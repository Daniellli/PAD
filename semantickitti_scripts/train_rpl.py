# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py
import copy
import sys
sys.path.append('../')
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import spconv.pytorch as spconv
from tqdm import tqdm
import torch.nn as nn
import scipy

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from magic_numbers_rpl import *
import matplotlib.pyplot as plt

import torch.nn.functional as func

import warnings

warnings.filterwarnings("ignore")

def smooth(arr, lamda1):
    new_array = arr
    copy_of_arr = 1 * arr

    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]

    # added the third direction for 3D points
    arr_3 = torch.zeros_like(copy_of_arr)
    arr_3[:, :, :, :-1] = copy_of_arr[:, :, :, 1:]
    arr_3[:, :, :, -1] = copy_of_arr[:, :, :, -1]

    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2) + torch.sum((arr_3 - copy_of_arr) ** 2)) / 3
    return lamda1 * loss

# TODO: Should it be calculated one by one for each point cloud in the batch?
def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def energy_loss(logits, targets):
    ood_ind = 5
    void_ind = 0
    num_class = 20
    T = 1.
    m_in = -12
    m_out = -6
    # Exclude class 0 from energy computation
    in_distribution_logits = torch.hstack([logits[:, 1:5],  logits[:, 6:]])
    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    ####################################################################
    if False:

        plt.hist(Ec_in.detach().cpu(), bins=100, color='blue')
        plt.hist(Ec_out.detach().cpu(), bins=100, color='red')
        plt.show()
        plt.hist(Ec_in.detach().cpu(), bins=100, color='blue')
        plt.show()
        plt.hist(Ec_out.detach().cpu(), bins=100, color='red')
        plt.show()
    ####################################################################




    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


class GaussianLayer_3D(nn.Module):
    def __init__(self, device=None):
        super(GaussianLayer_3D, self).__init__()
        self.seq = nn.Sequential(
            #nn.ReflectionPad2d(10),
            nn.Conv3d(1, 1, 7, stride=1, padding='same', bias=False, groups=1).to(device)
        )
        self.weights_init()
        self.device = device

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((7,7,7))
        n[3,3,3] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=1)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


class Gambler(torch.nn.Module):
    def __init__(self, reward, device, pretrain=-1, ood_reg=.1):
        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).to(device)
        self.pretrain = pretrain
        self.ood_reg = ood_reg
        self.device = device
        self.gaussian_layer_3d = GaussianLayer_3D(device=device)

    def forward(self, pred, targets, wrong_sample=False):
        # Exclude index 0 from softmax probability computation
        pred[:, 0] *= 0
        pred[:, 0] -= 99999.0
        pred_prob = torch.softmax(pred, dim=1)
        pred_prob = torch.clamp(pred_prob, min=1e-7)

        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])

        # ID and OOD probabilities
        true_pred, reservation = torch.hstack([pred_prob[:, :5], pred_prob[:, 6:]]), pred_prob[:, 5]

        # compute the reward via the energy score. Exclude class 0
        reward = torch.logsumexp(torch.hstack([pred[:, 1:5], pred[:, 6:]]), dim=1).pow(2)

        # 3D gaussian smoothing
        if reward.nelement() > 0:
            # TODO: improve gaussian filter when bs > 1
            if GAUSSIAN_FILTER:
                reward_list = []
                for i_reward in range(reward.shape[0]):
                    current_reward = reward[i_reward]
                    current_reward = current_reward.unsqueeze(0)
                    #assert (current_reward > 0).all(), "check reward before gaussian > 0"
                    current_reward = self.gaussian_layer_3d(current_reward)
                    #assert (current_reward > 0).all(), "check reward after gaussian > 0"
                    current_reward = current_reward.squeeze(0)
                    reward_list.append(current_reward)
                reward = torch.stack(reward_list)
        else:
            reward = self.reward

        if wrong_sample:  # if there's ood pixels inside the image
            reservation = torch.div(reservation, reward)
            mask = targets == 5
            # mask out each of the ood output channel
            # Exclude class 0 by left appending a mask of all False's
            mask_for_reserve_boosting_energy = torch.hstack([(mask.unsqueeze(1) & False),  mask.unsqueeze(1).repeat(1, 19-1, 1, 1,1)])
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask_for_reserve_boosting_energy]

            gambler_loss_out = torch.tensor([.0], device=self.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            void_mask = targets == 0
            targets[void_mask] = 0  # make void pixel to 0
            targets[mask] = 0  # make ood pixel to 0
            shifted_targets = targets - torch.tensor((targets > 5), dtype=int).to(targets.device)

            gambler_loss_in = torch.gather(true_pred, index=shifted_targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            intermediate_value = gambler_loss_in[(~mask) & (~void_mask)]
            assert not torch.any(intermediate_value <= 0), "nan check 3"

            gambler_loss_in = intermediate_value.log()
            return -(gambler_loss_in.mean() + gambler_loss_out.mean())
        else:
            reservation = torch.div(reservation, reward)
            mask = targets == 0
            targets[mask] = 0
            shifted_targets = targets - torch.tensor((targets > 5), dtype=int).to(targets.device)

            gambler_loss = torch.gather(true_pred, index=shifted_targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)

            intermediate_value = gambler_loss[~mask]
            gambler_loss = intermediate_value.log()
            assert not torch.any(torch.isnan(gambler_loss)), "nan check"

            return -gambler_loss.mean()


# vanilla_logits := logits w/o att_ block
# logits := logits w/ att_ block
def disimilarity_entropy(logits, vanilla_logits, t=1.):
    n_prob = torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)

    n_entropy = -torch.sum(n_prob * torch.log(n_prob), dim=1) / t
    a_entropy = -torch.sum(a_prob * torch.log(a_prob), dim=1) / t

    entropy_disimilarity = torch.nn.functional.mse_loss(input=a_entropy, target=n_entropy, reduction="none")
    assert ~torch.isnan(entropy_disimilarity).any(), print(torch.min(n_entropy), torch.max(a_entropy))

    return entropy_disimilarity


def rpl_energy_loss(logits, targets, vanilla_logits, out_idx=5, t=1.):
    out_msk = (targets == out_idx)
    void_msk = (targets == 0)

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)
    outlier_msk = (out_msk | void_msk)
    entropy_part = func.cross_entropy(input=logits, target=pseudo_targets, reduction='none')[~outlier_msk]
    reg = disimilarity_entropy(logits=logits, vanilla_logits=vanilla_logits)[~outlier_msk]
    if torch.sum(out_msk) > 0:
        logits = logits.flatten(start_dim=2).permute(0, 2, 1)
        # Clamp the logits so that torch.exp(logits) does not return inf
        # logits = torch.clamp(logits, max=50)
        # energy_part = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits), dim=2))[out_msk.flatten(start_dim=1)]).mean()
        energy_part = torch.nn.functional.relu(torch.logsumexp(logits, dim=2)[out_msk.flatten(start_dim=1)]).mean()
    else:
        energy_part = torch.tensor([.0], device=targets.device)
    return {"entropy_part": entropy_part.mean(), "reg": reg.mean(), "energy_part": energy_part}


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']

    experiment_output_dir = OUTPUT_DIR + '/' + args.experiment_name

    if not os.path.exists(experiment_output_dir):
        os.mkdir(experiment_output_dir)

    model_save_path = experiment_output_dir + '/' + 'model.pt'
    model_latest_path = experiment_output_dir + '/' + 'model_latest.pt'

    lamda_1 = train_hypers['lamda_1']
    lamda_2 = train_hypers['lamda_2']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config, ENABLE_RPL=ENABLE_RPL)

    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)

    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    # Freeze the closed-set model
    # In other words, only train logits2 and residual_block
    if not TRAIN_WHOLE_MODEL:
        for name, param in my_model.named_parameters():
            if 'logits2' in name or 'residual_block' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    my_model.to(pytorch_device)
    # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=train_hypers["learning_rate"])

    if LR_DROP_STEP_SIZE > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DROP_STEP_SIZE)

    loss_func_train, lovasz_softmax = loss_builder.build_ood(wce=True, lovasz=True,
                                                            num_class=num_class, ignore_label=ignore_label, weight=lamda_2)
    loss_func_val, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                                num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    writer = SummaryWriter('../runs/' + args.experiment_name)
    gambler_loss = Gambler(reward=[4.5], pretrain=-1, device=pytorch_device)

    while epoch < train_hypers['max_num_epochs']:
        # Validate
        if True:
            my_model.eval()
            hist_list = []
            val_loss_list = []
            with torch.no_grad():
                pbar_val = tqdm(total=len(val_dataset_loader))
                for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                        val_dataset_loader):

                    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                      val_pt_fea]
                    val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                    val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
                    # TODO: check this when using RPL
                    predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                    # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                    loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                          ignore=0) + loss_func_val(predict_labels.detach(), val_label_tensor)
                    predict_labels = torch.argmax(predict_labels, dim=1)
                    predict_labels = predict_labels.cpu().detach().numpy()
                    for count, i_val_grid in enumerate(val_grid):
                        hist_list.append(fast_hist_crop(predict_labels[
                                                            count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                            val_grid[count][:, 2]], val_pt_labs[count],
                                                        unique_label))
                    val_loss_list.append(loss.detach().cpu().numpy())
                    pbar_val.update(1)
            my_model.train()
            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100
            del val_vox_label, val_grid, val_pt_fea, val_grid_ten

            torch.save(my_model.state_dict(), model_latest_path)
            # Save checkpoint from each epoch
            save_path_for_current_epoch = model_latest_path[:model_latest_path.rfind('/')] + '/' + 'model_epoch_' + str(epoch) + '.pt'
            torch.save(my_model.state_dict(), save_path_for_current_epoch)
            # save model if performance is improved
            if best_val_miou < val_miou:
                best_val_miou = val_miou
                torch.save(my_model.state_dict(), model_save_path)

            print('Current val miou is %.3f while the best val miou is %.3f' %
                  (val_miou, best_val_miou))
            print('Current val loss is %.3f' %
                  (np.mean(val_loss_list)))

            writer.add_scalar('mIoU/valid', val_miou, epoch)

        e_loss_list, g_loss_id_list, g_loss_ood_list = [], [], []
        normal_loss_list, dummy_loss_list = [], []
        rpl_inlier_loss_list, rpl_outlier_loss_list = [], []
        loss_list = []

        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)


        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):

            # Train
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            energy_point_label_tensor = copy.deepcopy(point_label_tensor)
            gambler_point_label_tensor = copy.deepcopy(point_label_tensor)
            rpl_point_label_tensor = copy.deepcopy(point_label_tensor)
            point_label_tensor[point_label_tensor == 5] = 0

            # forward + backward + optimize
            coor_ori, output_normal_dummy, combined_output_normal_dummy = my_model.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                                        train_batch_size,
                                                                                        args.dummynumber)

            if ENERGY_LOSS or GAMBLER_LOSS or ENABLE_RPL:
                logits_for_loss_computing = torch.hstack(
                    [output_normal_dummy[:, :5], output_normal_dummy[:, -1:], output_normal_dummy[:, 6:-1]])
                combined_logits_for_loss_computing = torch.hstack(
                    [combined_output_normal_dummy[:, :5], combined_output_normal_dummy[:, -1:], combined_output_normal_dummy[:, 6:-1]])
            voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            if REAL_LOSS:
                loss_normal = loss_func_train(output_normal_dummy, point_label_tensor)
            else:
                loss_normal = torch.tensor(0, device=pytorch_device)

            output_normal_dummy = output_normal_dummy.permute(0,2,3,4,1)
            output_normal_dummy = output_normal_dummy[coor_ori.permute(1,0).chunk(chunks=4, dim=0)].squeeze()
            index_tmp = torch.arange(0,coor_ori.shape[0]).unsqueeze(0).cuda()
            voxel_label_origin[voxel_label_origin == 20] = 0
            index_tmp = torch.cat([index_tmp, voxel_label_origin], dim=0)
            output_normal_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            label_dummy = torch.ones(output_normal_dummy.shape[0]).type(torch.LongTensor).cuda()*20
            label_dummy[voxel_label_origin.squeeze() == 0] = 0
            if CALIBRATION_LOSS:
                loss_dummy = loss_func_train(output_normal_dummy, label_dummy)
            else:
                loss_dummy = torch.tensor(0, device=pytorch_device)

            # Energy loss
            # Labels for energy loss: Do not use class 5. Use scaled id objets (class 20) as ood objects during training
            if ENERGY_LOSS:
                energy_point_label_tensor[energy_point_label_tensor == 5] = 0
                energy_point_label_tensor[energy_point_label_tensor == 20] = 5
                e_loss, _ = energy_loss(logits_for_loss_computing, energy_point_label_tensor)
            else:
                e_loss = torch.tensor(0).to(pytorch_device)

            # Gambler loss
            if GAMBLER_LOSS:
                gambler_point_label_tensor[gambler_point_label_tensor == 5] = 0
                gambler_point_label_tensor[gambler_point_label_tensor == 20] = 5
                num_ood_samples = torch.sum((gambler_point_label_tensor == 5),
                                            dim=tuple(np.arange(len(gambler_point_label_tensor.shape))[1:]))
                is_ood = num_ood_samples > 0
                in_logits, in_target = logits_for_loss_computing[~is_ood], gambler_point_label_tensor[~is_ood]
                out_logits, out_target = logits_for_loss_computing[is_ood], gambler_point_label_tensor[is_ood]
                # 1. in distribution
                if in_logits.shape[0] > 0:
                    g_loss_id = gambler_loss(pred=in_logits, targets=in_target, wrong_sample=False)
                else:
                    g_loss_id = torch.tensor(0).cuda()
                # 2. out-of-distribution
                if torch.any(is_ood):
                    g_loss_ood = gambler_loss(pred=out_logits, targets=out_target, wrong_sample=True)
                else:
                    g_loss_ood = torch.tensor(0).cuda()
            else:
                g_loss_id = torch.tensor(0).to(pytorch_device)
                g_loss_ood = torch.tensor(0).to(pytorch_device)

            # RPL loss
            if ENABLE_RPL:
                rpl_point_label_tensor[rpl_point_label_tensor == 5] = 0
                rpl_point_label_tensor[rpl_point_label_tensor == 20] = 5
                rpl_loss_dict = rpl_energy_loss(logits=combined_logits_for_loss_computing,
                                                targets=rpl_point_label_tensor,
                                                vanilla_logits=logits_for_loss_computing,
                                                out_idx=5)
                rpl_inlier_loss = rpl_loss_dict["entropy_part"] + rpl_loss_dict["reg"]
                rpl_outlier_loss = rpl_loss_dict["energy_part"]
            else:
                rpl_inlier_loss = torch.tensor(0).to(pytorch_device)
                rpl_outlier_loss = torch.tensor(0).to(pytorch_device)

            loss = (loss_normal+lamda_1*loss_dummy) + \
                   (0.1 * e_loss + g_loss_id + g_loss_ood) + \
                   (rpl_inlier_loss + rpl_outlier_loss * RPL_ENERGY_WEIGHT)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            normal_loss_list.append(loss_normal.item())
            dummy_loss_list.append(loss_dummy.item())
            e_loss_list.append(e_loss.item())
            g_loss_id_list.append(g_loss_id.item())
            g_loss_ood_list.append(g_loss_ood.item())
            rpl_inlier_loss_list.append(rpl_inlier_loss.item())
            rpl_outlier_loss_list.append(rpl_outlier_loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            # if i_iter > 10:
            #     break

        writer.add_scalar('Loss/train/energy_loss', np.array(e_loss_list).mean(), epoch)
        writer.add_scalar('Loss/train/g_loss_id', np.array(g_loss_id_list).mean(), epoch)
        writer.add_scalar('Loss/train/g_loss_ood', np.array(g_loss_ood_list).mean(), epoch)
        writer.add_scalar('Loss/train/total_loss', np.array(loss_list).mean(), epoch)
        writer.add_scalar('Loss/train/real_normal', np.array(normal_loss_list).mean(), epoch)
        writer.add_scalar('Loss/train/real_calibration', np.array(dummy_loss_list).mean(), epoch)
        writer.add_scalar('Loss/train/rpl_inlier_loss', np.array(rpl_inlier_loss_list).mean(), epoch)
        writer.add_scalar('Loss/train/rpl_outlier_loss', np.array(rpl_outlier_loss_list).mean(), epoch)
        writer.add_scalar('Misc/LR', optimizer.param_groups[0]["lr"], epoch)
        pbar.close()
        epoch += 1

        if LR_DROP_STEP_SIZE > 0:
            lr_scheduler.step()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/rpl.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--experiment_name', default='debug_rpl', type=str)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    print()
    print('TRAIN_WHOLE_MODEL: ', TRAIN_WHOLE_MODEL)
    print()
    print('REAL_LOSS:         ', REAL_LOSS)
    print('CALIBRATION_LOSS:  ', CALIBRATION_LOSS)
    print()
    print('ENERGY_LOSS:       ', ENERGY_LOSS)
    print('GAMBLER_LOSS:      ', GAMBLER_LOSS)
    print('GAUSSIAN_FILTER:   ', GAUSSIAN_FILTER)
    print()
    print('ENABLE_RPL:        ', ENABLE_RPL)
    print('RPL_ENERGY_WEIGHT: ', RPL_ENERGY_WEIGHT)
    print()

    main(args)