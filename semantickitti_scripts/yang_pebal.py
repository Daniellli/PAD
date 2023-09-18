# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py
import copy
import sys

import scipy.ndimage

sys.path.append('../')
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import spconv.pytorch as spconv

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
import torch.nn.functional as F
from torchvision import transforms

import warnings
from shutil import copyfile
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def plot_rankfeat_effects(val_pt_labs, point_predict, my_model, val_pt_fea_ten, val_grid_ten, val_batch_size, val_grid, idx):
    # find out predicted classes of ood points (whose target label == 5)
    assert len(val_pt_labs) == 1
    point_target = val_pt_labs[0]
    point_target = point_target.T.squeeze(0)
    assert point_predict.shape == point_target.shape
    ood_predicted_labels = point_predict[point_target == 5]
    id_predicted_labels = point_predict[point_target != 5]

    # another round
    coor_ori, output_normal_dummy = my_model.forward_dummy_final(
        val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber, PROCESS_BLOCK_3=True,
        PROCESS_BLOCK_4=True)
    predict_labels = output_normal_dummy[:, :-1, ...]

    uncertainty_scores_logits = output_normal_dummy[:, -1, ...]
    uncertainty_scores_logits = uncertainty_scores_logits.cpu().detach().numpy()

    softmax_layer = torch.nn.Softmax(dim=1)
    uncertainty_scores_softmax = softmax_layer(output_normal_dummy)[:, -1, ...]
    uncertainty_scores_softmax = uncertainty_scores_softmax.cpu().detach().numpy()

    predict_labels = torch.argmax(predict_labels, dim=1)
    predict_labels = predict_labels.cpu().detach().numpy()

    count = 0
    point_predict = predict_labels[
        count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]].astype(np.int32)
    point_uncertainty_logits = uncertainty_scores_logits[
        count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]
    point_uncertainty_softmax = uncertainty_scores_softmax[
        count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]

    assert len(val_pt_labs) == 1
    point_target = val_pt_labs[0]
    point_target = point_target.T.squeeze(0)
    assert point_predict.shape == point_target.shape
    rank_feat_ood_predicted_labels = point_predict[point_target == 5]
    rank_feat_id_predicted_labels = point_predict[point_target != 5]

    num_points = 300
    s = plt.rcParams['lines.markersize'] ** 2 * 0.25

    id_random_indices = np.random.choice(len(id_predicted_labels), num_points)
    plt.scatter(np.arange(len(id_random_indices)), id_predicted_labels[id_random_indices], s=s)
    plt.scatter(np.arange(len(id_random_indices)), rank_feat_id_predicted_labels[id_random_indices], s=s)
    plt.gca().legend(("ID Data", "ID Data with RankFeat"))
    plt.title('ID Class Prediction' + ' ' + str(idx))
    plt.xlabel("Data Samples")
    plt.ylabel("Class Index")
    plt.show()

    ood_random_indices = np.random.choice(len(ood_predicted_labels), num_points)
    plt.scatter(np.arange(len(ood_random_indices)), ood_predicted_labels[ood_random_indices], s=s)
    plt.scatter(np.arange(len(ood_random_indices)), rank_feat_ood_predicted_labels[ood_random_indices], s=s)
    plt.gca().legend(("OOD Data", "OOD Data with RankFeat"))
    plt.title('OOD Class Prediction' + ' ' + str(idx))
    plt.xlabel("Data Samples")
    plt.ylabel("Class Index")
    plt.show()
    return



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

    energy = -(T * torch.logsumexp(logits[:, :num_class] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


# TODO
class Gambler(torch.nn.Module):
    def __init__(self, reward, device, pretrain=-1, ood_reg=.1):
        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).to(device)
        self.pretrain = pretrain
        self.ood_reg = ood_reg
        self.device = device

    def forward(self, pred, targets, wrong_sample=False):

        pred_prob = torch.softmax(pred, dim=1)
        pred_prob = torch.clamp(pred_prob, min=1e-7)

        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])
        true_pred, reservation = torch.hstack([pred_prob[:, :5], pred_prob[:, 6:]]), pred_prob[:, 5]

        # compute the reward via the energy score;
        reward = torch.logsumexp(torch.hstack([pred_prob[:, :5], pred_prob[:, 6:]]), dim=1).pow(2)

        # TODO: 3D gaussian smoothing
        if reward.nelement() > 0:
            # gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            # reward = reward.unsqueeze(0)
            # reward = gaussian_smoothing(reward)
            # reward = reward.squeeze(0)
            pass
        else:
            reward = self.reward

        if wrong_sample:  # if there's ood pixels inside the image
            reservation = torch.div(reservation, reward)
            mask = targets == 5
            # mask out each of the ood output channel
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask.unsqueeze(1).repeat(1, 19, 1, 1,1)]

            gambler_loss_out = torch.tensor([.0], device=self.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            void_mask = targets == 0
            targets[void_mask] = 0  # make void pixel to 0
            targets[mask] = 0  # make ood pixel to 0
            shifted_targets = targets
            shifted_targets[shifted_targets > 5] -= 1

            gambler_loss_in = torch.gather(true_pred, index=shifted_targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            gambler_loss_in = gambler_loss_in[(~mask) & (~void_mask)].log()
            return -(gambler_loss_in.mean() + gambler_loss_out.mean())
        else:
            mask = targets == 255
            targets[mask] = 0
            reservation = torch.div(reservation, reward)
            gambler_loss = torch.gather(true_pred, index=targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)
            gambler_loss = gambler_loss[~mask].log()
            # assert not torch.any(torch.isnan(gambler_loss)), "nan check"
            return -gambler_loss.mean()


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
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)
        print('Load checkpoint file successfully!')
    else:
        print('Checkpoint file does not exit!')

    my_model.to(pytorch_device)
    # Freeze parameters except for the last block
    for name, param in my_model.cylinder_3d_spconv_seg.named_parameters():
        if 'logits' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    best_val_miou = 0

    # lr_scheduler.step(epoch)
    my_model.train()
    hist_list = []
    val_loss_list = []
    pbar = tqdm(total=len(train_dataset_loader))
    global_iter = 0

    gambler_loss = Gambler(reward=[4.5], pretrain=-1, device=pytorch_device)

    for i_iter_val, (_, val_vox_label, val_grid, _, val_pt_fea) in enumerate(
            train_dataset_loader):

        optimizer.zero_grad()

        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                          val_pt_fea]
        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

        coor_ori, output_normal_dummy = my_model.forward_dummy_final(
            val_pt_fea_ten, val_grid_ten, train_batch_size, args.dummynumber, PROCESS_BLOCK_3=False,
            PROCESS_BLOCK_4=False)
        predict_labels = output_normal_dummy[:, :-1, ...]

        # Energy loss
        logits_for_loss_computing = torch.hstack([output_normal_dummy[:, :5], output_normal_dummy[:, -1:], output_normal_dummy[:, 6:-1]])
        e_loss, _ = energy_loss(logits_for_loss_computing, val_label_tensor)

        # determine which scenes contain ood points
        num_ood_samples = torch.sum(val_label_tensor, dim=tuple(np.arange(len(val_label_tensor.shape))[1:]))
        is_ood = num_ood_samples > 0
        in_logits, in_target = logits_for_loss_computing[~is_ood], val_label_tensor[~is_ood]
        out_logits, out_target = logits_for_loss_computing[is_ood], val_label_tensor[is_ood]

        # Gambler loss. TODO: 3D Gaussian smoothing
        # 1. in distribution
        g_loss_id = gambler_loss(pred=in_logits, targets=in_target, wrong_sample=False)
        # 2. out-of-distribution
        if torch.any(is_ood):
            g_loss_ood = gambler_loss(pred=out_logits, targets=out_target, wrong_sample=True)
        else:
            g_loss_ood = 0

        loss = 0.1 * e_loss + g_loss_id + g_loss_ood

        # Back-prop
        loss.backward()
        optimizer.step()


        uncertainty_scores_logits = output_normal_dummy[:, -1, ...]
        uncertainty_scores_logits = uncertainty_scores_logits.cpu().detach().numpy()

        softmax_layer = torch.nn.Softmax(dim=1)
        uncertainty_scores_softmax = softmax_layer(output_normal_dummy)[:, -1, ...]
        uncertainty_scores_softmax = uncertainty_scores_softmax.cpu().detach().numpy()

        predict_labels = torch.argmax(predict_labels, dim=1)
        predict_labels = predict_labels.cpu().detach().numpy()
        # val_grid_ten: [batch, points, 3]
        # val_vox_label: [batch, 480, 360, 32]
        # val_pt_fea_ten: [batch, points, 9]
        # val_pt_labs: [batch, points, 1]
        count = 0
        point_predict = predict_labels[
            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]].astype(np.int32)
        point_uncertainty_logits = uncertainty_scores_logits[
            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]
        point_uncertainty_softmax = uncertainty_scores_softmax[
            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]

        # Plot the predicted classes of id and ood points before and
        # after applying rankfeat
        # Note: Need to map 99 to 5 instead of to 0 in learning_map
        # in semantic-kitti.yaml.
        PLOT_RANKFEAT_EFFECTS = False
        # if PLOT_RANKFEAT_EFFECTS:
        #     plot_rankfeat_effects(val_pt_labs, point_predict, my_model, val_pt_fea_ten, val_grid_ten,
        #                           val_batch_size,
        #                           val_grid, idx)

        # idx_s = "%06d" % idx[0]
        # # point_uncertainty_logits.tofile(
        # #         '/harddisk/jcenaa/semantic_kitti/predictions/sequences/08/scores_logits_dummy_latest/' + idx_s + '.label')
        # point_uncertainty_softmax.tofile(
        #     '../../semantic_kitti/predictions/sequences/08/scores_softmax_2dummy_1_01_final_latest/' + idx_s + '.label')
        # point_predict.tofile(
        #     '../../semantic_kitti/predictions/sequences/08/predictions_2dummy_1_01_final_cross_latest/' + idx_s + '.label')

        # for count, i_val_grid in enumerate(val_grid):
        #     hist_list.append(fast_hist_crop(predict_labels[
        #                                         count, val_grid[count][:, 0], val_grid[count][:, 1],
        #                                         val_grid[count][:, 2]], val_pt_labs[count],
        #                                     unique_label))
        val_loss_list.append(loss.detach().cpu().numpy())
        pbar.update(1)

    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten

    print('Current val miou is %.3f while the best val miou is %.3f' %
          (val_miou, best_val_miou))
    print('Current val loss is %.3f' %
          (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_ood_final.yaml')
    parser.add_argument('--dummynumber', default=2, type=int, help='number of dummy label.')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
