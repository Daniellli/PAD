# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py
import copy
import sys


import os
from os.path import split, join, exists, isdir,isfile
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import wandb 
warnings.filterwarnings("ignore")

sys.path.insert(0,os.getcwd())



#!  why DistributedSampler is not used ?
import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, all_reduce
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ConstantLR
import torch.nn.functional as F



import spconv.pytorch as spconv
from utils.load_save_util import load_checkpoint
from utils.utils import *
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.image_utils import * 



from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from pad_losses.gambler_loss import *
from loguru import logger 

#!+======================================================

# OUTPUT_DIR = '../../semantic_kitti/checkpoints'
TRAIN_WHOLE_MODEL = True
# REAL
REAL_LOSS = True            # 是否使用real的cross entropy loss
CALIBRATION_LOSS = True     # 是否使用real的calibration loss
# PEBAL
ENERGY_LOSS = True          # 是否使用pebal的energy loss
GAMBLER_LOSS = True         # 是否使用pebal的gambler loss
GAUSSIAN_FILTER = True      # 是否使用pebal的gaussian filter

LR_DROP_STEP_SIZE = 10

# Shapenet Anomaly
SHAPENET_ANOMALY = False    # 是否将shapnet的物体作为训练时的异常

VAL_ENERGY_UNCERTAINTY = False  # 生成validation的输出时是否使用energy值作为uncertainty

SAVE_MODIFIED_DATA = False
SAVE_ALL_SEQUENCES = False
SAVE_PATH_DIR_FOR_MODIFIED_DATA = 'modified_dataset'
DISTRIBUTED = 'LOCAL_RANK' in os.environ



#!+======================================================


def ddp_setup():
  init_process_group(backend="nccl")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print





"""
类似NMS,  

"""
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
"""
让model 输出的异常的数量不要太多, 因为真实场景中, 异常也不会很多. 
"""
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



# def save_ckpt(model, optimizer, )



def main(args):
    if DISTRIBUTED:
        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])

        torch.distributed.barrier()
        setup_for_distributed(global_rank == 0)
    else:
        gpu_id = 0

    ############################################################################
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
    print('SHAPENET_ANOMALYL  ', SHAPENET_ANOMALY)
    print()
    if SAVE_MODIFIED_DATA:
        print('SAVE_MODIFIED_DATA ', SAVE_MODIFIED_DATA)
        print('SAVE_ALL_SEQUENCES ', SAVE_ALL_SEQUENCES)
        print('SAVE_PATH_DIR_FOR_MODIFIED_DATA: ', SAVE_PATH_DIR_FOR_MODIFIED_DATA)
    ############################################################################

    torch.cuda.set_device(gpu_id)
    pytorch_device = torch.device('cuda:' + str(gpu_id))

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    # val_batch_size = val_dataloader_config['batch_size']
    # train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    #* the only line need config model's parameter 
    model_load_path = train_hypers['model_load_path']

    if int(os.environ['LOCAL_RANK']) == 0:
        experiment_output_dir = join(os.getcwd(),'runs', time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time())))
        # experiment_output_dir = join(os.getcwd(),'runs', args.experiment_name)
        make_dir(experiment_output_dir)
        model_save_path = join(experiment_output_dir, 'model_best.pt')
        model_latest_path = join(experiment_output_dir,'model_latest.pt')
        print(f"save path : {experiment_output_dir}")
        writer = SummaryWriter(experiment_output_dir)


    lamda_1 = train_hypers['lamda_1']
    lamda_2 = train_hypers['lamda_2']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    #* init model 
    my_model = model_builder.build(model_config)

    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)


    #!===========================
    # * resume model 
    if args.resume is not None and exists(args.resume):
        my_model = load_checkpoint(args.resume, my_model, device=pytorch_device)
        print(f" model {args.resume} has been resumed ")
    else:
        assert exists(model_load_path)
        my_model = load_checkpoint(model_load_path, my_model, device=pytorch_device)
        print(f" the model, {model_load_path} ,trained in stage 1 has been loaded")
    #!===========================
    
    #* init the optimizer 
    if DISTRIBUTED:
        print('World size:', torch.distributed.get_world_size())
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=train_hypers["learning_rate"] * torch.distributed.get_world_size())
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=train_hypers["learning_rate"] )
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=train_hypers["learning_rate"])


    #* init the scheduler 
    if LR_DROP_STEP_SIZE > 0:
        if DISTRIBUTED:
            world_size = torch.distributed.get_world_size()
            warmup_lr_factor_start = 1 / world_size
            warmup_lr_factor_end = 1
            warmup_lr_increment_amount = (warmup_lr_factor_end - warmup_lr_factor_start) / 5

            #? why so complex
            scheduler_0 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 0 * warmup_lr_increment_amount, total_iters=1)
            scheduler_1 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 1 * warmup_lr_increment_amount, total_iters=1)
            scheduler_2 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 2 * warmup_lr_increment_amount, total_iters=1)
            scheduler_3 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 3 * warmup_lr_increment_amount, total_iters=1)
            scheduler_4 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 4 * warmup_lr_increment_amount, total_iters=1)

            drop_lr_every_n_epochs = 10
            mile_stones = list(0 + np.arange(drop_lr_every_n_epochs, train_hypers['max_num_epochs'], drop_lr_every_n_epochs))
            scheduler_5 = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, gamma=0.1)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_0, scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5], milestones=[1, 2, 3, 4, 5])

            if args.last_epoch > -1:
                for i in range(args.last_epoch):
                    lr_scheduler.step()
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DROP_STEP_SIZE)
            if args.last_epoch > -1:
                for i in range(args.last_epoch):
                    lr_scheduler.step()


    #* distributed training for model 
    if DISTRIBUTED:
        my_model.to(gpu_id)
        print('In SyncBatchNorm.convert_sync_batchnorm')
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
        my_model = DDP(my_model, device_ids=[gpu_id], find_unused_parameters=True)
        # my_model = DDP(my_model, device_ids=[gpu_id], find_unused_parameters=True,broadcast_buffers = True)


    loss_func_train, lovasz_softmax = loss_builder.build_ood(wce=True, lovasz=True,
                                                            num_class=num_class, ignore_label=ignore_label, weight=lamda_2)
    loss_func_val, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                                num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  SHAPENET_ANOMALY=SHAPENET_ANOMALY)

    # training
    epoch = args.last_epoch + 1
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    
    gambler_loss = Gambler(reward=[4.5], pretrain=-1, device=pytorch_device)

    while epoch < train_hypers['max_num_epochs']:

        torch.distributed.barrier()
        
        

        # Debug: write lr of each epoch to tensorboard
        if False:
            writer.add_scalar('Misc/LR', optimizer.param_groups[0]["lr"], epoch)
            lr_scheduler.step()
            print(epoch)
            epoch += 1
            continue


        if SAVE_MODIFIED_DATA and epoch == 1:
            break
        # Validate
        if False and not SAVE_MODIFIED_DATA:
            my_model.eval()
            hist_list = []
            val_loss_list = []
            with torch.no_grad():
                pbar_val = tqdm(total=len(val_dataset_loader), disable=DISTRIBUTED and global_rank != 0, desc='Valid')
                for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                        val_dataset_loader):

                    val_batch_size = len(val_pt_fea)

                    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                      val_pt_fea]
                    val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                    val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

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
                    if gpu_id == 0:
                        pbar_val.update(1)
                    if DISTRIBUTED:
                        torch.distributed.barrier()
            # my_model.train()
            if DISTRIBUTED:
                # Gather ious from all cards
                torch.distributed.barrier()
                hist_list = torch.tensor(sum(hist_list)).contiguous().cuda()
                torch.distributed.all_reduce(hist_list, op=torch.distributed.ReduceOp.SUM) #* reduce the hist_lists in all nodes with op operation, sum by default
            if not DISTRIBUTED or global_rank == 0:
                if DISTRIBUTED:
                    hist_list = hist_list.cpu().numpy()
                    iou = per_class_iu(hist_list)
                else:
                    iou = per_class_iu(sum(hist_list))

                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                if DISTRIBUTED:
                    torch.save(my_model.module.state_dict(), model_latest_path)
                else:
                    torch.save(my_model.state_dict(), model_latest_path)

                # Save checkpoint from each epoch
                save_path_for_current_epoch = join(model_latest_path[:model_latest_path.rfind('/')],'model_epoch_%s.pt'%(epoch))
                if DISTRIBUTED:
                    torch.save(my_model.module.state_dict(), save_path_for_current_epoch)
                else:
                    torch.save(my_model.state_dict(), save_path_for_current_epoch)
                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    if DISTRIBUTED:
                        torch.save(my_model.module.state_dict(), model_save_path)
                    else:
                        torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                
                writer.add_scalar('mIoU/valid', val_miou, epoch)

            torch.cuda.empty_cache()
            

        e_loss_list, g_loss_id_list, g_loss_ood_list, normal_loss_list, dummy_loss_list, loss_list = [], [], [], [], [], []
        pbar = tqdm(total=len(train_dataset_loader), disable=DISTRIBUTED and global_rank != 0, desc='Train')
        time.sleep(10)

        my_model.train()
        if DISTRIBUTED:
            train_dataset_loader.sampler.set_epoch(epoch)

        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if SAVE_MODIFIED_DATA:
                pbar.update(1)
                continue
            # Train
            train_batch_size,W,H,C=train_vox_label.shape
            # print(f"train_batch_size: {train_batch_size}")

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            energy_point_label_tensor = copy.deepcopy(point_label_tensor)
            gambler_point_label_tensor = copy.deepcopy(point_label_tensor)
            point_label_tensor[point_label_tensor == 5] = 0

            # forward + backward + optimize
            if DISTRIBUTED:
                coor_ori, output_normal_dummy = my_model.module.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                             train_batch_size,
                                                                             args.dummynumber)
            else:
                coor_ori, output_normal_dummy = my_model.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                                        train_batch_size,
                                                                                        args.dummynumber)
            if ENERGY_LOSS or GAMBLER_LOSS:
                logits_for_loss_computing = torch.hstack(
                    [output_normal_dummy[:, :5], output_normal_dummy[:, -1:], output_normal_dummy[:, 6:-1]])
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


            loss = (loss_normal+lamda_1*loss_dummy) + (0.1 * e_loss + g_loss_id + g_loss_ood)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            normal_loss_list.append(loss_normal.item())
            dummy_loss_list.append(loss_dummy.item())
            e_loss_list.append(e_loss.item())
            g_loss_id_list.append(g_loss_id.item())
            g_loss_ood_list.append(g_loss_ood.item())

            optimizer.zero_grad()

            
            if DISTRIBUTED:
                torch.distributed.barrier()
                pbar.update(1)
                torch.distributed.barrier()
            else:
                pbar.update(1)
            global_iter += 1

       


        if not 'train_batch_size' in locals().keys():
            train_batch_size = train_dataloader_config['batch_size']

            

        if DISTRIBUTED:
            torch.distributed.barrier()
            tb_energy_loss = torch.tensor(np.array(e_loss_list).mean()).cuda()
            tb_g_loss_id = torch.tensor(np.array(g_loss_id_list).mean()).cuda()
            tb_g_loss_ood = torch.tensor(np.array(g_loss_ood_list).mean()).cuda()
            tb_total_loss = torch.tensor(np.array(loss_list).mean()).cuda()
            tb_real_normal =  torch.tensor(np.array(normal_loss_list).mean()).cuda()
            tb_real_calibration = torch.tensor(np.array(dummy_loss_list).mean()).cuda()

            # reduce losses across all gpus
            all_reduce(tb_energy_loss, op=torch.distributed.ReduceOp.SUM)
            all_reduce(tb_g_loss_id, op=torch.distributed.ReduceOp.SUM)
            all_reduce(tb_g_loss_ood, op=torch.distributed.ReduceOp.SUM)
            all_reduce(tb_total_loss, op=torch.distributed.ReduceOp.SUM)
            all_reduce(tb_real_normal, op=torch.distributed.ReduceOp.SUM)
            all_reduce(tb_real_calibration, op=torch.distributed.ReduceOp.SUM)

            # normalize losses with respect to gpu quantity and batch size
            
            normalization_factor = torch.distributed.get_world_size() * train_batch_size
            tb_energy_loss = tb_energy_loss / normalization_factor
            tb_g_loss_id = tb_g_loss_id / normalization_factor
            tb_g_loss_ood = tb_g_loss_ood / normalization_factor
            tb_total_loss = tb_total_loss / normalization_factor
            tb_real_normal = tb_real_normal / normalization_factor
            tb_real_calibration = tb_real_calibration / normalization_factor

            if global_rank == 0:
                writer.add_scalar('Loss/train/energy_loss', tb_energy_loss, epoch)
                writer.add_scalar('Loss/train/g_loss_id', tb_g_loss_id.mean(), epoch)
                writer.add_scalar('Loss/train/g_loss_ood', tb_g_loss_ood, epoch)
                writer.add_scalar('Loss/train/total_loss', tb_total_loss, epoch)
                writer.add_scalar('Loss/train/real_normal', tb_real_normal, epoch)
                writer.add_scalar('Loss/train/real_calibration', tb_real_calibration, epoch)
                writer.add_scalar('Misc/LR', optimizer.param_groups[0]["lr"], epoch)
        else:
            # normalize losses with respect to batch size
            normalization_factor = train_batch_size
            writer.add_scalar('Loss/train/energy_loss', np.array(e_loss_list).mean() / normalization_factor, epoch)
            writer.add_scalar('Loss/train/g_loss_id', np.array(g_loss_id_list).mean() / normalization_factor, epoch)
            writer.add_scalar('Loss/train/g_loss_ood', np.array(g_loss_ood_list).mean()/ normalization_factor, epoch)
            writer.add_scalar('Loss/train/total_loss', np.array(loss_list).mean() / normalization_factor, epoch)
            writer.add_scalar('Loss/train/real_normal', np.array(normal_loss_list).mean() / normalization_factor, epoch)
            writer.add_scalar('Loss/train/real_calibration', np.array(dummy_loss_list).mean() / normalization_factor, epoch)
            writer.add_scalar('Misc/LR', optimizer.param_groups[0]["lr"], epoch)
            
        
        #* save model 
        # logger.info(f'lcoal rank == {os.environ["LOCAL_RANK"]},{type(os.environ["LOCAL_RANK"])}')
        if int(os.environ["LOCAL_RANK"]) == 0:
            #* save model 
            model_name = 'model_epoch_%s'%(epoch)

            save_path_for_current_epoch = join(model_latest_path[:model_latest_path.rfind('/')],model_name,model_name+'.pt')
            make_dir(os.path.dirname(save_path_for_current_epoch))
            


            #* , lr_scheduler, my_model,optimizer
            if DISTRIBUTED:   
                torch.save(my_model.module.state_dict(),save_path_for_current_epoch)

                # save_ckpt(
                #     {
                #         "model":my_model.module.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #         "scheduler": lr_scheduler.state_dict(),
                #         "epoch": epoch,
                #     },
                #     file_name= save_path_for_current_epoch
                # )
            else:
                torch.save(my_model.state_dict(),save_path_for_current_epoch)
                # save_ckpt(
                #     {
                #         "model":my_model.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #         "scheduler": lr_scheduler.state_dict(),
                #         "epoch": epoch,
                #     },
                #     file_name= save_path_for_current_epoch
                # )

          
        pbar.close()
        epoch += 1
        
        if LR_DROP_STEP_SIZE > 0:
            lr_scheduler.step()



# def save_ckpt(ckpt_in_dict,file_name):
#     torch.save(ckpt_in_dict,file_name)


# def load_ckpt(file_name):
#     return torch.load(file_name,map_location=torch.device('cpu'))




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_ood_final.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--experiment_name', default=None, type=str,required=False)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--last_epoch', default=-1, type=int)
    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = 'debug'
        
    main(args)
