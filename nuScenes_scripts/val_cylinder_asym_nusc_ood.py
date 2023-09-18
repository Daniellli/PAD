# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import spconv.pytorch as spconv

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings
from shutil import copyfile

warnings.filterwarnings("ignore")

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce, get_world_size, all_reduce
import os

def ddp_setup():
  init_process_group(backend="nccl")

DISTRIBUTED = 'LOCAL_RANK' in os.environ

def main(args):
    if DISTRIBUTED:
        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    torch.cuda.set_device(gpu_id)
    pytorch_device = torch.device('cuda:' + str(gpu_id))

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
    # Replace model load path with the one in args if the one in args is not empty
    if len(args.load_path) > 0:
        model_load_path = args.load_path
        EXPERIMENT_NAME = model_load_path[
                          model_load_path.rfind('checkpoints') + len('checkpoints') + 1:model_load_path.rfind('/')]
        MODEL_NAME = model_load_path[model_load_path.rfind('/') + 1:model_load_path.rfind('.')]
    else:
        EXPERIMENT_NAME = ''
        MODEL_NAME = ''

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model, device=pytorch_device)
        print('Load checkpoint file successfully!')

    if DISTRIBUTED:
        my_model.to(gpu_id)
        my_model = DDP(my_model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        my_model.to(gpu_id)

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    best_val_miou = 0

    # lr_scheduler.step(epoch)
    my_model.eval()
    hist_list = []
    val_loss_list = []
    pbar = tqdm(total=len(val_dataset_loader))
    global_iter = 0
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            if DISTRIBUTED:
                coor_ori, output_normal_dummy = my_model.module.forward_dummy_final(
                    val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber)
            else:
                coor_ori, output_normal_dummy = my_model.forward_dummy_final(
                    val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber)

            predict_labels = output_normal_dummy[:, :-1, ...]

            # aux_loss = loss_fun(aux_outputs, point_label_tensor)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                  ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
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
            idx_s = "%06d" % idx[0]

            if len(EXPERIMENT_NAME) == 0 and len(MODEL_NAME) == 0:
                uncertainty_save_path = '../../nuScenes/predictions/scores_softmax_2dummy_1_01_final_latest/' + idx_s + '.label'
                point_predict_save_path = '../../nuScenes/predictions/predictions_2dummy_1_01_final_cross_latest/' + idx_s + '.label'
            elif len(EXPERIMENT_NAME) > 0 and len(MODEL_NAME) > 0:
                experiment_folder = '../../nuScenes/predictions/' + EXPERIMENT_NAME + '/'
                model_folder = experiment_folder + MODEL_NAME + '/'
                uncertainty_folder = model_folder + 'uncertainty' + '/'
                point_predict_folder = model_folder + 'point_predict' + '/'
                uncertainty_save_path = uncertainty_folder + idx_s + '.label'
                point_predict_save_path = point_predict_folder + idx_s + '.label'
            else:
                raise ValueError('Incorrect experiment name or model name')

            # create folder for experiment
            if len(args.load_path) > 0:
                if not os.path.exists(experiment_folder[:experiment_folder.rfind('nuScenes') + len('nuScenes')] + '/predictions'):
                    os.mkdir(experiment_folder[:experiment_folder.rfind('nuScenes') + len('nuScenes')] + '/predictions')
                if not os.path.exists(experiment_folder):
                    os.mkdir(experiment_folder)
                # create folder for model
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                # create folder for uncertainty and point predict
                if not os.path.exists(uncertainty_folder):
                    os.mkdir(uncertainty_folder)
                if not os.path.exists(point_predict_folder):
                    os.mkdir(point_predict_folder)

            point_uncertainty_softmax.tofile(uncertainty_save_path)
            point_predict.tofile(point_predict_save_path)

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
            if DISTRIBUTED:
                torch.distributed.barrier()
            pbar.update(1)
            if DISTRIBUTED:
                torch.distributed.barrier()

    if DISTRIBUTED:
        # Gather ious from all cards
        torch.distributed.barrier()
        hist_list = torch.tensor(sum(hist_list)).contiguous().cuda()
        torch.distributed.all_reduce(hist_list, op=torch.distributed.ReduceOp.SUM)
        hist_list = hist_list.cpu().numpy()
        iou = per_class_iu(hist_list)
    else:
        iou = per_class_iu(sum(hist_list))

    if gpu_id == 0:
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
    parser.add_argument('-y', '--config_path', default='config/nuScenes_ood_final.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--load_path', default='', type=str, help='.')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
