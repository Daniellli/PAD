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
from tqdm import tqdm
import spconv.pytorch as spconv

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings
from shutil import copyfile
import matplotlib.pyplot as plt
from yang_train_pebal import GaussianLayer_3D
from magic_numbers_yang_train_pebal import *


from utils.utils import *
from utils.image_utils import *
warnings.filterwarnings("ignore")


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


def plot_3d_points(coordinate_list, color_list, index=0):

    coordinates = coordinate_list[index]
    colors = color_list[index]

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]

    fig = plt.figure(figsize=(20, 14))
    ax = plt.axes(projection="3d")
    ax.scatter3D(xs, ys, zs, c=colors)
    plt.show()
    return

import matplotlib.colors as mcolors


color_mapping = [
    'white',  # "unlabeled", and others ignored
    'blue',  # "car"
    'indigo',  # "bicycle"
    'magenta',  # "motorcycle"
    'orchid',  # "truck"
    'red',  # "other-vehicle"
    'cyan',  # "person"
    'lightskyblue',  # "bicyclist"
    'steelblue',  # "motorcyclist"
    'darkgray',  # "road"
    'darkred',  # "parking"
    'saddlebrown',  # "sidewalk"
    'white',  # "other-ground"
    'cadetblue',  # "building"
    'orange',  # "fence"
    'darkgreen',  # "vegetation"
    'chocolate',  # "trunk"
    'black',  # "terrain"
    'navy',  # "pole"
    'violet',  # "traffic-sign"
]

index_to_name = [
    'unlabeled',
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'other-vehicle',
    'person',
    'bicyclist',
    'motorcyclist',
    'road',
    'parking',
    'sidewalk',
    'other-ground',
    'building',
    'fence',
    'vegetation',
    'trunk',
    'terrain',
    'pole',
    'traffic-sign'
]



def plot_3d_points_by_group(coordinate_list, color_list, index=0, angle_x=0, angle_y=0):
    coordinates = coordinate_list[index]
    colors = color_list[index]

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]


    if False:

        for color in np.unique(colors):
            fig = plt.figure(figsize=(20, 15))
            ax = plt.axes(projection="3d")
            indices_for_current_color = colors == color
            indices_for_current_color = [item[0] for item in indices_for_current_color]
            ax.scatter3D(xs[indices_for_current_color], ys[indices_for_current_color], zs[indices_for_current_color],
                         label=index_to_name[color], color=color_mapping[color])

            plt.legend()
            if angle_x is not None and angle_y is not None:
                ax.view_init(angle_x, angle_y)
            plt.show()


    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection="3d")

    for color in np.unique(colors):
        indices_for_current_color = colors == color
        indices_for_current_color = [item[0] for item in indices_for_current_color]
        if color_mapping[color] != 'white':
            ax.scatter3D(xs[indices_for_current_color], ys[indices_for_current_color], zs[indices_for_current_color], label=index_to_name[color], color=color_mapping[color])

    plt.legend()


    if angle_x is not None and angle_y is not None :
        ax.view_init(angle_x, angle_y)
    plt.show()
    return

def plot_anomaly_score(coordinate_list, anomay_score, index=0, angle_x=0, angle_y=0):
    coordinates = coordinate_list[index]

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection="3d")

    ax.scatter3D(xs[:], ys[:], zs[:], c=torch.tensor(anomay_score*255, dtype=int))

    plt.show()

def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path # 'config/semantickitti_ood_final.yaml'

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
    # Replace model load path with the one in args if the one in args is not empty

    if len(args.load_path) > 0:

        # print(f"ready to load model {args.load_path}")
        model_load_path = args.load_path
        #* not suitable for all path format
        split_parts = args.load_path.split('/')
        EXPERIMENT_NAME ='/'.join(split_parts[:-1])
        MODEL_NAME = split_parts[-1].split('.')[0]
        # print(f'EXPERIMENT_NAME :{EXPERIMENT_NAME}, MODEL_NAME: {MODEL_NAME}')
    else:
        raise Exception
    
    model_save_path = train_hypers['model_save_path']
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)

    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)
    if os.path.exists(model_load_path):
        print('Loading ' + model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
        print('Load checkpoint file successfully!')
    else:
        print('Checkpoint file does not exit!')

    my_model.to(pytorch_device)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)

    print("number of params:", n_parameters)

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

    torch.backends.cudnn.benchmark = True

    #!+========================================================================================================
    if len(EXPERIMENT_NAME) == 0 and len(MODEL_NAME) == 0:
        uncertainty_folder = '../../semantic_kitti/predictions/sequences/08/scores_softmax_2dummy_1_01_final_latest/' 
        point_predict_folder = '../../semantic_kitti/predictions/sequences/08/predictions_2dummy_1_01_final_cross_latest/' 

    elif len(EXPERIMENT_NAME) > 0 and len(MODEL_NAME) > 0:
        experiment_folder = EXPERIMENT_NAME
        model_folder = join(experiment_folder, MODEL_NAME )

        uncertainty_folder = join(model_folder ,'uncertainty' )
        point_predict_folder =join( model_folder ,'point_predict' )
    else:
        raise ValueError('Incorrect experiment name or model name')
    
    if exists( uncertainty_folder) and exists(point_predict_folder) and len(os.listdir(point_predict_folder)) == val_dataset_loader.dataset.__len__():
        return 
    
    #!+========================================================================================================
    

    with torch.no_grad():

        gaussian_layer_3d = GaussianLayer_3D(device=pytorch_device)

        start_time = time.time()
        total_time_without_io = 0

        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                val_dataset_loader):
            B = len(idx)
            start_time_without_io = time.time()
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            coor_ori, output_normal_dummy = my_model.forward_dummy_final(
                val_pt_fea_ten, val_grid_ten, B, args.dummynumber, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False)
            predict_labels = output_normal_dummy[:,:-1,...]

            # aux_loss = loss_fun(aux_outputs, point_label_tensor)
            if args.speed_test is None:
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                      ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
            uncertainty_scores_logits = output_normal_dummy[:,-1,...]
            uncertainty_scores_logits = uncertainty_scores_logits.cpu().detach().numpy()

            softmax_layer = torch.nn.Softmax(dim=1)
            uncertainty_scores_softmax = softmax_layer(output_normal_dummy)[:,-1,...]
            uncertainty_scores_softmax = uncertainty_scores_softmax.cpu().detach().numpy()

            if False:
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=None, angle_y=None)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=0)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=30)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=60)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=90)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=120)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=150)
                plot_3d_points_by_group(val_grid, val_pt_labs, index=0, angle_x=0, angle_y=180)



            if VAL_ENERGY_UNCERTAINTY:
                # Compute anomaly score using energy of ID classes.
                # Additionally, exclude class 0 from energy computation.
                logits_for_loss_computing = torch.hstack(
                    [output_normal_dummy[:, 1:5], output_normal_dummy[:, 6:-1]])
                anomaly_score = -(1. * torch.logsumexp(logits_for_loss_computing / 1., dim=1))
                ####################################################################
                if False:
                    targets = val_label_tensor
                    # Note: when visualizing, need to map 99 to 5 instead of to 0 in
                    #  config/label_mapping/semantic-kitti.yaml
                    ood_ind = 5
                    void_ind = 0
                    Ec_out = anomaly_score[targets == ood_ind].cpu()
                    Ec_in = anomaly_score[(targets != ood_ind) & (targets != void_ind)].cpu()

                    plt.hist(Ec_in, bins=100, color='blue')
                    plt.hist(Ec_out, bins=100, color='red')
                    plt.show()
                    plt.hist(Ec_in, bins=100, color='blue')
                    plt.show()
                    plt.hist(Ec_out, bins=100, color='red')
                    plt.show()
                ####################################################################


                # Cast anomaly_score to [0,1]
                anomaly_score = anomaly_score - anomaly_score.min()
                anomaly_score = anomaly_score / anomaly_score.max()
                # TODO: smoothing
                anomaly_score = anomaly_score.cpu().detach().numpy()

                REAL_uncertainty = copy.deepcopy(uncertainty_scores_softmax)
                PEBAL_uncertainty = copy.deepcopy(anomaly_score)

                uncertainty_scores_softmax = anomaly_score


            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            # val_grid_ten: [batch, points, 3]
            # val_vox_label: [batch, 480, 360, 32]
            # val_pt_fea_ten: [batch, points, 9]
            # val_pt_labs: [batch, points, 1]
            count = 0
            point_predict = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]].astype(np.int32)
            point_uncertainty_logits = uncertainty_scores_logits[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]
            point_uncertainty_softmax = uncertainty_scores_softmax[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]

            # do not save predictions when doing speed test
            if args.speed_test is not None and args.speed_test == True:
                pbar.update(1)
                total_time_without_io += time.time() - start_time_without_io

                continue



            # Plot heatmap of PEBAL uncertainty and REAL uncertainty in 3D space
            if False:
                PEBAL_point_uncertainty = PEBAL_uncertainty[
                    count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]
                REAL_point_uncertainty = REAL_uncertainty[
                    count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]

                labels_as_one_row = np.array([k[0] for k in val_pt_labs[0]])

                # Plot the distribution of energy when labels are from different classes
                # Note: need to map 99 to 5 instead of to 0 in semantic-kitti.yaml
                if False:
                    for i in range(1, 20):
                        if i == 5:
                            continue
                        if (labels_as_one_row == i).sum() > 0:
                            plt.hist(PEBAL_point_uncertainty[labels_as_one_row == i], bins=100,
                                     label=str(i) + ' ' + index_to_name[i], log=True, color=color_mapping[i])
                    plt.hist(PEBAL_point_uncertainty[labels_as_one_row == 5], bins=100,
                             label=str(5) + ' ' + index_to_name[5], log=True, color=color_mapping[5])
                    plt.legend()
                    plt.show()



                plot_anomaly_score(val_grid, PEBAL_point_uncertainty)
                plot_anomaly_score(val_grid, REAL_point_uncertainty)

            # Plot the predicted classes of id and ood points before and
            # after applying rankfeat
            # Note: Need to map 99 to 5 instead of to 0 in learning_map
            # in semantic-kitti.yaml.
            PLOT_RANKFEAT_EFFECTS = False
            if PLOT_RANKFEAT_EFFECTS:
                plot_rankfeat_effects(val_pt_labs, point_predict, my_model, val_pt_fea_ten, val_grid_ten,
                                      val_batch_size,
                                      val_grid, idx)

            idx_s = "%06d" % idx[0]
            # point_uncertainty_logits.tofile(
            #         '/harddisk/jcenaa/semantic_kitti/predictions/sequences/08/scores_logits_dummy_latest/' + idx_s + '.label')
            #!+============================================
            # if len(EXPERIMENT_NAME) == 0 and len(MODEL_NAME) == 0:
            #     uncertainty_save_path = '../../semantic_kitti/predictions/sequences/08/scores_softmax_2dummy_1_01_final_latest/' + idx_s + '.label'
            #     point_predict_save_path = '../../semantic_kitti/predictions/sequences/08/predictions_2dummy_1_01_final_cross_latest/' + idx_s + '.label'
            # elif len(EXPERIMENT_NAME) > 0 and len(MODEL_NAME) > 0:
            #     experiment_folder = '../../semantic_kitti/predictions/sequences/08/' + EXPERIMENT_NAME + '/'
            #     model_folder = experiment_folder + MODEL_NAME + '/'
            #     uncertainty_folder = model_folder + 'uncertainty' + '/'
            #     point_predict_folder = model_folder + 'point_predict' + '/'
            #     uncertainty_save_path = uncertainty_folder + idx_s + '.label'
            #     point_predict_save_path = point_predict_folder + idx_s + '.label'
            # else:
            #     raise ValueError('Incorrect experiment name or model name')
                
            uncertainty_save_path = join(uncertainty_folder ,idx_s + '.label')
            point_predict_save_path = join(point_predict_folder, idx_s + '.label')
            #!+============================================

            # create folder for experiment
            if len(args.load_path) > 0:
                make_dir(experiment_folder)
                make_dir(model_folder)
                make_dir(uncertainty_folder)
                make_dir(point_predict_folder)
                
            make_dir(os.path.dirname(uncertainty_save_path))
            make_dir(os.path.dirname(point_predict_save_path))

            point_uncertainty_softmax.tofile(uncertainty_save_path)
            point_predict.tofile(point_predict_save_path)

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
            pbar.update(1)

    if args.speed_test is not None and args.speed_test == True:
        current_time = time.time()
        time_used = current_time - start_time
        print('Time used:', time_used)
        print('FPS:', len(val_dataset_loader) / time_used)
        print('Time without IO:', total_time_without_io)
        print('FPS exclude IO:', len(val_dataset_loader) / total_time_without_io)
        return

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
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--load_path', default='', type=str, help='.')
    parser.add_argument('--speed_test', default=None, type=bool, help='.')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    print('Energy VAL_ENERGY_UNCERTAINTY:       ', VAL_ENERGY_UNCERTAINTY)

    tic = time.time()    
    main(args)
    print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

