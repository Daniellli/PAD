

import sys
import numpy as np
import torch


import warnings
import matplotlib.pyplot as plt


from utils.utils import *
from utils.image_utils import *
warnings.filterwarnings("ignore")


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
