# -*- coding:utf-8 -*-


"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import sys
from torch.utils import data
import glob
import pickle
from os.path import join, split,exists,isdir,isfile,dirname
#!+==============================================
sys.path.append(os.getcwd())
from .introduce_shapenet_objects import *
# from dataloader.introduce_shapenet_objects import *
#!+==============================================
from utils.pc_utils import * 


sys.path.append('../')
from semantickitti_scripts.magic_numbers_yang_train_pebal import *

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple

""""
# transformation between Cartesian coordinates and polar coordinates

format note: 
    rho :  the radius , x,y 距离原点的距离, 也就是半径
    phi: the degree  of the  direction , x-y 平面的极角
    z-axis : 原始的z轴分量
"""
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@register_dataset
class cylinder_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4, ds_sample=False, incre=None):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        #* the radius, the degree, and the  z-axis value
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)#* max radius 
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)#* minimal radius 
        max_bound = np.max(xyz_pol[:, 1:], axis=0) #* max degree and max z-value,  return a 2-dimension value 
        min_bound = np.min(xyz_pol[:, 1:], axis=0)#* min .. 
        max_bound = np.concatenate(([max_bound_r], max_bound)) #* concatenate as 3-dimension vector represent the max bound of radius, degree and the z-axis
        min_bound = np.concatenate(([min_bound_r], min_bound))#* the min 

        #* use fixed volume space, so the code above is useless? 
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
       
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple

@register_dataset
class cylinder_dataset_test(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4, ds_sample=False, incre=None):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data, path_save = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, path_save)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple

@register_dataset
class cylinder_dataset_panop(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4, ds_sample=False, SHAPENET_ANOMALY=None,shapenet_path=None,debug =False):
        self.debug = debug
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std
        #* if debug,  turn it into true
        self.ds_sample = ds_sample if not self.debug else True
        self.keep_points = 2000

        self.noise_rotation = np.random.uniform(min_rad, max_rad)
        
        #* if debug,  turn it into true
        self.SHAPENET_ANOMALY = SHAPENET_ANOMALY if not self.debug else True

        if self.SHAPENET_ANOMALY:

            # TODO: each epoch select a different class of shapenet object
            assert shapenet_path is not None 

            self.shapenet_object_paths = []

            tic = time.time()
            # for filename in glob.iglob(shapenet_path + '/**/pointcloud.npz', recursive=True): #* take long time ~
            #     self.shapenet_object_paths.append(os.path.abspath(filename))
            
            self.shapenet_object_paths = [ join(shapenet_path,x) for x in np.loadtxt(join(shapenet_path,'object_path_list.txt'),dtype=np.str0).tolist()]
            

            print('num shapenet objects: ', len(self.shapenet_object_paths),'spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T
    

    def get_name(self,idx):
        
        return self.point_cloud_dataset.im_idx[idx].split('/')[-1].split('.')[0]

    def __getitem__(self, index):
        # tic = time.time()
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]

        if len(data) == 3:
            xyz, labels, instances = data
        elif len(data) == 4:
            xyz, labels, instances, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        
        if self.ds_sample:
            minimum_pts_thre = 300
            instances = instances.squeeze()
            cls, cnt = np.unique(instances, return_counts=True)

            inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]

            # Scale up or down objects and set their labels to 20
            # for each instance in all instances that has more than 300 points
            for instance_idx in inst_basic_idx:
                # scale this instance with a probability equals 0.5
                # skip this object if it is an anomaly (label is class 5)
                rnd = np.random.rand()
                if rnd > 0.5 or labels[instances == instance_idx][0]==5:
                    continue

                # find out this instance's xyz and center
                obj_ins = xyz[instances==instance_idx]
                obj_ins_center = np.mean(obj_ins, axis=0)

                # calculate relative position w.r.t instance center for each point
                obj_ins = obj_ins - obj_ins_center

                # Scale up or scale down this instance by shifting its points
                scale_ds_large = np.random.rand()*1.5+1.5
                scale_ds_small = np.random.rand()*0.25+0.25
                rnd = np.random.rand()
                scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
                obj_ins = obj_ins * scale_ds + obj_ins_center
                xyz[instances == instance_idx] = obj_ins

                # change the labels of the scaled object to 20
                if self.debug:
                    #* different with the shapenet point for debug
                    labels[instances == instance_idx] = 21
                else:
                    labels[instances == instance_idx] = 20
        #!======================================================
        # write_ply_color(xyz,labels.squeeze(),'logs/debug2/scene1.ply')
        #!==========================================
        # tic12 = time.time()
        # Introduce shapenet objects
        if self.SHAPENET_ANOMALY:
            # the number of shapnet objects to introduce is from a binomial distribution
            num_of_shapenet_objects = np.random.binomial(n=20, p=0.3)#* return the success time during n times experiments, the success rate of each time is 0.3
            # introduce shapenet objects
            for spn_obj_index in range(num_of_shapenet_objects):
                
                random_path_index = np.random.randint(low=0, high=len(self.shapenet_object_paths))
                path = self.shapenet_object_paths[random_path_index]
                # Load shapenet object
                spn_points = load_shapenet_object(path)

                #!======================================================
                #* down sample 
                choices = np.random.choice(
                    spn_points.shape[0],
                    self.keep_points,#* 这个是固定值,函数接受的参数定义好了
                    replace=len(spn_points) < self.keep_points
                )
                spn_points = spn_points[choices]
                #!======================================================


                # Rotate shapenet object to upright orientation
                spn_points = rotate_shapenet_object(spn_points, dx=90, dy=0, dz=0)
                

                # Move shapenet object
                min_r = np.sqrt(
                    np.power(xyz[(labels > 0).squeeze(1)][:, 0], 2) + np.power(xyz[(labels > 0).squeeze(1)][:, 1],
                                                                               2)).min()
                max_r = np.sqrt(
                    np.power(xyz[(labels > 0).squeeze(1)][:, 0], 2) + np.power(xyz[(labels > 0).squeeze(1)][:, 1],
                                                                               2)).max()
                random_dx = np.random.uniform(low=min_r, high=max_r * 0.8)
                random_dlon = np.random.uniform(low=0, high=360)
                #!======================================================
                # write_ply(spn_points,'logs/debug2/origin_spn.ply')
                #!======================================================

                spn_points = move_shapenet_object(spn_points, dx=random_dx)
                #!======================================================
                # write_ply(spn_points,'logs/debug2/radius_move.ply')
                #!======================================================

                spn_points = move_shapenet_object_in_sphere(spn_points, dlon=random_dlon)
                #!======================================================
                # write_ply(spn_points,'logs/debug2/_0_360_move.ply')
                #!======================================================


                # Skip this object if it does not overlap with kitti
                #  points when looking from the above (if it is not on the road)
                spn_center_x = spn_points[:, 0].mean()
                spn_center_y = spn_points[:, 1].mean()
                min_dist_between_spn_and_kitti = (np.abs(spn_center_x - xyz[:, 0]) + np.abs(spn_center_y - xyz[:, 1])).min()
                if min_dist_between_spn_and_kitti > 1:
                    continue
                # Resize shapenet object
                random_resize_factor = np.random.uniform(low=1, high=7)
                spn_points = resize_shapenet_object(spn_points, resize_factor=random_resize_factor)


                # Put shapenet object on the ground
                spn_points = put_shapenet_object_on_the_ground(spn_points, xyz)

                # Add shapenet points into kitti points
                xyz, introduced_anomaly_indices = add_shapenet_objects_to_kitti_points(spn_points, xyz)


                # change labels
                # labels[introduced_anomaly_indices] = 20

                """
                generate the dynamic label for the point radius


                start from 21 to be different with the resized point label
                thus, the the possible label for synthesized point is [21,121),including 21, not 121, 
                """
                
                labels[introduced_anomaly_indices] = 21 + int((random_dx / (max_r - min_r) )*100)


        # print('the novel class generation (ShapeNet) spends time: ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic12)))

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)
        #* radius bound 
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)#* max radius 
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)#* minimal radius 
        #*  bound  of degree and z-axis ???? 
        max_bound = np.max(xyz_pol[:, 1:], axis=0)#* max degree and max z-value,  return a 2-dimension value 
        min_bound = np.min(xyz_pol[:, 1:], axis=0)#* min .. 
        
        max_bound = np.concatenate(([max_bound_r], max_bound))#* concatenate as 3-dimension vector represent the max bound of radius, degree and the z-axis
        min_bound = np.concatenate(([min_bound_r], min_bound))#* the min 


        #* use fixed volume space, so the code above is useless? 
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        
        
        """"
            # get grid index

            #* intervals decide the voxel size.  a 3-dimension vector
            specifically, 
                cur_grid_size: the  final valume size, such as [480,320,32] , 
                crop_range: current pc scene range, 
        """
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)#* 

        if (intervals == 0).any(): print("Zero interval!")
        """
        #? grid_ind : the representation of point in voxel? 
        #* from radius, degree, z-axis to  the voxel index,
        """
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array =  np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        """"
        label_voxel_pair: [radius,degree,z,label] represent the label of each voxel
        #? voxel_position : the position of voxel, [3,self.grid_size[0],self.grid_size[1],self.grid_size[2]]
        """
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :] #* sorted according to radius, degree, z, 
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)#* 
        data_tuple = (voxel_position, processed_label)

        
        """"
        # center data on each voxel for PTnet

        
        #?  what does the variable, xyz_pol - voxel_centers , use for ?
        """
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound#* transfer back into xyz_pol
        return_xyz = xyz_pol - voxel_centers 
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        """"
        grid_ind: [radius, degree, z], shape = [126756,3]
        labels: [label], shape == [126756,1]
        return_fea: the voxel feature, [?, ?, ?, radius, degree, z, x,y, intensity]
        """
        if len(data) == 3:
            return_fea = return_xyz
        elif len(data) == 4:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
        
        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        # print('loading one sample (including the novel class generation) spend time: ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

        return data_tuple

@register_dataset
class cylinder_dataset_panop_incre(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4, ds_sample=False, incre=None):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std
        self.ds_sample = ds_sample
        self.incre = incre

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 4:
            xyz, labels, instances, dis_labels = data
        elif len(data) == 5:
            xyz, labels, instances, dis_labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        if self.ds_sample:
            minimum_pts_thre = 300
            instances = instances.squeeze()
            cls, cnt = np.unique(instances, return_counts=True)
            inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]
            for instance_idx in inst_basic_idx:
                rnd = np.random.rand()
                if rnd > 0.2 or labels[instances == instance_idx][0]!=5:
                    continue

                obj_ins = xyz[instances==instance_idx]
                obj_ins_center = np.mean(obj_ins, axis=0)
                obj_ins = obj_ins - obj_ins_center
                scale_ds_large = np.random.rand()*1.5+1.5
                scale_ds_small = np.random.rand()*0.25+0.25
                rnd = np.random.rand()
                scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
                obj_ins = obj_ins * scale_ds + obj_ins_center
                xyz[instances == instance_idx] = obj_ins
                labels[instances == instance_idx] = 20

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        if self.return_test == False:
            processed_dis_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            dis_label_voxel_pair = np.concatenate([grid_ind, dis_labels], axis=1)
            dis_label_voxel_pair = dis_label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            processed_dis_label = nb_process_label(np.copy(processed_dis_label), dis_label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 4:
            return_fea = return_xyz
        elif len(data) == 5:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea, processed_dis_label)
        return data_tuple

@register_dataset
class polar_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 45) - np.pi / 8
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV_incre(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    dis_labels = np.stack([d[5] for d in data]).astype(np.int)
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, torch.from_numpy(dis_labels)

def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz

def collate_fn_BEV_val(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index

def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    path_save = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, path_save




if __name__ == "__main__":
    np.random.seed(1184)
    from config.config import load_config_data
    from dataloader.pc_dataset import get_pc_model_class
    configs  = load_config_data('config/semantickitti_ood_final.yaml')


    SemKITTI = get_pc_model_class(configs['dataset_params']['pc_dataset_type'])

    nusc=None
    if "nusc" in configs['dataset_params']['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', 
                        dataroot=configs['train_data_loader']['data_path'],
                          verbose=True)

    train_pt_dataset = SemKITTI(configs['train_data_loader']['data_path'],
                                imageset=configs['train_data_loader']['imageset'],
                                return_ref=configs['train_data_loader']['return_ref'], 
                                label_mapping=configs['dataset_params']['label_mapping'],
                                  nusc=nusc)

    train_dataset = cylinder_dataset_panop(
        train_pt_dataset,
        grid_size=configs['model_params']['output_shape'],
        flip_aug=True,
        fixed_volume_space=configs['dataset_params']['fixed_volume_space'],
        max_volume_space=configs['dataset_params']['max_volume_space'],
        min_volume_space=configs['dataset_params']['min_volume_space'],
        ignore_label=configs['dataset_params']["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True,
        ds_sample=True,
        SHAPENET_ANOMALY=True,
        shapenet_path=configs['dataset_params']["shapenet_path"]
    )

    from tqdm import tqdm
    

    bar = tqdm(enumerate(train_dataset),total=train_dataset.__len__())
    for idx,sample in bar:
        bar.update(1)
        pass
    

    
    print(f"load done ")