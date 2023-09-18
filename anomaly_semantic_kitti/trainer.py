# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py
import copy
import sys
import yaml

import os
from os.path import split, join, exists, isdir,isfile,dirname
import time
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import wandb 
from IPython import embed
warnings.filterwarnings("ignore")

sys.path.insert(0,os.getcwd())

from utils.pc_utils import *
import json 
#!  why DistributedSampler is not used ?
import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, all_reduce
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ConstantLR
import torch.nn.functional as F


import torch.distributed as dist
import spconv.pytorch as spconv
from utils.load_save_util import load_checkpoint
from utils.utils import *
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.image_utils import * 


import random
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from pad_losses.gambler_loss import *

from loguru import logger 

from IPython import embed

from pad_losses.energe_loss import * 
from semantic_kitti_api.entity.semantic_evaluator import SementicEvaluator

from utils.utils import * 

# import pyvista as pv

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score


from dataloader.dataset_semantickitti import cart2polar,polar2cat
import pandas as pd

from plyfile import PlyData, PlyElement

'''
description: 
param {*} xyz
return {*}
following code to call this function and inference the extra data

                file_dir = 'deign_scene.ply'  #文件的路径
                plydata = PlyData.read(file_dir)  # 读取文件
                data = plydata.elements[0].data
                data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
                data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
                property_names = data[0].dtype.names  # 读取property的名字
                for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
                    data_np[:, i] = data_pd[name]

                _,val_grid , val_pt_fea , idx = get_voxelize_input(data_np)
                val_grid = val_grid[None,...]
                val_pt_fea = val_pt_fea[None,...]


'''
def get_voxelize_input(xyz):


    
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
    max_bound = np.asarray([50.0, 3.1415926, 2.0])
    min_bound = np.asarray([0.0, -3.1415926, -4.0])
    
    
    """"
        # get grid index

        #* intervals decide the voxel size.  a 3-dimension vector
        specifically, 
            cur_grid_size: the  final valume size, such as [480,320,32] , 
            crop_range: current pc scene range, 
    """
    crop_range = max_bound - min_bound
    grid_size = np.array([480, 360,  32])
    cur_grid_size = grid_size

    intervals = crop_range / (cur_grid_size - 1)#* 

    if (intervals == 0).any(): print("Zero interval!")
    """
    #? grid_ind : the representation of point in voxel? 
    #* from radius, degree, z-axis to  the voxel index,
    """
    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

    voxel_position = np.zeros(grid_size, dtype=np.float32)
    dim_array =  np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
    voxel_position = polar2cat(voxel_position)

    """"
    label_voxel_pair: [radius,degree,z,label] represent the label of each voxel
    #? voxel_position : the position of voxel, [3,self.grid_size[0],self.grid_size[1],self.grid_size[2]]
    """
    #!+=============================================================
    # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
    # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
    # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :] #* sorted according to radius, degree, z,     
    # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)#* 
    # data_tuple = (voxel_position, processed_label)
    #!+=============================================================
    data_tuple = (voxel_position,)

    
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
    #!+=============================================================
    # return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
    return_fea = np.concatenate((return_xyz, np.ones([return_xyz.shape[0],1])), axis=1)
    
    #!+=============================================================
    


    #!+=============================================================
    # data_tuple += (grid_ind, labels, return_fea, index)
    data_tuple += (grid_ind, return_fea, [33])
    #!+=============================================================
    
    #* origin : voxel_position, processed_label, grid_ind, labels, return_fea, index
    #* nessessary: voxel_position, grid_ind, return_fea, index

    return data_tuple






'''
description:  calcua
param {*} config
param {*} abortion
param {*} correct
param {*} results
return {*}
'''
def bisection_method(coverages_list, abortion, correct, gt_anomaly, results):
    upper = 1.
    while True:
        mask_up = abortion <= upper
        passed_up = torch.sum(mask_up.long()).item()
        if passed_up / len(correct) * 100. < coverages_list[0]:
            upper *= 2.
        else:
            break
    test_thres = 1.
    """
    三个概念: 
    1. coverage:  手动调节threashold h 使得剩余没有absteining的样本满足这个coverage 
    2. coverage决定了剩余多少个样本,  剩余样本中, 测量有多少个是预测正确的, 多少是预测错误的就是不同coverage下的acc做的事

    """
    for coverage in coverages_list:
        mask = abortion <= test_thres #* 这个coverage下,  哪些样本没有被过滤掉,  
        passed = torch.sum(mask.long()).item() #* 这个coverage 剩余多少样本, 
        lower = 0.
        while math.fabs(passed / len(correct) * 100. - coverage) > 0.3:#* 就是搜索这个coverage 下的 mask and passed
            if passed / len(correct) * 100. > coverage:
                upper = min(test_thres, upper)
                test_thres = (test_thres + lower) / 2
            elif passed / len(correct) * 100. < coverage:
                lower = max(test_thres, lower)
                test_thres = (test_thres + upper) / 2
            mask = abortion <= test_thres
            passed = torch.sum(mask.long()).item()
        masked_correct = correct[mask]

        masked_anomaly_correct = gt_anomaly[~mask] #* pick the point predicted as anomaly 

        

        correct_data = torch.sum(masked_correct.long()).item()

        correct_anoamly_data = torch.sum(masked_anomaly_correct.long()).item()
        passed_acc = correct_data / passed
        passed_anomaly_acc = correct_anoamly_data / passed
        #* actual coverage, acc, threashold for currrent coverage,  acc for anomaly 
        results.append((passed / len(correct), passed_acc,test_thres,passed_anomaly_acc))







'''
description:  calcua
param {*} config
param {*} abortion
param {*} correct
param {*} results
return {*}
'''
def bisection_method_mIoU(coverages_list, abortion, prediction, semantic_gt, results,DATA):
    
    unique_label = np.array(list(DATA['learning_map_inv'].keys())[1:] ) - 1

    upper = 1.
    while True:
        mask_up = abortion <= upper
        passed_up = torch.sum(mask_up.long()).item()
        if passed_up / len(prediction) * 100. < coverages_list[0]:
            upper *= 2.
        else:
            break
    test_thres = 1.
    """
    三个概念: 
    1. coverage:  手动调节threashold h 使得剩余没有absteining的样本满足这个coverage 
    2. coverage决定了剩余多少个样本,  剩余样本中, 测量有多少个是预测正确的, 多少是预测错误的就是不同coverage下的acc做的事

    """
    mIoU_per_class_per_coverage = {}
    for coverage in coverages_list:
        mask = abortion <= test_thres #* 这个coverage下,  哪些样本没有被过滤掉,  
        passed = torch.sum(mask.long()).item() #* 这个coverage 剩余多少样本, 
        lower = 0.
        while math.fabs(passed / len(prediction) * 100. - coverage) > 0.3:#* 就是搜索这个coverage 下的 mask and passed
            if passed / len(prediction) * 100. > coverage:
                upper = min(test_thres, upper)
                test_thres = (test_thres + lower) / 2
            elif passed / len(prediction) * 100. < coverage:
                lower = max(test_thres, lower)
                test_thres = (test_thres + upper) / 2
            mask = abortion <= test_thres
            passed = torch.sum(mask.long()).item()

        #* ID performance
        filterred_prediction = prediction[mask]
        filterred_semantic_gt = semantic_gt[mask]
        confuse_matrix = fast_hist_crop(filterred_prediction.numpy(),filterred_semantic_gt.numpy(),unique_label)
        iou = per_class_iu(confuse_matrix)


        #* OOD performance
        #* ood label is 5 
        # uncertainty_prediction = abortion[~mask]
        # uncertainty_label = semantic_gt[~mask].clone()

        uncertainty_prediction = abortion[mask]
        uncertainty_label = semantic_gt[mask].clone()

        aupr_score = 0 
        auroc_score = 0 
        if len(uncertainty_label) != 0 and len(uncertainty_label) == len(uncertainty_prediction): 
            uncertainty_label[uncertainty_label != 5] = 0
            uncertainty_label[uncertainty_label == 5] = 1

            precision, recall, threasholds = precision_recall_curve(uncertainty_label, uncertainty_prediction)#* take long time        
            aupr_score = auc(recall, precision)
            
            fpr, tpr, _ = roc_curve(uncertainty_label, uncertainty_prediction)
            auroc_score = auc(fpr, tpr)



        mIoU_per_class_per = {}
        for label_idx, class_iou in zip(unique_label+1, iou):
            mIoU_per_class_per[DATA["labels"][DATA['learning_map_inv'][label_idx]]] = round(class_iou * 100,2)
            # print('%s : %.2f%%' % (DATA["labels"][DATA['learning_map_inv'][label_idx]], class_iou * 100))
            
        mIoU_per_class_per_coverage[coverage] = mIoU_per_class_per
        
        #* actual coverage, acc, threashold for currrent coverage,  
        results.append((passed / len(prediction), np.nanmean(iou) ,aupr_score, auroc_score, test_thres))
    return mIoU_per_class_per_coverage






class AnomalyTrainer:


    def __init__(self,args):

        self.args = args
        
        self.load_cfg()
        self.init_path()
        
        if self.args.wandb :
            self.init_wandb()

        # if self.args.local_rank  != -1:
        

        self.best_aupr = 0
        self.best_epoch = -1
        if args.eval:
            self.log(f"only eval, do not need to load model again")
            return 
            
        self.init_distributed()
        self.init_dataloader()
        gpu_id = self.args.local_rank if self.args.local_rank != -1 else 0
        torch.cuda.set_device(gpu_id)
        
        self.log(f" distributed train init done ")


        self.init_model()
        self.unknown_clss = [5]
        self.init_criterion()
        

        #* init the optimizer 
        # if DISTRIBUTED:
        self.log('World size:'+ str(torch.distributed.get_world_size()))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.my_model.parameters()), 
                               lr=self.configs['train_params']["learning_rate"])

         

        #* init the scheduler 
        #* novel scheduler 

        # world_size = torch.distributed.get_world_size()
        # warmup_lr_factor_start = 1 / world_size
        # warmup_lr_factor_end = 1
        # warmup_lr_increment_amount = (warmup_lr_factor_end - warmup_lr_factor_start) / 5

        # scheduler_0 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 0 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_1 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 1 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_2 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 2 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_3 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 3 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_4 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 4 * warmup_lr_increment_amount, total_iters=1)

        # drop_lr_every_n_epochs = 10
        # mile_stones = list(0 + np.arange(drop_lr_every_n_epochs, self.configs['train_params']['max_num_epochs'], drop_lr_every_n_epochs))
        # scheduler_5 = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
        #                                                     schedulers=[scheduler_0, scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5],
        #                                                     milestones=[1, 2, 3, 4, 5])

        #* common scheduler
        
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_freq,gamma = args.decay_weight) 


        n_iter_per_epoch = len(self.train_dataset_loader)
        warmup_epoch = -1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer = optimizer, 
                            milestones = [(m - warmup_epoch) * n_iter_per_epoch for m in args.lr_decay_epochs], 
                            gamma= args.decay_weight,
                        )



        if args.last_epoch > -1:
            for i in range(args.last_epoch):
                for j in range(n_iter_per_epoch):
                    lr_scheduler.step()


        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        #* does self.local_rank == 0 when  distributed train is not init 
        #todo does error happen when single card train
        self.my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.my_model) 
        self.my_model = DDP(self.my_model, device_ids=[self.args.local_rank], 
                                find_unused_parameters=True,broadcast_buffers = True) 

        self.log(f"init done ")

        #* save config 
        if args.local_rank ==0:
            config_save_path = join(self.save_root,'config.txt')
            if  not exists(config_save_path):
                content = '\n'.join([f"{k}: {v}" for k,v in  self.args.__dict__.items()])
                content += '\n config file items : \n\n'
                content += '\n'.join([f"{k}: {v}"  for k,v in self.configs.items()])
                with open(config_save_path,'w') as f :
                    f.write(content)
                self.log(f"config save successfully")


    def init_wandb(self):
        if self.args.local_rank ==0 or self.args.local_rank == -1 :
            wandb_info_path  = join(self.save_root,'wandb_resume_info.json')


            """"
                resume wandb :
            """
            if self.args.resume is not None and exists(wandb_info_path)  :
                with open(wandb_info_path, 'r') as f :
                    last_run_info  = json.load(f)
                run  = wandb.init(project='3d_anomaly_detection',id=last_run_info['id'], resume="must")
                
                if os.environ.get('NCCL_SOCKET_TIMEOUT') is  not None:
                    setattr(wandb.config,"NCCL_SOCKET_TIMEOUT",os.environ['NCCL_SOCKET_TIMEOUT'])
            else:
                run = wandb.init(project='3d_anomaly_detection')
                run.name = split(self.save_root)[-1]
                with open(wandb_info_path,'w') as f :
                    json.dump({
                        "id":run.id,
                        "name":run.name,
                    },f)
                
                # wandb.log({'hello':12312312})
                for k, v in self.args.__dict__.items():
                    setattr(wandb.config,k,v)


                for k,v in self.configs.items():
                    setattr(wandb.config,k,v)

                setattr(wandb.config,"save_root",self.save_root)

                if os.environ.get('NCCL_SOCKET_TIMEOUT') is  not None:
                    setattr(wandb.config,"NCCL_SOCKET_TIMEOUT",os.environ['NCCL_SOCKET_TIMEOUT'])

            self.use_wandb = True
            

            


            

    def log_wandb(self,message_dict):

        if ( self.args.local_rank ==0 or self.args.local_rank == -1 ) and hasattr(self,'use_wandb') and self.use_wandb:
            wandb.log(message_dict)
            

    def init_distributed(self):
        torch.backends.cudnn.benchmark = True

        init_process_group(backend="nccl")
        torch.autograd.set_detect_anomaly(True) 
        
        # torch.distributed.barrier()

        #* disable the printer of other node program
        # import builtins as __builtin__
        # builtin_print = __builtin__.print
        # is_master = self.args.local_rank == 0
        # def print(*args, **kwargs):
        #     force = kwargs.pop('force', False)
        #     if is_master or force:
        #         builtin_print(*args, **kwargs)

        # __builtin__.print = print

    
    def load_cfg(self):

        self.configs  = load_config_data(self.args.config_path)

        #* update the learning rate 
        self.configs['train_params']['learning_rate'] = self.args.lr

        #*=========================================================
        self.configs['train_data_loader']['batch_size'] = self.args.train_bs
        self.configs['train_data_loader']['num_workers'] = self.configs['train_data_loader']['batch_size'] if self.configs['train_data_loader']['batch_size'] < 8 else 8


        self.configs['val_data_loader']['batch_size'] = self.args.val_bs
        self.configs['val_data_loader']['num_workers'] = self.configs['val_data_loader']['batch_size'] if self.configs['val_data_loader']['batch_size'] < 8 else 8

        self.configs['train_params']['max_num_epochs'] = self.args.max_epoch
        #*=========================================================
        


        # SemKITTI_label_name = get_SemKITTI_label_name(self.configs['dataset_params']['label_mapping'])
        # self.unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        # unique_label_str = [SemKITTI_label_name[x] for x in self.unique_label  + 1]
        

    def log(self,message):
        if self.args.local_rank == 0  or self.args.local_rank  == -1:
            print(message)

    def log_progress(self,message):
        if self.args.local_rank == 0  or self.args.local_rank  == -1:
            print(message,end='\r')


    def init_path(self):
        # if self.args.local_rank == 0 :

        if self.args.resume is not None :
            
            #* skip model_epoch_x and model real name
            self.save_root = '/'.join(self.args.resume.split('/')[:-2])
            experiment_output_dir = self.save_root

            
        else:


            status = ""

            if self.args.REAL_LOSS:
                status = status + '#CE'
            if self.args.CALIBRATION_LOSS:
                status = status + '#CCE'

            if self.args.ENERGY_LOSS:
                status = status + '#E'
                
            if self.args.GAMBLER_LOSS:
                status = status + '#G'
                
                
            if self.args.SHAPENET_ANOMALY:
                status = status + '#S'


            experiment_output_dir = join(os.getcwd(),'runs', time.strftime("%Y-%m-%d-%H:%M", \
                    time.gmtime(time.time())) +f"{self.args.energy_type}#{self.args.resize_m_out}#{self.args.m_out_max}#{self.args.no_resized_point}{status}" \
                    + self.args.save_dir_suffix )
            
            # experiment_output_dir = join(os.getcwd(),'nuscenes_runs', 'dist_save_test')
            if self.args.local_rank == 0 :
                make_dir(experiment_output_dir)

            self.save_root = experiment_output_dir
        self.model_save_path = join(experiment_output_dir, 'model_best.pt')
        self.model_latest_path = join(experiment_output_dir,'model_latest.pt')

        self.log(f"save path : {experiment_output_dir}")
        self.writer = SummaryWriter(experiment_output_dir)
    
    
    '''
    description: useless 
    param {*} self
    param {*} epoch
    return {*}
    '''
    def update_model(self,epoch):

        model_name = 'model_epoch_%s'%(epoch)
        model_path = join(self.save_root,model_name,model_name+'.pt')

        assert model_path is not None  and exists(model_path)

        
        self.my_model = load_checkpoint(model_path, self.my_model)

        self.log(f"{model_path} has been loaded! ")

        return model_path


        

        




    def init_model(self):
        
        my_model = model_builder.build(self.configs['model_params'])
        

        my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, self.args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True)
        # * resume model 
        if self.args.resume is not None and exists(self.args.resume):
            my_model = load_checkpoint(args.resume, my_model)
            self.log(f" model {self.args.resume} has been resumed ")
            
        else:
            assert exists(self.configs['train_params']['model_load_path'])
            my_model = load_checkpoint(self.configs['train_params']['model_load_path'], my_model)
            self.log(f" the model, {self.configs['train_params']['model_load_path']} ,trained in stage 1 has been loaded")

        self.my_model = my_model.cuda()



    def init_dataloader(self):
        self.train_dataset_loader, self.val_dataset_loader = data_builder.build(self.configs['dataset_params'],
                                                                self.configs['train_data_loader'],
                                                                self.configs['val_data_loader'],
                                                                grid_size=self.configs['model_params']['output_shape'],
                                                                SHAPENET_ANOMALY = self.args.SHAPENET_ANOMALY,
                                                                gen_resized_point_for_train = not self.args.no_resized_point)
        

    

    def init_criterion(self):

        self.loss_func_train, lovasz_softmax = loss_builder.build_ood(wce=True, lovasz=True,
                                                                num_class=self.configs['model_params']['num_class'],
                                                                  ignore_label=self.configs['dataset_params']['ignore_label'], 
                                                                  weight=self.configs['train_params']['lamda_2'])
        
        self.loss_func_val, self.lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                            num_class=self.configs['model_params']['num_class'], 
                                                            ignore_label=self.configs['dataset_params']['ignore_label'])


        # self.gambler_loss = Gambler(reward=[4.5], pretrain=-1, device=torch.device('cuda:' + str(self.args.local_rank)),class_num = 20)
        """

        self.unknown_clss = [5]
        """
        self.gambler_loss = Gambler(reward=[4.5],
                                            device=torch.device('cuda:' + str(self.args.local_rank)),
                                            valid_class_num=19,novel_class_num=1,
                                            unknown_cls_idx=5,
                                            novel_class_list = self.unknown_clss)
        
        


    def inference_epoch(self,epoch,model_name=None):
        

        if model_name is None:
            model_name = 'model_epoch_%s'%(epoch)

        uncertainty_folder = join(self.save_root,model_name,'sequences/08','uncertainty',)
        point_predict_folder = join(self.save_root,model_name,'sequences/08','point_predict')


        
        if self.args.local_rank == 0:
            make_dir(uncertainty_folder)
            make_dir(point_predict_folder)

        torch.distributed.barrier()
        

        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        self.log(f"start to inference")
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(self.val_dataset_loader):
                val_batch_size = len(idx)
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                
                #* forward
                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final( 
                                        val_pt_fea_ten, val_grid_ten, val_batch_size, 
                                        args.dummynumber, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False)
                
                #* save 
                output_normal_dummy = F.softmax(output_normal_dummy, dim=1)
                predict_labels = torch.argmax(output_normal_dummy[:,:-1,...], dim=1).cpu().detach().numpy()
                uncertainty_scores_softmax = output_normal_dummy[:,-1,...].cpu().detach().numpy()
                
                
                #* old post-processing 
                # predict_labels = torch.argmax(output_normal_dummy[:,:-1,...], dim=1).cpu().detach().numpy()
                # uncertainty_scores_softmax = torch.nn.Softmax(dim=1)(output_normal_dummy)[:,-1,...].cpu().detach().numpy()
                
                # uncertainty_scores_softmax = output_normal_dummy[:,-1,...].cpu().detach().numpy()
                # uncertainty_scores_softmax = -( 1. * torch.logsumexp(torch.hstack([output_normal_dummy[:,1:self.unknown_clss[0],...], output_normal_dummy[:,self.unknown_clss[0]+1:-1,...]]) / 1. ,dim=1)).cpu().detach().numpy()
                # uncertainty_scores_softmax = gaussian_blur_3d(uncertainty_scores_softmax).squeeze(1).cpu().detach().numpy().cpu().detach().numpy()

                

                for count in range(val_batch_size):
                    sample_name = self.val_dataset_loader.dataset.get_name(idx[count])
                    uncertainty_save_path = join(uncertainty_folder ,sample_name + '.label')
                    point_predict_save_path = join(point_predict_folder, sample_name + '.label')

                    if exists(uncertainty_save_path) and exists(point_predict_save_path):
                        continue

                    point_predict = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]].astype(np.int32)
                    point_uncertainty_softmax = uncertainty_scores_softmax[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]

                    
                    
                    point_uncertainty_softmax.tofile(uncertainty_save_path)
                    point_predict.tofile(point_predict_save_path)

                pbar.update(1)
            
        
        


        torch.cuda.empty_cache()


    def train_epoch(self,epoch):
        self.my_model.train()

        e_loss_list, g_loss_id_list, g_loss_ood_list, normal_loss_list, dummy_loss_list, loss_list = [], [], [], [], [], []
        pbar = tqdm(self.train_dataset_loader,total=len(self.train_dataset_loader), desc='Train')
        # time.sleep(10)

        """"
        #? voxel_position : the position of voxel, [3,self.grid_size[0],self.grid_size[1],self.grid_size[2]]
        train_vox_label: [radius,degree,z,label] represent the label of each voxel
        train_grid: the representation of point in voxel? [radius, degree, z-axis]
        point_label: [label], shape == [126756,1]
        train_pt_fea: the voxel feature, [?, ?, ?, radius, degree, z, x,y, intensity]
        useful variable : train_vox_label, train_grid, train_pt_fea, 
        """
        # next(iter(self.train_dataset_loader)).shape
        
        train_batch_size =  self.train_dataset_loader.batch_size
        self.log(f"train_batch_size : {train_batch_size}")
        
        for i_iter, (voxel_position, train_vox_label, train_grid, point_label, train_pt_fea) in enumerate(pbar):            


            # Train
            train_batch_size,W,H,C=train_vox_label.shape
            # print(f"train_batch_size: {train_batch_size}")
            """"
            visilize             

            a = np.concatenate([train_pt_fea[0][:,-3][...,None],train_pt_fea[0][:,-2][...,None],train_pt_fea[0][:,-4][...,None]],1)
            write_ply_color(a,point_label[0].squeeze(),'logs/debug/pc_label_anomaly.ply')
            write_ply_color_anomaly(a,(point_label[0].squeeze()>=20).astype(np.int32),'logs/debug/pc_label_anomaly_high_light.ply')
            tmp = point_label[0].squeeze().copy()
            tmp[tmp<20] = 0 
            remap_dict = {v:idx for idx,v in enumerate(np.unique(tmp))}
            remap_tmp =np.array([remap_dict[x] for x in tmp])
            
            
            write_ply_color(a,tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy.ply')
            write_ply_color(a,remap_tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy_remapped.ply')
            
            """


            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in train_pt_fea]#* [N,9]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).cuda() for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).cuda() for i in train_grid]#* [N,3]
            long_rang_point_label_tensor = train_vox_label.type(torch.LongTensor).cuda() #* [2, 480, 360, 32]

            
            unknown_clss = [5]
            noval_clas = 20

            """
                long_rang_point_label_tensor: including the [21,120] synthesis label, and label 20 for resized point 
                turn point_label_tensor back into [0,20] for REAL loss computation
            """
            point_label_tensor = copy.deepcopy(long_rang_point_label_tensor)
            point_label_tensor[point_label_tensor >= 20] =  noval_clas


            for unknown_cls in unknown_clss:
                point_label_tensor[point_label_tensor == unknown_cls] = 0 #? why 

            
            energy_point_label_tensor = copy.deepcopy(point_label_tensor)
            """
                all of the synthesis/shapent object label are set as unknown_clss[0], 5. 
                so the prediction of synthesis/shapent are all aligned to 5 in REAL CE loss. 
                so the unknown class prediction process become predict the proability of the point belong to the 5-th class

                #!  so the 5-th channel of prediction represent the unknown class proability in CE loss. 
                在最初版的REAL, the 5-th classifier  其实是一个redundancy classifier, 但是在修改后的版本, 变成了novel class classifier... 
            """

            energy_point_label_tensor[energy_point_label_tensor == noval_clas] = unknown_clss[0]

            gambler_point_label_tensor = copy.deepcopy(energy_point_label_tensor)

            #* coor_ori: [N,4], the [N,0]is the batch index while the [N,1:] is the voxel coordinate
            coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                            train_batch_size,self.args.dummynumber)
            

            if self.args.ENERGY_LOSS or self.args.GAMBLER_LOSS:
                logits_for_loss_computing = torch.hstack(
                    [   
                        output_normal_dummy[:, :unknown_clss[0]], 
                        output_normal_dummy[:, -1:],
                        output_normal_dummy[:, unknown_clss[0]+1:-1]
                    ])

            #* using the index of predict voxel label to query the  gt voxel label, reognisation as the prediction
            voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            if self.args.REAL_LOSS:
                loss_normal = self.loss_func_train(output_normal_dummy, point_label_tensor)
            else:
                loss_normal = torch.tensor(0).cuda()
           

            if self.args.CALIBRATION_LOSS: 
                """
                preparson for the calibration loss 
                1. mask the true positive 
                2. generate the dummy label  
                3. calculation the calibration loss 
                """
                output_normal_dummy = output_normal_dummy.permute(0,2,3,4,1)
                output_normal_dummy = output_normal_dummy[coor_ori.permute(1,0).chunk(chunks=4, dim=0)].squeeze()
                index_tmp = torch.arange(0,coor_ori.shape[0]).unsqueeze(0).cuda()
                #* only consider the common class, let the prediction  logit of common class has second largest logit for dummy class
                voxel_label_origin[voxel_label_origin == noval_clas] = 0 
                index_tmp = torch.cat([index_tmp, voxel_label_origin], dim=0)
                output_normal_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9 #* mask  the true positive 
                #* generate the dummy label
                label_dummy = torch.ones(output_normal_dummy.shape[0]).type(torch.LongTensor).cuda()*noval_clas 
                label_dummy[voxel_label_origin.squeeze() == 0] = 0 #* the ignore label, namely, the 0-th class 
                #* calcualtion the calibration loss 
                loss_dummy = self.loss_func_train(output_normal_dummy, label_dummy)
            else:
                loss_dummy = torch.tensor(0).cuda()

            
            """"
            # Energy loss
            #* Labels for energy loss: Do not use class 5. 
            #* Use scaled id objets (class 20) as ood objects during training
            
            """
            if self.args.ENERGY_LOSS:


                if self.args.energy_type == "origin":
                    e_loss, _ = energy_loss(logits_for_loss_computing, 
                                            energy_point_label_tensor,
                                            ood_ind=unknown_clss[0])
                elif self.args.energy_type == "dynamic":
                    e_loss, _ = dynamic_energy_loss(logits_for_loss_computing, 
                                    energy_point_label_tensor,
                                    ood_ind=unknown_clss[0],
                                    details_targets = long_rang_point_label_tensor,
                                    m_out_max = self.args.m_out_max,
                                    resized_point_label = noval_clas)
                elif self.args.energy_type == "crude_dynamic":
                    e_loss, _ = crude_dynamic_energy_loss(logits_for_loss_computing, 
                                    energy_point_label_tensor,
                                    ood_ind=unknown_clss[0],
                                    details_targets = long_rang_point_label_tensor,
                                    m_out_max = self.args.m_out_max,
                                    resized_point_label = noval_clas,
                                    resize_m_out=self.args.resize_m_out)

                else:
                    raise Exception

                                        
            else:
                e_loss = torch.tensor(0).cuda()


            #* Gambler loss/ abstention loss
            if self.args.GAMBLER_LOSS:
                num_ood_samples = torch.sum((gambler_point_label_tensor == 5),
                                            dim=tuple(np.arange(len(gambler_point_label_tensor.shape))[1:]))
                is_ood = num_ood_samples > 0
                in_logits, in_target = logits_for_loss_computing[~is_ood], gambler_point_label_tensor[~is_ood]
                out_logits, out_target = logits_for_loss_computing[is_ood], gambler_point_label_tensor[is_ood]
                # 1. in distribution
                if in_logits.shape[0] > 0:
                    g_loss_id = self.gambler_loss(pred=in_logits, 
                                                  targets=in_target, wrong_sample=False)
                else:
                    g_loss_id = torch.tensor(0).cuda()
                # 2. out-of-distribution
                if torch.any(is_ood):
                    g_loss_ood = self.gambler_loss(pred=out_logits, 
                                                   targets=out_target, wrong_sample=True)
                else:
                    g_loss_ood = torch.tensor(0).cuda()
            else:
                g_loss_id = torch.tensor(0).cuda()
                g_loss_ood = torch.tensor(0).cuda()


            """"
            loss gradient statistics
            #!loss_normal
                #! 1
                self.optimizer.zero_grad()
                loss_normal.backward()
                gradient_list  = np.array([ x.grad.mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ) #* 147
                print(gradient_list.shape)
                print("gradient_list.mean(): ",gradient_list.mean())#* 0.00027712295 ~= 1e-4 
                print("0.25 quantile: ",np.quantile(gradient_list, 0.25)) #* 4.615521191908556e-07
                print("0.50 quantile: ",np.quantile(gradient_list, 0.50)) #* 1.2568330930662341e-05
                print("0.75 quantile: ", np.quantile(gradient_list, 0.75)) #*  9.465916082262993e-05
            
            #!loss_dummy
                #! 1e-1
                self.optimizer.zero_grad()
                loss_dummy.backward()
                gradient_list  = np.array([ x.grad.mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ) #* 147
                print(gradient_list.shape)
                print("gradient_list.mean(): ",gradient_list.mean())#* 0.0056795147 ~= 1e-3
                print("0.25 quantile: ",np.quantile(gradient_list, 0.25)) #*  5.040372343501076e-06
                print("0.50 quantile: ",np.quantile(gradient_list, 0.50)) #* 9.880780999083072e-05
                print("0.75 quantile: ", np.quantile(gradient_list, 0.75)) #*  0.0006744059501215816 ~= 1e-4
                

            #!e_loss         
                #! 1e-2       
                self.optimizer.zero_grad()
                e_loss.backward()
                gradient_list  = np.array([ x.grad.mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ) #* 147
                print(gradient_list.shape)
                print("gradient_list.mean(): ",gradient_list.mean())#* 0.011877371 ~= 1e-2
                print("0.25 quantile: ",np.quantile(gradient_list, 0.25)) #*  2.3672482711845078e-05 
                print("0.50 quantile: ",np.quantile(gradient_list, 0.50)) #*  0.0004951321752741933 ~= 1e-4
                print("0.75 quantile: ", np.quantile(gradient_list, 0.75)) #*  0.002407068503089249 ~= 1e-3
            
           #!g_loss_ood   
                #! 1 
                self.optimizer.zero_grad()
                g_loss_ood.backward()
                gradient_list  = np.array([ x.grad.mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ) #* 147
                print(gradient_list.shape)
                print("gradient_list.mean(): ",gradient_list.mean())#*  0.0007919007 ~= 1e-4
                print("0.25 quantile: ",np.quantile(gradient_list, 0.25)) #*  6.861129406843247e-07 
                print("0.50 quantile: ",np.quantile(gradient_list, 0.50)) #*   3.9361377275781706e-05
                print("0.75 quantile: ", np.quantile(gradient_list, 0.75)) #*  0.00020090502221137285 ~= 1e-4

            #!e_loss   
                self.optimizer.zero_grad()
                e_loss.backward()
                gradient_list  = np.array([ x.grad.mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ) #* 147
                gradient_list.mean(): 0.016685408

            """




            #* self.configs['train_params']['lamda_1'] : 1e-1
            """
            #? does the e_loss weight need to be 1e-2 as analysis of above. 

            1e-2 and 1e-3 都会出现nan
            
            """
            if torch.isnan(e_loss):
                self.log(f"e_loss: {e_loss}, only nan is printed")
                loss = (loss_normal + self.configs['train_params']['lamda_1'] * loss_dummy) +\
                    (g_loss_id + g_loss_ood)
            else:
                loss = (loss_normal + self.configs['train_params']['lamda_1'] * loss_dummy) +\
                    (0.1 * e_loss + g_loss_id + g_loss_ood)


            self.optimizer.zero_grad()
            loss.backward()
            
            #!========================================================   
            # if self.args.clip_norm > 0:
            #     grad_total_norm = torch.nn.utils.clip_grad_norm_(
            #         self.my_model.parameters(), self.args.clip_norm
            #     )
            #     self.log_wandb({"Misc/grad_norm": grad_total_norm})
            # else:
            #     self.log_wandb({'mean gradients: ':
            #                     np.array([ x.grad.clone().mean().abs().cpu().numpy() for x in self.my_model.parameters() if x.grad is not None ] ).mean()})
            #!========================================================
            self.optimizer.step()
            
            loss_list.append(loss.item())
            normal_loss_list.append(loss_normal.item())
            dummy_loss_list.append(loss_dummy.item())
            if not torch.isnan(e_loss):
                e_loss_list.append(e_loss.item())
            g_loss_id_list.append(g_loss_id.item())
            g_loss_ood_list.append(g_loss_ood.item())

            #!========================================================
            # self.log_wandb(
            #     {
            #         "EpochLoss/loss_normal":loss_normal.item(),
            #         "EpochLoss/loss_dummy":(self.configs['train_params']['lamda_1'] * loss_dummy).item(),
            #         "EpochLoss/e_loss":(0.1 * e_loss).item(),
            #         "EpochLoss/g_loss_id":g_loss_id.item(),
            #         "EpochLoss/g_loss_ood":g_loss_ood.item(),
            #     }
            # )
            #!========================================================

            pbar.update(1)
            self.lr_scheduler.step()
            
        #*  collect loss information and record
        # if DISTRIBUTED:
        # torch.distributed.barrier()
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

        if self.args.local_rank == 0 or self.args.local_rank == -1:


            self.writer.add_scalar('Loss/train/energy_loss', tb_energy_loss, epoch)
            self.writer.add_scalar('Loss/train/g_loss_id', tb_g_loss_id.mean(), epoch)
            self.writer.add_scalar('Loss/train/g_loss_ood', tb_g_loss_ood, epoch)
            self.writer.add_scalar('Loss/train/total_loss', tb_total_loss, epoch)
            self.writer.add_scalar('Loss/train/real_normal', tb_real_normal, epoch)
            self.writer.add_scalar('Loss/train/real_calibration', tb_real_calibration, epoch)
            self.writer.add_scalar('Misc/LR', self.optimizer.param_groups[0]["lr"], epoch)

            self.log_wandb({
                "Loss/train/energy_loss":tb_energy_loss,
                "Loss/train/g_loss_id":tb_g_loss_id.mean(),
                "Loss/train/g_loss_ood":tb_g_loss_ood,
                "Loss/train/total_loss":tb_total_loss,
                "Loss/train/real_normal":tb_real_normal,
                "Loss/train/real_calibration":tb_real_calibration,
                "Misc/LR":self.optimizer.param_groups[0]["lr"],
            })
        # else:
            # normalize losses with respect to batch size
            # normalization_factor = train_batch_size
            # self.writer.add_scalar('Loss/train/energy_loss', np.array(e_loss_list).mean() / normalization_factor, epoch)
            # self.writer.add_scalar('Loss/train/g_loss_id', np.array(g_loss_id_list).mean() / normalization_factor, epoch)
            # self.writer.add_scalar('Loss/train/g_loss_ood', np.array(g_loss_ood_list).mean()/ normalization_factor, epoch)
            # self.writer.add_scalar('Loss/train/total_loss', np.array(loss_list).mean() / normalization_factor, epoch)
            # self.writer.add_scalar('Loss/train/real_normal', np.array(normal_loss_list).mean() / normalization_factor, epoch)
            # self.writer.add_scalar('Loss/train/real_calibration', np.array(dummy_loss_list).mean() / normalization_factor, epoch)
            # self.writer.add_scalar('Misc/LR', self.optimizer.param_groups[0]["lr"], epoch)
        torch.cuda.empty_cache()
                
    def save_ckpt(self,model_name):
            
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            #* , lr_scheduler, my_model,optimizer
            model_state = self.my_model.state_dict() if not hasattr(self.my_model,'module') else self.my_model.module.state_dict()
            model_path = join(self.save_root,model_name,model_name+'.pt')
            make_dir(dirname(model_path))

            torch.save(model_state,model_path)

            self.log(f'model save at {model_path} successfully')
        
            
        
    def train(self):

        #* distributed training for model 

        # training
        epoch = self.args.last_epoch + 1
        while epoch < self.configs['train_params']['max_num_epochs']:
            self.train_dataset_loader.sampler.set_epoch(epoch)
            self.val_dataset_loader.sampler.set_epoch(epoch)
            self.log(f"len(self.train_dataset_loader): {len(self.train_dataset_loader)}; len(self.val_dataset_loader): {len(self.val_dataset_loader)}")
            #* for the error process 
            with torch.autograd.detect_anomaly():
                self.train_epoch(epoch)
            
            #* save model 
            self.save_ckpt(model_name = 'model_epoch_%s'%(epoch))

            self.inference_epoch(epoch)

            # torch.distributed.barrier()
            # if self.args.local_rank == -1 or self.args.local_rank == 0 : 
            # torch.distributed.barrier()
            self.log_wandb({"epoch":epoch})

            epoch += 1
            # self.lr_scheduler.step()

    def eval_epoch(self,epoch,model_name = None):

        if model_name is None:
            model_name = 'model_epoch_%s'%(epoch)
        res_json_path = join(self.save_root,model_name,'anomaly_eval_results.json')

            
        if not exists(res_json_path):
            tic = time.time()
            
            self.evaluator = SementicEvaluator(self.configs['dataset_params']['semantic_kitti_root'],
                                            prediction_path=join(self.save_root,model_name),
                                            data_config_path='semantic_kitti_api/config/semantic-kitti.yaml',
                                            split='valid')
            
            self.evaluator()
            print('eval spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))
        
        # * only test not eval
        with open(res_json_path,'r')as f :
            data  = json.load(f)
        self.log_wandb(data)
        
        if self.best_aupr < data['OOD/AUPR']:
            self.best_aupr = data['OOD/AUPR']
            self.log_wandb({'Misc/best_aupr':self.best_aupr,'Misc/best_epoch':epoch})
            self.best_epoch = epoch
            self.log('best_aupr : %s \t best_epoch : %d '%(self.best_aupr,epoch))
        else:
            #todo remove the model 
            model_path = join(self.save_root,model_name,model_name+'.pt')
            prediction_path = join(self.save_root,model_name,'sequences')
            if exists(model_path):
                os.remove(model_path)

            if exists(prediction_path):
                shutil.rmtree(prediction_path)


            
            
            
        
    def eval(self,max_epoch = 10):
        

        if  self.args.last_epoch == -1 :
            self.log(f'please give the first model path before evaluation')

        epoch = self.args.last_epoch

        #* given first model then, update to the following model
        
        while epoch < max_epoch:            
            self.log(f"eval {epoch}-th epoch ")
            self.eval_epoch(epoch)
            epoch+=1
            
        self.log('best_aupr : %s \t best_epoch : %d '%(self.best_aupr,self.best_epoch))


        
    def eval_one_epoch(self):
        
        #* update save root due to the wrong save root 
        path_parts = self.args.resume.split('/')
        self.reset_save_root('/'.join(path_parts[:-1]))

        model_name = path_parts[-1].split('.')[-2]
        
        #* check where the inference is done 
        uncertainty_folder = join(self.save_root,model_name,'sequences/08','uncertainty',)
        point_predict_folder = join(self.save_root,model_name,'sequences/08','point_predict')

        if not (exists(uncertainty_folder) and exists(point_predict_folder) \
            and self.val_dataset_loader.dataset.__len__() == len(os.listdir(point_predict_folder)) and \
                self.val_dataset_loader.dataset.__len__() == len(os.listdir(uncertainty_folder))):
            self.inference_epoch(self.args.last_epoch,model_name=model_name)

        if dist.get_rank()==0:
            self.eval_epoch(self.args.last_epoch,model_name = model_name )
        
        self.log('best_aupr : %s \t best_epoch : %d '%(self.best_aupr,self.best_epoch))


    def get_mIoU(self):

        """
        1. compute acc/mIoU for closed-set prediction 
        2. compute 
        
        """
        tic = time.time()
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        hist_list = []
        val_loss_list = []
        

        """
            这个代码存在覆盖... 
        """

        
        DATA = yaml.safe_load(open(self.configs['dataset_params']['label_mapping'], 'r'))
        unique_label = np.array(list(DATA['learning_map_inv'].keys())[1:] ) - 1

    
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) \
                in enumerate(self.val_dataset_loader):
                val_batch_size = len(idx)

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final(
                    val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber)
                predict_labels = output_normal_dummy[:,:-1,...]

                # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                    ignore=0) + self.loss_func_val(predict_labels.detach(), val_label_tensor)
                uncertainty_scores_logits = output_normal_dummy[:,-1,...].cpu().detach().numpy()

                softmax_layer = torch.nn.Softmax(dim=1)
                uncertainty_scores_softmax = softmax_layer(output_normal_dummy)[:,-1,...].cpu().detach().numpy()
                predict_labels = torch.argmax(predict_labels, dim=1).cpu().detach().numpy()


                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(\
                        predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]], 
                        val_pt_labs[count],unique_label))
                
                val_loss_list.append(loss.detach().cpu().numpy())
                pbar.update(1)
                if i_iter_val == 3 :
                    break

        """
        #* calculation acc 
        the output size of sum(hist_list) : [19,19],
        iou size : [19]
        """
        sum_res = torch.tensor(sum(hist_list)).cuda().contiguous()
        dist.all_reduce(sum_res, op=torch.distributed.ReduceOp.SUM)
            
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten
        
        if dist.get_rank() == 0:
            
            iou = per_class_iu(sum_res.cpu().numpy())

            print('Validation per class iou: ')

            for label_idx, class_iou in zip(unique_label+1, iou):
                print('%s : %.2f%%' % (DATA["labels"][DATA['learning_map_inv'][label_idx]], class_iou * 100))



            print('Current val miou is %.3f' %(np.nanmean(iou) * 100))
            print('Current val miou is %.3f' %(np.nanmean(np.array(iou.tolist() + [0])) * 100 ))
            print('Current val loss is %.3f' %(np.mean(val_loss_list)))

        print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))
            





    def get_pixel_wise_absteining_acc(self):

        """
        1. compute acc/mIoU for closed-set prediction 
        2. compute 
        
        """
        def collect_data(data,collect_operation = torch.distributed.ReduceOp.SUM):
            tmp = torch.tensor(data).cuda().contiguous()
            dist.all_reduce(tmp, op=collect_operation)

            return tmp


        self.log(f"self.save_root: {self.save_root,}")

        tic = time.time()
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        hist_list = []
        val_loss_list = []
        

        """
            这个代码存在覆盖... 
        """

        
        DATA = yaml.safe_load(open(self.configs['dataset_params']['label_mapping'], 'r'))
        unique_label_without_novel_class = list(DATA['learning_map_inv'].keys())[1:].copy()
        unique_label_without_novel_class.remove(5)
        unique_label_without_novel_class = np.array(unique_label_without_novel_class)-1

        unique_label = np.array(list(DATA['learning_map_inv'].keys())[1:] ) - 1

        """
        consider absteining:
            1. collect all of the  uncertainty score and prediction label 


        """
        
        abortion_results = [[], [],[]]

        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) \
                in enumerate(self.val_dataset_loader):
                val_batch_size = len(idx)

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final(
                    val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber)
                predict_labels = output_normal_dummy[:,:-1,...]

                # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                    ignore=0) + self.loss_func_val(predict_labels.detach(), val_label_tensor)
                uncertainty_scores_logits = output_normal_dummy[:,-1,...].cpu().detach().numpy()

                softmax_layer = torch.nn.Softmax(dim=1)
                uncertainty_scores_softmax = softmax_layer(output_normal_dummy)[:,-1,...].cpu().detach().numpy()
                predict_labels_in_tensor = torch.argmax(predict_labels, dim=1).cpu().detach()
                predict_labels = predict_labels_in_tensor.numpy()


                #* collect data for risk-coverage curve
                # mask = torch.any(torch.stack([val_vox_label ==  unknown for unknown in self.unknown_clss] + [val_vox_label == 0]),axis=0)
                
                mask = val_vox_label == 0
                abortion_results[0].extend(uncertainty_scores_softmax[~mask])
                abortion_results[1].extend(predict_labels_in_tensor[~mask] ==  val_vox_label[~mask])
                abortion_results[2].extend(val_vox_label[~mask]  == self.unknown_clss[0])
                
                
                
                
            

                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(\
                        predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]], 
                        val_pt_labs[count],
                        unique_label))


                # for count, i_val_grid in enumerate(val_grid):
                #     hist_list.append(fast_hist_crop(\
                #         predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]], 
                #         val_pt_labs[count],
                #         unique_label_without_novel_class))


                val_loss_list.append(loss.detach().cpu().numpy())
                
                # if i_iter_val == 10:
                #     break
                pbar.update(1)

        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        abortion = torch.tensor(abortion_results[0])
        correct = torch.tensor(abortion_results[1]) #* mean: whether the prediction is correct or not 
        gt_anomaly = torch.tensor(abortion_results[2]) #* mean: whether the prediction is correct or not 

        results = []
        coverages_list = [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.]
        #* can not collect from other ranks
        bisection_method(coverages_list, abortion, correct, gt_anomaly,results)


        with open(os.path.join(self.save_root, 'coverage_VS_err.csv'), 'w') as f :
            for idx, result in enumerate(results):
                """
                #* error is equal to (1 - acc )
                #* the 3 columes is :  
                    target coverage 
                    actual coverage 
                    risk at actual coverage 
                    the threshold for current coverage
                """
                f.write('test{:.0f},{:.2f},{:.3f},{:.3f},{:.3f}\n'.format(coverages_list[idx],results[idx][0]*100.,(1-results[idx][1])*100,results[idx][2],(1 - results[idx][3]) * 100))




        

        

        """
        #* calculation acc 
        the output size of sum(hist_list) : [19,19],
        iou size : [19]
        """
        # sum_res = torch.tensor(sum(hist_list)).cuda().contiguous()
        # dist.all_reduce(sum_res, op=torch.distributed.ReduceOp.SUM)
        sum_res = collect_data(sum(hist_list))
            
        if dist.get_rank() == 0:
            
            iou = per_class_iu(sum_res.cpu().numpy())

            print('Validation per class iou: ')

            for label_idx, class_iou in zip(unique_label+1, iou):
                print('%s : %.2f%%' % (DATA["labels"][DATA['learning_map_inv'][label_idx]], class_iou * 100))

            # for label_idx, class_iou in zip(unique_label_without_novel_class+1, iou):
            #     print('%s : %.2f%%' % (DATA["labels"][DATA['learning_map_inv'][label_idx]], class_iou * 100))

            print('Current val miou is %.3f' %(np.nanmean(iou) * 100))
            print('Current val loss is %.3f' %(np.mean(val_loss_list)))

        print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))
            







    def get_pixel_wise_absteining_mIoU(self):

        """
        1. compute acc/mIoU for closed-set prediction 
        2. compute 
        
        """
        def collect_data(data,collect_operation = torch.distributed.ReduceOp.SUM):
            tmp = torch.tensor(data).cuda().contiguous()
            dist.all_reduce(tmp, op=collect_operation)

            return tmp


        # self.log(f"self.save_root: {self.save_root,}")

        tic = time.time()
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        hist_list = []
        val_loss_list = []
        

        """
            这个代码存在覆盖... 
        """
        DATA = yaml.safe_load(open(self.configs['dataset_params']['label_mapping'], 'r'))
        # unique_label_without_novel_class = list(DATA['learning_map_inv'].keys())[1:].copy()
        # unique_label_without_novel_class.remove(5)
        # unique_label_without_novel_class = np.array(unique_label_without_novel_class)-1

        unique_label = np.array(list(DATA['learning_map_inv'].keys())[1:] ) - 1

        """
        consider absteining:
            1. collect all of the  uncertainty score and prediction label 


        """
        
        abortion_results = [[], [],[]]

        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) \
                in enumerate(self.val_dataset_loader):
                val_batch_size = len(idx)

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final(
                    val_pt_fea_ten, val_grid_ten, val_batch_size, args.dummynumber)
                

                # loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                #                     ignore=0) + self.loss_func_val(predict_labels.detach(), val_label_tensor)
                uncertainty_scores_logits = output_normal_dummy[:,-1,...].cpu().detach().numpy()
                output_normal_dummy = F.softmax(output_normal_dummy)
                predict_labels = output_normal_dummy[:,:-1,...]
                
                
                uncertainty_scores_softmax = output_normal_dummy[:,-1,...].cpu().detach().numpy()
                predict_labels_in_tensor = torch.argmax(predict_labels, dim=1).cpu().detach()
                predict_labels = predict_labels_in_tensor.numpy()


                #* collect data for risk-coverage curve
                mask = val_vox_label != 0
                abortion_results[0].extend(uncertainty_scores_softmax[mask])
                abortion_results[1].extend(predict_labels_in_tensor[mask])
                abortion_results[2].extend(val_vox_label[mask])
                
                # abortion_results[1].extend(predict_labels_in_tensor[~mask] ==  val_vox_label[~mask])
                # abortion_results[2].extend(val_vox_label[~mask]  == self.unknown_clss[0])
                

                # for count, i_val_grid in enumerate(val_grid):
                #     hist_list.append(fast_hist_crop(\
                #         predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]], 
                #         val_pt_labs[count],
                #         unique_label))


                # val_loss_list.append(loss.detach().cpu().numpy())                

                # if i_iter_val == 5:
                #     break
                pbar.update(1)
            

        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        # correct = torch.tensor(abortion_results[1]) #* mean: whether the prediction is correct or not 
        # gt_anomaly = torch.tensor(abortion_results[2]) #* mean: whether the prediction is correct or not 

        results = []
        coverages_list = [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.]
        #* can not collect from other ranks
        # bisection_method(coverages_list, abortion, correct, gt_anomaly,results)
        bisection_method_mIoU(coverages_list, torch.tensor(abortion_results[0]), torch.tensor(abortion_results[1]), 
                            torch.tensor(abortion_results[2]),results,DATA)
        

        with open(os.path.join(self.save_root, 'coverage_VS_err_debug.csv'), 'w') as f :
            for idx, result in enumerate(results):
                """
                #* error is equal to (1 - acc )
                #* the 3 columes is :  
                    target coverage 
                    actual coverage 
                    risk at actual coverage / mIoU
                    the threshold for current coverage
                """
                f.write('test{:.0f},{:.2f},{:.3f},{:.3f},{:.3f},{}\n'.format(coverages_list[idx],results[idx][0]*100.,results[idx][1] * 100,results[idx][2]*100, results[idx][3] * 100,results[idx][4] ))

        self.log('total spend  time  : ' + time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))
            



    def statistic_data(self):

        """
        1. compute acc/mIoU for closed-set prediction 
        2. compute 
        
        """
        tic = time.time()
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))


        """
            这个代码存在覆盖... 
        """
        hist = torch.zeros([20])
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) \
                in enumerate(self.val_dataset_loader):
                
                hist += torch.histc(val_vox_label.float(), bins=20, min=0, max=19)
                
                pbar.update(1)

        """
        #* calculation acc 
        the output size of sum(hist_list) : [19,19],
        iou size : [19]
        """


        hist = torch.tensor(hist).cuda().contiguous()
        dist.all_reduce(hist, op=torch.distributed.ReduceOp.SUM)
        
        for idx in range(20):
            print(f"{idx}: {hist[idx]}")
        




    def risk_coverage(self):
        pass




    def reset_save_root(self,path):
        self.save_root = path


class DebugTrainer(AnomalyTrainer):
    

    def __init__(self,args):
        super(DebugTrainer,self).__init__(args)
        self.debug_save_dir = join(f'logs/analysis/{time.strftime("%Y-%m-%d-%H:%M",time.gmtime(time.time()))}')
        os.makedirs(self.debug_save_dir)

        


    def init_dataloader(self):
        self.train_dataset_loader, self.val_dataset_loader = data_builder.build(self.configs['dataset_params'],
                                                                self.configs['train_data_loader'],
                                                                self.configs['val_data_loader'],
                                                                grid_size=self.configs['model_params']['output_shape'],
                                                                SHAPENET_ANOMALY = self.args.SHAPENET_ANOMALY,
                                                                debug = self.args.debug)
        

    
    def debug(self):
        self.inference_epoch(self.args.last_epoch)
        self.inference_trainset_epoch(self.args.last_epoch)
        
        # self.inference_epoch_save_energy_anomaly(self.args.last_epoch)
        # self.eval(max_epoch = self.args.last_epoch+1)
        
        



    '''
    description: 
    param {*} self
    param {*} pred_logit
    return {*}
    '''
    def get_energy(self,pred_logit,targets,shapenet_cls_id,resized_cls_id,true_anomaly_id=5):

        
        # logits_for_loss_computing = torch.hstack(
        #     [   
        #         pred_logit[:, :self.unknown_clss[0]],
        #         pred_logit[:, -1:],
        #         pred_logit[:, self.unknown_clss[0]+1:-1]
        #     ])
        T = 1.
        #!================================================
        # in_distribution_logits = pred_logit[:,:-1]
        in_distribution_logits = torch.hstack([pred_logit[:,1:true_anomaly_id,...],pred_logit[:,true_anomaly_id+1:-1,...]])

        

        #!================================================
        
        energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))


        Ec_shapenet = energy[targets==shapenet_cls_id]
        Ec_resised = energy[targets == resized_cls_id]
        Ec_true_anomaly = energy[targets == true_anomaly_id]
        Ec_in = energy[(targets != resized_cls_id) &  (targets != shapenet_cls_id) & (targets != 0) & (targets != true_anomaly_id)]

        return Ec_shapenet,Ec_resised,Ec_true_anomaly,Ec_in






    '''
    description: save energy statistic plot
    param {*} self
    param {*} epoch
    return {*}
    '''
    def inference_testset_epoch(self,epoch):
        torch.distributed.barrier()
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        self.log(f"start to inference")

        Ec_shapenet_all = torch.tensor([])
        Ec_resised_all = torch.tensor([])
        Ec_true_anomaly_all = torch.tensor([])
        Ec_in_all = torch.tensor([])

        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(self.val_dataset_loader):
                val_batch_size = len(idx)

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                
                #* forward
                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final( 
                                        val_pt_fea_ten, val_grid_ten, val_batch_size, 
                                        args.dummynumber, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False)
                
                #* get energy
                #!=====================
                val_vox_label[val_vox_label>20] = 21 #* remap shapenet point
                #!=====================

                Ec_shapenet,Ec_resised,Ec_true_anomaly,Ec_in = self.get_energy(output_normal_dummy,val_vox_label, shapenet_cls_id = 21, resized_cls_id = 20)
                Ec_resised_all = torch.cat([Ec_resised_all,Ec_resised.detach().cpu()])
                Ec_shapenet_all = torch.cat([Ec_shapenet_all,Ec_shapenet.detach().cpu()])
                Ec_true_anomaly_all = torch.cat([Ec_true_anomaly_all,Ec_true_anomaly.detach().cpu()])
                Ec_in_all = torch.cat([Ec_in_all,Ec_in.detach().cpu()])

                pbar.update(1)
                if i_iter_val ==5 :
                    break
                
        save_dir = join(self.debug_save_dir,'val')
        os.makedirs(save_dir)
        
        plt.hist(Ec_in_all, bins=100, color='green',label='ID')
        plt.legend()
        plt.savefig(join(save_dir,'ID.png'))
        plt.cla()

        plt.hist(Ec_shapenet_all, bins=100, color='blue',label='shapenet point')
        plt.legend()
        plt.savefig(join(save_dir,'shapenet.png'))
        plt.cla()

        plt.hist(Ec_resised_all, bins=100, color='red',label='resize point')
        plt.legend()
        
        plt.savefig(join(save_dir,'resized.png'))
        plt.cla()

        plt.hist(Ec_true_anomaly_all, bins=100, color='lightblue',label='true anomaly (5)')
        plt.legend()
        plt.savefig(join(save_dir,'true_anomaly.png'))
        plt.cla()


        plt.hist(Ec_in_all, bins=100, color='green',label='ID')
        plt.hist(Ec_shapenet_all, bins=100, color='blue',label='shapenet point')
        plt.hist(Ec_resised_all, bins=100, color='red',label='resize point')
        plt.hist(Ec_true_anomaly_all, bins=100, color='lightblue',label='true anomaly (5)')
        plt.legend()
        plt.savefig(join(save_dir,'all.png'))
        plt.cla()
        
        torch.cuda.empty_cache()

    
    '''
    description: ineference and draw  energy 
    param {*} self
    param {*} epoch
    return {*}
    '''
    def inference_trainset_epoch(self,epoch):
        
        torch.distributed.barrier()

        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.train_dataset_loader))

        self.my_model.eval()
        self.log(f"start to inference")

        Ec_shapenet_all = torch.tensor([])
        Ec_resised_all = torch.tensor([])
        Ec_true_anomaly_all = torch.tensor([])
        Ec_in_all = torch.tensor([])

        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(self.train_dataset_loader):
                val_batch_size = val_vox_label.shape[0]

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                
                #* forward
                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final( 
                                        val_pt_fea_ten, val_grid_ten, val_batch_size, 
                                        args.dummynumber, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False)
                
                
                #!================================================================================
                val_vox_label[val_vox_label>20] = 21 #* remap shapenet point
                #!================================================================================


                #* get energy
                Ec_shapenet,Ec_resised,Ec_true_anomaly,Ec_in = self.get_energy(output_normal_dummy,val_vox_label, shapenet_cls_id = 21, resized_cls_id = 20)
                Ec_resised_all = torch.cat([Ec_resised_all,Ec_resised.detach().cpu()])
                Ec_shapenet_all = torch.cat([Ec_shapenet_all,Ec_shapenet.detach().cpu()])
                Ec_true_anomaly_all = torch.cat([Ec_true_anomaly_all,Ec_true_anomaly.detach().cpu()])
                Ec_in_all = torch.cat([Ec_in_all,Ec_in.detach().cpu()])

                pbar.update(1)
                if i_iter_val == 5 :
                    break
            
        save_dir = join(self.debug_save_dir,'train')
        os.makedirs(save_dir)


        plt.hist(Ec_in_all, bins=1000, color='green',label='ID')
        plt.legend()
        plt.savefig(join(save_dir,'ID.png'))
        plt.cla()

        plt.hist(Ec_shapenet_all, bins=1000, color='blue',label='shapenet point')
        plt.legend()
        plt.savefig(join(save_dir,'shapenet.png'))
        plt.cla()

        plt.hist(Ec_resised_all, bins=1000, color='red',label='resize point')
        plt.legend()
        
        plt.savefig(join(save_dir,'resized.png'))
        plt.cla()

        plt.hist(Ec_true_anomaly_all, bins=1000, color='lightblue',label='true anomaly (5)')
        plt.legend()
        plt.savefig(join(save_dir,'true_anomaly.png'))
        plt.cla()


        plt.hist(Ec_in_all, bins=1000, color='green',label='ID')
        plt.hist(Ec_shapenet_all, bins=1000, color='blue',label='shapenet point')
        plt.hist(Ec_resised_all, bins=1000, color='red',label='resize point')
        plt.hist(Ec_true_anomaly_all, bins=1000, color='lightblue',label='true anomaly (5)')
        plt.legend()
        plt.savefig(join(save_dir,'all.png'))
        plt.cla()
        
        torch.cuda.empty_cache()




def setup_seed(seed):
     torch.manual_seed(seed)
     #*===================
     torch.cuda.manual_seed_all(seed)
     #*===================
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_ood_final.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--experiment_name', default=None, type=str,required=False)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--last_epoch', default=-1, type=int)
    #* eval all epochs
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_max_epoch', default=10, type=int)
    #* eval one epoch
    parser.add_argument('--eval_one_epoch', action='store_true')

    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--val-print-freq', default=1, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--clip_norm', default=0.0, type=float)

    parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--decay-freq', default=10, type=int)
    parser.add_argument('--lr_decay_epochs',type=int, default=[10, 20,30],nargs='+',) #*  decay every 10 epoch 
    parser.add_argument('--decay-weight', default=0.1, type=float)
    parser.add_argument('--train_bs', default=1, type=int)
    parser.add_argument('--val_bs', default=4, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    
    

    parser.add_argument('--REAL_LOSS', action='store_true',)
    parser.add_argument('--CALIBRATION_LOSS', action='store_true',)
    parser.add_argument('--ENERGY_LOSS', action='store_true',)
    parser.add_argument('--GAMBLER_LOSS', action='store_true',)
    parser.add_argument('--SHAPENET_ANOMALY', action='store_true',)
    parser.add_argument('--no_resized_point', action='store_true',)
    parser.add_argument('--m_out_max', default=-6, type=int)
    parser.add_argument('--resize_m_out', default=-6, type=int)

    parser.add_argument('--energy_type', default='origin', type=str,choices=['origin','dynamic','crude_dynamic'])
    parser.add_argument('--debug', action='store_true',)
    parser.add_argument('--save_dir_suffix', default="", type=str)


    args = parser.parse_args()
    print(args.local_rank)

    if args.debug :
        trainer =DebugTrainer(args)
        trainer.debug()
        exit(0)


    #* seed of previous exp : 10
    #* random seed for not resize experiment
    # seed = np.random.randint(1,100)
    seed = 10
    setup_seed(seed)
    print(f'seed == {seed}')

    trainer =AnomalyTrainer(args)    
    if args.eval:
        trainer.eval(args.eval_max_epoch)
        # trainer.get_mIoU()
        # trainer.statistic_data()
        # trainer.reset_save_root('/'.join(args.resume.split('/')[:-1]))
        # trainer.get_pixel_wise_absteining_mIoU()
        exit(0)

    if args.eval_one_epoch:
        trainer.eval_one_epoch()
        exit(0)

        
        
    trainer.train()


    
