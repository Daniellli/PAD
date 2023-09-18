# -*- coding:utf-8 -*-

# @file: train_cylinder_asym.py
import copy
import sys
import os
sys.path.insert(0,os.getcwd())

from os.path import split, join, exists, isdir,isfile,dirname
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import wandb 
from IPython import embed
from utils.pc_utils import *
warnings.filterwarnings("ignore")


import json

import random


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


#!  why DistributedSampler is not used ?
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, all_reduce
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist



import spconv.pytorch as spconv
from utils.load_save_util import load_checkpoint
from utils.utils import *
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.image_utils import * 



from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data




from pad_losses.gambler_loss import *
from pad_losses.energe_loss import * 

from loguru import logger 

from os.path import exists

class AnomalyTrainer:


    def __init__(self,args):

        self.args = args
        

        
        self.load_cfg()
        self.init_path()
        
        if self.args.wandb :
            self.init_wandb()

        
        self.best_aupr = 0
        self.best_epoch = -1

        if args.eval:
            self.log(f"only eval, do not need to load model again")
            return 
        
        gpu_id = self.args.local_rank if self.args.local_rank != -1 else 0
        torch.cuda.set_device(gpu_id)
        self.init_distributed()
        self.init_dataloader()

        # if self.args.local_rank  != -1:
        
        self.log(f" distributed train init done ")


        self.init_model()

        self.unknown_clss = [1,5,8,9]
        self.init_criterion()
        

        #* init the optimizer 
        # if DISTRIBUTED:
        self.log('World size:'+ str(torch.distributed.get_world_size()))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.my_model.parameters()), 
                               lr=self.configs['train_params']["learning_rate"] )
        #  * torch.distributed.get_world_size()
        # else:
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.my_model.parameters()),
        #                             lr= self.configs['train_params']["learning_rate"])

        
        #* init the scheduler 
        # world_size = torch.distributed.get_world_size()
        # warmup_lr_factor_start = 1 / world_size
        # warmup_lr_factor_end = 1
        # warmup_lr_increment_amount = (warmup_lr_factor_end - warmup_lr_factor_start) / 5

        #? why so complex
        # scheduler_0 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 0 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_1 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 1 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_2 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 2 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_3 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 3 * warmup_lr_increment_amount, total_iters=1)
        # scheduler_4 = ConstantLR(optimizer, factor=warmup_lr_factor_start + 4 * warmup_lr_increment_amount, total_iters=1)

        # drop_lr_every_n_epochs = 10
        # mile_stones = list(0 + np.arange(drop_lr_every_n_epochs, self.configs['train_params']['max_num_epochs'], drop_lr_every_n_epochs))
        # scheduler_5 = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
        #                                                         schedulers=[scheduler_0, scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5], milestones=[1, 2, 3, 4, 5])
                

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

        #* save config 
        if args.local_rank ==0:
            config_save_path = join(self.save_root,'config.txt')
            if  not exists(config_save_path):
                content = '\n'.join([f"{k}: {v}" for k,v in  self.args.__dict__.items()])
                content += '\n config file items : \n\n'
                content += '\n'.join([f"{k}: {v}"  for k,v in self.configs.items()])
                with open(config_save_path,'w') as f :
                    f.write(content)
                
        
    def init_wandb(self):

        if self.args.local_rank ==0 or self.args.local_rank == -1 :


            """"
                resume wandb :
            
            
            """
            wandb_info_path  = join(self.save_root,'wandb_resume_info.json')

            if self.args.resume is not None and exists(wandb_info_path) :
                with open(wandb_info_path, 'r') as f :
                    last_run_info  = json.load(f)
                run  = wandb.init(project='3d_anomaly_detection',id=last_run_info['id'], resume="must")
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

            

            experiment_output_dir = join(os.getcwd(),'nuscenes_runs',\
                 time.strftime("%Y-%m-%d-%H:%M",time.gmtime(time.time())) + \
                    f"{self.args.energy_type}#{self.args.resize_m_out}#{self.args.m_out_max}#{self.args.lr}#{'-'.join([str(x) for x in self.args.lr_decay_epochs])}{status}" \
                    + self.args.save_dir_suffix )

            
            if self.args.local_rank == 0 :
                make_dir(experiment_output_dir)

            self.save_root = experiment_output_dir
        self.model_save_path = join(experiment_output_dir, 'model_best.pt')
        self.model_latest_path = join(experiment_output_dir,'model_latest.pt')

        self.log(f"save path : {experiment_output_dir}")
        self.writer = SummaryWriter(experiment_output_dir)


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
                                                                SHAPENET_ANOMALY=self.args.SHAPENET_ANOMALY)
        

    

    def init_criterion(self):

        self.loss_func_train, lovasz_softmax = loss_builder.build_ood(wce=True, lovasz=True,
                                                                num_class=self.configs['model_params']['num_class'],
                                                                  ignore_label=self.configs['dataset_params']['ignore_label'], 
                                                                  weight=self.configs['train_params']['lamda_2'])
        
        self.loss_func_val, self.lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                            num_class=self.configs['model_params']['num_class'], 
                                                            ignore_label=self.configs['dataset_params']['ignore_label'])

        
        """"
        #* class number = 16 + 1 , namely , the 0-th class
        unknown_clss = [1,5,8,9]
        """                 
        self.gambler_loss = Gambler(reward=[4.5],
                                    device=torch.device('cuda:' + str(self.args.local_rank)),
                                    valid_class_num=16,novel_class_num=len(self.unknown_clss),unknown_cls_idx=1,
                                    novel_class_list = self.unknown_clss)


    def inference_epoch(self,epoch,model_name=None):
        
        
        if model_name is None :
            model_name = 'model_epoch_%s'%(epoch)
        uncertainty_folder = join(self.save_root,model_name,'uncertainty')
        point_predict_folder = join(self.save_root,model_name,'point_predict')

        if dist.get_rank() == 0:
            make_dir(uncertainty_folder)
            make_dir(point_predict_folder)


        self.my_model.eval()

        self.val_dataset_loader.sampler.set_epoch(epoch)
        pbar_val = tqdm(enumerate(self.val_dataset_loader),total=len(self.val_dataset_loader))
        with torch.no_grad():
            for i_iter_val, (voxel_position, val_vox_label, val_grid, point_label, val_pt_fea, idx) in pbar_val:
                val_batch_size = len(val_pt_fea)

                """"
                visilize             

                a = np.concatenate([val_pt_fea[0][:,-3][...,None],val_pt_fea[0][:,-2][...,None],val_pt_fea[0][:,-4][...,None]],1)
                s_labels = point_label[0].squeeze()
                unknown_clss = [1,5,8,9]
                anomaly_labels = np.array([(x in unknown_clss) for x in s_labels]).astype(np.int32)

                write_ply_color(a,s_labels,'logs/debug/nuscenes/pc_label_anomaly.ply')
                
                write_ply_color_anomaly(a,anomaly_labels,'logs/debug/nuscenes/pc_label_anomaly_high_light.ply')
                
                """

                


                #* prepare data 
                
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).cuda() for i in val_grid]
                # val_label_tensor = val_vox_label.type(torch.LongTensor).cuda()




                #* old validation forward
                # predict_labels = self.my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)

                #* forward
                coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final( 
                                        val_pt_fea_ten, val_grid_ten, val_batch_size, 
                                        args.dummynumber, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False)
                
                #* save 
                predict_labels = torch.argmax(output_normal_dummy[:,:-1,...], dim=1).cpu().detach().numpy()
                uncertainty_scores_softmax = torch.nn.Softmax(dim=1)(output_normal_dummy)[:,-1,...].cpu().detach().numpy() 
                

                #todo                 
                # gather_list = [torch.zeros_like(output_normal_dummy) for _ in range(dist.get_world_size())]
                # dist.all_gather(gather_list, output_normal_dummy)


                for count in range(val_batch_size):
                    # sample_name = self.val_dataset_loader.dataset.get_name(idx[count])
                    sample_name = "%06d" % idx[count]

                    uncertainty_save_path = join(uncertainty_folder ,sample_name + '.label')
                    point_predict_save_path = join(point_predict_folder, sample_name + '.label')

                    if exists(uncertainty_save_path) and exists(point_predict_save_path):
                        continue

                    point_predict = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]].astype(np.int32)
                    point_uncertainty_softmax = uncertainty_scores_softmax[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]
                    
                    point_uncertainty_softmax.tofile(uncertainty_save_path)
                    point_predict.tofile(point_predict_save_path)

                    #!+===================================

                    """"
                    visualization:
                        write_ply_color(a,point_predict,num_classes=np.max(point_label[0].squeeze())+1,filename='logs/debug/nuscenes/pc_label_anomaly_prediction.ply')
                        uncertainty = (point_uncertainty_softmax*255).astype(np.int64)
                        write_ply_color(a,uncertainty,'logs/debug/nuscenes/pc_label_anomaly_uncertainty.ply')
                    """

                    #!+====================================

   
                pbar_val.update(1)

             

        torch.cuda.empty_cache()


    def train_epoch(self,epoch):
        self.my_model.train()

        e_loss_list, g_loss_id_list, g_loss_ood_list, normal_loss_list, dummy_loss_list, loss_list = [], [], [], [], [], []
        pbar = tqdm(self.train_dataset_loader,total=len(self.train_dataset_loader), desc='Train')
        # time.sleep(10)

        
        for i_iter, (__voxel_position, train_vox_label, train_grid, __point_label, train_pt_fea) in enumerate(pbar):            
            # Train
            train_batch_size,W,H,C=train_vox_label.shape
            """"
            visilize   
            

            #* with batch size
            a = np.concatenate([train_pt_fea[0][:,-3][...,None],train_pt_fea[0][:,-2][...,None],train_pt_fea[0][:,-4][...,None]],1)
            write_ply_color(a,__point_label[0].squeeze(),'logs/debug/pc_label_anomaly.ply')
            write_ply_color_anomaly(a,(__point_label[0].squeeze()>=20).astype(np.int32),'logs/debug/pc_label_anomaly_high_light.ply')
            tmp = __point_label[0].squeeze().copy()
            tmp[tmp<20] = 0 
            remap_dict = {v:idx for idx,v in enumerate(np.unique(tmp))}
            remap_tmp =np.array([remap_dict[x] for x in tmp])
            
            
            write_ply_color(a,tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy.ply')
            write_ply_color(a,remap_tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy_remapped.ply')


            #* without batch size

            __voxel_position, train_vox_label, train_grid, __point_label, train_pt_fea = self.train_dataset_loader.dataset[1]
            a = np.concatenate([train_pt_fea[:,-3][...,None],train_pt_fea[:,-2][...,None],train_pt_fea[:,-4][...,None]],1)
            write_ply_color(a,__point_label.squeeze(),'logs/debug/pc_label_anomaly.ply')
            write_ply_color_anomaly(a,(__point_label.squeeze()>=20).astype(np.int32),'logs/debug/pc_label_anomaly_high_light.ply')
            tmp = __point_label.squeeze().copy()
            tmp[tmp<20] = 0 
            remap_dict = {v:idx for idx,v in enumerate(np.unique(tmp))}
            remap_tmp =np.array([remap_dict[x] for x in tmp])
            
            
            write_ply_color(a,tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy.ply')
            write_ply_color(a,remap_tmp.astype(int),'logs/debug/pc_label_anomaly_high_light_dynamic_energy_remapped.ply')
            
            
            """
            

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).cuda() for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).cuda() for i in train_grid]
                        
            unknown_clss = [1,5,8,9]
            noval_clas = 17 #* the 17-th is dummy class 

            #!====================================================================================
            # point_label_tensor = train_vox_label.type(torch.LongTensor).cuda()
            #! dynamic energy code 
            long_rang_point_label_tensor = train_vox_label.type(torch.LongTensor).cuda() #* [2, 480, 360, 32]
            point_label_tensor = copy.deepcopy(long_rang_point_label_tensor)
            point_label_tensor[point_label_tensor >= noval_clas] =  noval_clas
            #!====================================================================================
            

            #!+========================

            
            for unknown_cls in unknown_clss:
                point_label_tensor[point_label_tensor == unknown_cls] = 0

            energy_point_label_tensor = copy.deepcopy(point_label_tensor)

            energy_point_label_tensor[energy_point_label_tensor == noval_clas] = unknown_clss[0]

            gambler_point_label_tensor = copy.deepcopy(energy_point_label_tensor)

            #!+========================


            #* forward + backward + optimize
            coor_ori, output_normal_dummy = self.my_model.module.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                            train_batch_size,self.args.dummynumber)
            

            if self.args.ENERGY_LOSS or self.args.GAMBLER_LOSS:
                #!+=======================================================
                # todo what about the class of unknown_clss[1:]
                logits_for_loss_computing = torch.hstack(
                        [
                            output_normal_dummy[:, :unknown_clss[0]], 
                            output_normal_dummy[:, -1:],
                            output_normal_dummy[:, unknown_clss[0]+1: unknown_clss[1]],
                            output_normal_dummy[:, unknown_clss[1]+1: unknown_clss[2]],
                            output_normal_dummy[:, unknown_clss[2]+1: unknown_clss[3]],
                            output_normal_dummy[:, unknown_clss[3]+1: -1],
                        ]
                    )
                #!+=======================================================


                
            voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            if self.args.REAL_LOSS:
                """
                    #* including the systhesis loss  and the origin calibration loss.  
                    # the true anomaly object and noise align to 0 while the normal and sythesis objects align to their corrsponding class 
                """
                loss_normal = self.loss_func_train(output_normal_dummy, point_label_tensor) 
            else:
                loss_normal = torch.tensor(0).cuda()

    

            if self.args.CALIBRATION_LOSS:
                output_normal_dummy = output_normal_dummy.permute(0, 2, 3, 4, 1)
                output_normal_dummy = output_normal_dummy[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
                index_tmp = torch.arange(0, coor_ori.shape[0]).unsqueeze(0).cuda()
                voxel_label_origin[voxel_label_origin == noval_clas] = 0 #!=============== thus, the [1,5,8,9,17] are all assigned as 0 
                index_tmp = torch.cat([index_tmp, voxel_label_origin], dim=0)
                """
                 并没有ignore掉 noval_clas 对应的classifier,  而是ignore掉noval_clas的 第0个classifier 输出,
                 但是下面又将 voxel_label_origin ==0 的label_dummy 设置成0, 
                 所以最终计算calibration loss 的时候还是把 true anomaly, noise and systhesis point 排除在外, 只对normal object 计算calibration loss
                """
                output_normal_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9 #* ignore all of the prediction of common object 
                label_dummy = torch.ones(output_normal_dummy.shape[0]).type(torch.LongTensor).cuda() * noval_clas #!===============
                label_dummy[voxel_label_origin.squeeze() == 0] = 0 #*  except the  [1,5,8,9,17] 
                loss_dummy = self.loss_func_train(output_normal_dummy, label_dummy) #*  all of point prediction have as large response value as possible to the unknown class 
            else:
                loss_dummy = torch.tensor(0).cuda()

            
            """"
            # Energy loss
            #* Labels for energy loss: Do not use class 5. 
            #* Use scaled id objets (class 20) as ood objects during training            
            只将[1,5,8,9] 中的[1] 替换成noval class 其他不管, 还是除了替换noval class 还将剩下的切片切走  对最后的loss 没用影响, namely, 都一样....
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
                #!+=====================
                num_ood_samples = torch.sum((gambler_point_label_tensor == unknown_clss[0]),
                                            dim=tuple(np.arange(len(gambler_point_label_tensor.shape))[1:]))
                #!+=====================

                is_ood = num_ood_samples > 0
                in_logits, in_target = logits_for_loss_computing[~is_ood], gambler_point_label_tensor[~is_ood]
                out_logits, out_target = logits_for_loss_computing[is_ood], gambler_point_label_tensor[is_ood]
                # 1. in distribution
                if in_logits.shape[0] > 0:
                    g_loss_id = self.gambler_loss(pred=in_logits, targets=in_target, wrong_sample=False)
                else:
                    g_loss_id = torch.tensor(0).cuda()
                # 2. out-of-distribution
                if torch.any(is_ood):
                    g_loss_ood = self.gambler_loss(pred=out_logits, targets=out_target, wrong_sample=True)
                else:
                    g_loss_ood = torch.tensor(0).cuda()
            else:
                g_loss_id = torch.tensor(0).cuda()
                g_loss_ood = torch.tensor(0).cuda()

             
            if torch.isnan(e_loss):
                self.log(f"e_loss: {e_loss}, only nan is printed")
                loss = (loss_normal + self.configs['train_params']['lamda_1'] * loss_dummy) +\
                    (g_loss_id + g_loss_ood)
            else:
                loss = (loss_normal + self.configs['train_params']['lamda_1'] * loss_dummy) +\
                    (0.1 * e_loss + g_loss_id + g_loss_ood)
                

            self.optimizer.zero_grad()
            loss.backward()

            
            self.optimizer.step()
            
            loss_list.append(loss.item())
            normal_loss_list.append(loss_normal.item())
            dummy_loss_list.append(loss_dummy.item())
            e_loss_list.append(e_loss.item())
            g_loss_id_list.append(g_loss_id.item())
            g_loss_ood_list.append(g_loss_ood.item())

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
                "epoch":epoch,
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
            torch.cuda.empty_cache()
            self.train_dataset_loader.sampler.set_epoch(epoch)
            self.val_dataset_loader.sampler.set_epoch(epoch)

            with torch.autograd.detect_anomaly():
                self.train_epoch(epoch)

            #* save model 
            model_name = 'model_epoch_%s'%(epoch)

            self.save_ckpt(model_name = model_name)
            
            self.inference_epoch(epoch)

            # torch.distributed.barrier()
            # torch.distributed.barrier()

            epoch += 1
            # self.lr_scheduler.step()
            

    def eval_epoch(self,epoch,model_name=None):

        if model_name is None :
            model_name = 'model_epoch_%s'%(epoch)

        res_json_path = join(self.save_root,model_name,'anomaly_eval_results.json')

        if not exists(res_json_path):                
            command = f"python nuscenes_api/evaluate_semantics3.py \
                --dataset datasets/nuScenes --predictions {join(self.save_root,model_name)} \
                    --data_cfg_path nuscenes_api/config/nuscenes.yaml "

            os.system(command)
        
        with open(res_json_path,'r') as f :
            data = json.load(f)

        self.log_wandb(data)

        if self.best_aupr < data['Metrics/AUPR']:
            self.best_aupr = data['Metrics/AUPR']
            self.log_wandb({'Misc/best_aupr':self.best_aupr,'Misc/best_epoch':epoch})
            self.best_epoch = epoch
            self.log('best_aupr : %f \t best_epoch : %d '%(self.best_aupr,epoch))
        else:
            #todo remove the model 
            model_path = join(self.save_root,model_name,model_name+'.pt')
            point_predict_path = join(self.save_root,model_name,'point_predict')
            uncertainty_path = join(self.save_root,model_name,'uncertainty')
            try: 
                if exists(model_path):
                    os.remove(model_path)

                if exists(point_predict_path):
                    shutil.rmtree(point_predict_path)
                
                
                if exists(uncertainty_path):
                    shutil.rmtree(uncertainty_path)
            except Exception as e :
                print(f" model_path : {model_path}point_predict_path: {point_predict_path}" + f" uncertainty_path: {uncertainty_path} " )
                print(e)
                

        
        
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

        self.root = '/'.join(path_parts[:-1])
        model_name = path_parts[-1].split('.')[-2]


        #* check where the inference is done 
        uncertainty_folder = join(self.save_root,model_name,'uncertainty')
        point_predict_folder = join(self.save_root,model_name,'point_predict')

        if exists(uncertainty_folder) and exists(point_predict_folder) \
            and self.val_dataset_loader.dataset.__len__() == len(os.listdir(point_predict_folder)) and \
                self.val_dataset_loader.dataset.__len__() == len(os.listdir(uncertainty_folder)):
                
            #* only eval
            self.eval_epoch(self.args.last_epoch,model_name = model_name )
        else:
            self.inference_epoch(self.args.last_epoch,model_name=model_name)
            self.eval_epoch(self.args.last_epoch,model_name = model_name )
        
        self.log('best_aupr : %s \t best_epoch : %d '%(self.best_aupr,self.best_epoch))
        



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
    parser.add_argument('-y', '--config_path', default='config/nuScenes_ood_final.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    parser.add_argument('--experiment_name', default='debug_original', type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--last_epoch', default=-1, type=int)

    #* eval all epochs
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_max_epoch', default=10, type=int)
    #* eval one epoch
    parser.add_argument('--eval_one_epoch', action='store_true')

    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--val-print-freq', default=1, type=int)
    parser.add_argument('--wandb', action='store_true',)

    parser.add_argument('--max_epoch', default=50, type=int)


    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay_epochs',type=int, default=[40, 100],nargs='+',)
    parser.add_argument('--decay-weight', default=0.1, type=float)
    parser.add_argument('--train_bs', default=1, type=int)
    parser.add_argument('--val_bs', default=4, type=int)

    
    parser.add_argument('--REAL_LOSS', action='store_true',)
    parser.add_argument('--CALIBRATION_LOSS', action='store_true',)
    parser.add_argument('--ENERGY_LOSS', action='store_true',)
    parser.add_argument('--GAMBLER_LOSS', action='store_true',)
    parser.add_argument('--SHAPENET_ANOMALY', action='store_true',)


    parser.add_argument('--m_out_max', default=-6, type=int)
    parser.add_argument('--resize_m_out', default=-6, type=int)
    parser.add_argument('--energy_type', default='origin', type=str,choices=['origin','dynamic','crude_dynamic'])
    parser.add_argument('--save_dir_suffix', default='', type=str)



    args = parser.parse_args()
    print(args.local_rank)

    seed = np.random.randint(0,100)
    # seed = 70
    setup_seed(seed)
    print(f'seed == {seed}')

    trainer =AnomalyTrainer(args)
    
    if args.eval:
        trainer.eval(args.eval_max_epoch)
        # trainer.eval_one_epoch()

        exit(0)

    if args.eval_one_epoch:
        trainer.eval_one_epoch()
        exit(0)

    trainer.train()


    
