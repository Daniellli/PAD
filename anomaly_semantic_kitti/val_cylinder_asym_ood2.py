import sys
import os
import time
import argparse
import numpy as np
import warnings
from tqdm import tqdm
import torch
from os.path import join

import torch.distributed as dist
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(),'semantic_kitti_api'))




import spconv.pytorch as spconv
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from utils.utils import *
from utils.image_utils import *

warnings.filterwarnings("ignore")

from semantic_kitti_api.entity.semantic_evaluator import SementicEvaluator




class Inferencor:


    def __init__(self,args):

        if args.local_rank != -1:
            torch.backends.cudnn.benchmark = True
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)
            torch.autograd.set_detect_anomaly(True) 
            self.distributed = True
        
        
        self.args = args
        self.config  = load_config_data(self.args.config_path )
        self.init_path()
        

        train_dataset_loader, val_dataset_loader = data_builder.build(self.config['dataset_params'],
                                                                      self.config['train_data_loader'],
                                                                    self.config['val_data_loader'],
                                                                    grid_size=self.config['model_params']['output_shape'])

        self.val_dataset_loader = val_dataset_loader
        
        #* todo check whether distribued train affect the judgement of following: 
        # if len(os.listdir(self.point_predict_folder)) == self.val_dataset_loader.dataset.__len__():
        #     print('already evaluated ')
        #     exit(0)

        self.init_model()


        


        
    def init_path(self):

        split_parts = self.args.load_path.split('/')

        experiment_path ='/'.join(split_parts[:-1])

        #* exists because it has been create by the train process 
        assert exists(experiment_path)
        
        self.save_dir = experiment_path

        uncertainty_folder = join(experiment_path ,'sequences/08','uncertainty' )
        point_predict_folder =join( experiment_path , 'sequences/08', 'point_predict' )
        
        make_dir(uncertainty_folder)
        make_dir(point_predict_folder)

        self.uncertainty_folder = uncertainty_folder
        self.point_predict_folder = point_predict_folder

        

    def log(self,message):
        if self.args.local_rank == -1 or self.args.local_rank == 0 : 
            print(message)

    def init_model(self):
        # SemKITTI_label_name = get_SemKITTI_label_name(self.config['dataset_params']['label_mapping'])

        my_model = model_builder.build(self.config['model_params'])

        my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, self.args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,bias=True)
        
        # print(self.args.load_path)
        #* load model and generate the save path 
        assert exists(self.args.load_path)
        
        #* not suitable for all path format
        self.log('Loading ' + self.args.load_path)
        #!+============
        #* new load code 

        
        ckpt = torch.load(self.args.load_path,map_location = 'cpu')
        my_model.load_state_dict(ckpt['model'])


        # my_model = load_checkpoint(self.args.load_path, my_model)

        #!+============
        my_model = my_model.cuda()

        if hasattr(self,'distributed') and  self.distributed:
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            my_model = torch.nn.parallel.DistributedDataParallel(my_model,device_ids=[self.args.local_rank],
                                find_unused_parameters=True,broadcast_buffers = True)
            
    
        self.my_model = my_model
        
        
    '''
    description:  remap and eval 
    param {*} self
    return {*}
    '''    
    def __eval__(self):
        # sys.path.insert(0,'/data1/liyang/Open_world_3D_semantic_segmentation/semantic_kitti_api')
        torch.distributed.barrier() #* to be consistency to ensure the inference process in different card is finished 
        if self.args.local_rank == -1 or self.args.local_rank == 0 and self.inferenced : 
            self.evaluator  =SementicEvaluator(self.config['dataset_params']['semantic_kitti_root'],
                                            prediction_path=self.save_dir,
                                            data_config_path=self.args.data_config,
                                            split='valid')
            
            
            self.evaluator()
            
        

    def __call__(self):
        
        tic = time.time()    
        self.__inference__()
        print('inference spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

        ttic = time.time()    
        self.__eval__()
        
        print('eval spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - ttic)))
        
    
        

    def __inference__(self):

        if len(os.listdir(self.point_predict_folder))  == self.val_dataset_loader.dataset.__len__():
            
            self.inferenced = True
            return 
        pytorch_device =  torch.cuda.current_device()
        pbar = tqdm(total=len(self.val_dataset_loader))

        self.my_model.eval()
        
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

                predict_labels = torch.argmax(output_normal_dummy[:,:-1,...], dim=1).cpu().detach().numpy()
                uncertainty_scores_softmax = torch.nn.Softmax(dim=1)(output_normal_dummy)[:,-1,...].cpu().detach().numpy()
                

                
                for count in range(val_batch_size):
                    sample_name = self.val_dataset_loader.dataset.get_name(idx[count])
                    uncertainty_save_path = join(self.uncertainty_folder ,sample_name + '.label')
                    point_predict_save_path = join(self.point_predict_folder, sample_name + '.label')

                    if exists(uncertainty_save_path) and exists(point_predict_save_path):
                        continue


                    point_predict = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]].astype(np.int32)
                    point_uncertainty_softmax = uncertainty_scores_softmax[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]

                    
                    
                    point_uncertainty_softmax.tofile(uncertainty_save_path)
                    point_predict.tofile(point_predict_save_path)

                pbar.update(1)


        self.inferenced = True


def parse_args():
        
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-y', '--config_path', default='config/semantickitti_ood_final.yaml')

    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')

    parser.add_argument('--load_path', default='', type=str, help='.')

    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distrubuted training')

    parser.add_argument('--data-config', default='semantic_kitti_api/config/semantic-kitti.yaml', type=str, help='.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    if args.local_rank == -1 or args.local_rank == 0 : 
        for k, v in args.__dict__.items():
            print(k,v)


    tic = time.time()    
    inferencor = Inferencor(args)
    # inferencor.__inference__()
    inferencor()
    # inferencor.__eval__()

    print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

