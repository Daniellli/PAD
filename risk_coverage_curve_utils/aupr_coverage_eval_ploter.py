



import os  

from os.path import join, split, exists,dirname
import numpy as np 

from IPython import embed 

import matplotlib.pyplot as plt
import math
from risk_coverage_ploter import RiskCoveragePloter,RiskCoverageAUPRAUROCPloter
import sys
sys.path.append(os.getcwd())

from semantic_kitti_api.entity.semantic_kitti_gt_loader import * 
from semantic_kitti_api.entity.prediction_loader import * 
import yaml
from tqdm import  tqdm
from utils.metric_util import  * 
import torch
from tqdm import  tqdm

import argparse





class RiskCoverageEvalPloter:


    def __init__(self,
            prediciton_save_path,
            dataset_path = 'datasets/semantic_kitti/dataset',
            dataset_cfg_path =  "semantic_kitti_api/config/semantic-kitti.yaml",
            sequence = ['08']):

        self.prediction_save_root = prediciton_save_path
        self.root = join(prediciton_save_path,'sequences/08')
        self.load_cfg(dataset_cfg_path)
    
    
        self.prediction_loader = MultiPredictionLoader(prediciton_save_path,sequence)
        self.gt_loader = MultiSementicKittiGtLoader(dataset_path,sequence)
        self.coverages_list = [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.]



    def load_cfg(self,data_cfg_path):
        # print(f"data_cfg_path : {data_cfg_path}")
        DATA = yaml.safe_load(open(data_cfg_path, 'r'))
        self.data = DATA
        class_remap = DATA["learning_map"]
        maxkey = max(class_remap.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(class_remap.keys())] = list(class_remap.values())

        self.remap_lut = remap_lut

        

    def save(self,results,save_name = 'reverse_aupr#auroc_coverage.csv'):

        with open(os.path.join(self.prediction_save_root, save_name), 'w') as f :

            for idx, result in enumerate(results):
                """
                #* error is equal to (1 - acc )
                #* the 3 columes is :  
                    target coverage 
                    actual coverage 
                    risk at actual coverage / mIoU,
                    AUPR for current coverage,
                    AUROC for current coverage,
                    the threshold for current coverage
                """
                f.write('test{:.0f},{:.2f},{:.3f},{:.3f},{:.3f},{}\n'.format(self.coverages_list[idx],results[idx][0]*100.,results[idx][1] * 100,results[idx][2]*100, results[idx][3] * 100,results[idx][4] ))


    def __call__(self,gpu_id = 3):


        progress = tqdm(self.gt_loader)
        length =self.gt_loader.__len__()

        pred_uncertainty = []
        gts = []
        pred_semantics = []
        for idx,label in enumerate(progress):

            pred,scores = self.prediction_loader[idx]
            valid_index = label != 0 #* 0 is the unlabeled object 

            label = self.remap_lut[label]       # remap to xentropy format

            label = label[valid_index]
            pred = pred[valid_index]
            scores = scores[valid_index]

            gts.append(label.reshape(-1))
            pred_uncertainty.append(scores.reshape(-1))
            pred_semantics.append(pred.reshape(-1))
            progress.update(idx//length)
        
        
        gts = torch.cat([torch.from_numpy(x).cuda(gpu_id) for x in gts])
        pred_uncertainty = torch.cat([torch.from_numpy(x).cuda(gpu_id) for x in pred_uncertainty])
        pred_semantics = torch.cat([torch.from_numpy(x).cuda(gpu_id) for x in pred_semantics])

        #* filter 0
        mask = gts!= 0
        gts = gts[mask]
        pred_uncertainty = pred_uncertainty[mask]
        pred_semantics = pred_semantics[mask]

        
        results = []
        
        bisection_method_mIoU(self.coverages_list, pred_uncertainty, pred_semantics, gts, results,self.data)
        self.save(results)
        return results





if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--prediction_save_path', default='', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    

    args = parser.parse_args()
    

    

    # eval_ploter = RiskCoverageEvalPloter("runs/no_real_ep28_aupr@43/no_real_ep28_aupr@43")
    # results = eval_ploter(2)


    eval_ploter = RiskCoverageEvalPloter(args.prediction_save_path)
    eval_ploter(args.gpu_id)
