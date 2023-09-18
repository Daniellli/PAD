

'''
description:  give the model save root,  return the best model information
return {*}
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from astropy import coordinates as ac
import copy
import torch
import os 
from os.path import join, split,exists, isdir,isfile

# file_root =  join(os.getcwd(),'..')
# os.chdir(file_root)

from utils import * 
from utils.utils import * 
import numpy as np 
import time 
from tqdm import tqdm


def get_anomaly_matrics(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data 



def get_best_metrics(prediction_root,model_pathes):
    best_AUPR = best_AUROC = 0
    best_FPR95=1e+2
    best_AUPR_epoch = best_AUROC_epoch =  best_FPR95_epoch = None
    error_model_name_list = []
    for model_name in model_pathes:
        # print(model_name)
        try:
            prediciton_model_path = join(prediction_root,model_name,'anomaly_eval_results.json')
            data = get_anomaly_matrics(prediciton_model_path)
        except Exception as e :
            error_model_name_list.append(model_name)
            print(e)
            continue


        if data['OOD/AUPR'] > best_AUPR:
            best_AUPR = data['OOD/AUPR']
            best_AUPR_epoch = model_name

        if data['OOD/AUROC'] > best_AUROC:
            best_AUROC = data['OOD/AUROC'] 
            best_AUROC_epoch = model_name
        
        if data['OOD/FPR95'] <  best_FPR95:
            best_FPR95 = data['OOD/FPR95'] 
            best_FPR95_epoch = model_name
    return best_AUPR,best_AUPR_epoch,best_AUROC,best_AUROC_epoch,best_FPR95,best_FPR95_epoch



def get_best_metrics2(prediction_root,model_pathes):
    best_AUPR = best_AUROC = 0
    best_FPR95=1e+2
    best_AUPR_epoch = best_AUROC_epoch =  best_FPR95_epoch = None
    error_model_name_list = []
    for model_name in model_pathes:
        # print(model_name)
        try:
            prediciton_model_path = join(prediction_root,model_name,'anomaly_eval_results.json')
            data = get_anomaly_matrics(prediciton_model_path)
        except Exception as e :
            error_model_name_list.append(model_name)
            print(e)
            continue


        if data['AUPR'] > best_AUPR:
            best_AUPR = data['AUPR']
            best_AUPR_epoch = model_name

        if data['AUROC'] > best_AUROC:
            best_AUROC = data['AUROC'] 
            best_AUROC_epoch = model_name
        
        if data['FPR95'] <  best_FPR95:
            best_FPR95 = data['FPR95'] 
            best_FPR95_epoch = model_name
    return best_AUPR,best_AUPR_epoch,best_AUROC,best_AUROC_epoch,best_FPR95,best_FPR95_epoch



def get_best_metrics3(prediction_root,model_pathes):
    best_AUPR = best_AUROC = 0
    best_FPR95=1e+2
    best_AUPR_epoch = best_AUROC_epoch =  best_FPR95_epoch = None
    error_model_name_list = []
    for model_name in model_pathes:
        # print(model_name)
        try:
            prediciton_model_path = join(prediction_root,model_name,'anomaly_eval_results.json')
            data = get_anomaly_matrics(prediciton_model_path)
        except Exception as e :
            error_model_name_list.append(model_name)
            print(e)
            continue


        if data['Metrics/AUPR'] > best_AUPR:
            best_AUPR = data['Metrics/AUPR']
            best_AUPR_epoch = model_name

        if data['Metrics/AUROC'] > best_AUROC:
            best_AUROC = data['Metrics/AUROC'] 
            best_AUROC_epoch = model_name
        
        if data['Metrics/FPR95'] <  best_FPR95:
            best_FPR95 = data['Metrics/FPR95'] 
            best_FPR95_epoch = model_name
    return best_AUPR,best_AUPR_epoch,best_AUROC,best_AUROC_epoch,best_FPR95,best_FPR95_epoch







if __name__ == "__main__":
        
    tic = time.time()

    prediction_root = "/data1/liyang/Open_world_3D_semantic_segmentation/nuscenes_runs/2023-04-24-06:44"


    model_pathes = sorted([x for x in os.listdir(prediction_root) if x.startswith('model')],key=lambda x:int(x.split('_')[-1]))
    best_AUPR,best_AUPR_epoch,best_AUROC,best_AUROC_epoch,best_FPR95,best_FPR95_epoch = get_best_metrics3(prediction_root,model_pathes)
    print('total spend  time  : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))
    print(best_AUPR,best_AUPR_epoch,best_AUROC,best_AUROC_epoch,best_FPR95,best_FPR95_epoch)


    # print(get_anomaly_matrics( join(prediction_root,'model_epoch_13','anomaly_eval_results.json')))