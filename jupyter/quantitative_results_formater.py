


import numpy as np
import matplotlib.pyplot as plt



import os 
from os.path import join, split,exists, isdir,isfile
import sys 



from utils.utils import * 
from utils import * 
import numpy as np 

import time 
from tqdm import tqdm
import shutil
import json





class QuantitativeResultFormater:

    def __init__(self,path):

        self.path = path 

        
        self.root = '/'.join(split(path)[:-1])
        
        self.prediction_path = join(self.root, 'point_predict')
        self.uncertainty_path = join(self.root, 'uncertainty')

        
        self.pickup_dir = join(self.root,'pickup_dir')
        
        self.pickup_prediction_path = join(self.pickup_dir,'point_predict')
        self.pickup_uncertainty_path = join(self.pickup_dir,'uncertainty')

        make_dir(self.pickup_prediction_path)
        make_dir(self.pickup_uncertainty_path)

        



        data = np.loadtxt(self.path,dtype=np.str0,delimiter=' ')
        
        data = {line[0]:{line[1]:float(line[2]), line[3]:float(line[4]) } for line in data}
        self.data = data 
        self.keys = list(data.keys())
        # for k,v in data.items() :

        #     print(k,v)
    def __len__(self):
        return len(self.keys)

    
    def getitem(self,idx):
        key = self.keys[idx]
        return self.data[key]

    def get_name(self,idx):
        return self.keys[idx]

    def get_file_name(self,idx):
        
        name = self.get_name(idx)

        return join(self.prediction_path,name+".label"),join(self.uncertainty_path,name+".label")

    def get_file_name_by_name(self,name):
        
        
        return join(self.prediction_path,name+".label"),join(self.uncertainty_path,name+".label")

    def pick_file_by_name(self,name):
        pred_file_name, uncertainty_file_name = self.get_file_name_by_name(name)



        # 复制文件
        shutil.copy(pred_file_name, self.pickup_prediction_path)
        shutil.copy(uncertainty_file_name, self.pickup_uncertainty_path)


        



    def getitem_by_name(self,name):
        
        # self.keys.index(name)
        return self.data[name]

    





if __name__ == "__main__":

    formater_b  = QuantitativeResultFormater("nuscenes_runs/PAL#PEL#S#aupr30#ep66#nusc/sorted_performance_each_image.txt")
    