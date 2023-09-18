'''

Date: 2023-08-18 19:53:44
LastEditTime: 2023-08-18 19:54:01

Description: clear the empty directory 
FilePath: /openset_anomaly_detection/jupyter/clear_directory.py
have a nice day
'''


import os 
from os.path import join, split,exists, isdir,isfile

import numpy as np 

import glob
import shutil



def get_model_list_nusc(path):
    return glob.glob(path+"/*.pt")


def get_model_list(path):
    return glob.glob(path+"/model*")



runs_path = "runs"
for name in os.listdir(runs_path):
    if (len(get_model_list(join(runs_path,name))))==0:
    # if isfile(join(runs_path,name)):
        shutil.rmtree(join(runs_path,name))
        # os.remove(join(runs_path,name))
        # print(join(runs_path,name))

        