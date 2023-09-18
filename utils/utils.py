import torch
from PIL import Image
import math
from os.path import  join, split
import threading
import shutil
import numpy as np
import os
import torch.nn.functional as F
import argparse
# import drn
from concurrent import futures

try:
    from modules import batchnormsync
except ImportError:
    pass



import scipy.io as scio
import sys



def load_mat(path):
    return scio.loadmat(path)






'''
description: 
param {*} tensor
param {*} width
param {*} height
return {*}
'''
def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


'''
description: 
param {*} depth_output_dir
return {*}
'''
def make_dir(depth_output_dir):
    if not os.path.exists(depth_output_dir):
        os.makedirs(depth_output_dir)


'''
description: 
param {*} pred
param {*} label
param {*} n
return {*}
'''
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)




'''
description: 
param {*} hist
return {*}
'''
def per_class_iu(hist):#? 这是什么 
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))






'''
description:   保存model
param {*} state
param {*} is_best
param {*} filename
return {*}
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #* 在save 一份
    if is_best:
        shutil.copyfile(filename, join(split(filename)[-2] ,'model_best.pth.tar'))




'''
description:  计算可以学习的模型参数
param {*} model : 需要计算的模型
return {*}
'''

def calculate_param_num(model):
    total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        multvalue = np.prod(param.size())
        total_params += multvalue
        if param.requires_grad:
            Trainable_params += multvalue  # 可训练参数量
        else:
            NonTrainable_params += multvalue  # 非可训练参数量
    
    
    return total_params,Trainable_params,NonTrainable_params











# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def process_mp(function,function_parameter_list,num_threads=64,\
                prefix='processing with multiple threads:',suffix = "done"):

    num_sample = len(function_parameter_list)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        
        fs = [executor.submit(function, parameters) for parameters in function_parameter_list]

        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, num_sample, prefix=prefix, suffix=suffix, barLength=40)









