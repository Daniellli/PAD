# -*- coding:utf-8 -*-

# @file: metric_util.py 

import numpy as np
import torch
from tqdm import  tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import math 
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) )
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)  + 1e-15)
    


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist




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
    progress = tqdm(coverages_list)
    length = len(coverages_list)

    for idx,coverage in enumerate(progress):
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
        confuse_matrix = fast_hist_crop(filterred_prediction.cpu().numpy(),filterred_semantic_gt.cpu().numpy(),unique_label)
        iou = per_class_iu(confuse_matrix)


        #* OOD performance
        #* ood label is 5 
        # uncertainty_prediction = abortion[~mask]
        # uncertainty_label = semantic_gt[~mask].clone()

        uncertainty_prediction = abortion[~mask]
        uncertainty_label = semantic_gt[~mask].clone()

        aupr_score = 0 
        auroc_score = 0 
        if len(uncertainty_label) != 0 and len(uncertainty_label) == len(uncertainty_prediction): 
            uncertainty_label[uncertainty_label != 5] = 0
            uncertainty_label[uncertainty_label == 5] = 1

            precision, recall, threasholds = precision_recall_curve(uncertainty_label.cpu().numpy(), uncertainty_prediction.cpu().numpy())#* take long time        
            aupr_score = auc(recall, precision)
            
            fpr, tpr, _ = roc_curve(uncertainty_label.cpu().numpy(), uncertainty_prediction.cpu().numpy())
            auroc_score = auc(fpr, tpr)

        
        #* actual coverage, acc, threashold for currrent coverage,  
        results.append((passed / len(prediction), np.nanmean(iou) ,aupr_score, auroc_score, test_thres))
        progress.update(idx/length)
        
        
