'''

Date: 2023-08-18 17:11:23
LastEditTime: 2023-08-18 17:26:06

Description: 
FilePath: /openset_anomaly_detection/jupyter/aupr_auroc_ploter.py
have a nice day
'''



import numpy as np 

import matplotlib.pyplot as plt

import os  

from os.path import join, split, exists
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score



class AUPRAUROCCURVEPloter:


    def __init__(self,path):
        self.path = path 
        self.fpr = np.load(join(path,'fpr.npz'))['data']
        self.tpr = np.load(join(path,'tpr.npz'))['data']
        self.recall = np.load(join(path,'recall.npz'))['data']
        self.precision = np.load(join(path,'precision.npz'))['data']

        assert self.precision.shape[0] == self.recall.shape[0]

        assert self.fpr.shape[0] == self.tpr.shape[0]
        # assert self.precision.shape[0] == self.fpn.shape[0]
        idx = np.argsort(self.recall)

        length = self.recall.shape[0]
        
        # __from = int(length * 0.000001) 
        # self.recall = self.recall[idx][__from:]
        # self.precision = self.precision[idx][__from:]

        tmp_len = 1
        self.recall = self.recall[idx][tmp_len:]
        self.precision = self.precision[idx][tmp_len:]

        


    def __len__(self):
        return self.precision.shape[0],self.tpr.shape[0]


    def aupr(self,legend):


        aupr_score = auc(self.recall , self.precision)
        # length = self.recall.shape
        # int(length[0] * 0.5)

        
        # plt.plot(self.recall[:500000] , self.precision[:500000])
        # plt.plot(self.recall[500000:] , self.precision[500000:])
        # plt.plot(self.recall[50000:] , self.precision[50000:])
        # plt.plot(self.recall[int(length[0] * 0.1):] , self.precision[int(length[0] * 0.1):])
        # plt.plot(self.recall[int(length[0] * 0.1):][::-1] , self.precision[int(length[0] * 0.1):][::-1])


        # plt.plot(self.recall, self.precision,label = legend + ": %.2f"%(aupr_score * 100)  )
        plt.plot(self.recall, self.precision,label = legend )
        plt.legend()


    def auroc(self,legend):
    
        # plt.plot(self.fpr, self.tpr,label = legend + ": %.2f"%(auc(self.fpr, self.tpr) * 100) )
        plt.plot(self.fpr, self.tpr,label = legend )

        plt.legend()
        


if __name__ == "__main__":
    ce_ploter = AUPRAUROCCURVEPloter("runs/model_archive/real_ablation/ce#aupr24#ep11")
    ce_cce_ploter =  AUPRAUROCCURVEPloter("runs/model_archive/real_ablation/ce#cce#aupr21#ep4")

    #*======================================= aupr=======================================
    
    # ce_ploter.aupr('CE')
    # ce_cce_ploter.aupr('CE&CCE')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title('AUPR')
    # plt.savefig('%s.png'%('CE_vs_CE&CCE_AUPR'), dpi=500,bbox_inches='tight')
    # plt.savefig('%s.png'%('sweep'), dpi=500,bbox_inches='tight')

    #*====================================================================================




    #*======================================= AUROC=======================================

    ce_ploter.auroc('CE')
    ce_cce_ploter.auroc('CE&CCE')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('AUROC')
    plt.savefig('%s.png'%('sweep_auroc'), dpi=500,bbox_inches='tight')
    # plt.savefig('%s.png'%('CE_vs_CE&CCE_AUROC'), dpi=500,bbox_inches='tight')
    #*====================================================================================



