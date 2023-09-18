



import os
from os.path import join, split, exists
import numpy as np 
import matplotlib.pyplot as plt


class RiskCoverageCurvePloter:


    def __init__(self,path):

        self.path = path 
        # self.data = 

        self.coverages_list = [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.]

        self.get_risk_coverage()


        self.id_score_path = join('/'.join(path.split('/')[:-1]),"scores_softmax_3dummy_base_in.score")
        self.ood_score_path  = join('/'.join(path.split('/')[:-1]),"scores_softmax_3dummy_base_ood.score")
        self.ood_score = np.fromfile(self.ood_score_path, dtype=np.float32)
        self.id_score = np.fromfile(self.id_score_path, dtype=np.float32)


    def draw_ood_score(self,title=None):        

        

        bins = np.arange(0, 1.1, 0.05)

        hist, _ = np.histogram(self.ood_score, bins=bins)
        
        fig, ax = plt.subplots()

        ax.bar(bins[:-1], hist, width=0.1)


        ax.set_xlabel('Score of unknown class for all real unknown class points')
        ax.set_ylabel('Number')

        ax.set_yscale('log')

        ax.set_ylim([100, 1e+9])


        if title is not None : 
            ax.set_title(title)
        else:
            title = 'ood_no_title'

    
        plt.savefig('%s.png'%(title), dpi=300,bbox_inches='tight')
        







    def draw_id_score(self,title = None):


        

        bins = np.arange(0, 1.1, 0.05)

        hist, _ = np.histogram(self.id_score, bins=bins)
        
        fig, ax = plt.subplots()

        ax.bar(bins[:-1], hist, width=0.1)


        ax.set_xlabel('Score of unknown class for all real unknown class points')
        ax.set_ylabel('Number')


        ax.set_yscale('log')

        ax.set_ylim([100, 1e+9])

        

        if title is not None : 
            ax.set_title(title)
        else:
            title = 'id_no_title'

        
        
        plt.savefig('%s.png'%(title), dpi=300,bbox_inches='tight')
        


    def get_risk_coverage(self):

        data = np.loadtxt(self.path,dtype=np.str0,delimiter=',')
        # low_bound = 55
        low_bound = 65
        data = data[:,1:][:len(np.where(data[:,1].astype(np.float32) > low_bound)[0]),:]
        # data = data[:,1:]
        # print(data)
        
        self.actual_coverages =  data[:,0][::-1].astype(np.float64)
        self.risks =  data[:,1][::-1].astype(np.float64)
        # todo : from mIoU to Risk: 1 - mIoU
        self.risks =  (1- self.risks / 100.0) * 100

        # self.risks = np.exp(self.risks)
        # self.risks = np.log(self.risks)

        self.thresholds =  data[:,2][::-1].astype(np.float64)
        # print(self.thresholds)

        
    def __len__(self):
        return len(self.thresholds)


        

        
    def __call__(self,legend= 'risk-coverage curve',color=None):

        if color is not None :
            plt.plot(self.actual_coverages, self.risks,label=legend,color=color)
            plt.scatter(self.actual_coverages, self.risks,color=color)
        else:
            plt.plot(self.actual_coverages, self.risks,label=legend)
            plt.scatter(self.actual_coverages, self.risks)
        plt.legend()

        # plt.show()

    def threashold_coverage_curve(self,legend= 'risk-coverage curve',color=None):

        if color is not None :
            plt.plot(self.actual_coverages, self.thresholds,label=legend,color=color)
            plt.scatter(self.actual_coverages, self.thresholds,color=color)
        else:
            plt.plot(self.actual_coverages, self.thresholds,label=legend)
            plt.scatter(self.actual_coverages, self.thresholds)
        plt.legend()
        
        # plt.show()
        
    
        
    def twin_plot_trails(self,legend= 'risk-coverage curve'):


        fig, ax1 = plt.subplots()

        ax1.plot(self.actual_coverages, self.risks,label=legend)
        ax1.scatter(self.actual_coverages, self.risks)
        ax1.set_ylabel('Risk')

        ax2 = ax1.twinx()
        ax2.plot(self.actual_coverages, self.thresholds)
        ax2.scatter(self.actual_coverages, self.thresholds)
        ax2.set_ylabel('Threashold')
        
        
        # ax1.legend()

        return ax1,ax2
        
    



        
if __name__ == "__main__":

    plt.xticks(list(range(70,91,5)) + list(range(92,102,2)))
    plt.grid(True)
    ce_cce_ploter = RiskCoverageCurvePloter('runs/model_archive/real_ablation/ce#cce#aupr21#ep4/coverage_VS_err3.csv')
    energy_abstein_S_dynamic_ploter = RiskCoverageCurvePloter('runs/model_archive/main_contribution_ablation/energy#abstein#S#dynamic#aupr44#ep34/energy#abstein#S#dynamic#aupr44#ep34/coverage_VS_err3.csv')
    
    ce_cce_ploter(legend='REAL')
    energy_abstein_S_dynamic_ploter(legend='Ours')

    plt.xlabel('Coverage')
    plt.ylabel('Risk')


    plt.savefig('risk_coverage_sweep.png', dpi=300,bbox_inches='tight')
    # plt.show()