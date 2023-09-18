'''

Date: 2023-08-18 19:33:49
LastEditTime: 2023-08-18 20:13:20

Description: 
FilePath: /openset_anomaly_detection/risk_coverage_curve_utils/risk_coverage_curve_aupr_auroc_ploter.py
have a nice day
'''




from .risk_coverage_curve_ploter import *


class RiskCoverageCurveAUPRAUROCPloter(RiskCoverageCurvePloter):


    def __init__(self,path):
        super(RiskCoverageCurveAUPRAUROCPloter,self).__init__(path)



    def get_risk_coverage(self,low_bound = 65):

        data = np.loadtxt(self.path,dtype=np.str0,delimiter=',')
        
        
        data = data[:,1:][:len(np.where(data[:,1].astype(np.float32) > low_bound)[0]),:]
        data = np.array([x[x != ''] for x in data])
        
        self.actual_coverages =  data[:,0][::-1].astype(np.float64)

        self.risks =  data[:,1][::-1].astype(np.float64)
        # todo : from mIoU to Risk: 1 - mIoU
        self.risks =  (1- self.risks / 100.0) * 100


        assert data.shape[1]>3, "no aupr, auroc in this evaluation result file"

        
        self.aupr = data[:,2][::-1].astype(np.float64)
        self.auroc = data[:,3][::-1].astype(np.float64)

        self.thresholds =  data[:,4][::-1].astype(np.float64)
    
        

    
    def aupr_coverage_curve(self,legend= 'risk-coverage curve',color=None):

        if color is not None :
            plt.plot(self.actual_coverages, self.aupr,label=legend,color=color)
            plt.scatter(self.actual_coverages, self.aupr,color=color)
        else:
            plt.plot(self.actual_coverages, self.aupr,label=legend)
            plt.scatter(self.actual_coverages, self.aupr)
        plt.legend()
        
    def auroc_coverage_curve(self,legend= 'risk-coverage curve',color=None):
        
        if color is not None :
            plt.plot(self.actual_coverages, self.auroc,label=legend,color=color)
            plt.scatter(self.actual_coverages, self.auroc, color=color)

        else:
            plt.plot(self.actual_coverages, self.auroc,label=legend)
            plt.scatter(self.actual_coverages, self.auroc)
        plt.legend()



        
if __name__ == "__main__":

    plt.xticks(list(range(70,91,5)) + list(range(92,102,2)))
    plt.grid(True)
    ce_cce_ploter = RiskCoverageCurveAUPRAUROCPloter('runs/model_archive/real_ablation/ce#cce#aupr21#ep4/coverage_VS_err_aupr_auroc1.csv')
    energy_abstein_S_dynamic_ploter = RiskCoverageCurveAUPRAUROCPloter('runs/model_archive/main_contribution_ablation/energy#abstein#S#dynamic#aupr44#ep34/energy#abstein#S#dynamic#aupr44#ep34/coverage_VS_err_aupr_auroc1.csv')
    
    ce_cce_ploter(legend='REAL')
    energy_abstein_S_dynamic_ploter(legend='Ours')

    plt.xlabel('Coverage')
    plt.ylabel('Risk')


    plt.savefig('risk_coverage_sweep2.png', dpi=300,bbox_inches='tight')
    # plt.show()