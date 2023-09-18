



from quantitative_results_formater import*







class QuantitativeCompareer:


    def __init__(self,quantitativer1,quantitativer2):

        self.quantitativer1 = quantitativer1
        self.quantitativer2 = quantitativer2
        


    '''
    description:  quantitativer1 - quantitativer2
    param {*} self
    return {*}
    '''
    def get_distances(self):

        
        distances = {}
        for idx in range(self.quantitativer1.__len__()):
            tmp1 = self.quantitativer1.getitem(idx)         
            name =    self.quantitativer1.get_name(idx)
            tmp2 = self.quantitativer2.getitem_by_name(name)

            aupr_distance = tmp1['AUPR'] - tmp2['AUPR']
            auroc_distance = tmp1['AUROC'] - tmp2['AUROC']

            distances[name] = {
                "AUPR":aupr_distance,
                "AUROC":auroc_distance,
            }

        return distances
            
            
            

            
            



if __name__ == "__main__":
            
            
    formater_a  = QuantitativeResultFormater("nuscenes_runs/CE#CCE#aupr7#ep3#nusc/sorted_performance_each_image.txt")
    formater_b  = QuantitativeResultFormater("nuscenes_runs/PAL#PEL#S#aupr30#ep66#nusc/sorted_performance_each_image.txt")

    compareer= QuantitativeCompareer(formater_b,formater_a)

    distances = compareer.get_distances()



    sorted_distances = sorted(distances.items(),key = lambda x : x[1]['AUPR'],reverse=True)

    #* copy the file from a to pickup_dir directory
    
    for line in sorted_distances[:100]:
        formater_a.pick_file_by_name(line[0])
        formater_b.pick_file_by_name(line[0])
        







