
###
 # @Author: daniel
 # @Date: 2023-01-16 12:45:04
 # @LastEditTime: 2023-03-22 20:27:06
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Open_world_3D_semantic_segmentation/semantickitti_scripts/train_naive.sh
 # have a nice day
### 


name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym_naive.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt

