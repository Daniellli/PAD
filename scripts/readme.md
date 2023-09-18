<!--
 * @Author: daniel
 * @Date: 2023-03-20 22:51:21
 * @LastEditTime: 2023-04-05 09:35:18
 * @LastEditors: daniel
 * @Description: 
 * @FilePath: /Open_world_3D_semantic_segmentation/scripts/readme.md
 * have a nice day
-->


# Liyang Verison 


semantickitti_scripts/train_cylinder_asym_ood_final.py   is the final code merging the REAL and PEBAL.


## pipeline:


1. 训练：train_cylinder_asym_ood_final.py
2. 用训好的模型的checkpoint预测：val_cylinder_asym_ood.py
    - config file is 'config/semantickitti_ood_final.yaml' by default
    - Q. how to set the mode for inference 
    - A use --load_path for ckpt path to infernece one ckpt
3. 用 /data2/liyang/semantic_kitti_api-master 进行remap和evaluate
5
```
ps aux | grep python | grep liyang  | grep -v root | awk '{print $2}' | xargs kill -9
```

## Problem 
1. what do i need to modify at semantickitti_scripts/magic_numbers_yang_train_pebal.py?







##  usage for semantickitti_scripts 

 git clone https://gitee.com/xu-shaocong/semantic_kitti_api.git


 

in  root directory of semantickitti_scripts run `python setup.py sdist`



then 

python setup.py install



python -c "import torch; print(torch.cuda.is_available(),torch.__version__)"



srun -t 3-0 -G 4 -w  discover-05 bash scripts/train.sh 





## Note 

1. to resume must give the resume model path and last epoch 


