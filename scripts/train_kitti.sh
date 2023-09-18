
###
 # @Author: daniel
 # @Date: 2023-05-06 23:49:56
 # @LastEditTime: 2023-08-19 14:07:24
 # @LastEditors: daniel
 # @Description: 2023-05-07-15:22 for debug 
 # @FilePath: /openset_anomaly_detection/scripts/train_kitti.sh
 # have a nice day
### 

#* offline command :  volume-382-202/openset_anomaly_detection/train_tlcz7.sh
#* for the dataset


source  /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate yang_real

#* distributed train 

gpuids='6,7'
gpu_num=2;

#!====================================================================================!#
#!================================ semantic kitti ====================================!#
#!====================================================================================!#



CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
--nproc_per_node $gpu_num --master_port $RANDOM anomaly_semantic_kitti/trainer.py \
--ENERGY_LOSS --GAMBLER_LOSS --SHAPENET_ANOMALY --energy_type 'crude_dynamic' --m_out_max -7 --resize_m_out -6 \
--lr 1e-3 --lr_decay_epochs 10 20 30 40 --decay-weight 0.1 --train_bs 2 --val_bs 4 \
2>&1 | tee -a logs/train_semantic_kitti.log



# --wandb 
#* for resume
# --resume $resume_path --last_epoch 21 \