
###
 # @Author: daniel
 # @Date: 2023-05-06 23:49:56
 # @LastEditTime: 2023-07-01 17:10:22
 # @LastEditors: daniel
 # @Description:  for debug 
 # @FilePath: /openset_anomaly_detection/debug.sh
 # have a nice day
### 

#* offline command :  volume-382-202/openset_anomaly_detection/train_tlcz7.sh

#* for the dataset
sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh
mount | grep s3_common_dataset

source activate yang_real

#* distributed train 

gpuids='0,1,2,3,4,5,6,7'
gpu_num=1;

export NCCL_SOCKET_TIMEOUT=12000000

#!====================================================================================!#
#!================================ nuscene ===========================================!#
#!====================================================================================!#


# CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
# --nproc_per_node $gpu_num --master_port $RANDOM anomaly_nuscenes/nuscenes_trainer.py \
# --ENERGY_LOSS --GAMBLER_LOSS \
# --lr 1e-2 --lr_decay_epochs 20 40 60 80 100 120 140 160 180 200 --decay-weight 0.5 --train_bs 8 --val_bs 32 --max_epoch 200 \
# 2>&1 | tee -a logs/train_nuscenes2.log







#!====================================================================================!#
#!================================ semantic kitti ====================================!#
#!====================================================================================!#


#* 能否浮现origin energy 的精度
CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
--nproc_per_node $gpu_num --master_port $RANDOM anomaly_semantic_kitti/trainer.py \
--ENERGY_LOSS --GAMBLER_LOSS --energy_type 'crude_dynamic' --m_out_max -6 --resize_m_out -6 \
--lr 1e-3 --lr_decay_epochs 10 20 30 40 --decay-weight 0.1 --train_bs 2 --val_bs 32 --save_dir_suffix 'debug' \
2>&1 | tee -a logs/train_semantic_kitti_off36.log


# --resume $resume_path --last_epoch 21 \







