
###
 # @Author: daniel
 # @Date: 2023-05-06 23:49:56
 # @LastEditTime: 2023-08-19 14:06:24
 # @LastEditors: daniel
 # @Description: 2023-05-07-15:22 for debug 
 # @FilePath: /openset_anomaly_detection/scripts/train_nusc.sh
 # have a nice day
### 

#* offline command :  volume-382-202/openset_anomaly_detection/train_tlcz7.sh
#* for the dataset


source  /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate yang_real

#* distributed train 

gpuids='0,1'
gpu_num=2;

export NCCL_SOCKET_TIMEOUT=12000000

#!====================================================================================!#
#!================================ nuscene ===========================================!#
#!====================================================================================!#


CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
--nproc_per_node $gpu_num --master_port $RANDOM anomaly_nuscenes/nuscenes_trainer.py \
--ENERGY_LOSS --GAMBLER_LOSS --SHAPENET_ANOMALY --energy_type 'crude_dynamic' --m_out_max -5 --resize_m_out -6 \
--lr 1e-3 --lr_decay_epochs 20 40 60 --decay-weight 0.5 --train_bs 2 --val_bs 2 --max_epoch 70 \
2>&1 | tee -a logs/train_nuscenes_off40.log



#* for resume 
# --resume $resume_path --last_epoch 44 \
