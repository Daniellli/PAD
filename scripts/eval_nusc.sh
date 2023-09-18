###
 # @Author: daniel
 # @Date: 2023-03-20 22:51:21
 # @LastEditTime: 2023-08-19 14:23:01
 # @LastEditors: daniel
 # @Description:  train_kitti_offline6.sh
 # @FilePath: /openset_anomaly_detection/scripts/eval_nusc.sh
 # have a nice day
### 



source  /usr/local/miniconda3/etc/profile.d/conda.sh 

conda activate yang_real
    
#* distributed train 

gpuids='0'
gpu_num=1;





#* =========================eval all epochs=========================


# resume_path="nuscenes_runs/2023-08-17-10:26crude_dynamic#-6#-5#0.001#20-40-60#E#G#Srd_0.01_rrf_2_nodown-5-6—daniel/model_epoch_${eval_epoch}/model_epoch_${eval_epoch}.pt"
# part1=$(echo $resume_path | cut -d "/" -f 1)
# part2=$(echo $resume_path | cut -d "/" -f 2)
# echo $part1, $part2
# eval_epoch=0;
# eval_max_epoch=15;

# CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
# --nproc_per_node $gpu_num --master_port $RANDOM anomaly_nuscenes/nuscenes_trainer.py \
# --ENERGY_LOSS --GAMBLER_LOSS --SHAPENET_ANOMALY --energy_type 'crude_dynamic' --m_out_max -5 --resize_m_out -6 \
# --lr 1e-3 --lr_decay_epochs 20 40 60 --decay-weight 0.5 --train_bs 12 --val_bs 24 --max_epoch 70 \
# --resume $resume_path --last_epoch $eval_epoch --eval --eval_max_epoch ${eval_max_epoch} \
# 2>&1 | tee -a $part1/$part2/eval_nuscenes.log

# --wandb

#* =================================================================








#* =========================eval one epoch=========================

resume_path="nuscenes_runs/CE#CCE#aupr7#ep3#nusc/CE#CCE#aupr7#ep3#nusc.pt"


CUDA_VISIBLE_DEVICES=$gpuids python -m torch.distributed.launch \
--nproc_per_node $gpu_num --master_port $RANDOM anomaly_nuscenes/nuscenes_trainer.py \
--ENERGY_LOSS --GAMBLER_LOSS --SHAPENET_ANOMALY --energy_type 'crude_dynamic' --m_out_max -5 --resize_m_out -6 \
--lr 1e-3 --lr_decay_epochs 20 40 60 --decay-weight 0.5 --train_bs 6 --val_bs 12 --max_epoch 70 \
--resume $resume_path --eval_one_epoch \
2>&1 | tee -a logs/eval_nuscenes.log

#* =================================================================