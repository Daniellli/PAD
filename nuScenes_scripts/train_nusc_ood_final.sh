name=cylinder_asym_networks_nusc
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python -u nuScenes_scripts/train_cylinder_asym_nuscenes_ood_final.py \
2>&1 | tee logs/${name}_logs_tee.txt