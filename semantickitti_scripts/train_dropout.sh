name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym_dropout.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt