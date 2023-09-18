CUDA_VISIBLE_DEVICES=7 python val_cylinder_asym_ood.py --load_path '../../semantic_kitti/checkpoints/Feb23_6_cards/model_epoch_0.pt' &
CUDA_VISIBLE_DEVICES=7 python val_cylinder_asym_ood.py --load_path '../../semantic_kitti/checkpoints/Feb23_6_cards/model_epoch_10.pt' &
CUDA_VISIBLE_DEVICES=7 python val_cylinder_asym_ood.py --load_path '../../semantic_kitti/checkpoints/Feb23_6_cards/model_epoch_20.pt' &
CUDA_VISIBLE_DEVICES=7 python val_cylinder_asym_ood.py --load_path '../../semantic_kitti/checkpoints/Feb23_6_cards/model_epoch_30.pt'