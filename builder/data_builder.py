'''

Date: 2023-03-23 09:00:34
LastEditTime: 2023-08-19 17:07:52

Description: 
FilePath: /openset_anomaly_detection/builder/data_builder.py
have a nice day
'''
# -*- coding:utf-8 -*-

# @file: data_builder.py 

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_test, collate_fn_BEV_val, collate_fn_BEV_incre
from dataloader.pc_dataset import get_pc_model_class
from torch.utils.data.distributed import DistributedSampler
from loguru import logger 

def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32],
          incre=None,
          SHAPENET_ANOMALY=None,debug=False,gen_resized_point_for_train = True):
    print(f"gen_resized_point_for_train : {gen_resized_point_for_train}")
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)

    train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                return_ref=train_ref, label_mapping=label_mapping, nusc=nusc)
    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)
    
    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True,
        ds_sample=gen_resized_point_for_train,
        SHAPENET_ANOMALY=SHAPENET_ANOMALY,
        shapenet_path=dataset_config["shapenet_path"],
        debug = debug
    )

    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        return_test=True,
        ds_sample=False,
        SHAPENET_ANOMALY=False,
        debug = debug,
        shapenet_path=dataset_config["shapenet_path"],
    )

    if torch.distributed.is_initialized():
        logger.info(f'ready to use distributed dataloader ')
        
        train_sampler = DistributedSampler(train_dataset)

        
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=train_dataloader_config["batch_size"],
                                                           collate_fn=collate_fn_BEV_incre if incre is not None else collate_fn_BEV,
                                                           sampler=train_sampler,
                                                           num_workers=train_dataloader_config["num_workers"],
                                                        #    shuffle=train_dataloader_config["shuffle"],
                                                           pin_memory=False,
                                                           drop_last=True)
        # TODO: when val batch size > 1, the prediction and label sizes does not match (4 v.s. 3)
        val_sampler = DistributedSampler(val_dataset)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_dataloader_config["batch_size"],
                                                         collate_fn=collate_fn_BEV_test,
                                                         sampler=val_sampler,
                                                        #  shuffle=val_dataloader_config["shuffle"],
                                                         num_workers=val_dataloader_config["num_workers"],
                                                         pin_memory=False)
    else:
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=train_dataloader_config["batch_size"],
                                                           collate_fn=collate_fn_BEV_incre if incre is not None else collate_fn_BEV,
                                                           shuffle=train_dataloader_config["shuffle"],
                                                           num_workers=train_dataloader_config["num_workers"],
                                                           pin_memory=False,
                                                           drop_last=True)
        # TODO: when val batch size > 1, the prediction and label sizes does not match (4 v.s. 3)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_dataloader_config["batch_size"],
                                                         collate_fn=collate_fn_BEV_test,
                                                         shuffle=val_dataloader_config["shuffle"],
                                                         num_workers=val_dataloader_config["num_workers"],
                                                         pin_memory=False)

    return train_dataset_loader, val_dataset_loader