'''

Date: 2023-04-05 16:34:55
LastEditTime: 2023-04-10 19:26:56

Description: 
FilePath: /openset_anomaly_detection/builder/model_builder.py
have a nice day
'''
# -*- coding:utf-8 -*-

# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.segmentator_3d_asymm_spconv import RPL_Asymm_3d_spconv
from network.segmentator_3d_asymm_spconv import rpl_residual_block
from network.cylinder_fea_generator import cylinder_fea


def build(model_config, ENABLE_RPL=False):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']

    # Whether to enable RPL
    if ENABLE_RPL:
        # build a residual block here and pass it into RPL_Asymm_3d_spconv
        residual_block = rpl_residual_block(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)

        cylinder_3d_spconv_seg = RPL_Asymm_3d_spconv(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class,
            residual_block=residual_block)
    else:
        cylinder_3d_spconv_seg = Asymm_3d_spconv(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    """
    fea_dim = 9 by default, and the 9-th is the intensity 
    out_fea_dim == 256 by default
    num_input_features == 16 
    the channel from 9 to 256, then from 256 to 16 for cylinder_3d_spconv_seg  processing 
    
    """
    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
