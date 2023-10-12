'''
Author: Radillus
Date: 2023-06-05 14:51:56
LastEditors: Radillus
LastEditTime: 2023-06-10 21:25:00
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from optuna.trial import Trial

from predictor.vanilla_layer import *
from predictor.tuneutils import ACTIVATE_FUNCTION_LIST


MAX_SOURCE_OUT_CHANNELS = 10
MAX_SOURCE_HIDDEN_SIZE = 500

MAX_MAXPOOL_KERNEL_SIZE = 10
MAX_MAXPOOL_STRIDE = 10

MAX_CONVOLUTION_OUT_CHANNELS = 10
MAX_CONVOLUTION_KERNEL_SIZE = 50

MAX_TFNO_OUT_CHANNELS = 3
MAX_TFNO_MODES = 30
MAX_TFNO_HIDDEN_CHANNELS = 20
MAX_TFNO_LIFTING_CHANNELS = 100
MAX_TFNO_PROJECTION_CHANNELS = 100
MAX_TFNO_N_LAYERS = 10
MAX_TFNO_RANK = 1.0

def generate_source_layer(trail:Trial, layer_id:int, now_channel:int):
    out_channels = trail.suggest_int(f'l{layer_id}_source_oc', 1, MAX_SOURCE_OUT_CHANNELS)
    return SourceLayer(
        out_channels=out_channels,
        hidden_size=trail.suggest_int(f'l{layer_id}_source_hs', 10, MAX_SOURCE_HIDDEN_SIZE),
    ), out_channels

def generate_cat_layer(trail:Trial, layer_id:int, now_channel:int):
    return CatLayer(), now_channel + 1

def generate_add_layer(trail:Trial, layer_id:int, now_channel:int):
    return AddLayer(), now_channel

def generate_matadd_layer(trail:Trial, layer_id:int, now_channel:int):
    return MatAddLayer(
        hidden_size=trail.suggest_int(f'l{layer_id}_matadd_hs', 10, MAX_SOURCE_HIDDEN_SIZE),
    ), now_channel

def generate_matdot_layer(trail:Trial, layer_id:int, now_channel:int):
    return MatDotLayer(
        hidden_size=trail.suggest_int(f'l{layer_id}_matdot_hs', 10, MAX_SOURCE_HIDDEN_SIZE),
    ), now_channel

def generate_matmul_layer(trail:Trial, layer_id:int, now_channel:int):
    return MatMulLayer(
        hidden_size=trail.suggest_int(f'l{layer_id}_matmul_hs', 10, MAX_SOURCE_HIDDEN_SIZE),
    ), now_channel

def generate_activate_layer(trail:Trial, layer_id:int, now_channel:int):
    return ActivateLayer(
        activate_function=trail.suggest_categorical(f'l{layer_id}_af', ACTIVATE_FUNCTION_LIST)
    ), now_channel

def generate_batch_normalization_layer(trail:Trial, layer_id:int, now_channel:int):
    return nn.BatchNorm2d(now_channel), now_channel

def generate_layer_normalization_layer(trail:Trial, layer_id:int, now_channel:int):
    return nn.LayerNorm(now_channel), now_channel

def generate_scaling_maxpool_layer(trail:Trial, layer_id:int, now_channel:int):
    return ScalingMaxPoolingLayer(
        kernel_size=trail.suggest_int(f'l{layer_id}_maxpool_ks', 1, MAX_MAXPOOL_KERNEL_SIZE),
        stride=trail.suggest_int(f'l{layer_id}_maxpool_st', 1, MAX_MAXPOOL_STRIDE),
    ), now_channel

def generate_convolution_layer(trail:Trial, layer_id:int, now_channel:int):
    out_channels = trail.suggest_int(f'l{layer_id}_conv_oc', 1, MAX_CONVOLUTION_OUT_CHANNELS)
    return ConvolutionLayer(
        in_channels=now_channel,
        out_channels=out_channels,
        kernel_size=trail.suggest_int(f'l{layer_id}_conv_ks', 1, MAX_CONVOLUTION_KERNEL_SIZE),
    ), out_channels

def generate_tfno_layer(trail:Trial, layer_id:int, now_channel:int):
    out_channels = trail.suggest_int(f'l{layer_id}_tfno_oc', 1, MAX_TFNO_OUT_CHANNELS)
    return TFNOLayer(
        in_channels=now_channel,
        out_channels=out_channels,
        modes=trail.suggest_int(f'l{layer_id}_tfno_m', 2, MAX_TFNO_MODES),
        hidden_channels=trail.suggest_int(f'l{layer_id}_tfno_hc', 2, MAX_TFNO_HIDDEN_CHANNELS),
        lifting_channels=trail.suggest_int(f'l{layer_id}_tfno_lc', 2, MAX_TFNO_LIFTING_CHANNELS),
        projection_channels=trail.suggest_int(f'l{layer_id}_tfno_pc', 2, MAX_TFNO_PROJECTION_CHANNELS),
        n_layers=trail.suggest_int(f'l{layer_id}_tfno_nl', 2, MAX_TFNO_N_LAYERS),
        rank=trail.suggest_float(f'l{layer_id}_tfno_r', 0.1, MAX_TFNO_RANK),
        skip=trail.suggest_categorical(f'l{layer_id}_tfno_s', ['linear', 'identity', 'soft-gating']),
        factorization=trail.suggest_categorical(f'l{layer_id}_tfno_f', ['tucker', 'cp', 'tt']),
        implementation=trail.suggest_categorical(f'l{layer_id}_tfno_i', ['factorized', 'reconstructed']),
        activate_function=trail.suggest_categorical(f'l{layer_id}_af', ACTIVATE_FUNCTION_LIST)
    ), out_channels

def generate_apart_convolution_layer(trail:Trial, layer_id:int, now_channel:int):
    return ApartConvolutionLayer(
        in_channels=now_channel,
        kernel_size=trail.suggest_int(f'l{layer_id}_apart_conv_ks', 1, MAX_CONVOLUTION_KERNEL_SIZE),
    ), now_channel

layer_list = [
    'source',
    'cat',
    'add',
    'matadd',
    'matdot',
    'matmul',
    'activate',
    'batch_normalization',
    # 'layer_normalization',
    'scaling_maxpool',
    'convolution',
    'tfno',
    'apart_convolution'
]

def match_layer_type(trail:Trial, layer_type:str, layer_id:int, now_channel:int):
    match layer_type:
        case 'source':
            return generate_source_layer(trail, layer_id, now_channel)
        case 'cat':
            return generate_cat_layer(trail, layer_id, now_channel)
        case 'add':
            return generate_add_layer(trail, layer_id, now_channel)
        case 'matadd':
            return generate_matadd_layer(trail, layer_id, now_channel)
        case 'matdot':
            return generate_matdot_layer(trail, layer_id, now_channel)
        case 'matmul':
            return generate_matmul_layer(trail, layer_id, now_channel)
        case 'activate':
            return generate_activate_layer(trail, layer_id, now_channel)
        case 'batch_normalization':
            return generate_batch_normalization_layer(trail, layer_id, now_channel)
        case 'layer_normalization':
            return generate_layer_normalization_layer(trail, layer_id, now_channel)
        case 'scaling_maxpool':
            return generate_scaling_maxpool_layer(trail, layer_id, now_channel)
        case 'convolution':
            return generate_convolution_layer(trail, layer_id, now_channel)
        case 'tfno':
            return generate_tfno_layer(trail, layer_id, now_channel)
        case 'apart_convolution':
            return generate_apart_convolution_layer(trail, layer_id, now_channel)
        case _:
            raise ValueError(f'Unknown layer type {layer_type}')

def generate_last_layer(trail:Trial, now_channel:int):
    match trail.suggest_categorical(f'last', ['convolution', 'tfno']):
        case 'convolution':
            return ConvolutionLayer(
                in_channels=now_channel,
                out_channels=1,
                kernel_size=trail.suggest_int(f'last_conv_ks', 1, MAX_CONVOLUTION_KERNEL_SIZE),
            )
        case 'tfno':
            return TFNOLayer(
                in_channels=now_channel,
                out_channels=1,
                modes=trail.suggest_int(f'last_tfno_m', 2, MAX_TFNO_MODES),
                hidden_channels=trail.suggest_int(f'last_tfno_hc', 1, MAX_TFNO_HIDDEN_CHANNELS),
                lifting_channels=trail.suggest_int(f'last_tfno_lc', 1, MAX_TFNO_LIFTING_CHANNELS),
                projection_channels=trail.suggest_int(f'last_tfno_pc', 1, MAX_TFNO_PROJECTION_CHANNELS),
                n_layers=trail.suggest_int(f'last_tfno_nl', 2, MAX_TFNO_N_LAYERS),
                rank=trail.suggest_float(f'last_tfno_r', 0.1, MAX_TFNO_RANK),
                skip=trail.suggest_categorical(f'last_tfno_s', ['linear', 'identity', 'soft-gating']),
                factorization=trail.suggest_categorical(f'last_tfno_f', ['tucker', 'cp', 'tt']),
                implementation=trail.suggest_categorical(f'last_tfno_i', ['factorized', 'reconstructed']),
                activate_function=trail.suggest_categorical(f'last_af', ACTIVATE_FUNCTION_LIST),
            )
