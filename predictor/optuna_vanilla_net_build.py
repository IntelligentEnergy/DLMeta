'''
Author: Radillus
Date: 2023-06-05 13:18:01
LastEditors: Radillus
LastEditTime: 2023-06-09 15:03:59
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from optuna.trial import Trial
import torch
import torch.nn as nn

from predictor.tuneutils import ACTIVATE_FUNCTION_LIST
from predictor.optuna_vanilla_layer_generate import layer_list, match_layer_type, generate_last_layer
from predictor.vanilla_layer import ApartConvolutionLayer, AddLayer, CatLayer


MAX_SKIP_LAYER_NUM = 10

def generate_layer(trail:Trial, layer_id:int, now_channel:int, skip_table:dict):
    if layer_id in skip_table.values():
        skip_layer_type = trail.suggest_categorical(f'l{layer_id}_skip', ['add', 'cat'])
        return match_layer_type(trail, skip_layer_type, layer_id, now_channel)
    layer_type = trail.suggest_categorical(f'l{layer_id}', layer_list)
    return match_layer_type(trail, layer_type, layer_id, now_channel)


def generate_model(trail:Trial):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_layer_num = trail.suggest_int('layer_num', 1, 20)
            self.model_layer_list = nn.ModuleList()
            now_channel = 1
            self.skip_table = dict()
            self.skip_transition = dict()
            for layer_id in range(self.model_layer_num):
                layer, now_channel = generate_layer(trail, layer_id, now_channel, self.skip_table)
                self.model_layer_list.append(layer)
                if isinstance(layer, ApartConvolutionLayer):
                    skip_to = layer_id + trail.suggest_int(f'l{layer_id}_skip_to', 1, MAX_SKIP_LAYER_NUM)
                    self.skip_table[layer_id] = skip_to
            # if now_channel != 1:
            self.last_layer = generate_last_layer(trail, now_channel)
            # else:
            #     self.last_layer = nn.Identity()
        def forward(self, x:torch.Tensor):
            orign_surface = x
            now_surface = x
            for layer_id, layer in enumerate(self.model_layer_list):
                if isinstance(layer, ApartConvolutionLayer):
                    now_surface, residual = layer(now_surface)
                    self.skip_transition[self.skip_table[layer_id]] = residual
                    continue
                if layer_id in self.skip_table.values():
                    now_surface = layer(now_surface, self.skip_transition[layer_id])
                    continue
                if isinstance(layer, (AddLayer, CatLayer)):
                    now_surface = layer(now_surface, orign_surface)
                    continue
                now_surface = layer(now_surface)
            now_surface = self.last_layer(now_surface)
            return now_surface
    return Model()
