'''
Author: Radillus
Date: 2023-06-19 23:59:44
LastEditors: Radillus
LastEditTime: 2023-07-02 16:55:51
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PDE2DConv(nn.Module):
    # this model do not support batch
    def __init__(
        self,
        hidden_channel:int=100,
        out_chanel:int=12,
        xy_size:int=100,
        step_alpha:float=0.01,
        convolution_times:int=5,
        convolution_kernel_size:int=5,
    ):
        super().__init__()
        # self.activation_function = ActivationFunction(clip=activation_function_clip, decay=activation_function_decay)
        self.xy_size = xy_size
        self.field = nn.Parameter(torch.randn(hidden_channel, xy_size, xy_size,))
        self.step_alpha = step_alpha
        if convolution_kernel_size % 2 == 0:
            convolution_kernel_size -= 1
        self.convolution_layers = nn.Conv2d(hidden_channel, hidden_channel, convolution_kernel_size, padding=convolution_kernel_size//2)
        self.convolution_times = convolution_times
        self.output_layer = nn.Conv2d(hidden_channel, out_chanel, 1)

    def forward(self, x:torch.Tensor):
        field = self.field.clone()

        for i in range(self.convolution_times):
            field = field + self.step_alpha * self.convolution_layers(field) * x

        field = self.output_layer(field)
        return field

import torchviz
model = PDE2DConv()
model = torch.compile(model)
sf = torch.randn(100,100)
out = model(sf).mean()
torchviz.make_dot(out,dict(model.named_parameters()),show_saved=True).render('model',format='pdf')

