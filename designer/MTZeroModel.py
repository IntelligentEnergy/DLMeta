'''
Author: Radillus
Date: 2023-06-24 19:46:23
LastEditors: Radillus
LastEditTime: 2023-06-24 23:48:02
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Callable

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from utils import caculate_convolution_output_size


class Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int = 12,
        input_size: int = 100,
        output_size: int = 500,
    ):
        super().__init__()
        assert (input_size ** 2) >= output_size, f'input_size: {input_size}, embedding_size: {output_size}'
        now_size, now_channel = input_size, input_channel
        self.convolution_layer = []
        while now_channel * (now_size ** 2) > output_size * 2:
            self.convolution_layer.append(
                nn.Conv2d(
                    in_channels=now_channel,
                    out_channels=int(now_channel*8/3),
                    kernel_size=3,
                    stride=2,
                )
            )
            self.convolution_layer.append(nn.Tanh())
            now_size = caculate_convolution_output_size(now_size, 3, int(now_channel*8/3))
            now_channel = int(now_channel*8/3)
        self.convolution_layer = nn.Sequential(*self.convolution_layer)
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(now_channel * (now_size ** 2), now_channel * (now_size ** 2) * 3 // 2),
            nn.Tanh(),
            nn.Linear(now_channel * (now_size ** 2) * 3 // 2, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )
    def forward(self, surface:torch.Tensor):
        return self.linear_layer(self.convolution_layer(surface))


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        output_channel: int = 12,
        output_size: int = 100,
    ):
        self.split_channel_layer = nn.Linear(input_size, output_channel * input_size)
        self.single_channel_image_layer = nn.Sequential(
            nn.Linear(input_size , output_size ** 2),
            nn.Unflatten(2, (output_size, output_size))
        )
    def forward(self, embedding:torch.Tensor):
        embedding = self.split_channel_layer(embedding)
        embedding = embedding.reshape(embedding.shape[0], -1, embedding.shape[-1])
        return self.single_channel_image_layer(embedding)


class DesignNetwork(nn.Module):
    # like multihead attention, but q is fixed with target's q so there's no Wq
    # and attention score is actual score, so there's no Wk
    def __init__(
        self,
        surface_embedding_size: int = 500,
        field_embedding_size: int = 10000,
        design_sequence_length: int = 10,
    ):
        super().__init__()
        self.output_size = surface_embedding_size
        self.design_sequence_length = design_sequence_length

    def forward(self, embedded_target_field:torch.Tensor, design_sequence:torch.Tensor, score_sequence:torch.Tensor):
         nn.TransformerDecoder
