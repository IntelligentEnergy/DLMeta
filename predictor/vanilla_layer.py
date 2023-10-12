'''
Author: Radillus
Date: 2023-06-04 22:16:25
LastEditors: Radillus
LastEditTime: 2023-06-09 14:50:24
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import TFNO

from predictor.tuneutils import get_activate_function


def caculate_conv_output_size(h, kernel_size, stride, padding=0):
    return int((h+2*padding-kernel_size)/stride+1)


class SourceLayer(nn.Module):
    def __init__(
        self,
        out_channels:int=1,
        hidden_size:int=100,
    ):
        super().__init__()
        self.source = nn.Parameter(torch.randn(1, out_channels, hidden_size, hidden_size))

    def forward(self, feature_map:torch.Tensor):
        out_size = feature_map.shape[-1]
        return F.interpolate(self.source, size=(out_size, out_size))


class CatLayer(nn.Module):
    def forward(self, feature_map:torch.Tensor, origin_surface:torch.Tensor):
        feature_map = torch.cat([feature_map, origin_surface], dim=1)
        return feature_map


class AddLayer(nn.Module):
    def forward(self, feature_map:torch.Tensor, origin_surface:torch.Tensor):
        feature_map = feature_map + origin_surface
        return feature_map


class MatAddLayer(nn.Module):
    def __init__(
        self,
        hidden_size:int=100,
    ):
        super().__init__()
        self.source = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))

    def forward(self, feature_map:torch.Tensor):
        out_size = feature_map.shape[-1]
        feature_map = feature_map + F.interpolate(self.source, size=(out_size, out_size))
        return feature_map


class MatDotLayer(nn.Module):
    def __init__(
        self,
        hidden_size:int=100,
    ):
        super().__init__()
        self.source = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))

    def forward(self, feature_map:torch.Tensor):
        out_size = feature_map.shape[-1]
        feature_map = feature_map * F.interpolate(self.source, size=(out_size, out_size))
        return feature_map


class MatMulLayer(nn.Module):
    def __init__(
        self,
        hidden_size:int=100,
    ):
        super().__init__()
        self.source = nn.Parameter(torch.randn(1,1,hidden_size, hidden_size))

    def forward(self, feature_map:torch.Tensor):
        out_size = feature_map.shape[-1]
        feature_map = torch.matmul(feature_map, F.interpolate(self.source, size=(out_size, out_size)))
        return feature_map


class ActivateLayer(nn.Module):
    def __init__(self, activate_function:str='ReLU'):
        super().__init__()
        self.activate_function = get_activate_function(activate_function)
    def forward(self, feature_map:torch.Tensor):
        feature_map = self.activate_function(feature_map)
        return feature_map


class ScalingMaxPoolingLayer(nn.Module):
    def __init__(self, kernel_size:int=2, stride:int=2):
        super().__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size, stride)
    def forward(self, feature_map:torch.Tensor):
        in_size = feature_map.shape[-1]
        feature_map = self.max_pooling(feature_map)
        feature_map = F.interpolate(feature_map, size=(in_size, in_size))
        return feature_map


class ConvolutionLayer(nn.Module):
    def __init__(
        self,
        in_channels:int=1,
        out_channels:int=1,
        kernel_size:int=5,
    ):
        if kernel_size % 2 == 0:
            kernel_size -= 1
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, padding_mode='circular')

    def forward(self, feature_map:torch.Tensor):
        feature_map = self.convolution(feature_map)
        return feature_map


class TFNOLayer(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        modes=50, # min 2 max in_size(included)
        hidden_channels=50, # min 1
        lifting_channels=250, # min 1
        projection_channels=250, # min 1
        n_layers=10, # min 2
        rank=1.0, # min 0.0 no max, this will affect the number of parameters
        skip='soft-gating', # {'linear', 'identity', 'soft-gating'}
        factorization='tucker', # {'tucker', 'cp', 'tt'}
        implementation='factorized', #  {'factorized', 'reconstructed'}
        activate_function='GELU',
    ):
        # assert 1 <= in_channels <= 5, f'in_channels should be in range [1, 5], got {in_channels}'
        assert 1 <= out_channels <= 5, f'out_channels should be in range [1, 5], got {out_channels}'
        assert 2 <= modes <= 50, f'modes should be in range [2, 50], got {modes}'
        assert 1 <= hidden_channels <= 50, f'hidden_channels should be greater than 0, got {hidden_channels}'
        assert 1 <= lifting_channels <= 250, f'lifting_channels should be greater than 0, got {lifting_channels}'
        assert 1 <= projection_channels <= 250, f'projection_channels should be greater than 0, got {projection_channels}'
        assert 2 <= n_layers <= 10, f'n_layers should be in range [2, 10], got {n_layers}'
        assert 0.0 < rank <= 1, f'rank should be greater than 0.0, got {rank}'
        assert skip in ['linear', 'identity', 'soft-gating'], f'skip should be in ["linear", "identity", "soft-gating"], got {skip}'
        assert factorization in ['tucker', 'cp', 'tt'], f'factorization should be in ["tucker", "cp", "tt"], got {factorization}'
        assert implementation in ['factorized', 'reconstructed'], f'implementation should be in ["factorized", "reconstructed"], got {implementation}'

        super().__init__()
        self.activate_funcion = get_activate_function(activate_function)
        self.tfno = TFNO(
            n_modes=(modes, modes),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=self.activate_funcion,
            rank=rank,
            skip=skip,
            factorization=factorization,
            implementation=implementation,
        )
    def forward(self, feature_map:torch.Tensor):
        return self.tfno(feature_map)


class ApartConvolutionLayer(nn.Module):
    def __init__(
        self,
        in_channels:int=1,
        kernel_size:int=5,
    ):
        super().__init__()
        self.convolution = ConvolutionLayer(in_channels, 1, kernel_size)

    def forward(self, feature_map:torch.Tensor):
        return feature_map, self.convolution(feature_map)
