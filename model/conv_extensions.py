'''
Author: Radillus
Date: 2023-10-02 11:30:33
LastEditors: Radillus
LastEditTime: 2023-10-02 18:43:42
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.nn import init


class Deconv2d(nn.Module):

    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_channels, *kernel_size, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)

        Returns
        -------
        x : torch.Tensor
        (batch_size, out_channels, (height-1)*stride[0]+kernel_size[0], (width-1)*stride[1]+kernel_size[1])
        """
        b, c, h, w = x.shape
        x = torch.einsum('bihw,oixy->boxyhw', x, self.weight)
        if self.bias is not None:
            x += self.bias
        x = x.reshape(b, self.out_channels*self.kernel_size[0]*self.kernel_size[1], h*w)
        nh, nw = (h-1)*self.stride[0]+self.kernel_size[0], (w-1)*self.stride[1]+self.kernel_size[1]
        x = F.fold(x, (nh, nw), self.kernel_size, stride=self.stride)
        return x


class PatchWiseConv2d(nn.Module):

    kernel_size: Tuple[int, int]
    out_shape: Tuple[int, int]
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    stride: Tuple[int, int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: _size_2_t,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        bias: bool = True,
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        self.kernel_size = kernel_size
        self.stride = stride

        self.out_shape = (in_shape[0]-kernel_size[0])//stride[0] + 1, (in_shape[1]-kernel_size[1])//stride[1] + 1
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size, *self.out_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_channels, *self.out_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)

        Returns
        -------
        x : torch.Tensor
        (batch_size, out_channels, (height-kernel_size[0])//stride[0] + 1, (width-kernel_size[1])//stride[1] + 1)
        """
        b, c, h, w = x.shape
        x = F.unfold(x, self.kernel_size, stride=self.stride)
        x = x.reshape(b, c, *self.kernel_size, *self.out_shape)
        x = torch.einsum('bixyhw,oixyhw->bohw', x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class PatchWiseDeconv2d(nn.Module):

    out_channels: int
    out_shape: Tuple[int, int]
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: _size_2_t,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        self.out_shape = (in_shape[0]-1)*stride[0]+kernel_size[0], (in_shape[1]-1)*stride[1]+kernel_size[1]
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *in_shape, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_channels, *self.out_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)

        Returns
        -------
        x : torch.Tensor
        (batch_size, out_channels, (height-1)*stride[0]+kernel_size[0], (width-1)*stride[1]+kernel_size[1])
        """
        b, c, h, w = x.shape
        x = torch.einsum('bihw,oihwxy->boxyhw', x, self.weight)
        x = x.reshape(b, self.out_channels*self.kernel_size[0]*self.kernel_size[1], h*w)
        x = F.fold(x, self.out_shape, self.kernel_size, stride=self.stride)
        if self.bias is not None:
            x += self.bias
        return x
