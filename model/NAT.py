'''
Author: Radillus
Date: 2023-09-09 11:24:13
LastEditors: Radillus
LastEditTime: 2023-10-11 15:40:38
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from .conv_extensions import Deconv2d, PatchWiseConv2d, PatchWiseDeconv2d


valid_padding_modes = ('zeros', 'reflect', 'replicate', 'circular')

def combine_pad(
    x: torch.Tensor,
    height_pad: int,
    weight_pad: int,
    height_padding_mode: str = 'circular',
    weight_padding_mode: str = 'circular',
    height_pad_first: bool = True,
):
    if height_padding_mode == weight_padding_mode:
        return F.pad(x, (weight_pad, weight_pad, height_pad, height_pad), height_padding_mode)
    if height_pad_first:
        if height_pad:
            x = F.pad(x, (0, 0, height_pad, height_pad), height_padding_mode)
        if weight_pad:
            x = F.pad(x, (weight_pad, weight_pad, 0, 0), weight_padding_mode)
        return x
    else:
        if weight_pad:
            x = F.pad(x, (weight_pad, weight_pad, 0, 0), weight_padding_mode)
        if height_pad:
            x = F.pad(x, (0, 0, height_pad, height_pad), height_padding_mode)
        return x


class OverlapConvPatcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        origin_in_shape: _size_2_t,
        patch_size: _size_2_t = 3,
        overlap: _size_2_t = 1,
        use_patch_wise_conv: bool = False,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        origin_in_shape = _pair(origin_in_shape)
        patch_size = _pair(patch_size)
        overlap = _pair(overlap)
        self.padding = (patch_size[0]-overlap[0], patch_size[1]-overlap[1])
        self.height_padding_mode = height_padding_mode
        self.weight_padding_mode = weight_padding_mode
        if use_patch_wise_conv:
            self.conv = PatchWiseConv2d(
                in_channels=in_channels,
                out_channels=d_model,
                in_shape=(origin_in_shape[0]+self.padding[0]*2, origin_in_shape[1]+self.padding[1]*2),
                kernel_size=patch_size,
                stride=(patch_size[0]-overlap[0], patch_size[1]-overlap[1]),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=(patch_size[0]-overlap[0], patch_size[1]-overlap[1]),
            )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            this module does not check if height and width are divisible by patch_size separately.
            you need ensure that h - overlap and w - overlap are divisible by patch_size - overlap

        Returns
        -------
        x : torch.Tensor (batch_size, patch_nums[0], patch_nums[1], d_model)
            where patch_nums = (h-overlap)//(patch_size-overlap) + 2
        """
        x = combine_pad(x, *self.padding, self.height_padding_mode, self.weight_padding_mode)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class OverlapDeconvDepatcher(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_channels: int,
        origin_in_shape: _size_2_t,
        patch_size: _size_2_t = 3,
        overlap: _size_2_t = 1,
        use_patch_wise_conv: bool = False,
    ):
        super().__init__()
        patch_size = _pair(patch_size)
        overlap = _pair(overlap)
        origin_in_shape = _pair(origin_in_shape)
        patch_shape = (
            (origin_in_shape[0]-overlap[0])//(patch_size[0]-overlap[0]) + 2,
            (origin_in_shape[1]-overlap[1])//(patch_size[1]-overlap[1]) + 2,
        )
        self.patch_size = patch_size
        self.overlap = overlap
        if use_patch_wise_conv:
            self.deconv = PatchWiseDeconv2d(
                in_channels=d_model,
                out_channels=out_channels,
                in_shape=patch_shape,
                kernel_size=patch_size,
                stride=(patch_size[0]-overlap[0], patch_size[1]-overlap[1]),
            )
        else:
            self.deconv = Deconv2d(
                in_channels=d_model,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=(patch_size[0]-overlap[0], patch_size[1]-overlap[1]),
            )

    def forward(self, x: torch.Tensor):
        """
        reverse operation of OverlapConvPatcher
        """

        x = x.permute(0, 3, 1, 2)
        x = self.deconv(x)
        x = x[:, :,
              (self.patch_size[0]-self.overlap[0]):-(self.patch_size[0]-self.overlap[0]),
              (self.patch_size[1]-self.overlap[1]):-(self.patch_size[1]-self.overlap[1])]
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        upscale: float = 4.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, int(in_features*upscale))
        self.act = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(int(in_features*upscale), in_features)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class NATLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        kernel_size: int = 7,
        upscale: float = 4.0,
        dropout: float = 0.0,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.height_padding_mode = height_padding_mode
        self.weight_padding_mode = weight_padding_mode

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = NeighborhoodAttention(
            dim=d_model,
            kernel_size=kernel_size,
            dilation=1,
            num_heads=nhead,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, upscale)

    def forward(self, x: torch.Tensor):
        out = self.norm1(x)

        # due to Neighborhood Attention dose not support padding, instead it does not move window at the edge
        # we need to pad manually to ensure consistency
        # see https://github.com/SHI-Labs/Neighborhood-Attention-Transformer#how-neighborhood-attention-works
        # for how Neighborhood Attention works
        out = out.permute(0, 3, 1, 2)
        out = combine_pad(out, self.kernel_size//2, self.kernel_size//2, self.height_padding_mode, self.weight_padding_mode)
        out = out.permute(0, 2, 3, 1)
        out = self.attn(out)
        out = out[:, self.kernel_size//2:-(self.kernel_size//2), self.kernel_size//2:-(self.kernel_size//2), :]

        out = x + self.drop_path(out)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out


class PatchWiseConv2dSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: _size_2_t,
        kernel_size: _size_2_t,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        kernel_size = _pair(kernel_size)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        self.padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.height_padding_mode = height_padding_mode
        self.weight_padding_mode = weight_padding_mode
        self.conv = PatchWiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            in_shape=(in_shape[0]+2*(kernel_size[0]//2), in_shape[1]+2*(kernel_size[1]//2)),
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, x: torch.Tensor):
        x = combine_pad(x, *self.padding, self.height_padding_mode, self.weight_padding_mode)
        x = self.conv(x)
        return x


class ResLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_shape: _size_2_t,
        kernel_size: _size_2_t,
        upscale: float = 4.,
        drop: float = 0.0,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        kernel_size = _pair(kernel_size)
        self.cv1 = PatchWiseConv2dSame(
            in_channels=in_channels,
            out_channels=int(in_channels*upscale),
            in_shape=in_shape,
            kernel_size=kernel_size,
            height_padding_mode=height_padding_mode,
            weight_padding_mode=weight_padding_mode,
        )
        self.ac = nn.LeakyReLU(inplace=True)
        self.cv2 = nn.Conv2d(int(in_channels*upscale), in_channels, 1)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.drop2 = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        out = self.drop1(self.cv1(x))
        out = self.ac(out)
        out = self.cv2(out)
        return self.drop2(out) + x


class OverlapNAT(nn.Module):
    def __init__(
        self,
        in_shape: _size_2_t,
        in_channels: int,
        out_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.01,
        patcher_kernel_size: _size_2_t = 4,
        patcher_overlap: _size_2_t = 2,
        nat_kernel_size: int = 5,
        nat_layers: int = 3,
        res_kernel_size: _size_2_t = 5,
        res_layers: int = 1,
        upscale: float = 4.0,
        use_patch_wise_conv: bool = True,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        patcher_kernel_size = _pair(patcher_kernel_size)
        patcher_overlap = _pair(patcher_overlap)
        res_kernel_size = _pair(res_kernel_size)
        assert (in_shape[0]-patcher_overlap[0]) % (patcher_kernel_size[0]-patcher_overlap[0]) == 0, \
            f'img_size[0]-patcher_overlap[0] must be divisible by patcher_kernel_size[0]-encoder_overlap[0]'
        assert (in_shape[1]-patcher_overlap[1]) % (patcher_kernel_size[1]-patcher_overlap[1]) == 0, \
            f'img_size[1]-patcher_overlap[1] must be divisible by patcher_kernel_size[1]-encoder_overlap[1]'
        assert res_kernel_size[0] % 2 == 1 and res_kernel_size[1] % 2 == 1, \
            f'res_kernel_size must be odd'
        assert d_model % nhead == 0, f'd_model must be divisible by nhead'
        assert patcher_kernel_size[0] - patcher_overlap[0] >= 1, f'overlap must be smaller than kernel_size'
        assert patcher_kernel_size[1] - patcher_overlap[1] >= 1, f'overlap must be smaller than kernel_size'
        assert dropout >= 0.0 and dropout <= 1.0, f'dropout must be in [0, 1]'
        assert upscale > 0.0, f'upscale must be greater than 0'
        assert height_padding_mode in valid_padding_modes, f'height_padding_mode must be in {valid_padding_modes}'
        assert weight_padding_mode in valid_padding_modes, f'weight_padding_mode must be in {valid_padding_modes}'

        self.positional_embedding_w = nn.Parameter(
            torch.rand(in_channels, *in_shape)+0.5
        )
        self.positional_embedding_b = nn.Parameter(
            torch.rand(in_channels, *in_shape)-0.5
        )
        self.position_dropout = nn.Dropout(p=dropout, inplace=True)

        self.patcher = OverlapConvPatcher(
            in_channels=in_channels,
            d_model=d_model,
            origin_in_shape=in_shape,
            patch_size=patcher_kernel_size,
            overlap=patcher_overlap,
            use_patch_wise_conv=use_patch_wise_conv,
            height_padding_mode=height_padding_mode,
            weight_padding_mode=weight_padding_mode,
        )
        self.nomalizer = nn.LayerNorm(d_model)
        self.depatcher = OverlapDeconvDepatcher(
            d_model=d_model,
            out_channels=out_channels,
            origin_in_shape=in_shape,
            patch_size=patcher_kernel_size,
            overlap=patcher_overlap,
            use_patch_wise_conv=use_patch_wise_conv,
        )
        self.nat_layers = nn.ModuleList([NATLayer(
                                            d_model=d_model,
                                            nhead=nhead,
                                            kernel_size=nat_kernel_size,
                                            upscale=upscale,
                                            dropout=dropout,
                                            height_padding_mode=height_padding_mode,
                                            weight_padding_mode=weight_padding_mode,
                                        ) for _ in range(nat_layers)])

        self.res_layers = nn.ModuleList([ResLayer(
                                            in_channels=in_channels,
                                            in_shape=in_shape,
                                            kernel_size=res_kernel_size,
                                            upscale=upscale,
                                            drop=dropout,
                                            height_padding_mode=height_padding_mode,
                                            weight_padding_mode=weight_padding_mode,
                                        ) for _ in range(res_layers)])


    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)

        Returns
        -------
        x : torch.Tensor (batch_size, out_channels, height, width)
            same as input
        """

        x = x * self.positional_embedding_w
        x = x + self.positional_embedding_b
        x = self.patcher(x)
        x = self.nomalizer(x)
        for nat_layer in self.nat_layers:
            x = nat_layer(x)
        x = self.depatcher(x)
        for res_layer in self.res_layers:
            x = self.position_dropout(x)
            x = res_layer(x)
        return x


class OverlapNATSupervisorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: _size_2_t,
        out_shape: _size_2_t,
        nhead: int = 8,
        nat_kernel_size: int = 5,
        nat_layers: int = 3,
        upscale: float = 4.0,
        use_patch_wise_conv: bool = True,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        out_shape = _pair(out_shape)
        conv_kernel_size = (in_shape[0]-out_shape[0]+1, in_shape[1]-out_shape[1]+1)
        self.nats = nn.ModuleList([NATLayer(
            d_model=in_channels,
            nhead=nhead,
            kernel_size=nat_kernel_size,
            upscale=upscale,
            dropout=0.0,
            height_padding_mode=height_padding_mode,
            weight_padding_mode=weight_padding_mode,
        ) for _ in range(nat_layers)])
        if use_patch_wise_conv:
            self.conv = PatchWiseConv2d(in_channels, out_channels, in_shape, conv_kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (b, hi, wi, ci)

        Returns:
            x (torch.Tensor): (b, ho, wo, co)
        """
        for nat in self.nats:
            x = nat(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class OverlapNATSupervisor(nn.Module):
    def __init__(
        self,
        in_shape: _size_2_t,
        in_channels: int,
        d_model: int = 8,
        nhead: int = 2,
        nat_kernel_size: int = 5,
        nat_layers: int = 3,
        blocks: int = 8,
        upscale: float = 4.0,
        use_patch_wise_conv: bool = True,
        height_padding_mode: str = 'circular',
        weight_padding_mode: str = 'circular',
    ):
        super().__init__()
        in_shape = _pair(in_shape)
        assert d_model % nhead == 0, f'd_model must be divisible by nhead'
        assert upscale > 0.0, f'upscale must be greater than 0'
        assert height_padding_mode in valid_padding_modes, f'height_padding_mode must be in {valid_padding_modes}'
        assert weight_padding_mode in valid_padding_modes, f'weight_padding_mode must be in {valid_padding_modes}'

        self.positional_embedding_w = nn.Parameter(
            torch.rand(in_channels, *in_shape)+0.5
        )
        self.positional_embedding_b = nn.Parameter(
            torch.rand(in_channels, *in_shape)-0.5
        )

        if use_patch_wise_conv:
            self.patcher = PatchWiseConv2d(in_channels, d_model, in_shape, 1)
        else:
            self.patcher = nn.Conv2d(in_channels, d_model, 1)
        self.nomalizer = nn.LayerNorm(d_model)
        
        hs = torch.linspace(in_shape[0], 1, blocks+1, dtype=torch.int)
        ws = torch.linspace(in_shape[1], 1, blocks+1, dtype=torch.int)
        self.intermediate_shapes = [(int(h), int(w)) for h, w in zip(hs, ws)]
        
        self.intermediate_channels = torch.linspace(d_model, d_model*round(upscale), blocks+1)
        self.intermediate_channels = torch.round(self.intermediate_channels / nhead).int() * nhead
        self.intermediate_channels = self.intermediate_channels.int().tolist()
        
        self.blocks = nn.ModuleList([OverlapNATSupervisorBlock(
            in_channels=self.intermediate_channels[i],
            out_channels=self.intermediate_channels[i+1],
            in_shape=self.intermediate_shapes[i],
            out_shape=self.intermediate_shapes[i+1],
            nhead=nhead,
            nat_kernel_size=nat_kernel_size,
            nat_layers=nat_layers,
            upscale=upscale,
            use_patch_wise_conv=use_patch_wise_conv,
            height_padding_mode=height_padding_mode,
            weight_padding_mode=weight_padding_mode,
        ) for i in range(blocks)])
        
        self.outs = nn.Sequential(
            nn.Linear(self.intermediate_channels[-1], int(self.intermediate_channels[-1]*upscale)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(self.intermediate_channels[-1]*upscale), 1),
            nn.Sigmoid(),
        )


    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)

        Returns
        -------
        x : torch.Tensor (batch_size)
        """

        x = x * self.positional_embedding_w
        x = x + self.positional_embedding_b
        x = self.patcher(x)
        x = x.permute(0, 2, 3, 1)
        x = self.nomalizer(x)
        for b in self.blocks:
            x = b(x)
        x = torch.squeeze(x)
        x = self.outs(x)
        x = torch.squeeze(x)
        return x
