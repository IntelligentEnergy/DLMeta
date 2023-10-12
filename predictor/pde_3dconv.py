'''
Author: Radillus
Date: 2023-06-19 17:23:48
LastEditors: Radillus
LastEditTime: 2023-06-23 00:21:02
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationFunction(nn.Module):
    def __init__(self, clip:bool=True, decay:bool=False):
        super().__init__()
        self.clip = clip
        self.decay = decay
        if self.clip:
            self.linear_weight = nn.Parameter(torch.Tensor([1.0]))
            self.linear_bias = nn.Parameter(torch.Tensor([0.0]))
            self.postive_term_weight = nn.Parameter(torch.Tensor([-1.0]))  # similar to atan
            self.negative_term_weight = nn.Parameter(torch.Tensor([1.0]))
            self.total_bias = nn.Parameter(torch.Tensor([0.0]))
        else:
            self.negative_weight = nn.Parameter(torch.Tensor([0.0]))  # ReLU
            self.postive_weight = nn.Parameter(torch.Tensor([1.0]))
            self.linear_bias = nn.Parameter(torch.Tensor([0.0]))
            self.total_bias = nn.Parameter(torch.Tensor([0.0]))
        if self.decay:
            self.decay_weight = nn.Parameter(torch.Tensor([1.0]))
            self.decay_bias = nn.Parameter(torch.Tensor([0.0]))
    def forward(self, x):
        if self.clip:
            lin_x = self.linear_weight * x + self.linear_bias
            out = self.postive_term_weight / (1+torch.exp(lin_x)) + self.negative_term_weight / (1+torch.exp(-lin_x)) + self.total_bias
        else:
            out = F.relu(x+self.linear_bias) * self.postive_weight + F.relu(-x-self.linear_bias) * self.negative_weight + self.total_bias
        if self.decay:
            out = out * torch.exp(-torch.square(self.decay_weight * (x + self.decay_bias)))
        return out
    
class PDE3DConv(nn.Module):
    # this model do not support batch
    def __init__(
        self,
        hidden_channel:int=20,
        out_channel:int=12,
        xy_size:int=100,
        z_size:int=100,
        step_alpha:float=0.01,
        convolution_kernel_sizes:Tuple[int]=tuple([3]*100),
        shift_kernel_size:int=3,
    ):
        super().__init__()
        
        self.field_xy_size = xy_size
        self.field_z_size = z_size
        self.init_field = nn.Parameter(torch.randn(hidden_channel, z_size, xy_size, xy_size,))
        
        self.step_alpha = step_alpha
        self.convolution_layer_list = nn.ModuleList()
        cost_memory = (len(convolution_kernel_sizes)+2) * self.init_field.nelement() * self.init_field.element_size()
        for now_convolution_layer_kernel_size in convolution_kernel_sizes:
            if now_convolution_layer_kernel_size % 2 == 0:
                print(f'WARNING: kernel size = {now_convolution_layer_kernel_size} is even, it will be reduced by 1')
                now_convolution_layer_kernel_size -= 1
            now_convolution_padding = (now_convolution_layer_kernel_size - 1) // 2
            cost_memory += (hidden_channel ** 2) * (now_convolution_layer_kernel_size ** 3) * self.init_field.element_size()
            self.convolution_layer_list.append(nn.Conv3d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=now_convolution_layer_kernel_size,
                padding=now_convolution_padding,
            ))
        self.out_convolution = nn.Conv3d(hidden_channel, out_channel, (z_size, 1, 1,))
        print(f'Model will cost about {cost_memory / 1024 / 1024 / 1024:.2f} GB memory')
        if shift_kernel_size % 2 == 0:
            print(f'WARNING: shift_kernel_size = {shift_kernel_size} is even, it will be reduced by 1')
            shift_kernel_size -= 1
        self.shift_convolution = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=shift_kernel_size,
            padding=shift_kernel_size//2,
            groups=out_channel,
            bias=False,
        )
    
    def forward(self, surface:torch.Tensor):
        assert len(surface.shape) == 2, f"surface.shape = {surface.shape}"
        assert surface.shape[0] == surface.shape[1] == self.field_xy_size
        
        # surface is normalization Refractive rate [-1,1], for TiO2, Refractive rate = 3.4
        surface = (surface+1) / 2 * 2.4 + 1  # Refractive rate [1, 3.4]
        surface = 1 / torch.square(surface)  # 1 / epsilon
        
        field = self.init_field
        
        # for _ in range(self.convolution_times):
        for convolution_layer in self.convolution_layer_list:
            field = surface * convolution_layer(field) * self.step_alpha + field
        
        field = self.out_convolution(field) # [out_chanel, 1, xy_size, xy_size]
        field = torch.squeeze(field, 1) # [out_chanel, xy_size, xy_size]
        field = self.shift_convolution(field) # [out_chanel, xy_size, xy_size]
        return field

# if __name__ == "__main__":
#     import torchviz
#     # with torch.no_grad():
    
#     # test_xy_size = int(np.random.randint(10, 100))
#     test_xy_size = 50
#     test_out_channel = 12
#     print(f'target xy_size: {test_xy_size}, target out_channel: {test_out_channel}')
#     model = PDE3DConv(
#         hidden_channel=20,
#         out_chanel=test_out_channel,
#         xy_size=test_xy_size,
#         convolution_kernel_sizes=[3]*300,
#     )
#     model = torch.compile(model)
#     out = model(torch.randn(test_xy_size, test_xy_size))
#     print(out.shape)
#     torchviz.make_dot(out.mean(), params=dict(model.named_parameters()), show_saved=True).render("model", format="pdf")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    test_xy_size = 50
    test_hidden_channel = 5
    convolution_layer_num = 50
    model = torch.compile(PDE3DConv(
        hidden_channel=test_hidden_channel,
        out_channel=1,
        xy_size=test_xy_size,
        z_size=21,
        step_alpha=10.0 / convolution_layer_num,
        convolution_kernel_sizes=[5]*convolution_layer_num,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fig, axs = plt.subplots(1,3)
    plt.ion()
    plt.pause(0.001)
    i = 0
    while True:
        print(i)
        x = torch.rand(test_xy_size, test_xy_size)
        y = 1 / (1 + torch.tanh(x))
        y = y.unsqueeze(0)
        predict_y = model(x)
        loss = F.l1_loss(predict_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        axs[0].cla()
        axs[0].imshow(sy_:=np.squeeze(predict_y.detach().numpy()))
        axs[1].cla()
        axs[1].imshow(sy:=np.squeeze(y.detach().numpy()))
        axs[2].cla()
        axs[2].imshow(np.abs(sy-sy_),cmap='hot')
        plt.pause(0.001)
        i+=1
