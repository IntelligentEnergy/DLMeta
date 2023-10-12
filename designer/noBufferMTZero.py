'''
Author: Radillus
Date: 2023-06-23 15:14:16
LastEditors: Radillus
LastEditTime: 2023-07-02 14:18:05
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import tools.TopoSampler

class Mat2MatEnv:
    def __init__(self, size, channel):
        self.size = size
        self.channel = channel
        self.samplers = [tools.TopoSampler.TopoSampler(size) for _ in range(channel)]
        self.reset()
    def reset(self):
        self.count = 0
        self.target_mat = [torch.from_numpy(samplers.sample()) for samplers in self.samplers]
        self.target_mat = torch.stack(self.target_mat)
        self.transformed_target_mat = torch.matmul(self.target_mat, self.target_mat)
        self.mat = torch.zeros((self.channel, self.size, self.size))
    def step(self, action:torch.Tensor):
        self.mat = action
        self.transformed_mat = torch.matmul(self.mat, self.mat)
        self.loss = F.l1_loss(self.transformed_mat, self.transformed_target_mat)
        self.count += 1
        if self.count >= 100:
            self.reset()


def convolution_output_size(input_size:int, kernel_size:int, stride:int, padding:int):
    return (input_size - kernel_size + 2 * padding) // stride + 1

class H(nn.Module):
    """a encoder"""
    def __init__(
        self,
        input_size: int = 100,
        input_channel: int = 12,
        embedded_size: int = 100,
    ):
        super().__init__()
        convolution_layers = []
        now_size, now_channel = input_size, input_channel
        while True:
            convolution_layers.append(
                nn.Conv2d(now_channel, now_channel * 3 // 2, 3, 2)
            )
            convolution_layers.append(
                nn.Tanh()
            )
            now_size = convolution_output_size(now_size, 5, 2, 0)
            now_channel = now_channel * 3 // 2
            if now_size <= np.sqrt(embedded_size) + 1:
                break
        self.convolution = nn.Sequential(*convolution_layers)
        # NOTICE: this only works for non batched input
        self.linear = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(now_size * now_size * now_channel, embedded_size),
            nn.Tanh(),
            nn.Linear(embedded_size, embedded_size),
        )
    def forward(self, x):
        # input: (channel, size, size) output: (embedded_size)
        x = self.convolution(x)
        x = self.linear(x)
        return x


class invH(nn.Module):
    """a decoder"""
    def __init__(
        self,
        embedded_size: int = 100,
        output_size: int = 100,
        output_channel: int = 12,
    ):
        super().__init__()
        # NOTICE: this only works for non batched input
        self.image = nn.Sequential(
            nn.Linear(embedded_size,output_channel * embedded_size),
            nn.Unflatten(0, (output_channel, embedded_size)),
            nn.Linear(embedded_size, output_size * output_size),
            nn.Unflatten(1, (output_size, output_size)),
        )
    def forward(self, x):
        return self.image(x)


class ViT(nn.Module):
    def __init__(
        self,
        input_size:int,
        input_channels:int,
        output_size:int,
        output_channels:int,
        encoding_size:int = 100,
        encoding_length:int = 100,
    ):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(input_channels,input_size,input_size))
        
        convolution_kernel_size = input_size * 2 //(int(np.sqrt(encoding_size)) + 1)
        convolution_stride = convolution_kernel_size // 2
        convolution_output_size = (input_size - convolution_kernel_size)//convolution_stride + 1
        # NOTICE: this only works for non batched input
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoding_length, convolution_kernel_size, convolution_stride),
            nn.Tanh(),
            nn.Flatten(1),
            nn.Linear(convolution_output_size ** 2, encoding_size)
        )
        
        self.encoding_length = encoding_length
        self.seed_array = self.register_buffer('seed_array', torch.zeros(1,encoding_size))
        self.prior_transformer = nn.Transformer(d_model=encoding_size)
        self.posterior_transformer = nn.Transformer(d_model=encoding_size)
        
        square_size = int(np.sqrt(encoding_size))
        transpose_convolution_kernerl_size = int(output_size / ((square_size - 1)/2 +1))
        transpose_convolution_stride = transpose_convolution_kernerl_size // 2
        transpose_convolution_kernerl_size = output_size - (square_size - 1) * transpose_convolution_stride
        # NOTICE: this only works for non batched input
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, square_size ** 2),
            nn.Tanh(),
            nn.Unflatten(1, (square_size, square_size)),
            nn.ConvTranspose2d(encoding_length, output_channels, transpose_convolution_kernerl_size, transpose_convolution_stride),
        )
    def forward(self, x:torch.Tensor):
        assert x.ndim == 3, f'Unsupported input shape {x.shape}, only non batch input is available'
        x = self.encoder(x)
        tgt = self.seed_array
        for _ in range(self.encoding_length):
            out = self.prior_transformer(x, tgt)[-1,:].reshape(1,-1)
            tgt = torch.cat([tgt,out],dim=0)

        # remove seed
        tgt = tgt[1:,:]
        
        # here we do not use mask because there is no sequence here
        tgt = self.posterior_transformer(x, tgt)
        
        tgt = self.decoder(tgt)
        return tgt

class G(nn.Module):
    """use s_mat_t,s_target,s_ to predict s_t+1"""
    def __init__(
        self,
        embedded_size: int = 100,
        output_size: int = 100,
        output_channel: int = 12,
    ):
        super().__init__()
        # NOTICE: this only works for non batched input
        self.image = nn.Sequential(
            nn.Linear(embedded_size,output_channel * embedded_size),
            nn.Unflatten(0, (output_channel, embedded_size)),
            nn.Linear(embedded_size, output_size * output_size),
            nn.Unflatten(1, (output_size, output_size)),
        )
    def forward(self, x):
        return self.image(x)

class MTZero:
    def __init__(
        self,
        size = 100,
        from_channel = 1,
        to_channel = 12,
        embedding_size:int = 100,
        first_predictor_predict_times:int = 10,
    ):
        self.environment = Mat2MatEnv(size, from_channel, to_channel)
        self.mat_H = H(size,from_channel, embedding_size * from_channel)
        self.mat_invH = invH(embedding_size * from_channel, size, from_channel)
        self.target_H = H(size, to_channel, embedding_size * to_channel)
        self.target_invH = invH(embedding_size * to_channel, size, to_channel)
        
        self.first_predictor = nn.Transformer(embedding_size)
        self.first_predictor_predict_times = first_predictor_predict_times
        
    def step(self):
        now_mat_state = self.mat_H(self.environment.mat)
        predict_mat_sequence = self.mat_H(self.environment.transformed_target_mat)
        first_predictor_output = now_mat_state
        for _ in range(self.first_predictor_predict_times):
            first_predictor_output = self.first_predictor(
                first_predictor_output,
                predict_mat_sequence,
            )[-1,:]
            predict_mat_sequence = torch.cat([predict_mat_sequence, first_predictor_output], dim=0)
        
    
    def state_value(self, state, target_state=None):
        if target_state is None:
            return F.l1_loss(self.mat_invH(state), self.environment.transformed_target_mat)
