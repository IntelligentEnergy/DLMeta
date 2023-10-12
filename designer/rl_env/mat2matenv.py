'''
Author: Radillus
Date: 2023-07-02 00:54:45
LastEditors: Radillus
LastEditTime: 2023-07-02 11:20:47
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.TopoSampler

class Mat2MatEnv:
    def __init__(self, size, from_channel, to_channel):
        self.size = size
        self.from_channel = from_channel
        self.to_channel = to_channel
        self.samplers = [tools.TopoSampler.TopoSampler(size) for _ in range(from_channel)]
        self.spliter = nn.Linear(from_channel, to_channel)
        for param in self.spliter.parameters():
            param.requires_grad = False
        self.reset()
    def _transform(self, src:torch.Tensor):
        out = src.reshape((self.size, self.size, self.from_channel))
        out = self.spliter(out)
        out = out.reshape((self.to_channel, self.size, self.size,))
        out = torch.matmul(out, out)
        return out
    def reset(self):
        self.count = 0
        self.target_mat = [torch.from_numpy(samplers.sample()) for samplers in self.samplers]
        self.target_mat = torch.stack(self.target_mat)
        self.transformed_target_mat = self._transform(self.target_mat)
        self.mat = torch.zeros((self.channel, self.size, self.size))
    def step(self, action:torch.Tensor):
        self.mat = action
        self.transformed_mat = self._transform(self.mat)
        self.loss = F.l1_loss(self.transformed_mat, self.transformed_target_mat)
        self.count += 1
        if self.count >= 100:
            self.reset()
