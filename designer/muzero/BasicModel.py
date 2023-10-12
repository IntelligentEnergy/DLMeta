'''
Author: Radillus
Date: 2023-07-27 16:52:35
LastEditors: Radillus
LastEditTime: 2023-07-29 15:11:47
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasicModel(ABC, nn.Module):
    @abstractmethod
    def initial_inference(self, observation: torch.Tensor):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state: torch.Tensor, action: torch.Tensor):
        pass
