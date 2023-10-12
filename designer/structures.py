'''
Author: Radillus
Date: 2023-06-09 13:51:14
LastEditors: Radillus
LastEditTime: 2023-06-24 19:57:04
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Optional

import numpy as np
import torch


class State:
    def __init__(
        self,
        surface: np.ndarray,
        field: np.ndarray,
    ):
        assert surface.ndim == 2 and surface.shape[0] == surface.shape[1]
        assert field.ndim == 3 and field.shape[1] == field.shape[2] == surface.shape[0]
        self.surface = surface
        self.field = field

    @property
    def size(self):
        return self.surface.shape[0]

    @property
    def field_channel(self):
        return self.field.shape[0]


class Replay:
    def __init__(
        self,
        target_field:np.ndarray,
    ):
        self.target = target_field
        self.play_list:list[State] = []
        self.score_list:list[float] = []

    def add(self, state:State):
        self.play_list.append(state)
        self.score_list.append(1 - np.mean(np.abs(state.surface - self.target)))
