'''
Author: Radillus
Date: 2023-08-04 11:50:04
LastEditors: Radillus
LastEditTime: 2023-08-04 12:14:29
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Any

import numpy as np
import gymnasium
import torch


def turn_raw_observation_to_tensor_observation(observation: Any) -> any:
    """ this will pass to gymnasium.wrappers.TransformObservation
    we did not constrain return type, you may want use composite tensor in your nn
    for example, obs is a dict contain two tensors with different shape,
    they cannot be concatenated unless you pad one of them or flatten them
    """
    pass

def turn_raw_action_to_tensor_action(action: Any) -> torch.Tensor:
    pass

def turn_tensor_action_to_raw_action(tensor_action: torch.Tensor) -> Any:
    pass
