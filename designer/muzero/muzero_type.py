'''
Author: Radillus
Date: 2023-08-02 19:01:47
LastEditors: Radillus
LastEditTime: 2023-08-11 17:08:38
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Any, Optional, Union, List, Dict, Sequence, NamedTuple
import numpy as np
import torch
from torchtyping import TensorType


# theses corresponding to gymnasium.spaces (box, discrete, multi_discrete, multi_binary), dict, sequence
# graph space is not supported now, maybe later
# tuple is not supported, use sequence or dict instead
Observation = Union[torch.Tensor, Dict[str, torch.Tensor], Sequence[torch.Tensor]]
Action = Union[torch.Tensor, Dict[str, torch.Tensor], Sequence[torch.Tensor]]
Reward = torch.Tensor

class Trial(NamedTuple):
    """A trial is a sequence of observations, actions, rewards
    we assume that take action[i] when observation[i] is observed and get reward[i] then observation to [i+1]
    so len(observation) is always 1 more than other
    """
    observations: List[Observation] = []
    actions: List[Action] = []
    rewards: List[Reward] = []
    
    # first layer child, turn MCT result into a policy
    flchild_actions: List[Action] = []
    flchild_visit_counts: List[int] = []
    root_values: List[float] = []
    reanalysed_predicted_root_values: Optional[List[float]] = None
    
    # For PER
    priorities: Optional[List[float]] = None
    game_priority: Optional[float] = None
