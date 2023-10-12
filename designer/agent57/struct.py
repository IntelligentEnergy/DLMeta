'''
Author: Radillus
Date: 2023-05-31 16:43:02
LastEditors: Radillus
LastEditTime: 2023-05-31 17:15:20
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Mapping, Iterable, Optional, Any, NamedTuple, Tuple

import numpy as np
import torch


class TimeStep(NamedTuple):
    """Environment timestep"""
    observation: Optional[np.ndarray]
    reward: Optional[float]  # reward couble be clipped or scaled
    raw_reward: Optional[float]  # non-clipped/unscaled reward, only used by the trackers
    done: Optional[bool]  # This may not be the real episode termination, since Atari we often use done-on-loss-life
    first: Optional[bool]  # first step of an episode

class Agent57Transition(NamedTuple):
    """
    done is True if the truncated OR terminated
    last_action is the last agent the agent took, before in s_t.
    """
    observation: np.ndarray
    action: np.ndarray
    done: bool
    q: Optional[np.ndarray]  # q values for s_t, computed from both ext_q_network and int_q_network
    prob_action: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    extrinsic_reward: Optional[float]  # extrinsic reward for (s_tm1, a_tm1)
    intrinsic_reward: Optional[float]  # intrinsic reward for (s_tm1)
    policy_index: Optional[int]  # intrinsic reward scale beta index
    beta: Optional[float]  # intrinsic reward scale beta value
    discount: Optional[float]
    extrinsic_init_memory: Optional[Tuple[torch.Tensor,torch.Tensor]]  # nn.LSTM initial (h,c), from extrinsic_q_network
    intrinsic_init_memory: Optional[Tuple[torch.Tensor,torch.Tensor]]  # nn.LSTM initial (h,c), from intrinsic_q_network
