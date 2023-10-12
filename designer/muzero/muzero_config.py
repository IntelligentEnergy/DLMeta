'''
Author: Radillus
Date: 2023-07-29 15:16:54
LastEditors: Radillus
LastEditTime: 2023-08-02 20:01:57
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Union, Optional

import numpy as np
from gymnasium import Env


################################################ GENERA CONFIG START ##############################################
# muzero will store replaybuffer and model under this path
run_path:str = './data/muzero/'
model_path:str = run_path + 'model/model.pth'
def get_train_env() -> Env:
    pass

def get_eval_env() -> Env:
    pass

discount = 0.997
batch_size = 8
################################################ GENERAL CONFIG END ################################################



############################################# REPLAYBUFFER CONFIG START ################################################
# priority replay buffer
use_PER:bool = True
buffer_size:int = 10000
replay_buffer_size:int = 10000
num_unroll_steps:int = 10
td_steps:int = 50
############################################# REPLAYBUFFER CONFIG END ################################################



############################################# MCTS CONFIG START ################################################
pw_alpha:float = 0.49
pb_c_base:int = 19652
pb_c_init:int = 1.25
# uniform or density
node_prior:str = 'uniform'
support_size:int = 21
num_simulations:int = 50
############################################# MCTS CONFIG END ################################################



################################################ ACTOR CONFIG START ##############################################
each_actor_sample_trial_num:int = 10000
temperature:Union[float, np.ndarray] = 1
temperature_threshold:Optional[float] = None
################################################ ACTOR CONFIG END ################################################
