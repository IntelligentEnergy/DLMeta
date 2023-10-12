'''
Author: Radillus
Date: 2023-05-31 18:38:45
LastEditors: Radillus
LastEditTime: 2023-05-31 19:25:39
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import ray
import gymnasium.envs


@ray.remote
class RemoteEnv:
    def __init__():
        pass
    def book(self) -> int:
        pass
    def step(self, action: int) -> tuple:
        pass
    