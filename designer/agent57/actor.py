'''
Author: Radillus
Date: 2023-05-31 16:43:25
LastEditors: Radillus
LastEditTime: 2023-06-09 15:29:02
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import numpy as np
import torch
import ray

from designer.network import RemoteNetwork, LocalNetwork
from designer.rl_env.remoteenv import RemoteEnv
from designer.buffer import RemoteBuffer
from designer.utils import no_autograd


@ray.remote
class Actors:
    def __init__(
        self,
        environment_handler:RemoteEnv,
        buffer_handler:RemoteBuffer,

        extrinsic_q_remote_network:RemoteNetwork,
        intrinsic_q_remote_network:RemoteNetwork,
        rnd_target_remote_network:RemoteNetwork,
        rnd_predictor_remote_network:RemoteNetwork,
        state_embedding_remote_network:RemoteNetwork,
        
        actor_num:int,
        num_policies: int,
        episodic_memory_capacity: int,
        actor_update_frequency: int,
        unroll_length:int,
        burn_in_length:int,
        num_neighbors: int,
        ucb_window_size: int,
        
        ucb_beta: float,
        extrinsic_discount:float,
        intrinsic_discount:float,
        ucb_epsilon: float,
        policy_beta: float,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
    ):
        self.environment_handler = environment_handler
        self.buffer_handler = buffer_handler
        
        self.extrinsic_q_remote_network = extrinsic_q_remote_network
        self.intrinsic_q_remote_network = intrinsic_q_remote_network
        self.rnd_target_remote_network = rnd_target_remote_network
        self.rnd_predictor_remote_network = rnd_predictor_remote_network
        self.state_embedding_remote_network = state_embedding_remote_network
        
        self.extrinsic_q_local_network = LocalNetwork(extrinsic_q_remote_network)
        self.intrinsic_q_local_network = LocalNetwork(intrinsic_q_remote_network)
        self.rnd_target_local_network = LocalNetwork(rnd_target_remote_network)
        self.rnd_predictor_local_network = LocalNetwork(rnd_predictor_remote_network)
        self.state_embedding_local_network = LocalNetwork(state_embedding_remote_network)
        
        no_autograd(self.extrinsic_q_local_network)
        no_autograd(self.intrinsic_q_local_network)
        no_autograd(self.rnd_target_local_network)
        no_autograd(self.rnd_predictor_local_network)
        no_autograd(self.state_embedding_local_network)
        
        self.actor_num = actor_num
        self.num_policies = num_policies
        self.episodic_memory_capacity = episodic_memory_capacity
        self.actor_update_frequency = actor_update_frequency
        self.unroll_length = unroll_length
        self.burn_in_length = burn_in_length
        self.num_neighbors = num_neighbors
        self.ucb_window_size = ucb_window_size
        
        self.ucb_beta = ucb_beta
        self.extrinsic_discount = extrinsic_discount
        self.intrinsic_discount = intrinsic_discount
        self.ucb_epsilon = ucb_epsilon
        self.policy_beta = policy_beta
        self.cluster_distance = cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.max_similarity = max_similarity
        
        self.envrioment_id:int = ray.get(self.environment_handler.book.remote())
        
        def step(self):
            pass
        def reset(self):
            pass
        
