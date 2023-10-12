'''
Author: Radillus
Date: 2023-07-29 17:13:38
LastEditors: Radillus
LastEditTime: 2023-08-17 19:01:07
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
from typing import Any, Optional, NamedTuple, List

import numpy as np
import torch

import models
import muzero_config
import muzero_type


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MCTNodeTransition(NamedTuple):
    """
    Hold by a parent node
    parent node's observation --- action --> child node's observation
    """
    action: muzero_type.Action
    node: 'MCTNode'

class MCTNode:
    def __init__(
        self,
        parent: 'MCTNode' = None,
        prior: Optional[float] = None,
    ):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: List[MCTNodeTransition] = []
        self.hidden_state = None
        self.reward = 0

        self.mu = None
        self.std = None

        self.parent = parent

    @property
    def expanded(self):
        return self.parent is not None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        reward:torch.Tensor,
        mu:torch.Tensor,
        log_std:torch.Tensor,
        hidden_state:torch.Tensor,
    ):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.reward = reward
        self.hidden_state = hidden_state
        self.mu = mu
        self.std = torch.exp(log_std)

        policy = torch.distributions.normal.Normal(self.mu, self.std)
        action = policy.sample().squeeze(0).detach().cpu()

        self.children.append(MCTNodeTransition(action, MCTNode(self)))

    # def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
    #     """
    #     At the start of each search, we add dirichlet noise to the prior of the root to
    #     encourage the search to explore new actions.
    #     """
    #     actions = list(self.children.keys())
    #     noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    #     frac = exploration_fraction
    #     for a, n in zip(actions, noise):
    #         self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


def _select_child(node: MCTNode, min_max_stats: MinMaxStats):
    """
    Select the child with the highest UCB score.
    """
    # Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
    if len(node.children) < (node.visit_count + 1) ** muzero_config.pw_alpha:
        distribution = torch.distributions.normal.Normal(node.mu, node.std)
        action = distribution.sample().squeeze(0).detach().cpu()
        
        children_actions = [child.action for child in node.children]
        while action in children_actions:
            action = distribution.sample().squeeze(0).detach().cpu()

        node.children.append(MCTNodeTransition(action, MCTNode(node)))
        return action, node.children[action]

    else:
        max_ucb = max(
            _ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if _ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
    return action, node.children[action]


def _ucb_score(parent: MCTNode, child: MCTNode, min_max_stats: MinMaxStats):
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    """
    pb_c = (
        np.log(
            (parent.visit_count + muzero_config.pb_c_base + 1) / muzero_config.pb_c_base
        )
        + muzero_config.pb_c_init
    )
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    # Uniform prior for continuous action space
    if muzero_config.node_prior == "uniform":
        prior_score = pb_c * (1 / len(parent.children))
    elif muzero_config.node_prior == "density":
        prior_score = pb_c * (
            child.prior / sum([child.prior for child in parent.children.values()])
        )
    else:
        raise ValueError("{} is unknown prior option, choose uniform or density")

    if child.visit_count > 0:
        # Mean value Q
        value_score = min_max_stats.normalize(
            child.reward
            + muzero_config.discount * child.value()
        )
    else:
        value_score = 0

    return prior_score + value_score


def backpropagate(search_path: list[MCTNode], value, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.reward + muzero_config.discount * node.value())

        value = node.reward + muzero_config.discount * value


# Game independent
def MCTS(model: models.AbstractNetwork, observation: np.ndarray):
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    """
    At the root of the search tree we use the representation function to obtain a
    hidden state given the current observation.
    We then run a Monte Carlo Tree Search using only action sequences and the model
    learned by the network.
    """

    root = MCTNode()
    observation = (
        torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
    )
    (
        root_predicted_value,
        reward,
        mu,
        log_std,
        hidden_state,
    ) = model.initial_inference(observation)
    root_predicted_value = models.support_to_scalar(
        root_predicted_value, muzero_config.support_size
    ).item()
    reward = models.support_to_scalar(reward, muzero_config.support_size).item()
    root.expand(
        reward,
        mu,
        log_std,
        hidden_state,
    )

    min_max_stats = MinMaxStats()

    max_tree_depth = 0
    for _ in range(muzero_config.num_simulations):
        node = root
        search_path = [node]
        current_tree_depth = 0

        while node.expanded():
            current_tree_depth += 1
            action, node = _select_child(node, min_max_stats)
            search_path.append(node)


        # Inside the search tree we use the dynamics function to obtain the next hidden
        # state given an action and the previous hidden state
        parent = search_path[-2]
        value, reward, mu, log_std, hidden_state = model.recurrent_inference(
            parent.hidden_state,
            torch.tensor(np.array([action.value]), dtype=torch.float32).to(
                parent.hidden_state.device
            ),
        )
        value = models.support_to_scalar(value, muzero_config.support_size).item()
        reward = models.support_to_scalar(reward, muzero_config.support_size).item()
        node.expand(
            reward,
            mu,
            log_std,
            hidden_state,
        )

        backpropagate(search_path, value, min_max_stats)

        max_tree_depth = max(max_tree_depth, current_tree_depth)

    extra_info = {
        "max_tree_depth": max_tree_depth,
        "root_predicted_value": root_predicted_value,
    }
    return root, extra_info
