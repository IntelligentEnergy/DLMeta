'''
Author: Radillus
Date: 2023-06-07 20:28:20
LastEditors: Radillus
LastEditTime: 2023-06-11 00:02:58
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import numpy as np

import structures


class Node:
    def __init__(
        self,
        state: structures.State,
        depth: int = 0,
        parent = None,
    ):
        self.state = state
        self.depth = depth
        self.children = []
        self.actions = []
        self.parent = parent
        self.max_value = self.state.value
    
    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None
    
    @property
    def choosen_value(self):
        return self.max_value * 0.1 + (self.max_value - self.state.value) - self.depth * 0.3 - 0.1 * len(self.children)
    
    def add_child(self, child_state:structures.State, action:structures.Action):
        self.children.append(Node(child_state, depth=self.depth + 1, parent=self))
        self.actions.append(action)
        self.max_value = max(self.max_value, child_state.value)
        parent = self.parent
        while not parent.is_root:
            parent.max_value = max(parent.max_value, child_state.value)
            parent = parent.parent

    def choose(self, remain_hp=1.0):
        if self.is_leaf or remain_hp < 0:
            return self
        else:
            if np.random.random() < 0.1:
                return self.children[np.random.randint(len(self.children))].choose(remain_hp)
            else:
                choose_probability = np.array([child.choosen_value for child in self.children])
                damage = -choose_probability
                choose_probability = (choose_probability - np.min(choose_probability)) / np.sum(choose_probability)
                dice = np.random.random()
                for i in range(len(self.children)):
                    if dice <= choose_probability[i]:
                        return self.children[i].choose(remain_hp + damage[i])
                    else:
                        dice -= choose_probability[i]
                return self
    
    def to_next(self):
        values = np.array([child.state.value for child in self.children])
        choose_idx = np.argmax(values)
        return self.children[choose_idx].state, self.actions[choose_idx]
