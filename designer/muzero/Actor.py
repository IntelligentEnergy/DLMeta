'''
Author: Radillus
Date: 2023-07-27 15:16:04
LastEditors: Radillus
LastEditTime: 2023-08-12 17:00:45
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env

from replay_buffer import ReplayBuffer
from MCTS import MCTS, MCTNode
import models
import muzero_config
from muzero_type import Trial


class Actor:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, Game, config, seed):
        self.config = config
        self.game:Env = Game(seed)

        # Initialize the network
        self.model:models.AbstractNetwork = models.MuZeroNetwork()
        self._load_model()
        

    def run(self, replay_buffer: ReplayBuffer, eval: bool = False):
        for sample_trail_count in range(muzero_config.each_actor_sample_trial_num):
            self._load_model()

            if not eval:
                trail = self._sample_a_trial(
                    temperature=muzero_config.act_temperature,
                    temperature_threshold=muzero_config.act_temperature_threshold,
                    render=False,
                    opponent="self",
                    muzero_player=0,
                )

                replay_buffer.save_game(trail)

            else:
                #TODO
                raise NotImplementedError
                # # Take the best action (no exploration) in test mode
                # game_history = self.play_game(
                #     0,
                #     self.config.temperature_threshold,
                #     False,
                #     "self" if len(self.config.players) == 1 else self.config.opponent,
                #     self.config.muzero_player,
                # )

                # # Save to the shared storage
                # shared_storage.set_info.remote(
                #     {
                #         "episode_length": len(game_history.action_history) - 1,
                #         "total_reward": sum(game_history.reward_history),
                #         "mean_value": np.mean(
                #             [value for value in game_history.root_values if value]
                #         ),
                #     }
                # )
                # if 1 < len(self.config.players):
                #     shared_storage.set_info.remote(
                #         {
                #             "muzero_reward": sum(
                #                 reward
                #                 for i, reward in enumerate(game_history.reward_history)
                #                 if game_history.to_play_history[i - 1]
                #                 == self.config.muzero_player
                #             ),
                #             "opponent_reward": sum(
                #                 reward
                #                 for i, reward in enumerate(game_history.reward_history)
                #                 if game_history.to_play_history[i - 1]
                #                 != self.config.muzero_player
                #             ),
                #         }
                #     )

        self.game.close()

    @torch.no_grad()
    def _sample_a_trial(self, temperature: float):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        we assume that take action[i] when observation[i] is observed and get reward[i] then observation to [i+1]
        """
        trial = Trial()

        observation, info = self.game.reset()
        trial.observation_history.append(observation)

        terminated, truncated = False, False

        while not (terminated or truncated):
            stacked_observations = trial.get_stacked_observations(
                -1, self.config.stacked_observations
            )

            root, mcts_info = MCTS(self.model, stacked_observations)
            action = self._select_action(root, temperature)

            observation, reward, terminated, truncated, info = self.game.step(action.value)
            trial.store_search_statistics(root)


            trial.action_history.append(action.value) # a[t]
            trial.reward_history.append(reward) # r[t]

            trial.observation_history.append(observation) # o[t+1]

        return trial

    def _load_model(self):
        self.model.load_state_dict(torch.load(muzero_config.model_path))
        self.model = self.model.eval()
        for params in self.model.parameters():
            params.requires_grad_(False)

    @staticmethod
    def _select_action(node: MCTNode, temperature: float):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype=int
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    @staticmethod
    def _store_MCTS_statistics(trial: Trial, MCT_root: MCTNode):
        """
        Store the MCTS statistics of the last search.
        """
        for action, child in MCT_root.children.items():
            trial.child_actions.append(action)
            trial.child_visits.append(child.visit_count)


# class Trial:
#     """
#     Store only usefull information of a self-play game.
#     """

#     def __init__(self):
#         self.observation_history = []
#         self.action_history = []
#         self.reward_history = []
#         self.child_actions = []
#         self.child_visits = []
#         self.root_values = []
#         self.reanalysed_predicted_root_values = None
#         # For PER
#         self.priorities = None
#         self.game_priority = None

#     def store_search_statistics(self, root:MCTNode):
#         # Turn visit count from root into a policy
#         sum_visits = sum(child.visit_count for child in root.children.values())
#         self.child_visits.append(
#             np.array([child.visit_count for child in root.children.values()])
#         )

#         self.root_values.append(root.value())
#         self.child_actions.append(
#             np.array([action.value for action in root.children.keys()])
#         )


#     def get_stacked_observations(self, index, num_stacked_observations):
#         """
#         Generate a new observation with the observation at the index position
#         and num_stacked_observations past observations and actions stacked.
#         """
#         # Convert to positive index
#         index = index % len(self.observation_history)

#         stacked_observations = self.observation_history[index].copy()
#         for past_observation_index in reversed(
#             range(index - num_stacked_observations, index)
#         ):
#             if 0 <= past_observation_index:
#                 previous_observation = np.concatenate(
#                     (
#                         self.observation_history[past_observation_index],
#                         [
#                             np.ones_like(stacked_observations[0])
#                             * self.action_history[past_observation_index + 1]
#                         ],
#                     )
#                 )
#             else:
#                 previous_observation = np.concatenate(
#                     (
#                         np.zeros_like(self.observation_history[index]),
#                         [np.zeros_like(stacked_observations[0])],
#                     )
#                 )

#             stacked_observations = np.concatenate(
#                 (stacked_observations, previous_observation)
#             )

#         return stacked_observations
