'''
Author: Radillus
Date: 2023-07-27 15:16:04
LastEditors: Radillus
LastEditTime: 2023-08-03 23:45:51
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import copy
import time

import numpy
import ray
import torch

import models
import muzero_config


class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )


    def save_game(self, game_history, shared_storage=None):
        if muzero_config.use_PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** muzero_config.use_PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if muzero_config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_action_batch,
            policy_visit_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [])
        weight_batch = [] if muzero_config.use_PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
            muzero_config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policy_actions, policy_visits, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_action_batch.append(policy_actions)
            policy_visit_batch.append(policy_visits)
            gradient_scale_batch.append(
                [
                    min(
                        muzero_config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if muzero_config.use_PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if muzero_config.use_PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_action_batch,
                policy_visit_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if muzero_config.use_PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if muzero_config.use_PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if muzero_config.use_PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if muzero_config.use_PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + muzero_config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * muzero_config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        (
            target_values,
            target_rewards,
            target_policy_actions,
            target_policy_visits,
            actions,
        ) = ([], [], [], [], [])
        for current_index in range(
            state_index, state_index + muzero_config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policy_actions.append(game_history.child_actions[current_index])
                target_policy_visits.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                target_policy_actions.append(
                    2 * numpy.random.uniform(0, 1, target_policy_actions[-1].shape) - 1
                )
                target_policy_visits.append(numpy.ones_like(target_policy_visits[-1]))
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy to give the tensor a valid dimension
                target_policy_actions.append(
                    2 * numpy.random.uniform(0, 1, target_policy_actions[-1].shape) - 1
                )
                target_policy_visits.append(
                    numpy.ones_like(target_policy_visits[-1])
                    * (muzero_config.num_simulations // len(target_policy_visits[-1]))
                )
                idx = numpy.random.choice(len(game_history.action_history))
                actions.append(game_history.action_history[idx])

        return (
            target_values,
            target_rewards,
            target_policy_actions,
            target_policy_visits,
            actions,
        )
