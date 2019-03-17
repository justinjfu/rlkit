import abc
from typing import Iterable
import pickle
import time
from collections import OrderedDict
import torch

import gtimer as gt
import numpy as np

from rlkit.core.rl_algorithm import RLAlgorithm, set_to_train_mode, set_to_eval_mode
from rlkit.torch import pytorch_util as ptu
from rlkit.torch import torch_rl_algorithm 
from rlkit.torch.core import PyTorchModule
from rlkit.core import eval_util, logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch.mbpo import model_env


class ModelBased(torch_rl_algorithm.TorchRLAlgorithm):
    def __init__(self, *args, model_step_ratio=4,
            model=None, 
            model_train_steps=10000,
            **kwargs):
        super(ModelBased, self).__init__(*args, **kwargs)
        self.model = model
        tmp_env = pickle.loads(pickle.dumps(self.training_env))
        self.model_env = model_env.ModelEnv(tmp_env, model=model)
        self.model_step_ratio = model_step_ratio
        self.model_rollout_buffer = EnvReplayBuffer(
            self.num_env_steps_per_epoch * model_step_ratio,
            self.env,
        )
        self.real_eval_sampler = self.eval_sampler
        model_sampler = InPlacePathSampler(
            env=self.model_env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval + self.max_path_length,
            max_path_length=self.max_path_length,
        )
        self.model_eval_sampler = model_sampler
        self.model_train_steps = model_train_steps

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        self.model.to(device)

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        self._model_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')
            logger.record_tabular('replay_buffer_size', len(self.replay_buffer))

            self.train_model()

            set_to_train_mode(self.model_env)
            observation = self._start_new_rollout_model()
            for _ in range(self.num_env_steps_per_epoch * self.model_step_ratio):
                observation = self._take_step_in_model(observation)
                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')
            logger.record_tabular('model_buffer_size', len(self.model_rollout_buffer))

            set_to_eval_mode(self.env)
            set_to_eval_mode(self.model_env)

            self.eval_sampler = self.model_eval_sampler
            with logger.tabular_prefix('model_eval/'):
                self.evaluate(epoch)
            self.eval_sampler = self.real_eval_sampler
            with logger.tabular_prefix('real_eval/'):
                self.evaluate(epoch)
            self.need_to_update_eval_statistics = False #TODO: idk what this does but I need it to log
            self._try_to_eval(epoch, no_test=True)
            gt.stamp('eval')
            self._end_epoch(epoch)

    def train_model(self):
        if self.model is None:
            logger.record_tabular('begin_model_loss', 0)
            logger.record_tabular('final_model_loss', 0)
            return

        loss = torch.nn.modules.SmoothL1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for i in range(self.model_train_steps):
            batch = self.replay_buffer.random_batch(self.batch_size)
            pt_batch = torch_rl_algorithm.np_to_pytorch_batch(batch)
            #pred_nobs = self.model(pt_batch['observations'], pt_batch['actions'])
            #model_loss = loss(pt_batch['next_observations'], pred_nobs)
            model_loss = self.model.loss(pt_batch['observations'],
                                         pt_batch['actions'],
                                         pt_batch['next_observations'])

            self.model.zero_grad()
            model_loss.backward()
            optimizer.step()
            if i==0:
                logger.record_tabular('begin_model_loss', ptu.get_numpy(model_loss))
        logger.record_tabular('final_model_loss', ptu.get_numpy(model_loss))
        gt.stamp('train_model')

    def get_batch(self):
        batch = self.model_rollout_buffer.random_batch(self.batch_size)
        return torch_rl_algorithm.np_to_pytorch_batch(batch)

    def _take_step_in_model(self, observation):
        action, agent_info = self._get_action_and_info(
            observation,
        )
        #if self.render:
        #    self.model_env.render()
        next_ob, raw_reward, terminal, env_info = (
            self.model_env.step(action)
        )
        if np.any(np.isnan(next_ob)):
            # this happens when the model rollout blows up
            # and goes to infinity -- 
            # typically towards the end of a rollout
            import pdb; pdb.set_trace()
        reward = raw_reward * self.reward_scale
        terminal = np.array([terminal])
        reward = np.array([reward])
        self._handle_step_model(
            observation,
            action,
            reward,
            next_ob,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )
        if terminal or len(self._model_path_builder) >= self.max_path_length:
            self._handle_rollout_ending_model()
            new_observation = self._start_new_rollout_model()
        else:
            new_observation = next_ob
        return new_observation

    def _handle_step_model(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._model_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.model_rollout_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending_model(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.model_rollout_buffer.terminate_episode()
        if len(self._model_path_builder) > 0:
            self._model_path_builder = PathBuilder()

    def _start_new_rollout_model(self):
        self.exploration_policy.reset()
        return self.model_env.reset()


class ModelBasedv2(torch_rl_algorithm.TorchRLAlgorithm):
    # crank up num_updates_per_train_call
    def __init__(self, *args,
            model_step_ratio=4,
            num_updates_per_train_call=1,
            **kwargs):
        self.model_env = pickle.loads(pickle.dumps(self.training_env))
        self.model_step_ratio = model_step_ratio
        super(ModelBasedv2, self).__init__(*args, **kwargs, 
                num_updates_per_train_call=model_step_ratio*num_updates_per_train_call)
        self.model_rollout_buffer = EnvReplayBuffer(
            self.replay_buffer_size,
            self.env,
        )

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')

            self.train_model()

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)

    def train_model(self):
        pass

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        raise NotImplementedError("Take steps in model")
        return np_to_pytorch_batch(batch)

    def _take_step_in_model(self, observation):
        action, agent_info = self._get_action_and_info(
            observation,
        )
        if self.render:
            self.model_env.render()
        next_ob, raw_reward, terminal, env_info = (
            self.model_env.step(action)
        )
        reward = raw_reward * self.reward_scale
        terminal = np.array([terminal])
        reward = np.array([reward])
        if terminal or len(self._model_path_builder) >= self.max_path_length:
            new_observation = self.model_env.reset()
        else:
            new_observation = next_ob
        return action, reward, new_observation, terminal


