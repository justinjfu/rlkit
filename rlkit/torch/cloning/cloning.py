from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class Cloning(TorchRLAlgorithm):
    """
    Behavioral Cloning
    """

    def __init__(
            self,
            env,
            policy,
            policy_lr=1e-3,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        super().__init__(
            env,
            policy,
            eval_policy=policy,
            collection_mode='offline',
            **kwargs
        )
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        #rewards = batch['rewards']
        #terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        #next_obs = batch['next_observations']

        # Soft policy
        policy_outputs = self.policy(obs)
        if isinstance(policy_outputs, tuple):
            policy_logprobs = self.policy.logprob(
                    obs,
                    actions
            )
            policy_loss = torch.mean(policy_logprobs)
            policy_actions, policy_mean = policy_outputs[0:2]
        else:
            policy_actions = policy_outputs
            policy_error = (policy_actions - actions) ** 2
            policy_loss = policy_error.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.policy,
        ]
