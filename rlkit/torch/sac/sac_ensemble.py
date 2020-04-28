import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class SacEnsemble(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploitation_policy,
            exploration_policy,
            qf_class,
            qf_kwargs,
            vf_class,
            vf_kwargs,

            ensemble_size=2,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=True,
            soft_target_tau=1e-2,
            policy_update_period=1,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(exploitation_policy)
        else:
            eval_policy = exploitation_policy
        super().__init__(
            env=env,
            exploration_policy=exploration_policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.exploitation_policy = exploitation_policy
        self.qfs = [qf_class(**qf_kwargs) for _ in range(ensemble_size)]
        #self.vfs = [vf_class(**vf_kwargs) for _ in range(ensemble_size)]
        self.vf = vf_class(**vf_kwargs)
        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = self.vf.copy()
        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()

        self.exploitation_policy_optimizer = optimizer_class(
            self.exploitation_policy.parameters(),
            lr=policy_lr,
        )
        self.exploration_policy_optimizer = optimizer_class(
            self.exploration_policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizers = [optimizer_class(qf.parameters(), lr=qf_lr) for qf in self.qfs]
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_preds = [qf(obs, actions) for qf in self.qfs]
        #v_preds = [vf(obs) for vf in self.vfs]
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.exploitation_policy(obs,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        Alpha Loss (if applicable)
        """
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        qf_losses = [self.qf_criterion(q_pred, q_target.detach()) for q_pred in q_preds]
        #qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions_conservative = torch.min(
            *[qf(obs, new_actions) for qf in self.qfs]
        )
        v_target = q_new_actions_conservative - alpha*log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        for qf_loss, qf_optimizer in zip(qf_losses, self.qf_optimizers):
            qf_optimizer.zero_grad()
            qf_loss.backward()
            qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Exploitation Policy Loss
            """
            if self.train_policy_with_reparameterization:
                exploit_policy_loss = (alpha.detach()*log_pi - q_new_actions_conservative).mean()
            else:
                log_policy_target = q_new_actions_conservative - v_pred
                exploit_policy_loss = (
                    log_pi * (alpha.detach()*log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            exploitation_policy_loss = exploit_policy_loss + policy_reg_loss

            self.exploitation_policy_optimizer.zero_grad()
            exploitation_policy_loss.backward()
            self.exploitation_policy_optimizer.step()

            """
            Exploration Policy Loss
            """
            exploration_policy_outputs = self.exploration_policy(obs,
                                         reparameterize=self.train_policy_with_reparameterization,
                                         return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = exploration_policy_outputs[:4]
            q_new_actions_aggressive = torch.max(
                *[qf(obs, new_actions) for qf in self.qfs]
            )
            if self.train_policy_with_reparameterization:
                explore_policy_loss = (alpha.detach()*log_pi - q_new_actions_aggressive).mean()
            else:
                log_policy_target = q_new_actions_aggressive - v_pred
                explore_policy_loss = (
                    log_pi * (alpha.detach()*log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            #policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            exploration_policy_loss = explore_policy_loss #+ policy_reg_loss
            self.exploration_policy_optimizer.zero_grad()
            exploration_policy_loss.backward()
            self.exploration_policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            #self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
            #    policy_loss
            #))
            for i, (qf_loss, q_pred) in enumerate(zip(qf_losses, q_preds)):
                self.eval_statistics['QF%d Loss'%i] = np.mean(ptu.get_numpy(qf_loss))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q%d Predictions'%i,
                    ptu.get_numpy(q_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    @property
    def networks(self):
        networks = [
            self.exploration_policy,
            self.exploitation_policy,
            self.vf,
            self.target_vf,
        ]
        networks.extend(self.qfs)
        return networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        for i, qf in enumerate(self.qfs):
            snapshot['qf%d'%i] = qf
        snapshot['policy'] = self.exploitation_policy
        snapshot['exploration_policy'] = self.exploration_policy
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        return snapshot
