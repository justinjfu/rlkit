import copy
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac_ensemble import SacEnsemble
from rlkit.torch.networks import RandomizedFlattenMlp

from rlkit.doodad_helper import *
from rlkit import hyper_sweep

def main(net_size=300, repeat=0, **algo_params):
    env = NormalizedBoxEnv(HalfCheetahEnv())
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    qf_class = RandomizedFlattenMlp
    qf_args = dict(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf_class = RandomizedFlattenMlp
    vf_args = dict(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    exploitation_policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    exploration_policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    default_params = dict(
        num_epochs=1000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        batch_size=128,
        max_path_length=999,
        discount=0.99,
        reward_scale=1,

        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
    )
    default_params.update(algo_params)
    algorithm = SacEnsemble(
        env=env,
        exploitation_policy=exploitation_policy,
        exploration_policy=exploration_policy,
        qf_class=qf_class,
        qf_kwargs=qf_args,
        vf_class=vf_class,
        vf_kwargs=vf_args,
        **default_params,
    )
    algorithm.to(ptu.device)

    # log params
    variant = copy.copy(default_params)
    variant['net_size'] = net_size
    setup_logger('debug_doodad', variant=variant)

    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = dict(
        net_size=[400],
        repeat=range(5),
        ensemble_size=[2,4,1],
        soft_target_tau=[0.1, 0.01, 0.001],
        reward_scale=[10],
        num_updates_per_env_step=[1,2,4],
    )

    #hyper_sweep.run_sweep_parallel(main, args, repeat=3)
    #hyper_sweep.run_sweep_serial(main, args, repeat=1)
    #SWEEPER_WEST1.run_single_gcp(main, args)
    SWEEPER_EAST1.run_test_docker(main, args)

    #SWEEPER_WEST1.run_sweep_gcp_chunked(main, args, 120, instance_type='n1-standard-4', s3_log_name='exact_weighting_distr_shift', region='us-west1-a')
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 69, instance_type='n1-standard-4', s3_log_name='exact_weighting_fqi_2', region='us-east1-b')
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 69, instance_type='n1-standard-4', s3_log_name='exact_weighting_adversarial', region='us-east1-b')
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 69, instance_type='n1-standard-4', s3_log_name='exact_weighting_stateaction', region='us-east1-b')
    #SWEEPER_CENTRAL1.run_sweep_gcp_chunked(main, args, 124, instance_type='n1-standard-4', s3_log_name='newenv_exact_weighting_highent_fixed', region='us-central1-b')

