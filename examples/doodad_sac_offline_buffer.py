import os
import copy
import uuid
import numpy as np
import pickle
from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.mujoco import SwimmerEnv
from gym.envs.mujoco import AntEnv
from gym.envs.mujoco import HopperEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.doodad_helper import *
from rlkit import hyper_sweep

EXPERT_POLICY_DIR = '/buffers'

def main(net_size=300, repeat=0, env_name='cheetah', exploration_policy_file=None, 
        buffer_name=None, **algo_params):
    if env_name == 'cheetah':
        env = NormalizedBoxEnv(HalfCheetahEnv())
    elif env_name == 'hopper':
        env = NormalizedBoxEnv(HopperEnv())
    else:
        raise NotImplementedError()
    #elif env == 'ant':
    #    env = NormalizedBoxEnv(AntEnv())
    #else:
    #    env = NormalizedBoxEnv(SwimmerEnv())
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    default_params = dict(
        num_epochs=2000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        batch_size=128,
        max_path_length=999,
        discount=0.99,
        reward_scale=1,

        soft_target_tau=0.001,
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,

        use_automatic_entropy_tuning=True,
        #exploration_policy_type='expert'

        replay_buffer_file='/buffers/buffer.npz',
        collection_mode='offline',
    )
    default_params.update(algo_params)

    # load the expert policy
    #with open(os.path.join(EXPERT_POLICY_DIR, exploration_policy_file), 'rb') as f:
    #    data = pickle.load(f)
    #expert_exploration_policy = data['exploration_policy']

    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        #expert_exploration_policy=expert_exploration_policy,
        qf=qf,
        vf=vf,
        **default_params
    )
    algorithm.to(ptu.device)

    # log params
    variant = copy.copy(default_params)
    exp_uuid = str(uuid.uuid4())
    variant['uuid'] = exp_uuid
    variant['net_size'] = net_size
    variant['env_name'] = env_name
    variant['buffer_name'] = buffer_name
    setup_logger('debug_doodad', variant=variant,
            log_dir='/data/%s'%exp_uuid)

    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = dict(
        net_size=[300],
        repeat=range(20),
        env_name=['cheetah'],
        num_replay_samples=[1000, 10000, 100000, 1000000]
    )
    buffer_name = 'buffers/sac_halfcheetah/sac_expert_1e6'
    args['buffer_name'] = [buffer_name]

    import doodad
    policy_data_mount = doodad.mount.MountLocal(local_dir=buffer_name, mount_point=EXPERT_POLICY_DIR)
    mounts = [policy_data_mount]
    #hyper_sweep.run_sweep_parallel(main, args, repeat=3)
    #hyper_sweep.run_sweep_serial(main, args, repeat=1)
    #SWEEPER_EAST1.run_test_docker(main, args, extra_mounts=mounts)

    SWEEPER_WEST1.run_sweep_gcp_chunked(main, args, 80, instance_type='n1-standard-2', s3_log_name='sac_cheetah_expert_offpol_fixed', region='us-west1-a', preemptible=True, extra_mounts=mounts)
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 120, instance_type='n1-standard-4', s3_log_name='badness_cheetah_sac', region='us-east1-b')
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 69, instance_type='n1-standard-4', s3_log_name='exact_weighting_adversarial', region='us-east1-b')
    #SWEEPER_EAST1.run_sweep_gcp_chunked(main, args, 69, instance_type='n1-standard-4', s3_log_name='exact_weighting_stateaction', region='us-east1-b')
    #SWEEPER_CENTRAL1.run_sweep_gcp_chunked(main, args, 124, instance_type='n1-standard-4', s3_log_name='newenv_exact_weighting_highent_fixed', region='us-central1-b')

