"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
from rlkit.envs.mbpo.halfcheetah import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

import heapdict

class Path(object):
    def __init__(self, init_state=None):
        self.states = []
        if init_state:
            self.states.append(init_state)
        self.returns = 0

    def extend(self, action, next_state, reward):
        new_path = Path()
        new_path.returns = reward + self.returns
        new_path.states = self.states + [next_state]
        return new_path

def aggregate(state, granularity=10):
    # the higher the granularity, the more fine the discretization
    rounded = np.floor(state * granularity).astype(np.int32)
    return tuple(rounded[1:])

def search(env):
    heap = heapdict.heapdict()
    visited = set()

    start_state = env.sim_reset()
    key = (state, Path(state))
    heap[key] = 0

    for _ in range(10000):
        (state, path), priority = heap.popitem()

        agg_state = aggregate(state)
        if agg_state in visited:
            raise NotImplementedError("need to update paths??")
        visited.add(agg_state)

        # sample actions
        sampled_actions = [env.action_space.sample() for _ in range(5)]
        for action in sampled_actions:
            next_state = env.sim_transition(state, action)
            reward = env.sim_reward(state, action, next_state)
            new_path = path.extend(action, next_state, reward)

            priority = -new_path.returns
            key = (succ, new_path)
            heap[key] = priority

    return best_path

def experiment(variant):
    env = NormalizedBoxEnv(HalfCheetahEnv())


if __name__ == "__main__":
    # noinspection PyTypeChecker
    experiment({})
