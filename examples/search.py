"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

import heapdict

class Path(object):
    def __init__(self):
        self.states = []

def search():
    heap = heapdict.heapdict()
    visited = set()
    key = (state, 0, None)
    heap[key] = 0

    while len(heap)>0:
        (state, path), priority = heap.popitem()
        if state in visited:
            continue
        visited.add(state)

        #
        #cur_heuristic = heuristic(state, goal)
        for succ, succ_action in succs:
            priority = 0
            key = (succ, new_path)
            heap[key] = priority

    return best_path

def experiment(variant):
    env = NormalizedBoxEnv(HalfCheetahEnv())


if __name__ == "__main__":
    # noinspection PyTypeChecker
    experiment({})
