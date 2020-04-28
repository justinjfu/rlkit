import numpy as np
import os

from rlkit.data_management import simple_replay_buffer
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from scipy.spatial.distance import pdist, squareform
import torch

def main():
    data_dir = 'data/sac_halfcheetah/itr250_1e7'
    filename = os.path.join(data_dir, 'buffer.npz')
    #env = NormalizedBoxEnv(HalfCheetahEnv())
    replay = simple_replay_buffer.SimpleReplayBuffer(1,1,1)
    replay.load_from(filename)

    device = torch.device('cuda:0')

    # load
    obs = replay._observations
    obs = torch.tensor(obs, device=device)
    
    #obs = obs[:500]
    N = obs.shape[0]
    with open(os.path.join(data_dir, 'min_dists.txt'), 'w') as f:
        for i in tqdm.tqdm(range(N)):
            dists = torch.abs(obs - obs[i].unsqueeze(0))
            dists = torch.sum(dists, dim=1)
            dists[i] = float('inf')
            f.write('%f\n' % torch.min(dists))







if __name__ == "__main__":
    main()
