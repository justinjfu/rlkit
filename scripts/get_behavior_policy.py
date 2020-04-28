from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
from PIL import Image
import os
import numpy as np


def get_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)

    return env, policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    # obtaining the policy
    env, agent = get_policy(args)
    # import ipdb; ipdb.set_trace()

    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    returns = 0
    discounted_return = 0.0
    discount = 0.99
    
    while path_length < 1000:
        # This is how we get action out of the behavior policy
        a, agent_info = agent.get_action(o)
        if args.add_noise:
            a = (a + np.random.uniform(-0.02, 0.02)).clip(-0.96, 0.96)
        next_o, r, d, env_info = env.step(a)
        returns += r
        discounted_return += (np.power(discount, path_length) * r)
        
        path_length += 1
        if d:
            break
        o = next_o
    
    print ('Return: ', returns)
    print ('Discounted return: ', discounted_return)