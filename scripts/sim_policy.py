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

filename = str(uuid.uuid4())

def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))

def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
        )
        import ipdb; ipdb.set_trace()
        save_video('./videos/' + 'walker' + '_d4rl', 'website', path['frames'], 1)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
