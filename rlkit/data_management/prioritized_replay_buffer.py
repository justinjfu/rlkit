import numpy as np
import random
import tqdm
from gym.spaces import Discrete

from rlkit.data_management import replay_buffer, env_replay_buffer
from rlkit.data_management.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(replay_buffer.ReplayBuffer):
    """
    A class used to save and replay data.
    """
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
            alpha=0.6,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._maxsize = int(max_replay_buffer_size)
        self._storage = []
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        idx = self._next_idx
        self.super_add(observation, action, reward, terminal, next_observation, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def super_add(self, obs_t, action, reward, done, obs_tp1, **kwargs):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def random_batch(self, batch_size, beta=0.5):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        #assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return dict(
            observations=encoded_sample[0],
            actions=encoded_sample[1],
            rewards=encoded_sample[2],
            next_observations=encoded_sample[3],
            terminals=encoded_sample[4],
            weights=weights,
            indices=np.array(idxes),
        )

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def terminate_episode(self):
        pass

    def __len__(self):
        return len(self._storage)

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        return len(self)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities, eps=1e-6):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        eps: [float]
            Small constant value to add to priorities
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority += eps
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def load_from(self, filename):
        npz_data = np.load(filename)
        self._observation_dim = npz_data['observation_dim']
        self._action_dim = npz_data['action_dim']
        self._maxsize = npz_data['max_replay_buffer_size']

        observations = npz_data['observations']
        next_obs = npz_data['next_obs']
        actions = npz_data['actions']
        rewards = npz_data['rewards']
        terminals = npz_data['terminals']
        for i in tqdm.tqdm(range(observations.shape[0])):
            self.add_sample(observations[i],
                    actions[i],
                    rewards[i],
                    terminals[i],
                    next_obs[i])


class PrioritizedEnvReplayBuffer(PrioritizedReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            **kwargs
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=env_replay_buffer.get_dim(self._ob_space),
            action_dim=env_replay_buffer.get_dim(self._action_space),
            **kwargs
        )

    def add_sample(self, observation, action, reward, terminal,
            next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        super(PrioritizedEnvReplayBuffer, self).add_sample(
                observation, action, reward, terminal, 
                next_observation, **kwargs)


