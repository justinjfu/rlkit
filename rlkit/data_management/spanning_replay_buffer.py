import numpy as np
import faiss
from rlkit.data_management import replay_buffer, env_replay_buffer
from gym.spaces import Discrete 

class SpanningReplayBuffer(replay_buffer.ReplayBuffer):
    """
    A class used to save and replay data.
    """
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
            l2_ball=1.0,
            dtype=np.float32
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._observations = np.zeros((self._max_replay_buffer_size, observation_dim), dtype=dtype)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((self._max_replay_buffer_size, observation_dim), dtype=dtype)
        self._actions = np.zeros((self._max_replay_buffer_size, action_dim), dtype=dtype)
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1), dtype=dtype)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0

        self._index = faiss.IndexFlatL2(self._observation_dim + self._action_dim)
        self.max_l2 = l2_ball

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        query_vec = np.r_[observation, action][np.newaxis,:].astype(np.float32)
        if self._index.ntotal > 0:
            D, _ = self._index.search(query_vec, 5)
            avg_dist = np.mean(D)
        else:
            avg_dist = float('inf')

        if avg_dist >= self.max_l2:
            self._index.add(query_vec)
            self._observations[self._top] = observation
            self._actions[self._top] = action
            self._rewards[self._top] = reward
            self._terminals[self._top] = terminal
            self._next_obs[self._top] = next_observation
            self._advance()


    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        return len(self)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            indices=indices,
        )

    def terminate_episode(self):
        pass

    def __len__(self):
        return self._index.ntotal


class SpanningEnvReplayBuffer(SpanningReplayBuffer):
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
        super(SpanningEnvReplayBuffer, self).add_sample(
                observation, action, reward, terminal, 
                next_observation, **kwargs)

