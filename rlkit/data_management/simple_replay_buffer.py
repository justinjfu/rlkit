import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            indices=indices,
        )

    def num_steps_can_sample(self):
        return self._size

    def __len__(self):
        return self._size

    def save_to(self, filename):
        np.savez(filename,
                observation_dim=self._observation_dim,
                action_dim=self._action_dim,
                max_replay_buffer_size=self._max_replay_buffer_size,
                observations=self._observations,
                next_obs=self._next_obs,
                actions=self._actions,
                rewards=self._rewards,
                terminals=self._terminals,
                top=self._top,
                size=self._size)

    def load_from(self, filename, num_replay_samples=None):
        npz_data = np.load(filename)
        self._observation_dim = npz_data['observation_dim']
        self._action_dim = npz_data['action_dim']
        self._max_replay_buffer_size = npz_data['max_replay_buffer_size']
        self._top = npz_data['top']
        self._size = npz_data['size']
        if num_replay_samples is not None:
            self._size = min(num_replay_samples, self._size)

        self._observations = npz_data['observations']
        self._next_obs = npz_data['next_obs']
        self._actions = npz_data['actions']
        self._rewards = npz_data['rewards']
        self._terminals = npz_data['terminals']

