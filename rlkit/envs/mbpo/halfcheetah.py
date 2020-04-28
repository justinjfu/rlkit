import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        
        self.qpos_idx = slice(0,self.model.nq)
        self.qvel_idx = slice(self.model.nq,self.model.nv)

    def step(self, action):
        #xposbefore = self.sim.data.qpos[0]
        ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        #xposafter = self.sim.data.qpos[0]
        nob = self._get_obs()
        reward = self.reward(ob, action, nob)
        done = False
        return nob, reward, done, {}

    def sim_reset(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        return np.c_[qpos, qvel]

    def sim_transition(self, state, action):
        self.set_state(state)
        self.do_simulation(action, self.frame_skip)
        return self.sim.data.qpos

    def sim_reward(self, state, action, next_state):
        reward_ctrl = - 0.1 * np.square(action).sum()
        xposbefore = state[0]
        xposafter = next_state[0]
        reward_run = (xposafter-xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        return reward

    def sim_obs(self, state):
        return np.concatenate([
            state[self.qpos_idx].flat[1:],
            state[self.qvel_idx].flat,
        ])

    def _get_obs(self, state):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
