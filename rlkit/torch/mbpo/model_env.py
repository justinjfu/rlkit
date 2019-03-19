from rlkit.core.serializable import Serializable
from rlkit.envs import wrappers

class ModelEnv(wrappers.ProxyEnv):
    def __init__(
            self,
            env,
            max_timesteps=float('inf'),
            reward_function=None,
            model=None,
    ):
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(ModelEnv, self).__init__(env)
        self.model = model
        self.current_obs = None
        self._t = 0
        self.max_t = max_timesteps
        self.reward_fn = reward_function

    def reset(self, **kwargs):
        self.current_obs = self._wrapped_env.reset(**kwargs)
        self._t = 0
        return self.current_obs

    def step(self, action):
        if self.model:
            next_obs = self.model.step(self.current_obs, action)
            reward = self._wrapped_env.reward(self.current_obs, action, next_obs)
        else:
            wrapped_step = self._wrapped_env.step(action)
            next_obs, reward, _, _ = wrapped_step
        self.current_obs = next_obs
        done = False

        self._t += 1
        if self._t >= self.max_t:
            done = True
        return next_obs, reward, done, {}

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

