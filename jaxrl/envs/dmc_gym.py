import gym
import numpy as np
from jaxrl.envs.single_precision import SinglePrecision
import copy
from typing import Dict, Optional, OrderedDict, Tuple
from dm_control import suite
from dm_env import specs
from gym import core, spaces

TimeStep = Tuple[np.ndarray, float, bool, dict]

def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError

class dmc2gym(core.Env):
    def __init__(self, domain_name: str, task_name: str, task_kwargs: Optional[Dict] = {}, environment_kwargs=None):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)
        self.action_space = dmc_spec2gym_space(self._env.action_spec())
        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())
        self.seed(seed=task_kwargs['random'])

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)
        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation
        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

def _make_env_dmc(env_name: str, seed: int) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = dmc2gym(domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed})
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    from gym.wrappers import RescaleAction
    env = RescaleAction(env, -1.0, 1.0)
    env = SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class make_env_dmc(gym.Env):
    def __init__(self, env_name: str, seed: int, num_envs: int, max_t=1000):
        env_fns = [lambda i=i: _make_env_dmc(env_name, seed + i) for i in range(num_envs)]
        self.envs = [env_fn() for env_fn in env_fns]
        self.max_t = max_t
        self.num_seeds = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)

    def _reset_idx(self, idx):
        return self.envs[idx].reset()

    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return observations, terms, truns, resets

    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.stack(obs)

    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(False)
            trun = True if 'TimeLimit.truncated' in info else False
            truns.append(trun)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), None

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        num_seeds = self.num_seeds
        returns_eval = []
        for episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(num_seeds)
            for i in range(self.max_t): # CHANGE?
                actions = agent.sample_actions(observations, temperature=temperature)
                next_observations, rewards, terms, truns, goals = self.step(actions)
                returns += rewards
                observations = next_observations            
            returns_eval.append(returns)
        return {'return': np.array(returns_eval).mean(axis=0)}