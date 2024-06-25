from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class make_env_mw(gym.Env):
    def __init__(self, env_name, num_envs, seed, max_t=200):
        
        np.random.seed(seed)
        self.num_seeds = num_envs
        self.constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        seeds = np.random.randint(0,1e6,(self.num_seeds))
        self.max_t = max_t
        self.envs = [self.constructor(seed=int(seed)) for seed in seeds]
        self.timesteps = np.zeros(self.num_seeds)             
        self.num_envs = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)

    def _reset_idx(self, idx):
        seed = np.random.randint(0,1e6)
        self.envs[idx] = self.constructor(seed=int(seed))
        obs, _ = self.envs[idx].reset()
        return obs

    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 0 
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
        seeds = np.random.randint(0,1e6,(self.num_seeds))
        self.timesteps = np.zeros(self.num_seeds)             
        self.envs = [self.constructor(seed=int(seed)) for seed in seeds]
        obs = []
        for env in self.envs:
            ob, _ = env.reset()
            obs.append(ob)
        return np.stack(obs)

    def step(self, actions):
        obs, rews, terms, truns, goals = [], [], [], [], []
        self.timesteps += 1
        for timestep, env, action in zip(self.timesteps, self.envs, actions):
            ob, reward, term, _, info = env.step(action)
            if timestep == self.max_t:
                trun = True
            else:
                trun = False
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goals.append(info['success'])
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        n_seeds = self.num_seeds
        goals = []
        returns_eval = []
        for _episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(n_seeds)
            goal = 0.0
            for i in range(self.max_t): # CHANGE?
                actions = agent.sample_actions(observations, temperature=temperature)
                next_observations, rewards, terms, truns, goals = self.step(actions)
                goal += goals / self.max_t
                returns += rewards
                observations = next_observations            
            goal[goal > 0] = 1.0
            goals.append(goal)
            returns_eval.append(returns)
        return {'goal': np.array(goals).mean(axis=0), 'return': np.array(returns_eval).mean(axis=0)}
