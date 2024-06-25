import gymnasium as gym
import numpy as np
from gymnasium import spaces

class make_env_dexhand(gym.Env):
    def __init__(self, env_name='HandManipulateBlock-v1', seed=0, num_envs=2, max_t=100):
        np.random.seed(seed)
        seeds = np.random.randint(0,1e6,(num_envs))
        self.max_t = max_t
        self.envs = [gym.wrappers.RescaleAction(gym.make(env_name, max_episode_steps=self.max_t), -1.0, 1.0) for seed in seeds]
        self.num_envs = len(self.envs)
        self.timesteps = np.zeros(self.num_envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        lows = np.concatenate((self.envs[0].observation_space['observation'].low[None], self.envs[0].observation_space['achieved_goal'].low[None], self.envs[0].observation_space['desired_goal'].low[None]), axis=-1).repeat(num_envs, axis=0),
        high = np.concatenate((self.envs[0].observation_space['observation'].high[None], self.envs[0].observation_space['achieved_goal'].high[None], self.envs[0].observation_space['desired_goal'].high[None]), axis=-1).repeat(num_envs, axis=0),
        self.observation_space = spaces.Box(low=lows[0],
                                            high=high[0],
                                            shape=(len(self.envs), lows[0].shape[1]),
                                            dtype=self.envs[0].observation_space['observation'].dtype)
        self.action_dim = self.envs[0].action_space.shape[0]

    def _concatenate_state(self, raw_state):
        state = np.concatenate((raw_state['observation'], raw_state['achieved_goal'], raw_state['desired_goal']))
        return state
        
    def _reset_idx(self, idx):
        seed_ = np.random.randint(0,1e6)
        obs, _ = self.envs[idx].reset(seed=seed_)
        obs = self._concatenate_state(obs)
        return obs
    
    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 0
        return observations, terms, truns, resets
    
    def reset(self):
        obs = []
        for env in self.envs:
            seed_ = np.random.randint(0,1e6)
            ob, _ = env.reset(seed=seed_)
            ob = self._concatenate_state(ob)
            obs.append(ob)
        return np.stack(obs)
    
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
    
    def step(self, actions):
        obs, rews, terms, truns, goals = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, term, trun, info = env.step(action)
            ob = self._concatenate_state(ob)
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goal = 1.0 if info['is_success'] else 0.0
            goals.append(goal)
        self.timesteps += 1
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)

    def random_step(self):
        actions = np.random.uniform(-1, 1, (self.num_envs, self.action_dim))
        obs, rews, terms, truns, goals = self.step(actions)
        return obs, rews, terms, truns, goals, actions
    
    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        returns_mean = np.zeros(self.num_envs)
        returns = np.zeros(self.num_envs)
        goals_mean = np.zeros(self.num_envs)
        goals = np.zeros(self.num_envs)
        episode_count = np.zeros(self.num_envs)
        state = self.reset()
        while episode_count.min() < num_episodes:
            actions = agent.sample_actions(state, temperature=temperature)
            new_state, reward, term, trun, goal = self.step(actions)
            returns += reward
            goals += goal
            state = new_state
            state, term, trun, reset_mask = self.reset_where_done(state, term, trun)
            episode_count += reset_mask
            goals = np.where(goals>0, 1.0, 0.0)
            goals_mean += reset_mask * goals
            returns_mean += reset_mask * returns
            returns *= (1 - reset_mask)
            goals *= (1 - reset_mask)
        return {"return": returns_mean/episode_count, "goals_mean": goals_mean/episode_count}
    