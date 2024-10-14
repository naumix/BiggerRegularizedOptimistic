import metaworld
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
        
class make_env_mt(gym.Env):
    def __init__(self, seed):
        #seed = 0
        np.random.seed(seed)
        mt_env = metaworld.MT10(seed=seed)
        envs = []
        for name, env_cls in mt_env.train_classes.items():
            env = env_cls()
            task = random.choice([task for task in mt_env.train_tasks if task.env_name == name])
            env.set_task(task)
            envs.append(env)
        self.envs = envs 
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)
        
    def reset_idx(self, idx):
        seed = np.random.randint(0,1e6)
        obs, _ = self.envs[idx].reset(seed=seed)
        return obs
    
    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return observations, terms, truns, resets
    
    def reset(self):
        obs = []
        for env in self.envs:
            ob, _ = env.reset()
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
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goals.append(info['success'])
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)
    
    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        n_seeds = 10
        goals = []
        returns_eval = []
        task_ids = np.eye(10)[:,:,None]
        for _episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(n_seeds)
            goal = 0.0
            for i in range(500): # CHANGE?
                actions = agent.sample_actions(observations, task_ids, temperature=temperature)
                next_observations, rewards, terms, truns, goals_ = self.step(actions)
                goal += goals_ / 500
                returns += rewards
                observations = next_observations            
            goal[goal > 0] = 1.0
            goals.append(goal)
            returns_eval.append(returns)
        return {'goal': np.array(goals).mean(axis=0), 'return': np.array(returns_eval).mean(axis=0)}
