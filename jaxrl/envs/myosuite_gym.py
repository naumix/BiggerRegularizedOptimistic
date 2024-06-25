import gym
import numpy as np
from gymnasium import spaces
import myosuite

MYOSUITE_TASKS = {
    'myo-test' : 'myoElbowPose1D6MRandom-v0',
	'myo-reach': 'myoHandReachFixed-v0',
	'myo-reach-hard': 'myoHandReachRandom-v0',
	'myo-pose': 'myoHandPoseFixed-v0',
	'myo-pose-hard': 'myoHandPoseRandom-v0',
	'myo-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
	'myo-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
	'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}

class make_env_myo(gym.Env):
    def __init__(self, env_name='myo-test', num_envs=2, seed=0, max_t=100):
        np.random.seed(seed)
        seeds = np.random.randint(0,1e6,(num_envs))
        self.max_t = max_t
        self.envs = [gym.make(MYOSUITE_TASKS[env_name]) for seed in seeds]
        self.num_envs = len(self.envs)
        self.timesteps = np.zeros(self.num_envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)
        self.action_wrapper_multiplier = self.envs[0].action_space.high[0]
        self.action_dim = self.envs[0].action_space.shape[0]
        
    def _reset_idx(self, idx):
        obs = self.envs[idx].reset()
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
        self.timesteps = np.zeros(self.num_envs)
        for env in self.envs:
            ob = env.reset()
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
        self.timesteps += 1
        actions_ = actions * self.action_wrapper_multiplier
        for timestep, env, action in zip(self.timesteps, self.envs, actions_):
            ob, reward, _, info = env.step(action)
            term = False
            obs.append(ob)
            goal = info['solved']
            #if timestep == self.max_t or info['solved'] == 1:
            if timestep == self.max_t:
                trun = True
            else:
                trun = False
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goals.append(goal)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)

    def random_step(self):
        actions = self.action_space.sample()
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
            #actions = self.action_space.sample()
            new_state, reward, term, trun, goal = self.step(actions)
            goals += goal
            goals[goals > 0.0] = 1.0
            returns += reward
            state = new_state
            state, term, trun, reset_mask = self.reset_where_done(state, term, trun)
            episode_count += reset_mask
            returns_mean += reset_mask * returns
            returns *= (1 - reset_mask)
            goals_mean += reset_mask * goals
            goals *= (1 - reset_mask)
        return {"return": returns_mean/episode_count, "goals_mean": goals_mean/episode_count}
    
#env = make_env_myo()
#env.evaluate(None, 10, 0.0)
