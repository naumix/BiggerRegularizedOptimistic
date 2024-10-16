import gym
import numpy as np

import os
import pickle
import collections

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'dones', 'next_observations', 'task_ids'])


class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_seeds: int):
        self.observations = np.empty((capacity*num_seeds, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((capacity*num_seeds, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity*num_seeds, ), dtype=np.float32)
        self.masks = np.empty((capacity*num_seeds, ), dtype=np.float32)
        self.dones_float = np.empty((capacity*num_seeds, ), dtype=np.float32)
        self.next_observations = np.empty((capacity*num_seeds, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.task_ids = np.empty((capacity*num_seeds, num_seeds), dtype=np.float32)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.num_seeds = num_seeds        
        
    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray, task_ids: np.ndarray):
        self.observations[self.insert_index:self.insert_index+self.num_seeds] = observation
        self.actions[self.insert_index:self.insert_index+self.num_seeds] = action
        self.rewards[self.insert_index:self.insert_index+self.num_seeds] = reward
        self.masks[self.insert_index:self.insert_index+self.num_seeds] = mask
        self.dones_float[self.insert_index:self.insert_index+self.num_seeds] = done_float
        self.next_observations[self.insert_index:self.insert_index+self.num_seeds] = next_observation
        self.task_ids[self.insert_index:self.insert_index+self.num_seeds] = task_ids
        self.insert_index = (self.insert_index + self.num_seeds) % self.capacity
        self.size = min(self.size + self.num_seeds, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     dones=self.dones_float[indx],
                     next_observations=self.next_observations[indx],
                     task_ids=self.task_ids[indx])

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        #indxs = np.arange(128)[None, :]
        return Batch(observations=self.observations[indxs],
                     actions=self.actions[indxs],
                     rewards=self.rewards[indxs],
                     masks=self.masks[indxs],
                     dones=self.dones_float[indxs],
                     next_observations=self.next_observations[indxs],
                     task_ids=self.task_ids[indxs])
    
    def sample_state(self, batch_size: int) -> np.ndarray:
        indx = np.random.randint(self.size, size=batch_size)
        return self.observations[:, indx]
