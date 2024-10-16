"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLPClassic, BroNet


class Critic(nn.Module):
    hidden_dims: int = 1024
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    use_bronet: bool = True
    num_envs: int = 10

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions, task_ids], -1)
        if self.use_bronet:
            critic = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes)(inputs)
        else:
            critic = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes)(inputs)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic        
        
class DoubleCritic(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    use_bronet: bool = False
    num_envs: int = 10

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, task_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, use_bronet=self.use_bronet)(observations, actions, task_ids)
        critic2 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, use_bronet=self.use_bronet)(observations, actions, task_ids)
        return critic1, critic2
    
