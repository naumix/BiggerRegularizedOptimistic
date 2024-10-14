import functools
from typing import Optional, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLPClassic, BroNet, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

class DualTanhPolicy(nn.Module):
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_means: float = 0.01
    use_bronet: bool = True
    num_envs: int = 10

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 task_ids: jnp.ndarray,
                 means: jnp.ndarray,
                 stds: jnp.ndarray,
                 std_multiplier: float,
                 return_params: bool = True) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, means], -1)
        if self.use_bronet:
            outputs = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(inputs)
        else:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(inputs)
        action_shift = nn.Dense(self.action_dim*self.num_envs, kernel_init=default_init(scale=self.scale_means), use_bias=False)(outputs)
        action_shift = jnp.stack(jnp.split(action_shift, self.num_envs, axis=-1))
        action_shift = (action_shift * task_ids).sum(0)
        optimistic_means = means + action_shift
        mult = jnp.ones(1) * std_multiplier
        optimistic_stds = stds * mult
        base_dist = tfd.MultivariateNormalDiag(loc=optimistic_means, scale_diag=optimistic_stds)
        if return_params is False:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), optimistic_means, optimistic_stds

class NormalTanhPolicy(nn.Module):
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    use_bronet: bool = True
    num_envs: int = 10

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, task_ids: jnp.ndarray, temperature: float = 1.0, training: bool = False, return_params: bool = False
    ) -> tfd.Distribution:
        if self.use_bronet:
            outputs = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(observations)
        else:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(observations)
        means = nn.Dense(self.action_dim*self.num_envs, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim*self.num_envs, kernel_init=default_init(self.log_std_scale))(outputs)
        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + nn.tanh(log_stds))
        stds = jnp.exp(log_stds)
        stds = stds * temperature
        means = jnp.stack(jnp.split(means, self.num_envs, axis=-1))
        stds = jnp.stack(jnp.split(stds, self.num_envs, axis=-1))
        means = (means * task_ids).sum(0)
        stds = (stds * task_ids).sum(0)
        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=stds)
        if return_params is False:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), means, stds
        #return means, stds
    
@functools.partial(jax.jit, static_argnames=('actor_def'))
def _sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    task_ids: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, task_ids, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions

def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    task_ids: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, task_ids, temperature)

@functools.partial(jax.jit, static_argnames=('actor_o_def', 'actor_c_def'))
def _sample_actions_o(
        rng: PRNGKey,
        actor_o_def: nn.Module,
        actor_o_params: Params,
        actor_c_def: nn.Module,
        actor_c_params: Params,
        observations: np.ndarray,
        task_ids: np.ndarray,
        std_multiplier: float,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    _, mu_c, std_c = actor_c_def.apply({'params': actor_c_params}, observations=observations, task_ids=task_ids, temperature=temperature, return_params=True)
    dist, mu_o, std_o = actor_o_def.apply({'params': actor_o_params}, observations=observations, task_ids=task_ids, means=mu_c, stds=std_c, std_multiplier=std_multiplier)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)

def sample_actions_o(
        rng: PRNGKey,
        actor_o_def: nn.Module,
        actor_o_params: Params,
        actor_c_def: nn.Module,
        actor_c_params: Params,
        observations: np.ndarray,
        task_ids: np.ndarray,
        std_multiplier: float,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions_o(rng, actor_o_def, actor_o_params, actor_c_def, actor_c_params, observations, task_ids, std_multiplier, temperature)
