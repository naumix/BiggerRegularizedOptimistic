import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.sac import temperature
from jaxrl.sac.actor import update as update_actor
from jaxrl.sac.actor import update_optimistic as update_actor_optimistic

from jaxrl.sac.critic import target_update
from jaxrl.sac.critic import update as update_critic
from jaxrl.sac.critic import update_quantile as update_critic_quantile

from jaxrl.replay_buffer import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

def _update(
    rng: PRNGKey, 
    actor: Model, 
    actor_o: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    optimism: Model, 
    regularizer: Model, 
    batch: Batch, 
    discount: float, 
    tau: float, 
    target_entropy: float, 
    distributional: bool, 
    quantile_taus: jnp.ndarray, 
    std_multiplier: float, 
    action_dim: int, 
    kl_target: float, 
    pessimism: float
):
    rng, key = jax.random.split(rng)
    if distributional:
        new_critic, critic_info = update_critic_quantile(key, actor, critic, target_critic, temp, batch, discount, pessimism, taus=quantile_taus)
    else:
        new_critic, critic_info = update_critic(key, actor, critic, target_critic, temp, batch, discount, pessimism)
    new_target_critic = target_update(new_critic, target_critic, tau)
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, pessimism, distributional)   
    rng, key = jax.random.split(rng)
    new_actor_o = actor_o
    new_temp, alpha_info = temperature.update_temperature(temp, actor_info['entropy'], target_entropy)
    #empirical_kl = actor_o_info['kl'] / action_dim
    new_optimism = optimism
    new_regularizer = regularizer
    return rng, new_actor, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@functools.partial(
    jax.jit,
    static_argnames=(
        'discount',
        'tau',
        'target_entropy',
        'distributional',
        'std_multiplier',
        'action_dim',
        'kl_target',
        'pessimism',
        'num_updates'
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    actor_o: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    optimism: Model,
    regularizer: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    distributional: bool,
    quantile_taus: jnp.ndarray, 
    std_multiplier: float,
    action_dim: int,
    kl_target: float,
    pessimism: bool,
    step: int,
    num_updates: int
):
    def one_step(i, state):
        step, rng, actor, actor_o, critic, target_critic, temp, optimism, regularizer, info = state
        step = step + 1
        new_rng, new_actor, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, info = _update(
            rng,
            actor,
            actor_o,
            critic,
            target_critic,
            temp,
            optimism,
            regularizer,
            jax.tree_map(lambda x: jnp.take(x, i, axis=0), batches),
            discount,
            tau,
            target_entropy,
            distributional,
            quantile_taus,
            std_multiplier,
            action_dim,
            kl_target,
            pessimism
        )
        return step, new_rng, new_actor, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, info

    step, rng, actor, actor_o, critic, target_critic, temp, optimism, regularizer, info = one_step(
        0, (step, rng, actor, actor_o, critic, target_critic, temp, optimism, regularizer, {})
    )
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, actor_o, critic, target_critic, temp, optimism, regularizer, info))

class SAC(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        adj_lr: float = 3e-5,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        init_optimism: float = 1.0,
        init_regularizer: float = 0.25,
        pessimism: float = 1.0,
        num_seeds: int = 5,
        updates_per_step: int = 2,
        distributional: bool = False,
        n_quantiles: int = 1,
        kl_target: float = 0.05,
        std_multiplier: float = 0.75,
    ) -> None:
        
        self.seed = seed
        self.distributional = False
        self.n_quantiles = n_quantiles
        self.std_multiplier = std_multiplier
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.kl_target = kl_target
        self.pessimism = 1.0
        quantile_taus = jnp.arange(0, n_quantiles+1) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_seeds = num_seeds
        
        output_nodes = self.n_quantiles if self.distributional else 1

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key, pessimism_key, actor_o_key, optimism_key, regularizer_key = jax.random.split(rng, 8)
            actor_def = policies.NormalTanhPolicy(action_dim, hidden_dims=256, depth=1, use_bronet=False)
            actor_o_def = policies.DualTanhPolicy(action_dim, hidden_dims=256, depth=1, use_bronet=False)
            critic_def = critic_net.DoubleCritic(output_nodes=output_nodes, hidden_dims=256, depth=1, use_bronet=False)
            
            actor = Model.create(actor_def, inputs=[actor_key, observations, jnp.eye(10)[:,:,None]], tx=optax.adamw(learning_rate=actor_lr))
            actor_o = Model.create(actor_o_def, inputs=[actor_o_key, observations, jnp.eye(10)[:,:,None], actions, actions, self.std_multiplier], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions, jnp.eye(10)[:,:,None]], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, jnp.eye(10)[:,:,None]])
            
            temp = Model.create(temperature.Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            log_val_min, log_val_max = -10.0, 7.5
            init_optimism_ = self.calculate_init_values(init_optimism, log_val_min, log_val_max)
            init_regularizer_ = self.calculate_init_values(init_regularizer, log_val_min, log_val_max)
            optimism = Model.create(temperature.Adjustment(init_value=init_optimism_, log_val_min=log_val_min, log_val_max=log_val_max), inputs=[optimism_key], tx=optax.adam(learning_rate=adj_lr, b1=0.5))
            regularizer = Model.create(temperature.Adjustment(init_value=init_regularizer_, log_val_min=log_val_min, log_val_max=log_val_max), inputs=[regularizer_key], tx=optax.adam(learning_rate=adj_lr, b1=0.5))
            return actor, actor_o, critic, target_critic, temp, optimism, regularizer, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer, self.rng = self.init_models(self.seed)
        self.step = 1
            
    def calculate_init_values(self, init_value, log_val_min, log_val_max) -> float:
        value = np.exp(np.arctanh((np.log(init_value) - log_val_min)/((log_val_max - log_val_min) * 0.5) - 1))
        return value

    def sample_actions(self, observations: np.ndarray, task_ids: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, task_ids, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def sample_actions_o(self, observations: np.ndarray, task_ids: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, task_ids, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int, env_step: int) -> InfoDict:
        if env_step in self.reset_list:
            self.reset()

        step, rng, actor, actor_o, critic, target_critic, temp, optimism, regularizer, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.actor_o,
            self.critic,
            self.target_critic,
            self.temp,
            self.optimism,
            self.regularizer,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.distributional,
            self.quantile_taus, 
            self.std_multiplier,
            self.action_dim,
            self.kl_target,
            self.pessimism,
            self.step,
            num_updates
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.actor_o = actor_o
        self.optimism = optimism
        self.regularizer = regularizer
        return info

    def reset(self):
        self.step = 1
        self.actor, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer, self.rng = self.init_models(self.seeds)
