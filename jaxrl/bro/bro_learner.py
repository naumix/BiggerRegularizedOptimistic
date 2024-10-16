import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.bro import temperature
from jaxrl.bro.actor import update as update_actor
from jaxrl.bro.actor import update_optimistic as update_actor_optimistic

from jaxrl.bro.critic import target_update
from jaxrl.bro.critic import update as update_critic
from jaxrl.bro.critic import update_quantile as update_critic_quantile

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
    new_actor_o, actor_o_info = update_actor_optimistic(key, new_actor, actor_o, new_critic, optimism, regularizer, batch, std_multiplier, distributional)
    new_temp, alpha_info = temperature.update_temperature(temp, actor_info['entropy'], target_entropy)
    empirical_kl = actor_o_info['kl'] / action_dim
    new_optimism, optimism_info = temperature.update_optimism(optimism, empirical_kl, kl_target, pessimism)
    new_regularizer, regularizer_info = temperature.update_regularizer(regularizer, empirical_kl, kl_target)
    return rng, new_actor, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, {
        **critic_info,
        **actor_info,
        **actor_o_info,
        **alpha_info,
        **optimism_info,
        **regularizer_info
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

class BRO(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_type: int = 10,
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
        pessimism: float = 0.0,
        num_seeds: int = 5,
        updates_per_step: int = 10,
        distributional: bool = True,
        n_quantiles: int = 100,
        kl_target: float = 0.05,
        std_multiplier: float = 0.75,
    ) -> None:
        
        self.seed = seed
        self.task_type = task_type
        self.distributional = distributional
        self.n_quantiles = n_quantiles
        self.std_multiplier = std_multiplier
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.kl_target = kl_target
        self.pessimism = pessimism
        quantile_taus = jnp.arange(0, n_quantiles+1) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.reset_list = [15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001]
        if updates_per_step == 2:
            self.reset_list = self.reset_list[:1]
        self.num_seeds = num_seeds
        
        output_nodes = self.n_quantiles if self.distributional else 1

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key, pessimism_key, actor_o_key, optimism_key, regularizer_key = jax.random.split(rng, 8)
            actor_def = policies.NormalTanhPolicy(action_dim, use_bronet=True)
            actor_o_def = policies.DualTanhPolicy(action_dim, use_bronet=True)
            critic_def = critic_net.DoubleCritic(output_nodes=output_nodes, use_bronet=True)
            
            actor = Model.create(actor_def, inputs=[actor_key, observations, jnp.eye(self.task_type)[0][None, :]], tx=optax.adamw(learning_rate=actor_lr))
            actor_o = Model.create(actor_o_def, inputs=[actor_o_key, observations, jnp.eye(self.task_type)[0][None, :], actions, actions, self.std_multiplier], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions, jnp.eye(self.task_type)[0][None, :]], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, jnp.eye(self.task_type)[0][None, :]])
            
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
        rng, actions = policies.sample_actions_o(self.rng, self.actor_o.apply_fn, self.actor_o.params, self.actor.apply_fn, self.actor.params, observations, task_ids, self.std_multiplier, temperature)
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
        self.actor, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer, self.rng = self.init_models(self.seed)
