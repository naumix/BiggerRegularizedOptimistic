import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.bro_minimal import temperature
from jaxrl.bro_minimal.actor import update as update_actor

from jaxrl.bro_minimal.critic import target_update
from jaxrl.bro_minimal.critic import update as update_critic
from jaxrl.bro_minimal.critic import update_quantile as update_critic_quantile

from jaxrl.replay_buffer import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None))
def _update(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    tau: float, 
    target_entropy: float, 
    distributional: bool, 
    quantile_taus: jnp.ndarray, 
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
    new_temp, alpha_info = temperature.update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
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
        'pessimism',
        'num_updates'
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    distributional: bool,
    quantile_taus: jnp.ndarray, 
    pessimism: bool,
    step: int,
    num_updates: int
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches),
            discount,
            tau,
            target_entropy,
            distributional,
            quantile_taus,
            pessimism
        )
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    step, rng, actor, critic, target_critic, temp, info = one_step(
        0, (step, rng, actor, critic, target_critic, temp, {})
    )
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, info))

class BROMinimal(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        pessimism: float = 0.0,
        num_seeds: int = 5,
        updates_per_step: int = 10,
        distributional: bool = True,
        n_quantiles: int = 100,
    ) -> None:
        
        self.distributional = distributional
        self.n_quantiles = n_quantiles
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.pessimism = pessimism
        quantile_taus = jnp.arange(0, n_quantiles+1) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.seeds = jnp.arange(seed, seed + num_seeds)
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
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 3)
            actor_def = policies.NormalTanhPolicy(action_dim, use_bronet=True)
            critic_def = critic_net.DoubleCritic(output_nodes=output_nodes, use_bronet=True)
            actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
            temp = Model.create(temperature.Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    #for compatibility
    def sample_actions_o(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int, env_step: int) -> InfoDict:
        if env_step in self.reset_list:
            self.reset()

        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.distributional,
            self.quantile_taus, 
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
        return info

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
