from typing import Tuple
import jax
import jax.numpy as jnp
from jaxrl.replay_buffer import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm

def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, pessimism: float) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations, batch.task_ids)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions, batch.task_ids)
    next_q = (next_q1 + next_q2) / 2 - 1.0 * jnp.abs(next_q1 - next_q2) / 2
    target_q = batch.rewards + discount * batch.masks * next_q
    target_q -= discount * temp() * batch.masks * next_log_probs
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_fn = lambda actions: critic.apply({"params": critic_params}, batch.observations, actions, batch.task_ids)
        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5 * (q1 + q2).mean(), (q1, q2)
        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
        }
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info

def update_quantile(key: PRNGKey, actor: Model, quantile_critic: Model, target_quantile_critic: Model,
           temp: Model, batch: Batch, discount: float, pessimism: float, taus: jnp.ndarray) -> Tuple[Model, InfoDict]:
    kappa = 1.0
    dist = actor(batch.next_observations, batch.task_ids)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_quantile_critic(batch.next_observations, next_actions, batch.task_ids)
    next_q = (next_q1 + next_q2) / 2 - 1.0 * jnp.abs(next_q1 - next_q2) / 2
    target_q = batch.rewards[..., None, None] + discount * batch.masks[..., None, None]  * next_q[:, None, :]
    target_q -= discount * temp().mean() * batch.masks[..., None, None] * next_log_probs[..., None, None]
    def critic_loss_fn(quantile_critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_fn = lambda actions: quantile_critic.apply({"params": quantile_critic_params}, batch.observations, actions, batch.task_ids)
        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5 * (q1 + q2).mean(), (q1, q2)
        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        td_errors1 = target_q - q1[..., None]
        td_errors2 = target_q - q2[..., None] 
        critic_loss = calculate_quantile_huber_loss(td_errors1, taus, kappa=kappa) + calculate_quantile_huber_loss(td_errors2, taus, kappa=kappa)
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(quantile_critic_params),
            "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
        }
    new_quantile_critic, info = quantile_critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_quantile_critic, info

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)

def huber_replace(td_errors, kappa: float = 1.0):
    return jnp.where(jnp.absolute(td_errors) <= kappa, 0.5 * td_errors ** 2, kappa * (jnp.absolute(td_errors) - 0.5 * kappa))

def calculate_quantile_huber_loss(td_errors, taus, kappa: float = 1.0):
    element_wise_huber_loss = huber_replace(td_errors, kappa)
    mask = jax.lax.stop_gradient(jnp.where(td_errors < 0, 1, 0)) # detach this
    element_wise_quantile_huber_loss = jnp.absolute(taus[..., None] - mask) * element_wise_huber_loss / kappa
    quantile_huber_loss = element_wise_quantile_huber_loss.sum(axis=1).mean()
    return quantile_huber_loss
