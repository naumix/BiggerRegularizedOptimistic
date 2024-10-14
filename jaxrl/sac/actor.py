from typing import Tuple
import jax.numpy as jnp
from jaxrl.replay_buffer import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch, pessimism: float, distributional: bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, mu, std = actor.apply({'params': actor_params}, batch.observations, batch.task_ids, return_params=True)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions, batch.task_ids)
        q = (q1 + q2)/2 - pessimism * jnp.abs(q1 - q2) / 2
        if distributional:
            q = q.mean(-1)
        actor_loss = (log_probs * temp().mean() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
            'std': std.mean(),
        }
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, info

def update_optimistic(key: PRNGKey, actor_c: Model, actor_o: Model, critic: Model, optimism: Model, regularizer: Model, batch: Batch, std_multiplier: float, distributional: bool) -> Tuple[Model, InfoDict]:
    def actor_o_loss_fn(actor_o_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, mu_c, std_c = actor_c(batch.observations, batch.task_ids, temperature=1.0, return_params=True)
        dist, mu_o, std_o = actor_o.apply({'params': actor_o_params}, observations=batch.observations, task_ids=batch.task_ids, means=mu_c, stds=std_c, std_multiplier=std_multiplier, return_params=True)        
        actions = dist.sample(seed=key)
        q1, q2 = critic(batch.observations, actions, batch.task_ids)
        kl = (jnp.log(std_c/(std_o / std_multiplier)) + ((std_o / std_multiplier) ** 2 + (mu_o - mu_c) ** 2)/(2 * std_c ** 2) - 1/2).sum(-1)
        q_ub = (q1 + q2) / 2 + optimism() * jnp.abs(q1 - q2) / 2
        if distributional:
            q_ub = q_ub.mean(-1)
        actor_e_loss = (-q_ub).mean() + regularizer() * kl.mean()    
        return actor_e_loss, {
            'actor_o_loss': actor_e_loss,
            'kl': kl.mean(),
            'actor_o_pnorm': tree_norm(actor_o_params),
            'std_c': std_c.mean(),
            'std_o': std_o.mean(),
            'Q_mean': ((q1 + q2) / 2).mean(),
            'Q_std': (jnp.abs(q1 - q2) / 2).mean(),
        }
    new_actor_o, info = actor_o.apply_gradient(actor_o_loss_fn)
    info['actor_o_gnorm'] = info.pop('grad_norm')
    return new_actor_o, info



