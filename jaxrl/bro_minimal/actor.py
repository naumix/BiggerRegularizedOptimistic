from typing import Tuple
import jax.numpy as jnp
from jaxrl.replay_buffer import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm

def update(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch, pessimism: float, distributional: bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, mu, std = actor.apply({'params': actor_params}, batch.observations, return_params=True)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
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