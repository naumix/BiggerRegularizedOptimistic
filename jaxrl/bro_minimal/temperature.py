from typing import Tuple
from flax import linen as nn
from jaxrl.networks.common import InfoDict, Model
import jax.numpy as jnp

class Temperature(nn.Module):
    initial_temperature: float = 1.0
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

def update_temperature(temp: Model, entropy: float, target_entropy: float) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info
