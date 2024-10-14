from typing import Tuple
from flax import linen as nn
from jaxrl.networks.common import InfoDict, Model
import jax.numpy as jnp

class Adjustment(nn.Module):
    init_value: float = 1.0
    offset: float = 0.0
    log_val_min: float = -10.0
    log_val_max: float = 7.5
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        log_value = self.log_val_min + (self.log_val_max - self.log_val_min) * 0.5 * (1 + nn.tanh(log_value))
        return jnp.exp(log_value) - self.offset 

class Temperature(nn.Module):
    initial_temperature: float = 1.0
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

def update_optimism(
        optimism: Model, empirical_kl: float, target_kl: float, pessimism: float = 0.0) -> Tuple[Model, InfoDict]:
    def optimism_loss_fn(optimism_params):
        optimism_ = optimism.apply({'params': optimism_params})
        optimism_loss = (optimism_ - pessimism) * (empirical_kl - target_kl).mean()
        return optimism_loss, {'optimism': optimism_, 'optimism_loss': optimism_loss}
    new_optimism, info = optimism.apply_gradient(optimism_loss_fn)
    info.pop('grad_norm')
    return new_optimism, info

def update_regularizer(
        regularizer: Model, empirical_kl: float, target_kl: float) -> Tuple[Model, InfoDict]:
    def regularizer_loss_fn(regularizer_params):
        kl_weight = regularizer.apply({'params': regularizer_params})
        regularizer_loss = -kl_weight * (empirical_kl - target_kl).mean()
        return regularizer_loss, {'kl_weight': kl_weight, 'regularizer_loss': regularizer_loss}
    new_regularizer, info = regularizer.apply_gradient(regularizer_loss_fn)
    info.pop('grad_norm')
    return new_regularizer, info

def update_temperature(temp: Model, entropy: float, target_entropy: float) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info
