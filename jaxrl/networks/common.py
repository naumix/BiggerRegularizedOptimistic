import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLPClassic(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    output_nodes: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer3(x)
            x = self.activations(x)
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer4(x)
            x = self.activations(x)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 3:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer3(x)
            x = self.activations(x)
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer4(x)
            x = self.activations(x)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer5(x)
            x = self.activations(x)
            layer6 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer6(x)
            x = self.activations(x)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x

class BroNet(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    output_nodes: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer5(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 3:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer5(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer6 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer6(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer7 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer7(res)
            res = nn.LayerNorm()(res)
            x = res + x
            
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x

@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(params=self.params, opt_state=self.opt_state)))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(params=self.params, opt_state=self.opt_state), contents
            )
        return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)


def split_tree(tree, key):
    tree_head = tree.unfreeze()
    tree_enc = tree_head.pop(key)
    tree_head = flax.core.FrozenDict(tree_head)
    tree_enc = flax.core.FrozenDict(tree_enc)
    return tree_enc, tree_head
