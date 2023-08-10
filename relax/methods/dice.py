# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/methods/02_dice.ipynb.

# %% ../../nbs/methods/02_dice.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from .base import CFModule
from ..base import BaseConfig
from ..utils import auto_reshaping, grad_update, validate_configs

# %% auto 0
__all__ = []

# %% ../../nbs/methods/02_dice.ipynb 13
def _diverse_cf(
    x: jnp.DeviceArray,  # `x` shape: (k,), where `k` is the number of features
    y_target: Array, # `y_target` shape: (1,)
    pred_fn: Callable[[Array], Array],  # y = pred_fn(x)
    n_cfs: int,
    n_steps: int,
    lr: float,  # learning rate for each `cf` optimization step
    lambdas: Tuple[float, float, float, float], # (lambda_1, lambda_2, lambda_3, lambda_4)
    key: jrand.PRNGKey,
    validity_fn: Callable,
    cost_fn: Callable,
    apply_constraints_fn: Callable,
    compute_reg_loss_fn: Callable,
) -> Array:  # return `cf` shape: (k,)
    """Diverse Counterfactuals (Dice) algorithm."""

    def loss_fn(
        cfs: Array, # shape: (n_cfs, k)
        x: Array, # shape: (1, k)
        pred_fn: Callable[[Array], Array], # y = pred_fn(x)
        y_target: Array,
    ):
        cf_y_pred = pred_fn(cfs)
        loss_1 = validity_fn(y_target, cf_y_pred).mean()
        loss_2 = cost_fn(x, cfs).mean()
        loss_3 = - dpp_style_vmap(cfs).mean()
        loss_4 = compute_reg_loss_fn(x, cfs)
        return (
            lambda_1 * loss_1 + 
            lambda_2 * loss_2 + 
            lambda_3 * loss_3 + 
            lambda_4 * loss_4
        )
    
    @loop_tqdm(n_steps)
    def gen_cf_step(i, states: Tuple[Array, optax.OptState]):
        cf, opt_state = states
        grads = jax.grad(loss_fn)(cf, x, pred_fn, y_target)
        cf_updates, opt_state = grad_update(grads, cf, opt_state, opt)
        return cf, opt_state
    
    lambda_1, lambda_2, lambda_3, lambda_4 = lambdas
    key, subkey = jrand.split(key)
    cfs = jrand.normal(key, (n_cfs, x.shape[-1]))
    opt = optax.adam(lr)
    opt_state = opt.init(cfs)
    
    cfs, opt_state = lax.fori_loop(0, n_steps, gen_cf_step, (cfs, opt_state))
    # TODO: support return multiple cfs
    # cfs = apply_constraints_fn(x, cfs[:1, :], hard=True)
    cfs = apply_constraints_fn(x, cfs, hard=True)
    return cfs

