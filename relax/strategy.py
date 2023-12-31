# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_explain.strategy.ipynb.

# %% ../nbs/03_explain.strategy.ipynb 2
from __future__ import annotations
from .import_essentials import *
import einops

# %% auto 0
__all__ = ['BaseStrategy', 'IterativeStrategy', 'VmapStrategy', 'PmapStrategy', 'BatchedVmapStrategy', 'BatchedPmapStrategy',
           'StrategyFactory']

# %% ../nbs/03_explain.strategy.ipynb 3
class BaseStrategy:
    """Base class for mapping strategy."""
    
    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        raise NotImplementedError
    
    __ALL__ = ["__call__"]


# %% ../nbs/03_explain.strategy.ipynb 4
class IterativeStrategy(BaseStrategy):
    """Iterativly generate counterfactuals."""

    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        
        assert xs.ndim == 2
        cfs = jnp.stack([fn(xs[i], pred_fn=pred_fn, y_target=y_targets[i], rng_key=rng_keys[i], **kwargs) 
            for i in range(xs.shape[0])])
        return cfs


# %% ../nbs/03_explain.strategy.ipynb 5
class VmapStrategy(BaseStrategy):
    """Generate counterfactuals via `jax.vmap`."""

    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        
        def partial_fn(x, y_target, rng_key):
            return fn(x, pred_fn=pred_fn, y_target=y_target, rng_key=rng_key, **kwargs)
        
        assert xs.ndim == 2
        cfs = jax.vmap(partial_fn)(xs, y_targets, rng_keys)
        return cfs


# %% ../nbs/03_explain.strategy.ipynb 6
def _pad_divisible_X(
    xs: Array,
    n_devices: int
):
    """Pad `X` to be divisible by `n_devices`."""
    if xs.shape[0] % n_devices != 0:
        pad_size = n_devices - xs.shape[0] % n_devices
        xs_pad = einops.repeat(
            xs[-1:], "n ... -> (pad n) ...", pad=pad_size
        )
        xs = jnp.concatenate([xs, xs_pad])
    X_padded = xs.reshape(n_devices, -1, *xs.shape[1:])
    return X_padded


# %% ../nbs/03_explain.strategy.ipynb 8
class PmapStrategy(BaseStrategy):
    def __init__(
        self, 
        n_devices: int = None, # Number of devices. If None, use all available devices
        strategy: str = 'auto', # Strategy to generate counterfactuals
        **kwargs
    ):
        self.strategy = strategy
        self.n_devices = n_devices or jax.device_count()

    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        
        def partial_fn(x, y_target, rng_key, **kwargs):
            return fn(x, pred_fn=pred_fn, y_target=y_target, rng_key=rng_key, **kwargs)

        assert xs.ndim == 2
        X_padded = _pad_divisible_X(xs, self.n_devices)
        y_targets = _pad_divisible_X(y_targets, self.n_devices)
        rng_keys = _pad_divisible_X(rng_keys, self.n_devices)
        cfs = jax.pmap(jax.vmap(partial_fn))(X_padded, y_targets, rng_keys)
        cfs = cfs.reshape(-1, *cfs.shape[2:])
        cfs = cfs[:xs.shape[0]]
        return cfs


# %% ../nbs/03_explain.strategy.ipynb 9
def _pad_xs(
    xs: Array, pad_size: int, batch_size: int
):
    """Pad `X` to be divisible by `n_devices`."""
    xs_pad = einops.repeat(
        xs[-1:], "n ... -> (pad n) ...", pad=pad_size
    )
    xs = jnp.concatenate([xs, xs_pad])
    xs = einops.rearrange(xs, "(b n) ... -> b n ...", b=batch_size)
    return xs

def _batched_generation(
    gs_fn: Callable, # Generation strategy function
    cf_fn: Callable, # Function to generate cf for a single input
    xs: Array, # Input instances to be explained
    pred_fn: Callable[[Array], Array],
    y_targets: Array,
    rng_keys: Iterable[jrand.PRNGKey],
    batch_size: int,
    **kwargs
) -> Array: # Generated counterfactual explanations
    """Batched  of counterfactuals."""

    def gs_fn_partial(state):
        x, y_target, rng_key = state
        return gs_fn(cf_fn, x, pred_fn, y_target, rng_key, **kwargs)
    
    assert xs.ndim == 2, f"X must be a 2D array, got {xs.ndim}D array"
    x_shape = xs.shape
    batch_size = min(batch_size, x_shape[0])
    # pad X to be divisible by batch_size
    pad_size = batch_size - (xs.shape[0] % batch_size)
    xs = _pad_xs(xs, pad_size, batch_size)
    y_targets = _pad_xs(y_targets, pad_size, batch_size)
    rng_keys = _pad_xs(rng_keys, pad_size, batch_size)
    # generate cfs via lax.map
    cfs = lax.map(gs_fn_partial, (xs, y_targets, rng_keys))
    # cfs = cfs.reshape(-1, *x_shape[1:])[:x_shape[0]]
    cfs = einops.rearrange(cfs, "n b ... k -> (n b) ... k")
    cfs = cfs[:x_shape[0]]
    return cfs

# %% ../nbs/03_explain.strategy.ipynb 10
class BatchedVmapStrategy(BaseStrategy):
    """Auto-batching for generate counterfactuals via `jax.vmap`."""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        vmap_g = VmapStrategy()    
        cfs = _batched_generation(
            vmap_g, fn, xs, pred_fn, y_targets, rng_keys, self.batch_size, **kwargs
        )
        return cfs


# %% ../nbs/03_explain.strategy.ipynb 11
class BatchedPmapStrategy(BaseStrategy):
    """Auto-batching for generate counterfactuals via `jax.vmap`."""
    def __init__(self, batch_size: int, n_devices: int = None):
        self.batch_size = batch_size
        self.n_devices = n_devices

    def __call__(
        self, 
        fn: Callable, # Function to generate cf for a single input
        xs: Array, # Input instances to be explained
        pred_fn: Callable[[Array], Array],
        y_targets: Array,
        rng_keys: Iterable[jrand.PRNGKey],
        **kwargs
    ) -> Array: # Generated counterfactual explanations
        pmap_g = PmapStrategy(self.n_devices)
        cfs = _batched_generation(
            pmap_g, fn, xs, pred_fn, y_targets, rng_keys, self.batch_size, **kwargs
        )
        return cfs


# %% ../nbs/03_explain.strategy.ipynb 22
class StrategyFactory(object):
    """Factory class for Parallelism Strategy."""

    __strategy_map = {
        'iter': IterativeStrategy(),
        'vmap': VmapStrategy(),
        'pmap': PmapStrategy(),
    }

    def __init__(self) -> None:
        raise ValueError("This class should not be instantiated.")
        
    @staticmethod
    def get_default_strategy() -> BaseStrategy:
        """Get default strategy."""
        return VmapStrategy()

    @classmethod
    def get_strategy(cls, strategy: str | BaseStrategy) -> BaseStrategy:
        """Get strategy."""
        if isinstance(strategy, BaseStrategy):
            return strategy
        elif isinstance(strategy, str) and strategy in cls.__strategy_map:
            return cls.__strategy_map[strategy]
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
    __ALL__ = ["get_default_strategy", "get_strategy"]
