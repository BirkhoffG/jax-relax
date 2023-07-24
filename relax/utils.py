# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% ../nbs/00_utils.ipynb 3
from __future__ import annotations
from .import_essentials import *
import nbdev
from fastcore.basics import AttrDict
from nbdev.showdoc import BasicMarkdownRenderer
from inspect import isclass
from fastcore.test import *
from jax.core import InconclusiveDimensionOperation

# %% auto 0
__all__ = ['validate_configs', 'auto_reshaping', 'grad_update', 'load_json', 'get_config']

# %% ../nbs/00_utils.ipynb 5
def validate_configs(
    configs: dict | BaseParser,  # A configuration of the model/dataset.
    config_cls: BaseParser,  # The desired configuration class.
) -> BaseParser:
    """return a valid configuration object."""

    assert isclass(config_cls), f"`config_cls` should be a class."
    assert issubclass(config_cls, BaseParser), \
        f"{config_cls} should be a subclass of `BaseParser`."
    
    if isinstance(configs, dict):
        configs = config_cls(**configs)
    if not isinstance(configs, config_cls):
        raise TypeError(
            f"configs should be either a `dict` or an instance of {config_cls.__name__}.")
    return configs

# %% ../nbs/00_utils.ipynb 15
def _reshape_x(x: Array):
    x_size = x.shape
    if len(x_size) > 1 and x_size[0] != 1:
        raise ValueError(
            f"""Invalid Input Shape: Require `x.shape` = (1, k) or (k, ),
but got `x.shape` = {x.shape}. This method expects a single input instance."""
        )
    if len(x_size) == 1:
        x = x.reshape(1, -1)
    return x, x_size

# %% ../nbs/00_utils.ipynb 16
def auto_reshaping(reshape_argname: str):
    """
    Decorator to automatically reshape function's input into (1, k), 
    and out to input's shape.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            kwargs = inspect.getcallargs(func, *args, **kwargs)
            if reshape_argname in kwargs:
                reshaped_x, x_shape = _reshape_x(kwargs[reshape_argname])
                kwargs[reshape_argname] = reshaped_x
            else:
                raise ValueError(
                    f"Invalid argument name: `{reshape_argname}` is not a valid argument name.")
            cf = func(**kwargs)
            if not isinstance(cf, Array): 
                raise ValueError(
                    f"Invalid return type: must be a `jax.Array`, but got `{type(cf).__name__}`.")
            try: 
                cf = cf.reshape(x_shape)
            except InconclusiveDimensionOperation:
                raise ValueError(
                    f"Invalid return shape: Require `cf.shape` = {cf.shape} "
                    f"is not compatible with `x.shape` = {x_shape}.")
            return cf

        return wrapper
    return decorator

# %% ../nbs/00_utils.ipynb 21
def grad_update(
    grads, # A pytree of gradients.
    params, # A pytree of parameters.
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
): # Return (upt_params, upt_opt_state)
    updates, opt_state = opt.update(grads, opt_state, params)
    upt_params = optax.apply_updates(params, updates)
    return upt_params, opt_state

# %% ../nbs/00_utils.ipynb 23
def load_json(f_name: str) -> Dict[str, Any]:  # file name
    with open(f_name) as f:
        return json.load(f)


# %% ../nbs/00_utils.ipynb 25
@dataclass
class Config:
    rng_reserve_size: int
    global_seed: int

    @classmethod
    def default(cls) -> Config:
        return cls(rng_reserve_size=1, global_seed=42)

main_config = Config.default()

# %% ../nbs/00_utils.ipynb 26
def get_config() -> Config: 
    return main_config
