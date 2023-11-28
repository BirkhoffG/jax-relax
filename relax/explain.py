# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_explain.ipynb.

# %% ../nbs/03_explain.ipynb 2
from __future__ import annotations
from .import_essentials import *
from .data_module import DataModule, load_data
from .base import *
from .methods import *
from .strategy import *
from .ml_model import *
from .utils import get_config, save_pytree, load_pytree
import einops
from sklearn.datasets import make_classification

# %% auto 0
__all__ = ['Explanation', 'fake_explanation', 'prepare_pred_fn', 'prepare_cf_module', 'prepare_rng_keys',
           'generate_cf_explanations']

# %% ../nbs/03_explain.ipynb 4
class Explanation(DataModule):
    """Generated CF Explanations class. It inherits a `DataModule`."""

    def __init__(
        self,
        cfs: Array,  # Generated cf explanation of `xs` in `data`
        pred_fn: Callable[[Array], Array],  # Predict function
        data_module: DataModule = None,  # Data module
        xs: Array = None,  # Input data
        ys: Array = None,  # Target data
        total_time: float = None,  # Total runtime
        cf_name: str = "CFModule",  # CF method's name
        data=None, # Deprecated argument
    ):
        if data is not None:
            warnings.warn(
                "Argument `data` is deprecated. Use `data_module` instead.",
                DeprecationWarning,
            )
            data_module = data

        if (xs is None or ys is None) and data_module is None:
            raise ValueError(
                "Either `xs` and `ys` or `data_module` must be provided."
            )
        
        if data_module is None:
            data_module = DataModule.from_numpy(xs, ys, transformation='identity')
        # assign attributes
        # self.recourses = data_module.features.with_transformed_data(cfs)
        self._cfs = cfs
        self.pred_fn = pred_fn
        self.total_time = total_time
        self.cf_name = cf_name
        
        super().__init__(
            features=data_module.features, 
            label=data_module.label,
            config=data_module.config,
            data=data_module.data,
        )

    def __repr__(self):
        return f"Explanation(data_name={self.data_name}, cf_name={self.cf_name}, " \
               f"total_time={self.total_time}, xs={self.xs}, ys={self.ys}, cfs={self.cfs})"

    def __getitem__(self, name: Literal['train', 'val', 'test']) -> Dict[str, Array]:
        if name == 'train':
            indices = self.train_indices
        elif name in ['val', 'test']:
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown data name: {name}. Should be one of ['train', 'val', 'test']")

        if isinstance(indices, list):
            indices = jnp.array(indices)
        
        return {
            'xs': self.xs[indices],
            'ys': self.ys[indices],
            'cfs': self.cfs[indices],
        }    

    @property
    def cfs(self) -> Array:
        """Return the counterfactuals in the shape of (n, c, k)"""
        if self._cfs.ndim == 2:
            return einops.rearrange(self._cfs, "n d -> n () d")
        return self._cfs
    
    @property
    def data_name(self):
        return self.name
    
    @property
    def feature_indices(self):
        return self.features.feature_indices
    
    @property
    def features_and_indices(self):
        return self.features.features_and_indices
        
    def save(self, path: str):
        """Save the explanation to a directory."""
        # create directories
        dm_path = Path(path) / 'data'
        exp_path = Path(path) / 'explanations'
        exp_path.mkdir(parents=True, exist_ok=True)
        # save data module and explanations
        super().save(dm_path)        
        save_pytree({
            'cfs': self.cfs,
            'total_time': self.total_time,
            'cf_name': self.cf_name,
        }, exp_path)
    
    @classmethod
    def load_from_path(cls, path: str, *, ml_module_path: str = None):
        dm_path = Path(path) / 'data'
        exp_path = Path(path) / 'explanations'
        dm = DataModule.load_from_path(dm_path)
        explanations = load_pytree(exp_path)
        if ml_module_path is not None:
            pred_fn = MLModule.load_from_path(ml_module_path).pred_fn
        else:
            warnings.warn("`ml_module_path` is not provided. Setting `pred_fn=None`.")
            pred_fn = None
        return cls(
            pred_fn=pred_fn,
            data_module=dm,
            **explanations
        )


# %% ../nbs/03_explain.ipynb 5
def fake_explanation(n_cfs: int=1):
    dm = load_data('dummy')
    ml_model = load_ml_module('dummy')
    if n_cfs < 1: 
        raise ValueError(f'n_cfs must be greater than 0, but got n_cfs={n_cfs}.')
    elif n_cfs == 1:
        cfs = dm.xs
    else:
        # Allow for multiple counterfactuals
        cfs = einops.repeat(dm.xs, "n k -> n c k", c=n_cfs)

    return Explanation(
        data_module=dm, cfs=cfs, pred_fn=ml_model.pred_fn, total_time=0.0, cf_name='dummy_method'
    )

# %% ../nbs/03_explain.ipynb 9
def prepare_pred_fn(
    cf_module: CFModule,
    data: DataModule,
    pred_fn: Callable[[Array, ...], Array], # Predictive function. 
    pred_fn_args: Dict = None,
) -> Callable[[Array], Array]: # Return predictive function with signature `(x: Array) -> Array`.
    """Prepare the predictive function for the CF module. 
    We will train the model if `pred_fn` is not provided and `cf_module` does not have `pred_fn`.
    If `pred_fn` is found in `cf_module`, we will use it irrespective of `pred_fn` argument.
    If `pred_fn` is provided, we will use it.
    """
    # Train the model if `pred_fn` is not provided.
    if not hasattr(cf_module, 'pred_fn') and pred_fn is None:
        model = MLModule().train(data)
        return model.pred_fn
    # If `pred_fn` is detected in cf_module, 
    # use it irrespective of `pred_fn` argument.
    elif hasattr(cf_module, 'pred_fn'):
        return cf_module.pred_fn
    # If `pred_fn` is provided, use it.
    else:
        if pred_fn_args is not None:
            pred_fn = ft.partial(pred_fn, **pred_fn_args)
        return pred_fn

def prepare_cf_module(
    cf_module: CFModule,
    data_module: DataModule,
    pred_fn: Callable[[Array], Array] = None,
    train_config: Dict[str, Any] = None, 
):
    """Prepare the CF module. 
    It will hook up the data module, 
    and its apply functions via the `init_apply_fns` method
    (e.g., `apply_constraints_fn` and `compute_reg_loss_fn`).
    Next, it will train the model if `cf_module` is a `ParametricCFModule`.
    Finally, it will call `before_generate_cf` method.
    """
    cf_module.set_data_module(data_module)
    cf_module.set_apply_constraints_fn(data_module.apply_constraints)
    cf_module.set_compute_reg_loss_fn(data_module.compute_reg_loss)
    train_config = train_config or {}
    if isinstance(cf_module, ParametricCFModule):
        if not cf_module.is_trained:
            cf_module.train(data_module, pred_fn=pred_fn, **train_config)
    cf_module.before_generate_cf()
    return cf_module

def prepare_rng_keys(
    rng_key: jrand.PRNGKey,
    n_instances: int,
):
    """Prepare random number generator keys."""
    if rng_key is None:
        rng_key = jrand.PRNGKey(get_config().global_seed)
    rng_keys = jrand.split(rng_key, n_instances)
    return rng_keys


# %% ../nbs/03_explain.ipynb 10
def generate_cf_explanations(
    cf_module: CFModule, # CF Explanation Module
    data: DataModule, # Data Module
    pred_fn: Callable[[Array, ...], Array] = None, # Predictive function
    strategy: str | BaseStrategy = None, # Parallelism Strategy for generating CFs. Default to `vmap`.
    train_config: Dict[str, Any] = None, 
    pred_fn_args: dict = None, # auxiliary arguments for `pred_fn` 
    rng_key: jrand.PRNGKey = None, # Random number generator key
) -> Explanation: # Return counterfactual explanations.
    """Generate CF explanations."""

    # Prepare `pred_fn`, `cf_module`, and `strategy`.
    pred_fn = prepare_pred_fn(cf_module, data, pred_fn, pred_fn_args)
    cf_module = prepare_cf_module(cf_module, data, pred_fn, train_config)
    if strategy is None:
        strategy = StrategyFactory.get_default_strategy()
    strategy = StrategyFactory.get_strategy(strategy)
    # n_instances
    n_instances = data.xs.shape[0]
    # Prepare random number generator keys.
    rng_keys = prepare_rng_keys(rng_key, n_instances)
    y_targets = 1 - pred_fn(data.xs)
    
    # Generate CF explanations.
    start_time = time.time()
    cfs = strategy(cf_module.generate_cf, data.xs, pred_fn, y_targets, rng_keys)
    # cfs = jax.vmap(cf_module.generate_cf, in_axes=(0, None, 0, 0))(data.xs, pred_fn, y_targets, rng_keys)
    total_time = time.time() - start_time

    # Return CF explanations.
    return Explanation(
        cf_name=cf_module.name,
        data=data,
        cfs=cfs,
        total_time=total_time,
        pred_fn=pred_fn,
    )
