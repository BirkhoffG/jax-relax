# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/data_utils/transform.ipynb.

# %% ../../nbs/data_utils/transform.ipynb 3
from __future__ import annotations
from .preprocessing import *
from ..utils import get_config, gumbel_softmax
from ..import_essentials import *

# %% auto 0
__all__ = ['FEATURE_TRANSFORMATIONS', 'BaseTransformation', 'MinMaxTransformation', 'SoftmaxTransformation',
           'GumbelSoftmaxTransformation', 'OneHotTransformation', 'OrdinalTransformation', 'IdentityTransformation']

# %% ../../nbs/data_utils/transform.ipynb 5
class BaseTransformation:
    """Base class for all transformations."""
    
    def __init__(self, name, transformer: DataPreprocessor = None):
        self.name = name
        self.transformer = transformer

    @property
    def is_categorical(self) -> bool:   raise NotImplementedError

    def fit(self, xs, y=None):          raise NotImplementedError

    def transform(self, xs):            raise NotImplementedError

    def fit_transform(self, xs, y=None):raise NotImplementedError

    def inverse_transform(self, xs):    raise NotImplementedError

    def apply_constraints(self, xs, cfs, hard, rng_key, **kwargs):
        raise NotImplementedError
    
    def compute_reg_loss(self, xs, cfs, hard: bool = False):
        raise NotImplementedError

    def from_dict(self, params):        raise NotImplementedError    

    def to_dict(self):                  raise NotImplementedError

# %% ../../nbs/data_utils/transform.ipynb 6
class _DefaultTransformation(BaseTransformation):

    @property
    def is_categorical(self) -> bool:
        if self.transformer is None:
            return False
        return isinstance(self.transformer, EncoderPreprocessor)

    def fit(self, xs, y=None):
        if self.transformer is not None:
            self.transformer.fit(xs)
        return self
    
    def transform(self, xs):
        if self.transformer is None:
            return xs
        return self.transformer.transform(xs)

    def fit_transform(self, xs, y=None):
        if self.transformer is None:
            return xs
        return self.transformer.fit_transform(xs)
    
    def inverse_transform(self, xs):
        if self.transformer is None:
            return xs
        return self.transformer.inverse_transform(xs)

    def apply_constraints(self, xs: jax.Array, cfs: jax.Array, hard: bool = False, 
                          rng_key: jrand.PRNGKey = None, **kwargs):
        return cfs
    
    def compute_reg_loss(self, xs, cfs, hard: bool = False):
        return 0.
    
    def from_dict(self, params: dict):
        self.name = params["name"]
        if not 'transformer' in params.keys():
            self.transformer = None
        else:
            self.transformer.from_dict(params["transformer"])
        return self
    
    def to_dict(self) -> dict:
        return {"name": self.name, "transformer": self.transformer.to_dict()}

# %% ../../nbs/data_utils/transform.ipynb 7
class MinMaxTransformation(_DefaultTransformation):
    def __init__(self):
        super().__init__("minmax", MinMaxScaler())

    def apply_constraints(self, xs, cfs, **kwargs):
        return jnp.clip(cfs, 0., 1.)

# %% ../../nbs/data_utils/transform.ipynb 9
class _OneHotTransformation(_DefaultTransformation):
    def __init__(self, name: str = None):
        super().__init__(name, OneHotEncoder())

    @property
    def num_categories(self) -> int:
        return len(self.transformer.categories_)
    
    def hard_constraints(self, operand: tuple[jax.Array, jrand.PRNGKey, dict]): 
        x, rng_key, kwargs = operand
        return jax.nn.one_hot(jnp.argmax(x, axis=-1), self.num_categories)
    
    def soft_constraints(self, operand: tuple[jax.Array, jrand.PRNGKey, dict]):
        raise NotImplementedError

    def apply_constraints(self, xs, cfs, hard: bool = False, rng_key=None, **kwargs):
        return jax.lax.cond(
            hard,
            true_fun=self.hard_constraints,
            false_fun=self.soft_constraints,
            operand=(cfs, rng_key, kwargs),
        )
    
    def compute_reg_loss(self, xs, cfs, hard: bool = False):
        reg_loss_per_xs = (cfs.sum(axis=-1, keepdims=True) - 1.0) ** 2
        return reg_loss_per_xs.mean()

# %% ../../nbs/data_utils/transform.ipynb 10
class SoftmaxTransformation(_OneHotTransformation):
    def __init__(self): 
        super().__init__("ohe")

    def soft_constraints(self, operand: tuple[jax.Array, jrand.PRNGKey, dict]):
        x, rng_key, kwargs = operand
        return jax.nn.softmax(x, axis=-1)
    
class GumbelSoftmaxTransformation(_OneHotTransformation):
    """Apply Gumbel softmax tricks for categorical transformation."""

    def __init__(self, tau: float = .1):
        super().__init__("gumbel")
        self.tau = tau
    
    def soft_constraints(self, operand: tuple[jax.Array, jrand.PRNGKey, dict]):
        x, rng_key, _ = operand
        if rng_key is None: # No randomness
            rng_key = jrand.PRNGKey(get_config().global_seed)
        return gumbel_softmax(rng_key, x, self.tau)
    
    def apply_constraints(self, xs, cfs, hard: bool = False, rng_key=None, **kwargs):
        """Apply constraints to the counterfactuals. If `rng_key` is None, no randomness is used."""
        return super().apply_constraints(xs, cfs, hard, rng_key, **kwargs)
    
    def to_dict(self) -> dict:
        return super().to_dict() | {"tau": self.tau}
    
def OneHotTransformation():
    warnings.warn("OneHotTransformation is deprecated since v0.2.5. "
                  "Use `SoftmaxTransformation`.", DeprecationWarning)
    return SoftmaxTransformation()

# %% ../../nbs/data_utils/transform.ipynb 12
class OrdinalTransformation(_DefaultTransformation):
    def __init__(self):
        super().__init__("ordinal", OrdinalPreprocessor())

    @property
    def num_categories(self) -> int:
        return len(self.transformer.categories_)
    
class IdentityTransformation(_DefaultTransformation):
    def __init__(self):
        super().__init__("identity", None)

    def fit(self, xs, y=None):
        return self
    
    def transform(self, xs):
        return xs
    
    def fit_transform(self, xs, y=None):
        return xs

    def apply_constraints(self, xs, cfs, **kwargs):
        return cfs
    
    def to_dict(self):
        return {'name': 'identity'}
    
    def from_dict(self, params: dict):
        self.name = params["name"]
        return self

# %% ../../nbs/data_utils/transform.ipynb 14
FEATURE_TRANSFORMATIONS = {
    'ohe': SoftmaxTransformation,
    'softmax': SoftmaxTransformation,
    'gumbel': GumbelSoftmaxTransformation,
    'minmax': MinMaxTransformation,
    'ordinal': OrdinalTransformation,
    'identity': IdentityTransformation,
}
