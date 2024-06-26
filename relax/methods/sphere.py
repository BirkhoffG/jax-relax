# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/methods/05_sphere.ipynb.

# %% ../../nbs/methods/05_sphere.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from .base import CFModule, BaseConfig, default_apply_constraints_fn
from ..utils import auto_reshaping, grad_update, validate_configs
from ..data_utils import Feature, FeaturesList
from ..data_module import DataModule

# %% auto 0
__all__ = ['hyper_sphere_coordindates', 'sample_categorical', 'default_perturb_function', 'perturb_function_with_features',
           'GSConfig', 'GrowingSphere']

# %% ../../nbs/methods/05_sphere.ipynb 5
@partial(jit, static_argnums=(2, 5))
def hyper_sphere_coordindates(
    rng_key: jrand.PRNGKey, # Random number generator key
    x: Array, # Input instance with only continuous features. Shape: (1, n_features)
    n_samples: int, # Number of samples
    high: float, # Upper bound
    low: float, # Lower bound
    p_norm: int = 2 # Norm
):
    # Adapted from 
    # https://github.com/carla-recourse/CARLA/blob/24db00aa8616eb2faedea0d6edf6e307cee9d192/carla/recourse_methods/catalog/growing_spheres/library/gs_counterfactuals.py#L8
    key_1, key_2 = jrand.split(rng_key)
    delta = jrand.normal(key_1, shape=(n_samples, x.shape[-1]))
    dist = jrand.uniform(key_2, shape=(n_samples,)) * (high - low) + low
    norm_p = jnp.linalg.norm(delta, ord=p_norm, axis=1)
    d_norm = jnp.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta = jnp.multiply(delta, d_norm)
    candidates = x + delta

    return candidates

# %% ../../nbs/methods/05_sphere.ipynb 6
def sample_categorical(rng_key: jrand.PRNGKey, col_size: int, n_samples: int):
    rng_key, _ = jrand.split(rng_key)
    prob = jnp.ones(col_size) / col_size
    cat_sample = jrand.categorical(rng_key, prob, shape=(n_samples, 1))
    return cat_sample

# %% ../../nbs/methods/05_sphere.ipynb 7
def default_perturb_function(
    rng_key: jrand.PRNGKey,
    x: np.ndarray, # Shape: (1, k)
    n_samples: int,
    high: float,
    low: float,
    p_norm: int
):
    return hyper_sphere_coordindates(
        rng_key, x, n_samples, high, low, p_norm
    )

# def perturb_function_with_features(
#     rng_key: jrand.PRNGKey,
#     x: np.ndarray, # Shape: (1, k)
#     n_samples: int,
#     high, 
#     low,
#     p_norm,
#     feats: FeaturesList,
# ):
#     def perturb_feature(rng_key, x, feat):
#         if feat.is_categorical:
#             sampled_cat = sample_categorical(
#                 rng_key, feat.transformation.num_categories, n_samples
#             ) #<== sampled labels
#             transformation = feat.transformation.name
#             if transformation == 'ohe':
#                 return jax.nn.one_hot(
#                     sampled_cat.reshape(-1), num_classes=feat.transformation.num_categories
#                 ) #<== transformed labels
#             elif transformation == 'ordinal':
#                 return sampled_cat
#             else:
#                 raise NotImplementedError
#         else: 
#             return hyper_sphere_coordindates(
#                 rng_key, x, n_samples, high, low, p_norm
#             ) #<== transformed continuous features
        
#     rng_keys = jrand.split(rng_key, len(feats))
#     perturbed = jnp.repeat(x, n_samples, axis=0)
#     for rng_key, (start, end), feat in zip(rng_keys, feats.feature_indices, feats):
#         _perturbed_feat = perturb_feature(rng_keys[0], x[:, start: end], feat)
#         perturbed = perturbed.at[:, start: end].set(_perturbed_feat)
#     return perturbed


# %% ../../nbs/methods/05_sphere.ipynb 8
@partial(jit, static_argnums=(2, 5, 8, 9))
def perturb_function_with_features(
    rng_key: jrand.PRNGKey,
    x: np.ndarray, # Shape: (1, k)
    n_samples: int,
    high: float, 
    low: float,
    p_norm: int,
    cont_masks: Array,
    immut_masks: Array,
    num_categories: list[int],
    cat_perturb_fn: Callable
):
        
    def perturb_cat_feat(rng_key, num_categories):
        rng_key, next_key = jrand.split(rng_key)
        sampled = cat_perturb_fn(rng_key, num_categories, n_samples)
        return next_key, sampled
    
    # cont_masks, immut_masks, num_categories = feats_info
    key_1, key_2 = jrand.split(rng_key)
    perturbed_cont = cont_masks * hyper_sphere_coordindates(
        key_1, x, n_samples, high, low, p_norm
    )
    cat_masks = jnp.where(cont_masks, 0, 1)
    perturbed_cat = cat_masks * jnp.concatenate([
        perturb_cat_feat(key_2, num_cat)[1] for num_cat in num_categories
    ], axis=1)

    perturbed = jnp.where(
        immut_masks,
        jnp.repeat(x, n_samples, axis=0),
        perturbed_cont + perturbed_cat
    )
    
    return perturbed

# %% ../../nbs/methods/05_sphere.ipynb 9
def features_to_infos_and_perturb_fn(
    features: FeaturesList
) -> Tuple[List[Array,Array,Array,Array,Array], Callable]:
    cont_masks = []
    immut_masks = []
    n_categories = []
    cat_transformation_name = None
    for (start, end), feat in zip(features.feature_indices, features):
        if feat.is_categorical:
            cont_mask = jnp.zeros(feat.transformation.num_categories)
            immut_mask = jnp.ones_like(cont_mask) * np.array([feat.is_immutable], dtype=np.int32)
            n_categorie = feat.transformation.num_categories
            cat_transformation_name = feat.transformation.name
        else:
            cont_mask = jnp.ones(1)
            immut_mask = cont_mask * np.array([feat.is_immutable], dtype=np.int32)
            n_categorie = 1
        
        cont_masks, immut_masks, n_categories = map(lambda x, y: x + [y], 
            [cont_masks, immut_masks, n_categories],
            [cont_mask, immut_mask, n_categorie]
        )
    
    cont_masks, immut_masks = map(lambda x: jnp.concatenate(x, axis=0), [cont_masks, immut_masks])
    return (cont_masks, immut_masks, tuple(n_categories)), cat_perturb_fn(cat_transformation_name)

def cat_perturb_fn(transformation):
    def ohe_perturb_fn(rng_key, num_categories, n_samples):
        sampled_cat = sample_categorical(rng_key, num_categories, n_samples)
        return jax.nn.one_hot(
            sampled_cat.reshape(-1), num_classes=num_categories
        )
    
    def ordinal_perturb_fn(rng_key, num_categories, n_samples):
        return sample_categorical(
            rng_key, num_categories, n_samples
        )
    
    if transformation == 'ohe':         return ohe_perturb_fn
    elif transformation == 'ordinal':   return ordinal_perturb_fn
    else:                               return sample_categorical


# %% ../../nbs/methods/05_sphere.ipynb 11
@ft.partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _growing_spheres(
    rng_key: jrand.PRNGKey, # Random number generator key
    y_target: Array, # Target label
    x: Array, # Input instance. Shape: (n_features)
    pred_fn: Callable, # Prediction function
    n_steps: int, # Number of steps
    n_samples: int,  # Number of samples to sample
    step_size: float, # Step size
    p_norm: int, # Norm
    perturb_fn: Callable, # Perturbation function
    apply_constraints_fn: Callable, # Apply immutable constraints
    dtype: jnp.dtype = jnp.float32, # Data type
): 
    @jit
    def dist_fn(x, cf):
        if p_norm == 1:
            return jnp.abs(cf - x).sum(axis=1)
        elif p_norm == 2:
            return jnp.linalg.norm(cf - x, ord=2, axis=1)
        else:
            raise ValueError("Only p_norm = 1 or 2 is supported")
    
    @loop_tqdm(n_steps)
    def step(i, state):
        candidate_cf, count, rng_key = state
        rng_key, subkey = jrand.split(rng_key)
        low, high = step_size * count, step_size * (count + 1)
        # Sample around x
        candidates = perturb_fn(rng_key, x, n_samples, high=high, low=low, p_norm=p_norm)
        
        # Apply immutable constraints
        candidates = apply_constraints_fn(x, candidates, hard=True)
        # assert candidates.shape[1] == x.shape[1], f"candidates.shape = {candidates.shape}, x.shape = {x.shape}"

        # Calculate distance
        dist = dist_fn(x, candidates)

        # Calculate counterfactual labels
        candidate_preds = pred_fn(candidates).argmax(axis=1, keepdims=True)
        indices = candidate_preds == y_target

        # Select valid candidates and their distances
        candidates, dist = jax.tree_util.tree_map(
            lambda x: jnp.where(indices, x, jnp.ones_like(x) * jnp.inf), 
            (candidates, dist)
        )

        closest_idx = dist.argmin()
        candidate_cf_update = candidates[closest_idx].reshape(1, -1)

        candidate_cf = jnp.where(
            dist[closest_idx].mean() < dist_fn(x, candidate_cf).mean(),
            candidate_cf_update, 
            candidate_cf
        )
        return candidate_cf, count + 1, subkey
    
    y_target = y_target.reshape(1, -1).argmax(axis=1)
    candidate_cf = jnp.ones_like(x) * jnp.inf
    count = 0
    state = (candidate_cf, count, rng_key)
    candidate_cf, _, _ = lax.fori_loop(0, n_steps, step, state)
    # if `inf` is found, return the original input
    candidate_cf = jnp.where(jnp.isinf(candidate_cf), x, candidate_cf)
    return candidate_cf

# %% ../../nbs/methods/05_sphere.ipynb 12
class GSConfig(BaseConfig):
    n_steps: int = 100
    n_samples: int = 100
    step_size: float = 0.05
    p_norm: int = 2


# %% ../../nbs/methods/05_sphere.ipynb 13
class GrowingSphere(CFModule):
    def __init__(self, config: dict | GSConfig = None, *, name: str = None, perturb_fn = None):
        if config is None:
             config = GSConfig()
        config = validate_configs(config, GSConfig)
        name = "GrowingSphere" if name is None else name
        self.perturb_fn = perturb_fn
        super().__init__(config, name=name)

    def has_data_module(self):
        return hasattr(self, 'data_module') and self.data_module is not None
    
    def save(self, path: str, *, save_data_module: bool = True):
        self.config.save(Path(path) / 'config.json')
        if self.has_data_module() and save_data_module:
            self.data_module.save(Path(path) / 'data_module')
    
    @classmethod
    def load_from_path(cls, path: str):
        config = GSConfig.load_from_json(Path(path) / 'config.json')
        gs = cls(config=config)
        if (Path(path) / 'data_module').exists():
            dm = DataModule.load_from_path(Path(path) / 'data_module')
            gs.set_data_module(dm)
        return gs

    def before_generate_cf(self, *args, **kwargs):
        if self.perturb_fn is None:
            if self.has_data_module():
                feats_info, perturb_fn = features_to_infos_and_perturb_fn(self.data_module.features)
                cont_masks, immut_masks, num_categories = feats_info
                self.perturb_fn = ft.partial(
                    perturb_function_with_features, 
                    cont_masks=cont_masks,
                    immut_masks=immut_masks,
                    num_categories=num_categories,
                    cat_perturb_fn=perturb_fn
                )
                # self.apply_constraints = default_apply_constraints_fn
            else:
                self.perturb_fn = default_perturb_function
        
    @auto_reshaping('x')
    def generate_cf(
        self,
        x: Array,  # `x` shape: (k,), where `k` is the number of features
        pred_fn: Callable[[Array], Array],
        y_target: Array = None,
        rng_key: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.DeviceArray:
        # TODO: Currently assumes binary classification.
        if y_target is None:
            y_target = 1 - pred_fn(x)
        else:
            y_target = y_target.reshape(1, -1)
        if rng_key is None:
            raise ValueError("`rng_key` must be provided, but got `None`.")
        
        return _growing_spheres(
            rng_key=rng_key,
            x=x,
            y_target=y_target,
            pred_fn=pred_fn,
            n_steps=self.config.n_steps,
            n_samples=self.config.n_samples,
            step_size=self.config.step_size,
            p_norm=self.config.p_norm,
            perturb_fn=self.perturb_fn,
            apply_constraints_fn=self.apply_constraints,
        )
