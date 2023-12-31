# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_evaluate.ipynb.

# %% ../nbs/04_evaluate.ipynb 3
from __future__ import annotations

from .import_essentials import *
from .base import *
from .explain import *
from keras.metrics import sparse_categorical_accuracy
import einops

# %% auto 0
__all__ = ['BaseEvalMetrics', 'PredictiveAccuracy', 'compute_single_validity', 'compute_validity', 'Validity',
           'compute_single_proximity', 'compute_proximity', 'Proximity', 'compute_single_sparsity', 'compute_sparsity',
           'Sparsity', 'ManifoldDist', 'Runtime', 'evaluate_cfs', 'benchmark_cfs']

# %% ../nbs/04_evaluate.ipynb 6
class BaseEvalMetrics:
    """Base evaluation metrics class."""

    def __init__(self, name: str = None):
        if name is None: 
            name = type(self).__name__
        self.name = name

    def __str__(self) -> str:
        has_name = hasattr(self, 'name')
        if not has_name:
            raise ValidationError(
                "EvalMetrics must have a name. Add the following as the first line in your "
                f"__init__ method:\n\nsuper({self.__name__}, self).__init__()")
        return self.name

    def __call__(self, explanation: Explanation) -> Any:
        raise NotImplementedError


# %% ../nbs/04_evaluate.ipynb 7
class PredictiveAccuracy(BaseEvalMetrics):
    """Compute the accuracy of the predict function."""
    
    def __init__(self, name: str = "accuracy"):
        super().__init__(name=name)

    def __call__(self, explanation: Explanation) -> float:
        xs, ys = explanation.xs, explanation.ys
        pred_fn = explanation.pred_fn
        pred_ys = pred_fn(xs)
        accuracy = sparse_categorical_accuracy(ys, pred_ys)
        return accuracy.mean()

# %% ../nbs/04_evaluate.ipynb 9
def compute_single_validity(
    xs: Array, # (n, d)
    cfs: Array, # (n, d)
    pred_fn: Callable[[Array], Array],
):
    y_xs = pred_fn(xs).argmax(axis=-1)
    y_cfs = pred_fn(cfs).argmax(axis=-1)
    validity = 1 - jnp.equal(y_xs, y_cfs).mean()
    return validity

def compute_validity(
    xs: Array, # (n, d)
    cfs: Array, # (n, d) or (n, b, d)
    pred_fn: Callable[[Array], Array],
) -> float:
    cfs = einops.rearrange(cfs, 'n ... d -> n (...) d')
    valdity_batch = jax.vmap(compute_single_validity, in_axes=(None, 1, None))(xs, cfs, pred_fn)
    return valdity_batch.mean()

# %% ../nbs/04_evaluate.ipynb 11
class Validity(BaseEvalMetrics):
    """Compute fraction of input instances on which CF explanation methods output valid CF examples.
    Support binary case only.
    """
    
    def __init__(self, name: str = "validity"):
        super().__init__(name=name)

    def __call__(self, explanation: Explanation) -> float:
        xs, cfs, pred_fn = explanation.xs, explanation.cfs, explanation.pred_fn
        return compute_validity(xs, cfs, pred_fn)

# %% ../nbs/04_evaluate.ipynb 13
def compute_single_proximity(xs: Array, cfs: Array):
    prox = jnp.linalg.norm(xs - cfs, ord=1, axis=1).mean()
    return prox

def compute_proximity(xs: Array, cfs: Array) -> float:
    cfs = einops.rearrange(cfs, 'n ... d -> n (...) d')
    prox_batch = jax.vmap(compute_single_proximity, in_axes=(None, 1))(xs, cfs)
    return prox_batch.mean()

# %% ../nbs/04_evaluate.ipynb 15
class Proximity(BaseEvalMetrics):
    """Compute L1 norm distance between input datasets and CF examples divided by the number of features."""
    def __init__(self, name: str = "proximity"):
        super().__init__(name=name)
    
    def __call__(self, explanation: Explanation) -> float:
        xs, cfs = explanation.xs, explanation.cfs
        return compute_proximity(xs, cfs)

# %% ../nbs/04_evaluate.ipynb 17
def compute_single_sparsity(xs: Array, cfs: Array, feature_indices: List[Tuple[int, int]]):
    def _feat_sparsity(xs, cfs, feat_indices):
        start, end = feat_indices
        xs = xs[:, start: end]
        cfs = cfs[:, start: end]
        return jnp.linalg.norm(xs - cfs, ord=0, axis=1).mean()
    
    return jnp.stack([_feat_sparsity(xs, cfs, feat_indices) for feat_indices in feature_indices]).mean()

def compute_sparsity(xs: Array, cfs: Array, feature_indices: List[Tuple[int, int]]) -> float:
    cfs = einops.rearrange(cfs, 'n ... d -> n (...) d')
    sparsity_batch = jax.vmap(compute_single_sparsity, in_axes=(None, 1, None))(xs, cfs, feature_indices)
    return sparsity_batch.mean()

# %% ../nbs/04_evaluate.ipynb 18
class Sparsity(BaseEvalMetrics):
    """Compute the number of feature changes between input datasets and CF examples."""

    def __init__(self, name: str = "sparsity"):
        super().__init__(name=name)
    
    def __call__(self, explanation: Explanation) -> float:
        xs, cfs, feature_indices = explanation.xs, explanation.cfs, explanation.feature_indices
        return compute_sparsity(xs, cfs, feature_indices)

# %% ../nbs/04_evaluate.ipynb 20
@partial(jit, static_argnums=(2))
def pairwise_distances(
    x: Array, # [n, k]
    y: Array, # [m, k]
    metric: str = "euclidean" # Supports "euclidean" and "cosine"
) -> Array: # [n, m]
    def euclidean_distances(x: Array, y: Array) -> float:
        XX = jnp.dot(x, x)
        YY = jnp.dot(y, y)
        XY = jnp.dot(x, y)
        dist = jnp.clip(XX - 2 * XY + YY, a_min=0.)
        return jnp.sqrt(dist)
        # return jnp.linalg.norm(x - y, ord=2)
    
    def cosine_distances(x: Array, y: Array) -> float:
        return 1.0 - jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-8)
    
    if metric == "euclidean":
        dists_fn = vmap(vmap(euclidean_distances, in_axes=(None, 0)), in_axes=(0, None))
    elif metric == "cosine":
        dists_fn = vmap(vmap(cosine_distances, in_axes=(None, 0)), in_axes=(0, None))
    else:
        raise ValueError(f"metric='{metric}' not supported")
    
    return dists_fn(x, y)

# %% ../nbs/04_evaluate.ipynb 21
@ft.partial(jax.jit, static_argnames=["k", "recall_target"])
def l2_ann(
    qy, # Query vectors
    db, # Database
    k=10, # Number of nearest neighbors to return
    recall_target=0.95 # Recall target for the approximation.
) -> Tuple[Array, Array]: # Return (distance, neighbor_indices) tuples
    dists = pairwise_distances(qy, db)
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

# %% ../nbs/04_evaluate.ipynb 22
class ManifoldDist(BaseEvalMetrics):
    """Compute the L1 distance to the n-nearest neighbor for all CF examples."""
    def __init__(self, n_neighbors: int = 1, name: str = "manifold_dist"):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        
    def __call__(self, explanation: Explanation) -> float:
        xs, cfs = explanation.xs, explanation.cfs
        l2_ann_partial = ft.partial(l2_ann, k=self.n_neighbors)
        dists, _ = vmap(l2_ann_partial, in_axes=(1, None))(cfs, xs)
        return dists.mean()

# %% ../nbs/04_evaluate.ipynb 24
class Runtime(BaseEvalMetrics):
    """Compute the runtime of the CF explanation method."""
    def __init__(self, name: str = "runtime"):
        super().__init__(name=name)
    
    def __call__(self, explanation: Explanation) -> float:
        return explanation.total_time

# %% ../nbs/04_evaluate.ipynb 27
METRICS_CALLABLE = [
    PredictiveAccuracy('acc'),
    PredictiveAccuracy('accuracy'),
    Validity(),
    Proximity(),
    Runtime(),
    ManifoldDist(),
]

METRICS = { m.name: m for m in METRICS_CALLABLE }

DEFAULT_METRICS = ["acc", "validity", "proximity"]

# %% ../nbs/04_evaluate.ipynb 29
def _get_metric(metric: str | BaseEvalMetrics, cf_exp: Explanation):
    if isinstance(metric, str):
        if metric not in METRICS.keys():
            raise ValueError(f"'{metric}' is not supported. Must be one of {METRICS.keys()}")
        res = METRICS[metric](cf_exp)
    elif callable(metric):
        # f(cf_exp) not supported for now
        if not isinstance(metric, BaseEvalMetrics):
            raise ValueError(f"metric needs to be a subclass of `BaseEvalMetrics`.")
        res = metric(cf_exp)
    else:
        raise ValueError(f"{type(metric).__name__} is not supported as a metric.")
    
    # Get scalar value
    if isinstance(res, Array) and res.ravel().shape == (1,):
        res = res.item()
    return res


# %% ../nbs/04_evaluate.ipynb 31
def evaluate_cfs(
    cf_exp: Explanation, # CF Explanations
    metrics: Iterable[Union[str, BaseEvalMetrics]] = None, # A list of Metrics. Can be `str` or a subclass of `BaseEvalMetrics`
    return_dict: bool = True, # return a dictionary or not (default: True)
    return_df: bool = False # return a pandas Dataframe or not (default: False)
):
    cf_name = cf_exp.cf_name
    data_name = cf_exp.data_name
    result_dict = { (data_name, cf_name): dict() }

    if metrics is None:
        metrics = DEFAULT_METRICS

    for metric in metrics:
        metric_name = str(metric)
        result_dict[(data_name, cf_name)][metric_name] = _get_metric(metric, cf_exp)
    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    
    if return_dict and return_df:
        return (result_dict, result_df)
    elif return_dict or return_df:
        return result_df if return_df else result_dict


# %% ../nbs/04_evaluate.ipynb 33
def benchmark_cfs(
    cf_results_list: Iterable[Explanation],
    metrics: Optional[Iterable[str]] = None,
):
    dfs = [
        evaluate_cfs(
            cf_exp=cf_results, metrics=metrics, return_dict=False, return_df=True
        )
        for cf_results in cf_results_list
    ]
    return pd.concat(dfs)

