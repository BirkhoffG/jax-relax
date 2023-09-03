# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/methods/07_vaecf.ipynb.

# %% ../../nbs/methods/07_vaecf.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from .base import ParametricCFModule
from ..ml_model import MLP, MLPBlock
from ..data_module import DataModule
from ..utils import auto_reshaping, validate_configs
from keras_core.src.backend.jax.random import draw_seed

# %% auto 0
__all__ = ['sample_latent', 'VAE', 'VAECFConfig', 'VAECF']

# %% ../../nbs/methods/07_vaecf.ipynb 4
@jax.jit
def hindge_embedding_loss(
    inputs: Array, targets: Array, margin: float = 1.0
):
    """Hinge embedding loss."""
    assert targets.shape == (1,)
    loss = jnp.where(
        targets == 1,
        inputs,
        jax.nn.relu(margin - inputs)
    )
    return loss

# %% ../../nbs/methods/07_vaecf.ipynb 7
def sample_latent(rng_key, mean, logvar):
    eps = jax.random.normal(rng_key, mean.shape)
    return mean + eps * jnp.sqrt(logvar)

# %% ../../nbs/methods/07_vaecf.ipynb 8
class VAE(keras.Model):
    def __init__(
        self, 
        layers: list[int],
        # pred_fn: Callable,
        mc_samples: int = 50,
        # compute_regularization_fn=None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_layers = layers
        # self.pred_fn = pred_fn
        self.mc_samples = mc_samples
        # if compute_regularization_fn is None:
        #     self.compute_regularization_fn = lambda *args, **kwargs: 0.
        # elif callable(compute_regularization_fn):
        #     self.compute_regularization_fn = compute_regularization_fn
        # else:
        #     raise ValueError("`compute_regularization_fn` must be callable or None, ",
        #                      f"but got {type(compute_regularization_fn)} instead.")
    
    def set_pred_fn(self, pred_fn):
        self.pred_fn = pred_fn

    def set_compute_regularization_fn(self, compute_regularization_fn):
        self.compute_regularization_fn = compute_regularization_fn

    def _compile(self, x):
        pred_out = self.pred_fn(x)
        if pred_out.shape[-1] != 2: 
            raise ValueError("Only binary classification is supported.")
        
        mu = self.mu_enc(x)
        var = 0.5 + self.var_enc(x)
        z = sample_latent(draw_seed(None), mu, var)
        z = jnp.concatenate([z, pred_out.argmax(-1, keepdims=True)], axis=-1)
        mu_x = self.mu_dec(z)
    
    def build(self, input_shape):
        encoder = keras.Sequential([
            MLPBlock(size, use_batch_norm=True, dropout_rate=0.) for size in self.n_layers[:-1]
        ])
        decoder = keras.Sequential([
            MLPBlock(size, use_batch_norm=True, dropout_rate=0.) for size in self.n_layers[::-1][1:]
        ])

        self.mu_enc = keras.Sequential([encoder, keras.layers.Dense(self.n_layers[-1])])
        self.var_enc = keras.Sequential([encoder, keras.layers.Dense(self.n_layers[-1], activation='sigmoid')])
        self.mu_dec = keras.Sequential([
            decoder, keras.layers.Dense(input_shape[-1]), 
        ])
        self._compile(jnp.zeros(input_shape))

    def encode(self, x, training=None):
        mean = self.mu_enc(x, training=training)
        var = 0.5 + self.var_enc(x, training=training)
        return mean, var
    
    def decode(self, z, training=None):
        return self.mu_dec(z, training=training)
        
    def sample(
        self, 
        rng_key: jrand.PRNGKey, 
        inputs: Array, 
        mc_samples: int, 
        training=None
    ):
        @jit
        def step(rng_key, em, ev, c):
            # rng_key, _ = jrand.split(rng_key)
            z = sample_latent(rng_key, em, ev)
            z = jnp.concatenate([z, c], axis=-1)
            mu_x = self.decode(z)
            return mu_x

        keys = jrand.split(rng_key, mc_samples)
        x, c = inputs[:, :-1], inputs[:, -1:]
        em, ev = self.encode(x, training=training)
        step_fn = partial(step, em=em, ev=ev, c=c)
        mu_x = jax.vmap(step_fn)(keys) # [mc_samples, n, d]
        return em, ev, mu_x
    
    def compute_vae_loss(
        self,
        inputs: Array,
        em, ev, cfs
    ):
        def cf_loss(cf: Array, x: Array, y: Array):
            assert cf.shape == x.shape, f"cf.shape ({cf.shape}) != x.shape ({x.shape}))"
            # proximity loss
            recon_err = jnp.sum(jnp.abs(cf - x), axis=1).mean()
            # Sum to 1 over the categorical indexes of a feature
            cat_error = self.compute_regularization_fn(x, cf)
            # validity loss
            pred_prob = self.pred_fn(cf)
            # This is same as the following:
            # tempt_1, tempt_0 = pred_prob[y == 1], pred_prob[y == 0]
            # validity_loss = hindge_embedding_loss(tempt_1 - (1. - tempt_1), -1, 0.165) + \
            #     hindge_embedding_loss(1. - 2 * tempt_0, -1, 0.165)
            target = jnp.array([-1])
            hindge_loss_1 = hindge_embedding_loss(
                jax.nn.sigmoid(pred_prob[:, 1]) - jax.nn.sigmoid(pred_prob[:, 0]), target, 0.165)
            hindge_loss_0 = hindge_embedding_loss(
                jax.nn.sigmoid(pred_prob[:, 0]) - jax.nn.sigmoid(pred_prob[:, 1]), target, 0.165)
            tempt_1 = jnp.where(y == 1, hindge_loss_1, 0).sum() / y.sum()
            tempt_0 = jnp.where(y == 0, hindge_loss_0, 0).sum() / (y.shape[0] - y.sum())
            validity_loss = tempt_1 + tempt_0
            return recon_err + cat_error, - validity_loss
        
        xs, ys = inputs[:, :-1], inputs[:, -1]
        kl = 0.5 * jnp.mean(em**2 + ev - jnp.log(ev) - 1, axis=1)
        cf_loss_fn = partial(cf_loss, x=xs, y=ys)
        cf_losses, validity_losses = jax.vmap(cf_loss_fn)(cfs)
        return (cf_losses.mean() + kl).mean() + validity_losses.mean()
    
    def call(self, inputs, training=None):
        rng_key = draw_seed(None)
        ys = 1. - self.pred_fn(inputs).argmax(axis=1, keepdims=True)
        inputs = jnp.concatenate([inputs, ys], axis=-1)
        em, ev, cfs = self.sample(rng_key, inputs, self.mc_samples, training=training)
        loss = self.compute_vae_loss(inputs, em, ev, cfs)
        self.add_loss(loss)
        return cfs   


# %% ../../nbs/methods/07_vaecf.ipynb 9
class VAECFConfig(BaseParser):
    """Configurator of `VAECFModule`."""
    layers: List[int] = Field(
        [20, 16, 14, 12, 5],
        description="Sequence of Encoder/Decoder layer sizes."
    )
    dropout_rate: float = Field(
        0.1, description="Dropout rate."
    )
    opt_name: str = Field(
        "adam", description="Optimizer name."  
    )
    lr: float = Field(
        1e-3, description="Learning rate."
    )
    mc_samples: int = Field(
        50, description="Number of samples for mu."
    )
    validity_reg: float = Field(
        42.0, description="Regularization for validity."
    )


# %% ../../nbs/methods/07_vaecf.ipynb 10
class VAECF(ParametricCFModule):
    def __init__(self, config=None, vae=None, name: str = 'VAECF'):
        if config is None:
            config = VAECFConfig()
        config = validate_configs(config, VAECFConfig)
        self.vae = vae
        super().__init__(config, name=name)

    def _init_model(
        self, 
        config: VAECFConfig, 
        model: keras.Model, 
        # pred_fn: Callable,
        # compute_regularization_fn: Callable
    ):
        if model is None:
            model = VAE(
                config.layers,
                # pred_fn=pred_fn,
                mc_samples=config.mc_samples,
                # compute_regularization_fn=compute_regularization_fn
            )
            model.compile(
                optimizer=keras.optimizers.get({
                    'class_name': config.opt_name, 
                    'config': {'learning_rate': config.lr}
                }),
            )
        return model
    
    def train(
        self, 
        data: DataModule, 
        pred_fn: Callable, 
        batch_size: int = 128,
        epochs: int = 10,
        **fit_kwargs
    ):
        if not isinstance(data, DataModule):
            raise ValueError(f"Expected `data` to be `DataModule`, "
                             f"got type=`{type(data).__name__}` instead.")
        train_xs, train_ys = data['train']
        self.vae = self._init_model(self.config, self.vae)
        self.vae.set_pred_fn(pred_fn)
        self.vae.set_compute_regularization_fn(data.compute_reg_loss)
        self.vae.fit(
            train_xs, train_ys, 
            batch_size=batch_size, 
            epochs=epochs,
            **fit_kwargs
        )
        self._is_trained = True
        return self
    
    @auto_reshaping('x')
    def generate_cf(
        self,
        x: Array,
        pred_fn: Callable = None,
        y_target: Array = None,
        rng_key: jrand.PRNGKey = None,
        **kwargs
    ) -> Array:
        # TODO: Currently assumes binary classification.
        if y_target is None:
            y_target = 1 - pred_fn(x).argmax(axis=1, keepdims=True)
        else:
            y_target = jnp.array(y_target, copy=True)
        if rng_key is None:
            raise ValueError("`rng_key` must be provided, but got `None`.")
        
        @jit
        def sample_step(rng_key, y_target):
            inputs = jnp.concatenate([x, y_target], axis=-1)
            _, _, cfs = self.vae.sample(rng_key, inputs, 1, training=False)
            return cfs
        
        return sample_step(rng_key, y_target)
        
