# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/methods/03_proto.ipynb.

# %% ../../nbs/methods/03_proto.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from ..base import TrainableMixedin, BaseConfig
from .base import ParametricCFModule
from ..utils import validate_configs, auto_reshaping, grad_update
from ..ml_model import AutoEncoder
from ..data_module import DataModule

# %% auto 0
__all__ = ['ProtoCFConfig', 'ProtoCF']

# %% ../../nbs/methods/03_proto.ipynb 5
@auto_reshaping('x')
def _proto_cf(
    x: Array, 
    y_target: Array,
    pred_fn: Callable[[Array], Array],
    n_steps: int,
    lr: float,
    c: float, # hyperparameter for validity loss
    beta: float, # cost = beta *l1_norm + l2_norm
    gamma: float, # hyperparameter for loss_ae
    theta: float, # hyperparameter for loss_proto
    ae: keras.Model,
    validity_fn: Callable,
    sampled_data: Array,
    apply_constraints_fn: Callable,
) -> Array:
    
    def encode(x):
        return ae.encoder(x)
    
    def loss_fn(
        cf: Array,
        x: Array,
        y_target: Array,
        pred_fn: Callable[[Array], Array],
    ):
        y_cf = pred_fn(cf)
        loss_val = c * validity_fn(y_target, y_cf)
        loss_cost = beta * jnp.linalg.norm(cf - x, ord=1) + jnp.linalg.norm(cf - x, ord=2)
        loss_ae = gamma * jnp.square(ae(cf) - cf).mean()
        loss_proto = theta * jnp.square(
            jnp.linalg.norm(encode(cf) - encode(sampled_data).sum(axis=0) / n_sampled_data, ord=2)
        )
        return (loss_val + loss_cost + loss_ae + loss_proto).mean()
    
    @loop_tqdm(n_steps)
    def gen_cf_step(
        i, cf_opt_state: Tuple[Array, optax.OptState] 
    ) -> Tuple[Array, optax.OptState]:
        cf, opt_state = cf_opt_state
        cf_grads = jax.grad(loss_fn)(cf, x, y_target, pred_fn)
        cf, opt_state = grad_update(cf_grads, cf, opt_state, opt)
        cf = apply_constraints_fn(x, cf, hard=False)
        return cf, opt_state
    
    # Calculate the number of samples
    # If the sampled data is all zeros, which means that this is not a valid sample.
    # This is used to calculate the mean of encode(sampled_data)
    n_sampled_data = jnp.where((sampled_data == 0).all(axis=1), 0, 1).sum()
    cf = jnp.array(x, copy=True)
    opt = optax.adam(lr)
    opt_state = opt.init(cf)
    cf, opt_state = lax.fori_loop(0, n_steps, gen_cf_step, (cf, opt_state))
    cf = apply_constraints_fn(x, cf, hard=True)
    return cf

# %% ../../nbs/methods/03_proto.ipynb 6
class ProtoCFConfig(BaseConfig):
    """Configurator of `ProtoCF`."""
    
    n_steps: int = 100
    lr: float = 0.01
    c: float = Field(1, description="The weight for validity loss.")
    beta: float = Field(0.1, description="The weight for l1_norm in the cost function, where cost = beta * l1_norm + l2_norm.")
    gamma: float = Field(0.1, description="The weight for Autoencoder loss.")
    theta: float = Field(0.1, description="The weight for prototype loss.")
    n_samples: int = Field(128, description="Number of samples for prototype.")
    validity_fn: str = 'KLDivergence'
    # AE configs
    enc_sizes: List[int] = Field([64, 32, 16], description="List of hidden layers of Encoder.")
    dec_sizes: List[int] = Field([16, 32, 64], description="List of hidden layers of Decoder.")
    opt_name: str = Field("adam", description="Optimizer name of AutoEncoder.")
    ae_lr: float = Field(1e-3, description="Learning rate of AutoEncoder.")
    ae_loss: str = Field("mse", description="Loss function name of AutoEncoder.")


# %% ../../nbs/methods/03_proto.ipynb 7
class ProtoCF(ParametricCFModule):

    def __init__(
        self,
        configs: dict | ProtoCFConfig = None,
        ae: keras.Model = None,
        name: str = None,
    ):
        if configs is None:
            configs = ProtoCFConfig()
        configs = validate_configs(configs, ProtoCFConfig)
        self.ae = ae
        name = "ProtoCF" if name is None else name
        super().__init__(configs, name=name)

    def _init_model(self, config: ProtoCFConfig, model: keras.Model, output_size: int):
        if model is None:
            model = AutoEncoder(
                enc_sizes=config.enc_sizes,
                dec_sizes=config.dec_sizes,
                output_size=output_size,
            )
            model.compile(
                optimizer=keras.optimizers.get({
                    'class_name': config.opt_name, 
                    'config': {'learning_rate': config.ae_lr}
                }),
                loss=config.ae_loss,
            )
        return model
    
    def train(
        self, 
        data: DataModule, 
        batch_size: int = 128,
        epochs: int = 10,
        **fit_kwargs
    ):
        if not isinstance(data, DataModule):
            raise ValueError(f"Expected `data` to be `DataModule`, got type=`{type(data).__name__}` instead.")
        X_train, y_train = data['train'] 
        self.ae = self._init_model(self.config, self.ae, X_train.shape[1])
        self.ae.fit(
            X_train, X_train, 
            batch_size=batch_size, 
            epochs=epochs,
            **fit_kwargs
        )
        self._is_trained = True
        # self.sampled_data = data.sample(self.config.n_samples)
        sampled_xs, sampled_ys = data.sample(self.config.n_samples)
        self.sampled_data = (sampled_xs, sampled_ys)
        self.sampled_data_dict = {
            label.item(): sampled_xs[(sampled_ys == label).reshape(-1)]
                for label in jnp.unique(sampled_ys)
        }
        return self
    
    @auto_reshaping('x')
    def generate_cf(
        self,
        x: Array,  # `x` shape: (k,), where `k` is the number of features
        pred_fn: Callable[[Array], Array],
        y_target: Array = None,
        **kwargs,
    ) -> Array:
        # TODO: Select based on the closest prototype.
        if y_target is None:
            y_target = 1 - pred_fn(x)
        else:
            y_target = jnp.array(y_target, copy=True)

        sampled_data = jnp.where(
            y_target.argmax(axis=1) == self.sampled_data[1],
            self.sampled_data[0],
            jnp.zeros_like(self.sampled_data[0]),
        )

        return _proto_cf(
            x=x,
            y_target=y_target,
            pred_fn=pred_fn,
            n_steps=self.config.n_steps,
            lr=self.config.lr,
            c=self.config.c,
            beta=self.config.beta,
            gamma=self.config.gamma,
            theta=self.config.theta,
            ae=self.ae,
            sampled_data=sampled_data,
            validity_fn=keras.losses.get({'class_name': self.config.validity_fn, 'config': {'reduction': None}}),
            apply_constraints_fn=self.apply_constraints_fn,
        )
