# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/legacy/04_trainer.ipynb.

# %% ../../nbs/legacy/04_trainer.ipynb 3
from __future__ import annotations
from .import_essentials import *
from .module import BaseTrainingModule, PredictiveTrainingModule
from .logger import Logger
from .utils import validate_configs, load_json
from .ckpt_manager import CheckpointManager, load_checkpoint
from ..data_module import DataModule
from urllib.request import urlretrieve
from keras.src.trainers.epoch_iterator import EpochIterator


# %% auto 0
__all__ = ['TrainingConfigs', 'train_model_with_states', 'train_model']

# %% ../../nbs/legacy/04_trainer.ipynb 4
class TrainingConfigs(BaseParser):
    """Configurator of `train_model`."""
    
    n_epochs: int = Field(
        description="Number of epochs."
    )
    batch_size: int = Field(
        description="Batch size."
    )
    monitor_metrics: Optional[str] = Field(
        None, description="Monitor metrics used to evaluate the training result after each epoch."
    )
    seed: int = Field(
        42, description="Seed for generating random number."
    )
    log_dir: str = Field(
        "log", description="The name for the directory that holds logged data during training."
    )
    logger_name: str = Field(
        "debug", description="The name for the directory that holds logged data during training under log directory."
    )
    log_on_step: bool = Field(
        False, description="Log the evaluate metrics at the current step."
    )
    max_n_checkpoints: int = Field(
        3, description="Maximum number of checkpoints stored."
    )

    @property
    def PRNGSequence(self):
        return hk.PRNGSequence(self.seed)


# %% ../../nbs/legacy/04_trainer.ipynb 5
def train_model_with_states(
    training_module: BaseTrainingModule,
    params: hk.Params,
    opt_state: optax.OptState,
    data_module: DataModule,
    t_configs: Dict[str, Any] | TrainingConfigs,
) -> Tuple[hk.Params, optax.OptState]:
    """Train models with `params` and `opt_state`."""

    t_configs = validate_configs(t_configs, TrainingConfigs)
    keys = t_configs.PRNGSequence
    # define logger
    logger = Logger(
        log_dir=t_configs.log_dir,
        name=t_configs.logger_name,
        on_step=t_configs.log_on_step,
    )
    logger.save_hyperparams(t_configs.dict())
    if hasattr(training_module, "hparams") and training_module.hparams is not None:
        logger.save_hyperparams(training_module.hparams)

    training_module.init_logger(logger)
    # define checkpoint manageer
    if t_configs.monitor_metrics is None:
        monitor_metrics = None
    else:
        monitor_metrics = f"{t_configs.monitor_metrics}_epoch"

    ckpt_manager = CheckpointManager(
        log_dir=Path(training_module.logger.log_dir) / "checkpoints",
        monitor_metrics=monitor_metrics,
        max_n_checkpoints=t_configs.max_n_checkpoints,
    )
    # dataloaders
    # train_loader = jdl.DataLoader(jdl.ArrayDataset(*data_module['train']), backend='jax', batch_size=t_configs.batch_size, shuffle=True) 
    epoch_iterator = EpochIterator(*data_module['train'], batch_size=t_configs.batch_size, shuffle=True)
    val_epoch_iterator = EpochIterator(*data_module['test'], batch_size=t_configs.batch_size, shuffle=False)

    @jax.jit
    def train_step(params, opt_state, key, batch):
        return training_module.training_step(params, opt_state, key, batch)
    
    # start training
    for epoch in range(t_configs.n_epochs):
        training_module.logger.on_epoch_started()
        # for step, batch in epoch_iterator.enumerate_epoch('np'):
        with tqdm(
            epoch_iterator.enumerate_epoch(), 
            unit="batch", 
            leave=epoch == t_configs.n_epochs - 1,
            total=epoch_iterator.num_batches
        ) as t_loader:
            t_loader.set_description(f"Epoch {epoch}")
            for step, batch in t_loader:
                x, y = batch[0]
                logs, (params, opt_state) = train_step(params, opt_state, next(keys), (x, y))
                # TODO: tqdm becomes the bottleneck
                t_loader.set_postfix(**logs)
        
        # validation
        for step, batch in val_epoch_iterator.enumerate_epoch():
            x, y = batch[0]
            logs = training_module.validation_step(params, next(keys), (x, y))
            # logger.log(logs)
        epoch_logs = training_module.logger.on_epoch_finished()
        ckpt_manager.update_checkpoints(params, opt_state, epoch_logs, epoch)

    training_module.logger.close()
    return params, opt_state


# %% ../../nbs/legacy/04_trainer.ipynb 6
def train_model(
    training_module: BaseTrainingModule, # Training module
    data_module: DataModule, # Data module
    batch_size=128, # Batch size
    epochs=1, # Number of epochs
    **fit_kwargs # Positional arguments for `keras.Model.fit`
) -> Tuple[hk.Params, optax.OptState]:
    """Train models."""
    t_configs = TrainingConfigs(
        n_epochs=epochs,
        batch_size=batch_size,
        **fit_kwargs
    )
    keys = t_configs.PRNGSequence 
    params, opt_state = training_module.init_net_opt(data_module, next(keys))
    return train_model_with_states(
        training_module=training_module,
        params=params,
        opt_state=opt_state,
        data_module=data_module,
        t_configs=t_configs,
    )
