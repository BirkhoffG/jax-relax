from cfnet.training_module import PredictiveTrainingModule, CounterNetTrainingModule
from cfnet.train import train_model
from cfnet.datasets import TabularDataModule


data_configs = {
    "data_dir": "assets/data/s_adult.csv",
    "data_name": "adult",
    "batch_size": 256,
    'sample_frac': 0.1,
    "continous_cols": [
        "age",
        "hours_per_week"
    ],
    "discret_cols": [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "race",
        "gender"
    ],
}
mlp_configs = {
    "sizes": [50, 10, 50],
    "dropout_rate": 0.3,
    'lr': 0.003,
}
cfnet_configs = {
    "enc_sizes": [50,10],
    "dec_sizes": [10],
    "exp_sizes": [50, 50],
    "dropout_rate": 0.3,    
    'lr': 0.003,
    "lambda_1": 1.0,
    "lambda_3": 0.1,
    "lambda_2": 0.2,
}

t_configs = {
    'n_epochs': 10,
    'monitor_metrics': 'val/val_loss'
}


if __name__ == "__main__":
    params, opt_state = train_model(
        PredictiveTrainingModule(mlp_configs), 
        TabularDataModule(data_configs), 
        t_configs
    )
    params, opts_state = train_model(
        CounterNetTrainingModule(cfnet_configs), 
        TabularDataModule(data_configs), 
        t_configs
    )
