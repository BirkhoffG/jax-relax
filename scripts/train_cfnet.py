from cfnet.training_module import PredictiveTrainingModule, CounterNetTrainingModule
from cfnet.train import train_model
from cfnet.datasets import TabularDataModule
from cfnet.evaluate import generate_cf_results_cfnet, benchmark_cfs, DEFAULT_METRICS
from cfnet.import_essentials import *
import argparse
from .utils_configs import get_configs


data_configs = {
    "data_dir": "assets/data/s_adult.csv",
    "data_name": "adult",
    "batch_size": 256,
    # 'sample_frac': 0.1,
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
    "imutable_cols": [
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
    "use_immutable": True
}

t_configs = {
    'n_epochs': 100,
    'monitor_metrics': 'val/val_loss'
}

cf_results_list = []

def store_benchmark():
    benchmark_df = benchmark_cfs(cf_results_list, DEFAULT_METRICS + ['manifold_dist'])
    benchmark_df.to_csv('result.csv')
    print(benchmark_df)


def main(args):
    
    configs_list = get_configs(args.data_name)

    for configs in configs_list:
        dm = TabularDataModule(configs["data_configs"]) 
        model = CounterNetTrainingModule(configs["cfnet_configs"])
        params, opts_state = train_model(model, dm, t_configs)
        cf_results = generate_cf_results_cfnet(model, dm, params, rng_key=jax.random.PRNGKey(0))
        cf_results_list.append(cf_results)

    store_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        type=str, 
                        default='all')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print("something is wrong: ", e)
        store_benchmark()
    # params, opt_state = train_model(
    #     PredictiveTrainingModule(mlp_configs), 
    #     TabularDataModule(data_configs), 
    #     t_configs
    # )

    # dm = TabularDataModule(data_configs) 
    # model = CounterNetTrainingModule(cfnet_configs)

    # params, opts_state = train_model(model, dm, t_configs)

    # cf_results = generate_cf_results_cfnet(model, dm, params, rng_key=jax.random.PRNGKey(0))

    

    # print(benchmark_cfs([cf_results], DEFAULT_METRICS + ['manifold_dist']))
