from relax.module import PredictiveTrainingModule, PredictiveTrainingModuleConfigs, load_pred_model
from relax.trainer import train_model
from relax.utils import load_json
from relax.data import TabularDataModule, load_data
from relax.data.module import DEFAULT_DATA_CONFIGS
from relax.methods import *
from relax.evaluate import generate_cf_explanations, benchmark_cfs, _compute_acc
from relax.import_essentials import *
import argparse
import gc


# Datasets for benchmarking
DATASET_NAMES = list(DEFAULT_DATA_CONFIGS.keys())

# CFs for benchmarking
CF_NAMES = ["VanillaCF","DiverseCF","ProtoCF","CounterNet","CCHVAE","CLUE","GrowingSphere","VAECF"]

def load_cf_configs(
    cf_method: str, # The name of cf method
    data_name: str # The name of data
    ) -> dict:

    # validate data name and cf method
    if data_name not in DATASET_NAMES:
        raise ValueError(f'`data_name` must be one of {DATASET_NAMES}, '
            f'but got data_name={data_name}.')
    
    if cf_method not in CF_NAMES:
        raise ValueError(f'`data_name` must be one of {CF_NAMES}, '
            f'but got data_name={cf_method}.')
    
    # Fetch the cf configs from the configs file
    data_dir = Path(os.getcwd()) / "cf_data" / data_name 
    cf_configs = load_json(data_dir / "configs.json" )['cf_configs']

    return cf_configs[cf_method]


def main(args):
    print("start...")
    print("devices: ", jax.devices())

    if args.data_name == "all":
        data_names = DATASET_NAMES
    else:
        data_names = [args.data_name]

    if args.cf_methods == "all":
        cf_methods_list = CF_NAMES
    else:
        cf_methods_list = [args.cf_methods]

    # Print benchmarking CF methods and dataset
    if args.cf_methods != "all":
        print("CF method(s): ", args.cf_methods)
    else: 
        print("CF method(s): ", CF_NAMES)
    print("Dataset(s): ", data_names)
    
    # strategy
    strategy = args.strategy

    # list for storing generated CFEs
    exps = []

    for data_name in data_names:
        for i, cf_method in enumerate(cf_methods_list):

            print("Benchmarking CF method:", cf_method)
            print("Benchmarking dataset:", data_name)
                                  
            # load data and data configs
            dm = load_data(data_name = data_name)
            
            # load predict function
            params, training_module = load_pred_model(data_name)
            pred_fn = training_module.pred_fn

            # warm-up
            if i == 0:
                print("warm-up...")
                test_X, test_y = dm.test_dataset[:]
                pred = pred_fn(test_X, params, jrand.PRNGKey(0)).reshape(-1, 1).round()
                labels = test_y.reshape(-1, 1)
                print(f"{data_name}'s accuracy: ", (pred == labels).mean())
            
            # get cf configs
            cf_configs = load_cf_configs(cf_method, data_name)

            cf = globals()[cf_method](cf_configs)

            # Generate CFEs
            print("generate...")
            cf_exp = generate_cf_explanations(cf, dm, pred_fn=pred_fn, pred_fn_args=dict(params=params, rng_key=jrand.PRNGKey(0)), strategy = strategy)
            
            # Store CFEs
            exps.append(cf_exp)
    
    results = benchmark_cfs(cf_results_list=exps,metrics=["acc", "validity", "proximity","runtime"])

    # Output as csv
    if args.to_csv:
        csv_name = args.csv_name
        results.to_csv(f'assets/{csv_name}.csv')
    else:
        print(results)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        type=str, 
                        default='adult', 
                        choices=['all'] + DATASET_NAMES)
    parser.add_argument('--cf_methods', 
                        type=str, 
                        default='VanillaCF', 
                        choices=['all'] + CF_NAMES)
    parser.add_argument('--strategy', 
                        type=str, 
                        default='vmap', 
                        choices=['iter' ,'vmap', 'pmap'])
    parser.add_argument('--to_csv', 
                        type=bool, 
                        default=False, 
                        choices=[False,True])
    parser.add_argument('--csv_name', 
                        type=str, 
                        default='benchmark_results')
    args = parser.parse_args()
    
    main(args)


