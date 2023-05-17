from relax.module import PredictiveTrainingModule, PredictiveTrainingModuleConfigs, load_pred_model
from relax.trainer import train_model
from relax.utils import load_json
from relax.data import TabularDataModule, load_data
from relax.methods import *
from relax.evaluate import generate_cf_explanations, benchmark_cfs, _AuxPredFn
from relax.import_essentials import *
import argparse
import copy


# Datasets for benchmarking
DATASET_NAMES = ["adult","credit","heloc","oulad","student_performance","titanic","german","cancer","spam", "ozone", "qsar", "bioresponse", "churn", "road"]

# CFs for benchmarking
CF_NAMES = ["VanillaCF","DiverseCF","ProtoCF","CounterNet","CCHVAE","CLUE","GrowingSphere","VAECF"]

# Convert the input string into the class
def get_CF_classes(class_names):
    if class_names == 'all':
        # return a list of all available CF method classes
        names = CF_NAMES
        classes = [globals()[name]() for name in names]
        return classes
    else:
        # return a list of the specified CF method classes
        names = class_names.split(',')
        classes = [globals()[name]() for name in names]
        return classes

def main(args):
    print("start...")

    if args.data_name == "all":
        data_names = DATASET_NAMES
    else:
        data_names = [args.data_name]

    # Convert the input string into CF class
    cf_methods_list = get_CF_classes(args.cf_methods)

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
        for cf_method in cf_methods_list:

            cf = copy.deepcopy(cf_method)
                      
            # load data and data configs
            dm = load_data(data_name = data_name)
            
            # load predict function
            params, training_module = load_pred_model(data_name)
            pred_fn = training_module.pred_fn
            
            # Generate CFEs
            print("generate...")
            cf_exp = generate_cf_explanations(cf, dm, pred_fn=pred_fn, pred_fn_args=dict(params=params, rng_key=jrand.PRNGKey(0)), strategy = strategy)
            
            # Store CFEs
            exps.append(cf_exp)
    
    results = benchmark_cfs(exps)

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


