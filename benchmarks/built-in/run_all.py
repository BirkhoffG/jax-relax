import relax
from relax.methods import *
from relax.import_essentials import *
from relax.data_module import DEFAULT_DATA_CONFIGS
from relax.utils import load_json
import argparse
import gc


# Datasets for benchmarking
DATASET_NAMES = list(DEFAULT_DATA_CONFIGS.keys())

# CFs for benchmarking
CF_NAMES = ["VanillaCF", "DiverseCF", "ProtoCF", "CounterNet", "CCHVAE", "CLUE", "GrowingSphere", "VAECF", "L2C"]

def load_cf_configs(
    cf_method: str, # The name of cf method
    data_name: str # The name of data
) -> dict:

    # validate data name and cf method
    if data_name not in DATASET_NAMES:
        raise ValueError(f'`data_name` must be one of {DATASET_NAMES}, '
            f'but got data_name={data_name}.')
    
    if cf_method not in CF_NAMES:
        raise ValueError(f'`cf_name` must be one of {CF_NAMES}, '
            f'but got cf_name={cf_method}.')
    
    # Fetch the cf configs from the configs file
    data_dir = Path(os.getcwd()) / "cf_data" / data_name 
    cf_configs = load_json(data_dir / "configs.json" )['cf_configs']

    return cf_configs[cf_method]


def main(args):
    print("start...")
    print("devices: ", jax.devices())

    # list for storing dataset names
    data_names = DATASET_NAMES if args.data_name == "all" else [args.data_name]

    # list for storing CF method names
    cf_methods_list = CF_NAMES if args.cf_methods == "all" else [args.cf_methods]

    # Print benchmarking CF methods and dataset
    # if args.cf_methods != "all":
    #     print("CF method(s): ", args.cf_methods)
    # else: 
    #     print("CF method(s): ", CF_NAMES)
    # print("Dataset(s): ", data_names)
    
    # strategy
    strategy = args.strategy

    # list for storing generated CFEs
    exps = []

    for data_name in data_names:
        for i, cf_method in enumerate(cf_methods_list):

            print("Benchmarking CF method:", cf_method)
            print("Benchmarking dataset:", data_name)
                                  
            # load data and data configs
            dm = relax.load_data(data_name)

            # keras.mixed_precision.set_global_policy("mixed_float16")
            
            # load predict function
            ml_model = relax.load_ml_module(data_name)
            pred_fn = ml_model.pred_fn

            # warm-up
            if i == 0 and not args.disable_jit and strategy == "vmap":
                print("warm-up...")
                test_xs, test_ys = dm['test']
                pred = pred_fn(test_xs)
            
            # get cf configs
            cf_config = {}

            cf = globals()[cf_method](cf_config)

            # Generate CFEs
            print("generate...")
            cf_exp = relax.generate_cf_explanations(
                cf, dm, pred_fn, strategy=strategy
            )
            
            # Store CFEs
            exps.append(cf_exp)
    
    results = relax.benchmark_cfs(cf_results_list=exps, metrics=["acc", "validity", "proximity","runtime"])

    # Output as csv
    if args.to_csv:
        f_dir = Path(os.getcwd()) / "benchmarks" / "built-in" / "assets"
        # if not os.path.exists(f_dir):
        f_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(f_dir / f"{args.csv_name}.csv")
    else:
        print(results)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        '-d',
                        type=str, 
                        default='adult', 
                        choices=['all'] + DATASET_NAMES)
    parser.add_argument('--cf_methods', 
                        '-c',
                        type=str, 
                        default='VanillaCF', 
                        choices=['all'] + CF_NAMES)
    parser.add_argument('--strategy', 
                        type=str, 
                        default='vmap', 
                        choices=['iter' ,'vmap', 'pmap'])
    parser.add_argument('--to_csv', 
                        type=bool, 
                        default=True, 
                        choices=[False,True])
    parser.add_argument('--csv_name', 
                        type=str, 
                        default='results')
    parser.add_argument('--disable_jit',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
    
    # jax.profiler.start_trace("/tmp/tensorboard")
    main(args)
    # jax.profiler.stop_trace()

