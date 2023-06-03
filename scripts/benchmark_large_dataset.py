from relax.module import PredictiveTrainingModule, PredictiveTrainingModuleConfigs, load_pred_model
from relax.trainer import train_model
from relax.utils import load_json
from relax.data import TabularDataModule, TabularDataModuleConfigs
from relax.module import download_model
from relax.methods import *
from relax.evaluate import generate_cf_explanations, evaluate_cfs, BatchedVmapGenerationStrategy, BatchedPmapGenerationStrategy
from relax._ckpt_manager import load_checkpoint
from relax.import_essentials import *
import datasets as hfds
import argparse
from urllib.request import urlretrieve

# CFs for benchmarking
CF_NAMES = ["VanillaCF","DiverseCF","ProtoCF","CounterNet","CCHVAE","CLUE","GrowingSphere","VAECF"]

def hfds_to_dm(
    dataset: hfds.Dataset, 
    configs: TabularDataModuleConfigs
) -> TabularDataModule:
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    df = pd.concat([train_df, test_df])
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    print('df is loaded')
    dm = TabularDataModule(configs, df)
    return dm

def main(args):
    print("start...")

    if args.cf_methods == "all":
        cf_methods_list = CF_NAMES
    else:
        cf_methods_list = [args.cf_methods]

    # Print benchmarking CF methods
    if args.cf_methods != "all":
        print("CF method(s): ", cf_methods_list)
    else: 
        print("CF method(s): ", CF_NAMES)

    # data configs
    data_configs = TabularDataModuleConfigs(
        data_dir='',
        data_name='forktable',
        continous_cols=['AGEP', 'OCCP', 'POBP', 'RELP', 'WKHP'],
        discret_cols=['COW', 'SCHL', 'MAR', 'SEX', 'RAC1P', 'STATE', 'YEAR'],
    )

    # cf configs
    cf_configs_dict = {
        "VanillaCF":{},
        "DiverseCF":{},
        "ProtoCF":{},
        "CounterNet":{},
        "CCHVAE":{},
        "CLUE":{},
        "GrowingSphere":{"n_samples":400},
        "VAECF":{}
    }
    # t configs
    t_configs_dict = {
        "VanillaCF":{},
        "DiverseCF":{},
        "ProtoCF":{"n_epochs":5, "batch_size":256},
        "CounterNet":{"n_epochs":100, "batch_size":1024},
        "CCHVAE":{"n_epochs":5, "batch_size":256},
        "CLUE":{"n_epochs":5, "batch_size":256},
        "GrowingSphere":{},
        "VAECF":{"n_epochs":5, "batch_size":256}
    }

    # batch size
    batch_sizes = {
        "VanillaCF":2448543,
        "DiverseCF":524288,
        "ProtoCF":1048576,
        "CounterNet":2448543,
        "CCHVAE":32768,
        "CLUE":2448543,
        "GrowingSphere":16384,
        "VAECF":131072
    }

    # list for storing results
    results = []

    # load data
    ds = hfds.load_dataset("birkhoffg/folktables-acs-income")
    dm = hfds_to_dm(ds, data_configs)     

    module = PredictiveTrainingModule({
        'lr': 1e-3,
        'sizes': [110, 110, 50, 10],
        'dropout': 0.3,
    })

    # load predict function
    # get model urls
    _model_path = "assets/forktable/model"
    # create new dir
    data_dir = Path(os.getcwd()) / "cf_data"
    if not data_dir.exists():
        os.makedirs(data_dir)
    model_path = data_dir / "forktable" / "model"
    if not model_path.exists():
        os.makedirs(model_path)
    model_params_url = f"https://github.com/BirkhoffG/ReLax/raw/master/{_model_path}/params.npy"
    model_tree_url = f"https://github.com/BirkhoffG/ReLax/raw/master/{_model_path}/tree.pkl"

    # download trained model
    params_path = os.path.join(model_path, "params.npy")
    tree_path = os.path.join(model_path, "tree.pkl")
    if not os.path.isfile(params_path):
        urlretrieve(model_params_url, params_path)
    if not os.path.isfile(tree_path):
        urlretrieve(model_tree_url, tree_path)    
    params = load_checkpoint(model_path)
    pred_fn = module.pred_fn

    for cf_method in cf_methods_list:

        print("Benchmarking CF method:", cf_method)

        # strategy
        if args.strategy == "vmap":
            batch_size = batch_sizes[cf_method]
            strategy = BatchedVmapGenerationStrategy(batch_size)
        elif args.strategy == "pmap":
            batch_size = batch_sizes[cf_method]
            strategy = BatchedPmapGenerationStrategy(batch_size)
        else:
            strategy = args.strategy

        # get cf configs
        cf_configs = cf_configs_dict[cf_method]

        cf = globals()[cf_method](cf_configs)

        # Generate CFEs
        print("generate...")
        
        if bool(t_configs_dict[cf_method]):
            cf_exp = generate_cf_explanations(cf, dm, pred_fn=pred_fn, pred_fn_args=dict(params=params, rng_key=jrand.PRNGKey(0)), strategy=strategy, t_configs=t_configs_dict[cf_method])
        else:
            cf_exp = generate_cf_explanations(cf, dm, pred_fn=pred_fn, pred_fn_args=dict(params=params, rng_key=jrand.PRNGKey(0)), strategy=strategy)

        # Store benchmark results
        results.append(evaluate_cfs(cf_exp=cf_exp, metrics=["acc", "validity", "proximity","runtime"], return_dict=False, return_df=True))
        del cf_exp

    # Output as csv
    if args.to_csv:
        csv_name = args.csv_name
        dfs = pd.concat(results)
        dfs.to_csv(f'assets/{csv_name}.csv')
    else:
        print(results)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
                        default='forktable_benchmark_results')
    args = parser.parse_args()
    
    main(args)


