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
import gc

# CFs for benchmarking
CF_NAMES = ["VanillaCF","DiverseCF","ProtoCF","CounterNet","CCHVAE","CLUE","GrowingSphere","VAECF"]

# list of sample fractions
sample_fracs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def hfds_to_dm(
    dataset: hfds.Dataset, 
    configs: TabularDataModuleConfigs
) -> TabularDataModule:
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    size = int(len(test_df) * configs.sample_frac)
    test_df = test_df.head(size)
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
        "ProtoCF":{"n_epochs":10, "batch_size":256},
        "CounterNet":{"n_epochs":20, "batch_size":256},
        "CCHVAE":{"n_epochs":10, "batch_size":256},
        "CLUE":{"n_epochs":10, "batch_size":256},
        "GrowingSphere":{},
        "VAECF":{"n_epochs":10, "batch_size":256}
    }

    # batch size
    batch_sizes = {
        "VanillaCF":2448543,
        "DiverseCF":524288,
        "ProtoCF":1048576,
        "CounterNet":2097152,
        "CCHVAE":8192,
        "CLUE":1048576,
        "GrowingSphere":4096,
        "VAECF":65536
    }

    # list for storing results
    results = {}
    results['cf_methods\\#instances'] = []

    # load data
    # data configs
    ds = hfds.load_dataset("birkhoffg/folktables-acs-income")

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

    for sample_frac in sample_fracs:
        data_configs = TabularDataModuleConfigs(
            data_dir='',
            data_name='forktable',
            continous_cols=['AGEP', 'OCCP', 'POBP', 'RELP', 'WKHP'],
            discret_cols=['COW', 'SCHL', 'MAR', 'SEX', 'RAC1P', 'STATE', 'YEAR'],
            sample_frac=sample_frac
        )
        dm = hfds_to_dm(ds, data_configs)     
        print("Number of instance in train dataset",len(dm.train_dataset))
        print("Number of instance in test dataset",len(dm.test_dataset))
        results[len(dm.test_dataset)] = []

        for cf_method in cf_methods_list:
            print("Benchmarking CF method:", cf_method)

            if cf_method not in results['cf_methods\\#instances']:
                results['cf_methods\\#instances'].append(cf_method)

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
            results[len(dm.test_dataset)].append(evaluate_cfs(cf_exp=cf_exp, metrics=["runtime"], return_dict=True, return_df=False)[('forktable',cf_method)]["runtime"])
            del cf_exp
            gc.collect()

    # Output as csv
    if args.to_csv:
        csv_name = args.csv_name
        dfs = pd.DataFrame.from_dict(results)
        dfs.to_csv(f'assets/{csv_name}.csv', index=False)
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
                        action='store_true')
    parser.add_argument('--csv_name', 
                        type=str, 
                        default='forktable_scalability_results')
    args = parser.parse_args()
    
    main(args)


