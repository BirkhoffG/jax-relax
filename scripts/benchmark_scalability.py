from relax.module import PredictiveTrainingModule, PredictiveTrainingModuleConfigs, load_pred_model
from relax.trainer import train_model
from relax.utils import load_json
from relax.data import TabularDataModule, TabularDataModuleConfigs
from relax.module import download_model
from relax.methods import *
from relax.evaluate import generate_cf_explanations, evaluate_cfs
from relax.evaluate import BatchedVmapGenerationStrategy, BatchedPmapGenerationStrategy, StrategyFactory
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


class BenchmarkLogger:
    """A logger for storing the benchmarking results.
    
    Data format:
        method | sample_frac | time
        ---------------------------
        VanillaCF | 0.01 | 0.1
        VanillaCF | 0.05 | 0.2
    """
    def __init__(self, file_name: str) -> None:
        # check if the file exists
        self.file_name = file_name
        if os.path.exists(file_name):
            self.__result = pd.read_csv(file_name)\
                               .set_index(['method', 'sample_frac'])\
                               .to_dict("dict")
            print(f"Result loaded from {file_name}")
        else:
            self.__result = {'time': {}}

    def update(self, method: str, sample_frac: float, time: float) -> None:
        self.__result['time'].update({
            (method, sample_frac): time
        })

    def store(self):
        pd.DataFrame(self.__result).to_csv(self.file_name, index_label=['method', 'sample_frac'])
        print(f"Result updated in {self.file_name}")

    def print(self):
        print(pd.DataFrame(self.__result).to_string())


def hfds_to_dm(
    dataset: hfds.Dataset, 
    configs: TabularDataModuleConfigs
) -> TabularDataModule:
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    df = pd.concat([train_df, test_df])
    size = int(len(df) * configs.sample_frac)
    df = df.iloc[:size]
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    print('df is loaded')
    dm = TabularDataModule(configs, df)
    return dm


def get_strategy(
    dm: TabularDataModule,
    batch_size: int,
    args
):
    data_size = len(dm.test_dataset)    
    if args.strategy == "vmap":
        if data_size < batch_size:
            strategy = StrategyFactory.get_strategy('vmap')
        else:
            strategy = BatchedVmapGenerationStrategy(batch_size)
    elif args.strategy == "pmap":
        if data_size < batch_size:
            strategy = StrategyFactory.get_strategy('pmap')
        else:
            strategy = BatchedPmapGenerationStrategy(batch_size)
    else:
        strategy = StrategyFactory.get_strategy(args.strategy)
    return strategy


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
        "ProtoCF":{"n_epochs":1, "batch_size":256},
        "CounterNet":{"n_epochs":2, "batch_size":1024},
        "CCHVAE":{"n_epochs":1, "batch_size":256},
        "CLUE":{"n_epochs":1, "batch_size":256},
        "GrowingSphere":{},
        "VAECF":{"n_epochs":1, "batch_size":256}
    }

    # batch size
    batch_sizes = {
        "VanillaCF":2448543,
        "DiverseCF":524288,
        "ProtoCF":1048576,
        "CounterNet":2097152,
        "CCHVAE":32768,
        "CLUE":1048576,
        "GrowingSphere":16384,
        "VAECF":131072
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
    
    # Init logger
    logger = BenchmarkLogger(f'assets/{args.csv_name}.csv')

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

        for cf_method in cf_methods_list:
            print("Benchmarking CF method:", cf_method)

            # if cf_method not in results['cf_methods\\#instances']:
            #     results['cf_methods\\#instances'].append(cf_method)

            strategy = get_strategy(dm, batch_sizes[cf_method], args)

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
                        # results[len(dm.test_dataset)].append(evaluate_cfs(cf_exp=cf_exp, metrics=["runtime"], return_dict=True, return_df=False)[('forktable',cf_method)]["runtime"])

            cf_method_name = cf.name
            runtime = evaluate_cfs(cf_exp=cf_exp, metrics=["runtime"], return_dict=True, return_df=False)[('forktable', cf_method_name)]["runtime"]
            logger.update(cf_method_name, sample_frac, runtime)
            if args.to_csv: logger.store()
            del cf_exp, cf, cf_configs
            gc.collect()
        
        del dm, data_configs
        gc.collect()

    # Output as csv
    # if args.to_csv:
    #     csv_name = args.csv_name
    #     dfs = pd.DataFrame.from_dict(results)
    #     dfs.to_csv(f'assets/{csv_name}.csv', index=False)
    # else:
    #     print(results)
    #     return None


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
                        default=True)
    parser.add_argument('--csv_name', 
                        type=str, 
                        default='forktable_scalability_results')
    args = parser.parse_args()
    
    main(args)


