from carla import Benchmark
from carla.data.catalog import CsvCatalog, DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
from carla.recourse_methods import *
import pandas as pd
import time
import os
import torch

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    # load a catalog dataset
    continuous = ["age","hours_per_week"]
    categorical = ["workclass","education","marital_status","occupation","race","gender"]
    immutable = ["race","gender"]

    df = pd.read_csv("assets/adult/data.csv")
    df['income'] = df['income'].astype(int)
    df.to_csv("assets/adult/data.csv", index=None)

    dataset = CsvCatalog(file_path="assets/adult/data.csv",
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')

    # load artificial neural network from catalog
    model = MLModelCatalog(dataset, "ann", "pytorch", load_online=False)

    model.train(
        learning_rate=0.01,
        epochs=10,
        batch_size=128
    )

    # get factuals from the data to generate counterfactual examples
    factuals = dataset.df.iloc[:]

    # load a recourse model and pass black box model
    # gs = Wachter(model, hyperparams={"binary_cat_features": False})
    gs = Wachter(model, hyperparams={"binary_cat_features": True, 'learning_rate': 0.1})

    # generate counterfactual examples
    start_time = time.time()
    counterfactuals = gs.get_counterfactuals(factuals)
    total_time = time.time() - start_time

    print(total_time)

if __name__ == "__main__":
    main()
