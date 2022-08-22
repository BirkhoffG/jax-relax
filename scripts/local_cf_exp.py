from cfnet.train import train_model, TensorboardLogger
from cfnet.training_module import PredictiveTrainingModule
from cfnet.datasets import TabularDataModule
from cfnet.methods import VanillaCF
from cfnet.evaluate import generate_cf_results_local_exp, benchmark_cfs
from cfnet.import_essentials import *


def load_configs(config_path):
    with open(config_path) as f:
        return json.load(f)


m_configs = {
    'lr': 0.003,
    "sizes": [50, 10, 50],
    "dropout_rate": 0.3
}
t_configs = {
    'n_epochs': 20,
    'monitor_metrics': 'val/val_loss',
    'logger_name': 'pred'
}
cf_configs = {
    'n_steps': 1000,
    'lr': 0.001
}

if __name__ == "__main__":
    training_module = PredictiveTrainingModule(m_configs)

    data_configs = load_configs('assets/configs/data_configs/adult.json')
    dm = TabularDataModule(data_configs)

    params, opt_state = train_model(
        training_module, dm, t_configs
    )
    pred_fn = lambda x: training_module.forward(params, random.PRNGKey(0), x, is_training=False)

    cf_exp = VanillaCF(pred_fn, cf_configs)
        
    print("Start generating cf...")
    cf_results = generate_cf_results_local_exp(cf_exp, dm)
    print("DONE!")

    print(benchmark_cfs([cf_results]))
