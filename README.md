# CFNet: An Algorithmic Recourse Library in Jax
> A fast and scalable library for counterfactual explanations in Jax.


## Key Features

- **fast**: code runs significantly faster than existing CF explanation libraries.
- **scalable**: code can be accelerated over *CPU*, *GPU*, and *TPU*
- **flexible**: we provide flexible API for researchers to allow full customization.


TODO: 
- implement various methods of CF explanations


## Install

`cfnet` is built on top of [Jax](https://jax.readthedocs.io/en/latest/index.html). It also uses [Pytorch](https://pytorch.org/) to load data.

### Running on CPU

If you only need to run `cfnet` on CPU, you can simply install via `pip` or clone the `GitHub` project.

Installation via PyPI:
```bash
pip install cfnet
```

Editable Install:
```bash
git clone https://github.com/BirkhoffG/cfnet.git
pip install -e cfnet
```

### Running on GPU or TPU

If you wish to run `cfnet` on GPU or TPU, please first install this library via `pip install cfnet`.

Then, you should install the right GPU or TPU version of Jax by following steps in the [install guidelines](https://github.com/google/jax#installation).



## A Minimum Example

```python
from cfnet.utils import load_json
from cfnet.datasets import TabularDataModule
from cfnet.training_module import PredictiveTrainingModule
from cfnet.train import train_model
from cfnet.methods import VanillaCF
from cfnet.evaluate import generate_cf_results_local_exp, benchmark_cfs
from cfnet.import_essentials import *

data_configs = load_json('assets/configs/data_configs/adult.json')
m_configs = {
    'lr': 0.003,
    "sizes": [50, 10, 50],
    "dropout_rate": 0.3
}
t_configs = {
    'n_epochs': 10,
    'monitor_metrics': 'val/val_loss',
    'logger_name': 'pred'
}
cf_configs = {
    'n_steps': 1000,
    'lr': 0.001
}

# load data
dm = TabularDataModule(data_configs)

# specify the ML model 
training_module = PredictiveTrainingModule(m_configs)

# train ML model
params, opt_state = train_model(
    training_module, dm, t_configs
)

# define CF Explanation Module
pred_fn = lambda x: training_module.forward(
    params, random.PRNGKey(0), x, is_training=False)
cf_exp = VanillaCF(cf_configs)

# generate cf explanations
cf_results = generate_cf_results_local_exp(cf_exp, dm, pred_fn)

# benchmark different cf explanation methods
benchmark_cfs([cf_results])
```

    Epoch 9: 100%|██████████| 191/191 [00:02<00:00, 76.28batch/s, train/train_loss_1=0.0548]
    100%|██████████| 1000/1000 [00:06<00:00, 142.99it/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc</th>
      <th>validity</th>
      <th>proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VanillaCF</th>
      <td>0.825943</td>
      <td>0.894116</td>
      <td>7.257979</td>
    </tr>
  </tbody>
</table>
</div>


