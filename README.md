# ReLax


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

![Python](https://img.shields.io/pypi/pyversions/jax-relax.svg) ![CI
status](https://github.com/BirkhoffG/jax-relax/actions/workflows/test.yaml/badge.svg)
![Docs](https://github.com/BirkhoffG/jax-relax/actions/workflows/deploy.yaml/badge.svg)
![pypi](https://img.shields.io/pypi/v/jax-relax.svg) ![GitHub
License](https://img.shields.io/github/license/BirkhoffG/jax-relax.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06567/status.svg)](https://doi.org/10.21105/joss.06567)

[**Overview**](#overview) \| [**Installation**](#installation) \|
[**Tutorials**](https://birkhoffg.github.io/jax-relax/tutorials/getting_started.html)
\| [**Documentation**](https://birkhoffg.github.io/jax-relax/) \|
[**Citing ReLax**](#citing-relax)

## Overview

`ReLax` (**Re**course Explanation **L**ibrary in J**ax**) is an
efficient and scalable benchmarking library for recourse and
counterfactual explanations, built on top of
[jax](https://jax.readthedocs.io/en/latest/). By leveraging language
primitives such as *vectorization*, *parallelization*, and
*just-in-time* compilation in
[jax](https://jax.readthedocs.io/en/latest/), `ReLax` offers massive
speed improvements in generating individual (or local) explanations for
predictions made by Machine Learning algorithms.

Some of the key features are as follows:

- 🏃 **Fast and scalable** recourse generation.

- 🚀 **Accelerated** over `cpu`, `gpu`, `tpu`.

- 🪓 **Comprehensive** set of recourse methods implemented for
  benchmarking.

- 👐 **Customizable** API to enable the building of entire modeling and
  interpretation pipelines for new recourse algorithms.

## Installation

``` bash
pip install jax-relax
# Or install the latest version of `jax-relax`
pip install git+https://github.com/BirkhoffG/jax-relax.git 
```

To futher unleash the power of accelerators (i.e., GPU/TPU), we suggest
to first install this library via `pip install jax-relax`. Then, follow
steps in the [official install
guidelines](https://github.com/google/jax#installation) to install the
right version for GPU or TPU.

## Dive into `ReLax`

`ReLax` is a recourse explanation library for explaining (any) JAX-based
ML models. We believe that it is important to give users flexibility to
choose how to use `ReLax`. You can

- only use methods implemeted in `ReLax` (as a recourse methods
  library);
- build a pipeline using `ReLax` to define data module, training ML
  models, and generating CF explanation (for constructing recourse
  benchmarking pipeline).

### `ReLax` as a Recourse Explanation Library

We introduce basic use cases of using methods in `ReLax` to generate
recourse explanations. For more advanced usages of methods in `ReLax`,
See this [tutorials](tutorials/methods.ipynb).

``` python
from relax.methods import VanillaCF
from relax import DataModule, MLModule, generate_cf_explanations, benchmark_cfs
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import functools as ft
import jax
```

Let’s first generate synthetic data:

``` python
xs, ys = make_classification(n_samples=1000, n_features=10, random_state=42)
train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, random_state=42)
```

Next, we fit an MLP model for this data. Note that this model can be any
model implmented in JAX. We will use the
[`MLModule`](https://birkhoffg.github.io/jax-relax/ml_model.html#mlmodule)
in `ReLax` as an example.

``` python
model = MLModule()
model.train((train_xs, train_ys), epochs=10, batch_size=64)
```

Generating recourse explanations are straightforward. We can simply call
`generate_cf` of an implemented recourse method to generate *one*
recourse explanation:

``` python
vcf = VanillaCF(config={'n_steps': 1000, 'lr': 0.05})
cf = vcf.generate_cf(test_xs[0], model.pred_fn)
assert cf.shape == test_xs[0].shape
```

Or generate a bunch of recourse explanations with `jax.vmap`:

``` python
generate_fn = ft.partial(vcf.generate_cf, pred_fn=model.pred_fn)
cfs = jax.vmap(generate_fn)(test_xs)
assert cfs.shape == test_xs.shape
```

### `ReLax` for Building Recourse Explanation Pipelines

The above example illustrates the usage of the decoupled `relax.methods`
to generate recourse explanations. However, users are required to write
boilerplate code for tasks such as data preprocessing, model training,
and generating recourse explanations with feature constraints.

`ReLax` additionally offers a one-liner framework, streamlining the
process and helping users in building a standardized pipeline for
generating recourse explanations. You can write three lines of code to
benchmark recourse explanations:

``` python
data_module = DataModule.from_numpy(xs, ys)
exps = generate_cf_explanations(vcf, data_module, model.pred_fn)
benchmark_cfs([exps])
```

See [Getting Started with
ReLax](https://birkhoffg.github.io/jax-relax/tutorials/getting_started.html)
for an end-to-end example of using `ReLax`.

## Supported Recourse Methods

`ReLax` currently provides implementations of 9 recourse explanation
methods.

| Method                                                                                     | Type            | Paper Title                                                                                    | Ref                                       |
|--------------------------------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------|-------------------------------------------|
| [`VanillaCF`](https://birkhoffg.github.io/jax-relax/methods/vanilla.html#vanillacf)        | Non-Parametric  | Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR.   | [\[1\]](https://arxiv.org/abs/1711.00399) |
| [`DiverseCF`](https://birkhoffg.github.io/jax-relax/methods/dice.html#diversecf)           | Non-Parametric  | Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.           | [\[2\]](https://arxiv.org/abs/1905.07697) |
| [`ProtoCF`](https://birkhoffg.github.io/jax-relax/methods/proto.html#protocf)              | Semi-Parametric | Interpretable Counterfactual Explanations Guided by Prototypes.                                | [\[3\]](https://arxiv.org/abs/1907.02584) |
| [`CounterNet`](https://birkhoffg.github.io/jax-relax/methods/counternet.html#counternet)   | Parametric      | CounterNet: End-to-End Training of Prediction Aware Counterfactual Explanations.               | [\[4\]](https://arxiv.org/abs/2109.07557) |
| [`GrowingSphere`](https://birkhoffg.github.io/jax-relax/methods/sphere.html#growingsphere) | Non-Parametric  | Inverse Classification for Comparison-based Interpretability in Machine Learning.              | [\[5\]](https://arxiv.org/abs/1712.08443) |
| [`CCHVAE`](https://birkhoffg.github.io/jax-relax/methods/cchvae.html#cchvae)               | Semi-Parametric | Learning Model-Agnostic Counterfactual Explanations for Tabular Data.                          | [\[6\]](https://arxiv.org/abs/1910.09398) |
| [`VAECF`](https://birkhoffg.github.io/jax-relax/methods/vaecf.html#vaecf)                  | Parametric      | Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers. | [\[7\]](https://arxiv.org/abs/1912.03277) |
| [`CLUE`](https://birkhoffg.github.io/jax-relax/methods/clue.html#clue)                     | Semi-Parametric | Getting a CLUE: A Method for Explaining Uncertainty Estimates.                                 | [\[8\]](https://arxiv.org/abs/2006.06848) |
| [`L2C`](https://birkhoffg.github.io/jax-relax/methods/l2c.html#l2c)                        | Parametric      | Feature-based Learning for Diverse and Privacy-Preserving Counterfactual Explanations          | [\[9\]](https://arxiv.org/abs/2209.13446) |

## Citing `ReLax`

To cite this repository:

``` latex
@software{relax2023github,
  author = {Hangzhi Guo and Xinchang Xiong and Amulya Yadav},
  title = {{R}e{L}ax: Recourse Explanation Library in Jax},
  url = {http://github.com/birkhoffg/jax-relax},
  version = {0.2.0},
  year = {2023},
}
```
