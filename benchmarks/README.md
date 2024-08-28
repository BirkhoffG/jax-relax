# Benchmarking Scripts

This directory hosts scripts for benchmarking recourse methods.

## Installation

To install the necessary dependencies, run the following command:
```sh
pip install "jax-relax[dev]"
```

## Usage

To reproduce the results in the paper, you can run

```sh
python -m benchmarks.built-in.run_all
```

## Legacy Scripts

> [!WARNING]  
> These scripts are used to benchmark `jax-relax<0.2.0>=0.1.0`. 
>

To run these scripts, install the dependencies as:
```sh
pip install "jax-relax[dev]<0.2.0"
```

Next, run the script as:

```sh
# run large dataset
python -m benchmarks.legacy.benchmark_large_dataset
# run scalability test
python -m benchmarks.legacy.benchmark_scalability
```
