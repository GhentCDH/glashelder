# Glashelder AI Experiments

## Install

To use this code, you should have `uv`, the python dependency manager installed.

If you have a Nvidia GPU, check which CUDA version you are running (e.g. through `nvidia-smi`).
Modify the following lines in `pyproject.toml` to reflect your cuda version:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130" # CUDA 13.0
# url = "https://download.pytorch.org/whl/cu126" # CUDA 12.6
explicit = true
```

The example in this repo uses CUDA 13.0

## Jupyter Notebooks

There are some jupyter notebooks in `notebooks/`.

To run them, you should start the jupyter server within the right experiments' dependencies.

This can be done with `uv`, by specifying the specific package with `--package`.

For example, the `search.ipynb` notebook is part of the `clipcluster` experiment (its path is `notebooks/clipcluster/search.ipynb`), and thus you can start the jupyter server to run this one with:

```sh
uv run --package clipcluster --with jupyter jupyter lab
```

Then browse to the notebook you want to run, and open it.

## CLI

Use the CLI to run experimenst through:

```sh
uv run glashelder <command> [arguments]
```

get an overview of available experiments / commands using `--help`:

```sh
uv run glashelder --help
```

## Adding a new experiment

Make sure you have the command runner `just` installed, and run `just new <experiment_name>`.
This will add a new directory and workspace member named `experiment_name`.
