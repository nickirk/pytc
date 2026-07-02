# Installation

## Requirements

- Python 3.10 or higher

## Dependencies

- numpy
- scipy
- jax (autodiff and GPU acceleration)
- flax (neural network)
- folx (≥0.2.22, installed automatically as a dependency)
- optax (machine learning optimizers)
- pyscf

## Install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/nickirk/pytc.git
cd pytc
pip install -e .
```

## GPU support

To run on GPUs, install the correct version of JAX for your CUDA version.
See the [JAX installation guide](https://github.com/google/jax#installation).

For CUDA 12:

```bash
pip install -e .
pip install -U "jax[cuda12]"
```
