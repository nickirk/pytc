# pytc
**Py**thon **T**rans**C**orrelation package

## Dependencies
- numpy
- scipy
- unittest
- jax (autodiff and GPU acceleration)
- flax (neural network)
- optax (machine learning optimizers)
- kfac_jax (2nd order optimizer)
- tqdm (visual progress tracking)
- pyscf (![modified version](https://github.com/nickirk/pyscf/tree/tc-ccsd) for tc-ccsd only. Otherwise, official pyscf also works)

## Features
- Takes in user defined Jastrow factors (Neural Jastrow under test)
- Uses JAX autodiff to compute gradient of Jastrow on r and on parameters
- Supports simple Jastrow optimization by deterministic/second quantized optimization algorithm
- Real-space VMC Jastrow optimization
- Supports GPU acceleration via JAX (needs careful memory management)
- Calculates the transcorrelated 2-body integrals: K1, K2, K3
- Calculates the xTC approximated 3-body integrals
- Implements the Interpolative Separable Density Fitting (ISDF) approximation for the all the aforementioned integrals
- Seamless integration with PySCF rccsd solver, and more to come...

## Code overview


<img width="1041" height="731" alt="pytc" src="https://github.com/user-attachments/assets/b7ac0fb8-d960-4474-8283-1891b9de6228" />

## Usage
See the test directory for examples of how to use the package.


