# pytc

**Py**thon **T**rans**C**orrelation package

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Features

- **Modular Jastrow factors**: Boys-Handy, Nuclear Cusp, Neural Network (EE/EN/EEN), REXP, Polynomial, and Composite
- **JAX-based automatic differentiation** for Jastrow gradients and Laplacians via [folx](https://github.com/microsoft/folx)
- **VMC-based Jastrow optimization** with second-order Newton and first-order machine learning optimizers, e.g. Adam
- **Deterministic Jastrow optimization** via second-quantized optimization algorithm
- **GPU acceleration** via JAX for both VMC sampling and integral calculations using multiple GPUs
- **Transcorrelated integrals**: K1, K2, K3 two-body and xTC approximated three-body integrals
- **Interpolative Separable Density Fitting (ISDF)** for efficient integral calculations—up to 800+ orbitals with controlled accuracy
- **Seamless PySCF integration**: Works directly with PySCF mean-field objects and CCSD solvers
