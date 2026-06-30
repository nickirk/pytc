# PyTC

Python TransCorrelation package (to be released soon...)

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
gpu-memory
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

```{image} _static/isdf_timing_benzene.png
:alt: ISDF-xTC setup time vs basis-set size for benzene on a B200 GPU
:width: 700px
:align: center
```

- **JAX-based automatic differentiation** for Jastrow gradients and Laplacians via [folx](https://github.com/microsoft/folx) — single-pass forward Laplacian, JIT-compiled to fused GPU kernels, avoiding the cost of finite differences entirely.
- **Multi-GPU acceleration** via JAX sharding for both VMC sampling and integral evaluation — near-linear strong scaling across devices with no user-side code changes.
- **Transcorrelated integrals**: K1, K2, K3 two-body and xTC approximated three-body integrals — fully GPU-resident contractions, never materialising the 3-body tensor explicitly.
- **Interpolative Separable Density Fitting (ISDF)** for the xTC kernel — empirical $T \propto n_{\mathrm{orb}}^{1.76}$ scaling (see plot above) pushes routine calculations past 1200 orbitals on a single B200, which is beyond what standard quantum chemistry solvers can handle.
- **Modular Jastrow factors**: Boys-Handy, Nuclear Cusp, Neural Network (EE/EN/EEN), REXP, Polynomial, and Composite — share the same JIT-compiled evaluation path, so adding a new factor adds zero per-step overhead.
- **VMC-based Jastrow optimization** with second-order Newton and first-order ML optimizers (Adam) — second-order steps reach the same accuracy in an order of magnitude fewer iterations than SGD.
- **Seamless PySCF integration**: drop-in CCSD replacement that reuses existing PySCF mean-field objects, so transitioning from a standard workflow costs no extra setup time.

