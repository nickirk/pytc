# PyTC

<p align="center">
  <img src="https://raw.githubusercontent.com/nickirk/pytc/main/docs/_static/pytc-logo.svg" alt="PyTC logo" width="280">
</p>

![CI](https://github.com/nickirk/pytc/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

**Py**thon **T**rans**C**orrelation package

## Development provenance

PyTC was initiated in Prof. Ali Alavi's group at the Max Planck Institute for Solid State Research, where the early-stage development of the transcorrelated-method infrastructure took place. Its current development is based in Prof. Tianyu Zhu's group at Yale University.

## Features

- **Modular Jastrow factors**: Boys-Handy, Nuclear Cusp, Neural Network (EE/EN/EEN), REXP, Polynomial, and Composite
- **JAX-based automatic differentiation** for Jastrow gradients and Laplacians via [folx](https://github.com/microsoft/folx)
- **VMC-based Jastrow optimization** with second-order Newton and first-order machine learning optimizers, e.g. Adam
- **Deterministic Jastrow optimization** via second-quantized optimization algorithm
- **GPU acceleration** via JAX for both VMC sampling and integral calculations using multiple GPUs
- **Transcorrelated integrals**: K1, K2, K3 two-body and xTC approximated three-body integrals
- **Interpolative Separable Density Fitting (ISDF)** for efficient integral calculations — empirical T ∝ n_orb^1.76 scaling, demonstrated past 1200 orbitals on a single B200 GPU
- **PySCF interoperability with JAX-native CCSD**: Builds on PySCF mean-field objects and molecular data, then runs xTC-CCSD with PyTC's in-house JAX solver

## Dependencies

- Python >= 3.10
- numpy
- scipy
- jax (autodiff and GPU acceleration)
  > **Note**: To run on GPUs, you must install the correct version of JAX. See the [JAX installation guide](https://github.com/google/jax#installation).
  > For example, for CUDA 12:
  > ```bash
  > pip install -U "jax[cuda12]"
  > ```
- flax (neural network)
- folx (≥0.2.22, installed automatically as a dependency)
- optax (machine learning optimizers)
- pyscf

## Installation

**Requirements**: Python >= 3.10

Install from PyPI:

```bash
python -m pip install pytc-qc
```

The PyPI distribution is named `pytc-qc`; the Python import package remains `pytc`.

For GPU support (CUDA 12):

```bash
python -m pip install pytc-qc
python -m pip install -U "jax[cuda12]"
```

For source checkout / development install, see [docs/installation.md](https://github.com/nickirk/pytc/blob/main/docs/installation.md).

## Quick Start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import jax_xtc_ccsd

# Set up molecule
mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='ccpvdz')
mf = scf.RHF(mol)
mf.kernel()

# Create Jastrow factor
my_jastrow = rexp.REXP()
jastrow_params = {'alpha': jnp.array([1.0])}

# Exact XTC (for small systems)
my_xtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
eris_exact = my_xtc.make_eris(mf, jastrow_params)

# ISDF-accelerated XTC (scales to large systems)
n_rank = 10 * my_xtc.n_orb  # ISDF rank
my_isdf_xtc = xtc.ISDFXTC.from_xtc(my_xtc, n_rank=n_rank)
my_isdf_xtc = my_isdf_xtc.isdf(jastrow_params)
eris_isdf = my_isdf_xtc.make_eris(mf, jastrow_params)

# Run xTC-CCSD with pytc's own JAX-native solver
mycc = jax_xtc_ccsd.RCCSD(mf, my_isdf_xtc, jastrow_params)
e_corr, t1, t2 = mycc.kernel(eris=eris_isdf)
```

See [docs/quickstart.md](https://github.com/nickirk/pytc/blob/main/docs/quickstart.md) for additional examples and explanations.

## Code Overview


<div align="center">

```mermaid
flowchart TD
    PySCF["🔬 PySCF — gto.Mole · scf.RHF"]

    subgraph jastrow["pytc.jastrow"]
        J["BoysHandy · NuclearCusp · NeuralNet · REXP"]
    end

    subgraph ansatz["pytc.ansatz"]
        SJ["SlaterJastrow = SlaterDet + Jastrow"]
    end

    subgraph vmc["pytc.vmc"]
        V["Metropolis sampler — SR / Adam optimizer"]
    end

    subgraph xtc["pytc.xtc · kmat · df"]
        X["XTC exact / ISDFXTC D & X kernels / K1 · K3"]
    end

    subgraph solver["pytc.solver"]
        S["RCCSD — non-Hermitian CCSD"]
    end

    PySCF --> jastrow
    PySCF --> ansatz
    jastrow --> ansatz
    ansatz -->|VMC optimize| vmc
    vmc -->|optimized params| xtc
    jastrow -->|Jastrow factor| xtc
    xtc -->|ERIs| solver
    PySCF -->|mf| solver
```

</div>

## Usage

See the `pytc/examples/` directory for complete examples:
- `Be_vmc_ref_opt_xtc_ccsd.py` — VMC optimization with xTC-CCSD
- `co2_simple_jastrow_xtc_ccsd.py` — xTC-CCSD on CO₂ with a simple Jastrow factor
- `h2o_jastrow_xtc_isdf_ccsd.py` — ISDF convergence study
- `h2o_jax_isdf_xtc.py` — JAX-based ISDF example
- `benchmark_isdf_xtc_kdx.py` — stage-wise wall-clock benchmark for ISDF and K-kernel builds

To run the tests:
```bash
python -m unittest discover -v
```

## Publications

*No publications yet. This section will be updated when papers using pytc are published.*

## Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues
- Use the [GitHub Issues](https://github.com/nickirk/pytc/issues) page to report bugs or request features
- Include a minimal reproducible example when reporting bugs
- Describe the expected vs. actual behavior

### Submitting Pull Requests
1. Fork the repository and create a feature branch
2. Make your changes, following the existing code style
3. **Run the tests** before submitting:
   ```bash
   python -m unittest discover -v
   ```
4. Submit a pull request with a clear description of your changes

### Code Style
- Follow the existing code conventions in the repository
- Use type hints where appropriate
- Add docstrings to new functions and classes

## For AI Agents

If you are an AI coding assistant working on this repository, please read the documentation in the `.agents/` directory before proceeding. This directory contains important rules, architecture details, and step-by-step workflows.

- `.agents/rules.md`: Core conventions and constraints for `pytc`.
- `.agents/ARCHITECTURE.md`: High-level explanation of the codebase structure.
- `.agents/workflows/`: Checklists and standardized processes for adding code (e.g., adding a new Jastrow factor, creating PRs).
