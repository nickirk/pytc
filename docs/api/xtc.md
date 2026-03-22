# Core Modules

## XTC — Transcorrelated Integrals

`pytc.xtc`

Compute transcorrelated two-electron integrals (ERIs) using exact or ISDF-accelerated methods.

### `XTC`

`pytc.xtc.XTC`

Exact transcorrelated integral calculator.

**Class method:**

`XTC.from_pyscf(mf, jastrow, grid_lvl=2)`
: Create from a PySCF mean-field object and a Jastrow factor.

**Key methods:**

`make_eris(mf, jastrow_params)`
: Compute transcorrelated ERIs compatible with PySCF's CCSD solver.

### `ISDFXTC`

`pytc.xtc.ISDFXTC`

ISDF-accelerated transcorrelated integrals. Scales to large systems (800+ orbitals) with controlled accuracy.

**Class method:**

`ISDFXTC.from_xtc(xtc, n_rank=...)`
: Create from an `XTC` instance with a given ISDF rank.

**Key methods:**

`isdf(jastrow_params)`
: Perform the ISDF decomposition.

`make_eris(mf, jastrow_params)`
: Compute ISDF-accelerated transcorrelated ERIs.

```python
from pytc import xtc

# Exact (small systems)
my_xtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
eris = my_xtc.make_eris(mf, jastrow_params)

# ISDF-accelerated (large systems)
my_isdf_xtc = xtc.ISDFXTC.from_xtc(my_xtc, n_rank=10 * my_xtc.n_orb)
my_isdf_xtc = my_isdf_xtc.isdf(jastrow_params)
eris = my_isdf_xtc.make_eris(mf, jastrow_params)
```

## TC — Transcorrelation

`pytc.tc`

Core transcorrelation routines for computing similarity-transformed Hamiltonian matrix elements.

## SCF — Self-Consistent Field

`pytc.scf`

### `TCSCF`

`pytc.scf.TCSCF`

Transcorrelated Self-Consistent Field solver. Inherits from PySCF's `RHF` class and adds TC effective potentials to `H_core` and `V_eff`.

## DF — Density Fitting

`pytc.df`

Density fitting utilities including Interpolative Separable Density Fitting (ISDF) with pivoted Cholesky decomposition.

## KMat — Kinetic Matrix Elements

`pytc.kmat`

Two-body kinetic matrix element calculations (K1, K2, K3 kernels).

## LMat — L-Matrix

`pytc.lmat`

Three-body integral (L-matrix) calculations.
