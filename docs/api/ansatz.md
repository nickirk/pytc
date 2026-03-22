# Ansatz

`pytc.ansatz`

Quantum many-body wave function ansatz implementations.

## `SlaterDet`

`pytc.ansatz.det.SlaterDet`

Slater determinant ansatz. Stores molecular orbital coefficients and configuration as a JAX-compatible PyTree (Flax dataclass).

**Class method:**

`SlaterDet.create(mol, mo_coeff, ...)`
: Create a Slater determinant from a PySCF molecule and MO coefficients.

**Properties:**

`n_electrons`
: Total number of electrons.

`n_alpha`, `n_beta`
: Number of alpha/beta electrons.

**Key methods:**

`eval(positions)`
: Evaluate the determinant at given electron positions.

```python
from pytc.ansatz.det import SlaterDet
det = SlaterDet.create(mol, mf.mo_coeff)
```

## `SlaterJastrow`

`pytc.ansatz.sj.SlaterJastrow`

Quantum many-body wave function combining Slater determinants with a Jastrow factor.

**Class method:**

`SlaterJastrow.create(mol, jastrow, dets)`
: Create a Slater-Jastrow ansatz from a PySCF molecule, a Jastrow factor, and a list of Slater determinants.

**Properties:**

`n_electrons`, `n_alpha`, `n_beta`
: Electron counts (delegated to the first determinant).

**Key methods:**

`log_psi(positions, params)`
: Evaluate the log wave function at given electron positions.

`local_energy(positions, params)`
: Compute the local energy at given electron positions.

```python
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
sj = SlaterJastrow.create(mol, jastrow, [det])
```
