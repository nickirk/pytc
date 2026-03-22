# Solver

`pytc.solver`

Post-Hartree-Fock solvers for transcorrelated Hamiltonians.

## CCSD

`pytc.solver.ccsd`

Non-Hermitian Coupled Cluster Singles and Doubles (CCSD) solver adapted for transcorrelated Hamiltonians. Works with ERIs produced by the `pytc.xtc` module and integrates with PySCF's CC infrastructure.

```python
from pyscf import cc

# After computing transcorrelated ERIs
mycc = cc.rccsd.RCCSD(mf)
e_corr, t1, t2 = mycc.kernel(eris=eris)
```
