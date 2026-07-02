# Quick Start

## VMC-based Jastrow Optimization

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.vmc import optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import CompositeJastrow, NuclearCusp, BoysHandy

# Set up a molecule with PySCF
mol = gto.Mole()
mol.atom = "Be 0 0 0"
mol.basis = 'cc-pVDZ'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# Create Slater determinant and Jastrow factors
det = SlaterDet.create(mol, mf.mo_coeff)
jncusp = NuclearCusp.create(mol, name="ncusp")
jbh = BoysHandy.create(mol, name="bh")
jastrow = CompositeJastrow.create([jncusp, jbh])

# Create Slater-Jastrow ansatz
sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
params = [jastrow.init_params(), jnp.ones(1)]

# Optimize with VMC
opt_results = optimize_ref_var(
    sj_ansatz,
    params=params,
    n_walkers=5000,
    n_steps=50,
    n_opt_steps=10,
    optimizer_type='newton',
)
```

## ISDF-accelerated xTC Integrals

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

## Examples

See the `pytc/examples/` directory for more complete examples:

- `Be_vmc_ref_opt_xtc_ccsd.py` — VMC optimization with xTC-CCSD
- `co2_simple_jastrow_xtc_ccsd.py` — xTC-CCSD on CO₂ with a simple Jastrow factor
- `h2o_jastrow_xtc_isdf_ccsd.py` — ISDF convergence study
- `h2o_jax_isdf_xtc.py` — JAX-based ISDF example
- `benchmark_isdf_xtc_kdx.py` — stage-wise wall-clock benchmark for ISDF and K-kernel builds
