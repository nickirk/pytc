"""End-to-end VMC validation.

This is the load-bearing 'VMC works correctly' test for the periodic
stack. It runs two independent VMC samplings — one through the
molecular pipeline, one through the PBC pipeline — on the same H2
trial wavefunction (using the *same* MO coefficients and the *same*
initial Jastrow parameters) in a cell large enough that periodic image
contributions are below 10 mHa. The two mean energies must agree to
within combined statistical error plus that finite-cell offset.

If this test fails, the PBC pipeline is mis-composing somewhere
between walker init, moves, Ewald, and local-energy assembly. The
component-level tests in the other files don't catch composition bugs
on their own.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto as molgto
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet as MolSlaterDet
from pytc.jastrow import (
    CompositeJastrow,
    NuclearCusp as MolNuclearCusp,
    BoysHandy as MolBoysHandy,
)
from pytc.vmc.sampling import sample as mol_sample

from pytc.pbc.ansatz import create_slater_det
from pytc.pbc.jastrow import NuclearCusp, BoysHandy
from pytc.pbc.vmc import make_ewald_params, sample as pbc_sample


class TestVMCValidation(unittest.TestCase):
    """End-to-end correctness: PBC VMC at large cell matches molecular VMC."""

    def test_pbc_matches_molecular_at_large_cell(self):
        L = 12.0
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 1.4'
        cell.basis = 'sto-3g'
        cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
        cell.unit = 'B'
        cell.cart = True
        cell.verbose = 0
        cell.build()
        mf = pbcscf.RHF(cell)
        mf.exxdiv = None
        mf.kernel()

        mol = molgto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 1.4'
        mol.basis = 'sto-3g'
        mol.unit = 'B'
        mol.cart = True
        mol.verbose = 0
        mol.build()

        # Build both SJs with the SAME mo_coeff (from the PBC SCF) so
        # determinants are sign- and normalisation-compatible. Same Jastrow
        # construction parameters → same init_params values.
        pbc_det = create_slater_det(cell, mo_coeff=mf.mo_coeff)
        pbc_ncusp = NuclearCusp.create(cell, n_radial=300)
        pbc_bh = BoysHandy.create(cell)
        pbc_jastrow = CompositeJastrow.create([pbc_ncusp, pbc_bh])
        pbc_sj = SlaterJastrow.create(cell, jastrow=pbc_jastrow, dets=[pbc_det])

        mol_det = MolSlaterDet.create(mol, mo_coeff=mf.mo_coeff)
        mol_ncusp = MolNuclearCusp.create(mol, n_radial=300)
        mol_bh = MolBoysHandy.create(mol)
        mol_jastrow = CompositeJastrow.create([mol_ncusp, mol_bh])
        mol_sj = SlaterJastrow.create(mol, jastrow=mol_jastrow, dets=[mol_det])

        params = pbc_sj.init_params(random.PRNGKey(0))
        # NuclearCusp and BoysHandy init_params are deterministic functions
        # of (atom_coords, charges, basis) — both SJs see the same numbers,
        # so the parameter trees are identical leaf-wise.
        mol_params = mol_sj.init_params(random.PRNGKey(0))

        ewald = make_ewald_params(cell.lattice_vectors())

        # PBC VMC
        pbc_result = pbc_sample(
            pbc_sj, cell, ewald,
            n_walkers=64, n_steps=300, burn_in_steps=200,
            step_size=0.4, params=params,
            key=random.PRNGKey(1), log=False,
        )

        # Molecular VMC with the same MCMC parameters
        mol_result = mol_sample(
            mol_sj,
            n_walkers=64, n_steps=300, burn_in_steps=200,
            step_size=0.4, params=mol_params,
            key=random.PRNGKey(2),
            move_type='one',
            thinning=1,
        )

        e_pbc = pbc_result['mean']
        err_pbc = pbc_result['stderr']
        e_mol = float(mol_result['energy_mean'])
        err_mol = float(mol_result['energy_error'])

        diff = abs(e_pbc - e_mol)
        combined_err = float(np.sqrt(err_pbc ** 2 + err_mol ** 2))
        finite_cell_tol = 1e-2  # ~10 mHa at L=12 for H2 (Ewald vs bare)
        tolerance = 5 * combined_err + finite_cell_tol

        # Acceptance rates should also be in a reasonable range.
        self.assertGreater(pbc_result['acceptance'], 0.2)
        self.assertLess(pbc_result['acceptance'], 0.95)
        self.assertGreater(mol_result['acceptance_rates'].mean(), 0.2)

        self.assertLess(
            diff, tolerance,
            msg=(
                f"VMC mean energies disagree:\n"
                f"  PBC:     {e_pbc:.5f} +/- {err_pbc:.5f} Ha "
                f"(acc {pbc_result['acceptance']:.3f})\n"
                f"  Mol:     {e_mol:.5f} +/- {err_mol:.5f} Ha\n"
                f"  |diff|:  {diff:.5f} > tolerance {tolerance:.5f}\n"
                f"  combined statistical error: {combined_err:.5f}\n"
                f"  finite-cell allowance:      {finite_cell_tol:.5f}"
            ),
        )


if __name__ == '__main__':
    unittest.main()
