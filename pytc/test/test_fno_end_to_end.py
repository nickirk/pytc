import os
import unittest

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.integrals.xtc import XTC
from pytc.jastrow.rexp import REXP
from pytc.solver import xtc_ccsd
from pytc.fno import make_fno_mo_coeff

_ON_CI = os.environ.get("CI", "").lower() == "true"


@unittest.skipIf(_ON_CI, "end-to-end xTC-CCSD; too slow for the 16 GB CI runner")
class TestFNOEndToEnd(unittest.TestCase):
    """Drive the full FNO -> ISDFXTC/XTC -> RCCSD pipeline end-to-end.

    Guards the Design-B property that a truncated ``mf`` flows through the
    existing pipeline unchanged: (a) a truncated run completes with
    ``n_orb < n_ao`` through ``XTC.from_pyscf`` + ``RCCSD``, and (b) the
    all-virtual FNO (a pure virtual rotation) reproduces the full xTC-CCSD
    energy to a tight tolerance (CCSD is invariant under occupied-preserving
    virtual rotations).
    """

    def setUp(self):
        atom = "; ".join(f"H 0 0 {i*1.4}" for i in range(4))
        self.mol = gto.M(atom=atom, basis="cc-pvdz", verbose=0)
        self.mf = scf.RHF(self.mol).run()
        self.jastrow = REXP()
        self.params = {"alpha": jnp.array([0.5])}
        self.grid_lvl = 0

    def _run_ccsd(self, mf):
        xtc_obj = XTC.from_pyscf(mf, self.jastrow, grid_lvl=self.grid_lvl)
        cc = xtc_ccsd.RCCSD(mf, xtc_obj, self.params)
        cc.kernel()
        return float(cc.e_tot), int(xtc_obj.n_orb)

    def test_truncated_runs_and_all_virtual_matches_full(self):
        e_full, n_full = self._run_ccsd(self.mf)

        # (a) truncated active space runs through the pipeline with n_orb < n_ao
        r = make_fno_mo_coeff(self.mf, n_keep=4)
        e_trunc, n_trunc = self._run_ccsd(r.mf)
        self.assertLess(n_trunc, self.mol.nao_nr(), "truncated n_orb should be < n_ao")
        self.assertEqual(n_trunc, r.nocc + 4)

        # (b) all-virtual FNO (no truncation, just a virtual rotation) must
        # reproduce the full energy (virtual-rotation invariance).
        r_all = make_fno_mo_coeff(self.mf)
        e_all, n_all = self._run_ccsd(r_all.mf)
        self.assertEqual(n_all, n_full)
        self.assertAlmostEqual(e_all, e_full, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
