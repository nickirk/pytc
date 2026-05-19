"""Tests for ``pytc.solver.jax_xtc_eom_ccsd``.

M1 (skeleton): just confirm the module imports, EOMEE constructs, vector
size is computed correctly.

M5 (validation, not yet implemented): with zero Jastrow parameters,
XTC-EOM-CCSD must reproduce pyscf's stock EOM-EE-CCSD to ≤ 10⁻⁶ Eh.

M6 (CBD diradical): check that EOM finds a negative eigenvalue at D₄ₕ
corresponding to the 1¹B₁g singlet state.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

jax.config.update("jax_enable_x64", True)


class TestEOMEEImports(unittest.TestCase):
    """M1 — module imports and class can be constructed."""

    def test_module_imports(self):
        from pytc.solver import xtc_eom_ccsd  # noqa: F401

    def test_constructor_requires_converged_cc(self):
        from pytc.solver import xtc_eom_ccsd

        class _StubCC:
            t1 = None
            t2 = None

        with self.assertRaises(RuntimeError):
            xtc_eom_ccsd.EOMEE(_StubCC())


class TestEOMEEAgainstPySCF(unittest.TestCase):
    """M5 — zero-Jastrow XTC-EOM-CCSD must match pyscf EOM-EE-CCSD."""

    @unittest.skip("M5 not yet implemented — σ-vector still NotImplementedError")
    def test_h2o_sto3g(self):
        from pyscf import cc as pcc

        from pytc import xtc as xtc_mod
        from pytc.jastrow.rexp import REXP
        from pytc.solver import jax_xtc_ccsd
        from pytc.solver.xtc_eom_ccsd import EOMEE

        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis="sto-3g",
            verbose=0,
        )
        mf = scf.RHF(mol).run()

        # Stock pyscf EOM-EE-CCSD reference
        ref_cc = pcc.CCSD(mf)
        ref_cc.kernel()
        ref_eom = ref_cc.eomee_ccsd_singlet(nroots=2)
        e_ref = np.asarray(ref_eom[0])

        # XTC-EOM-CCSD with ALL-ZERO Jastrow should match ref
        jastrow = REXP()
        zero_params = {"alpha": jnp.array([0.0])}
        my_xtc = xtc_mod.XTC.from_pyscf(mf, jastrow)
        cc = jax_xtc_ccsd.RCCSD(mf, my_xtc, zero_params)
        cc.kernel()
        e_xtc, _ = EOMEE(cc).kernel(nroots=2)

        # Should agree to micro-Hartree
        np.testing.assert_allclose(np.sort(e_xtc), np.sort(e_ref), atol=1e-6)


class TestEOMEECBDDiradical(unittest.TestCase):
    """M6 — find the 1¹B₁g state of CBD D₄ₕ as a negative EOM root."""

    @unittest.skip("M6 — requires σ-vector implementation")
    def test_cbd_d4h_finds_negative_root(self):
        # Build CBD at the D4h square geometry (1.451 Å C-C), aug-cc-pVDZ.
        # The closed-shell 1¹A_g RHF reference is *above* the true 1¹B₁g
        # ground state. EOM-EE should find a root with ω < 0 corresponding
        # to 1¹B₁g.
        pass


if __name__ == "__main__":
    unittest.main()
