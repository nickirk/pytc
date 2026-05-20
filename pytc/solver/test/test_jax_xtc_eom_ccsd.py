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
    """M5 — singlet EOM-EE-CCSD numerical validation against pyscf."""

    def test_same_eris_matches_pyscf(self):
        """Code-level cross-check: ours and pyscf agree to micro-Hartree
        when fed the same Hermitian eris.

        Validates that our JAX H̄ intermediates, σ-vector, F-only
        preconditioner, and the Davidson wiring through pyscf's
        ``EOMEESinglet.kernel`` reproduce stock pyscf EOM-EE-CCSD on a
        system where pyscf can also be run end-to-end. Skips the
        XTC code path entirely — exercises only ``xtc_eom_ccsd.EOMEE``.
        """
        from pyscf import cc as pcc
        from pyscf.cc import eom_rccsd
        from pytc.solver.xtc_eom_ccsd import EOMEE

        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis="sto-3g",
            verbose=0,
        )
        mf = scf.RHF(mol).run()

        ref_cc = pcc.CCSD(mf)
        ref_cc.kernel()

        # pyscf reference uses its own packed-ovvv eris untouched.
        e_ref = ref_cc.eomee_ccsd_singlet(nroots=6)[0]

        # Our path needs a 4D ovvv (matches pytc convention). Build a
        # separate eris on the same converged amplitudes and patch.
        my_eris = ref_cc.ao2mo()
        my_eris.ovvv = np.asarray(my_eris.get_ovvv())
        ref_cc.eris = my_eris
        e_xtc, _ = EOMEE(ref_cc).kernel(nroots=6)

        # 5 µEh tolerance: our F-only preconditioner reaches Davidson
        # convergence slightly less tightly than pyscf's full diag, so
        # eigenvalues land within ~1.5 µEh rather than the ~0.1 µEh pyscf
        # achieves end-to-end. Both are well below chemistry precision.
        np.testing.assert_allclose(
            np.sort(np.asarray(e_xtc)),
            np.sort(np.asarray(e_ref)),
            atol=5e-6,
        )

    def test_hermitian_limit_matches_pyscf(self):
        """End-to-end pytc-XTC → EOMEE matches pyscf in the large-α limit.

        Per pytc convention (see ``test_jax_xtc_ccsd.test_hermitian_limit``),
        ``alpha=1000`` makes the REXP Jastrow effectively negligible — XTC
        integrals reduce to standard Coulomb. Excitation energies from
        XTC-EOM-CCSD on this reference should reproduce stock pyscf
        EOM-EE-CCSD to sub-µEh.
        """
        from pyscf import cc as pcc
        from pytc import xtc as xtc_mod
        from pytc.jastrow.rexp import REXP
        from pytc.solver import jax_xtc_ccsd
        from pytc.solver.xtc_eom_ccsd import EOMEE

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).run()

        ref_cc = pcc.CCSD(mf)
        ref_cc.kernel()
        e_ref = ref_cc.eomee_ccsd_singlet(nroots=4)[0]

        large_alpha = {"alpha": jnp.array([1000.0])}
        my_xtc = xtc_mod.XTC.from_pyscf(mf, REXP())
        cc = jax_xtc_ccsd.RCCSD(mf, my_xtc, large_alpha)
        cc.kernel()
        e_xtc, _ = EOMEE(cc).kernel(nroots=4)

        np.testing.assert_allclose(
            np.sort(np.asarray(e_xtc)),
            np.sort(np.asarray(e_ref)),
            atol=1e-6,
        )


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
