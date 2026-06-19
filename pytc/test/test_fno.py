import os
import unittest

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf
from pytc.fno import make_fno_mo_coeff

_ON_CI = os.environ.get("CI", "").lower() == "true"


def _h_chain(n, dist=1.4):
    return "; ".join(f"H 0 0 {i*dist}" for i in range(n))


class TestMakeFNOMoCoeff(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom=_h_chain(6), basis="cc-pvdz", verbose=0)
        self.mf = scf.RHF(self.mol).run()
        self.nocc = int(np.sum(self.mf.mo_occ > 0))
        self.nvir = self.mf.mo_coeff.shape[1] - self.nocc
        self.S = self.mf.get_ovlp()

    def test_n_keep_truncates_and_shapes(self):
        r = make_fno_mo_coeff(self.mf, n_keep=4)
        self.assertEqual(r.mo_coeff.shape, (self.mol.nao_nr(), self.nocc + 4))
        self.assertEqual(r.n_keep, 4)
        self.assertEqual(r.nocc, self.nocc)
        self.assertEqual(r.nvir_full, self.nvir)
        # occupied occupations preserved, virtuals zeroed
        np.testing.assert_allclose(r.mo_occ[: self.nocc], self.mf.mo_occ[: self.nocc])
        np.testing.assert_allclose(r.mo_occ[self.nocc:], 0.0)
        # n_orb < n_ao (the whole point)
        self.assertLess(r.mo_coeff.shape[1], self.mol.nao_nr())
        # mf copy is self-consistent with the returned arrays
        np.testing.assert_allclose(r.mf.mo_coeff, r.mo_coeff)
        np.testing.assert_allclose(r.mf.mo_occ, r.mo_occ)

    def test_orthonormal_in_overlap_metric(self):
        # MOs are orthonormal in the AO-overlap metric C^T S C = I, not C^T C = I.
        for k in (2, 4, self.nvir):
            r = make_fno_mo_coeff(self.mf, n_keep=k)
            eye = r.mo_coeff.T @ self.S @ r.mo_coeff
            np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-10)

    def test_occupations_descending(self):
        r = make_fno_mo_coeff(self.mf, n_keep=4)
        self.assertTrue(np.all(np.diff(r.no_occ) <= 1e-12), "NO occupations not sorted descending")
        # virtual NO occupations are small and non-negative for an RHF MP2
        self.assertTrue(np.all(r.no_occ >= -1e-12))

    def test_kept_virtuals_semicanonical(self):
        r = make_fno_mo_coeff(self.mf, n_keep=5)
        fock_mo = r.mo_coeff.T @ self.mf.get_fock() @ r.mo_coeff
        fv = fock_mo[self.nocc:, self.nocc:]
        # the kept-virtual Fock block must be diagonal (semicanonicalized)
        off = np.max(np.abs(fv - np.diag(np.diag(fv))))
        self.assertLess(off, 1e-10, f"kept-virtual Fock not diagonal: {off:.2e}")
        # and its diagonal equals the returned orbital energies
        np.testing.assert_allclose(r.mo_energy[self.nocc:], np.diag(fv), atol=1e-10)

    def test_occ_threshold_matches_n_keep(self):
        r4 = make_fno_mo_coeff(self.mf, n_keep=4)
        thr = float(r4.no_occ[3])  # 4th-largest occupation
        r_thr = make_fno_mo_coeff(self.mf, occ_threshold=thr)
        self.assertGreaterEqual(r_thr.n_keep, 4)
        np.testing.assert_allclose(np.sort(r_thr.no_occ[r_thr.kept_mask])[::-1],
                                   np.sort(r4.no_occ[r4.kept_mask])[::-1])

    def test_no_truncation_spans_full_space(self):
        # With neither argument, rotate to the full MP2-NO basis: the spanned
        # space (projector) must equal the input MO space.
        r = make_fno_mo_coeff(self.mf)
        self.assertEqual(r.mo_coeff.shape[1], self.mf.mo_coeff.shape[1])
        P1 = self.mf.mo_coeff @ self.mf.mo_coeff.T
        P2 = r.mo_coeff @ r.mo_coeff.T
        np.testing.assert_allclose(P1, P2, atol=1e-9)

    def test_mf_copy_preserves_eri_cache(self):
        # mf.copy() must keep mf._eri so the solver's ao2mo path works; the
        # generic copy.copy nulls it via pyscf __getstate__.
        r = make_fno_mo_coeff(self.mf, n_keep=4)
        self.assertIsNotNone(r.mf._eri)
        # and it is the same tensor object as the input (shared, read-only)
        self.assertIs(r.mf._eri, self.mf._eri)

    def test_errors(self):
        with self.assertRaises(ValueError):
            make_fno_mo_coeff(self.mf, n_keep=self.nvir + 1)
        with self.assertRaises(ValueError):
            make_fno_mo_coeff(self.mf, n_keep=-1)
        # UHF is rejected
        from pyscf import gto as _gto, scf as _scf
        umol = _gto.M(atom=_h_chain(2), basis="sto-3g", verbose=0, spin=2)
        with self.assertRaises(ValueError):
            make_fno_mo_coeff(_scf.UHF(umol).run())


if __name__ == "__main__":
    unittest.main()
