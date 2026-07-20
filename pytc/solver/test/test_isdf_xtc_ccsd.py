"""Tests for the factorized-dispatch solver class isdf_xtc_ccsd.RCCSD.

These tests EXECUTE the class's own seams on a tiny synthetic deck so that a
missing or mis-signed dependency (an import target that exists only in a
diagnostic harness, a renamed attribute) fails in the suite instead of on a
cluster.  Identity checks alone cannot catch that class: they prove which
code would run, not that everything it calls exists.
"""

import types
import unittest

import numpy as np
from pyscf import gto, scf

from pytc.solver import isdf_xtc_ccsd, jax_xtc_ccsd


def _fake_xtc_obj(nmo, rank, seed=20260720):
    """Minimal xtc_obj attribute surface consumed by the factorized state.

    Shapes mirror the real ISDFXTC object: phi_isdf (nmo, rank),
    grad_phi_isdf (nmo, rank, 3), X (nmo, nmo, rank).
    """
    rng = np.random.default_rng(seed)
    kernels = {
        "K1_kernel": rng.standard_normal((rank, rank, 3)),
        "K3_kernel": rng.standard_normal((rank, rank)),
        "D": rng.standard_normal((rank, rank)),
        "X": rng.standard_normal((nmo, nmo, rank)),
    }
    return types.SimpleNamespace(
        phi_isdf=rng.standard_normal((nmo, rank)),
        grad_phi_isdf=rng.standard_normal((nmo, rank, 3)),
        isdf_kernels=kernels,
    )


class FactorizedDispatchSurfaceTest(unittest.TestCase):
    """The dispatch seam's structure, pinned so a refactor fails loudly."""

    def test_subclass_inherits_jax_solver(self):
        self.assertTrue(issubclass(isdf_xtc_ccsd.RCCSD, jax_xtc_ccsd.RCCSD))

    def test_hook_owned_by_subclass(self):
        self.assertEqual(isdf_xtc_ccsd.RCCSD._contract_vvvv_t2.__module__,
                         "pytc.solver.isdf_xtc_ccsd")

    def test_parent_has_no_hook_attribute(self):
        # The parent's update path falls back to its module-level function via
        # getattr(cc, name, fn); an attribute on the parent class would change
        # what every materialized run resolves to.
        self.assertFalse(hasattr(jax_xtc_ccsd.RCCSD, "_contract_vvvv_t2"))


class FactorizedStateExecutionTest(unittest.TestCase):
    """Call _factorized_state() for real: extraction, fit, caching, guards."""

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="cc-pVDZ", unit="B",
                    verbose=0)
        cls.mf = scf.RHF(mol).density_fit()
        cls.mf.run()
        cls.nocc = mol.nelectron // 2
        cls.nmo = cls.mf.mo_coeff.shape[1]
        cls.rank = 12

    def _make_cc(self, drop_kernel=None):
        fake = _fake_xtc_obj(self.nmo, self.rank)
        if drop_kernel is not None:
            del fake.isdf_kernels[drop_kernel]
        return isdf_xtc_ccsd.RCCSD(self.mf, fake, None, on_the_fly_vvvv=True)

    def test_state_builds_and_caches(self):
        cc = self._make_cc()
        state1 = cc._factorized_state()
        tc, b, fit, x_backing = state1
        nvir = self.nmo - self.nocc
        self.assertEqual(tc["p"].shape, (nvir, self.rank))
        self.assertEqual(tc["grad_p"].shape, (nvir, self.rank, 3))
        self.assertNotIn("x", tc)  # X stays on its backing, streamed by panels
        self.assertIs(x_backing, cc.xtc_obj.isdf_kernels["X"])
        self.assertEqual(x_backing.shape, (self.nmo, self.nmo, self.rank))
        self.assertEqual(b.shape[:2], (nvir, nvir))
        self.assertGreater(b.shape[2], 0)
        for name, arr in tc.items():
            self.assertEqual(arr.dtype, np.float64, name)
        self.assertEqual(b.dtype, np.float64)
        self.assertEqual(fit.p_virtual.shape[0], nvir)
        # The lazy seam caches: a second call returns the same state object.
        self.assertIs(cc._factorized_state(), state1)

    def test_hook_executes_streamed_contraction(self):
        # Execute the hook end-to-end on the synthetic deck: the streamed
        # factor-direct terms plus the JAX sandwich must produce a finite,
        # nonzero t2 update.  This is the seam the 1200-orbital card relies
        # on; identity checks alone cannot prove it runs.
        cc = self._make_cc()
        nvir = self.nmo - self.nocc
        rng = np.random.default_rng(7)
        t2_raw = rng.standard_normal((self.nocc, self.nocc, nvir, nvir))
        t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        eris = types.SimpleNamespace(vvvv=None)
        t2new = np.zeros_like(t2)
        cc._contract_vvvv_t2(cc, t2, eris, t2new)
        self.assertTrue(np.all(np.isfinite(t2new)))
        self.assertGreater(np.linalg.norm(t2new), 0.0)

    def test_missing_kernel_fails_closed_naming_key(self):
        cc = self._make_cc(drop_kernel="K3_kernel")
        with self.assertRaisesRegex(RuntimeError, "K3_kernel"):
            cc._factorized_state()

    def test_hook_refuses_materialized_vvvv(self):
        cc = self._make_cc()
        eris = types.SimpleNamespace(vvvv=np.zeros((1, 1, 1, 1)))
        with self.assertRaisesRegex(RuntimeError, "materialized VVVV"):
            cc._contract_vvvv_t2(cc, None, eris, None)


if __name__ == "__main__":
    unittest.main()
