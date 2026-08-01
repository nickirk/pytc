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

    def test_state_prefers_rank_major_x_rm_when_present(self):
        # Stores carrying the rank-major twin serve the contraction from
        # X_rm (panel-contiguous reads); X stays for legacy consumers.
        cc = self._make_cc()
        x_rm = np.ascontiguousarray(
            np.asarray(cc.xtc_obj.isdf_kernels["X"]).transpose(2, 0, 1))
        cc.xtc_obj.isdf_kernels["X_rm"] = x_rm
        _, _, _, x_backing = cc._factorized_state()
        self.assertIs(x_backing, x_rm)
        self.assertEqual(x_backing.shape, (self.rank, self.nmo, self.nmo))
        # And the layout detector agrees with the preference.
        from pytc.solver import factor_direct_vvvv
        self.assertEqual(
            factor_direct_vvvv._x_backing_layout(
                x_rm, self.nocc, self.nmo - self.nocc, self.rank),
            "rank_major")

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


class PreloadSuppressionTest(unittest.TestCase):
    """The whole-X host preload fires on the materialized parent but is
    suppressed on the factorized subclass, with identical X content.

    At the 1200-orbital deck the preload is 247 GB of host RAM the streamed
    contraction never reads; the subclass suppresses it by construction
    (``_preload_x_for_eris = False``) while the parent's materialized default
    stays byte-identical.  X CONTENT is identical either way -- the store
    bytes are the store bytes; only the backing changes.
    """

    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import tempfile
        from pytc import xtc
        from pytc.jastrow import rexp
        cls._tmp = tempfile.TemporaryDirectory()
        mol = gto.M(atom="C 0 0 0; O 0 0 1.128", basis="sto-6g", verbose=0)
        cls.mf = scf.RHF(mol).density_fit()
        cls.mf.run()
        cls.jastrow = rexp.REXP()
        cls.jastrow_params = {"alpha": jnp.array([0.5])}
        cls.xtc_obj = xtc.XTC.from_pyscf(cls.mf, cls.jastrow, grid_lvl=1)
        n_rank = cls.xtc_obj.n_orb * 12
        import os
        cls.store = os.path.join(cls._tmp.name, "isdf_test.h5")
        cls.isdf_xtc = xtc.ISDFXTC.from_xtc(
            cls.xtc_obj, n_rank=n_rank, save_path=cls.store)
        cls.isdf_xtc = cls.isdf_xtc.isdf(cls.jastrow_params)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_subclass_suppresses_preload_parent_keeps_it(self):
        import h5py
        # Parent (materialized default): the preload fires, X becomes a host
        # numpy array in the kernels dict.
        cc_parent = jax_xtc_ccsd.RCCSD(self.mf, self.isdf_xtc,
                                       self.jastrow_params,
                                       on_the_fly_vvvv=True)
        cc_parent.ao2mo()
        x_parent = cc_parent.xtc_obj.isdf_kernels["X"]
        self.assertIsInstance(x_parent, np.ndarray)

        # Subclass (factorized): suppression holds, X stays store-backed.
        cc_sub = isdf_xtc_ccsd.RCCSD(self.mf, self.isdf_xtc,
                                     self.jastrow_params,
                                     on_the_fly_vvvv=True)
        cc_sub.ao2mo()
        x_sub = cc_sub.xtc_obj.isdf_kernels["X"]
        self.assertIsInstance(x_sub, h5py.Dataset)

        # X CONTENT is identical either way (the ISDF fingerprint gate).
        np.testing.assert_array_equal(np.asarray(x_parent),
                                      np.asarray(x_sub[:]))


if __name__ == "__main__":
    unittest.main()
