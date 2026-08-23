"""Gauge-sync ordering gate for ISDF store reuse.

The fail-closed gauge guard in ISDFXTC.from_xtc refuses to consume a fresh
mf whose gauge differs from the cache -- so a reuse run MUST call
sync_mf_from_cache BEFORE XTC.from_pyscf, not after.  This test builds a
real store, flips the fresh gauge, and proves: (1) without the sync, the
guard fires (exactly the production failure); (2) with the sync placed
before any mo_coeff consumer, the guard passes and the gauge is adopted
from the cache.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import rexp
from pytc.utils import cache_state


class GaugeReuseTest(unittest.TestCase):
    """sync_mf_from_cache must run BEFORE XTC.from_pyscf consumes the gauge."""

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="C 0 0 0; O 0 0 1.128", basis="sto-6g", verbose=0)
        cls.mol = mol
        cls.mf = scf.RHF(mol).density_fit()
        cls.mf.run()
        cls.jastrow = rexp.REXP()
        cls.jastrow_params = {"alpha": jnp.array([0.5])}
        cls.xtc_obj = xtc.XTC.from_pyscf(cls.mf, cls.jastrow, grid_lvl=1)
        cls.n_rank = cls.xtc_obj.n_orb * 12
        cls._tmp = tempfile.TemporaryDirectory()
        cls.store = os.path.join(cls._tmp.name, "isdf_gauge_test.h5")
        obj = xtc.ISDFXTC.from_xtc(cls.xtc_obj, n_rank=cls.n_rank,
                                   save_path=cls.store)
        obj.isdf(cls.jastrow_params)
        assert cache_state.cache_has_mf_state(cls.store), (
            "test fixture broken: the store carries no mf state to sync from")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _fresh_flipped_mf(self):
        mf = scf.RHF(self.mol).density_fit()
        mf.run()
        mf.mo_coeff[:, 1] *= -1.0   # gauge flip: sign-change one orbital
        return mf

    def test_unsynced_flipped_gauge_is_refused(self):
        mf = self._fresh_flipped_mf()
        xo = xtc.XTC.from_pyscf(mf, self.jastrow, grid_lvl=1)
        with self.assertRaises(ValueError):
            xtc.ISDFXTC.from_xtc(xo, n_rank=self.n_rank,
                                 save_path=self.store)

    def test_synced_flipped_gauge_passes_and_adopts_cache(self):
        mf = self._fresh_flipped_mf()
        # The reuse-ordering contract: sync BEFORE any mo_coeff consumer.
        cache_state.sync_mf_from_cache(mf, self.store)
        # Gauge adopted from the cache (the original run's gauge), not merely
        # "no error": the flipped column is restored to the cached gauge.
        np.testing.assert_allclose(mf.mo_coeff, self.mf.mo_coeff,
                                   rtol=0.0, atol=1e-12)
        xo = xtc.XTC.from_pyscf(mf, self.jastrow, grid_lvl=1)
        obj = xtc.ISDFXTC.from_xtc(xo, n_rank=self.n_rank,
                                   save_path=self.store)
        self.assertIsNotNone(obj)


if __name__ == "__main__":
    unittest.main()
