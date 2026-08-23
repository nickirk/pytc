"""Tile-size parity gate for the K1/K3 ISDF build.

Tiling is a blocking parameter (a device-memory lever), never a numerical
input: r2/host tile boundaries change only the scan accumulation order (FP
reassociation).  Forced small tiles must therefore reproduce the default
auto-sized tiles to reassociation level, gated at 1e-12 relative -- beyond
that, a difference is a real defect, not a tile effect.
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


def _relative_l2(actual, reference):
    return float(jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference))


class KmatTileParityTest(unittest.TestCase):
    """Forced r2/host tiles vs default tiles: same kernels to 1e-12 relative.

    Two independent production-shaped builds (own ISDFXTC object and store
    each, the way the driver runs them), differing only in tile sizes.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="C 0 0 0; O 0 0 1.128", basis="sto-6g", verbose=0)
        cls.mf = scf.RHF(mol).run()
        cls.jastrow = rexp.REXP()
        cls.jastrow_params = {"alpha": jnp.array([0.5])}
        cls.xtc_obj = xtc.XTC.from_pyscf(cls.mf, cls.jastrow, grid_lvl=1)
        cls.n_rank = cls.xtc_obj.n_orb * 12
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _build(self, tag, **isdf_kwargs):
        obj = xtc.ISDFXTC.from_xtc(
            self.xtc_obj, n_rank=self.n_rank,
            save_path=os.path.join(self._tmp.name, f"isdf_{tag}.h5"))
        return obj.isdf(self.jastrow_params, **isdf_kwargs)

    def test_forced_tiles_match_default(self):
        default = self._build("default")
        forced = self._build("forced", host_grid_block_size=8192,
                             r2_tile_size=8192)
        # Sanity: the forced run really used more than one tile on this deck
        # (n_grid=10360 -> 2 tiles), otherwise the gate is vacuous.
        self.assertGreater(self.xtc_obj.grid_points.shape[0], 8192)
        for name in ("K1_kernel", "K3_kernel"):
            with self.subTest(kernel=name):
                self.assertLessEqual(
                    _relative_l2(np.asarray(forced.isdf_kernels[name]),
                                 np.asarray(default.isdf_kernels[name])),
                    1e-12)


if __name__ == "__main__":
    unittest.main()
