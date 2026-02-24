import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.jastrow.rexp import REXP
from pytc.xtc import XTC, ISDFXTC


jax.config.update("jax_enable_x64", True)


class TestISDFXTCPanelization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.kernel()

        jastrow = REXP()
        cls.jparams = {"alpha": jnp.array([1.0])}

        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=0)
        n_rank = max(8, 3 * xtc.n_orb)
        cls.isdf_xtc = ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True)

    def test_x_s_panel_blocks_matches_baseline(self):
        batch_size = 64
        orb_block_size = 2
        host_grid_block_size = 512

        l_aux = self.isdf_xtc._compute_L_aux(
            self.jparams,
            batch_size=batch_size,
            host_grid_block_size=host_grid_block_size,
        )

        kernels_ref = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=batch_size,
            L_aux=l_aux,
            orb_block_size=orb_block_size,
            host_grid_block_size=host_grid_block_size,
            x_s_panel_blocks=1,
        )
        kernels_panel = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=batch_size,
            L_aux=l_aux,
            orb_block_size=orb_block_size,
            host_grid_block_size=host_grid_block_size,
            x_s_panel_blocks=2,
        )

        np.testing.assert_allclose(
            np.asarray(kernels_panel["D"]),
            np.asarray(kernels_ref["D"]),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(kernels_panel["X"]),
            np.asarray(kernels_ref["X"]),
            atol=1e-9,
            rtol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
