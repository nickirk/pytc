import unittest
from unittest import mock
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
            d_reduce_group_blocks=2,
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

    def test_delta_u_tile_assembly_matches_public_api(self):
        kernels = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        ranges = (slice(0, 2), slice(1, 3), slice(0, 2), slice(1, 3))
        assembled = self.isdf_xtc._assemble_delta_u_tile(kernels, ranges)
        public = self.isdf_xtc.get_delta_U(self.jparams, ranges=ranges, batch_size=64)
        np.testing.assert_allclose(
            np.asarray(assembled),
            np.asarray(public),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_delta_u_direct_tile_is_not_chunk_wrapper_for_fitting_tile(self):
        kernels = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        ranges = (slice(0, 2), slice(1, 3), slice(0, 2), slice(1, 3))
        ref = ISDFXTC._contract_delta_U_kernels(self.isdf_xtc, kernels, ranges)

        with mock.patch("pytc.utils.gpu_memory._get_gpu_free_bytes", return_value=10**12):
            with mock.patch.object(
                ISDFXTC,
                "_contract_delta_U_kernels",
                side_effect=AssertionError("direct tile should not route through chunk scheduler"),
            ):
                got = self.isdf_xtc._get_delta_u_direct_tile(kernels, ranges)

        np.testing.assert_allclose(
            np.asarray(got),
            np.asarray(ref),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_delta_u_direct_tile_raises_cleanly_when_tile_is_too_large(self):
        kernels = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        ranges = (slice(0, 2), slice(1, 3), slice(0, 2), slice(1, 3))
        with mock.patch.object(
            ISDFXTC,
            "_contract_delta_U_kernels",
            side_effect=AssertionError("direct tile should not route through chunk scheduler"),
        ):
            with mock.patch("pytc.xtc._get_device_free_bytes", return_value=1):
                with self.assertRaises(RuntimeError):
                    self.isdf_xtc._get_delta_u_direct_tile(kernels, ranges)

    def test_delta_u_direct_tile_padding_trims_back_to_reference(self):
        kernels = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        ranges = (slice(0, 1), slice(1, 3), slice(0, 1), slice(1, 3))
        ref = self.isdf_xtc._get_delta_u_direct_tile(kernels, ranges)
        padded = self.isdf_xtc._get_delta_u_direct_tile(kernels, ranges, panel_size=2)
        np.testing.assert_allclose(
            np.asarray(padded)[:1, :, :1, :],
            np.asarray(ref),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_isdf_device_cache_reuses_phi_grad_and_d(self):
        kernels = self.isdf_xtc.compute_delta_u_kernels(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        device = jax.devices("cpu")[0]
        cache_a = self.isdf_xtc._get_isdf_device_cache(
            kernels, device=device, include_grad=True, include_delta_u=True
        )
        cache_b = self.isdf_xtc._get_isdf_device_cache(
            kernels, device=device, include_grad=True, include_delta_u=True
        )
        self.assertIs(cache_a["phi_isdf"], cache_b["phi_isdf"])
        self.assertIs(cache_a["grad_phi_isdf"], cache_b["grad_phi_isdf"])
        self.assertIs(cache_a["D"], cache_b["D"])

    def test_isdf_device_cache_reuses_tc_kernels(self):
        isdf_xtc = self.isdf_xtc.isdf(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        kernels = isdf_xtc.isdf_kernels
        device = jax.devices("cpu")[0]
        cache_a = isdf_xtc._get_isdf_device_cache(
            kernels, device=device, include_grad=True, include_tc=True
        )
        cache_b = isdf_xtc._get_isdf_device_cache(
            kernels, device=device, include_grad=True, include_tc=True
        )
        self.assertIs(cache_a["K1_kernel"], cache_b["K1_kernel"])
        self.assertIs(cache_a["K3_kernel"], cache_b["K3_kernel"])

    def test_2b_tile_assembly_matches_public_api(self):
        isdf_xtc = self.isdf_xtc.isdf(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        kernels = isdf_xtc.isdf_kernels
        ranges = (slice(0, 2), slice(1, 3), slice(0, 2), slice(1, 3))
        assembled = isdf_xtc._assemble_2b_tile(self.jparams, kernels, ranges)
        public = isdf_xtc.get_2b(self.jparams, ranges=ranges, batch_size=64)
        np.testing.assert_allclose(
            np.asarray(assembled),
            np.asarray(public),
            atol=1e-10,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
