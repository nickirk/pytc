import unittest
from unittest import mock
import os
import tempfile

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc import xtc as xtc_module
from pytc.jastrow.rexp import REXP
from pytc.tc import ISDFTC
from pytc.xtc import XTC, ISDFXTC
from pytc.solver import jax_xtc_ccsd


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
        cls.mf = mf

        jastrow = REXP()
        cls.jparams = {"alpha": jnp.array([1.0])}

        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=0)
        n_rank = max(8, 3 * xtc.n_orb)
        cls.isdf_xtc = ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True)

    def test_drop_x_preserves_direct_kernel_and_zeroes_exchange(self):
        """The no-X study lever only removes the exchange kernel on a real toy system."""
        kwargs = dict(
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        l_aux = self.isdf_xtc._compute_L_aux(
            self.jparams,
            batch_size=kwargs["batch_size"],
            host_grid_block_size=kwargs["host_grid_block_size"],
        )
        with mock.patch.dict(os.environ, {"PYTC_XTC_DROP_X": "0"}):
            full = self.isdf_xtc.compute_delta_u_kernels(
                self.jparams, L_aux=l_aux, **kwargs
            )
        with mock.patch.dict(os.environ, {"PYTC_XTC_DROP_X": "1"}):
            no_x = self.isdf_xtc.compute_delta_u_kernels(
                self.jparams, L_aux=l_aux, **kwargs
            )

        np.testing.assert_array_equal(np.asarray(no_x["D"]), np.asarray(full["D"]))
        self.assertGreater(np.linalg.norm(np.asarray(full["X"])), 0.0)
        np.testing.assert_array_equal(np.asarray(no_x["X"]), np.zeros_like(no_x["X"]))
        full_delta_u = self.isdf_xtc.replace(isdf_kernels=full).get_delta_U(
            self.jparams
        )
        no_x_delta_u = self.isdf_xtc.replace(isdf_kernels=no_x).get_delta_U(
            self.jparams
        )
        self.assertGreater(
            np.linalg.norm(np.asarray(full_delta_u - no_x_delta_u)), 0.0,
            "The exchange kernel must affect the downstream Delta-U integral.",
        )

    def test_aux_recovered_kernels_preserve_h2_xtc_normal_order(self):
        """Exact auxiliary K recovery must leave the full H2 XTC view intact."""
        kwargs = dict(batch_size=64, orb_block_size=2, host_grid_block_size=512)
        clean_env = {
            "PYTC_XTC_DROP_X": "0",
            "PYTC_XTC_DROP_X_NORMAL_ORDER": "0",
            "PYTC_XTC_DROP_X_RESIDUAL": "0",
        }
        with mock.patch.dict(os.environ, clean_env):
            direct = self.isdf_xtc.isdf(self.jparams, **kwargs)
            recovered = self.isdf_xtc.isdf(
                self.jparams, reuse_aux_kernels=True, **kwargs
            )

            np.testing.assert_allclose(
                np.asarray(recovered.isdf_kernels["K1_kernel"]),
                np.asarray(direct.isdf_kernels["K1_kernel"]),
                rtol=0,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                np.asarray(recovered.isdf_kernels["K3_kernel"]),
                np.asarray(direct.isdf_kernels["K3_kernel"]),
                rtol=0,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                np.asarray(recovered.get_2b(self.jparams)),
                np.asarray(direct.get_2b(self.jparams)),
                rtol=0,
                atol=2e-12,
            )

            direct_h = direct.get_delta_h(self.jparams)
            recovered_h = recovered.get_delta_h(self.jparams)
            np.testing.assert_allclose(
                np.asarray(recovered_h), np.asarray(direct_h), rtol=0, atol=2e-12
            )
            np.testing.assert_allclose(
                np.asarray(recovered.get_delta_U(self.jparams)),
                np.asarray(direct.get_delta_U(self.jparams)),
                rtol=0,
                atol=2e-12,
            )
            self.assertAlmostEqual(
                float(recovered.get_const(self.jparams, delta_h=recovered_h)),
                float(direct.get_const(self.jparams, delta_h=direct_h)),
                places=12,
            )

    def test_aux_reuse_streams_out_of_core_xtc_normal_order(self):
        """The production path streams HDF5 auxiliary panels and preserves XTC."""
        kwargs = dict(batch_size=64, orb_block_size=2, host_grid_block_size=127)
        clean_env = {
            "PYTC_XTC_DROP_X": "0",
            "PYTC_XTC_DROP_X_NORMAL_ORDER": "0",
            "PYTC_XTC_DROP_X_RESIDUAL": "0",
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, clean_env):
            direct = self.isdf_xtc.isdf(self.jparams, **kwargs)
            store_path = os.path.join(tmpdir, "h2_aux_streamed.h5")
            # Match a genuine out-of-core ISDF object: xi lives only in the
            # persistent store and the object carries None for both fields.
            with h5py.File(store_path, "w") as f:
                f.create_dataset("xi_phi", data=np.asarray(self.isdf_xtc.xi_phi))
                f.create_dataset("xi_grad", data=np.asarray(self.isdf_xtc.xi_grad))
            out_of_core = self.isdf_xtc.replace(
                is_incore=False,
                xi_phi=None,
                xi_grad=None,
                save_path=store_path,
            )
            recovered = out_of_core.isdf(
                self.jparams,
                save_path=store_path,
                reuse_aux_kernels=True,
                **kwargs,
            )

            np.testing.assert_allclose(
                np.asarray(recovered.isdf_kernels["K1_kernel"]),
                np.asarray(direct.isdf_kernels["K1_kernel"]),
                rtol=0,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                np.asarray(recovered.isdf_kernels["K3_kernel"]),
                np.asarray(direct.isdf_kernels["K3_kernel"]),
                rtol=0,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                np.asarray(recovered.get_delta_U(self.jparams)),
                np.asarray(direct.get_delta_U(self.jparams)),
                rtol=0,
                atol=2e-12,
            )
            with h5py.File(store_path, "r") as f:
                self.assertIn("L_aux", f)
                self.assertNotIn("H_aux", f)
                self.assertEqual(f.attrs["pytc_kmat_kernel_mode"], "aux-recovery")

    def test_x_normal_order_and_residual_switches_are_independent(self):
        """A full X store can isolate normal-order and residual-X effects."""
        kwargs = dict(batch_size=64, orb_block_size=2, host_grid_block_size=512)
        l_aux = self.isdf_xtc._compute_L_aux(
            self.jparams,
            batch_size=kwargs["batch_size"],
            host_grid_block_size=kwargs["host_grid_block_size"],
        )
        clean_env = {
            "PYTC_XTC_DROP_X": "0",
            "PYTC_XTC_DROP_X_NORMAL_ORDER": "0",
            "PYTC_XTC_DROP_X_RESIDUAL": "0",
        }
        with mock.patch.dict(os.environ, clean_env):
            full = self.isdf_xtc.compute_delta_u_kernels(
                self.jparams, L_aux=l_aux, **kwargs
            )

        no_x = {"D": full["D"], "X": np.zeros_like(np.asarray(full["X"]))}
        full_obj = self.isdf_xtc.replace(isdf_kernels=full)
        no_x_obj = self.isdf_xtc.replace(isdf_kernels=no_x)
        ranges = (slice(0, 2), slice(1, 3), slice(0, 2), slice(1, 3))

        with mock.patch.dict(os.environ, clean_env):
            full_h = full_obj.get_delta_h(self.jparams)
            full_e0 = full_obj.get_const(self.jparams, delta_h=full_h)
            full_du = full_obj.get_delta_U(self.jparams, ranges=ranges)
            no_x_h = no_x_obj.get_delta_h(self.jparams)
            no_x_e0 = no_x_obj.get_const(self.jparams, delta_h=no_x_h)
            no_x_du = no_x_obj.get_delta_U(self.jparams, ranges=ranges)

        with mock.patch.dict(
            os.environ, {**clean_env, "PYTC_XTC_DROP_X_NORMAL_ORDER": "1"}
        ):
            normal_order_dropped_h = full_obj.get_delta_h(self.jparams)
            normal_order_dropped_e0 = full_obj.get_const(
                self.jparams, delta_h=normal_order_dropped_h
            )
            normal_order_dropped_du = full_obj.get_delta_U(
                self.jparams, ranges=ranges
            )

        with mock.patch.dict(
            os.environ, {**clean_env, "PYTC_XTC_DROP_X_RESIDUAL": "1"}
        ):
            residual_dropped_h = full_obj.get_delta_h(self.jparams)
            residual_dropped_e0 = full_obj.get_const(
                self.jparams, delta_h=residual_dropped_h
            )
            residual_dropped_du = full_obj.get_delta_U(
                self.jparams, ranges=ranges
            )
            residual_dropped_direct = full_obj._assemble_delta_u_tile(full, ranges)

        # Each narrow switch exactly reproduces its corresponding component of
        # the all-X-dropped Hamiltonian *and* leaves the complementary
        # component unchanged within FP64 tolerance.
        np.testing.assert_allclose(
            normal_order_dropped_h, no_x_h, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            normal_order_dropped_e0, no_x_e0, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            residual_dropped_du, no_x_du, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            residual_dropped_direct, no_x_du, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            normal_order_dropped_du, full_du, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            residual_dropped_h, full_h, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            residual_dropped_e0, full_e0, atol=1e-10, rtol=1e-10
        )

        # Mutation oracles: prove the complement assertions reject either
        # possible broadened switch, rather than merely passing on today's
        # implementation.
        real_drop_residual = xtc_module._drop_x_from_residual_integrals
        with mock.patch.object(
            xtc_module,
            "_drop_x_from_residual_integrals",
            side_effect=lambda: (
                real_drop_residual()
                or os.environ.get("PYTC_XTC_DROP_X_NORMAL_ORDER") == "1"
            ),
        ), mock.patch.dict(
            os.environ, {**clean_env, "PYTC_XTC_DROP_X_NORMAL_ORDER": "1"}
        ):
            broadened_normal_du = full_obj.get_delta_U(self.jparams, ranges=ranges)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                broadened_normal_du, full_du, atol=1e-10, rtol=1e-10
            )

        real_drop_normal = xtc_module._drop_x_from_normal_order
        with mock.patch.object(
            xtc_module,
            "_drop_x_from_normal_order",
            side_effect=lambda: (
                real_drop_normal()
                or os.environ.get("PYTC_XTC_DROP_X_RESIDUAL") == "1"
            ),
        ), mock.patch.dict(
            os.environ, {**clean_env, "PYTC_XTC_DROP_X_RESIDUAL": "1"}
        ):
            broadened_residual_h = full_obj.get_delta_h(self.jparams)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                broadened_residual_h, full_h, atol=1e-10, rtol=1e-10
            )

        self.assertGreater(np.linalg.norm(np.asarray(full_h - no_x_h)), 1e-10)
        self.assertGreater(abs(float(full_e0 - no_x_e0)), 1e-10)
        self.assertGreater(np.linalg.norm(np.asarray(full_du - no_x_du)), 1e-10)

    def test_x_partition_switches_match_literal_zero_x_end_to_end(self):
        """The JAX-CCSD split agrees with an explicit zero-X Hamiltonian."""
        clean_env = {
            "PYTC_XTC_DROP_X": "0",
            "PYTC_XTC_DROP_X_NORMAL_ORDER": "0",
            "PYTC_XTC_DROP_X_RESIDUAL": "0",
        }
        kwargs = dict(batch_size=64, orb_block_size=2, host_grid_block_size=512)
        with mock.patch.dict(os.environ, clean_env):
            full_obj = self.isdf_xtc.isdf(self.jparams, **kwargs)

        zero_x_kernels = dict(full_obj.isdf_kernels)
        zero_x_kernels["X"] = np.zeros_like(np.asarray(zero_x_kernels["X"]))
        literal_zero_x_obj = full_obj.replace(isdf_kernels=zero_x_kernels)

        def solve(obj, drop_normal_order, drop_residual):
            env = {
                **clean_env,
                "PYTC_XTC_DROP_X_NORMAL_ORDER": "1" if drop_normal_order else "0",
                "PYTC_XTC_DROP_X_RESIDUAL": "1" if drop_residual else "0",
            }
            with mock.patch.dict(os.environ, env):
                cc = jax_xtc_ccsd.RCCSD(
                    self.mf, obj, self.jparams,
                    max_memory=2_000, gpu_max_memory=2_000,
                    on_the_fly_vvvv=True,
                )
                cc.max_cycle = 50
                cc.kernel()
                energy = float(cc.e_tot)
                if hasattr(cc, "eris") and hasattr(cc.eris, "close"):
                    cc.eris.close()
                return energy

        full = solve(full_obj, False, False)
        drop_normal = solve(full_obj, True, False)
        drop_residual = solve(full_obj, False, True)
        drop_all = solve(full_obj, True, True)
        literal_zero_x = solve(literal_zero_x_obj, False, False)

        self.assertAlmostEqual(drop_all, literal_zero_x, places=10)
        self.assertGreater(abs(full - drop_normal), 1e-10)
        self.assertGreater(abs(full - drop_residual), 1e-10)
        self.assertGreater(abs(drop_normal - drop_all), 1e-10)
        self.assertGreater(abs(drop_residual - drop_all), 1e-10)

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

    def test_incomplete_kernel_store_closes_read_handle(self):
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        try:
            with h5py.File(path, "w") as store:
                store.create_dataset("D", data=np.zeros((1, 1)))

            opened_reads = []
            real_file = h5py.File

            def tracking_file(name, mode="r", *args, **kwargs):
                handle = real_file(name, mode, *args, **kwargs)
                if mode == "r":
                    opened_reads.append(handle)
                return handle

            base = self.isdf_xtc.replace(isdf_kernels={})
            with mock.patch.object(ISDFTC, "isdf", return_value=base):
                with mock.patch.object(
                    ISDFXTC,
                    "compute_delta_u_kernels",
                    return_value={"D": np.zeros((1, 1)), "X": np.zeros((1, 1, 1))},
                ):
                    with mock.patch("pytc.xtc.h5py.File", side_effect=tracking_file):
                        self.isdf_xtc.isdf(self.jparams, save_path=path)

            self.assertEqual(len(opened_reads), 1)
            self.assertFalse(opened_reads[0].id.valid)
        finally:
            os.remove(path)

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

    def test_tc_direct_tile_panel_padding_slice_p_eq_slice_q(self):
        """Regression: ``_get_tc_direct_tile`` must tolerate panel padding
        on only one of p/q when ``slice_p == slice_q``.

        Before the fix, the antisymmetrization shortcut
        ``k12 - k12.transpose(1, 0, 2, 3)`` crashed with
        ``sub got incompatible shapes for broadcasting`` whenever the
        panel_layout padded one side of the (p, q) pair but not the
        other — which happens in the oovv medium block when
        ``nocc < panel_blk`` (layout ``"pr"`` pads p → panel_size while
        q stays at nocc).  The slice length must be ≥ 2 to turn the bug
        into a hard crash instead of a silent NumPy broadcast.
        """
        isdf_xtc = self.isdf_xtc.isdf(
            self.jparams,
            batch_size=64,
            orb_block_size=2,
            host_grid_block_size=512,
        )
        kernels = isdf_xtc.isdf_kernels

        # slice_p == slice_q → antisymmetrization branch.  Use a length-2
        # slice so panel_size=3 is strictly larger AND the transposed
        # shape (2, 3, ...) does not broadcast against (3, 2, ...) — this
        # is the actual bug mode seen in production (nocc=21, panel=22).
        pq_slice = slice(0, 2)
        r_slice  = slice(0, 2)
        s_slice  = slice(0, 2)
        ranges = (pq_slice, pq_slice, r_slice, s_slice)
        ref = isdf_xtc._get_tc_direct_tile(kernels, ranges)

        # --- "pr" layout: pads p (axis 0) and r (axis 2) only ------------
        # Before the fix this call raised TypeError from k12 - k12.T with
        # shapes (3, 2, 3, 2) vs (2, 3, 3, 2).
        pr_padded = isdf_xtc._get_tc_direct_tile(
            kernels, ranges, panel_size=3, panel_layout="pr")
        # Convention: axes listed in the layout are padded, others are not.
        self.assertEqual(np.asarray(pr_padded).shape, (3, 2, 3, 2))
        np.testing.assert_allclose(
            np.asarray(pr_padded)[:2, :2, :2, :],
            np.asarray(ref),
            atol=1e-10, rtol=1e-10,
        )

        # --- "qr" layout: pads q (axis 1) and r (axis 2) only ------------
        # Symmetric case — exercises the branch where phi_q is the padded
        # side and phi_p gets re-padded inside the fix.
        qr_padded = isdf_xtc._get_tc_direct_tile(
            kernels, ranges, panel_size=3, panel_layout="qr")
        self.assertEqual(np.asarray(qr_padded).shape, (2, 3, 3, 2))
        np.testing.assert_allclose(
            np.asarray(qr_padded)[:2, :2, :2, :],
            np.asarray(ref),
            atol=1e-10, rtol=1e-10,
        )

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


class TestAssembleTileShortcutAsymmetricPadding(unittest.TestCase):
    """Regression tests for the ``direct + direct.transpose(2,3,0,1)``
    shortcut in ``_assemble_tc_tile`` and ``_assemble_delta_u_tile``.

    The shortcut is taken when ``slice_p == slice_r and slice_q == slice_s``
    and was assumed always safe.  It is **only** safe when the panel
    padding is invariant under the ``(p↔r, q↔s)`` axis swap — i.e. when
    padded-axis set equals ``{0, 2}`` (``"pr"``).  For ``"qr"`` (pads
    ``{1, 2}``) and ``"ps"`` (pads ``{0, 3}``) the shortcut broadcast-adds
    mis-shaped tensors.

    Production manifestation (collaborator's aug-cc-pVDZ benzene run,
    nocc=14, nvir=114, panel_blk=114)::

        add got incompatible shapes for broadcasting:
            (14, 114, 114, 114), (114, 114, 14, 114)

    which is exactly the ovov single-tile case (``slice_q == slice_s``
    when ``i_len == nvir``) with panel_layout ``"qr"``.

    To trigger the crash (not a silent size-1 broadcast) the test fixture
    needs ``nocc >= 2`` **and** ``nvir >= 2``.  H2/sto-3g (nocc=nvir=1)
    silently broadcasts size-1 axes; LiH/sto-3g (nocc=2, nvir=4) forces
    a hard broadcast failure and distinct numerical checks.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.595",
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
        base_isdf_xtc = ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True)
        cls.isdf_xtc = base_isdf_xtc.isdf(
            cls.jparams, batch_size=64, orb_block_size=2,
            host_grid_block_size=512,
        )
        cls.kernels = cls.isdf_xtc.isdf_kernels

        nmo = int(cls.isdf_xtc.phi_isdf.shape[0])
        nocc = int(cls.isdf_xtc.nocc)
        cls.nocc = nocc
        cls.nvir = nmo - nocc
        cls.nmo  = nmo
        # Sanity: need both nocc and nvir >= 2 for hard-failure coverage.
        assert cls.nocc >= 2 and cls.nvir >= 2, (
            f"fixture must have nocc>=2 and nvir>=2, got "
            f"nocc={cls.nocc} nvir={cls.nvir}"
        )


    def _ovov_ranges(self):
        """ovov single-tile ranges: ``slice_p==slice_r``, ``slice_q==slice_s``."""
        return (
            slice(0, self.nocc), slice(self.nocc, self.nmo),
            slice(0, self.nocc), slice(self.nocc, self.nmo),
        )

    def _vovo_ranges(self):
        """vovo single-tile ranges: same slice-symmetry pattern."""
        return (
            slice(self.nocc, self.nmo), slice(0, self.nocc),
            slice(self.nocc, self.nmo), slice(0, self.nocc),
        )

    def _trim_padded(self, padded, layout, p_len, q_len, r_len, s_len):
        """Slice a padded tile back to its natural (unpadded) extents."""
        padded = np.asarray(padded)
        if layout == "pr":
            return padded[:p_len, :q_len, :r_len, :s_len]
        if layout == "qr":
            return padded[:p_len, :q_len, :r_len, :s_len]
        return padded[:p_len, :q_len, :r_len, :s_len]

    def _padded_tile_shape(self, layout, p_len, q_len, r_len, s_len, ps):
        if layout == "pr":
            return (ps,    q_len, ps,    s_len)
        if layout == "qr":
            return (p_len, ps,    ps,    s_len)
        return     (ps,    q_len, r_len, ps)


    def _check_assemble_tc_tile_all_layouts(self, ranges, tag):
        p_len = ranges[0].stop - ranges[0].start
        q_len = ranges[1].stop - ranges[1].start
        r_len = ranges[2].stop - ranges[2].start
        s_len = ranges[3].stop - ranges[3].start
        ps = self.nmo  # strictly larger than every natural extent

        ref = np.asarray(self.isdf_xtc._assemble_tc_tile(self.kernels, ranges))
        self.assertEqual(ref.shape, (p_len, q_len, r_len, s_len))

        for layout in ("pr", "qr", "ps"):
            padded = self.isdf_xtc._assemble_tc_tile(
                self.kernels, ranges, panel_size=ps, panel_layout=layout,
            )
            self.assertEqual(
                np.asarray(padded).shape,
                self._padded_tile_shape(layout, p_len, q_len, r_len, s_len, ps),
                f"{tag}/_assemble_tc_tile: unexpected padded shape for "
                f"layout={layout!r}",
            )
            trimmed = self._trim_padded(padded, layout, p_len, q_len, r_len, s_len)
            np.testing.assert_allclose(
                trimmed, ref, atol=1e-10, rtol=1e-10,
                err_msg=f"{tag}/_assemble_tc_tile mismatch for layout={layout!r}",
            )

    def test_assemble_tc_tile_ovov_single_tile_all_layouts(self):
        self._check_assemble_tc_tile_all_layouts(self._ovov_ranges(), "ovov")

    def test_assemble_tc_tile_vovo_single_tile_all_layouts(self):
        self._check_assemble_tc_tile_all_layouts(self._vovo_ranges(), "vovo")


    def _check_assemble_delta_u_tile_all_layouts(self, ranges, tag):
        p_len = ranges[0].stop - ranges[0].start
        q_len = ranges[1].stop - ranges[1].start
        r_len = ranges[2].stop - ranges[2].start
        s_len = ranges[3].stop - ranges[3].start
        ps = self.nmo

        ref = np.asarray(
            self.isdf_xtc._assemble_delta_u_tile(self.kernels, ranges)
        )
        self.assertEqual(ref.shape, (p_len, q_len, r_len, s_len))

        for layout in ("pr", "qr", "ps"):
            padded = self.isdf_xtc._assemble_delta_u_tile(
                self.kernels, ranges, panel_size=ps, panel_layout=layout,
            )
            self.assertEqual(
                np.asarray(padded).shape,
                self._padded_tile_shape(layout, p_len, q_len, r_len, s_len, ps),
                f"{tag}/_assemble_delta_u_tile: unexpected padded shape for "
                f"layout={layout!r}",
            )
            trimmed = self._trim_padded(padded, layout, p_len, q_len, r_len, s_len)
            np.testing.assert_allclose(
                trimmed, ref, atol=1e-10, rtol=1e-10,
                err_msg=f"{tag}/_assemble_delta_u_tile mismatch for "
                        f"layout={layout!r}",
            )

    def test_assemble_delta_u_tile_ovov_single_tile_all_layouts(self):
        self._check_assemble_delta_u_tile_all_layouts(
            self._ovov_ranges(), "ovov")

    def test_assemble_delta_u_tile_vovo_single_tile_all_layouts(self):
        self._check_assemble_delta_u_tile_all_layouts(
            self._vovo_ranges(), "vovo")


    def test_assemble_2b_tile_ovov_single_tile_all_layouts(self):
        ranges = self._ovov_ranges()
        p_len, q_len, r_len, s_len = (self.nocc, self.nvir, self.nocc, self.nvir)
        ps = self.nmo

        ref = np.asarray(
            self.isdf_xtc._assemble_2b_tile(
                self.jparams, self.kernels, ranges,
            )
        )
        self.assertEqual(ref.shape, (p_len, q_len, r_len, s_len))

        for layout in ("pr", "qr", "ps"):
            padded = self.isdf_xtc._assemble_2b_tile(
                self.jparams, self.kernels, ranges,
                panel_size=ps, panel_layout=layout,
            )
            self.assertEqual(
                np.asarray(padded).shape,
                self._padded_tile_shape(layout, p_len, q_len, r_len, s_len, ps),
                f"ovov/_assemble_2b_tile: unexpected padded shape for "
                f"layout={layout!r}",
            )
            trimmed = self._trim_padded(padded, layout, p_len, q_len, r_len, s_len)
            np.testing.assert_allclose(
                trimmed, ref, atol=1e-10, rtol=1e-10,
                err_msg=f"ovov/_assemble_2b_tile mismatch for "
                        f"layout={layout!r}",
            )


if __name__ == "__main__":
    unittest.main()
