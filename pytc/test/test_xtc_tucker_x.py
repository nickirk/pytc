"""Algebra and streaming-build controls for orbital-leg Tucker X."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc as xtc_mod
from pytc.jastrow.rexp import REXP
from pytc.solver import jax_xtc_ccsd


jax.config.update("jax_enable_x64", True)


class _FakeTuckerBuild:
    """Minimal ISDFXTC-shaped object backed by a synthetic dense X oracle."""

    def __init__(self, x):
        self.x = np.asarray(x)
        self.n_orb = self.x.shape[0]
        self.phi_isdf = np.eye(self.n_orb)
        self.panel_shapes = []

    def _get_mf_dm(self):
        return np.eye(self.n_orb)

    def _compute_L_aux(self, *args, **kwargs):
        return object()

    def _compute_X_kernel(
        self, _params, ranges, _batch_size, _l_aux, *, Gb=None, L_Q=None,
        host_grid_block_size=None, orbital_rows=None,
    ):
        del Gb, L_Q, host_grid_block_size
        r_slice, s_slice = ranges[2], ranges[3]
        if orbital_rows is None:
            panel = self.x[r_slice, s_slice]
        else:
            # phi_isdf is identity, so projected rows are exactly U.T.
            rows = np.asarray(orbital_rows)
            panel = np.einsum(
                "ra,abc,sb->rsc", rows[r_slice], self.x, rows[s_slice],
                optimize=True,
            )
        self.panel_shapes.append(panel.shape)
        return panel


class _FakeTuckerRuntime:
    """Minimal runtime object for the dense-vs-factor solver entry points."""

    def __init__(self, phi, dm1):
        self.phi_isdf = jnp.asarray(phi)
        self.n_orb = phi.shape[0]
        self.dm1 = jnp.asarray(dm1)
        self.gpu_max_memory = None

    def _get_mf_dm(self):
        return self.dm1

    def _get_fixed_rank_block_size(self):
        return 2

    def _contract_delta_U_kernels(self, kernels, ranges):
        return xtc_mod.ISDFXTC._contract_delta_U_kernels(self, kernels, ranges)


class TestTuckerXAlgebra(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.n_orb = 7
        self.n_rank = 5
        x = rng.normal(size=(self.n_orb, self.n_orb, self.n_rank))
        self.x = 0.5 * (x + x.swapaxes(0, 1))
        self.u, _ = np.linalg.qr(rng.normal(size=(self.n_orb, self.n_orb)))
        self.z = np.einsum("ra,rsc,sb->abc", self.u, self.x, self.u,
                           optimize=True)

    def test_factor_direct_residual_matches_dense_x(self):
        rng = np.random.default_rng(18)
        phi_p = rng.normal(size=(3, self.n_rank))
        phi_q = rng.normal(size=(2, self.n_rank))
        r_idx = np.array([1, 3, 4])
        s_idx = np.array([0, 2])

        expected = -np.einsum(
            "pc,qc,rsc->pqrs", phi_p, phi_q, self.x[r_idx][:, s_idx],
            optimize=True,
        )
        got = xtc_mod._contract_tucker_x_residual(
            phi_p, phi_q, self.u[r_idx], self.u[s_idx], self.z,
        )
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-11,
                                   rtol=1e-11)

    def test_normal_order_intermediates_match_dense_x(self):
        rng = np.random.default_rng(19)
        dm1 = rng.normal(size=(self.n_orb, self.n_orb))
        phi_tilde = rng.normal(size=(self.n_orb, self.n_rank))
        gb = rng.normal(size=(self.n_rank,))

        expected_wc = np.einsum("rsc,rs->c", self.x, dm1, optimize=True)
        expected_y = np.einsum("rqc,rc->qc", self.x, phi_tilde,
                               optimize=True)
        expected_j = -np.einsum("pqc,c->pq", self.x, gb, optimize=True)
        wc, y_all, j_x_sym = xtc_mod._tucker_x_normal_order_intermediates(
            self.u, self.z, dm1, phi_tilde, gb,
        )

        np.testing.assert_allclose(np.asarray(wc), expected_wc, atol=1e-11,
                                   rtol=1e-11)
        np.testing.assert_allclose(np.asarray(y_all), expected_y, atol=1e-11,
                                   rtol=1e-11)
        np.testing.assert_allclose(np.asarray(j_x_sym), expected_j, atol=1e-11,
                                   rtol=1e-11)


class TestStreamedTuckerBuild(unittest.TestCase):
    def test_full_rank_streamed_basis_and_core_recover_x(self):
        rng = np.random.default_rng(20)
        n_orb, n_rank = 6, 6
        x = rng.normal(size=(n_orb, n_orb, n_rank))
        x = 0.5 * (x + x.swapaxes(0, 1))
        fake = _FakeTuckerBuild(x)

        u = xtc_mod.ISDFXTC.select_tucker_x_orbital_basis(
            fake, None, n_orb, oversampling=3, seed=21, orb_block_size=2,
        )
        factors = xtc_mod.ISDFXTC.compute_tucker_x_core(fake, None, u)
        reconstructed = np.einsum(
            "ra,abc,sb->rsc", factors["U"], factors["Z"], factors["U"],
            optimize=True,
        )

        np.testing.assert_allclose(reconstructed, x, atol=1e-11, rtol=1e-11)
        self.assertGreater(len(fake.panel_shapes), 1)
        self.assertTrue(
            all(shape[0] <= 2 for shape in fake.panel_shapes[:-1]),
            f"sketch did not stream r panels: {fake.panel_shapes}",
        )

    def test_nonorthogonal_basis_is_rejected(self):
        x = np.zeros((4, 4, 3))
        fake = _FakeTuckerBuild(x)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            xtc_mod.ISDFXTC.compute_tucker_x_core(
                fake, None, np.ones((4, 2)),
            )


class TestTuckerXSolverViews(unittest.TestCase):
    """The production tile and normal-order paths accept X_tucker only."""

    def setUp(self):
        rng = np.random.default_rng(22)
        self.n_orb, self.n_rank = 6, 4
        phi = rng.normal(size=(self.n_orb, self.n_rank))
        dm1 = rng.normal(size=(self.n_orb, self.n_orb))
        self.runtime = _FakeTuckerRuntime(phi, dm1)
        self.d = rng.normal(size=(self.n_rank, self.n_rank))
        x = rng.normal(size=(self.n_orb, self.n_orb, self.n_rank))
        self.x = 0.5 * (x + x.swapaxes(0, 1))
        self.u, _ = np.linalg.qr(rng.normal(size=(self.n_orb, self.n_orb)))
        self.z = np.einsum("ra,rsc,sb->abc", self.u, self.x, self.u,
                           optimize=True)
        self.dense = {"D": self.d, "X": self.x}
        # Deliberately no dense-X key: an accidental fallback raises KeyError.
        self.factor = {"D": self.d, "X_tucker": {"U": self.u, "Z": self.z}}

    def test_direct_tile_factor_view_matches_dense_x(self):
        ranges = (slice(0, 3), slice(2, 5), slice(1, 4), slice(0, 2))
        with mock.patch.object(xtc_mod, "_get_device_free_bytes", return_value=2**40):
            expected = xtc_mod.ISDFXTC._get_delta_u_direct_tile(
                self.runtime, self.dense, ranges,
            )
            actual = xtc_mod.ISDFXTC._get_delta_u_direct_tile(
                self.runtime, self.factor, ranges,
            )
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                   rtol=1e-11, atol=1e-11)

    def test_padded_direct_tile_factor_view_matches_dense_x(self):
        ranges = (slice(0, 3), slice(2, 5), slice(1, 4), slice(0, 2))
        with mock.patch.object(xtc_mod, "_get_device_free_bytes", return_value=2**40):
            expected = xtc_mod.ISDFXTC._get_delta_u_direct_tile(
                self.runtime, self.dense, ranges, panel_size=4,
            )
            actual = xtc_mod.ISDFXTC._get_delta_u_direct_tile(
                self.runtime, self.factor, ranges, panel_size=4,
            )
        self.assertEqual(actual.shape, (4, 3, 4, 2))
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                   rtol=1e-11, atol=1e-11)

    def test_generic_delta_u_and_normal_order_factor_views_match_dense_x(self):
        ranges = (slice(0, 3), slice(1, 5), slice(1, 4), slice(0, 2))
        expected_u = xtc_mod.ISDFXTC._contract_delta_U_kernels(
            self.runtime, self.dense, ranges,
        )
        actual_u = xtc_mod.ISDFXTC._contract_delta_U_kernels(
            self.runtime, self.factor, ranges,
        )
        np.testing.assert_allclose(np.asarray(actual_u), np.asarray(expected_u),
                                   rtol=1e-11, atol=1e-11)

        self.runtime.isdf_kernels = self.dense
        expected_public_u = xtc_mod.ISDFXTC.get_delta_U(
            self.runtime, None, ranges=ranges,
        )
        self.runtime.isdf_kernels = self.factor
        actual_public_u = xtc_mod.ISDFXTC.get_delta_U(
            self.runtime, None, ranges=ranges,
        )
        np.testing.assert_allclose(np.asarray(actual_public_u),
                                   np.asarray(expected_public_u),
                                   rtol=1e-11, atol=1e-11)

        self.runtime.isdf_kernels = self.dense
        expected_h = xtc_mod.ISDFXTC.get_delta_h(
            self.runtime, None, ranges=(slice(0, 4), slice(1, 6)),
        )
        self.runtime.isdf_kernels = self.factor
        actual_h = xtc_mod.ISDFXTC.get_delta_h(
            self.runtime, None, ranges=(slice(0, 4), slice(1, 6)),
        )
        np.testing.assert_allclose(np.asarray(actual_h), np.asarray(expected_h),
                                   rtol=1e-11, atol=1e-11)


class TestRealFactorOnlyH2(unittest.TestCase):
    """Exercise the actual ISDF, ERI, and CCSD paths with no dense-X key."""

    def test_full_rank_factor_only_view_matches_dense_x(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", verbose=0,
        )
        mf = scf.RHF(mol).run()
        jparams = {"alpha": jnp.array([1.0])}
        base = xtc_mod.XTC.from_pyscf(mf, REXP(), grid_lvl=0)
        isdf = xtc_mod.ISDFXTC.from_xtc(
            base, n_rank=max(8, 3 * base.n_orb), is_incore=True,
        )
        isdf = isdf.isdf(
            jparams, batch_size=64, orb_block_size=2,
            host_grid_block_size=512,
        )
        l_aux = isdf._compute_L_aux(jparams, batch_size=64,
                                    host_grid_block_size=512)
        dense = isdf.isdf_kernels
        u = isdf.select_tucker_x_orbital_basis(
            jparams, base.n_orb, oversampling=2, seed=23, batch_size=64,
            L_aux=l_aux, orb_block_size=2, host_grid_block_size=512,
        )
        factors = isdf.compute_tucker_x_core(
            jparams, u, batch_size=64, L_aux=l_aux,
            host_grid_block_size=512,
        )
        factor_kernels = dict(dense)
        factor_kernels.pop("X")
        factor_kernels["X_tucker"] = factors
        dense_obj = isdf.replace(isdf_kernels=dense)
        factor_obj = isdf.replace(isdf_kernels=factor_kernels)
        self.assertNotIn("X", factor_obj.isdf_kernels)
        np.testing.assert_allclose(
            np.asarray(factor_obj.get_delta_h(jparams)),
            np.asarray(dense_obj.get_delta_h(jparams)), rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(factor_obj.get_delta_U(jparams)),
            np.asarray(dense_obj.get_delta_U(jparams)), rtol=1e-10, atol=1e-10,
        )

        dense_cc = jax_xtc_ccsd.RCCSD(
            mf, dense_obj, jparams, max_memory=2_000, gpu_max_memory=2_000,
            on_the_fly_vvvv=False,
        )
        factor_cc = jax_xtc_ccsd.RCCSD(
            mf, factor_obj, jparams, max_memory=2_000, gpu_max_memory=2_000,
            on_the_fly_vvvv=False,
        )
        dense_eris = dense_cc.ao2mo()
        factor_eris = factor_cc.ao2mo()
        try:
            np.testing.assert_allclose(np.asarray(factor_eris.fock),
                                       np.asarray(dense_eris.fock),
                                       rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(np.asarray(factor_eris.ovov),
                                       np.asarray(dense_eris.ovov),
                                       rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(np.asarray(factor_eris.vvvv),
                                       np.asarray(dense_eris.vvvv),
                                       rtol=1e-10, atol=1e-10)
            dense_cc.max_cycle = factor_cc.max_cycle = 50
            e_dense = float(dense_cc.kernel(eris=dense_eris)[0])
            e_factor = float(factor_cc.kernel(eris=factor_eris)[0])
            self.assertTrue(dense_cc.converged and factor_cc.converged)
            self.assertAlmostEqual(e_factor, e_dense, places=10)
        finally:
            dense_eris.close()
            factor_eris.close()


if __name__ == "__main__":
    unittest.main()
