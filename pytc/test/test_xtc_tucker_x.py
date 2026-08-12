"""Algebra and streaming-build controls for orbital-leg Tucker X."""

import unittest

import jax
import numpy as np

from pytc import xtc as xtc_mod


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


if __name__ == "__main__":
    unittest.main()
