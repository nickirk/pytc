"""Numerical and physical controls for the opt-in L_aux hierarchy."""

import tempfile
import unittest

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from pytc.df import LauxHMatrixConfig
from pytc.df.hmatrix import _cross_factor, apply_pair_gradient_interpolative_hmatrix


def _direct_aux(points, weights, xi_phi, gradient):
    value = gradient(points, points)
    weighted_xi = xi_phi * weights[None, :]
    return (
        np.einsum("ah,ihk->aik", weighted_xi, value),
        weighted_xi @ np.sum(value * value, axis=2).T,
    )


class TestLauxHMatrixConfig(unittest.TestCase):
    def test_controls_are_explicit_and_validated(self):
        config = LauxHMatrixConfig(8, 0.05, 1e-3, 4, 6)
        self.assertEqual(
            config.cache_tag,
            "hmatrix[leaf=8,eta=0.05,tol=0.001,rank=4,heldout=6,fallback=1]",
        )
        invalid = (
            (1, 0.05, 1e-3, 4, 6),
            (8, 0.0, 1e-3, 4, 6),
            (8, 0.05, -1.0, 4, 6),
            (8, 0.05, 1e-3, 0, 6),
            (8, 0.05, 1e-3, 4, 0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                LauxHMatrixConfig(*values)

        invalid_types = (
            (8.0, 0.05, 1e-3, 4, 6, True),
            (8, np.inf, 1e-3, 4, 6, True),
            (8, 0.05, np.nan, 4, 6, True),
            (8, 0.05, 1e-3, True, 6, True),
            (8, 0.05, 1e-3, 4, 6, 1),
        )
        for values in invalid_types:
            with self.subTest(values=values), self.assertRaises(TypeError):
                LauxHMatrixConfig(*values)


class TestPairGradientHMatrix(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(701)
        self.points = np.concatenate(
            (
                rng.normal(loc=(-2.0, 0.0, 0.0), scale=0.20, size=(24, 3)),
                rng.normal(loc=(2.0, 0.0, 0.0), scale=0.20, size=(24, 3)),
            )
        )
        self.weights = rng.uniform(0.2, 1.1, size=len(self.points))
        self.xi_phi = rng.normal(size=(7, len(self.points)))

    @staticmethod
    def _gradient(rows, cols):
        displacement = rows[:, None, :] - cols[None, :, :]
        radius2 = np.sum(displacement * displacement, axis=2, keepdims=True)
        return -2.0 * displacement * np.exp(-radius2)

    def test_validation_fallback_matches_direct(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        config = LauxHMatrixConfig(8, 0.05, 1e-13, 2, 8)
        l_aux, h_aux, metadata = apply_pair_gradient_interpolative_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            config=config,
        )
        self.assertGreater(metadata["far_blocks"], 0)
        self.assertGreater(metadata["far_direct_fallbacks"], 0)
        np.testing.assert_allclose(l_aux, direct_l, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(h_aux, direct_h, rtol=0.0, atol=2e-12)

    def test_rank_capped_cur_has_small_smooth_kernel_error(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        config = LauxHMatrixConfig(
            8, 0.05, 0.0, 8, 8, direct_fallback=False
        )
        l_aux, h_aux, metadata = apply_pair_gradient_interpolative_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            config=config,
        )
        self.assertGreater(metadata["far_rank_max"], 0)
        self.assertLess(
            np.linalg.norm(l_aux - direct_l) / np.linalg.norm(direct_l),
            2e-5,
        )
        self.assertLess(
            np.linalg.norm(h_aux - direct_h) / np.linalg.norm(direct_h),
            2e-5,
        )

    def test_zero_candidate_cross_cannot_hide_nonzero_validation_patch(self):
        factor = _cross_factor(
            np.zeros((4, 2)),
            np.zeros((2, 4)),
            np.array([0, 3]),
            np.array([0, 3]),
            2,
            np.array([1, 2]),
            np.array([1, 2]),
            np.ones((2, 2)),
            1e-3,
        )
        self.assertEqual(factor[0].shape, (4, 0))
        self.assertEqual(factor[2], 1.0)

    def test_nonfinite_kernel_is_rejected(self):
        def invalid_gradient(rows, cols):
            return np.full((len(rows), len(cols), 3), np.nan)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            apply_pair_gradient_interpolative_hmatrix(
                self.points,
                self.weights,
                self.xi_phi,
                invalid_gradient,
                config=LauxHMatrixConfig(8, 0.05, 1e-3, 4, 4),
            )


class TestPhysicalH2ResidualHMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf

        from pytc.jastrow import BoysHandy
        from pytc.kmat import calc_kmat_kernels_from_aux
        from pytc.tc import ISDFTC, TC

        molecule = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mean_field = scf.RHF(molecule).run()
        cls.boys_handy = BoysHandy.create(molecule)
        cls.params = cls.boys_handy.init_params()
        base = TC.from_pyscf(mean_field, cls.boys_handy, grid_lvl=0)
        full = ISDFTC.from_tc(
            base, n_rank=max(8, 3 * base.n_orb), is_incore=True
        )
        selection = np.linspace(0, len(full.grid_points) - 1, 64, dtype=int)
        cls.points = np.asarray(full.grid_points)[selection]
        cls.weights = np.asarray(full.weights)[selection]
        cls.xi_phi = np.asarray(full.xi_phi)[:, selection]
        cls.subset = full.replace(
            grid_points=jnp.asarray(cls.points),
            weights=jnp.asarray(cls.weights),
            xi_phi=jnp.asarray(cls.xi_phi),
            xi_grad=jnp.asarray(full.xi_grad)[:, selection, :],
        )
        cls.direct_l, cls.direct_h = map(
            np.asarray,
            cls.subset._compute_L_aux(
                cls.params,
                batch_size=16,
                host_grid_block_size=64,
                include_h_aux=True,
                use_laux_fast_grad=True,
            ),
        )
        cls.direct_kernels = calc_kmat_kernels_from_aux(
            cls.subset.xi_phi,
            cls.subset.xi_grad,
            cls.subset.weights,
            cls.direct_l,
            cls.direct_h,
        )

    @property
    def exact_config(self):
        return LauxHMatrixConfig(8, 0.05, 0.0, 16, 16)

    def test_exact_control_reaches_kernels_and_two_body(self):
        direct = self.subset.isdf(
            self.params,
            batch_size=16,
            host_grid_block_size=64,
            reuse_aux_kernels=True,
            use_laux_fast_grad=True,
        )
        hierarchy = self.subset.isdf(
            self.params,
            batch_size=16,
            host_grid_block_size=64,
            reuse_aux_kernels=True,
            use_laux_fast_grad=True,
            laux_hmatrix=self.exact_config,
        )
        atol = 2e-12 if jax.config.x64_enabled else 2e-6
        for key in ("L_aux", "K1_kernel", "K3_kernel"):
            np.testing.assert_allclose(
                hierarchy.isdf_kernels[key],
                direct.isdf_kernels[key],
                rtol=0.0,
                atol=atol,
            )
        np.testing.assert_allclose(
            hierarchy.get_2b(self.params),
            direct.get_2b(self.params),
            rtol=0.0,
            atol=atol,
        )

    def test_hmatrix_requires_auxiliary_kernel_recovery(self):
        with self.assertRaisesRegex(ValueError, "reuse_aux_kernels=True"):
            self.subset.isdf(
                self.params,
                batch_size=16,
                host_grid_block_size=64,
                laux_hmatrix=self.exact_config,
            )

    def test_out_of_core_cache_is_recoverable_and_fully_tagged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/hmatrix.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("xi_phi", data=np.asarray(self.subset.xi_phi))
                handle.create_dataset("xi_grad", data=np.asarray(self.subset.xi_grad))

            streamed = self.subset.replace(
                is_incore=False,
                xi_phi=None,
                xi_grad=None,
                save_path=path,
            ).isdf(
                self.params,
                save_path=path,
                batch_size=16,
                host_grid_block_size=64,
                reuse_aux_kernels=True,
                use_laux_fast_grad=True,
                laux_hmatrix=self.exact_config,
            )
            observed = {
                key: np.asarray(streamed.isdf_kernels[key])
                for key in ("L_aux", "K1_kernel", "K3_kernel")
            }
            streamed.isdf_kernels["L_aux"].file.close()

            atol = 2e-12 if jax.config.x64_enabled else 2e-6
            references = {
                "L_aux": self.direct_l,
                "K1_kernel": self.direct_kernels["K1_kernel"],
                "K3_kernel": self.direct_kernels["K3_kernel"],
            }
            for key in observed:
                np.testing.assert_allclose(
                    observed[key], references[key], rtol=0.0, atol=atol
                )
            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    handle.attrs["pytc_kmat_kernel_mode"], "aux-recovery"
                )
                self.assertEqual(
                    handle.attrs["pytc_laux_gradient_mode"],
                    self.exact_config.cache_tag + "-fast",
                )
                self.assertEqual(
                    handle.attrs["pytc_laux_hmatrix_mode"],
                    "interpolative-cur-v1",
                )
                self.assertGreater(
                    handle.attrs["pytc_laux_hmatrix_far_blocks"], 0
                )
                self.assertEqual(
                    handle.attrs["pytc_laux_hmatrix_far_fallbacks"],
                    handle.attrs["pytc_laux_hmatrix_far_blocks"],
                )
                self.assertNotIn("H_aux", handle)


class TestH2XTCWithHMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf

        from pytc.jastrow import REXP
        from pytc.xtc import ISDFXTC, XTC

        molecule = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        cls.mean_field = scf.RHF(molecule).run()
        cls.params = {"alpha": jnp.array([0.4])}
        cls.base = ISDFXTC.from_xtc(
            XTC.from_pyscf(cls.mean_field, REXP(), grid_lvl=0),
            n_rank=8,
            is_incore=True,
        )

    def test_approximate_full_x_keeps_relaxed_ccsd_energy(self):
        from pytc.solver import jax_xtc_ccsd

        direct = self.base.isdf(
            self.params,
            batch_size=32,
            orb_block_size=2,
            host_grid_block_size=512,
            reuse_aux_kernels=True,
        )
        hierarchy = self.base.isdf(
            self.params,
            batch_size=32,
            orb_block_size=2,
            host_grid_block_size=512,
            reuse_aux_kernels=True,
            laux_hmatrix=LauxHMatrixConfig(128, 0.05, 1e-2, 16, 16),
        )

        def solve(isdf_object):
            coupled_cluster = jax_xtc_ccsd.RCCSD(
                self.mean_field,
                isdf_object,
                self.params,
                max_memory=2_000,
                gpu_max_memory=2_000,
                on_the_fly_vvvv=False,
            )
            eris = coupled_cluster.ao2mo()
            try:
                coupled_cluster.max_cycle = 50
                energy = float(coupled_cluster.kernel(eris=eris)[0])
                self.assertTrue(coupled_cluster.converged)
                return energy
            finally:
                eris.close()

        direct_energy = solve(direct)
        hierarchy_energy = solve(hierarchy)
        self.assertLess(abs(hierarchy_energy - direct_energy) * 1_000.0, 0.01)


if __name__ == "__main__":
    unittest.main()
