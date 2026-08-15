"""Controls for the experimental geometry-hierarchical L_aux operator."""

import os
import tempfile
import unittest

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from pytc.df.hmatrix import (
    apply_pair_gradient_hmatrix,
    apply_pair_gradient_interpolative_hmatrix,
)


def _direct_aux(points, weights, xi_phi, gradient):
    """Reference contraction for a small, fully materialized test kernel."""
    value = gradient(points, points)
    weighted_xi = xi_phi * weights[None, :]
    l_aux = np.einsum("ah,ihk->aik", weighted_xi, value)
    h_aux = weighted_xi @ np.sum(value * value, axis=2).T
    return l_aux, h_aux


class TestPairGradientHMatrix(unittest.TestCase):
    """The hierarchy must retain each pair and factor H_aux independently."""

    def setUp(self):
        rng = np.random.default_rng(701)
        # Two spatially separated clouds force a nontrivial far field while
        # retaining enough nearby pairs to exercise both paths.
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
        # A smooth, vector-valued pair gradient.  The scalar H_aux kernel is
        # intentionally not represented as the square of any factor here.
        return -2.0 * displacement * np.exp(-radius2)

    def test_exact_control_matches_direct_for_laux_and_haux(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        l_aux, h_aux, metadata = apply_pair_gradient_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=0.0,
        )

        self.assertGreater(metadata["near_blocks"], 0)
        self.assertGreater(metadata["far_blocks"], 0)
        self.assertGreater(metadata["far_rank_max"], 0)
        np.testing.assert_allclose(l_aux, direct_l, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(h_aux, direct_h, rtol=0.0, atol=2e-12)

    def test_tolerance_reduces_rank_with_reported_error(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        exact_l, exact_h, exact = apply_pair_gradient_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=0.0,
        )
        approx_l, approx_h, approx = apply_pair_gradient_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=1e-3,
        )

        np.testing.assert_allclose(exact_l, direct_l, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(exact_h, direct_h, rtol=0.0, atol=2e-12)
        self.assertLessEqual(approx["far_rank_mean"], exact["far_rank_mean"])
        self.assertLessEqual(approx["far_factor_storage"], exact["far_factor_storage"])
        self.assertLess(
            np.linalg.norm(approx_l - direct_l) / np.linalg.norm(direct_l),
            2e-3,
        )
        self.assertLess(
            np.linalg.norm(approx_h - direct_h) / np.linalg.norm(direct_h),
            2e-3,
        )

    def test_interpolative_blocks_fallback_when_validation_rejects_them(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        l_aux, h_aux, metadata = apply_pair_gradient_interpolative_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=1e-13,
            max_rank=2,
            direct_fallback=True,
        )

        self.assertGreater(metadata["far_blocks"], 0)
        self.assertGreater(metadata["far_direct_fallbacks"], 0)
        np.testing.assert_allclose(l_aux, direct_l, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(h_aux, direct_h, rtol=0.0, atol=2e-12)

    def test_interpolative_blocks_have_controlled_smooth_kernel_error(self):
        direct_l, direct_h = _direct_aux(
            self.points, self.weights, self.xi_phi, self._gradient
        )
        l_aux, h_aux, metadata = apply_pair_gradient_interpolative_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self._gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=None,
            max_rank=8,
            direct_fallback=False,
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


class TestPhysicalH2ResidualHMatrix(unittest.TestCase):
    """A physical residual control before wiring the approximation to ISDFTC."""

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf

        from pytc.jastrow import BoysHandy
        from pytc.kmat import calc_kmat_kernels_from_aux
        from pytc.tc import ISDFTC, TC

        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        cls.boys_handy = BoysHandy.create(mol)
        cls.params = cls.boys_handy.init_params()
        base = TC.from_pyscf(mf, cls.boys_handy, grid_lvl=0)
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
        cls.calc_kmat_kernels_from_aux = staticmethod(calc_kmat_kernels_from_aux)

        @jax.jit
        def pair_gradient(rows, cols):
            return cls.boys_handy.grad_r_batch_laux(rows, cols, cls.params)

        # Compile a representative call before the blockwise hierarchy asks
        # for a variety of small row/column panels.
        pair_gradient(
            jnp.asarray(cls.points[:2]), jnp.asarray(cls.points[:2])
        ).block_until_ready()
        cls.pair_gradient = staticmethod(
            lambda rows, cols: np.asarray(
                pair_gradient(jnp.asarray(rows), jnp.asarray(cols))
            )
        )

    def test_exact_hierarchy_recovers_physical_boys_handy_residual(self):
        l_aux, h_aux, metadata = apply_pair_gradient_hmatrix(
            self.points,
            self.weights,
            self.xi_phi,
            self.pair_gradient,
            leaf_size=8,
            eta=0.05,
            tolerance=0.0,
        )
        # The control is algebraically exact.  A float32-only JAX runtime
        # limits the physical evaluator itself to roughly 1e-7; standard
        # float64 PyTC runs satisfy the tighter branch of this bound.
        atol = 2e-12 if jax.config.x64_enabled else 2e-6
        self.assertGreater(metadata["far_blocks"], 0)
        np.testing.assert_allclose(l_aux, self.direct_l, rtol=0.0, atol=atol)
        np.testing.assert_allclose(h_aux, self.direct_h, rtol=0.0, atol=atol)
        recovered = self.calc_kmat_kernels_from_aux(
            self.subset.xi_phi,
            self.subset.xi_grad,
            self.subset.weights,
            l_aux,
            h_aux,
        )
        np.testing.assert_allclose(
            recovered["K1_kernel"], self.direct_kernels["K1_kernel"], rtol=0.0, atol=atol
        )
        np.testing.assert_allclose(
            recovered["K3_kernel"], self.direct_kernels["K3_kernel"], rtol=0.0, atol=atol
        )
        direct_2b = self.subset.replace(
            isdf_kernels={**self.direct_kernels, "L_aux": self.direct_l}
        ).get_2b(self.params)
        recovered_2b = self.subset.replace(
            isdf_kernels={**recovered, "L_aux": l_aux}
        ).get_2b(self.params)
        np.testing.assert_allclose(recovered_2b, direct_2b, rtol=0.0, atol=atol)

    def test_isdf_hmatrix_mode_reaches_kernels_and_two_body(self):
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
            use_laux_hmatrix=True,
            laux_hmatrix_leaf_size=8,
            laux_hmatrix_eta=0.05,
            # Zero makes every nonzero held-out residual take the direct
            # fallback, giving an exact end-to-end cache/recovery control.
            laux_hmatrix_tolerance=0.0,
            laux_hmatrix_max_rank=16,
            laux_hmatrix_heldout_size=16,
        )
        atol = 2e-12 if jax.config.x64_enabled else 2e-6
        for key in ("L_aux", "K1_kernel", "K3_kernel"):
            np.testing.assert_allclose(
                hierarchy.isdf_kernels[key], direct.isdf_kernels[key], rtol=0.0, atol=atol
            )
        np.testing.assert_allclose(
            hierarchy.get_2b(self.params), direct.get_2b(self.params), rtol=0.0, atol=atol
        )

    def test_isdf_hmatrix_requires_haux_recovery(self):
        with self.assertRaisesRegex(ValueError, "reuse_aux_kernels=True"):
            self.subset.isdf(
                self.params,
                batch_size=16,
                host_grid_block_size=64,
                use_laux_hmatrix=True,
            )

    def test_out_of_core_hmatrix_recovers_and_tags_its_cache(self):
        """The hierarchy must retain streamed K recovery and cache provenance."""
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
                use_laux_hmatrix=True,
                laux_hmatrix_leaf_size=8,
                laux_hmatrix_eta=0.05,
                laux_hmatrix_tolerance=0.0,
                laux_hmatrix_max_rank=16,
                laux_hmatrix_heldout_size=16,
            )

            atol = 2e-12 if jax.config.x64_enabled else 2e-6
            for key in ("L_aux", "K1_kernel", "K3_kernel"):
                np.testing.assert_allclose(
                    np.asarray(streamed.isdf_kernels[key]),
                    np.asarray(
                        self.direct_l if key == "L_aux" else self.direct_kernels[key]
                    ),
                    rtol=0.0,
                    atol=atol,
                )
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["pytc_kmat_kernel_mode"], "aux-recovery")
                self.assertEqual(
                    handle.attrs["pytc_laux_gradient_mode"],
                    "hmatrix[leaf=8,eta=0.05,tol=0,rank=16,heldout=16]-fast",
                )
                self.assertEqual(handle.attrs["pytc_laux_hmatrix_mode"], "interpolative-cur-v1")
                self.assertGreater(handle.attrs["pytc_laux_hmatrix_far_blocks"], 0)
                self.assertEqual(
                    handle.attrs["pytc_laux_hmatrix_far_fallbacks"],
                    handle.attrs["pytc_laux_hmatrix_far_blocks"],
                )
                self.assertNotIn("H_aux", handle)


class TestH2XTCWithHMatrix(unittest.TestCase):
    """The experimental path must reach full-X Delta-U unchanged in control mode."""

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf

        from pytc.jastrow import REXP
        from pytc.xtc import ISDFXTC, XTC

        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        cls.mf = mf
        cls.params = {"alpha": jnp.array([0.4])}
        cls.base = ISDFXTC.from_xtc(
            XTC.from_pyscf(mf, REXP(), grid_lvl=0), n_rank=8, is_incore=True
        )

    def test_full_x_control_follows_hmatrix_laux_path(self):
        x_variables = (
            "PYTC_XTC_DROP_X",
            "PYTC_XTC_DROP_X_NORMAL_ORDER",
            "PYTC_XTC_DROP_X_RESIDUAL",
        )
        previous = {name: os.environ.pop(name, None) for name in x_variables}
        try:
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
                use_laux_hmatrix=True,
                laux_hmatrix_leaf_size=128,
                laux_hmatrix_eta=0.05,
                laux_hmatrix_tolerance=0.0,
                laux_hmatrix_max_rank=16,
                laux_hmatrix_heldout_size=16,
            )
        finally:
            for name, value in previous.items():
                if value is not None:
                    os.environ[name] = value

        atol = 2e-12 if jax.config.x64_enabled else 2e-6
        self.assertGreater(np.linalg.norm(np.asarray(hierarchy.isdf_kernels["X"])), 0.0)
        for key in ("K1_kernel", "K3_kernel", "D", "X"):
            np.testing.assert_allclose(
                hierarchy.isdf_kernels[key], direct.isdf_kernels[key], rtol=0.0, atol=atol
            )
        np.testing.assert_allclose(
            hierarchy.get_delta_h(self.params), direct.get_delta_h(self.params), rtol=0.0, atol=atol
        )
        np.testing.assert_allclose(
            hierarchy.get_delta_U(self.params), direct.get_delta_U(self.params), rtol=0.0, atol=atol
        )

    def test_full_x_approximate_hierarchy_keeps_relaxed_ccsd_energy(self):
        """A nonzero hierarchy tolerance needs a relaxed-energy guard."""
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
            use_laux_hmatrix=True,
            laux_hmatrix_leaf_size=128,
            laux_hmatrix_eta=0.05,
            laux_hmatrix_tolerance=1e-2,
            laux_hmatrix_max_rank=16,
            laux_hmatrix_heldout_size=16,
        )

        def solve(isdf):
            cc = jax_xtc_ccsd.RCCSD(
                self.mf, isdf, self.params, max_memory=2_000,
                gpu_max_memory=2_000, on_the_fly_vvvv=False,
            )
            eris = cc.ao2mo()
            try:
                cc.max_cycle = 50
                energy = float(cc.kernel(eris=eris)[0])
                self.assertTrue(cc.converged)
                return energy
            finally:
                eris.close()

        e_direct = solve(direct)
        e_hmatrix = solve(hierarchy)
        # This is intentionally looser than the local observed 0.0002 mHa
        # shift, while still well below the 1 mHa scale-run gate.
        self.assertLess(abs(e_hmatrix - e_direct) * 1_000.0, 0.01)
