"""Controls for the experimental geometry-blocked ISDF kernel selector."""

import unittest

import numpy as np

from pytc.df.hierarchical_pivots import global_pivoted_cholesky
from pytc.df.kernel_block_pivots import (
    build_kernel_block_matrix,
    kernel_block_pivoted_cholesky,
    kernel_block_pivoted_cholesky_jax,
)


class TestKernelBlockPivots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(701)
        x = np.linspace(-5.0, 5.0, 96)
        cls.points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
        centers = np.linspace(-4.0, 4.0, 7)
        cls.features = np.asarray(
            [np.exp(-0.7 * (x - center) ** 2) for center in centers]
        )
        cls.features += 1e-5 * rng.normal(size=cls.features.shape)
        cls.gradients = np.stack(
            (
                np.gradient(cls.features, x, axis=1),
                np.zeros_like(cls.features),
                np.zeros_like(cls.features),
            ),
            axis=2,
        )

    def test_exact_fallback_columns_and_pivots_match_dense_reference(self):
        matrix = build_kernel_block_matrix(
            self.features,
            self.points,
            leaf_size=12,
            eta=0.5,
            tolerance=0.0,
            max_rank=12,
            direct_fallback=True,
        )
        dense = (self.features.T @ self.features) ** 2
        for pivot in (0, 17, 48, 95):
            np.testing.assert_allclose(matrix.column(pivot), dense[:, pivot], atol=2e-13)

        reference = global_pivoted_cholesky(self.features, 20)
        blocked = kernel_block_pivoted_cholesky(
            matrix,
            20,
            shift_scale=0.0,
            tie_break_scale=0.0,
            stopping_tolerance=0.0,
        )
        np.testing.assert_array_equal(blocked["pivots"], reference["pivots"])
        np.testing.assert_allclose(
            blocked["residual_diagonal"],
            reference["residual_diagonal"],
            atol=2e-12,
        )

    def test_validated_far_blocks_reduce_storage_with_bounded_column_error(self):
        matrix = build_kernel_block_matrix(
            self.features,
            self.points,
            leaf_size=12,
            eta=0.5,
            tolerance=1e-6,
            max_rank=8,
            heldout_size=8,
            direct_fallback=True,
        )
        self.assertGreater(matrix.metadata["compressed_far_blocks"], 0)
        self.assertLess(
            matrix.metadata["auxiliary_storage_ratio_to_packed_dense"], 1.0
        )
        dense = (self.features.T @ self.features) ** 2
        errors = []
        for pivot in range(len(self.points)):
            errors.append(
                np.linalg.norm(matrix.column(pivot) - dense[:, pivot])
                / max(np.linalg.norm(dense[:, pivot]), np.finfo(float).tiny)
            )
        self.assertLess(max(errors), 2e-5)

    def test_gradient_exact_fallback_matches_dense_columns(self):
        matrix = build_kernel_block_matrix(
            self.features,
            self.points,
            gradient_features=self.gradients,
            leaf_size=12,
            eta=0.5,
            tolerance=0.0,
            max_rank=12,
            direct_fallback=True,
        )
        orbital = self.features.T @ self.features
        gradient = sum(
            self.gradients[:, :, component].T
            @ self.gradients[:, :, component]
            for component in range(3)
        )
        dense = orbital * gradient
        for pivot in (3, 31, 62, 90):
            np.testing.assert_allclose(matrix.column(pivot), dense[:, pivot], atol=2e-13)

    def test_jax_dense_update_matches_host_selector(self):
        matrix = build_kernel_block_matrix(
            self.features,
            self.points,
            leaf_size=12,
            eta=0.5,
            tolerance=0.0,
            max_rank=12,
            direct_fallback=True,
        )
        host = kernel_block_pivoted_cholesky(
            matrix,
            12,
            shift_scale=0.0,
            tie_break_scale=0.0,
            stopping_tolerance=0.0,
        )
        accelerated = kernel_block_pivoted_cholesky_jax(
            matrix,
            12,
            shift_scale=0.0,
            tie_break_scale=0.0,
            stopping_tolerance=0.0,
        )
        np.testing.assert_array_equal(accelerated["pivots"], host["pivots"])
        np.testing.assert_allclose(
            accelerated["residual_diagonal"],
            host["residual_diagonal"],
            atol=2e-12,
        )


if __name__ == "__main__":
    unittest.main()
