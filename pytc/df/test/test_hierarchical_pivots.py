"""Controls for the small-grid hierarchical ISDF-pivot diagnostic."""

import unittest

import numpy as np

from pytc.df.hierarchical_pivots import (
    gradient_orbital_product_kernel_block,
    global_pivoted_cholesky,
    hierarchical_pivoted_cholesky,
    local_pivot_candidates,
    orbital_product_feature_sketch,
    orbital_product_kernel_block,
    orbital_product_projection_error,
    relative_block_rank_profile,
    relative_gradient_block_rank_profile,
)


class TestHierarchicalPivotPrototype(unittest.TestCase):
    """A leaf screen is useful only with a global residual control."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(701)
        centers = np.array(
            [[-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        cls.points = np.concatenate(
            [rng.normal(center, 0.12, size=(24, 3)) for center in centers]
        )
        # A nonsingular small orbital-feature test matrix.  It deliberately
        # does not assume spatial locality, so the refinement assertion is
        # stronger than an easy locality-only example.
        cls.features = rng.normal(size=(6, len(cls.points)))

    def test_leaf_candidates_are_bounded_by_leaf_rank(self):
        candidates, metadata = local_pivot_candidates(
            self.features, self.points, leaf_size=12, local_rank=2
        )
        self.assertEqual(metadata["n_leaf"], len(candidates))
        self.assertTrue(all(1 <= len(candidate) <= 2 for candidate in candidates))
        self.assertEqual(metadata["candidate_count"], sum(len(candidate) for candidate in candidates))
        self.assertLess(metadata["candidate_count"], len(self.points))

    def test_leaf_maximum_refinement_recovers_global_pivots(self):
        rank = 12
        global_result = global_pivoted_cholesky(self.features, rank)
        refined = hierarchical_pivoted_cholesky(
            self.features,
            self.points,
            rank,
            leaf_size=12,
            local_rank=2,
            refine_with_leaf_maxima=True,
        )
        np.testing.assert_array_equal(refined["pivots"], global_result["pivots"])
        np.testing.assert_allclose(
            refined["residual_diagonal"], global_result["residual_diagonal"], atol=1e-13
        )
        self.assertGreater(refined["metadata"]["refinement_additions"], 0)

    def test_screened_selection_exposes_its_quality_cost_tradeoff(self):
        rank = 12
        reference = global_pivoted_cholesky(self.features, rank)
        screened = hierarchical_pivoted_cholesky(
            self.features,
            self.points,
            rank,
            leaf_size=12,
            local_rank=2,
            refine_with_leaf_maxima=False,
        )
        reference_error = orbital_product_projection_error(self.features, reference["pivots"])
        screened_error = orbital_product_projection_error(self.features, screened["pivots"])
        self.assertLess(screened["metadata"]["argmax_fraction_initial"], 1.0)
        self.assertGreaterEqual(screened_error + 1e-13, reference_error)

    def test_orbital_product_block_ranks_are_exactly_measured(self):
        rows = np.arange(8)
        cols = np.arange(24, 40)
        block = orbital_product_kernel_block(self.features, rows, cols)
        expected = (self.features[:, rows].T @ self.features[:, cols]) ** 2
        np.testing.assert_allclose(block, expected, atol=0.0)

        profile = relative_block_rank_profile(
            self.features, rows, cols, (1e-4, 1e-6, 1e-8)
        )
        ranks = profile["ranks"]
        self.assertLessEqual(ranks["1e-04"], ranks["1e-06"])
        self.assertLessEqual(ranks["1e-06"], ranks["1e-08"])
        self.assertLessEqual(profile["relative_errors"]["1e-04"], 1e-4)
        self.assertLessEqual(profile["relative_errors"]["1e-06"], 1e-6)
        self.assertLessEqual(profile["relative_errors"]["1e-08"], 1e-8)

    def test_gradient_product_block_ranks_are_exactly_measured(self):
        rng = np.random.default_rng(17)
        gradients = rng.normal(size=(*self.features.shape, 3))
        rows = np.arange(8)
        cols = np.arange(24, 40)
        block = gradient_orbital_product_kernel_block(
            self.features, gradients, rows, cols
        )
        orbital = self.features[:, rows].T @ self.features[:, cols]
        gradient = sum(
            gradients[:, rows, component].T @ gradients[:, cols, component]
            for component in range(3)
        )
        np.testing.assert_allclose(block, orbital * gradient, atol=0.0)

        profile = relative_gradient_block_rank_profile(
            self.features, gradients, rows, cols, (1e-4, 1e-6, 1e-8)
        )
        ranks = profile["ranks"]
        self.assertLessEqual(ranks["1e-04"], ranks["1e-06"])
        self.assertLessEqual(ranks["1e-06"], ranks["1e-08"])

    def test_orbital_product_feature_sketch_is_reproducible(self):
        first = orbital_product_feature_sketch(self.features, dimension=5, seed=7)
        second = orbital_product_feature_sketch(self.features, dimension=5, seed=7)
        self.assertEqual(first.shape, (len(self.points), 5))
        np.testing.assert_allclose(first, second, atol=0.0)
        self.assertTrue(np.all(np.isfinite(first)))


if __name__ == "__main__":
    unittest.main()
