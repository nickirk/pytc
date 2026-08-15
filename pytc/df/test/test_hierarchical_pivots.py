"""Controls for the small-grid hierarchical ISDF-pivot diagnostic."""

import unittest

import numpy as np

from pytc.df.hierarchical_pivots import (
    global_pivoted_cholesky,
    hierarchical_pivoted_cholesky,
    local_pivot_candidates,
    orbital_product_projection_error,
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


if __name__ == "__main__":
    unittest.main()
