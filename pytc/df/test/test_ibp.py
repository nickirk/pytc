"""Direct tests for pytc.df.ibp's canonical single-IBP primitives (task #2,
#proj-isdf-ibp-coulomb) and typed provenance artifacts (task #5).
Exercises the primitives at their new home independently of the atom-
centered benchmark's re-export, so a future change to the benchmark's
import path cannot silently stop testing the canonical implementation.

Molecular end-to-end regression coverage (H2O atom-centered grid vs exact
4-center/analytic-DF references) stays in
pytc/test/test_atom_centered_single_ibp.py, which already re-runs
unchanged against pytc.df.ibp via the benchmark's import.
"""

import dataclasses
import hashlib
import os
import subprocess
import sys
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pytc.df.ibp import (
    IBPGrid,
    IBPOperatorPlan,
    build_ibp_grid,
    build_ibp_operator_plan,
    naive_coulomb_kernel,
    kernel,
)


class TestAtomCenteredSingleIBPCore(unittest.TestCase):
    def _case(self, dtype):
        rng = np.random.default_rng(21)
        coords = rng.normal(size=(9, 3))
        weights = rng.normal(size=9)
        density = rng.normal(size=(4, 9))
        gradient = rng.normal(size=(3, 3, 9))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            density = density + 1j * rng.normal(size=density.shape)
            gradient = gradient + 1j * rng.normal(size=gradient.shape)
        density = density.astype(dtype)
        gradient = gradient.astype(dtype)
        return coords, weights, density, gradient

    def test_single_ibp_matches_explicit_dense_real_and_complex(self):
        for dtype in (np.float64, np.complex128):
            coords, weights, density, gradient = self._case(dtype)
            diff = coords[:, None, :] - coords[None, :, :]
            radius = np.linalg.norm(diff, axis=-1)
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            vector = np.einsum("nj,j,ijc->nci", density, weights, rhat)
            expected = -0.5 * np.einsum(
                "mci,nci,i->mn", gradient.conj(), vector, weights
            )
            actual, coincident = kernel(
                gradient, density, coords, weights,
                eval_block_size=4, source_block_size=3,
            )
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
            self.assertEqual(coincident, len(weights))

    def test_direct_offdiagonal_matches_explicit_dense(self):
        coords, weights, density, _ = self._case(np.complex128)
        diff = coords[:, None, :] - coords[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        inv_r = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius != 0.0)
        expected = np.einsum(
            "mi,i,ij,nj,j->mn", density.conj(), weights, inv_r, density, weights
        )
        actual, coincident = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=4, source_block_size=3,
        )
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        self.assertEqual(coincident, len(weights))

    def test_cross_grid_matches_dense_and_has_no_coincident_points(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(7, 3))
        y = rng.normal(size=(8, 3)) + 0.123
        wx, wy = rng.normal(size=7), rng.normal(size=8)
        grad = rng.normal(size=(2, 3, 7))
        rho = rng.normal(size=(3, 8))
        diff = x[:, None, :] - y[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        rhat = diff / radius[..., None]
        vector = np.einsum("nj,j,ijc->nci", rho, wy, rhat)
        expected = -0.5 * np.einsum("mci,nci,i->mn", grad, vector, wx)
        actual, coincident = kernel(
            grad, rho, x, wx, coords_right=y, weights_right=wy,
            eval_block_size=3, source_block_size=5,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(coincident, 0)

    def test_block_sizes_do_not_change_result(self):
        coords, weights, density, gradient = self._case(np.float64)
        small, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_block_sizes_do_not_change_result_offdiagonal(self):
        coords, weights, density, _ = self._case(np.complex128)
        small, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_rejects_malformed_inputs(self):
        coords, weights, density, gradient = self._case(np.float64)
        with self.assertRaises(ValueError):
            kernel(gradient[:, :2], density, coords, weights)
        with self.assertRaises(ValueError):
            kernel(
                gradient.astype(np.float32), density, coords, weights
            )
        with self.assertRaises(ValueError):
            naive_coulomb_kernel(
                density, density, coords, weights, eval_block_size=0
            )
        with self.assertRaises(ValueError):
            # coords shape mismatch: wrong number of grid points
            kernel(
                gradient, density, coords[:-1], weights
            )
        with self.assertRaises(ValueError):
            # non-finite coordinates
            bad_coords = coords.copy()
            bad_coords[0, 0] = np.nan
            kernel(gradient, density, bad_coords, weights)
        with self.assertRaises(ValueError):
            # dtype mismatch between gradient and density_right
            kernel(
                gradient.astype(np.complex128), density, coords, weights
            )

    def test_explicit_coincident_point_rhat_zero_convention(self):
        """A single point coincident with itself must contribute rhat=0,
        not NaN/inf -- and coincident_pairs must report it."""
        coords = np.zeros((1, 3))
        weights = np.ones(1)
        density = np.ones((1, 1))
        gradient = np.ones((1, 3, 1))
        result, coincident = kernel(
            gradient, density, coords, weights,
        )
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, np.zeros((1, 1)))
        self.assertEqual(coincident, 1)


class TestIBPGrid(unittest.TestCase):
    def _random_grid_arrays(self, n=6, seed=1):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 3)), rng.normal(size=n)

    def test_numpy_grid_round_trip(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        self.assertEqual(grid.n_grid, 6)
        self.assertEqual(grid.backend, "numpy")
        self.assertEqual(grid.device, "cpu")
        self.assertIsNone(grid.jax_version)
        self.assertEqual(grid.dtype, "float64")
        self.assertEqual(grid.coincident_point_policy, "centered_zero")
        self.assertEqual(grid.schema_version, "1")
        np.testing.assert_allclose(grid.coords, coords)
        np.testing.assert_allclose(grid.weights, weights)

    def test_numpy_grid_defensive_copy_not_caller_array(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        self.assertIsNot(grid.coords, coords)
        self.assertIsNot(grid.weights, weights)
        self.assertFalse(grid.coords.flags.writeable)
        self.assertFalse(grid.weights.flags.writeable)
        coords[0, 0] = 999.0
        weights[0] = 999.0
        self.assertNotEqual(grid.coords[0, 0], 999.0)
        self.assertNotEqual(grid.weights[0], 999.0)

    def test_jax_grid_round_trip(self):
        coords_np, weights_np = self._random_grid_arrays()
        coords = jnp.asarray(coords_np)
        weights = jnp.asarray(weights_np)
        coords_id = hashlib.sha256(b"coords").hexdigest()
        weights_id = hashlib.sha256(b"weights").hexdigest()
        grid = build_ibp_grid(
            coords, weights, backend="jax",
            coords_identity=coords_id, weights_identity=weights_id,
        )
        self.assertEqual(grid.backend, "jax")
        self.assertEqual(grid.jax_version, jax.__version__)
        self.assertEqual(grid.device, str(coords.device))
        self.assertEqual(grid.coords_sha256, coords_id)
        self.assertEqual(grid.weights_sha256, weights_id)
        self.assertIs(grid.coords, coords)
        self.assertIs(grid.weights, weights)

    def test_jax_grid_requires_identity(self):
        coords = jnp.asarray(self._random_grid_arrays()[0])
        weights = jnp.asarray(self._random_grid_arrays()[1])
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights, backend="jax")

    def test_jax_grid_rejects_malformed_identity_syntax(self):
        coords = jnp.asarray(self._random_grid_arrays()[0])
        weights = jnp.asarray(self._random_grid_arrays()[1])
        good = hashlib.sha256(b"x").hexdigest()
        with self.assertRaises(ValueError):
            build_ibp_grid(
                coords, weights, backend="jax",
                coords_identity="not-a-hash", weights_identity=good,
            )
        with self.assertRaises(ValueError):
            build_ibp_grid(
                coords, weights, backend="jax",
                coords_identity=good.upper(), weights_identity=good,
            )

    def test_numpy_grid_rejects_caller_supplied_identity(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(
                coords, weights, coords_identity=hashlib.sha256(b"x").hexdigest()
            )

    def test_rejects_malformed_shapes(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords[:, :2], weights)
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights[:-1])
        with self.assertRaises(ValueError):
            build_ibp_grid(np.empty((0, 3)), np.empty(0))

    def test_rejects_non_finite(self):
        coords, weights = self._random_grid_arrays()
        bad_coords = coords.copy()
        bad_coords[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_ibp_grid(bad_coords, weights)
        bad_weights = weights.copy()
        bad_weights[0] = np.inf
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, bad_weights)

    def test_rejects_dtype_mismatch(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords.astype(np.float32), weights.astype(np.float64))

    def test_rejects_complex_dtype(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords.astype(np.complex128), weights.astype(np.complex128))

    def test_rejects_unsupported_backend(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights, backend="numba")

    def test_rejects_unsupported_coincident_point_policy(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights, coincident_point_policy="skip")

    def test_construction_metadata_is_deep_frozen(self):
        coords, weights = self._random_grid_arrays()
        meta = {"mol": "H2O", "nested": {"grid_level": 2}}
        grid = build_ibp_grid(coords, weights, construction_metadata=meta)
        self.assertIsInstance(grid.construction_metadata, types.MappingProxyType)
        self.assertIsInstance(grid.construction_metadata["nested"], types.MappingProxyType)
        meta["mol"] = "mutated"
        meta["nested"]["grid_level"] = 999
        self.assertEqual(grid.construction_metadata["mol"], "H2O")
        self.assertEqual(grid.construction_metadata["nested"]["grid_level"], 2)
        with self.assertRaises(TypeError):
            grid.construction_metadata["mol"] = "cannot-assign"

    def test_grid_spec_sha256_deterministic_and_content_sensitive(self):
        coords, weights = self._random_grid_arrays(seed=7)
        grid_a = build_ibp_grid(coords.copy(), weights.copy())
        grid_b = build_ibp_grid(coords.copy(), weights.copy())
        self.assertEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)
        other_coords, other_weights = self._random_grid_arrays(seed=8)
        grid_c = build_ibp_grid(other_coords, other_weights)
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_c.grid_spec_sha256)

    def test_tampering_via_dataclasses_replace_is_rejected(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        for kwargs in (
            {"grid_spec_sha256": "0" * 64},
            {"coords_sha256": "0" * 64},
            {"weights_sha256": "0" * 64},
            {"n_grid": grid.n_grid + 1},
            {"device": "gpu:0"},
            {"dtype": "float32"},
            {"schema_version": "999"},
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(grid, **kwargs)

    def test_direct_construction_bypassing_builder_is_still_validated(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        with self.assertRaises(ValueError):
            IBPGrid(
                coords=grid.coords, weights=grid.weights, n_grid=grid.n_grid,
                backend=grid.backend, device=grid.device, dtype=grid.dtype,
                coincident_point_policy=grid.coincident_point_policy,
                coords_sha256="0" * 64, weights_sha256=grid.weights_sha256,
                numpy_version=grid.numpy_version, jax_version=grid.jax_version,
                construction_metadata={}, schema_version=grid.schema_version,
                grid_spec_sha256=grid.grid_spec_sha256,
            )


class TestIBPOperatorPlan(unittest.TestCase):
    def _grid(self, n=6, seed=3):
        rng = np.random.default_rng(seed)
        return build_ibp_grid(rng.normal(size=(n, 3)), rng.normal(size=n))

    def test_plan_round_trip_numpy(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid, eval_block_size=16, source_block_size=64)
        self.assertIs(plan.grid, grid)
        self.assertEqual(plan.method, "direct")
        self.assertEqual(plan.eval_block_size, 16)
        self.assertEqual(plan.source_block_size, 64)
        self.assertIsNone(plan.tolerance)
        self.assertEqual(plan.backend, grid.backend)
        self.assertEqual(plan.device, grid.device)
        self.assertEqual(plan.dtype, grid.dtype)
        self.assertEqual(plan.method_version, "1")

    def test_plan_jax_backend_round_trip(self):
        rng = np.random.default_rng(4)
        coords = jnp.asarray(rng.normal(size=(5, 3)))
        weights = jnp.asarray(rng.normal(size=5))
        grid = build_ibp_grid(
            coords, weights, backend="jax",
            coords_identity=hashlib.sha256(b"c").hexdigest(),
            weights_identity=hashlib.sha256(b"w").hexdigest(),
        )
        plan = build_ibp_operator_plan(grid)
        self.assertEqual(plan.backend, "jax")
        self.assertEqual(plan.device, grid.device)
        self.assertEqual(plan.dtype, grid.dtype)

    def test_plan_requires_ibp_grid_type(self):
        with self.assertRaises(TypeError):
            build_ibp_operator_plan("not-a-grid")

    def test_plan_rejects_unsupported_method(self):
        grid = self._grid()
        for method in ("hierarchical", "nufft", "fft"):
            with self.assertRaises(ValueError):
                build_ibp_operator_plan(grid, method=method)

    def test_plan_rejects_tolerance_for_direct(self):
        grid = self._grid()
        with self.assertRaises(ValueError):
            build_ibp_operator_plan(grid, tolerance=1e-6)

    def test_plan_rejects_bad_block_sizes(self):
        grid = self._grid()
        for kwargs in (
            {"eval_block_size": 0}, {"eval_block_size": -4},
            {"source_block_size": 0}, {"eval_block_size": True},
            {"eval_block_size": 3.5},
        ):
            with self.assertRaises(ValueError):
                build_ibp_operator_plan(grid, **kwargs)

    def test_plan_block_size_accepts_integral_float(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid, eval_block_size=8.0, source_block_size=32.0)
        self.assertEqual(plan.eval_block_size, 8)
        self.assertIsInstance(plan.eval_block_size, int)

    def test_plan_provenance_is_deep_frozen(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(
            grid, upstream_provenance={"caller": "test", "nested": {"a": 1}}
        )
        self.assertIsInstance(plan.provenance, types.MappingProxyType)
        self.assertEqual(plan.provenance["upstream_provenance"]["caller"], "test")
        with self.assertRaises(TypeError):
            plan.provenance["caller"] = "cannot-assign"

    def test_plan_tampering_via_dataclasses_replace_is_rejected(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid)
        for kwargs in (
            {"operator_spec_sha256": "0" * 64},
            {"method": "direct", "backend": "jax"},
            {"device": "gpu:0"},
            {"dtype": "float32"},
            {"method_version": "999"},
            {"eval_block_size": plan.eval_block_size + 1},
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(plan, **kwargs)

    def test_plan_direct_construction_bypassing_builder_is_still_validated(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid)
        with self.assertRaises(ValueError):
            IBPOperatorPlan(
                grid=grid, method=plan.method, eval_block_size=plan.eval_block_size,
                source_block_size=plan.source_block_size, tolerance=plan.tolerance,
                backend=plan.backend, device=plan.device, dtype=plan.dtype,
                method_version=plan.method_version, provenance={},
                operator_spec_sha256="0" * 64,
            )

    def test_plan_holds_no_array_data_of_its_own(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid)
        for field in dataclasses.fields(plan):
            if field.name == "grid":
                continue
            value = getattr(plan, field.name)
            self.assertNotIsInstance(value, np.ndarray)
            self.assertNotIsInstance(value, jax.Array)

    def test_plan_scales_to_large_grid_without_materializing_pairs(self):
        rng = np.random.default_rng(5)
        n = 20000
        grid = build_ibp_grid(rng.normal(size=(n, 3)), rng.normal(size=n))
        # Must be cheap -- a dense (3, n, n) kernel here would be ~9.6GB at
        # float64, and building this plan does none of that work.
        plan = build_ibp_operator_plan(grid, eval_block_size=128, source_block_size=4096)
        self.assertEqual(plan.grid.n_grid, n)


class TestIBPProvenanceCanonicalEncoding(unittest.TestCase):
    """Regressions for Alice's task #5 review finding: repr()-based
    provenance hashing is hash-seed-dependent for sets/frozensets (a
    single construction could even disagree with itself) and silently
    truncates large ndarray content -- fixed by _canonical_encode."""

    def _grid_with_metadata(self, construction_metadata):
        rng = np.random.default_rng(9)
        coords = rng.normal(size=(4, 3))
        weights = rng.normal(size=4)
        return build_ibp_grid(coords, weights, construction_metadata=construction_metadata)

    def test_cross_process_hash_seed_determinism(self):
        script = (
            "import numpy as np\n"
            "from pytc.df.ibp import build_ibp_grid\n"
            "c = np.arange(12, dtype=float).reshape(4, 3)\n"
            "w = np.ones(4)\n"
            "g = build_ibp_grid(c, w, construction_metadata="
            "{'labels': {'alpha', 'beta', 'gamma', 'delta'}})\n"
            "print(g.grid_spec_sha256)\n"
        )
        digests = set()
        for seed in ("1", "2", "3", "4", "0", "100"):
            full_env = dict(os.environ)
            full_env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-W", "error", "-c", script],
                capture_output=True, text=True, env=full_env, check=True,
            )
            # pytc's own startup banner logs to stdout too; the digest is
            # always the last line since it's the script's final print().
            digests.add(result.stdout.strip().splitlines()[-1])
        self.assertEqual(len(digests), 1, f"hash-seed-dependent digests: {digests}")

    def test_dict_key_insertion_order_does_not_affect_hash(self):
        grid_a = self._grid_with_metadata({"a": 1, "b": 2, "c": 3})
        grid_b = self._grid_with_metadata({"c": 3, "a": 1, "b": 2})
        self.assertEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_set_construction_order_does_not_affect_hash(self):
        grid_a = self._grid_with_metadata({"labels": {"alpha", "beta", "gamma", "delta"}})
        grid_b = self._grid_with_metadata({"labels": {"delta", "gamma", "beta", "alpha"}})
        self.assertEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_distinct_set_content_gives_distinct_hash(self):
        grid_a = self._grid_with_metadata({"labels": {"alpha", "beta"}})
        grid_b = self._grid_with_metadata({"labels": {"alpha", "gamma"}})
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_large_array_middle_content_is_not_truncated(self):
        # repr()'s numpy summarization would show "..." in the middle and
        # hide a difference confined entirely to the middle of a long array.
        base = np.arange(2000, dtype=np.float64)
        modified = base.copy()
        modified[1000] += 1.0
        grid_a = self._grid_with_metadata({"payload": base})
        grid_b = self._grid_with_metadata({"payload": modified})
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_tuple_element_order_is_preserved_and_sensitive(self):
        grid_a = self._grid_with_metadata({"seq": (1, 2, 3)})
        grid_b = self._grid_with_metadata({"seq": (3, 2, 1)})
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_delimiter_collision_in_string_content_is_rejected(self):
        """Alice's round-2 finding: a delimiter-separated (non-length-
        prefixed) encoder let a string CONTAINING a literal delimiter
        collide with an unrelated sibling tuple -- ("a,str:b",) and
        ("a", "b") serialized to identical bytes. The TLV encoder must
        distinguish them since they are genuinely different structures."""
        grid_a = self._grid_with_metadata({"seq": ("a,str:b",)})
        grid_b = self._grid_with_metadata({"seq": ("a", "b")})
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)

    def test_delimiter_collision_in_raw_bytes_content_is_rejected(self):
        # Analogous case with raw bytes instead of str -- exercises the
        # "y" (bytes) TLV branch rather than "s" (str).
        grid_a = self._grid_with_metadata({"seq": (b"a,bytes:b",)})
        grid_b = self._grid_with_metadata({"seq": (b"a", b"bytes:b")})
        self.assertNotEqual(grid_a.grid_spec_sha256, grid_b.grid_spec_sha256)


if __name__ == "__main__":
    unittest.main()
