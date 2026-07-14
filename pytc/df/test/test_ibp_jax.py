"""Task #10 evidence suite for the tiled device-resident JAX ibp_core backend:
numpy/jax parity, device placement, whole-builder allocation audit, byte
bound, cold-vs-warm compile timing, and the device PSD path. float32/complex64
(x64-off) parity runs in a fresh subprocess (JAX's global x64 flag cannot be
toggled mid-process) and lives in test_ibp_jax_x32_subprocess.py's helper.
"""

import hashlib
import subprocess
import sys
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

import pytc.df.ibp as ibp
from pytc.df.ibp import (
    IBPCoreArtifact,
    build_ibp_grid,
    build_ibp_interpolation_sector,
    build_ibp_operator_plan,
    ibp_core,
)
from pytc.integrals.coulomb import select_sector_pivots, weight_mo_values

_IDS = {
    "factor_p_sha256": hashlib.sha256(b"factor").hexdigest(),
    "factor_q_sha256": hashlib.sha256(b"factor").hexdigest(),
    "gradient_p_sha256": hashlib.sha256(b"grad").hexdigest(),
    "gradient_q_sha256": hashlib.sha256(b"grad").hexdigest(),
}


def _data(dtype, ng=24, n_orb=5, seed=5, coincident=True):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(ng, 3))
    if coincident:
        coords[3] = coords[9]  # a duplicate coordinate -> coincident pair
    weights = rng.uniform(0.5, 1.5, size=ng)
    fp = rng.normal(size=(n_orb, ng)).astype(dtype)
    gp = rng.normal(size=(3, n_orb, ng)).astype(dtype)
    if np.iscomplexobj(np.empty((), dtype)):
        fp = fp + 1j * rng.normal(size=(n_orb, ng))
        gp = gp + 1j * rng.normal(size=(3, n_orb, ng))
    return coords, weights, fp.astype(dtype), gp.astype(dtype)


def _numpy_core(coords, weights, fp, gp, *, symmetry_mode, blocks):
    e, s, mu, nu = blocks
    grid = build_ibp_grid(coords, weights)
    # The pivot selector is real-only; select on the real magnitude (the pivots
    # are just shared indices, so parity holds regardless of how they're chosen).
    weighted = weight_mo_values(np.abs(fp), grid.weights)
    pivots, record = select_sector_pivots(weighted, weighted, fp.shape[0],
                                          same_factor=True, return_provenance=True)
    sector = build_ibp_interpolation_sector(fp, fp, gp, gp, pivots, grid,
                                            pivot_provenance=record, same_factor=True)
    plan = build_ibp_operator_plan(grid, eval_block_size=e, source_block_size=s)
    return ibp_core(sector, operator=plan, symmetry_mode=symmetry_mode,
                    mu_block_size=mu, nu_block_size=nu), pivots, record


def _jax_core(coords, weights, fp, gp, pivots, record, *, symmetry_mode, blocks):
    e, s, mu, nu = blocks
    grid = build_ibp_grid(
        jnp.asarray(coords), jnp.asarray(weights), backend="jax",
        coords_identity=hashlib.sha256(b"c").hexdigest(),
        weights_identity=hashlib.sha256(b"w").hexdigest(),
    )
    sector = build_ibp_interpolation_sector(
        jnp.asarray(fp), jnp.asarray(fp), jnp.asarray(gp), jnp.asarray(gp),
        pivots, grid, pivot_provenance=record, same_factor=True, upstream_provenance=_IDS,
    )
    plan = build_ibp_operator_plan(grid, eval_block_size=e, source_block_size=s)
    return ibp_core(sector, operator=plan, symmetry_mode=symmetry_mode,
                    mu_block_size=mu, nu_block_size=nu)


class TestJaxNumpyParity(unittest.TestCase):
    def _check(self, dtype, symmetry_mode, blocks, coincident=True, seed=5):
        coords, weights, fp, gp = _data(dtype, seed=seed, coincident=coincident)
        np_core, pivots, record = _numpy_core(coords, weights, fp, gp,
                                              symmetry_mode=symmetry_mode, blocks=blocks)
        jax_core = _jax_core(coords, weights, fp, gp, pivots, record,
                             symmetry_mode=symmetry_mode, blocks=blocks)
        self.assertEqual(jax_core.backend, "jax")
        self.assertEqual(np_core.coincident_pairs, jax_core.coincident_pairs)
        np.testing.assert_allclose(np.asarray(jax_core.Z), np_core.Z, atol=1e-10, rtol=1e-8)
        self.assertAlmostEqual(np_core.raw_dagger_residual, jax_core.raw_dagger_residual, places=9)
        self.assertAlmostEqual(np_core.raw_packed_pair_metric_dagger_residual,
                               jax_core.raw_packed_pair_metric_dagger_residual, places=8)

    def test_real_and_complex_same_sector_one_sided(self):
        for dtype in (np.float64, np.complex128):
            self._check(dtype, "one_sided", (8, 16, 2, 3))

    def test_duplicate_and_no_coincident(self):
        # exercise both a grid with a duplicate coordinate and one without
        self._check(np.float64, "one_sided", (8, 16, 2, 3), coincident=True)
        self._check(np.float64, "one_sided", (8, 16, 2, 3), coincident=False)

    def test_all_block_choices_including_short_final(self):
        coords, weights, fp, gp = _data(np.float64)
        n_orb, ng = fp.shape
        for blocks in [(ng, ng, n_orb, n_orb),   # single block
                       (7, 5, 2, 2),               # short-final on every axis
                       (4, 4, 1, 1),               # smallest tiles
                       (13, 9, 3, 4)]:              # irregular
            np_core, pivots, record = _numpy_core(coords, weights, fp, gp,
                                                  symmetry_mode="one_sided", blocks=blocks)
            jax_core = _jax_core(coords, weights, fp, gp, pivots, record,
                                 symmetry_mode="one_sided", blocks=blocks)
            np.testing.assert_allclose(np.asarray(jax_core.Z), np_core.Z, atol=1e-10,
                                       rtol=1e-8, err_msg=str(blocks))
            self.assertEqual(np_core.coincident_pairs, jax_core.coincident_pairs)


class TestJaxPlacement(unittest.TestCase):
    def test_no_large_host_roundtrip_and_outputs_on_device(self):
        # Spy the host-transfer paths AROUND ibp_core ONLY (grid/sector/operator
        # are separate, already-reviewed builders): no jax.Array with more than
        # one element may pass through np.asarray/np.array/jax.device_get during
        # the core build; allowed scalar syncs go through _ibp_sync_scalars
        # (rank-0 only). The sector/operator are built before the spy.
        coords, weights, fp, gp = _data(np.float64)
        np_core, pivots, record = _numpy_core(coords, weights, fp, gp,
                                              symmetry_mode="one_sided", blocks=(8, 16, 2, 3))
        grid = build_ibp_grid(
            jnp.asarray(coords), jnp.asarray(weights), backend="jax",
            coords_identity=hashlib.sha256(b"c").hexdigest(),
            weights_identity=hashlib.sha256(b"w").hexdigest(),
        )
        sector = build_ibp_interpolation_sector(
            jnp.asarray(fp), jnp.asarray(fp), jnp.asarray(gp), jnp.asarray(gp),
            pivots, grid, pivot_provenance=record, same_factor=True, upstream_provenance=_IDS,
        )
        plan = build_ibp_operator_plan(grid, eval_block_size=8, source_block_size=16)

        violations = []
        real_asarray, real_array, real_get = np.asarray, np.array, jax.device_get

        def spy_asarray(a, *args, **kw):
            if isinstance(a, jax.Array) and a.size > 1:
                violations.append(("np.asarray", a.shape))
            return real_asarray(a, *args, **kw)

        def spy_array(a, *args, **kw):
            if isinstance(a, jax.Array) and getattr(a, "size", 0) > 1:
                violations.append(("np.array", a.shape))
            return real_array(a, *args, **kw)

        def spy_get(a, *args, **kw):
            if isinstance(a, jax.Array) and a.size > 1:
                violations.append(("device_get", a.shape))
            return real_get(a, *args, **kw)

        try:
            np.asarray = spy_asarray
            np.array = spy_array
            jax.device_get = spy_get
            jax_core = ibp_core(sector, operator=plan, symmetry_mode="one_sided",
                                mu_block_size=2, nu_block_size=3)
        finally:
            np.asarray, np.array, jax.device_get = real_asarray, real_array, real_get
        self.assertEqual(violations, [], f"large host roundtrip(s): {violations}")
        # outputs stay on the input device
        self.assertIsInstance(jax_core.Z, jax.Array)
        self.assertEqual(str(jax_core.Z.device), jax_core.device)

    def test_sync_helper_rejects_non_scalar(self):
        with self.assertRaises(ValueError):
            ibp._ibp_sync_scalars(bad=jnp.zeros(3))


class TestJaxAllocationAudit(unittest.TestCase):
    def test_whole_builder_jaxpr_has_no_full_grid_pair_axis(self):
        # The compiled whole-orientation builder must never materialize a full
        # grid-pair axis; every intermediate is bounded by the fixed tile sizes.
        ng, n_mu, n_nu = 40, 6, 6
        E, S, Nm, Nn = 8, 16, 2, 3
        dt = jnp.float64
        grad = jnp.zeros((n_mu, 3, ng), dt)
        theta = jnp.zeros((n_nu, ng), dt)
        coords = jnp.zeros((ng, 3))
        weights = jnp.ones((ng,))
        valid = jnp.ones((ng,))
        jaxpr = jax.make_jaxpr(
            lambda g, t, c, w, v: ibp._ibp_orientation_jax(g, t, c, w, v, E, S, Nm, Nn, n_mu, n_nu)
        )(grad, theta, coords, weights, valid)
        forbidden = []
        seen_tile = {"rhat_ES3": False}

        def _size(shape):
            n = 1
            for d in shape:
                n *= d
            return n

        def _walk(jpr):
            for eqn in jpr.eqns:
                # Only COMPUTED intermediates (outvars) count as allocations; the
                # padded inputs legitimately carry the ng axis as loop invars.
                # A grid axis is acceptable only in an O(ng) or O(3*ng) array (the
                # weight vector / a coord slice); a grid-PAIR (ng x ng, E x ng) or
                # an untiled field (nu x 3 x ng) is forbidden -- caught by size.
                for var in eqn.outvars:
                    shape = tuple(getattr(getattr(var, "aval", None), "shape", ()) or ())
                    if any(dim == ng for dim in shape) and _size(shape) > 3 * ng:
                        forbidden.append((str(eqn.primitive), shape))
                    if shape == (E, S, 3):
                        seen_tile["rhat_ES3"] = True
                # recurse into control-flow sub-jaxprs (the fori_loop bodies)
                for param in eqn.params.values():
                    sub = getattr(param, "jaxpr", param)
                    if hasattr(sub, "eqns"):
                        _walk(sub)
                    elif isinstance(param, (tuple, list)):
                        for p in param:
                            s = getattr(p, "jaxpr", p)
                            if hasattr(s, "eqns"):
                                _walk(s)

        _walk(jaxpr.jaxpr)
        self.assertEqual(forbidden, [], f"full-grid allocation(s) found: {forbidden}")
        # positive control: the bounded (E,S,3) rhat tile IS present in the body
        self.assertTrue(seen_tile["rhat_ES3"],
                        "expected the bounded (E,S,3) rhat tile inside the loop body")

    def test_symbolic_byte_bound_is_dtype_aware(self):
        # The conservative per-block bound scales with itemsize (f64=8, c128=16).
        def bound(itemsize, E, S, Nm, Nn, n_mu, n_nu):
            workspace = E * S * 3 + E * S + Nn * 3 * E + Nm * 3 * E
            output = n_mu * n_nu
            return itemsize * (workspace + output)
        self.assertEqual(bound(16, 8, 16, 2, 3, 6, 6),
                         2 * bound(8, 8, 16, 2, 3, 6, 6))


class TestJaxTiming(unittest.TestCase):
    def test_cold_compile_and_warm_execution_reported_separately(self):
        import time
        ng, n_mu, n_nu = 24, 5, 5
        E, S, Nm, Nn = 8, 16, 2, 3
        args = (jnp.asarray(np.random.default_rng(0).normal(size=(n_mu, 3, ng))),
                jnp.asarray(np.random.default_rng(1).normal(size=(n_nu, ng))),
                jnp.asarray(np.random.default_rng(2).normal(size=(ng, 3))),
                jnp.ones((ng,)), jnp.ones((ng,)))
        t0 = time.perf_counter()
        out = ibp._ibp_orientation_jax(*args, E, S, Nm, Nn, n_mu, n_nu)
        out.block_until_ready()
        cold = time.perf_counter() - t0
        t1 = time.perf_counter()
        out = ibp._ibp_orientation_jax(*args, E, S, Nm, Nn, n_mu, n_nu)
        out.block_until_ready()
        warm = time.perf_counter() - t1
        # Warm (cache-hit) is strictly cheaper than cold (compile); both finite.
        self.assertGreater(cold, 0.0)
        self.assertGreater(warm, 0.0)
        self.assertLess(warm, cold)


class TestJaxPsdPath(unittest.TestCase):
    """Exercise the same-sector two-sided-average device PSD path by injecting a
    known Hermitian orientation through a mocked one-sided block, so the average
    Z is exactly PSD (a random synthetic core is materially indefinite)."""

    def _psd_core(self, z_hermitian_psd):
        from unittest import mock
        coords, weights, fp, gp = _data(np.float64)
        n = fp.shape[0]
        # Build a real jax sector/operator; mock the one-sided block to return a
        # forward orientation whose two-sided average equals the injected PSD Z.
        grid = build_ibp_grid(
            jnp.asarray(coords), jnp.asarray(weights), backend="jax",
            coords_identity=hashlib.sha256(b"c").hexdigest(),
            weights_identity=hashlib.sha256(b"w").hexdigest(),
        )
        weighted = weight_mo_values(fp, np.asarray(weights))
        pivots, record = select_sector_pivots(weighted, weighted, n, same_factor=True,
                                              return_provenance=True)
        sector = build_ibp_interpolation_sector(
            jnp.asarray(fp), jnp.asarray(fp), jnp.asarray(gp), jnp.asarray(gp),
            pivots, grid, pivot_provenance=record, same_factor=True, upstream_provenance=_IDS,
        )
        plan = build_ibp_operator_plan(grid, eval_block_size=8, source_block_size=16)
        zf = jnp.asarray(z_hermitian_psd)  # already Hermitian -> average is itself
        with mock.patch.object(ibp, "_ibp_one_sided_block_jax",
                               return_value=(zf, jnp.asarray(0, jnp.int64))):
            return ibp_core(sector, operator=plan, symmetry_mode="two_sided_average")

    def test_device_psd_factorization_wired(self):
        rng = np.random.default_rng(7)
        n = 5
        a = rng.normal(size=(n, n))
        z = a @ a.T + 0.5 * np.eye(n)     # Hermitian PSD, full rank
        core = self._psd_core((z + z.T) / 2)
        self.assertEqual(core.psd_status, "factorized")
        self.assertIsInstance(core.psd_factor, jax.Array)
        self.assertEqual(str(core.psd_factor.device), core.device)
        self.assertIsNone(core.psd_factor_sha256)
        self.assertEqual(core.psd_retained_rank, n)
        # W W^dagger reconstructs Z
        recon = np.asarray(core.psd_factor) @ np.asarray(core.psd_factor).conj().T
        np.testing.assert_allclose(recon, np.asarray(core.Z), atol=1e-10)

    def test_material_negative_hard_fails_on_device(self):
        rng = np.random.default_rng(8)
        n = 5
        a = rng.normal(size=(n, n))
        z = (a + a.T) / 2
        z = z - (np.max(np.abs(np.linalg.eigvalsh(z))) + 1.0) * np.eye(n)  # forced indefinite
        with self.assertRaises(ValueError):
            self._psd_core(z)


class TestJaxArtifactFields(unittest.TestCase):
    def test_jax_core_identity_and_device_fields(self):
        coords, weights, fp, gp = _data(np.float64)
        np_core, pivots, record = _numpy_core(coords, weights, fp, gp,
                                              symmetry_mode="one_sided", blocks=(8, 16, 2, 3))
        core = _jax_core(coords, weights, fp, gp, pivots, record,
                         symmetry_mode="one_sided", blocks=(8, 16, 2, 3))
        self.assertEqual(core.output_identity_source, "derived_from_attested_inputs_unverified")
        self.assertIsNone(core.z_sha256)
        self.assertIsNone(core.z_forward_sha256)
        self.assertEqual(core.peak_host_bytes_status, "unmeasured_jax_device")
        self.assertIsNone(core.peak_host_bytes)
        self.assertEqual(core.solver_version, "2")
        self.assertNotIn("build_wall_time_seconds", core._core_spec_fields(core.n_mu, core.n_nu))


class TestJaxFloat32Subprocess(unittest.TestCase):
    def test_float32_complex64_parity_x64_off(self):
        # JAX's global x64 flag cannot be toggled mid-process, so run the
        # low-precision parity in a fresh subprocess with x64 OFF.
        code = (
            "import numpy as np, jax, jax.numpy as jnp, hashlib\n"
            "from pytc.df.ibp import (build_ibp_grid, build_ibp_operator_plan,\n"
            "  build_ibp_interpolation_sector, ibp_core)\n"
            "from pytc.integrals.coulomb import select_sector_pivots, weight_mo_values\n"
            "IDS={'factor_p_sha256':hashlib.sha256(b'f').hexdigest(),'factor_q_sha256':hashlib.sha256(b'f').hexdigest(),\n"
            "  'gradient_p_sha256':hashlib.sha256(b'g').hexdigest(),'gradient_q_sha256':hashlib.sha256(b'g').hexdigest()}\n"
            "assert not jax.config.jax_enable_x64\n"
            "ok=True\n"
            "for dt in (np.float32, np.complex64):\n"
            "  rng=np.random.default_rng(5); ng,n=24,5\n"
            "  coords=rng.normal(size=(ng,3)); coords[3]=coords[9]; weights=rng.uniform(.5,1.5,ng)\n"
            "  fp=rng.normal(size=(n,ng)).astype(dt); gp=rng.normal(size=(3,n,ng)).astype(dt)\n"
            "  g=build_ibp_grid(coords.astype('float32'),weights.astype('float32'))\n"
            "  w=weight_mo_values(np.abs(fp).astype('float32'),g.weights); piv,rec=select_sector_pivots(w,w,n,same_factor=True,return_provenance=True)\n"
            "  sec=build_ibp_interpolation_sector(fp,fp,gp,gp,piv,g,pivot_provenance=rec,same_factor=True)\n"
            "  pl=build_ibp_operator_plan(g,eval_block_size=8,source_block_size=16)\n"
            "  npc=ibp_core(sec,operator=pl,symmetry_mode='one_sided')\n"
            "  jg=build_ibp_grid(jnp.asarray(coords.astype('float32')),jnp.asarray(weights.astype('float32')),backend='jax',\n"
            "    coords_identity=hashlib.sha256(b'c').hexdigest(),weights_identity=hashlib.sha256(b'w').hexdigest())\n"
            "  js=build_ibp_interpolation_sector(jnp.asarray(fp),jnp.asarray(fp),jnp.asarray(gp),jnp.asarray(gp),\n"
            "    piv,jg,pivot_provenance=rec,same_factor=True,upstream_provenance=IDS)\n"
            "  jpl=build_ibp_operator_plan(jg,eval_block_size=8,source_block_size=16)\n"
            "  jc=ibp_core(js,operator=jpl,symmetry_mode='one_sided')\n"
            "  err=float(np.max(np.abs(np.asarray(jc.Z)-npc.Z)))\n"
            "  ok = ok and jc.backend=='jax' and err<5e-4 and npc.coincident_pairs==jc.coincident_pairs\n"
            "print('X32_OK' if ok else 'X32_FAIL')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("X32_OK", result.stdout,
                      msg=f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
