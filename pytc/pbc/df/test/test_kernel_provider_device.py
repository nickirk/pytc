"""Tests for the device-path KernelProvider seam (design v2.1 section 7):
- raw_kernel_apply's derived q<->-q dagger law at the pre-phased,
  pre-conjugate seam.
- RawKernelProvider is a thin, correct wrapper around raw_kernel_apply.
- apply_kernel_and_solve_device (RawKernelProvider) matches the CPU/
  NumPy oracle apply_raw_kernel_and_solve bit-tier (V3's "JAX device
  path vs NumPy oracle <=1e-12" sub-gate).
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_kernel_and_solve_device,
    apply_raw_kernel_and_solve,
    build_coul_kpt_device,
    build_pi_eta,
    precompute_phase_all_q,
    raw_kernel_apply,
)
from pytc.pbc.df.kpts import canonicalize_kpts


def _make_cell():
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag([2.0, 2.0, 2.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 40.0
    cell.build()
    return cell


def _tr_symmetric_fixture(rng, n_kpts, neg, shape):
    X = np.zeros((n_kpts,) + shape, dtype=np.complex128)
    done = set()
    for k in range(n_kpts):
        if k in done:
            continue
        nk = int(neg[k])
        if nk == k:
            X[k] = rng.normal(size=shape)
        else:
            re, im = rng.normal(size=shape), rng.normal(size=shape)
            X[k] = re + 1j * im
            X[nk] = re - 1j * im
            done.add(nk)
        done.add(k)
    return X


class TestRawKernelApplyDaggerLaw(unittest.TestCase):
    def test_dagger_law_holds_for_arbitrary_lq(self):
        # The derived law apply(neg[q], conj(lq)) == conj(apply(q, lq))
        # holds for ANY (Nip, Ng) lq -- it does not depend on lq itself
        # being time-reversal symmetric across a k-collection (that
        # structure is what pair_convolve's eta guarantees upstream;
        # the provider seam's own law is a per-call identity).
        cell = _make_cell()
        rng = np.random.default_rng(80)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        n_ip = 3
        lq = (
            rng.normal(size=(n_ip, n_grid)) + 1j * rng.normal(size=(n_ip, n_grid))
        ).astype(np.complex128)
        q_kpt = rng.normal(size=3) * 0.3

        v_q = raw_kernel_apply(lq, cell=cell, q_kpt=q_kpt, grid_mesh=grid_mesh)
        v_negq = raw_kernel_apply(lq.conj(), cell=cell, q_kpt=-q_kpt, grid_mesh=grid_mesh)
        np.testing.assert_allclose(np.asarray(v_negq), np.asarray(v_q).conj(), atol=1e-9)

    def test_dagger_law_at_gamma_is_self_conjugate(self):
        # q = Gamma is its own neg[q]; the law degenerates to
        # apply(Gamma, conj(lq)) == conj(apply(Gamma, lq)), a genuine
        # constraint (not vacuous) since it must hold for the SAME q.
        cell = _make_cell()
        rng = np.random.default_rng(81)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        lq = (rng.normal(size=(2, n_grid)) + 1j * rng.normal(size=(2, n_grid))).astype(
            np.complex128
        )
        gamma = np.zeros(3)
        v = raw_kernel_apply(lq, cell=cell, q_kpt=gamma, grid_mesh=grid_mesh)
        v_conj_input = raw_kernel_apply(lq.conj(), cell=cell, q_kpt=gamma, grid_mesh=grid_mesh)
        np.testing.assert_allclose(np.asarray(v_conj_input), np.asarray(v).conj(), atol=1e-9)


class TestRawKernelProvider(unittest.TestCase):
    def test_apply_matches_raw_kernel_apply_directly(self):
        cell = _make_cell()
        rng = np.random.default_rng(82)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        lq = (rng.normal(size=(2, n_grid)) + 1j * rng.normal(size=(2, n_grid))).astype(
            np.complex128
        )
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=grid_mesh
        )
        for q in range(mesh_obj.n_kpts):
            v_direct = raw_kernel_apply(
                lq, cell=cell, q_kpt=mesh_obj.canonical_kpts[q], grid_mesh=grid_mesh
            )
            v_provider = provider.apply(q, lq)
            np.testing.assert_allclose(np.asarray(v_provider), np.asarray(v_direct), atol=0.0)

    def test_provenance_fields(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        prov = provider.provenance()
        self.assertEqual(prov["kernel_name"], "raw")
        self.assertIn("exxdiv", prov)
        self.assertIn("not_this_provider", prov["exxdiv"])
        self.assertEqual(prov["normalization"], "vol_over_ng_inside_provider")

    def test_rejects_out_of_range_q_index(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        with self.assertRaises(ValueError):
            provider.apply(mesh_obj.n_kpts, np.zeros((1, int(np.prod(cell.mesh)))))


class TestPrecomputePhaseAllQ(unittest.TestCase):
    """design v2.1 section 6 glue-fix (task #25/C2 item 1):
    precompute_phase_all_q batches the per-q Bloch phase computation
    that apply_kernel_and_solve_device used to redo from scratch (with
    a fresh grid_coords upload) on every call."""

    def test_matches_per_q_internal_computation(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)

        phase_all = precompute_phase_all_q(grid_coords, mesh_obj.canonical_kpts)
        self.assertEqual(phase_all.shape, (mesh_obj.n_kpts, grid_coords.shape[0]))
        for q in range(mesh_obj.n_kpts):
            expected = np.exp(-1j * (grid_coords @ mesh_obj.canonical_kpts[q]))
            np.testing.assert_allclose(np.asarray(phase_all[q]), expected, atol=1e-12)

    def test_rejects_malformed_shapes(self):
        with self.assertRaises(ValueError):
            precompute_phase_all_q(np.zeros((5, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            precompute_phase_all_q(np.zeros((5, 3)), np.zeros((3, 2)))


class TestApplyKernelAndSolveDeviceMatchesNumpyOracle(unittest.TestCase):
    def _setup(self, kmesh, seed=83, n_ip=3):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
        Pi, eta = build_pi_eta(X, ao, mesh_obj.phase)
        return cell, mesh_obj, grids, Pi, eta

    def test_device_path_matches_numpy_oracle_bit_tier(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        for q in range(mesh_obj.n_kpts):
            W_np, kern_np, info_np = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
            )
            W_dev, kern_dev, info_dev = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            np.testing.assert_allclose(np.asarray(kern_dev), kern_np, atol=1e-12, err_msg=f"kern_q q={q}")
            np.testing.assert_allclose(np.asarray(W_dev), W_np, atol=1e-10, err_msg=f"W_q q={q}")
            self.assertEqual(info_dev["n_retained"], info_np["n_retained"])

    def test_phase_q_gives_bit_identical_result_to_grid_coords_path(self):
        # design v2.1 section 6 glue-fix (task #25/C2 item 1): passing a
        # precomputed phase_q must reproduce EXACTLY (not just closely)
        # the same computation the internal grid_coords-based path did.
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        phase_all = precompute_phase_all_q(grids, mesh_obj.canonical_kpts)
        for q in range(mesh_obj.n_kpts):
            W_grid, kern_grid, _ = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            W_phase, kern_phase, _ = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], phase_q=phase_all[q], rtol=1e-8,
            )
            np.testing.assert_array_equal(np.asarray(kern_phase), np.asarray(kern_grid))
            np.testing.assert_array_equal(np.asarray(W_phase), np.asarray(W_grid))

    def test_rejects_neither_grid_coords_nor_phase_q(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        with self.assertRaises(ValueError):
            apply_kernel_and_solve_device(provider, 0, Pi[0], eta[0], rtol=1e-8)

    def test_rejects_malformed_phase_q_shape(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        with self.assertRaises(ValueError):
            apply_kernel_and_solve_device(
                provider, 0, Pi[0], eta[0], phase_q=np.zeros(3), rtol=1e-8
            )

    def test_self_paired_matches_numpy_oracle_and_is_exactly_real(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        q = 0  # Gamma, always self-paired
        self.assertEqual(int(mesh_obj.neg[q]), q)
        W_np, kern_np, _ = apply_raw_kernel_and_solve(
            Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
            grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8, self_paired=True,
        )
        W_dev, kern_dev, _ = apply_kernel_and_solve_device(
            provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8, self_paired=True,
        )
        np.testing.assert_array_equal(kern_np.imag, 0.0)
        np.testing.assert_array_equal(np.asarray(kern_dev).imag, 0.0)
        np.testing.assert_allclose(np.asarray(kern_dev), kern_np, atol=1e-12)
        np.testing.assert_allclose(np.asarray(W_dev), W_np, atol=1e-10)

    def test_device_w_q_is_hermitian(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        for q in range(mesh_obj.n_kpts):
            W_dev, _, _ = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            W_np = np.asarray(W_dev)
            np.testing.assert_allclose(W_np, W_np.conj().T, atol=1e-8, err_msg=f"q={q}")

    def test_zero_retained_modes_raises_with_q_index(self):
        # The device solve cannot raise on a traced value internally, so
        # this host wrapper must turn n_retained==0 into a precise,
        # q-indexed error rather than a silent W_q=0.
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        n_ip = Pi.shape[1]
        Pi_zero = np.zeros((n_ip, n_ip), dtype=np.complex128)
        with self.assertRaises(ValueError) as ctx:
            apply_kernel_and_solve_device(
                provider, 1, Pi_zero, eta[1], grid_coords=grids, rtol=1e-8,
            )
        self.assertIn("q_index=1", str(ctx.exception))

    def test_retained_solve_residual_gate_is_enforced(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        # A normal, well-conditioned q passes at the default gate...
        apply_kernel_and_solve_device(
            provider, 0, Pi[0], eta[0], grid_coords=grids, rtol=1e-8,
        )
        # ...but must raise if the gate is tightened below any possible
        # residual, proving the gate is actually enforced, not a no-op.
        with self.assertRaises(ValueError) as ctx:
            apply_kernel_and_solve_device(
                provider, 0, Pi[0], eta[0], grid_coords=grids, rtol=1e-8,
                retained_solve_residual_gate=-1.0,
            )
        self.assertIn("q_index=0", str(ctx.exception))


class TestBuildCoulKptDevice(unittest.TestCase):
    def _setup(self, kmesh, seed=90, n_ip=3):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
        Pi, eta = build_pi_eta(X, ao, mesh_obj.phase)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        return cell, mesh_obj, grids, Pi, eta, provider

    def test_matches_numpy_oracle_looped_over_all_q(self):
        cell, mesh_obj, grids, Pi, eta, provider = self._setup([1, 1, 3])
        coul_kpt, kern_kpt, infos, n_calls = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, rtol=1e-8,
        )
        for q in range(mesh_obj.n_kpts):
            W_np, kern_np, _ = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
            )
            np.testing.assert_allclose(np.asarray(kern_kpt[q]), kern_np, atol=1e-12, err_msg=f"q={q}")
            np.testing.assert_allclose(np.asarray(coul_kpt[q]), W_np, atol=1e-10, err_msg=f"q={q}")

    def test_conjugate_shortcut_matches_independent_build(self):
        # The strongest test: verify the SKIPPED neg[q] value against an
        # INDEPENDENT direct pipeline call at neg[q] (its own inputs,
        # not derived from q's result), not just self-consistency with
        # the shortcut's own output.
        cell, mesh_obj, grids, Pi, eta, provider = self._setup([1, 1, 3])
        coul_kpt, kern_kpt, infos, n_calls = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, rtol=1e-8,
        )
        neg = mesh_obj.neg
        paired_q = next(q for q in range(mesh_obj.n_kpts) if int(neg[q]) != q)
        nq = int(neg[paired_q])
        W_independent, kern_independent, _ = apply_kernel_and_solve_device(
            provider, nq, Pi[nq], eta[nq], grid_coords=grids, rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(coul_kpt[nq]), np.asarray(W_independent), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(kern_kpt[nq]), np.asarray(kern_independent), atol=1e-12
        )

    def test_all_self_paired_mesh_gives_exactly_real_kern_and_coul(self):
        # design v2.1 section 5 (retention-policy fix follow-up): a
        # [2,1,1] mesh has BOTH k-points self-paired (neg=[0,1]) --
        # build_coul_kpt_device must thread self_paired through to
        # BOTH, not just q=0/Gamma.
        cell, mesh_obj, grids, Pi, eta, provider = self._setup([2, 1, 1])
        self.assertEqual(list(mesh_obj.neg), [0, 1])
        coul_kpt, kern_kpt, infos, n_calls = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, rtol=1e-8,
        )
        for q in range(mesh_obj.n_kpts):
            np.testing.assert_array_equal(np.asarray(kern_kpt[q]).imag, 0.0, err_msg=f"q={q}")
            np.testing.assert_array_equal(np.asarray(coul_kpt[q]).imag, 0.0, err_msg=f"q={q}")

    def test_efficiency_count_is_half_plus_self_paired(self):
        cell, mesh_obj, grids, Pi, eta, provider = self._setup([1, 1, 3])
        neg = mesh_obj.neg
        n_self_paired = sum(1 for q in range(mesh_obj.n_kpts) if int(neg[q]) == q)
        n_pairs = (mesh_obj.n_kpts - n_self_paired) // 2
        expected_calls = n_self_paired + n_pairs
        _, _, _, n_calls = build_coul_kpt_device(provider, Pi, eta, grids, mesh_obj, rtol=1e-8)
        self.assertEqual(n_calls, expected_calls)
        self.assertLess(n_calls, mesh_obj.n_kpts)

    def test_rejects_mismatched_leading_dimension(self):
        cell, mesh_obj, grids, Pi, eta, provider = self._setup([1, 1, 2])
        with self.assertRaises(ValueError):
            build_coul_kpt_device(provider, Pi[:1], eta, grids, mesh_obj, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
