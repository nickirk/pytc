"""V1 oracle tests for pytc.pbc.df.kpts (task #21, design v2.1 section 4/8).
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.kpts import (
    _SPC_TILE_COLS,
    pair_convolve_device,
    KptsMesh,
    canonicalize_kpts,
    check_time_reversal_residual,
    kpt_to_spc,
    kpt_to_spc_fft,
    pair_convolve,
    pair_convolve_fft_device,
    pair_convolve_q_device,
    spc_to_kpt,
    spc_to_kpt_fft,
)


def _make_cell(a_diag=(2.0, 2.0, 2.0)):
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    lattice = np.asarray(a_diag, dtype=np.float64)
    cell.a = lattice if lattice.shape == (3, 3) else np.diag(lattice)
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.build()
    return cell


def _tr_symmetric_fixture(rng, n_kpts, neg, shape):
    """Build a (n_kpts,)+shape complex128 array satisfying X[neg[k]] =
    conj(X[k]) for every k (self-paired k get a purely real value)."""
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


def _direct_sum(T, kmesh):
    """Z[q] = (1/sqrt(Nk)) sum_k T[k] * T[(q-k) mod mesh], with
    (q-k) mod mesh meaning per-AXIS modular subtraction on the
    (n1,n2,n3) lattice -- NOT flat-index arithmetic mod Nk, which only
    coincides with the correct per-axis result for a mesh with a single
    nontrivial axis (any mesh used in a 1D-reduced test here).

    The 1/sqrt(Nk) prefactor (not 1/Nk) matches pair_convolve's own
    unitary kpt_to_spc/spc_to_kpt convention (1/sqrt(Nk) split evenly
    across the forward and inverse transforms) -- re-derived directly
    from the orthogonality relation sum_R exp(iR.(k1+k2-q)) = Nk *
    delta(k1+k2-q mod G) once per direct-sum collapse, giving an overall
    (1/sqrt(Nk))*(1/Nk)*Nk = 1/sqrt(Nk) prefactor, not 1/Nk."""
    n1, n2, n3 = kmesh
    n_k = T.shape[0]
    n_ip, n_f = T.shape[1], T.shape[2]
    T_mesh = T.reshape(n1, n2, n3, n_ip, n_f)
    Z_mesh = np.zeros_like(T_mesh)
    for q1 in range(n1):
        for q2 in range(n2):
            for q3 in range(n3):
                acc = np.zeros((n_ip, n_f), dtype=np.complex128)
                for k1 in range(n1):
                    for k2 in range(n2):
                        for k3 in range(n3):
                            qk1, qk2, qk3 = (q1 - k1) % n1, (q2 - k2) % n2, (q3 - k3) % n3
                            acc += T_mesh[k1, k2, k3] * T_mesh[qk1, qk2, qk3]
                Z_mesh[q1, q2, q3] = acc / np.sqrt(n_k)
    return Z_mesh.reshape(n_k, n_ip, n_f)


class TestCanonicalizeKpts(unittest.TestCase):
    def test_both_wrap_around_gauges_canonicalize_identically(self):
        cell = _make_cell()
        kpts_f = cell.make_kpts([1, 1, 3], wrap_around=False)
        kpts_t = cell.make_kpts([1, 1, 3], wrap_around=True)
        mesh_f = canonicalize_kpts(cell, kpts_f)
        mesh_t = canonicalize_kpts(cell, kpts_t)
        np.testing.assert_array_equal(mesh_f.permutation, mesh_t.permutation)
        np.testing.assert_array_equal(mesh_f.neg, mesh_t.neg)
        np.testing.assert_allclose(mesh_f.canonical_kpts, mesh_t.canonical_kpts)

    def test_shuffled_input_order_recovers_correct_permutation(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        order = [2, 0, 1]
        mesh_ref = canonicalize_kpts(cell, kpts)
        mesh_shuf = canonicalize_kpts(cell, kpts[order])
        for i, orig_idx in enumerate(order):
            self.assertEqual(mesh_shuf.permutation[i], mesh_ref.permutation[orig_idx])

    def test_transport_round_trip_on_arbitrary_axis(self):
        cell = _make_cell()
        canonical_kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, canonical_kpts[[2, 0, 1]])
        canonical = np.arange(18).reshape(2, 3, 3)
        caller = mesh.from_canonical(canonical, axis=1)
        np.testing.assert_array_equal(mesh.to_canonical(caller, axis=1), canonical)
        with self.assertRaises(ValueError):
            mesh.to_canonical(np.zeros((2, 4)), axis=1)

    def test_subset_indices_accept_wrap_gauge_and_repetitions(self):
        cell = _make_cell()
        canonical_kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        wrapped_kpts = cell.make_kpts([1, 1, 3], wrap_around=True)
        mesh = canonicalize_kpts(cell, canonical_kpts)
        wrapped_mesh = canonicalize_kpts(cell, wrapped_kpts)
        query_positions = [2, 0, 2, 1]
        indices = mesh.canonical_indices(cell, wrapped_kpts[query_positions])
        np.testing.assert_array_equal(
            indices, wrapped_mesh.permutation[query_positions]
        )

    def test_subset_indices_reject_point_outside_mesh(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        with self.assertRaises(ValueError):
            mesh.canonical_indices(cell, kpts[:1] + np.array([[0.013, 0.021, 0.034]]))

    def test_gamma_present_at_canonical_index_zero(self):
        cell = _make_cell()
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        np.testing.assert_allclose(mesh.canonical_kpts[0], 0.0, atol=1e-10)

    def test_neg_is_involution_and_bijection(self):
        cell = _make_cell()
        for kmesh in ([1, 1, 2], [1, 1, 3], [2, 2, 2], [2, 3, 4]):
            kpts = cell.make_kpts(kmesh, wrap_around=False)
            mesh = canonicalize_kpts(cell, kpts)
            self.assertEqual(sorted(mesh.neg.tolist()), list(range(mesh.n_kpts)))
            for k in range(mesh.n_kpts):
                self.assertEqual(mesh.neg[mesh.neg[k]], k)

    def test_bz_edge_points_are_self_paired(self):
        # A 2-point mesh along one axis has both points on/adjacent to the
        # BZ edge; verify neg correctly resolves via mod-1 folding rather
        # than requiring them to pair with each other.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        for k in range(mesh.n_kpts):
            self.assertEqual(mesh.neg[k], k)

    def test_anisotropic_odd_even_mesh(self):
        cell = _make_cell(a_diag=(2.0, 2.5, 3.0))
        kpts = cell.make_kpts([2, 3, 4], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        self.assertEqual(mesh.kmesh, (2, 3, 4))
        self.assertEqual(mesh.n_kpts, 24)

    def test_fft_layout_is_bijective_on_anisotropic_mesh(self):
        cell = _make_cell(a_diag=(2.0, 2.5, 3.0))
        mesh = canonicalize_kpts(
            cell, cell.make_kpts([2, 3, 4], wrap_around=False)
        )
        for indices in (mesh.fft_k_indices, mesh.fft_r_indices):
            flat = np.ravel_multi_index(indices.T, mesh.kmesh)
            np.testing.assert_array_equal(np.sort(flat), np.arange(mesh.n_kpts))
        self.assertFalse(mesh.fft_k_indices.flags.writeable)
        self.assertFalse(mesh.fft_r_indices.flags.writeable)

    def test_rejects_incomplete_mesh(self):
        cell = _make_cell()
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        with self.assertRaises(ValueError):
            canonicalize_kpts(cell, kpts[:-1])

    def test_rejects_malformed_shape(self):
        cell = _make_cell()
        with self.assertRaises(ValueError):
            canonicalize_kpts(cell, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            canonicalize_kpts(cell, np.zeros((0, 3)))


class TestKptsMeshValidation(unittest.TestCase):
    def _valid_kwargs(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        return dict(
            kpts=mesh.kpts, kmesh=mesh.kmesh, n_kpts=mesh.n_kpts,
            canonical_kpts=mesh.canonical_kpts, permutation=mesh.permutation,
            neg=mesh.neg, ktol=mesh.ktol, phase=mesh.phase,
            fft_k_indices=mesh.fft_k_indices,
            fft_r_indices=mesh.fft_r_indices,
        )

    def test_valid_construction_round_trips(self):
        kwargs = self._valid_kwargs()
        mesh = KptsMesh(**kwargs)
        self.assertEqual(mesh.n_kpts, 2)
        self.assertFalse(mesh.kpts.flags.writeable)

    def test_rejects_non_bijective_permutation(self):
        kwargs = self._valid_kwargs()
        kwargs["permutation"] = np.array([0, 0])
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)

    def test_rejects_non_involution_neg(self):
        kwargs = self._valid_kwargs()
        kwargs["neg"] = np.array([1, 1])  # not sorted-bijective even
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)

    def test_rejects_missing_gamma(self):
        kwargs = self._valid_kwargs()
        shifted = kwargs["canonical_kpts"].copy()
        shifted[0] = shifted[0] + 0.3
        kwargs["canonical_kpts"] = shifted
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)

    def test_rejects_bad_ktol(self):
        kwargs = self._valid_kwargs()
        kwargs["ktol"] = -1e-8
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)

    def test_rejects_non_bijective_fft_layout(self):
        kwargs = self._valid_kwargs()
        bad = kwargs["fft_k_indices"].copy()
        bad[1] = bad[0]
        kwargs["fft_k_indices"] = bad
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)

    def test_rejects_phase_that_disagrees_with_fft_layout(self):
        kwargs = self._valid_kwargs()
        bad = kwargs["phase"].copy()
        bad[0, 0] += 1e-3
        kwargs["phase"] = bad
        with self.assertRaises(ValueError):
            KptsMesh(**kwargs)


class TestCheckTimeReversalResidual(unittest.TestCase):
    def test_valid_tr_pair_passes(self):
        rng = np.random.default_rng(2)
        neg = np.array([0, 2, 1])
        ao = _tr_symmetric_fixture(rng, 3, neg, (5, 4))
        residual = check_time_reversal_residual(ao, neg)
        self.assertLess(residual, 1e-10)

    def test_independent_arrays_trip_the_gate(self):
        rng = np.random.default_rng(3)
        neg = np.array([0, 2, 1])
        ao = rng.normal(size=(3, 5, 4)) + 1j * rng.normal(size=(3, 5, 4))
        with self.assertRaises(ValueError):
            check_time_reversal_residual(ao, neg)


class TestPairConvolve(unittest.TestCase):
    def test_gamma_only_reduces_to_molecular_elementwise_square(self):
        cell = _make_cell()
        mesh = canonicalize_kpts(cell, cell.get_kpts([1, 1, 1], wrap_around=False))
        rng = np.random.default_rng(10)
        n_ip, n_f, n_ao = 3, 4, 5
        X = rng.normal(size=(1, n_ip, n_ao)).astype(np.complex128)
        Y = rng.normal(size=(1, n_f, n_ao)).astype(np.complex128)
        Z = pair_convolve(X, Y, mesh.phase)
        expected = np.einsum("Iu,fu->If", X[0], Y[0].conj()) ** 2
        np.testing.assert_allclose(Z[0], expected, atol=1e-12)

    def test_direct_sum_agreement_on_tiny_meshes(self):
        cell = _make_cell()
        rng = np.random.default_rng(11)
        n_ip, n_f, n_ao = 3, 4, 5
        for kmesh_spec in ([1, 1, 2], [1, 1, 3], [2, 2, 2]):
            kpts = cell.make_kpts(kmesh_spec, wrap_around=False)
            mesh = canonicalize_kpts(cell, kpts)
            X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
            Y = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_f, n_ao))
            Z = pair_convolve(X, Y, mesh.phase)
            T = np.einsum("kIu,kfu->kIf", X, Y.conj())
            Z_direct = _direct_sum(T, mesh.kmesh)
            np.testing.assert_allclose(Z, Z_direct, atol=1e-10, err_msg=str(kmesh_spec))

    def test_q_neg_q_dagger_closure(self):
        cell = _make_cell()
        rng = np.random.default_rng(12)
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (3, 5))
        Y = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (4, 5))
        Z = pair_convolve(X, Y, mesh.phase)
        np.testing.assert_allclose(Z[mesh.neg], Z.conj(), atol=1e-10)

    def test_delta_field_analytic_case(self):
        # X/Y are nonzero only at Gamma (a "delta field" in k-space) -> the
        # real-space T_R is CONSTANT across all Nk supercell images
        # (T_R[r] = T[0]/Nk, standard IFFT-of-a-delta identity), so
        # Z_R = T_R**2 is also constant = (T[0]/Nk)**2, and Z[q] is
        # nonzero ONLY at q=Gamma (index 0): the unitary spc_to_kpt
        # transform of a length-Nk constant array c sums Nk copies of c
        # (scaled by 1/sqrt(Nk)) at q=0 and cancels (DFT orthogonality)
        # elsewhere, giving Z[0] = sqrt(Nk) * (T[0]/sqrt(Nk))**2.
        cell = _make_cell()
        n_ip, n_f, n_ao = 2, 3, 4
        n_k = 4
        mesh = canonicalize_kpts(cell, cell.get_kpts([1, 1, n_k], wrap_around=False))
        rng = np.random.default_rng(14)
        X = np.zeros((n_k, n_ip, n_ao), dtype=np.complex128)
        Y = np.zeros((n_k, n_f, n_ao), dtype=np.complex128)
        X[0] = rng.normal(size=(n_ip, n_ao))
        Y[0] = rng.normal(size=(n_f, n_ao))
        Z = pair_convolve(X, Y, mesh.phase)
        T0 = X[0] @ Y[0].conj().T
        expected_gamma = (T0 ** 2) / np.sqrt(n_k)
        np.testing.assert_allclose(Z[0], expected_gamma, atol=1e-10)
        for q in range(1, n_k):
            np.testing.assert_allclose(Z[q], 0.0, atol=1e-10)

    def test_rejects_malformed_shapes_and_kmesh(self):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        Y = rng.normal(size=(2, 5, 4)).astype(np.complex128)
        phase2 = np.eye(2, dtype=np.complex128)
        with self.assertRaises(ValueError):
            pair_convolve(X, Y, np.eye(3, dtype=np.complex128))  # phase shape != (Nk,Nk)
        with self.assertRaises(ValueError):
            pair_convolve(X, rng.normal(size=(2, 5, 6)).astype(np.complex128), phase2)
        with self.assertRaises(ValueError):
            pair_convolve(X[:, :0], Y, phase2)


class TestKptToSpcSpcToKpt(unittest.TestCase):
    def test_round_trip_recovers_original(self):
        cell = _make_cell()
        mesh = canonicalize_kpts(cell, cell.get_kpts([1, 1, 3], wrap_around=False))
        rng = np.random.default_rng(60)
        m_spc = rng.normal(size=(mesh.n_kpts, 4, 5))
        m_kpt = spc_to_kpt(m_spc, mesh.phase)
        m_spc_recovered = kpt_to_spc(m_kpt, mesh.phase)
        np.testing.assert_allclose(m_spc_recovered, m_spc, atol=1e-12)

    def test_pair_convolve_matches_manual_kpt_to_spc_spc_to_kpt_composition(self):
        # pair_convolve is now implemented AS this composition -- an
        # independent manual composition must match it exactly.
        cell = _make_cell()
        mesh = canonicalize_kpts(cell, cell.get_kpts([1, 1, 3], wrap_around=False))
        rng = np.random.default_rng(61)
        n_ip, n_f, n_ao = 3, 4, 5

        X_tr = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        Y = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_f, n_ao))

        Z = pair_convolve(X_tr, Y, mesh.phase)

        T = np.einsum("kIu,kfu->kIf", X_tr, Y.conj(), optimize=True)
        T_R = kpt_to_spc(T, mesh.phase)
        Z_R = T_R * T_R
        Z_manual = spc_to_kpt(Z_R, mesh.phase)

        np.testing.assert_allclose(Z, Z_manual, atol=0.0)

    def test_kpt_to_spc_rejects_non_tr_symmetric_input(self):
        cell = _make_cell()
        mesh = canonicalize_kpts(cell, cell.get_kpts([1, 1, 3], wrap_around=False))
        rng = np.random.default_rng(62)
        m_kpt = (rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))).astype(np.complex128)
        with self.assertRaises(ValueError):
            kpt_to_spc(m_kpt, mesh.phase)

    def test_rejects_malformed_shapes_and_kmesh(self):
        rng = np.random.default_rng(63)
        m = rng.normal(size=(4, 3, 3))
        phase3 = np.eye(3, dtype=np.complex128)
        with self.assertRaises(ValueError):
            kpt_to_spc(m, phase3)  # phase shape != (4,4)
        with self.assertRaises(ValueError):
            spc_to_kpt(m, phase3)

    def test_mapped_fft_matches_dense_phase_on_nontrivial_meshes(self):
        cell = _make_cell(a_diag=(2.0, 2.5, 3.0))
        rng = np.random.default_rng(64)
        for kmesh in ([1, 1, 3], [2, 2, 2], [2, 3, 4]):
            with self.subTest(kmesh=kmesh):
                mesh = canonicalize_kpts(
                    cell, cell.make_kpts(kmesh, wrap_around=False)
                )
                spc = rng.normal(size=(mesh.n_kpts, 3, 5))
                kpt = spc_to_kpt(spc, mesh.phase)
                np.testing.assert_allclose(
                    kpt_to_spc_fft(kpt, mesh),
                    kpt_to_spc(kpt, mesh.phase),
                    rtol=0.0,
                    atol=2e-14,
                )
                np.testing.assert_allclose(
                    spc_to_kpt_fft(spc, mesh),
                    spc_to_kpt(spc, mesh.phase),
                    rtol=0.0,
                    atol=2e-14,
                )

    def test_mapped_fft_matches_dense_phase_on_skew_lattice(self):
        cell = _make_cell(
            np.array([[2.0, 0.1, 0.0], [0.2, 2.4, 0.1], [0.0, 0.3, 2.8]])
        )
        mesh = canonicalize_kpts(
            cell, cell.make_kpts([2, 3, 2], wrap_around=False)
        )
        rng = np.random.default_rng(65)
        spc = rng.normal(size=(mesh.n_kpts, 2, 7))
        kpt = spc_to_kpt(spc, mesh.phase)
        np.testing.assert_allclose(
            kpt_to_spc_fft(kpt, mesh),
            kpt_to_spc(kpt, mesh.phase),
            rtol=0.0,
            atol=2e-14,
        )


if __name__ == "__main__":
    unittest.main()


class TestPairConvolveDevice(unittest.TestCase):
    """The device path must match the numpy path and keep the same gate.

    Equality is to a tolerance, not bit-exact: device reductions reorder
    summations. The time-reversal gate is asserted on BOTH paths so the
    device route cannot become the one that silently accepts bad input.
    """

    def _inputs(self, n_k=8, n_ip=64, n_ao=26, n_f=128):
        rng = np.random.default_rng(0)
        # real-valued complex128 => conj(X[k]) == X[k], satisfying the gate
        X = rng.standard_normal((n_k, n_ip, n_ao)).astype(np.complex128)
        Y = rng.standard_normal((n_k, n_f, n_ao)).astype(np.complex128)
        phase = np.eye(n_k, dtype=np.complex128)
        return X, Y, phase

    def test_matches_numpy_path(self):
        X, Y, phase = self._inputs()
        host = pair_convolve(X, Y, phase)
        device = pair_convolve_device(X, Y, phase)
        self.assertEqual(host.shape, device.shape)
        np.testing.assert_allclose(device, host, rtol=0.0, atol=1e-12)

    def test_time_reversal_gate_fires_on_both_paths(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((8, 32, 16)) + 1j * rng.standard_normal((8, 32, 16))
        Y = rng.standard_normal((8, 64, 16)) + 1j * rng.standard_normal((8, 64, 16))
        phase = np.eye(8, dtype=np.complex128)
        for fn in (pair_convolve, pair_convolve_device):
            with self.assertRaises(ValueError):
                fn(X, Y, phase)

    def test_rejects_malformed_shapes(self):
        X, Y, phase = self._inputs()
        with self.assertRaises(ValueError):
            pair_convolve_device(X[0], Y, phase)
        with self.assertRaises(ValueError):
            pair_convolve_device(X, Y, phase[:, :3])

    def test_mapped_fft_and_selected_q_match_dense_oracle(self):
        cell = _make_cell(a_diag=(2.0, 2.5, 3.0))
        rng = np.random.default_rng(4)
        for kmesh in ([1, 1, 3], [2, 2, 2], [2, 3, 4]):
            with self.subTest(kmesh=kmesh):
                mesh = canonicalize_kpts(
                    cell, cell.make_kpts(kmesh, wrap_around=False)
                )
                X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (3, 5))
                Y = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (4, 5))
                dense = pair_convolve(X, Y, mesh.phase)
                mapped = pair_convolve_fft_device(X, Y, mesh)
                np.testing.assert_allclose(mapped, dense, rtol=0.0, atol=2e-12)

                q_indices = np.array(
                    [mesh.n_kpts - 1, 0, mesh.n_kpts // 2], dtype=np.int64
                )
                selected = pair_convolve_q_device(X, Y, mesh, q_indices)
                np.testing.assert_allclose(
                    selected, dense[q_indices], rtol=0.0, atol=2e-12
                )

    def test_selected_q_rejects_bad_indices(self):
        cell = _make_cell()
        mesh = canonicalize_kpts(
            cell, cell.make_kpts([2, 2, 2], wrap_around=False)
        )
        rng = np.random.default_rng(5)
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (3, 5))
        Y = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (4, 5))
        for bad in ([], [mesh.n_kpts], [-1], [0.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pair_convolve_q_device(X, Y, mesh, np.asarray(bad))


class TestKTransformTiling(unittest.TestCase):
    """The k-transforms tile the flattened axis for cache locality. Tiling must
    not perturb a single bit, and the result must stay contiguous -- the version
    this replaced returned a strided `.real` view that made every downstream
    elementwise op read 16 bytes per 8 it used."""

    @staticmethod
    def _phase(n_k):
        w = np.exp(2j * np.pi * np.outer(np.arange(n_k), np.arange(n_k)) / n_k)
        return w / np.sqrt(n_k)

    @staticmethod
    def _whole_array_kpt_to_spc(m, phase):
        n_k = m.shape[0]
        return (phase @ m.reshape(n_k, -1)).reshape((n_k,) + m.shape[1:]).real

    @staticmethod
    def _whole_array_spc_to_kpt(m, phase):
        n_k = m.shape[0]
        out = (phase.conj().T @ m.reshape(n_k, -1)).reshape((n_k,) + m.shape[1:])
        return out.astype(np.complex128)

    def test_bit_identical_across_tile_boundaries(self):
        n_k = 8
        phase = self._phase(n_k)
        rng = np.random.default_rng(0)
        # Straddle the boundary in both directions and land exactly on it.
        for n_cols in (17, _SPC_TILE_COLS - 1, _SPC_TILE_COLS,
                       _SPC_TILE_COLS + 1, 2 * _SPC_TILE_COLS + 5):
            with self.subTest(n_cols=n_cols):
                spc = rng.standard_normal((n_k, 3, n_cols))
                # Round-trip from real image data so the k-space input is
                # time-reversal symmetric and clears kpt_to_spc's gate.
                kpt = self._whole_array_spc_to_kpt(spc, phase)
                np.testing.assert_array_equal(
                    kpt_to_spc(kpt, phase), self._whole_array_kpt_to_spc(kpt, phase))
                np.testing.assert_array_equal(
                    spc_to_kpt(spc, phase), self._whole_array_spc_to_kpt(spc, phase))

    def test_results_are_contiguous(self):
        n_k = 8
        phase = self._phase(n_k)
        spc = np.random.default_rng(1).standard_normal((n_k, 3, _SPC_TILE_COLS + 9))
        kpt = self._whole_array_spc_to_kpt(spc, phase)
        self.assertTrue(kpt_to_spc(kpt, phase).flags["C_CONTIGUOUS"])
        self.assertTrue(spc_to_kpt(spc, phase).flags["C_CONTIGUOUS"])

    def test_gate_still_fires_on_non_time_reversal_input(self):
        n_k = 8
        phase = self._phase(n_k)
        rng = np.random.default_rng(2)
        bad = (rng.standard_normal((n_k, 2, 40))
               + 1j * rng.standard_normal((n_k, 2, 40)))
        with self.assertRaises(ValueError):
            kpt_to_spc(bad, phase)
