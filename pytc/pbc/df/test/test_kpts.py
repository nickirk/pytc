"""V1 oracle tests for pytc.pbc.df.kpts (task #21, design v2.1 section 4/8).
"""

import unittest

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.kpts import (
    KptsMesh,
    canonicalize_kpts,
    check_time_reversal_residual,
    pair_convolve,
)


def _make_cell(a_diag=(2.0, 2.0, 2.0)):
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag(a_diag)
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
    """Z[q] = (1/Nk) sum_k T[k] * T[(q-k) mod mesh], with (q-k) mod mesh
    meaning per-AXIS modular subtraction on the (n1,n2,n3) lattice -- NOT
    flat-index arithmetic mod Nk, which only coincides with the correct
    per-axis result for a mesh with a single nontrivial axis (any mesh
    used in a 1D-reduced test here). Verified this distinction matters by
    direct failure on a [2,2,2] mesh before this fix."""
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
                Z_mesh[q1, q2, q3] = acc / n_k
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
            neg=mesh.neg, ktol=mesh.ktol,
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
        rng = np.random.default_rng(10)
        n_ip, n_f, n_ao = 3, 4, 5
        X = rng.normal(size=(1, n_ip, n_ao)).astype(np.complex128)
        Y = rng.normal(size=(1, n_f, n_ao)).astype(np.complex128)
        Z = pair_convolve(X, Y, (1, 1, 1))
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
            Z = pair_convolve(X, Y, mesh.kmesh)
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
        Z = pair_convolve(X, Y, mesh.kmesh)
        np.testing.assert_allclose(Z[mesh.neg], Z.conj(), atol=1e-10)

    def test_delta_field_analytic_case(self):
        # X/Y are nonzero only at Gamma (a "delta field" in k-space) -> the
        # real-space T_R is CONSTANT across all Nk supercell images
        # (T_R[r] = T[0]/Nk, standard IFFT-of-a-delta identity, backward
        # norm), so Z_R = T_R**2 is also constant = (T[0]/Nk)**2, and
        # Z[q] = FFT(constant) is nonzero ONLY at q=Gamma (index 0): the
        # unnormalized forward-FFT of a length-Nk constant array c sums
        # Nk copies of c at q=0 and cancels (DFT orthogonality) elsewhere,
        # giving Z[0] = Nk * (T[0]/Nk)**2 = T[0]**2 / Nk.
        n_ip, n_f, n_ao = 2, 3, 4
        n_k = 4
        rng = np.random.default_rng(14)
        X = np.zeros((n_k, n_ip, n_ao), dtype=np.complex128)
        Y = np.zeros((n_k, n_f, n_ao), dtype=np.complex128)
        X[0] = rng.normal(size=(n_ip, n_ao))
        Y[0] = rng.normal(size=(n_f, n_ao))
        Z = pair_convolve(X, Y, (1, 1, n_k))
        T0 = X[0] @ Y[0].conj().T
        expected_gamma = (T0 ** 2) / n_k
        np.testing.assert_allclose(Z[0], expected_gamma, atol=1e-10)
        for q in range(1, n_k):
            np.testing.assert_allclose(Z[q], 0.0, atol=1e-10)

    def test_rejects_malformed_shapes_and_kmesh(self):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        Y = rng.normal(size=(2, 5, 4)).astype(np.complex128)
        with self.assertRaises(ValueError):
            pair_convolve(X, Y, (1, 1, 3))  # prod != Nk
        with self.assertRaises(ValueError):
            pair_convolve(X, rng.normal(size=(2, 5, 6)).astype(np.complex128), (1, 1, 2))
        with self.assertRaises(ValueError):
            pair_convolve(X[:, :0], Y, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
