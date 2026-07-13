"""Tests for pytc.pbc.df.isdf.build_pi_eta (task #21, design v2.1
sections 4/5)."""

import unittest

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import build_pi_eta
from pytc.pbc.df.kpts import canonicalize_kpts, pair_convolve


def _make_cell():
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag([2.0, 2.0, 2.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
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


class TestBuildPiEta(unittest.TestCase):
    def test_pi_matches_direct_pair_convolve(self):
        cell = _make_cell()
        rng = np.random.default_rng(50)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao, n_g = 3, 5, 7
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_g, n_ao))

        Pi, eta = build_pi_eta(X, ao, mesh.kmesh)
        Pi_expected = pair_convolve(X, X, mesh.kmesh)
        eta_expected = pair_convolve(X, ao, mesh.kmesh)
        np.testing.assert_allclose(Pi, Pi_expected, atol=1e-12)
        np.testing.assert_allclose(eta, eta_expected, atol=1e-12)

    def test_pi_is_hermitian_per_q(self):
        cell = _make_cell()
        rng = np.random.default_rng(51)
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao = 4, 6
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (5, n_ao))
        Pi, _ = build_pi_eta(X, ao, mesh.kmesh)
        for q in range(mesh.n_kpts):
            np.testing.assert_allclose(Pi[q], Pi[q].conj().T, atol=1e-10, err_msg=f"q={q}")

    def test_pi_neg_q_conj_closure(self):
        cell = _make_cell()
        rng = np.random.default_rng(52)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao = 3, 4
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (5, n_ao))
        Pi, eta = build_pi_eta(X, ao, mesh.kmesh)
        np.testing.assert_allclose(Pi[mesh.neg], Pi.conj(), atol=1e-10)
        np.testing.assert_allclose(eta[mesh.neg], eta.conj(), atol=1e-10)

    def test_grid_block_accumulation_matches_single_block(self):
        cell = _make_cell()
        rng = np.random.default_rng(53)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao, n_g = 3, 4, 9
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_g, n_ao))

        Pi_single, eta_single = build_pi_eta(X, ao, mesh.kmesh)
        blocks = [ao[:, :4, :], ao[:, 4:7, :], ao[:, 7:, :]]
        Pi_blocked, eta_blocked = build_pi_eta(X, blocks, mesh.kmesh)

        np.testing.assert_allclose(Pi_blocked, Pi_single, atol=1e-12)
        np.testing.assert_allclose(eta_blocked, eta_single, atol=1e-12)

    def test_rejects_malformed_x_shape(self):
        cell = _make_cell()
        rng = np.random.default_rng(54)
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        X_bad = rng.normal(size=(mesh.n_kpts, 3)).astype(np.complex128)
        ao = rng.normal(size=(mesh.n_kpts, 5, 4)).astype(np.complex128)
        with self.assertRaises(ValueError):
            build_pi_eta(X_bad, ao, mesh.kmesh)

    def test_rejects_empty_ao_blocks(self):
        cell = _make_cell()
        rng = np.random.default_rng(55)
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (3, 4))
        with self.assertRaises(ValueError):
            build_pi_eta(X, [], mesh.kmesh)


if __name__ == "__main__":
    unittest.main()
