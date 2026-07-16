"""CPU-JAX parity gates for the private periodic bare-K device core."""

import unittest
from unittest import mock

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb


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


def _tr_symmetric(rng, n_k, neg, shape):
    values = np.zeros((n_k,) + shape, dtype=np.complex128)
    for k in range(n_k):
        nk = int(neg[k])
        if nk < k:
            continue
        if nk == k:
            values[k] = rng.normal(size=shape)
        else:
            value = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            values[k] = value
            values[nk] = value.conj()
    return values


def _genuine_pair_fixture(seed=9):
    rng = np.random.default_rng(seed)
    cell = _make_cell()
    mesh = coulomb.canonicalize_kpts(cell, cell.make_kpts([1, 1, 4]))
    n_ip, n_ao = 3, 2
    inpv = _tr_symmetric(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
    coul_raw = _tr_symmetric(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ip))
    coul_kpt = coul_raw + coul_raw.conj().transpose(0, 2, 1)
    dm_raw = _tr_symmetric(rng, mesh.n_kpts, mesh.neg, (n_ao, n_ao))
    dm = dm_raw + dm_raw.conj().transpose(0, 2, 1)
    return dm, inpv, coul_kpt, mesh.phase, mesh.neg


class TestGetKBareDevice(unittest.TestCase):
    def test_gamma_single_and_batched_match_numpy_oracle(self):
        rng = np.random.default_rng(3)
        n_ao, n_ip = 3, 2
        x = rng.normal(size=(n_ip, n_ao)).astype(np.complex128)
        w = rng.normal(size=(n_ip, n_ip))
        w = ((w + w.T) / 2).astype(np.complex128)
        d = rng.normal(size=(n_ao, n_ao))
        d = ((d + d.T) / 2).astype(np.complex128)
        inpv, coul = x[None], w[None]
        phase = np.ones((1, 1), dtype=np.complex128)
        neg = np.array([0], dtype=np.int64)

        single = coulomb._get_k_bare_device(d[None], inpv, coul, phase, neg=neg)
        batch_dm = np.stack([d[None], (2 * d)[None]])
        batch = coulomb._get_k_bare_device(batch_dm, inpv, coul, phase, neg=neg)
        single_ref = coulomb.get_k(d[None], inpv, coul, phase, neg=neg)
        batch_ref = coulomb.get_k(batch_dm, inpv, coul, phase, neg=neg)

        np.testing.assert_allclose(np.asarray(single), single_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(batch), batch_ref, rtol=1e-12, atol=1e-12)

    def test_genuine_pair_matches_oracle_and_is_hermitian(self):
        dm, inpv, coul_kpt, phase, neg = _genuine_pair_fixture()
        vk = np.asarray(coulomb._get_k_bare_device(
            dm, inpv, coul_kpt, phase, neg=neg,
        ))
        vk_ref = coulomb.get_k(dm, inpv, coul_kpt, phase, neg=neg)

        np.testing.assert_allclose(vk, vk_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vk, vk.conj().transpose(0, 2, 1), rtol=1e-12, atol=1e-12)

    def test_per_set_gate_fails_closed(self):
        dm, inpv, _, _, neg = _genuine_pair_fixture()
        dm_sets = np.stack([np.zeros_like(dm), dm])
        phase = np.eye(neg.size, dtype=np.complex128)
        phase[1, 1] = 1j
        coul_kpt = np.zeros((neg.size, 3, 3), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, r"density sets \[1\]"):
            coulomb._get_k_bare_device(
                dm_sets, inpv, coul_kpt, phase, neg=neg,
            )

    def test_zero_norm_gate_is_zero(self):
        zeros = np.zeros((1, 2, 2), dtype=np.complex128)
        phase = np.ones((1, 1), dtype=np.complex128)
        rho, coul_spc, rho_ratio, coul_ratio = coulomb._get_k_bare_device_preflight(
            zeros[None], zeros, zeros, phase, np.array([0], dtype=np.int64),
        )
        self.assertEqual(float(np.asarray(rho_ratio)[0]), 0.0)
        self.assertEqual(float(np.asarray(coul_ratio)), 0.0)
        np.testing.assert_array_equal(np.asarray(rho), 0.0)
        np.testing.assert_array_equal(np.asarray(coul_spc), 0.0)

    def test_rejects_bad_shape_neg_dtype_and_x64(self):
        dm, inpv, coul_kpt, phase, neg = _genuine_pair_fixture()
        with self.assertRaises(ValueError):
            coulomb._get_k_bare_device(dm, inpv[:, :, :1], coul_kpt, phase, neg=neg)
        with self.assertRaises(ValueError):
            coulomb._get_k_bare_device(dm, inpv, coul_kpt, phase, neg=np.array([1, 1]))
        with self.assertRaises(ValueError):
            coulomb._get_k_bare_device(dm, inpv, coul_kpt, phase, neg=neg.astype(float))
        with self.assertRaises(ValueError):
            coulomb._get_k_bare_device(
                dm.astype(np.complex64), inpv, coul_kpt, phase, neg=neg,
            )
        with mock.patch.object(jax.config, "read", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "jax_enable_x64"):
                coulomb._get_k_bare_device(dm, inpv, coul_kpt, phase, neg=neg)


if __name__ == "__main__":
    unittest.main()
