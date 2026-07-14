"""Structural tests for pytc.pbc.coulomb.get_k/get_j (design v2.1
section 4/7).

get_k's absolute normalization is NOT independently confirmed here (see
pytc/pbc/coulomb.py's module docstring) -- these tests validate what CAN
be trusted without a full periodic FFTDF reference run: Hermiticity,
malformed-input rejection, exxdiv wiring, and an exact reduction at
Nk=1 to the standard ISDF exchange-matrix formula K = X^T (V (had) P) X
(P = X D X^T, the density projected onto interpolation points) --
derived independently from the ISDF literature, not from this module's
own code, and checked against a from-scratch NumPy computation.
"""

import unittest

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


class TestGetKGammaOnlyReduction(unittest.TestCase):
    def test_matches_independent_molecular_isdf_k_formula(self):
        # Real-valued fixtures throughout (Nk=1, real X/D/V) so
        # kpt_to_spc's imag_tol gate passes automatically and the
        # Gamma-only reduction is exact, not approximate.
        rng = np.random.default_rng(70)
        n_ao, n_ip = 5, 3
        X = rng.normal(size=(n_ip, n_ao))
        D_raw = rng.normal(size=(n_ao, n_ao))
        D = (D_raw + D_raw.T) / 2  # Hermitian (real symmetric) density matrix
        V_raw = rng.normal(size=(n_ip, n_ip))
        V = (V_raw + V_raw.T) / 2  # Hermitian (real symmetric) kernel

        inpv_kpt = X[None, :, :].astype(np.complex128)  # (1, Nip, Nao)
        coul_kpt = V[None, :, :].astype(np.complex128)  # (1, Nip, Nip)
        dm_kpts = D[None, :, :].astype(np.complex128)  # (1, Nao, Nao)

        K = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, kmesh=(1, 1, 1))

        P = X @ D @ X.T  # density projected onto interpolation points
        K_expected = X.T @ (V * P) @ X  # Hadamard product, then sandwich

        self.assertEqual(K.shape, (1, n_ao, n_ao))
        np.testing.assert_allclose(np.asarray(K[0]).real, K_expected, atol=1e-10)
        np.testing.assert_allclose(np.asarray(K[0]).imag, 0.0, atol=1e-10)

    def test_single_set_vs_multi_set_shape_convention(self):
        rng = np.random.default_rng(71)
        n_ao, n_ip = 4, 2
        inpv_kpt = rng.normal(size=(1, n_ip, n_ao)).astype(np.complex128)
        coul_kpt = rng.normal(size=(1, n_ip, n_ip)).astype(np.complex128)
        coul_kpt = coul_kpt + coul_kpt.conj().transpose(0, 2, 1)
        dm_single = rng.normal(size=(1, n_ao, n_ao)).astype(np.complex128)

        K_single = coulomb.get_k(dm_single, inpv_kpt, coul_kpt, kmesh=(1, 1, 1))
        self.assertEqual(K_single.shape, (1, n_ao, n_ao))

        K_multi = coulomb.get_k(dm_single[None], inpv_kpt, coul_kpt, kmesh=(1, 1, 1))
        self.assertEqual(K_multi.shape, (1, 1, n_ao, n_ao))
        np.testing.assert_allclose(K_multi[0], K_single, atol=0.0)


class TestGetKHermiticityAndValidation(unittest.TestCase):
    def test_vk_is_hermitian_per_k_for_hermitian_density(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        from pytc.pbc.df.kpts import canonicalize_kpts

        mesh_obj = canonicalize_kpts(cell, kpts)
        rng = np.random.default_rng(72)
        n_ip, n_ao = 3, cell.nao
        n_k = mesh_obj.n_kpts

        def _tr_symmetric_fixture(shape):
            arr = np.zeros((n_k,) + shape, dtype=np.complex128)
            done = set()
            for k in range(n_k):
                if k in done:
                    continue
                nk = int(mesh_obj.neg[k])
                if nk == k:
                    arr[k] = rng.normal(size=shape)
                else:
                    re, im = rng.normal(size=shape), rng.normal(size=shape)
                    arr[k] = re + 1j * im
                    arr[nk] = re - 1j * im
                    done.add(nk)
                done.add(k)
            return arr

        inpv_kpt = _tr_symmetric_fixture((n_ip, n_ao))
        coul_raw = _tr_symmetric_fixture((n_ip, n_ip))
        coul_kpt = coul_raw + coul_raw.conj().transpose(0, 2, 1)  # force Hermitian per k

        dm_raw = _tr_symmetric_fixture((n_ao, n_ao))
        dm_kpts = dm_raw + dm_raw.conj().transpose(0, 2, 1)  # Hermitian density per k

        K = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.kmesh)
        for k in range(n_k):
            Kk = np.asarray(K[k])
            np.testing.assert_allclose(Kk, Kk.conj().T, atol=1e-8, err_msg=f"k={k}")

    def test_rejects_malformed_shapes(self):
        rng = np.random.default_rng(73)
        inpv_kpt = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        coul_kpt = rng.normal(size=(2, 3, 3)).astype(np.complex128)
        dm_kpts = rng.normal(size=(2, 4, 4)).astype(np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt[:, :2, :2], kmesh=(1, 1, 2))
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts[:, :3, :3], inpv_kpt, coul_kpt, kmesh=(1, 1, 2))

    def test_rejects_bad_exxdiv(self):
        rng = np.random.default_rng(74)
        inpv_kpt = rng.normal(size=(1, 2, 3)).astype(np.complex128)
        coul_kpt = rng.normal(size=(1, 2, 2)).astype(np.complex128)
        dm_kpts = rng.normal(size=(1, 3, 3)).astype(np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, kmesh=(1, 1, 1), exxdiv="bogus")

    def test_exxdiv_ewald_requires_cell_and_kpts(self):
        rng = np.random.default_rng(75)
        inpv_kpt = rng.normal(size=(1, 2, 3)).astype(np.complex128)
        coul_kpt = rng.normal(size=(1, 2, 2)).astype(np.complex128)
        dm_kpts = rng.normal(size=(1, 3, 3)).astype(np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, kmesh=(1, 1, 1), exxdiv="ewald")

    def test_exxdiv_ewald_changes_the_result(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        from pytc.pbc.df.kpts import canonicalize_kpts

        mesh_obj = canonicalize_kpts(cell, kpts)
        rng = np.random.default_rng(76)
        n_ip, n_ao = 3, cell.nao
        n_k = mesh_obj.n_kpts

        def _tr_symmetric_fixture(shape):
            arr = np.zeros((n_k,) + shape, dtype=np.complex128)
            done = set()
            for k in range(n_k):
                if k in done:
                    continue
                nk = int(mesh_obj.neg[k])
                if nk == k:
                    arr[k] = rng.normal(size=shape)
                else:
                    re, im = rng.normal(size=shape), rng.normal(size=shape)
                    arr[k] = re + 1j * im
                    arr[nk] = re - 1j * im
                    done.add(nk)
                done.add(k)
            return arr

        inpv_kpt = _tr_symmetric_fixture((n_ip, n_ao))
        coul_raw = _tr_symmetric_fixture((n_ip, n_ip))
        coul_kpt = coul_raw + coul_raw.conj().transpose(0, 2, 1)
        dm_raw = _tr_symmetric_fixture((n_ao, n_ao))
        dm_kpts = dm_raw + dm_raw.conj().transpose(0, 2, 1)

        K_bare = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.kmesh)
        K_ewald = coulomb.get_k(
            dm_kpts, inpv_kpt, coul_kpt, mesh_obj.kmesh,
            exxdiv="ewald", cell=cell, kpts=mesh_obj.canonical_kpts,
        )
        self.assertGreater(np.abs(np.asarray(K_bare) - np.asarray(K_ewald)).max(), 0.0)


class TestGetJ(unittest.TestCase):
    def test_matches_direct_pyscf_fftdf_call(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        rng = np.random.default_rng(77)
        n_ao = cell.nao
        dm_raw = rng.normal(size=(2, n_ao, n_ao)) + 1j * rng.normal(size=(2, n_ao, n_ao))
        dm_kpts = (dm_raw + dm_raw.conj().transpose(0, 2, 1)).astype(np.complex128)

        vj = coulomb.get_j(cell, dm_kpts, kpts)

        from pyscf.pbc.df import FFTDF
        from pyscf.pbc.df.fft_jk import get_j_kpts

        vj_direct = get_j_kpts(FFTDF(cell), dm_kpts, kpts=kpts)
        np.testing.assert_allclose(np.asarray(vj), np.asarray(vj_direct), atol=0.0)
        self.assertEqual(np.asarray(vj).shape, (2, n_ao, n_ao))


if __name__ == "__main__":
    unittest.main()
