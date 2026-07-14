"""Tests for pytc.pbc.coulomb.get_ao_eri/get_mo_eri (design doc §2).
Primary gate: reconstructing get_k's K from ERI blocks is an exact
algebraic identity, rank-independent."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb
from pytc.pbc.df.kpts import build_kconserv


def _make_cell():
    cell = Cell()
    cell.atom = "He 0.0 0.0 0.0; He 1.0 1.0 1.0"
    cell.a = np.diag([3.0, 3.0, 3.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 20.0
    cell.build()
    return cell


def _tr_symmetric_hermitian(rng, n_kpts, neg, shape):
    arr = np.zeros((n_kpts,) + shape, dtype=np.complex128)
    done = set()
    for k in range(n_kpts):
        if k in done:
            continue
        nk = int(neg[k])
        if nk == k:
            h = rng.normal(size=shape)
            arr[k] = (h + h.T) / 2
        else:
            re, im = rng.normal(size=shape), rng.normal(size=shape)
            h = re + 1j * im
            arr[k] = (h + h.conj().T) / 2
            arr[nk] = arr[k].conj()
        done.add(k)
        done.add(nk)
    return arr


class TestGetAoEriMatchesGetK(unittest.TestCase):
    """Primary gate: rank-independent algebraic identity vs get_k."""

    def _check_at_rank(self, rank):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=rank, block_size=200, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])
        n_k = mesh_obj.n_kpts
        n_ao = cell.nao
        kconserv = build_kconserv(cell, mesh_obj.canonical_kpts)

        rng = np.random.default_rng(5 + rank)
        dm_kpts = _tr_symmetric_hermitian(rng, n_k, mesh_obj.neg, (n_ao, n_ao))
        vk_ref = np.asarray(coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase))

        K = np.zeros((n_k, n_ao, n_ao), dtype=np.complex128)
        for k1 in range(n_k):
            for k1d in range(n_k):
                # standard-convention exchange-diagonal case: querying
                # (a=q@k1d, b=r@k1 | c=p@k1, d=s@k1d) reduces to
                # k3=k1, k4=k1d.
                eri_block, k4 = coulomb.get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1d, k1, k1)
                self.assertEqual(k4, k1d)
                D = dm_kpts[k1d]  # D_sq
                K[k1] += np.einsum("qrps,sq->pr", eri_block, D, optimize=True)
        K = K / n_k

        rel = np.abs(K - vk_ref).max() / np.abs(vk_ref).max()
        self.assertLess(rel, 1e-10, msg=f"rank={rank}: rel={rel:.3e}")

    def test_matches_get_k_at_low_rank(self):
        self._check_at_rank(6)

    def test_matches_get_k_at_toy_rank(self):
        self._check_at_rank(15)

    def test_matches_get_k_at_overcomplete_rank(self):
        self._check_at_rank(25)


class TestGetAoEriStructure(unittest.TestCase):
    def test_shape_and_momentum_conservation_for_general_triples(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=10, block_size=200, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])
        n_ao = cell.nao
        n_k = mesh_obj.n_kpts
        kconserv = build_kconserv(cell, mesh_obj.canonical_kpts)

        for k1 in range(n_k):
            for k2 in range(n_k):
                for k3 in range(n_k):
                    eri_block, k4 = coulomb.get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3)
                    self.assertEqual(eri_block.shape, (n_ao, n_ao, n_ao, n_ao))
                    self.assertTrue(0 <= k4 < n_k)
                    # standard pyscf momentum relation: k1 - k2 + k3 - k4 = 0 (mod G)
                    diff = (
                        mesh_obj.canonical_kpts[k1]
                        - mesh_obj.canonical_kpts[k2]
                        + mesh_obj.canonical_kpts[k3]
                        - mesh_obj.canonical_kpts[k4]
                    )
                    scaled = cell.get_scaled_kpts(diff[None])[0]
                    resid = np.linalg.norm(scaled - np.round(scaled))
                    self.assertLess(resid, 1e-8, msg=f"(k1,k2,k3,k4)=({k1},{k2},{k3},{k4})")

    def test_rejects_malformed_shapes(self):
        rng = np.random.default_rng(80)
        inpv_kpt = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        coul_kpt = rng.normal(size=(2, 3, 3)).astype(np.complex128)
        kconserv = np.zeros((2, 2, 2), dtype=np.int64)
        with self.assertRaises(ValueError):
            coulomb.get_ao_eri(inpv_kpt, coul_kpt[:, :2, :2], kconserv, 0, 0, 0)
        with self.assertRaises(ValueError):
            coulomb.get_ao_eri(inpv_kpt, coul_kpt, np.zeros((3, 3, 3), dtype=np.int64), 0, 0, 0)

    def test_rejects_out_of_range_k_index(self):
        rng = np.random.default_rng(81)
        inpv_kpt = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        coul_kpt = rng.normal(size=(2, 3, 3)).astype(np.complex128)
        kconserv = np.zeros((2, 2, 2), dtype=np.int64)
        with self.assertRaises(ValueError):
            coulomb.get_ao_eri(inpv_kpt, coul_kpt, kconserv, 5, 0, 0)


class TestGetAoEriVsExactFftdf(unittest.TestCase):
    """Secondary gate: rank-matched parity vs real FFTDF exact ERI,
    following the same "compare at matched rank, not a fixed 1e-6" rule
    established for get_k (Flinn's D1 finding 2: at nip~1.5x nao the
    ISDF rank error is large but comparable between implementations,
    not a bug)."""

    def test_rank_matched_parity_vs_fftdf_exact_eri(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=15, block_size=200, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])
        n_ao = cell.nao
        kconserv = build_kconserv(cell, mesh_obj.canonical_kpts)

        from pyscf.pbc.df import FFTDF

        fftdf = FFTDF(cell)
        k1, k2, k3 = 0, 1, 2
        eri_mine, k4 = coulomb.get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3)
        # Standard pyscf convention throughout -- no axis/index relabeling.
        kpts_for_eri = [mesh_obj.canonical_kpts[i] for i in (k1, k2, k3, k4)]
        eri_exact = np.asarray(
            fftdf.get_eri(kpts_for_eri, compact=False)
        ).reshape(n_ao, n_ao, n_ao, n_ao)

        rel = np.linalg.norm(eri_mine - eri_exact) / np.linalg.norm(eri_exact)
        # Rank-matched parity band, not a fixed tight tolerance -- at
        # nip=15 (~1.5x nao) both pytc and an external ISDF reference
        # show O(1) relF vs exact (see get_k's own rank=15 findings);
        # this asserts pytc is not egregiously worse than that band.
        self.assertLess(rel, 1.5)


class TestGetMoEri(unittest.TestCase):
    def test_matches_manual_ao_to_mo_transform(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=10, block_size=200, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])
        n_ao = cell.nao
        kconserv = build_kconserv(cell, mesh_obj.canonical_kpts)

        rng = np.random.default_rng(90)
        n_k = mesh_obj.n_kpts
        mo_coeffs = [
            (rng.normal(size=(n_ao, n_ao)) + 1j * rng.normal(size=(n_ao, n_ao))).astype(
                np.complex128
            )
            for _ in range(n_k)
        ]

        k1, k2, k3 = 0, 1, 2
        eri_ao, k4 = coulomb.get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3)
        mo_coeff_kpts = (mo_coeffs[k1], mo_coeffs[k2], mo_coeffs[k3], mo_coeffs[k4])
        eri_mo, k4_mo = coulomb.get_mo_eri(
            inpv_kpt, coul_kpt, kconserv, mo_coeff_kpts, k1, k2, k3
        )
        self.assertEqual(k4_mo, k4)

        eri_mo_manual = np.einsum(
            "abcd,ai,bj,ck,dl->ijkl",
            eri_ao,
            mo_coeff_kpts[0],
            mo_coeff_kpts[1],
            mo_coeff_kpts[2],
            mo_coeff_kpts[3],
            optimize=True,
        )
        np.testing.assert_allclose(eri_mo, eri_mo_manual, atol=1e-10)

    def test_rejects_wrong_number_of_mo_coeff_sets(self):
        rng = np.random.default_rng(91)
        inpv_kpt = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        coul_kpt = rng.normal(size=(2, 3, 3)).astype(np.complex128)
        kconserv = np.zeros((2, 2, 2), dtype=np.int64)
        C = np.eye(4, dtype=np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_mo_eri(inpv_kpt, coul_kpt, kconserv, (C, C, C), 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
