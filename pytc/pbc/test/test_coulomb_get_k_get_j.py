"""Structural tests for pytc.pbc.coulomb.get_k/get_j (design v2.1
section 4/7).

Nk=1: exact reduction to the standard ISDF exchange-matrix formula
K = X^T (V (had) P) X (P = X D X^T, the density projected onto
interpolation points) -- derived independently from the ISDF
literature, not from this module's own code, checked against a
from-scratch NumPy computation. Nk>1: TestGetKVsRealFftdf below adds
rank-matched-parity (not exactness) checks against a real periodic
FFTDF get_k_kpts, for both exxdiv=None and exxdiv="ewald" -- ranks are
kept in the "sane regime" (see class docstring) since the retained-
mode solve-residual retention policy is a separate, not-yet-tuned knob
(design v2.1 section 5) that blows up at overcomplete rank on these
tiny test cells, independent of the get_k formula itself.
"""

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb

_PHASE_GAMMA_ONLY = np.array([[1.0 + 0.0j]])  # Nk=1: R=[0,0,0], k=Gamma, exp(i*0)/sqrt(1)=1


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

        K = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, _PHASE_GAMMA_ONLY)

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

        K_single = coulomb.get_k(dm_single, inpv_kpt, coul_kpt, _PHASE_GAMMA_ONLY)
        self.assertEqual(K_single.shape, (1, n_ao, n_ao))

        K_multi = coulomb.get_k(dm_single[None], inpv_kpt, coul_kpt, _PHASE_GAMMA_ONLY)
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

        K = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase)
        for k in range(n_k):
            Kk = np.asarray(K[k])
            np.testing.assert_allclose(Kk, Kk.conj().T, atol=1e-8, err_msg=f"k={k}")

    def test_rejects_malformed_shapes(self):
        rng = np.random.default_rng(73)
        inpv_kpt = rng.normal(size=(2, 3, 4)).astype(np.complex128)
        coul_kpt = rng.normal(size=(2, 3, 3)).astype(np.complex128)
        dm_kpts = rng.normal(size=(2, 4, 4)).astype(np.complex128)
        phase2 = np.eye(2, dtype=np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt[:, :2, :2], phase2)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts[:, :3, :3], inpv_kpt, coul_kpt, phase2)

    def test_rejects_bad_exxdiv(self):
        rng = np.random.default_rng(74)
        inpv_kpt = rng.normal(size=(1, 2, 3)).astype(np.complex128)
        coul_kpt = rng.normal(size=(1, 2, 2)).astype(np.complex128)
        dm_kpts = rng.normal(size=(1, 3, 3)).astype(np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, _PHASE_GAMMA_ONLY, exxdiv="bogus")

    def test_exxdiv_ewald_requires_cell_and_kpts(self):
        rng = np.random.default_rng(75)
        inpv_kpt = rng.normal(size=(1, 2, 3)).astype(np.complex128)
        coul_kpt = rng.normal(size=(1, 2, 2)).astype(np.complex128)
        dm_kpts = rng.normal(size=(1, 3, 3)).astype(np.complex128)
        with self.assertRaises(ValueError):
            coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, _PHASE_GAMMA_ONLY, exxdiv="ewald")

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

        K_bare = coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase)
        K_ewald = coulomb.get_k(
            dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase,
            exxdiv="ewald", cell=cell, kpts=mesh_obj.canonical_kpts,
        )
        self.assertGreater(np.abs(np.asarray(K_bare) - np.asarray(K_ewald)).max(), 0.0)


def _tr_symmetric_hermitian_fixture(rng, n_kpts, neg, shape):
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


class TestGetKVsRealFftdf(unittest.TestCase):
    """Rank-matched-parity (design v2.1 section 8, V3) checks against a
    real periodic FFTDF get_k_kpts on he_cubic_cell [1,1,3]. Rank kept
    at 6 (nao=5, within the "sane regime" nip ~ 2-3x nao per the D1 fix
    session's finding) -- past this the retained-mode solve residual
    retention policy (a separate, not-yet-tuned knob, design v2.1
    section 5) blows up on this tiny system independent of get_k's own
    formula, which is what this test isolates. Bounds are generous
    sanity/regression bounds, not accuracy claims -- ISDF at this rank
    is far from converged (see V0's own cisdf sweep in the task #20
    baseline), the point is confirming get_k stays in the same ballpark
    as a real reference, not exact agreement."""

    def _build_and_dm(self, kmesh, rank, seed):
        cell = _make_cell()
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        result = coulomb.build(cell, kpts, rank=rank, block_size=100, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        rng = np.random.default_rng(seed)
        dm_kpts = _tr_symmetric_hermitian_fixture(
            rng, mesh_obj.n_kpts, mesh_obj.neg, (cell.nao, cell.nao)
        )
        return cell, kpts, result, mesh_obj, dm_kpts

    def test_rank_matched_parity_exxdiv_none(self):
        cell, kpts, result, mesh_obj, dm_kpts = self._build_and_dm([1, 1, 3], rank=6, seed=42)
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])

        vk_mine = np.asarray(coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase))

        from pyscf.pbc.df import FFTDF
        from pyscf.pbc.df.fft_jk import get_k_kpts

        vk_ref = get_k_kpts(FFTDF(cell), dm_kpts, kpts=kpts, exxdiv=None)
        rel = np.linalg.norm(vk_mine - vk_ref) / np.linalg.norm(vk_ref)
        self.assertLess(rel, 1.5)

    def test_rank_matched_parity_exxdiv_ewald(self):
        cell, kpts, result, mesh_obj, dm_kpts = self._build_and_dm([1, 1, 3], rank=6, seed=42)
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])

        vk_mine = np.asarray(
            coulomb.get_k(
                dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase,
                exxdiv="ewald", cell=cell, kpts=mesh_obj.canonical_kpts,
            )
        )

        from pyscf.pbc.df import FFTDF
        from pyscf.pbc.df.fft_jk import get_k_kpts

        vk_ref = get_k_kpts(FFTDF(cell), dm_kpts, kpts=kpts, exxdiv="ewald")
        rel = np.linalg.norm(vk_mine - vk_ref) / np.linalg.norm(vk_ref)
        self.assertLess(rel, 2.5)


class TestRetentionPolicyDefaultAvoidsBlowup(unittest.TestCase):
    """Regression test for the retention-policy fix (design v2.1 section
    5): the old rtol=1e-8 default silently retained near-singular Pi
    modes at over-complete rank, whose inverse then amplified V's noise
    and blew up K (observed relF > 6 on this exact cell at rank=8 with
    the old default, and outright ValueError from kpt_to_spc's imag_tol
    gate at higher rank). rtol=1e-4 (the new default) must complete
    without error and stay in a bounded relF regime at ranks that used
    to blow up."""

    def test_rank_that_used_to_blow_up_now_stays_bounded(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        n_ao = cell.nao
        result = coulomb.build(cell, kpts, rank=10, block_size=100)  # default rtol
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])

        rng = np.random.default_rng(42)
        dm_kpts = _tr_symmetric_hermitian_fixture(
            rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ao, n_ao)
        )
        vk_mine = np.asarray(coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase))

        from pyscf.pbc.df import FFTDF
        from pyscf.pbc.df.fft_jk import get_k_kpts

        vk_ref = get_k_kpts(FFTDF(cell), dm_kpts, kpts=kpts, exxdiv=None)
        rel = np.linalg.norm(vk_mine - vk_ref) / np.linalg.norm(vk_ref)
        # The old default produced relF=39 (a genuine blow-up) at this
        # exact rank; bounding well below that, not claiming accuracy.
        self.assertLess(rel, 2.0)

    def test_explicit_old_default_still_blows_up_documenting_why_it_changed(self):
        # Not a contradiction with the fix -- this documents the ORIGINAL
        # failure mode still reproduces when a caller explicitly asks
        # for the old rtol, proving the new default is what changed the
        # outcome (not some other unrelated change).
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        n_ao = cell.nao
        result = coulomb.build(cell, kpts, rank=10, block_size=100, rtol=1e-8)
        mesh_obj = result["mesh_obj"]
        inpv_kpt = np.asarray(result["inpv_kpt"])
        coul_kpt = np.asarray(result["coul_kpt"])

        rng = np.random.default_rng(42)
        dm_kpts = _tr_symmetric_hermitian_fixture(
            rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ao, n_ao)
        )
        vk_mine = np.asarray(coulomb.get_k(dm_kpts, inpv_kpt, coul_kpt, mesh_obj.phase))

        from pyscf.pbc.df import FFTDF
        from pyscf.pbc.df.fft_jk import get_k_kpts

        vk_ref = get_k_kpts(FFTDF(cell), dm_kpts, kpts=kpts, exxdiv=None)
        rel = np.linalg.norm(vk_mine - vk_ref) / np.linalg.norm(vk_ref)
        self.assertGreater(rel, 5.0)


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
