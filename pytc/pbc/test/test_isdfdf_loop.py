"""ISDFDF.loop: the PySCF three-index DF contract, served from THC factors.

pytc/df/thc.py's extract_vv_df_factor refuses any with_df lacking loop(), and
pytc/solver/isdf_xtc_ccsd.py reaches it through that function, so this adapter
is what connects the periodic Coulomb provider to the existing xTC solver.

The equivalence test is the load-bearing one: loop() must reproduce the SAME
ERIs the provider's own factors define, or every downstream number is quietly
wrong rather than loudly broken.
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf import lib
from pyscf.pbc import gto

from pytc.pbc.coulomb import ISDFDF


def _cell(basis="gth-szv", ke=40.0):
    cell = gto.Cell()
    cell.atom = "C 0 0 0; C 0.8917 0.8917 0.8917"
    cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
    cell.unit = "A"
    cell.basis = basis
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = ke
    cell.verbose = 0
    cell.build()
    return cell


def _gamma_df(cell, **kw):
    nao = cell.nao_nr()
    kw.setdefault("rank", 6 * nao)
    kw.setdefault("block_size", 128)
    kw.setdefault("solve_backend", "host")
    df = ISDFDF(cell, cell.make_kpts([1, 1, 1]), **kw)
    df.build()
    return df


class TestReproducesTheProvidersOwnEris(unittest.TestCase):
    """loop() is only useful if it agrees with the factorization it came from."""

    @classmethod
    def setUpClass(cls):
        cls.cell = _cell()
        cls.df = _gamma_df(cls.cell)
        built = cls.df.build()
        cls.X = np.asarray(built["inpv_kpt"][0]).real
        cls.V = np.asarray(built["coul_kpt"][0]).real

    def _eri_from_loop(self):
        L = np.vstack(list(self.df.loop()))
        return L.T @ L

    def _eri_from_factors(self):
        # (pq|rs) = M^T V M with M[mu,pq] = X[mu,p] X[mu,q]; the definition the
        # provider's own ao2mo path contracts.
        M = lib.pack_tril(np.einsum("mp,mq->mpq", self.X, self.X))
        return M.T @ self.V @ M

    def test_matches_the_thc_contraction(self):
        got, want = self._eri_from_loop(), self._eri_from_factors()
        denom = np.linalg.norm(want)
        self.assertGreater(denom, 0.0)
        self.assertLess(np.linalg.norm(got - want) / denom, 1e-11)

    def test_direction_not_just_magnitude(self):
        # A norm alone cannot distinguish "same answer" from "same size"; the
        # cosine is what catches a factor that is scaled or rotated.
        got, want = self._eri_from_loop().ravel(), self._eri_from_factors().ravel()
        cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
        self.assertAlmostEqual(cos, 1.0, places=12)

    def test_eri_is_symmetric(self):
        eri = self._eri_from_loop()
        np.testing.assert_allclose(eri, eri.T, rtol=0, atol=1e-12 * np.abs(eri).max())


class TestPyscfContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _gamma_df(_cell())

    def test_blocks_are_2d_float64_and_nao_pair_wide(self):
        nao = self.df.cell.nao_nr()
        nao_pair = nao * (nao + 1) // 2
        for blk in self.df.loop():
            self.assertEqual(blk.ndim, 2)
            self.assertEqual(blk.shape[1], nao_pair)
            self.assertEqual(blk.dtype, np.float64)
            self.assertTrue(blk.flags["C_CONTIGUOUS"])

    def test_get_naoaux_matches_what_is_yielded(self):
        # Callers preallocate from get_naoaux() and fill from loop(); if these
        # disagree the failure is a silent truncation, so they share one cached
        # square root by construction.
        rows = sum(blk.shape[0] for blk in self.df.loop())
        self.assertEqual(rows, self.df.get_naoaux())

    def test_blksize_splits_without_changing_the_result(self):
        whole = np.vstack(list(self.df.loop()))
        split = np.vstack(list(self.df.loop(blksize=3)))
        self.assertGreater(len(list(self.df.loop(blksize=3))), 1)
        np.testing.assert_allclose(whole, split, rtol=0, atol=0)

    def test_rejects_nonpositive_blksize(self):
        for bad in (0, -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                list(self.df.loop(blksize=bad))

    def test_loop_cost_reports_the_streamed_size(self):
        cost = self.df.loop_cost()
        nao = self.df.cell.nao_nr()
        self.assertEqual(cost["nao"], nao)
        self.assertEqual(cost["nao_pair"], nao * (nao + 1) // 2)
        self.assertEqual(cost["naux"], self.df.get_naoaux())
        self.assertAlmostEqual(
            cost["stream_bytes"], 8.0 * cost["naux"] * cost["nao_pair"])


class TestGammaOnly(unittest.TestCase):
    def test_kpoints_refused_with_a_reason(self):
        cell = _cell()
        nao = cell.nao_nr()
        df = ISDFDF(cell, cell.make_kpts([2, 1, 1]), rank=6 * nao,
                    block_size=128, solve_backend="host")
        with self.assertRaises(NotImplementedError) as ctx:
            list(df.loop())
        # The refusal must say WHY, because the fix is a contract change (a real
        # square root of complex factors does not exist), not a missing branch.
        self.assertIn("Gamma-only", str(ctx.exception))


class TestConsumerIntegration(unittest.TestCase):
    """The reason this method exists at all."""

    def test_extract_vv_df_factor_accepts_it(self):
        from pyscf.pbc import scf
        from pytc.df import thc

        cell = _cell()
        mf = scf.RHF(cell).density_fit()
        mf.kernel()
        mo = np.asarray(mf.mo_coeff)
        nocc = int((mf.mo_occ > 0).sum())

        df = _gamma_df(cell)
        b = thc.extract_vv_df_factor(df, mo, nocc)

        nvir = mo.shape[1] - nocc
        self.assertEqual(b.shape[:2], (nvir, nvir))
        self.assertEqual(b.shape[2], df.get_naoaux())
        self.assertTrue(np.isfinite(b).all())


if __name__ == "__main__":
    unittest.main()
