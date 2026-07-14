"""Tests for pytc.pbc.coulomb.ISDFDF (design v2.1 section 8, V4 gate):
the thin pyscf-compatible with_df adapter that lets a real, unmodified
pyscf KRHF/KRKS SCF loop consume this module's build()/get_k/get_j.

Real pyscf KRHF runs (not synthetic get_jk-interface mocks) since the
whole point of this adapter is exercising OUR integrals inside PYSCF'S
UNMODIFIED SCF machinery -- a synthetic interface test would not catch
wiring bugs against pyscf's actual get_hcore/get_veff call chain (e.g.
the with_df.get_pp/get_nuc delegation this class needed once mf.get_hcore
was actually exercised).
"""

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF

from pytc.pbc.coulomb import ISDFDF


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


class TestISDFDFStructure(unittest.TestCase):
    def test_get_jk_rejects_omega(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(cell, kpts, rank=4, block_size=100)
        dm = np.zeros((2, cell.nao, cell.nao), dtype=np.complex128)
        with self.assertRaises(NotImplementedError):
            adapter.get_jk(dm, kpts=kpts, omega=0.3)

    def test_get_jk_rejects_kpts_band(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(cell, kpts, rank=4, block_size=100)
        dm = np.zeros((2, cell.nao, cell.nao), dtype=np.complex128)
        with self.assertRaises(NotImplementedError):
            adapter.get_jk(dm, kpts=kpts, kpts_band=kpts[:1])

    def test_get_jk_rejects_mismatched_kpts(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(cell, kpts, rank=4, block_size=100)
        dm = np.zeros((2, cell.nao, cell.nao), dtype=np.complex128)
        other_kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        with self.assertRaises(ValueError):
            adapter.get_jk(dm, kpts=other_kpts)

    def test_build_is_memoized(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(cell, kpts, rank=4, block_size=100)
        result1 = adapter.build()
        result2 = adapter.build()
        self.assertIs(result1, result2)

    def test_get_pp_delegates_to_real_fftdf(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(cell, kpts, rank=4, block_size=100)

        from pyscf.pbc.df import FFTDF

        expected = FFTDF(cell, kpts).get_pp(kpts)
        got = adapter.get_pp(kpts)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=0.0)


class TestISDFDFRealKrhf(unittest.TestCase):
    """Plumbing-correctness tests: a real pyscf KRHF run through the
    adapter must converge and track the real-FFTDF energy monotonically
    as rank increases (the same rank-matched-parity pattern as every
    other gate in this design, not exactness at toy rank)."""

    def test_converges_and_energy_improves_with_rank(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)

        mf_ref = KRHF(cell, kpts)
        mf_ref.verbose = 0
        e_ref = mf_ref.kernel()

        errors = []
        for rank in (6, 14):
            mf = KRHF(cell, kpts)
            mf.verbose = 0
            mf.with_df = ISDFDF(cell, kpts, rank=rank, block_size=100, rtol=1e-4)
            e = mf.kernel()
            self.assertTrue(mf.converged, msg=f"rank={rank} did not converge")
            errors.append(abs(e - e_ref) / cell.natm)

        self.assertLess(errors[1], errors[0])

    def test_exxdiv_ewald_is_the_pyscf_default_and_wires_through(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mf = KRHF(cell, kpts)
        mf.verbose = 0
        self.assertEqual(mf.exxdiv, "ewald")  # pyscf's own KSCF default
        mf.with_df = ISDFDF(cell, kpts, rank=6, block_size=100, rtol=1e-4)
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(np.isfinite(e))


if __name__ == "__main__":
    unittest.main()
