"""Tests for pytc.utils.cache_state.

Covers:
    * cache_has_mf_state detects the presence of cached orbital state.
    * save_orbital_state_to_cache + sync_mf_from_cache round-trip arrays
      bit-exactly.
    * ISDFXTC.from_xtc auto-saves mo_coeff / mo_occ on the first build.
    * sync_mf_from_cache is a no-op when the cache does not yet exist.
    * check_mo_coeff_matches_cache returns True for identical values and
      False (with a warning) for a different gauge.
    * sync_mf_from_cache refuses to sync when the cached mo_coeff has a
      different AO size than the current mol.
"""

import os
import tempfile
import unittest

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf

from pytc.jastrow.rexp import REXP
from pytc.xtc import XTC, ISDFXTC
from pytc.utils.cache_state import (
    cache_has_mf_state,
    check_mo_coeff_matches_cache,
    mo_coeff_fingerprint,
    prepare_mf,
    save_orbital_state_to_cache,
    sync_mf_from_cache,
)


def _tiny_mf():
    mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="321g", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


class TestCacheStateRoundTrip(unittest.TestCase):
    def setUp(self):
        self.mf = _tiny_mf()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.tmp.name, "cache.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def test_has_state_false_when_missing(self):
        self.assertFalse(cache_has_mf_state(self.cache_path))
        self.assertFalse(cache_has_mf_state(None))
        self.assertFalse(cache_has_mf_state(""))

    def test_save_then_has_state_true(self):
        save_orbital_state_to_cache(
            self.cache_path,
            mo_coeff=self.mf.mo_coeff,
            mo_energy=self.mf.mo_energy,
            mo_occ=self.mf.mo_occ,
            e_tot=self.mf.e_tot,
        )
        self.assertTrue(cache_has_mf_state(self.cache_path))

    def test_sync_round_trip_is_bit_exact(self):
        save_orbital_state_to_cache(
            self.cache_path,
            mo_coeff=self.mf.mo_coeff,
            mo_energy=self.mf.mo_energy,
            mo_occ=self.mf.mo_occ,
            e_tot=self.mf.e_tot,
        )
        # Fresh mf to verify sync overwrites.
        fresh = scf.RHF(self.mf.mol)
        fresh.mo_coeff = None
        fresh.mo_occ = None
        fresh.mo_energy = None
        fresh.e_tot = None
        fresh.converged = False

        returned = sync_mf_from_cache(fresh, self.cache_path)
        self.assertIs(returned, fresh, "sync should return the same mf object")
        np.testing.assert_array_equal(fresh.mo_coeff, self.mf.mo_coeff)
        np.testing.assert_array_equal(fresh.mo_occ, self.mf.mo_occ)
        np.testing.assert_array_equal(fresh.mo_energy, self.mf.mo_energy)
        self.assertEqual(float(fresh.e_tot), float(self.mf.e_tot))
        self.assertTrue(fresh.converged)

    def test_sync_noop_when_missing(self):
        """No cached state -> mf returned unchanged, no error."""
        fresh = scf.RHF(self.mf.mol)
        fresh.mo_coeff = np.zeros_like(self.mf.mo_coeff)
        returned = sync_mf_from_cache(fresh, self.cache_path)
        self.assertIs(returned, fresh)
        # mo_coeff must still be the zeros we set — sync did nothing.
        np.testing.assert_array_equal(fresh.mo_coeff, np.zeros_like(self.mf.mo_coeff))


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_encodes_shape_sum_norm(self):
        mo = np.arange(12, dtype=np.float64).reshape(3, 4)
        fp = mo_coeff_fingerprint(mo)
        self.assertIn("shape=(3, 4)", fp)
        self.assertIn("sum=", fp)
        self.assertIn("norm=", fp)


class TestCheckMoCoeffMatches(unittest.TestCase):
    def setUp(self):
        self.mf = _tiny_mf()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.tmp.name, "cache.h5")
        save_orbital_state_to_cache(
            self.cache_path,
            mo_coeff=self.mf.mo_coeff,
            mo_occ=self.mf.mo_occ,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_match_returns_true(self):
        self.assertTrue(
            check_mo_coeff_matches_cache(self.mf.mo_coeff, self.cache_path)
        )

    def test_sign_flip_detected(self):
        flipped = np.asarray(self.mf.mo_coeff).copy()
        flipped[:, 0] *= -1.0  # classic SCF gauge ambiguity
        self.assertFalse(
            check_mo_coeff_matches_cache(flipped, self.cache_path)
        )

    def test_shape_mismatch_detected(self):
        other = np.asarray(self.mf.mo_coeff)[:, :-1]
        self.assertFalse(
            check_mo_coeff_matches_cache(other, self.cache_path)
        )


class TestSyncRejectsWrongBasis(unittest.TestCase):
    def test_basis_mismatch_returns_mf_unchanged(self):
        """A cache saved from basis A must not silently overwrite basis B."""
        mol_a = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="321g", verbose=0)
        mf_a = scf.RHF(mol_a)
        mf_a.kernel()

        mol_b = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="sto-3g", verbose=0)
        mf_b = scf.RHF(mol_b)
        mf_b.kernel()
        original_mo_b = np.asarray(mf_b.mo_coeff).copy()

        tmp = tempfile.TemporaryDirectory()
        try:
            cache_path = os.path.join(tmp.name, "cache.h5")
            # Save 321g state.
            save_orbital_state_to_cache(
                cache_path,
                mo_coeff=mf_a.mo_coeff,
                mo_occ=mf_a.mo_occ,
            )
            # Try to sync it into the sto-3g mf — should be a no-op.
            sync_mf_from_cache(mf_b, cache_path)
            np.testing.assert_array_equal(mf_b.mo_coeff, original_mo_b)
        finally:
            tmp.cleanup()


class TestISDFXTCAutoSavesMoCoeff(unittest.TestCase):
    """ISDFXTC.from_xtc should persist mo_coeff to the cache on the first build."""

    def test_first_from_xtc_writes_mo_coeff(self):
        mf = _tiny_mf()
        jastrow = REXP()
        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            n_rank = 4 * mf.mo_coeff.shape[1]

            self.assertFalse(cache_has_mf_state(cache_path))
            _ = ISDFXTC.from_xtc(xtc, n_rank=n_rank, save_path=cache_path)
            self.assertTrue(
                cache_has_mf_state(cache_path),
                "ISDFXTC.from_xtc should auto-save mo_coeff to the cache.",
            )
            # And round-tripping recovers the same mo_coeff.
            fresh = scf.RHF(mf.mol)
            fresh.mo_coeff = None
            fresh.mo_occ = None
            sync_mf_from_cache(fresh, cache_path)
            np.testing.assert_array_equal(fresh.mo_coeff, np.asarray(mf.mo_coeff))


class TestPrepareMf(unittest.TestCase):
    """prepare_mf runs SCF when no cache, adopts cached state when present."""

    def test_first_call_runs_kernel_second_call_syncs(self):
        mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="321g", verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")

            # First call: no cache yet. prepare_mf should run kernel().
            mf1 = scf.RHF(mol)
            self.assertIsNone(mf1.mo_coeff)
            mf1 = prepare_mf(mf1, cache_path)
            self.assertIsNotNone(mf1.mo_coeff)
            self.assertTrue(mf1.converged)

            # Manually persist state so the second call can sync it.
            save_orbital_state_to_cache(
                cache_path,
                mo_coeff=mf1.mo_coeff,
                mo_occ=mf1.mo_occ,
                mo_energy=mf1.mo_energy,
                e_tot=mf1.e_tot,
            )

            # Second call: cache present. prepare_mf should skip kernel().
            mf2 = scf.RHF(mol)
            self.assertIsNone(mf2.mo_coeff)
            mf2 = prepare_mf(mf2, cache_path)
            np.testing.assert_array_equal(mf2.mo_coeff, np.asarray(mf1.mo_coeff))
            self.assertTrue(mf2.converged)


if __name__ == "__main__":
    unittest.main()
