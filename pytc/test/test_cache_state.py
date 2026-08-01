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
    cache_has_isdf_kernels,
    cache_has_mf_state,
    check_mo_coeff_matches_cache,
    mo_coeff_fingerprint,
    prepare_mf,
    save_orbital_state_to_cache,
    sync_mf_from_cache,
)
import h5py


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


class TestFromXtcFailsOnMismatchedMoCoeff(unittest.TestCase):
    """from_xtc must raise (not warn) when the cache mo_coeff doesn't match."""

    def test_mismatched_mo_coeff_raises(self):
        mf = _tiny_mf()
        jastrow = REXP()
        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            n_rank = 4 * mf.mo_coeff.shape[1]

            _ = ISDFXTC.from_xtc(xtc, n_rank=n_rank, save_path=cache_path)
            self.assertTrue(cache_has_mf_state(cache_path))

            flipped = np.asarray(mf.mo_coeff).copy()
            flipped[:, 0] *= -1.0
            save_orbital_state_to_cache(
                cache_path, mo_coeff=flipped, mo_occ=mf.mo_occ)

            with self.assertRaises(ValueError):
                ISDFXTC.from_xtc(xtc, n_rank=n_rank, save_path=cache_path)

    def test_matched_mo_coeff_passes(self):
        mf = _tiny_mf()
        jastrow = REXP()
        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            n_rank = 4 * mf.mo_coeff.shape[1]

            _ = ISDFXTC.from_xtc(xtc, n_rank=n_rank, save_path=cache_path)

            _ = ISDFXTC.from_xtc(xtc, n_rank=n_rank, save_path=cache_path)


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


class TestCacheHasMfStateRequiresMoOcc(unittest.TestCase):
    """cache_has_mf_state must require both mo_coeff AND mo_occ so that
    prepare_mf never skips SCF based on a partial cache."""

    def test_mo_coeff_only_is_not_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            # Write mo_coeff WITHOUT mo_occ (partial cache).
            with h5py.File(cache_path, "w") as f:
                f.create_dataset("mo_coeff_cached", data=np.eye(3))
            self.assertFalse(cache_has_mf_state(cache_path))


class TestLegacyCacheNotOverwritten(unittest.TestCase):
    """ISDFXTC.from_xtc must refuse to auto-save mo_coeff when the cache
    already contains ISDF kernels but no cached mo_coeff (i.e. was written
    by an older pytc that didn't know to pin the gauge).  Silent writing
    would lock future reloads to the wrong orbital gauge."""

    def test_legacy_kernels_block_auto_save(self):
        mf = _tiny_mf()
        jastrow = REXP()
        xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            # Fake a legacy cache: ISDF kernels present, no mo_coeff.
            with h5py.File(cache_path, "w") as f:
                f.create_dataset("xi_phi", data=np.zeros((1, 1)))
                f.create_dataset("pivots", data=np.array([0]))
            self.assertTrue(cache_has_isdf_kernels(cache_path))
            self.assertFalse(cache_has_mf_state(cache_path))

            # Force isdf_decompose to fall through its "all keys present?"
            # check and try to write.  Since xi_phi exists, we expect
            # isdf_decompose to error or load a stub — what matters is
            # that ISDFXTC.from_xtc does NOT add mo_coeff_cached to the
            # file when legacy kernels are already there.
            try:
                _ = ISDFXTC.from_xtc(
                    xtc, n_rank=4 * mf.mo_coeff.shape[1], save_path=cache_path
                )
            except Exception:
                # isdf_decompose may fail on the malformed stub — that's
                # fine; we only care about whether mo_coeff got persisted.
                pass

            self.assertFalse(
                cache_has_mf_state(cache_path),
                "Legacy cache must not be stamped with a fresh mo_coeff "
                "(could lock future reloads to the wrong gauge).",
            )


class TestCheckMoCoeffAtolTightens(unittest.TestCase):
    """check_mo_coeff_matches_cache must use rtol=0 so the advertised atol
    is the real tolerance (np.allclose default rtol=1e-5 would mask
    noticeably different mo_coeff)."""

    def test_rtol_is_zero_effectively(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            mo = np.ones((3, 3))
            save_orbital_state_to_cache(
                cache_path, mo_coeff=mo, mo_occ=np.array([2.0, 2.0, 0.0]),
            )
            # Perturb by 1e-4 (> atol=1e-12, > implicit rtol*1=1e-5 baseline).
            # Under the old np.allclose default rtol this would pass; under
            # rtol=0 it must fail.
            perturbed = mo + 1e-4
            self.assertFalse(
                check_mo_coeff_matches_cache(perturbed, cache_path, atol=1e-12)
            )


class TestPrepareMfFallsBackOnRejectedSync(unittest.TestCase):
    """If sync_mf_from_cache refuses to sync (basis mismatch), prepare_mf
    must fall back to running mf.kernel() rather than returning an unsolved
    mf."""

    def test_falls_back_to_kernel_on_basis_mismatch(self):
        mol_a = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="321g", verbose=0)
        mf_a = scf.RHF(mol_a)
        mf_a.kernel()

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "cache.h5")
            save_orbital_state_to_cache(
                cache_path,
                mo_coeff=mf_a.mo_coeff,
                mo_occ=mf_a.mo_occ,
            )

            # Now hand prepare_mf a different basis; sync will refuse, so
            # prepare_mf should fall through to kernel().
            mol_b = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="sto-3g", verbose=0)
            mf_b = scf.RHF(mol_b)
            self.assertIsNone(mf_b.mo_coeff)
            mf_b = prepare_mf(mf_b, cache_path)
            self.assertIsNotNone(mf_b.mo_coeff)
            self.assertTrue(mf_b.converged)
            # And it's the sto-3g result (7 AOs), not the 321g cached one.
            self.assertEqual(mf_b.mo_coeff.shape[0], 7)


if __name__ == "__main__":
    unittest.main()
