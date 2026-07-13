"""Tests for pytc.integrals.coulomb's build_core orchestration section
(task #8 follow-up, isdf-coulomb-cuda, 2026-07-12), TWO review rounds:

Round 1: build_sector/build_core must reproduce exactly what the
hand-rolled pivot-selection -> pair_collocation_at_pivots ->
compute_C_streamed -> compute_Z/compute_Z_cross pipeline produces
(bit-exact), while additionally assembling the full §4 provenance
record (pivot/grid/mo_coeff hashes, distinct rank fields, kernel
policy, upstream provenance).

Round 2 (Alice's re-review of the round-1 implementation, 3 real
findings, all covered here):
1. numerical_rank was reported as an exact value even when
   pivoted_cholesky_pair_pivots only established a LOWER BOUND (a
   capped prefix where every candidate remained "effective" proves
   rank >= n_rank_capped, not ==) -- independently reproduced with
   random 3x12/3x12 factors at requested_rank=2 (reported
   numerical_rank=2, true rank 9).
2. Cross-sector joins didn't validate that the two SectorFits came
   from the same molecule/basis/auxbasis/kernel-policy/pyscf-version
   identity -- same n_aux was silently accepted as sufficient.
3. SectorFit/CoreArtifact were only shallow-frozen (top-level
   MappingProxyType, mutable arrays) -- mutating a SectorFit's
   provenance or arrays after build_core already used it retroactively
   altered the built CoreArtifact.

Round 3 (Alice's re-review of round-2's compatibility_key/_deep_freeze
fixes, 2 more real findings):
4. compatibility_key fingerprinted the REQUESTED auxbasis via
   pyscf.df.addons.make_auxmol -- but stream_df_cderi_blocks silently
   REUSES mf.with_df whenever present, ignoring the requested auxbasis
   entirely in that case. Independently reproduced: an mf with a
   pre-existing with_df streamed bit-identical blocks under two
   different (both-ignored) requested auxbasis strings, yet got
   DIFFERENT keys -- a false rejection. Fixed by hashing the ACTUAL
   streamed DF block bytes (compute_C_streamed's df_factor_sha256)
   instead of an auxmol fingerprint of the requested label.
5. _deep_freeze never actually froze numpy arrays nested inside e.g.
   upstream_provenance (only top-level P/C/pivots went through
   _readonly_copy) -- mutating a caller's array nested in
   upstream_provenance after build leaked into the built artifact.
"""

import copy
import types
import unittest

import numpy as np
from pyscf import df as pyscf_df
from pyscf import gto, scf

from pytc.integrals.coulomb import (
    get_mo_coeff,
    get_grid_ao_values_and_weights,
    weight_mo_values,
    select_sector_pivots,
    pair_collocation_at_pivots,
    compute_C_streamed,
    compute_Z,
    compute_Z_cross,
    build_sector,
    build_core,
    SectorFit,
    CoreArtifact,
)


class TestBuildCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24",
            basis="cc-pvdz", verbose=0)
        cls.mf = scf.RHF(cls.mol).density_fit().run()
        cls.n_occ = cls.mol.nelectron // 2
        mo_coeff = get_mo_coeff(cls.mf)
        cls.mo_occ = mo_coeff[:, :cls.n_occ]
        cls.mo_vir = mo_coeff[:, cls.n_occ:]

        cls.ao_values, cls.weights, cls.coords = get_grid_ao_values_and_weights(cls.mf, grid_lvl=2)
        mo_values = (cls.ao_values @ mo_coeff).T
        cls.occ_raw = mo_values[:cls.n_occ]
        cls.vir_raw = mo_values[cls.n_occ:]
        mo_weighted = weight_mo_values(mo_values, cls.weights)
        cls.occ_weighted = mo_weighted[:cls.n_occ]
        cls.vir_weighted = mo_weighted[cls.n_occ:]

        cls.n_rank_ov = 300
        cls.n_rank_oo = 75

        # A second, DIFFERENT molecule/basis for cross-join rejection tests.
        cls.mol2 = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        cls.mf2 = scf.RHF(cls.mol2).density_fit().run()
        mo_coeff2 = get_mo_coeff(cls.mf2)
        ao_values2, weights2, coords2 = get_grid_ao_values_and_weights(cls.mf2, grid_lvl=2)
        mo_values2 = (ao_values2 @ mo_coeff2).T
        cls.mo2 = mo_coeff2
        cls.factor2_raw = mo_values2
        cls.coords2 = coords2
        cls.weights2 = weights2

        # A THIRD mf, same molecule/basis as mf2 but NOT density-fit up
        # front (no with_df yet) -- used to isolate "same molecule, same
        # nominal auxbasis argument, different ACTUAL with_df" from
        # "different molecule entirely" (Alice's round-4 coverage ask).
        cls.mf3 = scf.RHF(cls.mol2).run()
        ao_values3, weights3, coords3 = get_grid_ao_values_and_weights(cls.mf3, grid_lvl=2)
        mo_coeff3 = get_mo_coeff(cls.mf3)
        mo_values3 = (ao_values3 @ mo_coeff3).T
        cls.mo3 = mo_coeff3
        cls.factor3_raw = mo_values3
        cls.coords3 = coords3
        cls.weights3 = weights3

    def _reference_ov_pipeline(self):
        pivots = np.asarray(select_sector_pivots(
            self.occ_weighted, self.vir_weighted, self.n_rank_ov))
        occ_at_piv = np.asarray(self.occ_raw)[:, pivots]
        vir_at_piv = np.asarray(self.vir_raw)[:, pivots]
        P = pair_collocation_at_pivots(occ_at_piv, vir_at_piv)
        C = compute_C_streamed(self.mf, P, self.mo_occ, self.mo_vir, auxbasis="weigend")
        return pivots, P, C

    def _build_ov_sector(self):
        return build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov)

    def _build_oo_sector(self):
        return build_sector(
            self.mf, self.occ_raw, self.occ_raw, self.mo_occ, self.mo_occ,
            self.coords, self.weights, self.n_rank_oo, same_factor=True)

    # ---- Round 1: bit-exact pipeline equivalence ----

    def test_build_sector_matches_hand_rolled_pipeline_bit_exact(self):
        ref_pivots, ref_P, ref_C = self._reference_ov_pipeline()
        sector = self._build_ov_sector()
        self.assertIsInstance(sector, SectorFit)
        np.testing.assert_array_equal(np.asarray(sector.pivots), ref_pivots)
        np.testing.assert_allclose(sector.P, ref_P, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(sector.C, ref_C, atol=0.0, rtol=0.0)

    def test_build_core_same_sector_matches_compute_Z_directly(self):
        sector = self._build_ov_sector()
        core = build_core(sector)
        self.assertIsInstance(core, CoreArtifact)
        Z_direct, prov_direct = compute_Z(sector.P, sector.C)
        np.testing.assert_allclose(core.Z, Z_direct, atol=1e-12)
        self.assertEqual(core.provenance["solver"], prov_direct["solver"])
        self.assertEqual(core.provenance["jitter_used"], prov_direct["jitter_used"])
        self.assertIn("sector", core.provenance)

    def test_build_core_cross_sector_matches_compute_Z_cross_directly(self):
        sector_oo = self._build_oo_sector()
        sector_ov = self._build_ov_sector()

        core = build_core(sector_oo, sector_ov)
        Z_direct, prov_direct = compute_Z_cross(
            sector_oo.P, sector_oo.C, sector_ov.P, sector_ov.C, same_sector=False)
        np.testing.assert_allclose(core.Z, Z_direct, atol=1e-12)
        self.assertEqual(core.provenance["jitter_used"], prov_direct["jitter_used"])
        self.assertIn("left_sector", core.provenance)
        self.assertIn("right_sector", core.provenance)

    def test_provenance_hashes_present_and_reproducible(self):
        sector_a = self._build_ov_sector()
        sector_b = self._build_ov_sector()
        self.assertEqual(sector_a.provenance["pivot_indices_sha256"],
                          sector_b.provenance["pivot_indices_sha256"])
        self.assertEqual(sector_a.provenance["grid_sha256"], sector_b.provenance["grid_sha256"])
        self.assertEqual(sector_a.provenance["mo_coeff_p_sha256"],
                          sector_b.provenance["mo_coeff_p_sha256"])
        self.assertEqual(sector_a.provenance["factor_p_raw_sha256"],
                          sector_b.provenance["factor_p_raw_sha256"])

        sector_oo = self._build_oo_sector()
        self.assertNotEqual(sector_a.provenance["pivot_indices_sha256"],
                             sector_oo.provenance["pivot_indices_sha256"])
        self.assertEqual(sector_a.provenance["grid_sha256"], sector_oo.provenance["grid_sha256"])
        self.assertNotEqual(sector_a.provenance["mo_coeff_p_sha256"],
                             sector_a.provenance["mo_coeff_q_sha256"])

    def test_upstream_provenance_passthrough(self):
        upstream = {"pyscf_version": "2.10.0", "gpu4pyscf_version": None, "grid_lvl": 2}
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, upstream_provenance=upstream)
        self.assertEqual(dict(sector.provenance["upstream_provenance"]), upstream)

    def test_upstream_provenance_defaults_to_empty_dict(self):
        sector = self._build_ov_sector()
        self.assertEqual(dict(sector.provenance["upstream_provenance"]), {})

    def test_unsupported_kernel_policy_rejected(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_ov,
                kernel_policy="MolecularFreeSpacePoisson")

    def test_kernel_policy_recorded_in_provenance(self):
        sector = self._build_ov_sector()
        self.assertEqual(sector.provenance["kernel_policy"], "MolecularDFReference")
        self.assertEqual(sector.provenance["kernel_policy_params"]["auxbasis"], "weigend")

    # ---- Round 2, finding 1: honest rank semantics ----

    def test_numerical_rank_is_none_when_not_exhausted(self):
        # requested_rank=2 for the ov sector's true rank (94) -- the
        # capped selection never runs out of "effective" pivots within
        # only 2 candidates, so numerical_rank must NOT claim an exact
        # value (Alice's independent repro: capped-but-not-exhausted
        # runs previously reported a false exact rank).
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, requested_rank=2)
        prov = sector.provenance
        self.assertFalse(prov["rank_exhausted"])
        self.assertIsNone(prov["numerical_rank"])
        self.assertEqual(prov["numerical_rank_lower_bound"], prov["n_rank_capped"])
        self.assertEqual(prov["n_pivots"], prov["n_rank_capped"])

    def test_numerical_rank_is_exact_when_exhausted(self):
        # n_rank_oo=75 requested >> oo's true triangular-number rank
        # (15 for H2O/cc-pVDZ, n_occ=5) -- the analytic cap alone brings
        # n_rank_capped to 15, and the numerical selection may or may
        # not further truncate below that; whichever happens,
        # rank_exhausted's value must be internally consistent with
        # numerical_rank/numerical_rank_lower_bound.
        sector = self._build_oo_sector()
        prov = sector.provenance
        self.assertEqual(prov["requested_rank"], self.n_rank_oo)
        self.assertLess(prov["analytic_rank_bound"], self.n_rank_oo)
        if prov["rank_exhausted"]:
            self.assertIsNotNone(prov["numerical_rank"])
            self.assertEqual(prov["numerical_rank"], prov["numerical_rank_lower_bound"])
            self.assertEqual(prov["numerical_rank"], prov["n_pivots"])
        else:
            self.assertIsNone(prov["numerical_rank"])
            self.assertEqual(prov["numerical_rank_lower_bound"], prov["n_rank_capped"])

    # ---- Round 2, finding 2: cross-join compatibility validation ----

    def test_cross_join_rejects_mismatched_system(self):
        sector_ov = self._build_ov_sector()
        # A SectorFit built from a totally different molecule/basis,
        # requesting the SAME nominal auxbasis string -- proving the
        # rejection is driven by the actually-streamed factor, not a
        # trivial string mismatch (round-3 finding).
        sector_other = build_sector(
            self.mf2, self.factor2_raw, self.factor2_raw, self.mo2, self.mo2,
            self.coords2, self.weights2, requested_rank=1, same_factor=True,
            auxbasis="weigend")
        self.assertNotEqual(sector_ov.compatibility_key, sector_other.compatibility_key)
        with self.assertRaises(ValueError):
            build_core(sector_ov, sector_other)

    # ---- Round 3, finding 4: compatibility_key tracks the ACTUAL
    # streamed factor, not the (possibly-ignored) requested auxbasis ----

    def test_compatibility_key_invariant_to_ignored_auxbasis_when_with_df_reused(self):
        # self.mf already has with_df set (from .density_fit() in
        # setUpClass), so stream_df_cderi_blocks silently IGNORES
        # whatever auxbasis is requested here -- both calls must
        # stream the identical actual factor and therefore get the
        # SAME compatibility_key/df_factor_sha256, despite different
        # (both-ignored) requested auxbasis strings.
        self.assertIsNotNone(self.mf.with_df)
        sector_a = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, auxbasis="weigend")
        sector_b = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, auxbasis="def2-svp-jkfit")
        self.assertEqual(sector_a.provenance["df_factor_sha256"],
                          sector_b.provenance["df_factor_sha256"])
        self.assertEqual(sector_a.compatibility_key, sector_b.compatibility_key)
        self.assertTrue(sector_a.provenance["reused_existing_with_df"])
        self.assertEqual(sector_a.provenance["requested_auxbasis"], "weigend")
        self.assertEqual(sector_b.provenance["requested_auxbasis"], "def2-svp-jkfit")
        self.assertEqual(sector_a.provenance["effective_auxbasis"],
                          sector_b.provenance["effective_auxbasis"])

    def test_compatibility_key_invariant_to_blksize(self):
        sector_a = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, blksize=None)
        sector_b = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, blksize=8)
        self.assertEqual(sector_a.provenance["df_factor_sha256"],
                          sector_b.provenance["df_factor_sha256"])
        self.assertEqual(sector_a.compatibility_key, sector_b.compatibility_key)
        np.testing.assert_allclose(sector_a.C, sector_b.C)

    def test_compatibility_key_isolates_same_molecule_different_actual_factor(self):
        # Alice's round-4 coverage ask: test_cross_join_rejects_mismatched_
        # system (above) conflates "different molecule" with the narrower
        # claim this module actually makes. Isolate it: SAME molecule
        # (self.mol2/self.mf3), SAME nominal (ignored) auxbasis argument
        # passed to build_sector both times, but two genuinely DIFFERENT
        # pre-built with_df objects swapped onto mf3.with_df between
        # calls -- must still produce different keys and a rejected join.
        with_df_a = pyscf_df.df.DF(self.mol2, auxbasis="weigend")
        with_df_a.build()
        self.mf3.with_df = with_df_a
        sector_a = build_sector(
            self.mf3, self.factor3_raw, self.factor3_raw, self.mo3, self.mo3,
            self.coords3, self.weights3, requested_rank=1, same_factor=True,
            auxbasis="def2-svp-jkfit")  # ignored -- with_df_a already set

        with_df_b = pyscf_df.df.DF(self.mol2, auxbasis="def2-svp-jkfit")
        with_df_b.build()
        self.mf3.with_df = with_df_b
        sector_b = build_sector(
            self.mf3, self.factor3_raw, self.factor3_raw, self.mo3, self.mo3,
            self.coords3, self.weights3, requested_rank=1, same_factor=True,
            auxbasis="def2-svp-jkfit")  # same nominal argument as sector_a, still ignored

        self.assertTrue(sector_a.provenance["reused_existing_with_df"])
        self.assertTrue(sector_b.provenance["reused_existing_with_df"])
        self.assertEqual(sector_a.provenance["requested_auxbasis"],
                          sector_b.provenance["requested_auxbasis"])
        self.assertNotEqual(sector_a.provenance["df_factor_sha256"],
                             sector_b.provenance["df_factor_sha256"])
        self.assertNotEqual(sector_a.compatibility_key, sector_b.compatibility_key)
        with self.assertRaises(ValueError):
            build_core(sector_a, sector_b)

    def test_cross_join_accepts_matching_system(self):
        sector_oo = self._build_oo_sector()
        sector_ov = self._build_ov_sector()
        self.assertEqual(sector_oo.compatibility_key, sector_ov.compatibility_key)
        core = build_core(sector_oo, sector_ov)  # must not raise
        self.assertIsInstance(core, CoreArtifact)

    def test_compatibility_key_recorded_in_provenance(self):
        sector = self._build_ov_sector()
        self.assertEqual(sector.provenance["compatibility_key"], sector.compatibility_key)

    # ---- Round 2, finding 3: genuine immutability ----

    def test_provenance_top_level_is_read_only(self):
        sector = self._build_ov_sector()
        self.assertIsInstance(sector.provenance, types.MappingProxyType)
        with self.assertRaises(TypeError):
            sector.provenance["mutated_after_build"] = True

    def test_provenance_nested_dict_is_read_only(self):
        # A shallow top-level MappingProxyType still allows nested-key
        # mutation (sector.provenance['upstream_provenance']['x']=...)
        # unless nested structures are ALSO frozen -- exactly Alice's
        # repro.
        upstream = {"pyscf_version": "2.10.0"}
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, upstream_provenance=upstream)
        nested = sector.provenance["upstream_provenance"]
        self.assertIsInstance(nested, types.MappingProxyType)
        with self.assertRaises(TypeError):
            nested["mutated_after_build"] = True

    def test_kernel_policy_params_nested_dict_is_read_only(self):
        sector = self._build_ov_sector()
        nested = sector.provenance["kernel_policy_params"]
        self.assertIsInstance(nested, types.MappingProxyType)
        with self.assertRaises(TypeError):
            nested["auxbasis"] = "mutated"

    # ---- Round 3, finding 5: arrays nested inside provenance are frozen ----

    def test_nested_array_in_upstream_provenance_is_read_only(self):
        x = np.array([1.0, 2.0, 3.0])
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov,
            upstream_provenance={"some_array": x})
        frozen_array = sector.provenance["upstream_provenance"]["some_array"]
        self.assertIsInstance(frozen_array, np.ndarray)
        self.assertFalse(frozen_array.flags.writeable)
        with self.assertRaises(ValueError):
            frozen_array[0] = 0.0

    def test_mutating_callers_nested_array_after_build_does_not_alter_artifact(self):
        # Alice's exact repro: x = np.array([1.0]); frozen = _deep_freeze
        # ({'nested_array': x}); x[0] = 9.0 must NOT change
        # frozen['nested_array'][0] -- the frozen copy must be
        # independent of the caller's original array, not merely a
        # read-only VIEW of the same underlying buffer.
        x = np.array([1.0])
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov,
            upstream_provenance={"nested_array": x})
        x[0] = 9.0
        np.testing.assert_array_equal(
            sector.provenance["upstream_provenance"]["nested_array"], [1.0])

    def test_deep_freeze_rejects_unknown_mutable_object(self):
        # Alice's round-4 repro: a plain custom object with mutable
        # state must NOT be silently passed through unchanged -- the
        # provenance schema is CLOSED (Mapping/list/tuple/set/frozenset/
        # ndarray/immutable scalars only); anything else raises
        # TypeError rather than staying aliased and mutable.
        class Box:
            def __init__(self):
                self.value = 1

        with self.assertRaises(TypeError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_ov,
                upstream_provenance={"box": Box()})

    # ---- Round 5, findings 1-3: closing the remaining schema edges ----

    def test_deep_freeze_rejects_non_str_mapping_key(self):
        # Alice's round-5 repro: a mutable-but-hashable custom key
        # object stays aliased if only VALUES are frozen -- provenance
        # keys must be str.
        class Key:
            def __init__(self):
                self.value = 1

        with self.assertRaises(TypeError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_ov,
                upstream_provenance={"nested": {Key(): 1}})

    def test_deep_freeze_rejects_object_dtype_array(self):
        # Alice's round-5 repro: setflags(write=False) only blocks
        # reassigning array ELEMENTS, not mutating the arbitrary Python
        # objects an object-dtype array's elements reference.
        class Box:
            def __init__(self):
                self.value = 1

        obj_array = np.array([Box(), Box()], dtype=object)
        with self.assertRaises(TypeError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_ov,
                upstream_provenance={"boxes": obj_array})

    def test_deep_freeze_recurses_through_numpy_scalar_item(self):
        # np.generic.item() must be fed back through _deep_freeze, not
        # returned directly -- a structured/object numpy scalar can
        # .item() into something that itself still needs validation.
        # Plain numeric scalars are the common case and must still work
        # (a regression on this would break e.g. numerical_rank fields
        # derived from numpy computations upstream).
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov,
            upstream_provenance={"numpy_float": np.float64(3.5), "numpy_int": np.int64(7)})
        self.assertEqual(sector.provenance["upstream_provenance"]["numpy_float"], 3.5)
        self.assertIsInstance(sector.provenance["upstream_provenance"]["numpy_float"], float)
        self.assertEqual(sector.provenance["upstream_provenance"]["numpy_int"], 7)
        self.assertIsInstance(sector.provenance["upstream_provenance"]["numpy_int"], int)

    def test_mutating_caller_upstream_dict_after_build_does_not_alter_artifact(self):
        # build_sector must not alias the caller's own dict either --
        # mutating the ORIGINAL dict passed in after the call must not
        # retroactively change the (already frozen-copied) artifact.
        upstream = {"pyscf_version": "2.10.0"}
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov, upstream_provenance=upstream)
        upstream["mutated_after_build"] = True
        self.assertNotIn("mutated_after_build", sector.provenance["upstream_provenance"])

    def test_sector_arrays_are_read_only(self):
        sector = self._build_ov_sector()
        for arr in (sector.P, sector.C, sector.pivots):
            self.assertFalse(arr.flags.writeable)
            with self.assertRaises(ValueError):
                arr[0] = 0

    def test_readonly_copy_does_not_mutate_callers_array(self):
        # Passing the caller's own array in must not mark THAT array
        # read-only as a side effect -- construction must always copy
        # first (Alice's review: np.asarray on an already-ndarray input
        # can return the SAME object, so setflags without copying first
        # would leak the read-only flag back to the caller).
        occ_raw_copy = np.array(self.occ_raw, copy=True)
        self.assertTrue(occ_raw_copy.flags.writeable)
        build_sector(
            self.mf, occ_raw_copy, self.vir_raw, self.mo_occ, self.mo_vir,
            self.coords, self.weights, self.n_rank_ov)
        self.assertTrue(occ_raw_copy.flags.writeable)

    def test_core_artifact_z_is_read_only(self):
        sector = self._build_ov_sector()
        core = build_core(sector)
        self.assertFalse(core.Z.flags.writeable)

    def test_mutation_after_build_does_not_alter_already_built_core(self):
        # Alice's exact repro: build a core from a SectorFit, then
        # attempt to mutate the SectorFit's own provenance -- the
        # already-built CoreArtifact's embedded copy must be unaffected
        # (this test documents the intended behavior: mutation is
        # rejected outright by the frozen structures above, so there is
        # nothing left to leak).
        sector = self._build_ov_sector()
        core = build_core(sector)
        core_provenance_before = copy.deepcopy(_thaw(core.provenance))
        with self.assertRaises(TypeError):
            sector.provenance["mutated_after_build"] = True
        self.assertEqual(_thaw(core.provenance), core_provenance_before)

    # ---- Round 2: construction validation ----

    def test_same_factor_true_rejects_unequal_factors(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_oo, same_factor=True)

    def test_mismatched_grid_and_weights_length_rejected(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights[:-1], self.n_rank_ov)

    def test_mismatched_mo_coeff_column_count_rejected(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ[:, :-1], self.mo_vir,
                self.coords, self.weights, self.n_rank_ov)

    def test_malformed_1d_grid_coords_raises_value_error_not_index_error(self):
        # A flattened grid_coords (missing the (n_grid, 3) axis) must
        # raise the documented ValueError, not an incidental IndexError
        # from .shape[1] indexing (Alice's build_core review, round 3).
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords.ravel(), self.weights, self.n_rank_ov)

    def test_malformed_1d_factor_raises_value_error_not_index_error(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw.ravel(), self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights, self.n_rank_ov)

    def test_malformed_1d_mo_coeff_raises_value_error_not_index_error(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ[:, 0], self.mo_vir,
                self.coords, self.weights, self.n_rank_ov)

    def test_malformed_2d_grid_weights_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.mo_occ, self.mo_vir,
                self.coords, self.weights[:, None], self.n_rank_ov)


def _thaw(obj):
    """Inverse of build_core._deep_freeze, for test comparison only."""
    if isinstance(obj, types.MappingProxyType):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_thaw(v) for v in obj)
    return obj


if __name__ == "__main__":
    unittest.main()
