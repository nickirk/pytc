"""Tests for the PySCF-compatible IBPISDF density-fitting provider.

The fast suite exercises configuration validation, the constructor contract,
the streamed ``loop``/``get_naoaux`` surface (via injected artifacts and a
genuinely fast H2/STO-3G build), lifecycle (copy/reset), and the atomic build
gates (via a mocked core). One opt-in heavier physical acceptance builds
H2O/cc-pVDZ at grid level 1; it is skipped unless PYTC_RUN_SLOW_IBP is set.

The shared prepared normal-equation solver is JAX-backed and refuses to
downcast, so x64 must be enabled before any sector build.
"""

import io
import os
import types
import unittest
from unittest import mock

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf import gto

from pytc.df.ibp_pyscf import (
    IBPISDF,
    IBPISDFConfig,
    _canonical_molecule_digest,
)


def _h2():
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)


class TestIBPISDFConfig(unittest.TestCase):
    def test_valid_and_spec_is_stable_and_discriminating(self):
        a = IBPISDFConfig(rank=8, grid_level=1)
        b = IBPISDFConfig(rank=8, grid_level=1)
        self.assertEqual(a.config_spec_sha256, b.config_spec_sha256)
        self.assertNotEqual(
            a.config_spec_sha256, IBPISDFConfig(rank=9, grid_level=1).config_spec_sha256
        )
        # every artifact-affecting knob perturbs the digest
        for kw in (
            dict(grid_level=2),
            dict(backend="numpy", psd_rtol=1e-9),
            dict(packed_pair_tol=1e-4),
            dict(pivot_effective_rank_rtol=1e-5),
            dict(pivot_on_over_rank="raise"),
            dict(grid_batch_size=64),
            dict(eval_block_size=256),
            dict(source_block_size=2048),
            dict(core_mu_block_size=32),
            dict(core_nu_block_size=32),
        ):
            self.assertNotEqual(
                a.config_spec_sha256,
                IBPISDFConfig(rank=8, **{"grid_level": 1, **kw}).config_spec_sha256,
                msg=f"knob {kw} did not affect the config digest",
            )

    def test_rank_required_and_positive(self):
        with self.assertRaises(TypeError):
            IBPISDFConfig()
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                IBPISDFConfig(rank=bad)

    def test_backend_restricted_to_numpy(self):
        IBPISDFConfig(rank=4, backend="numpy")
        with self.assertRaises(ValueError):
            IBPISDFConfig(rank=4, backend="jax")

    def test_grid_level_nonnegative(self):
        IBPISDFConfig(rank=4, grid_level=0)  # coarsest grid is valid
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                IBPISDFConfig(rank=4, grid_level=bad)

    def test_tolerances_and_blocks_validated(self):
        for kw in (
            dict(psd_rtol=-1e-9),
            dict(psd_rtol=True),
            dict(packed_pair_tol=0.0),
            dict(packed_pair_tol=-1e-3),
            dict(pivot_effective_rank_rtol=0.0),
            dict(pivot_on_over_rank="nope"),
            dict(eval_block_size=0),
            dict(eval_block_size=True),
            dict(source_block_size=-1),
            dict(grid_batch_size=0),
            dict(core_mu_block_size=True),
            dict(core_nu_block_size=-2),
        ):
            with self.assertRaises(ValueError):
                IBPISDFConfig(rank=4, **kw)
        # psd_rtol=0 is allowed (non-negative); None block sizes are allowed
        IBPISDFConfig(rank=4, psd_rtol=0.0, grid_batch_size=None,
                      core_mu_block_size=None, core_nu_block_size=None)

    def test_pivot_effective_rank_rtol_open_unit_interval(self):
        # The selector contract is 0 < rtol < 1; anything outside is rejected.
        IBPISDFConfig(rank=4, pivot_effective_rank_rtol=0.5)
        for bad in (0.0, 1.0, 2.0, -0.1, True):
            with self.assertRaises(ValueError):
                IBPISDFConfig(rank=4, pivot_effective_rank_rtol=bad)


class TestIBPISDFConstruction(unittest.TestCase):
    def test_config_xor_direct_knobs(self):
        with self.assertRaises(ValueError):
            IBPISDF(_h2(), config=IBPISDFConfig(rank=3), rank=3)

    def test_config_type_checked(self):
        with self.assertRaises(TypeError):
            IBPISDF(_h2(), config=object())

    def test_auxbasis_rejected(self):
        with self.assertRaises(ValueError):
            IBPISDF(_h2(), auxbasis="weigend")

    def test_rank_required(self):
        with self.assertRaises(ValueError):
            IBPISDF(_h2())

    def test_starts_unbuilt_with_metric(self):
        p = IBPISDF(_h2(), rank=3)
        self.assertFalse(p._ibp_built)
        self.assertEqual(p.metric, "atom_centered_single_ibp")
        self.assertEqual(p.config.rank, 3)


class TestIBPISDFStreaming(unittest.TestCase):
    """loop()/get_naoaux() streaming, exercised on injected W/P so the math is
    isolated from the (separately tested) physical build."""

    def _inject(self, p, W, P):
        p._ibp_factor = W
        p._ibp_pair = P
        p._ibp_naoaux = int(W.shape[1])
        # Match the current molecule identity so build()'s idempotent identity
        # guard treats this injected cache as a valid built state.
        p._ibp_mol_digest = _canonical_molecule_digest(p.mol)
        p._ibp_built = True

    def test_loop_yields_W_dagger_P_in_blocks(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(5, 3))  # (n_mu, retained_rank)
        P = rng.normal(size=(5, 4))  # (n_mu, n_pair)
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, W, P)
        self.assertEqual(p.get_naoaux(), 3)
        blocks = list(p.loop(blksize=2))
        self.assertEqual([b.shape for b in blocks], [(2, 4), (1, 4)])
        np.testing.assert_allclose(np.vstack(blocks), W.conj().T @ P)

    def test_loop_default_blksize_single_block(self):
        rng = np.random.default_rng(1)
        W = rng.normal(size=(5, 3))
        P = rng.normal(size=(5, 4))
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, W, P)
        blocks = list(p.loop())  # default blockdim (240) >> naoaux
        self.assertEqual(len(blocks), 1)
        np.testing.assert_allclose(blocks[0], W.conj().T @ P)

    def test_blksize_validation(self):
        rng = np.random.default_rng(2)
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, rng.normal(size=(5, 3)), rng.normal(size=(5, 4)))
        for bad in (True, 0, -1, 2.5, "8"):
            with self.assertRaises(ValueError):
                list(p.loop(blksize=bad))

    def test_zero_naoaux_is_empty_iterator(self):
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, np.zeros((5, 0)), np.zeros((5, 4)))
        self.assertEqual(p.get_naoaux(), 0)
        self.assertEqual(list(p.loop()), [])

    def test_zero_naoaux_still_validates_blksize(self):
        # The block size must be validated BEFORE the zero-rank short circuit,
        # so an invalid blksize is rejected even when there is nothing to yield.
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, np.zeros((5, 0)), np.zeros((5, 4)))
        for bad in (False, 0, -1, 2.5):
            with self.assertRaises(ValueError):
                list(p.loop(blksize=bad))

    def test_corrupt_default_blockdim_is_rejected(self):
        # An invalid default blockdim must not be silently coerced to int.
        p = IBPISDF(_h2(), rank=3)
        self._inject(p, np.zeros((5, 0)), np.zeros((5, 4)))
        p.blockdim = True
        with self.assertRaises(ValueError):
            list(p.loop())


class TestIBPISDFLifecycle(unittest.TestCase):
    def _mark_built(self, p):
        p._ibp_built = True
        p._ibp_factor = np.ones((5, 3))
        p._ibp_pair = np.ones((5, 4))
        p._ibp_naoaux = 3

    def test_copy_is_unbuilt_same_config_independent(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        self._mark_built(p)
        q = p.copy()
        self.assertFalse(q._ibp_built)
        self.assertIsNone(q._ibp_factor)
        self.assertEqual(q.config.config_spec_sha256, p.config.config_spec_sha256)
        # mutating the copy's cache does not touch the original
        self.assertTrue(p._ibp_built)

    def test_copy_preserves_pyscf_runtime_settings(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        marker = io.StringIO()
        p.blockdim = 7
        p.max_memory = 123
        p.verbose = 5
        p.stdout = marker
        q = p.copy()
        self.assertEqual(q.blockdim, 7)
        self.assertEqual(q.max_memory, 123)
        self.assertEqual(q.verbose, 5)
        self.assertIs(q.stdout, marker)
        self.assertFalse(q._ibp_built)  # runtime preserved, build cache not

    def test_reset_clears_built_state(self):
        p = IBPISDF(_h2(), rank=3)
        self._mark_built(p)
        p.reset()
        self.assertFalse(p._ibp_built)
        self.assertIsNone(p._ibp_naoaux)
        self.assertIsNone(p._ibp_factor)

    def test_ao2mo_not_implemented(self):
        p = IBPISDF(_h2(), rank=3)
        with self.assertRaises(NotImplementedError):
            p.ao2mo(np.eye(2))

    def test_get_jk_not_implemented(self):
        p = IBPISDF(_h2(), rank=3)
        with self.assertRaises(NotImplementedError):
            p.get_jk(np.eye(2))


class TestIBPISDFBuildGates(unittest.TestCase):
    """The build is atomic: if either publish gate fails, no cache is set."""

    def _stub_core(self, pair_resid, psd_status="factorized"):
        return types.SimpleNamespace(
            raw_packed_pair_metric_dagger_residual=pair_resid,
            psd_status=psd_status,
            psd_retained_rank=3,
            psd_factor=np.ones((3, 3)),
            Z=np.eye(3),
        )

    def test_packed_pair_gate_leaves_provider_unbuilt(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1, packed_pair_tol=1e-6)
        with mock.patch("pytc.df.ibp_pyscf.ibp_core",
                        return_value=self._stub_core(1e-2)):
            with self.assertRaises(ValueError):
                p.build()
        self.assertFalse(p._ibp_built)
        self.assertIsNone(p._ibp_factor)
        self.assertIsNone(p._ibp_naoaux)

    def test_psd_gate_leaves_provider_unbuilt(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        with mock.patch("pytc.df.ibp_pyscf.ibp_core",
                        return_value=self._stub_core(1e-9, psd_status="not_applicable")):
            with self.assertRaises(ValueError):
                p.build()
        self.assertFalse(p._ibp_built)
        self.assertIsNone(p._ibp_factor)


class TestMoleculeDigest(unittest.TestCase):
    """The molecule identity must be canonical (no repr) and complete."""

    def test_cart_vs_spherical_distinguished(self):
        # Alice's repro: C/cc-pVDZ spherical (14 AO) vs Cartesian (15 AO) must
        # not share a digest even though the requested basis name is identical.
        sph = gto.M(atom="C 0 0 0", basis="cc-pvdz", spin=2, verbose=0)
        cart = gto.M(atom="C 0 0 0", basis="cc-pvdz", spin=2, cart=True, verbose=0)
        self.assertNotEqual(sph.nao, cart.nao)
        self.assertNotEqual(
            _canonical_molecule_digest(sph), _canonical_molecule_digest(cart)
        )

    def test_geometry_basis_charge_spin_distinguished(self):
        base = _canonical_molecule_digest(_h2())
        moved = _canonical_molecule_digest(
            gto.M(atom="H 0 0 0; H 0 0 0.90", basis="sto-3g", verbose=0)
        )
        rebased = _canonical_molecule_digest(
            gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        )
        self.assertNotEqual(base, moved)
        self.assertNotEqual(base, rebased)

    def test_digest_is_reproducible(self):
        self.assertEqual(_canonical_molecule_digest(_h2()),
                         _canonical_molecule_digest(_h2()))


class TestIBPISDFProvenanceBinding(unittest.TestCase):
    def test_provider_provenance_bound_in_every_artifact(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        p.build()
        prov = p.provenance
        self.assertEqual(
            sorted(prov),
            ["adapter_version", "config_spec_sha256", "metric", "mol_digest",
             "provider", "pyscf_version"],
        )
        self.assertEqual(prov["mol_digest"], _canonical_molecule_digest(p.mol))
        self.assertEqual(prov["config_spec_sha256"], p.config.config_spec_sha256)
        # The same closed record travels through grid, sector, plan, and core.
        self.assertEqual(
            p._ibp_grid.construction_metadata["provider_provenance"]["mol_digest"],
            prov["mol_digest"],
        )
        for artifact in (p._ibp_sector, p._ibp_plan, p._ibp_core):
            bound = artifact.provenance["upstream_provenance"]
            self.assertEqual(bound["mol_digest"], prov["mol_digest"])
            self.assertEqual(bound["config_spec_sha256"], prov["config_spec_sha256"])
            self.assertEqual(bound["adapter_version"], prov["adapter_version"])


class TestIBPISDFStaleMolecule(unittest.TestCase):
    def test_inplace_geometry_mutation_after_build_is_rejected(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        p.build()
        p.mol.set_geom_("H 0 0 0; H 0 0 0.90")
        for access in (p.build, p.get_naoaux, lambda: list(p.loop())):
            with self.assertRaises(RuntimeError):
                access()

    def test_reset_then_rebuild_after_mutation_succeeds(self):
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        p.build()
        p.mol.set_geom_("H 0 0 0; H 0 0 0.90")
        p.reset()
        p.build()  # clean rebuild on the new geometry
        self.assertTrue(p._ibp_built)
        self.assertEqual(p.provenance["mol_digest"], _canonical_molecule_digest(p.mol))


class TestIBPISDFPhysicalBuild(unittest.TestCase):
    def test_h2_sto3g_level1_build_and_reconstruction(self):
        """A genuine end-to-end build. The streamed factor B = W^dagger P must
        reconstruct the packed-pair metric P^dagger Z P since Z ~ W W^dagger."""
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        p.build()
        self.assertTrue(p._ibp_built)
        self.assertEqual(p.get_naoaux(), p._ibp_core.psd_retained_rank)
        self.assertEqual(p._ibp_core.psd_status, "factorized")
        full = np.vstack(list(p.loop()))
        P = p._ibp_pair
        Z = p._ibp_core.Z
        metric = P.conj().T @ Z @ P
        np.testing.assert_allclose(full.conj().T @ full, metric, atol=1e-10, rtol=1e-8)

    @unittest.skipUnless(os.environ.get("PYTC_RUN_SLOW_IBP"),
                         "opt-in heavier physical acceptance (set PYTC_RUN_SLOW_IBP=1)")
    def test_h2o_ccpvdz_level1_acceptance(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis="cc-pVDZ", verbose=0,
        )
        p = IBPISDF(mol, rank=200, grid_level=1)
        p.build()
        self.assertEqual(p._ibp_core.psd_status, "factorized")
        self.assertGreater(p.get_naoaux(), 0)
        self.assertLessEqual(
            p._ibp_core.raw_packed_pair_metric_dagger_residual, p.config.packed_pair_tol
        )


if __name__ == "__main__":
    unittest.main()
