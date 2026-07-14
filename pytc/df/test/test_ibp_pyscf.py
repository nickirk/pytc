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
    _mol_fingerprint,
)


def _h2():
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)


def _h3():
    # Three centers -> nao=3, so packed AO pairs and unequal MO blocks are
    # both non-trivial for the ao2mo shape/identity matrix.
    return gto.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.5", basis="sto-3g",
                 spin=1, verbose=0)


def _dense_ao2mo_oracle(provider, mos, compact):
    """A fully independent dense reference for IBPISDF.ao2mo. It expands the
    packed AO lower-triangle factors from ``loop()`` to the full symmetric AO
    pair tensor by hand, applies ``C1^T B C2`` / ``C3^T B C4``, and contracts
    over L -- deliberately never calling the provider's production packing or
    half-transform helpers, only its public streamed factor."""
    from pyscf.ao2mo.incore import iden_coeffs
    c1, c2, c3, c4 = mos
    b = np.vstack(list(provider.loop()))
    naoaux = b.shape[0]
    nao = provider.mol.nao
    b_full = np.zeros((naoaux, nao, nao))
    for ell in range(naoaux):
        idx = 0
        for i in range(nao):
            for j in range(i + 1):
                b_full[ell, i, j] = b[ell, idx]
                b_full[ell, j, i] = b[ell, idx]
                idx += 1

    def _flatten(ca, cb, identical):
        m = np.einsum("Lab,ai,bj->Lij", b_full, ca, cb, optimize=True)
        na, nb = ca.shape[1], cb.shape[1]
        if bool(compact) and identical:
            cols = [m[:, i, j] for i in range(na) for j in range(i + 1)]
            return np.stack(cols, axis=1)
        return m.reshape(naoaux, na * nb)

    l_bra = _flatten(c1, c2, iden_coeffs(c1, c2))
    l_ket = _flatten(c3, c4, iden_coeffs(c3, c4))
    return l_bra.T @ l_ket


class TestIBPISDFConfig(unittest.TestCase):
    def test_valid_config_accepts_all_knobs(self):
        cfg = IBPISDFConfig(
            rank=8, grid_level=1, backend="numpy", psd_rtol=1e-9,
            packed_pair_tol=1e-4, pivot_effective_rank_rtol=1e-5,
            pivot_on_over_rank="raise", grid_batch_size=64,
            eval_block_size=256, source_block_size=2048,
            core_mu_block_size=32, core_nu_block_size=32,
        )
        self.assertEqual(cfg.rank, 8)
        self.assertEqual(cfg.grid_level, 1)
        self.assertEqual(cfg.pivot_on_over_rank, "raise")
        self.assertEqual(cfg.core_mu_block_size, 32)

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
        p._ibp_mol_fingerprint = _mol_fingerprint(p.mol)
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
        self.assertEqual(q.config, p.config)
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
        self.assertEqual(p._ibp_mol_fingerprint, _mol_fingerprint(p.mol))

    def test_pseudopotential_molecule_is_rejected(self):
        # v1 supports all-electron/ECP molecules only; a pseudopotential
        # molecule must be rejected explicitly rather than silently omitted
        # from the identity digest.
        mol = _h2()
        mol._pseudo = {"H": "gth-pade"}  # simulate a pseudo molecule
        p = IBPISDF(mol, rank=3, grid_level=1)
        with self.assertRaises(NotImplementedError):
            p.build()


class TestIBPISDFao2mo(unittest.TestCase):
    """Real-only ao2mo: shape/identity matrix, compact truthiness, and value
    parity against the fully independent dense oracle."""

    def _provider(self):
        mol = _h3()
        nao = mol.nao
        p = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1)
        p.build()
        return p, nao

    def _cols(self, nao, ncol, seed):
        return np.random.default_rng(seed).normal(size=(nao, ncol))

    def test_identity_and_compact_matrix_vs_oracle(self):
        p, nao = self._provider()
        a = self._cols(nao, 2, 1)
        b = self._cols(nao, 3, 2)
        c = self._cols(nao, 4, 3)
        d = self._cols(nao, 2, 4)
        cases = [
            ("both-iden one-matrix", (a, a, a, a), True, (3, 3)),
            ("both-iden noncompact", (a, a, a, a), False, (4, 4)),
            ("bra-iden only", (a, a, c, d), True, (3, 8)),
            ("ket-iden only", (a, b, c, c), True, (6, 10)),
            ("all-distinct compact", (a, b, c, d), True, (6, 8)),
            ("all-distinct noncompact", (a, b, c, d), False, (6, 8)),
        ]
        for name, mos, compact, shape in cases:
            got = p.ao2mo(mos, compact=compact)
            self.assertEqual(got.shape, shape, msg=name)
            np.testing.assert_allclose(
                got, _dense_ao2mo_oracle(p, mos, compact), atol=1e-10, err_msg=name
            )

    def test_one_matrix_2d_input_expands_to_all_four(self):
        p, nao = self._provider()
        a = self._cols(nao, 3, 7)
        # passing a bare 2-D array == passing it for all four indices
        np.testing.assert_allclose(
            p.ao2mo(a, compact=True), p.ao2mo((a, a, a, a), compact=True), atol=1e-12
        )

    def test_compact_is_truthy_not_bool_only(self):
        # PySCF treats compact by truth value; do not reject non-bool.
        p, nao = self._provider()
        a = self._cols(nao, 2, 5)
        packed = p.ao2mo(a, compact=True).shape
        full = p.ao2mo(a, compact=False).shape
        self.assertEqual(p.ao2mo(a, compact=1).shape, packed)
        self.assertEqual(p.ao2mo(a, compact="yes").shape, packed)
        self.assertEqual(p.ao2mo(a, compact=0).shape, full)
        self.assertEqual(p.ao2mo(a, compact=None).shape, full)

    def test_complex_coefficients_rejected(self):
        p, nao = self._provider()
        a = self._cols(nao, 2, 9).astype(complex)
        with self.assertRaises(NotImplementedError):
            p.ao2mo(a)

    def test_dtype_parity_float64_only_and_rejected_before_build(self):
        # PySCF 2.10 DF ao2mo accepts float64 and rejects float32/int/bool/
        # object; v1 matches with no silent coercion, and the rejection happens
        # BEFORE the expensive lazy build (a fresh provider stays unbuilt).
        mol = _h3()
        nao = mol.nao
        f64 = np.random.default_rng(3).normal(size=(nao, 2))
        self.assertEqual(f64.dtype, np.float64)
        obj = np.array(f64, dtype=object)             # object dtype
        for bad in (f64.astype(np.float32),
                    f64.astype(np.int64),
                    (f64 > 0),                          # bool
                    obj):
            p = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1)
            with self.assertRaises(TypeError):
                p.ao2mo(bad)
            self.assertFalse(p._ibp_built, "dtype rejection must precede lazy build")
        # float64 is accepted (and builds)
        p = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1)
        p.ao2mo(f64)
        self.assertTrue(p._ibp_built)

    def test_malformed_inputs_rejected(self):
        p, nao = self._provider()
        good = self._cols(nao, 2, 3)
        bad_dim = self._cols(nao + 1, 2, 3)          # wrong AO row count
        nonfinite = self._cols(nao, 2, 3); nonfinite[0, 0] = np.inf
        for arg in (
            "not-an-array",
            np.zeros((nao, 2, 2)),                    # 3-D single
            (good, good, good),                       # length-3 tuple
            (good, good, good, bad_dim),              # wrong AO dim in one
            (good, good, good, nonfinite),            # non-finite
        ):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                p.ao2mo(arg)

    def test_zero_rank_ao2mo_returns_zero_and_validates_block(self):
        # An injected valid zero-rank cache: ao2mo iterates loop() (zero blocks
        # -> zero output) yet still validates the (default) block size.
        p, nao = self._provider()
        p._ibp_factor = np.zeros((p._ibp_pair.shape[0], 0))   # W with 0 columns
        p._ibp_naoaux = 0
        a = self._cols(nao, 2, 1)
        out = p.ao2mo(a, compact=True)
        self.assertEqual(out.shape, (3, 3))
        np.testing.assert_array_equal(out, 0.0)
        p.blockdim = True                                     # corrupt default block size
        with self.assertRaises(ValueError):
            p.ao2mo(a, compact=True)

    def test_multi_block_dgemm_accumulation_matches_oracle(self):
        # Force several L blocks (small blockdim) so the beta-accumulate is
        # exercised across blocks, spy the module-level dgemm to prove every
        # call accumulates in place into the one F-contiguous output object
        # (same buffer, overwrite_c=1), and confirm the streamed result still
        # matches the independent oracle.
        import pytc.df.ibp_pyscf as mod
        p, nao = self._provider()
        p.blockdim = 3
        a = self._cols(nao, 3, 5)
        b = self._cols(nao, 2, 6)
        self.assertGreater(len(list(p.loop())), 1)  # genuinely multi-block

        seen = []
        orig = mod.dgemm

        def spy(*args, **kwargs):
            ret = orig(*args, **kwargs)
            seen.append((kwargs.get("c"), kwargs.get("overwrite_c"), ret))
            return ret

        try:
            mod.dgemm = spy
            got = p.ao2mo((a, a, b, b), compact=True)
        finally:
            mod.dgemm = orig

        self.assertGreater(len(seen), 1, "dgemm was not called per block")
        first_c = seen[0][0]
        for c_arg, overwrite, ret in seen:
            self.assertEqual(overwrite, 1)
            self.assertTrue(c_arg.flags.f_contiguous)
            self.assertIs(c_arg, first_c)                 # one output object across blocks
            self.assertTrue(np.shares_memory(ret, c_arg))  # accumulated in place
        np.testing.assert_allclose(
            got, _dense_ao2mo_oracle(p, (a, a, b, b), True), atol=1e-10
        )

    def test_no_dense_cache_retained_and_cderi_none(self):
        p, nao = self._provider()
        a = self._cols(nao, 3, 2)
        _ = p.ao2mo(a)
        # ao2mo derives solely from W/P via loop(); no inherited _cderi and no
        # retained full-B / AO four-index cache. Whitelist the permitted W/P
        # arrays and reject any other array-valued attribute that is >=3-D or
        # carries a packed-AO-pair axis (the full-B shape is (naoaux, nao_pair)
        # with naoaux <= nao_pair, so a bare shape[0] > nao_pair test misses it).
        self.assertIsNone(p._cderi)
        nao_pair = nao * (nao + 1) // 2
        allowed = {"_ibp_factor", "_ibp_pair"}
        for name in vars(p):
            val = getattr(p, name)
            if not isinstance(val, np.ndarray) or name in allowed:
                continue
            if val.ndim >= 3:
                self.fail(f"unexpected dense (>=3-D) cache retained: {name} {val.shape}")
            if val.ndim == 2 and nao_pair in val.shape:
                self.fail(f"unexpected packed-AO-pair cache retained: {name} {val.shape}")

    def test_ao2mo_lazy_builds_and_honors_stale_guard(self):
        # Lazy build through ao2mo.
        mol = _h3(); nao = mol.nao
        p = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1)
        self.assertFalse(p._ibp_built)
        p.ao2mo(np.eye(nao))
        self.assertTrue(p._ibp_built)
        # Stale-molecule guard fires through ao2mo.
        p.mol.set_geom_("H 0 0 0; H 0 0 0.80; H 0 0 1.6")
        with self.assertRaises(RuntimeError):
            p.ao2mo(np.eye(nao))

    def test_ao2mo_after_copy_builds_independently(self):
        p, nao = self._provider()
        q = p.copy()
        self.assertFalse(q._ibp_built)
        a = self._cols(nao, 2, 1)
        np.testing.assert_allclose(q.ao2mo(a), p.ao2mo(a), atol=1e-10)


class TestIBPISDFPhysicalBuild(unittest.TestCase):
    def test_h2_sto3g_level1_build_and_reconstruction(self):
        """A genuine end-to-end build. The streamed factor B = W^dagger P must
        reconstruct the packed-pair metric P^dagger Z P since Z ~ W W^dagger."""
        p = IBPISDF(_h2(), rank=3, grid_level=1)
        p.build()
        self.assertTrue(p._ibp_built)
        self.assertEqual(p.get_naoaux(), p._ibp_diagnostics.psd_retained_rank)
        self.assertEqual(p._ibp_diagnostics.psd_status, "factorized")
        full = np.vstack(list(p.loop()))
        P = p._ibp_pair
        Z = p._ibp_diagnostics.Z
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
        self.assertEqual(p._ibp_diagnostics.psd_status, "factorized")
        self.assertGreater(p.get_naoaux(), 0)
        self.assertLessEqual(
            p._ibp_diagnostics.raw_packed_pair_metric_dagger_residual, p.config.packed_pair_tol
        )


@unittest.skipUnless(os.environ.get("PYTC_RUN_SLOW_IBP"),
                     "opt-in unchanged-consumer physical gate (set PYTC_RUN_SLOW_IBP=1)")
class TestIBPISDFUnchangedConsumers(unittest.TestCase):
    """Run stock PySCF DFMP2/CCSD with mf.with_df = IBPISDF -- no monkeypatch,
    no shim -- on frozen shared exact-RHF orbitals so the number isolates the
    integral metric. Spies prove no analytic-DF fallback and no ao2mo path."""

    @staticmethod
    def _h2o():
        return gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                     basis="cc-pVDZ", verbose=0)

    def test_dfmp2_within_0p1_mHa_of_exact_four_center(self):
        from pyscf import scf, mp, df as pyscf_df
        from pyscf.mp import dfmp2
        import pyscf.df.df as dfmod
        mol = self._h2o()
        nao = mol.nao
        mf = scf.RHF(mol)
        mf.kernel()  # exact (non-DF) RHF; its orbitals are shared by all three
        e_exact = mp.MP2(mf).kernel()[0]
        # analytic DF-MP2 reference, constructed OUTSIDE the no-fallback spy
        mf.with_df = pyscf_df.DF(mol).build()
        e_dfmp2 = dfmp2.DFMP2(mf).kernel()[0]
        # IBP provider, pre-built before the spy so its own DF.__init__ is not counted
        ibp = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1).build()
        rank = ibp.get_naoaux()
        pair_resid = ibp._ibp_diagnostics.raw_packed_pair_metric_dagger_residual
        calls = {"df_init": 0, "ao2mo": 0}
        orig_init, orig_ao2mo = dfmod.DF.__init__, IBPISDF.ao2mo

        def spy_init(self, *a, **k):
            calls["df_init"] += 1
            return orig_init(self, *a, **k)

        def spy_ao2mo(self, *a, **k):
            calls["ao2mo"] += 1
            return orig_ao2mo(self, *a, **k)

        mf.with_df = ibp
        try:
            dfmod.DF.__init__ = spy_init
            IBPISDF.ao2mo = spy_ao2mo
            e_ibp = dfmp2.DFMP2(mf).kernel()[0]
        finally:
            dfmod.DF.__init__ = orig_init
            IBPISDF.ao2mo = orig_ao2mo
        d_ibp = abs(e_ibp - e_exact) * 1e3
        print(f"\n[IBP-DFMP2 gate] nao={nao} requested_rank={nao*(nao+1)//2} "
              f"realized_rank={rank} pair_resid={pair_resid:.2e}\n"
              f"  e_exact4c={e_exact:.8f}  e_dfmp2={e_dfmp2:.8f}  e_ibp={e_ibp:.8f}\n"
              f"  |ibp-exact|={d_ibp:.4f} mHa  |dfmp2-exact|={abs(e_dfmp2-e_exact)*1e3:.4f} mHa")
        self.assertEqual(calls["df_init"], 0, "an analytic DF fallback was constructed")
        self.assertEqual(calls["ao2mo"], 0, "DFMP2 unexpectedly used ao2mo, not loop")
        self.assertLess(d_ibp, 0.1)

    def test_ccsd_ibp_drives_integrals_with_no_fallback(self):
        # Exact (non-DF) RHF reference: cc.CCSD(mf) is a conventional
        # four-center RCCSD (pyscf.cc.ccsd.CCSD, no with_df), built OUTSIDE the
        # spy. cc.CCSD(mf).density_fit(with_df=ibp) is a DF-CCSD
        # (pyscf.cc.dfccsd.RCCSD) whose CC integral build uses IBP. Both share
        # the same exact-RHF orbitals so the difference isolates the integral
        # metric. The provider and the DF-CCSD object are constructed before the
        # spy; the spy wraps only the IBP CC kernel and requires zero
        # analytic-DF construction, zero ao2mo, and loop() > 0.
        from pyscf import scf, cc
        import pyscf.df.df as dfmod
        mol = self._h2o()
        nao = mol.nao
        mf = scf.RHF(mol)
        mf.kernel()  # exact non-DF RHF; frozen orbitals shared by ref and IBP
        ref = cc.CCSD(mf)             # conventional four-center CCSD
        ref.max_cycle = 1
        ref.kernel()
        self.assertFalse(hasattr(ref, "with_df"))
        ibp = IBPISDF(mol, rank=nao * (nao + 1) // 2, grid_level=1).build()
        mycc = cc.CCSD(mf).density_fit(with_df=ibp)   # dfccsd.RCCSD, built outside spy
        mycc.max_cycle = 1
        self.assertIs(mycc.with_df, ibp)
        calls = {"df_init": 0, "ao2mo": 0, "loop": 0}
        orig_init, orig_ao2mo, orig_loop = (
            dfmod.DF.__init__, IBPISDF.ao2mo, IBPISDF.loop)

        def spy_init(self, *a, **k):
            calls["df_init"] += 1
            return orig_init(self, *a, **k)

        def spy_ao2mo(self, *a, **k):
            calls["ao2mo"] += 1
            return orig_ao2mo(self, *a, **k)

        def spy_loop(self, *a, **k):
            calls["loop"] += 1
            return orig_loop(self, *a, **k)

        try:
            dfmod.DF.__init__ = spy_init
            IBPISDF.ao2mo = spy_ao2mo
            IBPISDF.loop = spy_loop
            mycc.kernel()
        finally:
            dfmod.DF.__init__ = orig_init
            IBPISDF.ao2mo = orig_ao2mo
            IBPISDF.loop = orig_loop
        self.assertEqual(calls["df_init"], 0, "an analytic DF fallback built the CC integrals")
        self.assertEqual(calls["ao2mo"], 0, "CCSD unexpectedly used ao2mo, not loop")
        self.assertGreater(calls["loop"], 0, "IBP loop() did not drive the CC integrals")
        self.assertTrue(np.isfinite(mycc.e_corr))
        self.assertTrue(np.all(np.isfinite(mycc.t1)))
        self.assertTrue(np.all(np.isfinite(mycc.t2)))
        # Direct conventional-reference differences (no acceptance threshold yet).
        d_ecorr = abs(mycc.e_corr - ref.e_corr) * 1e3
        d_t1 = float(np.max(np.abs(mycc.t1 - ref.t1)))
        d_t2 = float(np.max(np.abs(mycc.t2 - ref.t2)))
        print(f"\n[IBP DF-CCSD 1-cycle vs conventional 4c, exact-RHF orbitals] "
              f"dE_corr={d_ecorr:.6f} mHa max|dt1|={d_t1:.3e} max|dt2|={d_t2:.3e}")


if __name__ == "__main__":
    unittest.main()
