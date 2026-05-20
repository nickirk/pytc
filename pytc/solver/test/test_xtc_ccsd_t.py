"""Correctness test for the xTC-CCSD(T) kernel scaffold.

The kernel in ``pytc.solver.xtc_ccsd_t`` is derived from
``pyscf.cc.ccsd_t_slow`` with two modifications: it drops the ``.conj()``
calls (xTC ERIs are real but non-Hermitian) and it consumes the
non-Hermitian xTC ``eris`` blocks built by ``xtc_ccsd._make_xtc_eris``.

When fed plain (Hermitian) RCCSD ERIs and plain RCCSD amplitudes, those
two kernels must agree to machine precision — that is the comparison
this test makes. It pins down the algebra independently of any xTC
machinery; full xTC-CCSD(T) regression at finite Jastrow strength is a
separate (and approximate) test left for follow-up.
"""
import unittest

import numpy as np
import jax
from pyscf import gto, scf, cc
from pyscf.cc import ccsd_t as pyscf_ccsd_t

from pytc.solver import xtc_ccsd_t

jax.config.update("jax_enable_x64", True)


class TestXTCCCSDT_AlgebraAgainstPySCF(unittest.TestCase):
    """At zero Jastrow, our kernel reduces to plain RCCSD(T) — compare directly."""

    @classmethod
    def setUpClass(cls):
        # 6-31g gives nvir ~ 6 on HF — enough to exercise the streaming
        # path with multiple slabs and non-trivial cache pressure.
        cls.mol = gto.M(
            atom="H 0 0 0; F 0 0 0.917",
            basis="6-31g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)

        cls.cc = cc.RCCSD(cls.mf)
        cls.cc.conv_tol = 1e-10
        cls.cc.kernel()
        cls.eris = cls.cc.ao2mo()

    def test_reference_kernel_matches_pyscf(self):
        e_ref_pyscf = pyscf_ccsd_t.kernel(self.cc, self.eris,
                                          t1=self.cc.t1, t2=self.cc.t2)
        e_ref_xtc = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                      t1=self.cc.t1, t2=self.cc.t2)
        # Identical algebra on real Hermitian inputs → agreement to ~1e-12.
        self.assertAlmostEqual(e_ref_xtc, e_ref_pyscf, places=10,
                               msg=f"xtc (T) = {e_ref_xtc!r}, pyscf (T) = {e_ref_pyscf!r}")

    def test_multigpu_path_matches_reference(self):
        """The JAX/multi-device path must agree with the NumPy reference.

        On a CPU-only host this still exercises the JAX/lax.scan kernel —
        we set ``ccsd_t_force_multigpu`` so the driver doesn't short-circuit
        to NumPy.
        """
        e_ref = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                  t1=self.cc.t1, t2=self.cc.t2)
        self.cc.ccsd_t_use_multigpu = True
        self.cc.ccsd_t_force_multigpu = True
        try:
            e_mgpu = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                       t1=self.cc.t1, t2=self.cc.t2)
        finally:
            self.cc.ccsd_t_use_multigpu = False
            self.cc.ccsd_t_force_multigpu = False
        self.assertAlmostEqual(e_mgpu, e_ref, places=10,
                               msg=f"multigpu={e_mgpu!r}, reference={e_ref!r}")

    def test_multigpu_cpu_falls_back_to_reference(self):
        """Without ``force_multigpu`` and no accelerators, the driver should
        transparently fall back to the NumPy reference."""
        e_ref = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                  t1=self.cc.t1, t2=self.cc.t2)
        self.cc.ccsd_t_use_multigpu = True
        try:
            e_fallback = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                           t1=self.cc.t1, t2=self.cc.t2)
        finally:
            self.cc.ccsd_t_use_multigpu = False
        # CPU-only + no force flag → exact same code path → exact same number.
        self.assertEqual(e_fallback, e_ref)

    def test_streaming_path_matches_reference(self):
        """The HDF5-streaming path (bounded slab cache) must match reference."""
        e_ref = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                  t1=self.cc.t1, t2=self.cc.t2)
        self.cc.ccsd_t_mode = "streaming"
        self.cc.ccsd_t_force_multigpu = True
        # Force a tight cache so the LRU/refetch path is actually exercised.
        self.cc.ccsd_t_max_cached_slabs = 3
        self.cc.ccsd_t_prefetch_lookahead = 2
        try:
            e_stream = xtc_ccsd_t.kernel(self.cc, eris=self.eris,
                                         t1=self.cc.t1, t2=self.cc.t2)
        finally:
            del self.cc.ccsd_t_mode
            self.cc.ccsd_t_force_multigpu = False
            del self.cc.ccsd_t_max_cached_slabs
            del self.cc.ccsd_t_prefetch_lookahead
        self.assertAlmostEqual(e_stream, e_ref, places=10,
                               msg=f"streaming={e_stream!r}, reference={e_ref!r}")

    def test_streaming_partition_covers_all_triples(self):
        """The contiguous-`a` partition must cover every triangular triple
        exactly once across devices — sanity check the partitioner."""
        from pytc.solver.xtc_ccsd_t import (
            _per_device_triples_contiguous_a, _balanced_a_partition,
        )
        nocc, nvir = self.cc.t1.shape
        for n_dev in (1, 2, 3, 4, 7):
            covered = set()
            for d in range(n_dev):
                for t in _per_device_triples_contiguous_a(nvir, n_dev, d):
                    self.assertNotIn(t, covered,
                                     f"duplicate triple {t} for n_dev={n_dev}")
                    covered.add(t)
            expected = {(a, b, c)
                        for a in range(nvir)
                        for b in range(a + 1)
                        for c in range(b + 1)}
            self.assertEqual(covered, expected,
                             f"missing triples for n_dev={n_dev}: "
                             f"{expected - covered}")
            # Partition boundaries are monotonic.
            parts = _balanced_a_partition(nvir, n_dev)
            for (lo1, hi1), (lo2, _) in zip(parts, parts[1:]):
                self.assertEqual(hi1, lo2)


class TestXTCCCSDT_WiredIntoSolver(unittest.TestCase):
    """The ccsd_t() method on xtc_ccsd.RCCSD should round-trip the same number.

    Strategy: drive xtc_ccsd.RCCSD with a degenerate xTC object that produces
    zero corrections, so the xTC eris equals the plain RCCSD eris. We do that
    by monkey-patching get_1b/get_2b/get_const to return zeros — this exercises
    the full xtc_ccsd.RCCSD.ccsd_t() dispatch path without requiring a real
    Jastrow that vanishes (the rexp Jastrow does not vanish at any α).
    """

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="H 0 0 0; F 0 0 0.917",
            basis="sto-3g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)

    def test_ccsd_t_method_dispatches(self):
        # Plain RCCSD reference values.
        plain_cc = cc.RCCSD(self.mf)
        plain_cc.conv_tol = 1e-10
        plain_cc.kernel()
        plain_eris = plain_cc.ao2mo()
        e_t_plain = pyscf_ccsd_t.kernel(plain_cc, plain_eris,
                                        t1=plain_cc.t1, t2=plain_cc.t2)

        # Build an xtc_ccsd.RCCSD with a stub xtc_obj whose corrections are
        # identically zero — the resulting eris equals the plain RCCSD eris
        # (up to whatever Fock dressing the xTC path applies; with zero
        # corrections that's the bare HF Fock too).
        from pytc.solver import xtc_ccsd

        nmo = self.mf.mo_coeff.shape[1]
        nocc = int(np.sum(self.mf.mo_occ > 0))

        class _ZeroXTC:
            """Minimal stub matching the get_*  interface used by _make_xtc_eris."""
            def get_1b(self, params, **kw):
                return np.zeros((nmo, nmo))

            def get_2b(self, params, block_str=None, ranges=None, **kw):
                if block_str is not None:
                    sizes = [nocc if c == "o" else (nmo - nocc) for c in block_str]
                    return np.zeros(sizes)
                # full or ranges-style call: derive shape from ranges
                if ranges is not None:
                    shape = tuple(_range_size(r, nmo) for r in ranges)
                    return np.zeros(shape)
                return np.zeros((nmo, nmo, nmo, nmo))

            def get_const(self, params, **kw):
                return 0.0

        xtc_obj = _ZeroXTC()
        cc_xtc = xtc_ccsd.RCCSD(self.mf, xtc_obj=xtc_obj, jastrow_params=None)
        cc_xtc.conv_tol = 1e-10
        cc_xtc.kernel()

        e_t_xtc = cc_xtc.ccsd_t()

        # Both reduce to plain RCCSD(T) on the same molecule → tight match.
        self.assertAlmostEqual(e_t_xtc, e_t_plain, places=8,
                               msg=f"xtc.ccsd_t() = {e_t_xtc!r}, pyscf (T) = {e_t_plain!r}")

    def test_ccsd_t_method_on_jax_xtc_ccsd(self):
        """jax_xtc_ccsd.RCCSD inherits from xtc_ccsd.RCCSD, so ccsd_t() should
        be available via inheritance. This is the class used by production
        scripts (pytc_calcs/.../h10/bh/isdf_xtc.py)."""
        from pytc.solver import jax_xtc_ccsd

        nmo = self.mf.mo_coeff.shape[1]
        nocc = int(np.sum(self.mf.mo_occ > 0))

        class _ZeroXTC:
            def get_1b(self, params, **kw):
                return np.zeros((nmo, nmo))

            def get_2b(self, params, block_str=None, ranges=None, **kw):
                if block_str is not None:
                    sizes = [nocc if c == "o" else (nmo - nocc) for c in block_str]
                    return np.zeros(sizes)
                if ranges is not None:
                    shape = tuple(_range_size(r, nmo) for r in ranges)
                    return np.zeros(shape)
                return np.zeros((nmo, nmo, nmo, nmo))

            def get_const(self, params, **kw):
                return 0.0

        # Plain RCCSD reference value.
        plain_cc = cc.RCCSD(self.mf)
        plain_cc.conv_tol = 1e-10
        plain_cc.kernel()
        plain_eris = plain_cc.ao2mo()
        e_t_plain = pyscf_ccsd_t.kernel(plain_cc, plain_eris,
                                        t1=plain_cc.t1, t2=plain_cc.t2)

        # jax_xtc_ccsd path with zero-correction xTC stub.
        cc_jax = jax_xtc_ccsd.RCCSD(
            self.mf, xtc_obj=_ZeroXTC(), jastrow_params=None,
        )
        cc_jax.conv_tol = 1e-10
        cc_jax.kernel()

        # Inherited ccsd_t() method should work identically.
        e_t_jax = cc_jax.ccsd_t()

        self.assertAlmostEqual(e_t_jax, e_t_plain, places=8,
                               msg=f"jax_xtc_ccsd.ccsd_t() = {e_t_jax!r}, "
                                   f"pyscf (T) = {e_t_plain!r}")


class TestHdf5VvovSlabView(unittest.TestCase):
    """The HDF5-backed slab view must produce identical (nvir, nocc, nvir)
    slabs to the in-RAM ``ovvv.transpose(1, 3, 0, 2)[a]`` formulation."""

    def test_hdf5_slab_equals_in_memory(self):
        import tempfile
        import h5py
        from pytc.solver.xtc_ccsd_t import (
            _Hdf5VvovSlabView, _make_host_slab_view,
        )

        nocc, nvir = 4, 7
        rng = np.random.default_rng(0)
        ovvv = rng.standard_normal((nocc, nvir, nvir, nvir))

        # Reference: build in-memory vvov.
        vvov_ref = np.ascontiguousarray(ovvv.transpose(1, 3, 0, 2))

        # HDF5 path.
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with h5py.File(tmp_path, "w") as f:
                dset = f.create_dataset("ovvv", data=ovvv)
            with h5py.File(tmp_path, "r") as f:
                view = _Hdf5VvovSlabView(f["ovvv"], nocc, nvir)
                for a in range(nvir):
                    np.testing.assert_allclose(view[a], vvov_ref[a],
                                               atol=0, rtol=0)

            # _make_host_slab_view should auto-select the HDF5 path when
            # eris.ovvv is an HDF5 dataset and the in-memory path otherwise.
            class _ErisHdf5:
                pass
            with h5py.File(tmp_path, "r") as f:
                eris = _ErisHdf5()
                eris.ovvv = f["ovvv"]
                slab_src = _make_host_slab_view(eris, nocc, nvir)
                self.assertIsInstance(slab_src, _Hdf5VvovSlabView)
                np.testing.assert_allclose(slab_src[0], vvov_ref[0])

            class _ErisMem:
                pass
            eris = _ErisMem()
            eris.ovvv = ovvv
            slab_src = _make_host_slab_view(eris, nocc, nvir)
            self.assertIsInstance(slab_src, np.ndarray)
            np.testing.assert_allclose(slab_src, vvov_ref)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


def _range_size(r, nmo):
    """Resolve a slice-or-int into a size; used by the _ZeroXTC stub."""
    if isinstance(r, slice):
        start, stop, step = r.indices(nmo)
        return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
    return int(r)


if __name__ == "__main__":
    unittest.main()
