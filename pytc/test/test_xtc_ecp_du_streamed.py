"""Tests for the streamed DF/ISDF-path plumbing of the *honest* Phase 3
ECP-Δu rank-4 correction.

The Phase 3 commit (``2f4e65e``) materialises the rank-4 ECP-Δu tensor in
full inside ``_make_xtc_eris`` and routes it into the Fock build plus the
small all-occupied blocks (oooo/ovoo/ooov/vooo).  The follow-up wiring
threads the *same* slice into the streamed virtual-index writers:

  - ``_compute_large_blocks``         → ovvv, vovv  (HDF5)
  - ``_compute_medium_blocks_tiled``  → oovv, vvoo, ovov, ovvo, vovo  (memory)
  - ``_compute_vvvv_block_df``        → vvvv        (HDF5, ECP-Δu slice
                                                    added per slab)

The strategy: we cannot directly compare DF-streamed-path blocks to the
full-tensor ``XTC.make_eris`` blocks block-by-block because the DF
auxiliary basis introduces a non-trivial fitting error in the standard
Coulomb part.  Instead we:

  1. Build the streamed ERIs twice — once with the ECP-Δu plumbing live
     (``use_ecp_du=True``-equivalent) and once with the correction
     monkey-patched to zero — then verify that the *difference* matches
     the corresponding slice of the full ECP-Δu rank-4 tensor exactly
     (to fp tolerance, since both paths use the same DF integrals and
     the same TC tiles; only the in-place ECP-Δu add differs).

  2. As a sanity test, also confirm
     ``get_2b_ecp_du(ranges=...)`` returns the same numbers as slicing
     the full tensor.
"""

from __future__ import annotations

import unittest

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc as xtc_mod
from pytc.jastrow import REXP
from pytc.solver import xtc_ccsd


def _make_be_ccecp():
    mol = gto.M(
        atom="Be 0 0 0",
        basis="ccecp-cc-pvdz",
        ecp="ccecp",
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _collect_eri_blocks(eris):
    """Materialise all (o/v) ERI blocks as dense numpy arrays for comparison."""
    block_names = [
        "oooo", "ovoo", "ooov", "vooo",
        "oovv", "vvoo", "ovov", "ovvo", "vovo", "voov",
        "ovvv", "vovv", "vvov", "vvvv",
    ]
    out = {}
    for name in block_names:
        arr = getattr(eris, name, None)
        if arr is None:
            continue
        if isinstance(arr, h5py.Dataset):
            arr = arr[...]
        out[name] = np.asarray(arr)
    return out


def _build_streamed_eris(mf, jastrow, params, grid_lvl=1, monkey_patch_du_to_zero=False):
    """Build streamed-path ERIs (DF + ISDFXTC).

    When ``monkey_patch_du_to_zero=True`` the ``get_2b_ecp_du`` method on
    the ISDF XTC object is overridden to return a zero rank-4 tensor;
    this lets us isolate the ECP-Δu contribution to each ERI block by
    diffing two streamed runs.
    """
    from pytc.xtc import ISDFXTC

    my_xtc = xtc_mod.XTC.from_pyscf(mf, jastrow, grid_lvl=grid_lvl)
    n_rank = max(8, my_xtc.grid_points.shape[0] // 4)
    isdf_xtc = ISDFXTC.from_xtc(my_xtc, n_rank=n_rank, is_incore=True)
    isdf_xtc = isdf_xtc.isdf(params)

    if monkey_patch_du_to_zero:
        n_orb = isdf_xtc.n_orb
        zero4 = jnp.zeros((n_orb, n_orb, n_orb, n_orb))
        # Replace get_2b_ecp_du on this single instance.  We have to bind
        # the override through the underlying flax dataclass __dict__
        # because ``ISDFXTC`` is a frozen ``struct.dataclass``.
        object.__setattr__(
            isdf_xtc, "get_2b_ecp_du",
            lambda *a, **kw: zero4,
        )

    mf_df = mf.density_fit(auxbasis="weigend")
    mf_df.with_df.build()
    cc = xtc_ccsd.RCCSD(mf_df, xtc_obj=isdf_xtc, jastrow_params=params,
                        gpu_max_memory=4000)
    eris = cc.ao2mo()
    blocks = _collect_eri_blocks(eris)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    fock = np.asarray(eris.fock)
    e_core = float(eris.e_core)
    eris.close()
    return blocks, fock, e_core, nocc, nmo, my_xtc


class TestStreamedEcpDuPlumbing(unittest.TestCase):
    """Streamed DF path must include the ECP-Δu Phase 3 correction in
    every virtual-index block."""

    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    def test_streamed_path_includes_ecp_du_in_all_blocks(self):
        """Diff-test: (streamed with ECP-Δu) − (streamed without ECP-Δu)
        must equal the corresponding slice of the full ECP-Δu rank-4
        tensor for *every* ERI block, including the streamed virtual-
        index blocks (ovvv/vovv/vvvv/oovv/ovov/...) under test."""
        mol, mf = _make_be_ccecp()
        jastrow = REXP()
        params = {"alpha": jnp.array([0.5])}

        # With ECP-Δu wiring live.
        blocks_with, fock_with, e_core_with, nocc, nmo, my_xtc = (
            _build_streamed_eris(mf, jastrow, params, grid_lvl=1)
        )
        # With ECP-Δu wiring suppressed at the XTC level.
        blocks_without, fock_without, e_core_without, _, _, _ = (
            _build_streamed_eris(mf, jastrow, params, grid_lvl=1,
                                 monkey_patch_du_to_zero=True)
        )

        # Reference: full rank-4 ECP-Δu tensor.
        du_full = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        self.assertGreater(
            float(np.linalg.norm(du_full)), 1e-6,
            "Test setup is degenerate: ECP-Δu correction is zero.",
        )

        # The two streamed runs share identical DF + TC tiles; only the
        # ECP-Δu in-place add differs.  So the per-block difference is
        # exactly the corresponding slice of the full ECP-Δu tensor.
        O = slice(0, nocc)
        V = slice(nocc, nmo)
        slices_per_block = {
            "oooo": (O, O, O, O),
            "ovoo": (O, V, O, O),
            "ooov": (O, O, O, V),
            "vooo": (V, O, O, O),
            "oovv": (O, O, V, V),
            "vvoo": (V, V, O, O),
            "ovov": (O, V, O, V),
            "ovvo": (O, V, V, O),
            "vovo": (V, O, V, O),
            "ovvv": (O, V, V, V),
            "vovv": (V, O, V, V),
            "vvvv": (V, V, V, V),
        }

        nontrivial = []
        for name, sl in slices_per_block.items():
            if name not in blocks_with:
                continue
            slice_full = du_full[sl]
            diff = blocks_with[name] - blocks_without[name]
            # Allowed slack: the streamed-path TC tiles are
            # *recomputed* for each ao2mo call (jax tracing + grid
            # rebuild can introduce sub-machine-epsilon perturbations),
            # so absolute tol is 1e-9 rather than 0.
            np.testing.assert_allclose(
                diff, slice_full, atol=1e-9, rtol=1e-9,
                err_msg=(
                    f"block {name}: streamed ECP-Δu contribution does NOT "
                    f"match the full-tensor slice "
                    f"(||slice||={np.linalg.norm(slice_full):.3e}, "
                    f"||diff||={np.linalg.norm(diff):.3e})"
                ),
            )
            if float(np.linalg.norm(slice_full)) > 1e-9:
                nontrivial.append(name)

        # Ensure at least one virtual-index block carries a non-trivial
        # ECP-Δu contribution — otherwise the test is vacuous.
        virtual_blocks = {"ovvv", "vovv", "vvvv", "oovv", "vvoo",
                          "ovov", "ovvo", "vovo"}
        nontrivial_virtual = [b for b in nontrivial if b in virtual_blocks]
        self.assertTrue(
            nontrivial_virtual,
            "Test setup is degenerate: no virtual-index ECP-Δu slice is "
            "non-zero.",
        )

        # Fock and e_core were already wired in Phase 3 — verify nothing
        # regressed.
        # Fock with ECP-Δu = fock_without + (2 (pq|ii) - (pi|iq)) on the
        # ECP-Δu rank-4 tensor over occupied i.
        fock_du_corr = (
            2 * np.einsum('pqii->pq', du_full[:, :, :nocc, :nocc])
            - np.einsum('piiq->pq', du_full[:, :nocc, :nocc, :])
        )
        np.testing.assert_allclose(
            fock_with - fock_without, fock_du_corr,
            atol=1e-10, rtol=1e-10,
        )
        # e_core is independent of ECP-Δu (it lives in h2e, not h0).
        self.assertAlmostEqual(e_core_with, e_core_without, places=10)

    def test_get_2b_ecp_du_ranges_matches_full_slice(self):
        """``get_2b_ecp_du(ranges=...)`` returns the same numbers as
        slicing the full rank-4 tensor."""
        mol, mf = _make_be_ccecp()
        jastrow = REXP()
        params = {"alpha": jnp.array([0.5])}
        my_xtc = xtc_mod.XTC.from_pyscf(mf, jastrow, grid_lvl=1)

        full = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        nocc = int(np.sum(mf.mo_occ > 0))
        nmo = full.shape[0]
        nvir = nmo - nocc
        self.assertGreater(nvir, 0)

        for ranges in [
            (slice(0, nocc), slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo)),
            (slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo)),
            (slice(0, 1), slice(0, nmo), slice(0, nmo), slice(0, nmo)),
        ]:
            got = np.asarray(my_xtc.get_2b_ecp_du(mf, params, ranges=ranges))
            ref = full[ranges[0], ranges[1], ranges[2], ranges[3]]
            self.assertEqual(got.shape, ref.shape)
            np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
