"""Tests for pytc.pbc.coulomb.ISDFDF (design doc §8, V4 gate): real,
unmodified pyscf KRHF runs consuming our integrals via with_df -- not
synthetic get_jk mocks."""

import unittest
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell
from pyscf.pbc import mp
from pyscf.pbc.scf import KRHF

import pytc.pbc.coulomb as coulomb
from pytc.pbc.coulomb import ISDFDF, get_mo_eri
from pytc.pbc.df.isdf import build_periodic_pivot_oracle, pivoted_cholesky_hermitian
from pytc.pbc.df.kpts import build_kconserv, canonicalize_kpts


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
    def _fixed_pivots(self, cell, kpts, rank=3):
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)
        diagonal, column = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size=13,
        )
        pivots, _, _ = pivoted_cholesky_hermitian(diagonal, column, rank=rank)
        return pivots

    def test_build_forwards_bpc_selection_configuration(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        adapter = ISDFDF(
            cell, kpts, rank=4, block_size=100, selection_mode="bpc_streamed",
            selection_metric="gamma",
            bpc_batch_size=8, bpc_min_separation=2.0,
            bpc_candidate_oversampling=4, bpc_n_topup=16,
        )
        sentinel = {"sentinel": True}
        with patch.object(coulomb, "build", return_value=sentinel) as mocked_build:
            self.assertIs(adapter.build(), sentinel)
        self.assertEqual(mocked_build.call_args.kwargs["selection_mode"], "bpc_streamed")
        self.assertEqual(mocked_build.call_args.kwargs["selection_metric"], "gamma")
        self.assertEqual(mocked_build.call_args.kwargs["bpc_batch_size"], 8)
        self.assertEqual(mocked_build.call_args.kwargs["bpc_min_separation"], 2.0)
        self.assertEqual(mocked_build.call_args.kwargs["bpc_candidate_oversampling"], 4)
        self.assertEqual(mocked_build.call_args.kwargs["bpc_n_topup"], 16)

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

    def test_kmp2_uses_fixed_pivot_adapter_ao2mo(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)
        diagonal, column = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size=13,
        )
        pivots, _, _ = pivoted_cholesky_hermitian(diagonal, column, rank=3)
        mf = KRHF(cell, kpts)
        mf.verbose = 0
        mf.with_df = ISDFDF(
            cell, kpts, rank=3, block_size=13,
            fixed_pivots=pivots,
        )
        mf.kernel()
        self.assertTrue(mf.converged)
        correlation, _ = mp.KMP2(mf).kernel()
        self.assertTrue(np.isfinite(correlation))
        self.assertGreater(mf.with_df._ao2mo_call_count, 0)

        built = mf.with_df.build()
        kconserv = build_kconserv(cell, built["mesh_obj"].canonical_kpts)
        k1, k2, k3 = 0, 1, 0
        k4 = int(kconserv[k1, k2, k3])
        mo_coeffs = [mf.mo_coeff[index] for index in (k1, k2, k3, k4)]
        expected, expected_k4 = get_mo_eri(
            built["inpv_kpt"], built["coul_kpt"], kconserv, mo_coeffs, k1, k2, k3,
        )
        actual = mf.with_df.ao2mo(
            mo_coeffs,
            built["mesh_obj"].canonical_kpts[[k1, k2, k3, k4]],
            compact=False,
        )
        self.assertEqual(expected_k4, k4)
        self.assertEqual(actual.shape, (expected.size,))
        np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=1e-11, rtol=1e-11)

    def test_get_jk_preserves_shuffled_and_wrapped_caller_order(self):
        cell = _make_cell()
        canonical_kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        pivots = self._fixed_pivots(cell, canonical_kpts)
        adapter = ISDFDF(
            cell, canonical_kpts, rank=3, block_size=13,
            fixed_pivots=pivots,
        )

        rng = np.random.default_rng(18)
        dm = np.empty((3, cell.nao, cell.nao), dtype=np.complex128)
        real = rng.normal(size=(cell.nao, cell.nao))
        dm[0] = real + real.T
        z = rng.normal(size=(cell.nao, cell.nao)) + 1j * rng.normal(
            size=(cell.nao, cell.nao)
        )
        dm[1] = z + z.conj().T
        dm[2] = dm[1].conj()
        _, vk_canonical = adapter.get_jk(
            dm, with_j=False, with_k=True, exxdiv=None
        )

        order = np.array([2, 0, 1])
        shuffled_kpts = canonical_kpts[order]
        shuffled = ISDFDF(
            cell, shuffled_kpts, rank=3, block_size=13,
            fixed_pivots=pivots,
        )
        _, vk_shuffled = shuffled.get_jk(
            dm[order], with_j=False, with_k=True, exxdiv=None
        )
        np.testing.assert_allclose(
            vk_shuffled, vk_canonical[order], atol=1e-11, rtol=1e-11
        )

        # nset == nk is the adversarial shape: permuting axis 0 would pass the
        # length check while silently reordering density sets instead of k-points.
        dm_sets = np.stack((dm, 0.5 * dm, -0.25 * dm), axis=0)
        _, vk_sets_canonical = adapter.get_jk(
            dm_sets, with_j=False, with_k=True, exxdiv=None
        )
        _, vk_sets_shuffled = shuffled.get_jk(
            dm_sets[:, order], with_j=False, with_k=True, exxdiv=None
        )
        np.testing.assert_allclose(
            vk_sets_shuffled, vk_sets_canonical[:, order],
            atol=1e-11, rtol=1e-11,
        )

        wrapped_kpts = cell.make_kpts([1, 1, 3], wrap_around=True)
        wrapped_mesh = canonicalize_kpts(cell, wrapped_kpts)
        wrapped = ISDFDF(
            cell, wrapped_kpts, rank=3, block_size=13,
            fixed_pivots=pivots,
        )
        dm_wrapped = wrapped_mesh.from_canonical(dm)
        _, vk_wrapped = wrapped.get_jk(
            dm_wrapped, with_j=False, with_k=True, exxdiv=None
        )
        np.testing.assert_allclose(
            vk_wrapped, wrapped_mesh.from_canonical(vk_canonical),
            atol=1e-11, rtol=1e-11,
        )

    def test_ao2mo_accepts_wrap_around_gauge(self):
        cell = _make_cell()
        canonical_kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        pivots = self._fixed_pivots(cell, canonical_kpts)
        adapter = ISDFDF(
            cell, canonical_kpts, rank=3, block_size=13,
            fixed_pivots=pivots,
        )
        built = adapter.build()
        mesh_obj = built["mesh_obj"]
        kconserv = build_kconserv(cell, mesh_obj.canonical_kpts)
        k1, k2, k3 = 2, 0, 1
        k4 = int(kconserv[k1, k2, k3])
        mo_coeffs = [
            np.eye(cell.nao, dtype=np.complex128)
            for _ in range(4)
        ]
        expected, _ = get_mo_eri(
            built["inpv_kpt"], built["coul_kpt"], kconserv, mo_coeffs,
            k1, k2, k3,
        )

        wrapped_kpts = cell.make_kpts([1, 1, 3], wrap_around=True)
        wrapped_mesh = canonicalize_kpts(cell, wrapped_kpts)
        wrapped_by_canonical = wrapped_mesh.to_canonical(wrapped_kpts)
        quartet = wrapped_by_canonical[[k1, k2, k3, k4]]
        actual = adapter.ao2mo(mo_coeffs, quartet, compact=False)
        np.testing.assert_allclose(
            actual.reshape(expected.shape), expected, atol=1e-11, rtol=1e-11
        )


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
            mf.with_df = ISDFDF(cell, kpts, rank=rank, block_size=100)
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
        mf.with_df = ISDFDF(cell, kpts, rank=6, block_size=100)
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(np.isfinite(e))


if __name__ == "__main__":
    unittest.main()
