"""Regression gate for the reference-normal-ordered X-channel study."""

import copy
import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow.rexp import REXP
from pytc.solver import jax_xtc_ccsd, xtc_ccsd


jax.config.update("jax_enable_x64", True)


_CLEAN_X_ENV = {
    "PYTC_XTC_DROP_X": "0",
    "PYTC_XTC_DROP_X_NORMAL_ORDER": "0",
    "PYTC_XTC_DROP_X_RESIDUAL": "0",
}
_ERI_BLOCKS = (
    "oooo", "ovoo", "ooov", "vooo", "ovov", "vovo", "ovvo", "voov",
    "oovv", "vvoo", "ovvv", "vvov", "vovv", "vvvv",
)


def _explicit_reference_energy(eris):
    """Independent reference-energy expansion for the normal-order gate."""
    nocc = eris.nocc
    fock = np.asarray(eris.fock)
    oooo = np.asarray(eris.oooo)
    return (
        2 * np.einsum("ii->", fock[:nocc, :nocc])
        - 2 * np.einsum("iijj->", oooo)
        + np.einsum("ijji->", oooo)
        + eris.e_core
    ).real


def _fock_from_delta_u(delta_u, nocc):
    occupied = slice(0, nocc)
    return (
        2 * np.einsum("pqii->pq", delta_u[:, :, occupied, occupied])
        - np.einsum("piiq->pq", delta_u[:, occupied, occupied, :])
    )


class TestNormalOrderedXView(unittest.TestCase):
    """Validate all scalar/one-body/two-body parts before an H10 study."""

    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        cls.mf = scf.RHF(mol).run()
        cls.jparams = {"alpha": jnp.array([1.0])}
        base = xtc.XTC.from_pyscf(cls.mf, REXP(), grid_lvl=0)
        cls.base_isdf = xtc.ISDFXTC.from_xtc(
            base, n_rank=max(8, 3 * base.n_orb), is_incore=True
        )

    def _build_full_and_no_x_eris(self):
        # Build X exactly once.  The global no-X endpoint below uses the same
        # object/kernel store and only changes the read-time X contribution.
        with mock.patch.dict(os.environ, _CLEAN_X_ENV):
            full_obj = self.base_isdf.isdf(
                self.jparams,
                batch_size=64,
                orb_block_size=2,
                host_grid_block_size=512,
            )
        self.assertGreater(np.linalg.norm(np.asarray(full_obj.isdf_kernels["X"])), 0)

        def build(env):
            with mock.patch.dict(os.environ, env):
                cc = jax_xtc_ccsd.RCCSD(
                    self.mf, full_obj, self.jparams,
                    max_memory=2_000,
                    gpu_max_memory=2_000,
                    on_the_fly_vvvv=False,
                )
                return cc.ao2mo()

        full_eris = build(_CLEAN_X_ENV)
        no_x_eris = build({**_CLEAN_X_ENV, "PYTC_XTC_DROP_X": "1"})
        return full_obj, full_eris, no_x_eris

    def test_x_normal_ordered_views_are_componentwise_exact(self):
        full_obj, full_eris, no_x_eris = self._build_full_and_no_x_eris()
        self.addCleanup(full_eris.close)
        self.addCleanup(no_x_eris.close)

        # Explicitly normal order Delta-U itself.  This is the physics gate:
        # δh_X = -1/2 F[ΔU_X], and the scalar follows from that same δh_X.
        with mock.patch.dict(os.environ, _CLEAN_X_ENV):
            delta_h_full = np.asarray(full_obj.get_delta_h(self.jparams))
            delta_u_full = np.asarray(full_obj.get_delta_U(self.jparams))
            e0_full = float(full_obj.get_const(self.jparams, delta_h=delta_h_full))
        with mock.patch.dict(os.environ, {**_CLEAN_X_ENV, "PYTC_XTC_DROP_X": "1"}):
            delta_h_no_x = np.asarray(full_obj.get_delta_h(self.jparams))
            delta_u_no_x = np.asarray(full_obj.get_delta_U(self.jparams))
            e0_no_x = float(full_obj.get_const(self.jparams, delta_h=delta_h_no_x))

        delta_h_x = delta_h_full - delta_h_no_x
        delta_u_x = delta_u_full - delta_u_no_x
        fock_delta_u_x = _fock_from_delta_u(delta_u_x, full_eris.nocc)
        dm1 = np.asarray(full_obj._get_mf_dm())
        np.testing.assert_allclose(delta_h_x, -0.5 * fock_delta_u_x, atol=1e-11, rtol=1e-11)
        self.assertAlmostEqual(
            e0_full - e0_no_x,
            float(-2 / 3 * np.einsum("qp,pq->", delta_h_x, dm1)),
            places=11,
        )
        np.testing.assert_allclose(
            np.asarray(full_eris.fock) - np.asarray(no_x_eris.fock),
            delta_h_x + fock_delta_u_x,
            atol=1e-11,
            rtol=1e-11,
        )

        drop_zero_one = xtc_ccsd.make_x_normal_ordered_eris_view(
            full_eris, no_x_eris, "zero_one"
        )
        drop_two_body = xtc_ccsd.make_x_normal_ordered_eris_view(
            full_eris, no_x_eris, "two_body"
        )

        # The two source endpoints define a linear normal-order triple.
        x_reference = (
            xtc_ccsd.eris_reference_energy(full_eris)
            - xtc_ccsd.eris_reference_energy(no_x_eris)
        )
        self.assertNotAlmostEqual(x_reference, 0.0, places=11)
        nocc = full_eris.nocc
        nmo = full_eris.fock.shape[0]
        for block in _ERI_BLOCKS:
            self.assertTrue(hasattr(full_eris, block))
            block_slices = tuple(
                slice(0, nocc) if index == "o" else slice(nocc, nmo)
                for index in block
            )
            np.testing.assert_allclose(
                np.asarray(getattr(full_eris, block))
                - np.asarray(getattr(no_x_eris, block)),
                delta_u_x[block_slices],
                atol=1e-11,
                rtol=1e-11,
            )

        # Check scalar, one-body, and every two-body block independently.
        # Endpoint reconstruction alone would not exercise these hybrids.
        for view, reference_source, two_body_source in (
            (drop_zero_one, no_x_eris, full_eris),
            (drop_two_body, full_eris, no_x_eris),
        ):
            self.assertIsNone(view.feri)
            self.assertFalse(view.fock.flags.writeable)
            np.testing.assert_array_equal(view.fock, reference_source.fock)
            np.testing.assert_array_equal(view.mo_energy, np.diag(reference_source.fock))
            self.assertAlmostEqual(
                _explicit_reference_energy(view),
                xtc_ccsd.eris_reference_energy(reference_source),
                places=12,
            )
            self.assertAlmostEqual(
                xtc_ccsd.eris_reference_energy(view),
                xtc_ccsd.eris_reference_energy(reference_source),
                places=12,
            )
            for block in _ERI_BLOCKS:
                np.testing.assert_array_equal(
                    np.asarray(getattr(view, block)),
                    np.asarray(getattr(two_body_source, block)),
                )

        # The construction must reject the JAX dynamic VVVV path: it would
        # regenerate Delta-U from xtc_obj rather than consume this ERI view.
        on_the_fly_like = copy.copy(full_eris)
        on_the_fly_like.vvvv = None
        with self.assertRaisesRegex(ValueError, "materialized VVVV"):
            xtc_ccsd.make_x_normal_ordered_eris_view(
                on_the_fly_like, no_x_eris, "zero_one"
            )

        # Finally run the two hybrid Hamiltonians through CCSD.  Supplying the
        # view must leave the installed Fock intact; a reconstruction here
        # would reintroduce the double contraction this test is designed to
        # prevent.
        totals = []
        with mock.patch.dict(os.environ, _CLEAN_X_ENV):
            for view in (drop_zero_one, drop_two_body):
                cc = jax_xtc_ccsd.RCCSD(
                    self.mf, full_obj, self.jparams,
                    max_memory=2_000,
                    gpu_max_memory=2_000,
                    on_the_fly_vvvv=False,
                )
                cc.max_cycle = 50
                real_update_amps = cc.update_amps
                seen_eris = []

                def record_eris(t1, t2, eris):
                    seen_eris.append(eris)
                    self.assertIs(eris, view)
                    np.testing.assert_array_equal(eris.fock, view.fock)
                    return real_update_amps(t1, t2, eris)

                with mock.patch.object(cc, "update_amps", side_effect=record_eris):
                    cc.kernel(eris=view)
                self.assertGreater(len(seen_eris), 0)
                self.assertAlmostEqual(cc.e_hf, _explicit_reference_energy(view), places=12)
                self.assertTrue(np.isfinite(cc.e_tot))
                totals.append(cc.e_tot)

        self.assertGreater(abs(totals[0] - totals[1]), 1e-10)


if __name__ == "__main__":
    unittest.main()
