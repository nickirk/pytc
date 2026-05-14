"""Tests for pytc.ecp.parser against PySCF mol._ecp data."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pytc.ecp.parser import parse_pyscf_ecp


def _carbon_with_ccecp():
    return gto.M(
        atom="C 0 0 0",
        basis="ccecp-cc-pvdz",
        ecp="ccecp",
        spin=2,
        unit="Bohr",
    )


def _ch_with_partial_ecp():
    # C has ccECP, H does not.
    return gto.M(
        atom="C 0 0 0; H 0 0 2.0",
        basis={"C": "ccecp-cc-pvdz", "H": "sto-3g"},
        ecp={"C": "ccecp"},
        spin=1,
        unit="Bohr",
    )


def _h2_no_ecp():
    return gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
    )


class TestParseCarbon(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.mol = _carbon_with_ccecp()
        self.ecp = parse_pyscf_ecp(self.mol)

    def test_has_ecp_flag(self):
        np.testing.assert_array_equal(np.asarray(self.ecp.has_ecp), [True])

    def test_n_core(self):
        # C 1s^2 core -> n_core = 2; valence Z_eff = 4.
        np.testing.assert_array_equal(np.asarray(self.ecp.n_core), [2])
        np.testing.assert_array_equal(self.mol.atom_charges(), [4])

    def test_local_channel(self):
        # Carbon ccECP local channel terms (verified by direct pyscf inspection):
        #   n=1: zeta=14.43502, c=4.0
        #   n=2: zeta= 7.38188, c=-25.81955
        #   n=3: zeta= 8.39889, c=57.74008
        loc_n = np.asarray(self.ecp.loc_n[0])
        loc_z = np.asarray(self.ecp.loc_zeta[0])
        loc_c = np.asarray(self.ecp.loc_c[0])

        # Sort by n_power to make comparison order-independent.
        nonzero = loc_c != 0
        order = np.argsort(loc_n[nonzero])
        ns = loc_n[nonzero][order]
        zs = loc_z[nonzero][order]
        cs = loc_c[nonzero][order]
        np.testing.assert_array_equal(ns, [1, 2, 3])
        np.testing.assert_allclose(zs, [14.43502, 7.38188, 8.39889], rtol=1e-10)
        np.testing.assert_allclose(cs, [4.0, -25.81955, 57.74008], rtol=1e-10)

    def test_nonlocal_l0_only(self):
        # Carbon ccECP has a single l=0 non-local channel:
        #   n=2: zeta=7.76079, c=52.13345
        l_mask = np.asarray(self.ecp.l_mask[0])
        # l = 0 active, no other l-channels.
        self.assertTrue(bool(l_mask[0]))
        if l_mask.size > 1:
            self.assertFalse(bool(np.any(l_mask[1:])))

        nl_n = np.asarray(self.ecp.nl_n[0, 0])
        nl_z = np.asarray(self.ecp.nl_zeta[0, 0])
        nl_c = np.asarray(self.ecp.nl_c[0, 0])
        nonzero = nl_c != 0
        ns = nl_n[nonzero]
        zs = nl_z[nonzero]
        cs = nl_c[nonzero]
        np.testing.assert_array_equal(ns, [2])
        np.testing.assert_allclose(zs, [7.76079], rtol=1e-10)
        np.testing.assert_allclose(cs, [52.13345], rtol=1e-10)

    def test_r_cut_reasonable(self):
        r_c = float(np.asarray(self.ecp.r_cut)[0])
        # Expected ~1.4 Bohr for C ccECP at 1e-5 Ha (see test_radial.py).
        self.assertGreater(r_c, 1.0)
        self.assertLess(r_c, 2.5)


class TestParseMixedAtoms(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.mol = _ch_with_partial_ecp()
        self.ecp = parse_pyscf_ecp(self.mol)

    def test_has_ecp_per_atom(self):
        # C is index 0 (has ECP), H is index 1 (no ECP).
        np.testing.assert_array_equal(np.asarray(self.ecp.has_ecp), [True, False])

    def test_h_atom_is_all_zero(self):
        loc_c = np.asarray(self.ecp.loc_c[1])
        nl_c = np.asarray(self.ecp.nl_c[1])
        np.testing.assert_array_equal(loc_c, np.zeros_like(loc_c))
        np.testing.assert_array_equal(nl_c, np.zeros_like(nl_c))
        # And r_cut = 0 for atoms without non-local channels.
        self.assertEqual(float(np.asarray(self.ecp.r_cut)[1]), 0.0)

    def test_padding_shape_consistent(self):
        # All per-atom arrays have a common leading dimension n_atoms = 2.
        self.assertEqual(self.ecp.has_ecp.shape, (2,))
        self.assertEqual(self.ecp.loc_n.shape[0], 2)
        self.assertEqual(self.ecp.nl_n.shape[0], 2)
        self.assertEqual(self.ecp.l_mask.shape[0], 2)


class TestParseNoEcp(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.mol = _h2_no_ecp()
        self.ecp = parse_pyscf_ecp(self.mol)

    def test_has_ecp_all_false(self):
        np.testing.assert_array_equal(np.asarray(self.ecp.has_ecp), [False, False])

    def test_zero_coefficients(self):
        np.testing.assert_array_equal(
            np.asarray(self.ecp.loc_c), np.zeros_like(self.ecp.loc_c)
        )
        np.testing.assert_array_equal(
            np.asarray(self.ecp.nl_c), np.zeros_like(self.ecp.nl_c)
        )

    def test_n_atoms_matches(self):
        self.assertEqual(self.ecp.n_atoms, 2)


if __name__ == "__main__":
    unittest.main()
