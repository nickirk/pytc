"""Tests for the ECP-sphere overlap warning.

When two ECP atoms sit closer than the sum of their non-local cutoff radii,
the trial-wavefunction representation in the overlap region is typically
poor.  ``parse_pyscf_ecp`` (and via it, ``SlaterJastrow.create``) emits a
``RuntimeWarning`` for each offending atom pair.
"""

import unittest
import warnings

import jax
from pyscf import gto

from pytc.ecp.parser import parse_pyscf_ecp


class TestEcpOverlapWarning(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_co_at_equilibrium_triggers(self):
        # C and O both ECP'd; bond ~2.13 Bohr; r_cut sum ~2.6 Bohr.
        mol = gto.M(
            atom="C 0 0 0; O 0 0 2.132",
            basis="ccecp-cc-pvdz",
            ecp="ccecp",
            spin=0, unit="Bohr", verbose=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_pyscf_ecp(mol)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(
            len(runtime) >= 1,
            "Expected at least one RuntimeWarning for CO overlap, got none.",
        )
        msg = str(runtime[0].message)
        self.assertIn("Overlapping ECP", msg)
        self.assertIn("C", msg)
        self.assertIn("O", msg)

    def test_co_far_separated_does_not_trigger(self):
        # Push C-O to 8 Bohr; no overlap expected.
        mol = gto.M(
            atom="C 0 0 0; O 0 0 8.0",
            basis="ccecp-cc-pvdz",
            ecp="ccecp",
            spin=0, unit="Bohr", verbose=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_pyscf_ecp(mol)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(
            len(runtime), 0,
            f"Expected no warning at 8 Bohr CO separation; got: "
            f"{[str(w.message) for w in runtime]}",
        )

    def test_h2o_bfd_does_not_trigger(self):
        # H2O/BFD: only O is ECP'd; H is all-electron.  Cross-pair O-H
        # cannot overlap because H has r_cut = 0 (no ECP).  Should not warn.
        mol = gto.M(
            atom="O 0 0 0; H 1.4307 0 -1.1075; H -1.4307 0 -1.1075",
            basis={"O": "bfd-vdz", "H": "bfd-vdz"},
            ecp={"O": "bfd"},
            spin=0, unit="Bohr", verbose=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_pyscf_ecp(mol)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 0)

    def test_warn_on_overlap_flag_disables(self):
        # Same CO that triggers — but with warn_on_overlap=False.
        mol = gto.M(
            atom="C 0 0 0; O 0 0 2.132",
            basis="ccecp-cc-pvdz",
            ecp="ccecp",
            spin=0, unit="Bohr", verbose=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_pyscf_ecp(mol, warn_on_overlap=False)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 0)


if __name__ == "__main__":
    unittest.main()
