"""Tests for the per-atom ECP gate on NuclearCusp.

Production QMC codes (QMCPACK, CASINO, TurboRVB, QWalk) all enforce
``cusp = 0`` at ECP centers; the standard reference is Drummond, Towler, Needs
PRB 70, 235119 (2004) §III.C.  These tests verify that pytc's NuclearCusp
implements the same gate: the per-nucleus contribution must be identically
zero for any atom carrying an ECP.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pytc.jastrow import NuclearCusp


class TestNuclearCuspEcpGating(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_ecp_atoms_flagged(self):
        # H2O with BFD on O only: index 0 is O (ECP), 1 and 2 are H (AE).
        mol = gto.M(
            atom="O 0 0 0; H 1.43 0 -1.11; H -1.43 0 -1.11",
            basis={"O": "bfd-vdz", "H": "bfd-vdz"},
            ecp={"O": "bfd"},
            unit="Bohr",
            spin=0,
        )
        cusp = NuclearCusp.create(mol)
        np.testing.assert_array_equal(
            np.asarray(cusp.has_ecp_per_atom), [True, False, False]
        )

    def test_all_electron_unchanged(self):
        # No ECP at all: every atom should be False.
        mol = gto.M(
            atom="O 0 0 0; H 1.43 0 -1.11; H -1.43 0 -1.11",
            basis="sto-3g",
            unit="Bohr",
            spin=0,
        )
        cusp = NuclearCusp.create(mol)
        np.testing.assert_array_equal(
            np.asarray(cusp.has_ecp_per_atom), [False, False, False]
        )

    def test_ecp_atom_zero_contribution(self):
        """The per-nucleus contribution at an ECP center must be zero for any
        electron position, including positions inside the cusp region rc."""
        mol = gto.M(
            atom="O 0 0 0; H 1.43 0 -1.11; H -1.43 0 -1.11",
            basis={"O": "bfd-vdz", "H": "bfd-vdz"},
            ecp={"O": "bfd"},
            unit="Bohr",
            spin=0,
        )
        cusp = NuclearCusp.create(mol)
        params = cusp.init_params()

        # Compare against an "all-electron variant" with the ECP flag forced
        # to False on every atom — the cusp value at electron positions close
        # to O should differ by the O-centered contribution.
        all_ae = cusp.replace(
            has_ecp_per_atom=jnp.zeros_like(cusp.has_ecp_per_atom)
        )

        # Probe several electron positions: close to O (inside rc), close to H,
        # and far away.  r2 is a fixed companion electron position, irrelevant
        # for cusp.
        r2 = jnp.array([5.0, 5.0, 5.0])
        probe_positions_near_O = [
            jnp.array([0.05, 0.0, 0.0]),
            jnp.array([0.0, 0.1, 0.0]),
            jnp.array([0.0, 0.0, 0.15]),
        ]
        probe_position_near_H = jnp.array([1.43, 0.0, -1.0])
        probe_position_far = jnp.array([10.0, 10.0, 10.0])

        # 1. Near O: cusp-gated value must differ from all-AE value
        #    (and gated value should be exactly the H1+H2 contribution).
        for r1 in probe_positions_near_O:
            gated = float(cusp._compute(r1, r2, params))
            ae = float(all_ae._compute(r1, r2, params))
            self.assertNotAlmostEqual(
                gated, ae, places=6,
                msg=f"O-cusp gate had no effect at r1 = {r1}",
            )

        # 2. Near H: gating O should not change the result (H is unaffected).
        gated_h = float(cusp._compute(probe_position_near_H, r2, params))
        ae_h = float(all_ae._compute(probe_position_near_H, r2, params))
        self.assertAlmostEqual(gated_h, ae_h, places=10)

        # 3. Far from any nucleus: both should be zero.
        gated_far = float(cusp._compute(probe_position_far, r2, params))
        ae_far = float(all_ae._compute(probe_position_far, r2, params))
        self.assertAlmostEqual(gated_far, 0.0, places=10)
        self.assertAlmostEqual(ae_far, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
