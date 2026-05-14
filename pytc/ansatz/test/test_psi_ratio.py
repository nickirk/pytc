"""Tests for slater_ratio_single and psi_ratio_single.

Both ratio APIs are used to query psi(R')/psi(R) under a one-electron move
without mutating the walker.  They are correctness-equivalent to recomputing
psi from scratch, just cheaper.  These tests verify that equivalence on real
ansatzes.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc.ansatz import SlaterDet, SlaterJastrow
from pytc.ansatz.det import slater_ratio_single
from pytc.ansatz.test.test_sj import create_test_walker
from pytc.jastrow import Poly


def _populated_walker(ansatz, positions, params):
    """Build a walker and run the ansatz once to populate caches."""
    walker = create_test_walker(positions, ansatz.dets[0])
    _, populated = ansatz(walker, params)
    return populated


def _setup_h2_singlet():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = Poly()
    jastrow_params = jnp.array([0.5])
    linear_coeffs = jnp.array([1.0])
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    params = (jastrow_params, linear_coeffs)
    return mol, ansatz, params


def _setup_lih_singlet():
    """LiH singlet: n_alpha = n_beta = 2 — RHF, both channels nontrivial."""
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 3.0",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = Poly()
    jastrow_params = jnp.array([0.3])
    linear_coeffs = jnp.array([1.0])
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    params = (jastrow_params, linear_coeffs)
    return mol, ansatz, params


class TestSlaterRatioSingle(unittest.TestCase):
    """slater_ratio_single must equal det(S')/det(S) from scratch."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def _check_ratio(self, ansatz, params, positions, electron_idx, new_pos):
        walker = _populated_walker(ansatz, positions, params)
        ratio = slater_ratio_single(
            ansatz.dets[0], walker, electron_idx, new_pos
        )
        # Reference: rebuild S' from scratch.
        new_positions = positions.at[electron_idx].set(new_pos)
        new_walker = _populated_walker(ansatz, new_positions, params)
        # det(S)/det(S) old:
        sign_up_old, logdet_up_old = walker.det_up
        sign_dn_old, logdet_dn_old = walker.det_down
        sign_up_new, logdet_up_new = new_walker.det_up
        sign_dn_new, logdet_dn_new = new_walker.det_down
        ref_ratio = (
            (sign_up_new * sign_dn_new)
            * jnp.exp(
                (logdet_up_new + logdet_dn_new)
                - (logdet_up_old + logdet_dn_old)
            )
            / (sign_up_old * sign_dn_old)
        )
        np.testing.assert_allclose(
            float(ratio), float(ref_ratio), rtol=1e-10, atol=1e-12
        )

    def test_h2_singlet_alpha_move(self):
        _, ansatz, params = _setup_h2_singlet()
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        self._check_ratio(ansatz, params, positions, 0, jnp.array([0.1, 0.0, 0.5]))

    def test_h2_singlet_beta_move(self):
        _, ansatz, params = _setup_h2_singlet()
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        self._check_ratio(ansatz, params, positions, 1, jnp.array([0.0, 0.2, 0.9]))

    def test_lih_alpha_moves(self):
        _, ansatz, params = _setup_lih_singlet()
        # LiH: n_alpha = n_beta = 2 (indices 0,1 alpha; 2,3 beta).
        positions = jnp.array(
            [[0.0, 0.0, 0.5],
             [0.3, 0.0, 0.0],
             [0.0, 0.4, 2.7],
             [0.0, -0.2, 3.1]]
        )
        self._check_ratio(ansatz, params, positions, 0, jnp.array([0.1, 0.2, 0.7]))
        self._check_ratio(ansatz, params, positions, 1, jnp.array([-0.3, 0.0, 0.4]))

    def test_lih_beta_moves(self):
        _, ansatz, params = _setup_lih_singlet()
        positions = jnp.array(
            [[0.0, 0.0, 0.5],
             [0.3, 0.0, 0.0],
             [0.0, 0.4, 2.7],
             [0.0, -0.2, 3.1]]
        )
        self._check_ratio(ansatz, params, positions, 2, jnp.array([0.0, -0.5, 2.6]))
        self._check_ratio(ansatz, params, positions, 3, jnp.array([0.1, 0.3, 3.4]))


class TestPsiRatioSingle(unittest.TestCase):
    """psi_ratio_single must equal psi(R')/psi(R) from scratch."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def _check_ratio(self, ansatz, params, positions, electron_idx, new_pos):
        # Old psi
        walker_old = _populated_walker(ansatz, positions, params)
        sign_old, logabs_old = walker_old.psi_sign, walker_old.log_psi

        # New psi from scratch
        new_positions = positions.at[electron_idx].set(new_pos)
        walker_new = _populated_walker(ansatz, new_positions, params)
        sign_new, logabs_new = walker_new.psi_sign, walker_new.log_psi

        ref_ratio = (sign_new / sign_old) * jnp.exp(logabs_new - logabs_old)

        ratio = ansatz.psi_ratio_single(walker_old, electron_idx, new_pos, params)

        np.testing.assert_allclose(
            float(ratio), float(ref_ratio), rtol=1e-9, atol=1e-11,
            err_msg=(
                f"electron_idx={electron_idx}: "
                f"got {float(ratio)}, ref {float(ref_ratio)}"
            ),
        )

    def test_h2_singlet_both_channels(self):
        _, ansatz, params = _setup_h2_singlet()
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        self._check_ratio(ansatz, params, positions, 0, jnp.array([0.1, 0.0, 0.5]))
        self._check_ratio(ansatz, params, positions, 1, jnp.array([0.0, 0.2, 0.9]))

    def test_h2_no_move_returns_one(self):
        # Moving an electron to its current position must give ratio 1.
        _, ansatz, params = _setup_h2_singlet()
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        walker = _populated_walker(ansatz, positions, params)
        for i in range(positions.shape[0]):
            r = ansatz.psi_ratio_single(walker, i, positions[i], params)
            np.testing.assert_allclose(float(r), 1.0, rtol=1e-10, atol=1e-12)

    def test_lih_all_electrons(self):
        _, ansatz, params = _setup_lih_singlet()
        positions = jnp.array(
            [[0.0, 0.0, 0.5],
             [0.3, 0.0, 0.0],
             [0.0, 0.4, 2.7],
             [0.0, -0.2, 3.1]]
        )
        for i, new_pos in [
            (0, jnp.array([0.1, 0.2, 0.7])),
            (1, jnp.array([-0.3, 0.0, 0.4])),
            (2, jnp.array([0.0, -0.5, 2.6])),
            (3, jnp.array([0.1, 0.3, 3.4])),
        ]:
            self._check_ratio(ansatz, params, positions, i, new_pos)

    def test_walker_unchanged_after_call(self):
        # psi_ratio_single must not mutate the input walker.
        _, ansatz, params = _setup_h2_singlet()
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        walker = _populated_walker(ansatz, positions, params)
        inv_up_before = walker.inv_up
        slater_up_before = walker.slater_up
        log_psi_before = walker.log_psi
        log_jastrow_before = walker.log_jastrow

        _ = ansatz.psi_ratio_single(walker, 0, jnp.array([0.5, 0.5, 0.5]), params)

        np.testing.assert_array_equal(np.asarray(walker.inv_up), np.asarray(inv_up_before))
        np.testing.assert_array_equal(np.asarray(walker.slater_up), np.asarray(slater_up_before))
        self.assertEqual(float(walker.log_psi), float(log_psi_before))
        self.assertEqual(float(walker.log_jastrow), float(log_jastrow_before))


class TestPsiRatioMultiDetRaises(unittest.TestCase):
    """Multi-determinant ansatzes are intentionally rejected in v1."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_raises(self):
        _, ansatz, params = _setup_h2_singlet()
        # Force a fake two-det ansatz by replicating the determinant.
        fake = ansatz.replace(dets=[ansatz.dets[0], ansatz.dets[0]])
        positions = jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, 1.1]])
        walker = _populated_walker(ansatz, positions, params)
        # Use two linear coeffs since the ansatz now has two dets.
        bad_params = (params[0], jnp.array([1.0, 0.0]))
        with self.assertRaises(NotImplementedError):
            fake.psi_ratio_single(walker, 0, jnp.array([0.5, 0.5, 0.5]), bad_params)


if __name__ == "__main__":
    unittest.main()
