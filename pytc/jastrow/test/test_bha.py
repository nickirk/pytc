"""Tests for analytical Boys-Handy implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.jastrow.bh import BoysHandy, BHTerm
from pytc.jastrow.bha import BoysHandyAnalytical

jax.config.update("jax_enable_x64", True)


def get_h2_molecule(bond_length=1.4):
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        unit="bohr",
    )


class TestBoysHandyAnalytical(unittest.TestCase):
    def setUp(self):
        self.key = random.PRNGKey(0)
        self.mol = get_h2_molecule()
        self.terms = [[
            BHTerm(0, 0, 1, 0.5),
            BHTerm(1, 0, 0, -0.1),
            BHTerm(2, 0, 0, -0.1),
            BHTerm(2, 0, 2, 1e-5),
        ]]
        self.bh = BoysHandy.create(self.mol, terms_per_nucleus=self.terms,
                                   analytical_gradients=False)
        self.bha = BoysHandyAnalytical.create(self.mol, terms_per_nucleus=self.terms)
        self.params = self.bh.init_params(key=self.key)

    def test_compute_matches_reference(self):
        for i in range(5):
            k1, k2 = random.split(random.fold_in(self.key, i))
            r1 = random.normal(k1, (3,))
            r2 = random.normal(k2, (3,))
            # Tolerance reflects float op-ordering between the vmap-based bh
            # implementation and the broadcast-based bha implementation; both
            # compute the same quantity but in different orders.
            np.testing.assert_allclose(
                np.array(self.bha._compute(r1, r2, self.params)),
                np.array(self.bh._compute(r1, r2, self.params)),
                rtol=1e-9,
                atol=1e-9,
            )

    def test_grad_and_laplacian_match_reference(self):
        for i in range(5):
            k1, k2 = random.split(random.fold_in(self.key, i + 100))
            r1 = random.normal(k1, (3,))
            r2 = random.normal(k2, (3,))
            grad_ref, lap_ref = self.bh.get_log_grads_r1(r1, r2, self.params)
            grad_new, lap_new = self.bha.get_log_grads_r1(r1, r2, self.params)
            np.testing.assert_allclose(np.array(grad_new), np.array(grad_ref), rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(np.array(lap_new), np.array(lap_ref), rtol=1e-7, atol=1e-7)

    def test_grad_r2_matches_reference(self):
        r1 = jnp.array([0.2, -0.3, 0.5])
        r2 = jnp.array([-0.4, 0.1, -0.2])
        grad_ref, lap_ref = self.bh.get_log_grads_r2(r1, r2, self.params)
        grad_new, lap_new = self.bha.get_log_grads_r2(r1, r2, self.params)
        np.testing.assert_allclose(np.array(grad_new), np.array(grad_ref), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.array(lap_new), np.array(lap_ref), rtol=1e-7, atol=1e-7)

    def test_parameter_gradient_nan_check(self):
        """Test that optimizing parameters beyond cutoff does not yield NaNs."""
        r1 = jnp.array([10.0, 10.0, 10.0])
        r2 = jnp.array([-10.0, -10.0, -10.0])
        def energy_fn(params):
            g, l = self.bha.get_log_grads_r1(r1, r2, params)
            return jnp.sum(g) + l
            
        grads = jax.grad(energy_fn)(self.params)
        for key in grads:
            self.assertFalse(jnp.any(jnp.isnan(grads[key])), f"NaN found in parameter gradient for {key}")


class TestBoysHandyRoutingGuard(unittest.TestCase):
    """Assert the production guard routes correctly based on atom types."""

    def test_single_type_defaults_to_bha(self):
        mol = get_h2_molecule()
        j = BoysHandy.create(mol)
        self.assertIsInstance(j, BoysHandyAnalytical)

    def test_single_type_forced_bh(self):
        mol = get_h2_molecule()
        j = BoysHandy.create(mol, analytical_gradients=False)
        self.assertIs(type(j), BoysHandy)

    def test_multi_type_defaults_to_bha(self):
        mol = get_h2o_molecule()
        j = BoysHandy.create(mol)
        self.assertIsInstance(j, BoysHandyAnalytical)

    def test_multi_type_forced_bh(self):
        mol = get_h2o_molecule()
        j = BoysHandy.create(mol, analytical_gradients=False)
        self.assertIs(type(j), BoysHandy)

    def test_multi_type_explicit_bha_still_works(self):
        mol = get_h2o_molecule()
        j = BoysHandyAnalytical.create(mol)
        self.assertIs(type(j), BoysHandyAnalytical)


def get_h2o_molecule():
    return gto.M(
        atom="O 0 0 0; H 0 -1.4 1.1; H 0 1.4 1.1",
        basis="sto-3g",
        unit="bohr",
        verbose=0,
    )


class TestBoysHandyAnalyticalMultiType(unittest.TestCase):
    """Multi-atom-type equivalence and FD tests for BHA vs BH.

    The existing TestBoysHandyAnalytical validates on H2 (single atom type).
    These tests extend coverage to H2O (O + H = 2 types), exercising BHA's
    padded_nuclei_by_type / nuclei_mask_by_type machinery with default
    (production) Boys-Handy terms.
    """

    def setUp(self):
        self.key = random.PRNGKey(42)
        self.mol = get_h2o_molecule()
        self.bh = BoysHandy.create(self.mol, analytical_gradients=False)
        self.bha = BoysHandyAnalytical.create(self.mol)
        self.params = self.bh.init_params(key=self.key)

    def test_compute_matches_multi_type(self):
        for i in range(10):
            k1, k2 = random.split(random.fold_in(self.key, i))
            r1 = random.normal(k1, (3,)) * 2.0
            r2 = random.normal(k2, (3,)) * 2.0
            np.testing.assert_allclose(
                np.array(self.bha._compute(r1, r2, self.params)),
                np.array(self.bh._compute(r1, r2, self.params)),
                rtol=1e-8, atol=1e-8,
            )

    def test_grad_and_laplacian_match_multi_type(self):
        for i in range(10):
            k1, k2 = random.split(random.fold_in(self.key, i + 200))
            r1 = random.normal(k1, (3,)) * 2.0
            r2 = random.normal(k2, (3,)) * 2.0
            grad_ref, lap_ref = self.bh.get_log_grads_r1(r1, r2, self.params)
            grad_new, lap_new = self.bha.get_log_grads_r1(r1, r2, self.params)
            np.testing.assert_allclose(
                np.array(grad_new), np.array(grad_ref), rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(
                np.array(lap_new), np.array(lap_ref), rtol=1e-6, atol=1e-6)

    def test_grad_r2_matches_multi_type(self):
        r1 = jnp.array([0.3, -0.5, 0.7])
        r2 = jnp.array([-0.4, 0.1, -0.2])
        grad_ref, lap_ref = self.bh.get_log_grads_r2(r1, r2, self.params)
        grad_new, lap_new = self.bha.get_log_grads_r2(r1, r2, self.params)
        np.testing.assert_allclose(
            np.array(grad_new), np.array(grad_ref), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(
            np.array(lap_new), np.array(lap_ref), rtol=1e-6, atol=1e-6)

    def test_fd_grad_and_lap_r1_multi_type(self):
        """Central FD validation of BHA grad/lap on 2-type system (H2O)."""
        r1 = jnp.array([0.3, -0.5, 0.7])
        r2 = jnp.array([-0.4, 0.1, -0.2])
        eps = 1e-5

        def u(r1_vec):
            return float(
                np.array(self.bha._compute(r1_vec, r2, self.params)).reshape(-1)[0])

        u0 = u(r1)

        fd_grad = np.zeros(3)
        fd_second = 0.0
        for d in range(3):
            rp = np.array(r1, dtype=np.float64)
            rm = np.array(r1, dtype=np.float64)
            rp[d] += eps
            rm[d] -= eps
            fp = u(jnp.array(rp))
            fm = u(jnp.array(rm))
            fd_grad[d] = (fp - fm) / (2 * eps)
            fd_second += (fp - 2 * u0 + fm) / eps ** 2

        grad_bha, lap_bha = self.bha.get_log_grads_r1(r1, r2, self.params)
        np.testing.assert_allclose(
            np.array(grad_bha), fd_grad, atol=1e-4,
            err_msg="BHA grad != FD grad on multi-type (H2O)")
        np.testing.assert_allclose(
            float(lap_bha), fd_second, atol=1e-2,
            err_msg="BHA lap != FD lap on multi-type (H2O)")


def get_lih_molecule():
    return gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="bohr",
        verbose=0,
    )


class TestBoysHandyAnalyticalMultiTypeLiH(unittest.TestCase):
    """Second multi-type equivalence case (Li+H), distinct charge/mass ratio
    from the O+H case in TestBoysHandyAnalyticalMultiType -- extends the
    validation scoped out in d21d7ed before routing multi-type by default
    (task #5 PR-A, #pro-pytc-efficiency-refactor).
    """

    def setUp(self):
        self.key = random.PRNGKey(7)
        self.mol = get_lih_molecule()
        self.bh = BoysHandy.create(self.mol, analytical_gradients=False)
        self.bha = BoysHandyAnalytical.create(self.mol)
        self.params = self.bh.init_params(key=self.key)

    def test_compute_matches_lih(self):
        for i in range(10):
            k1, k2 = random.split(random.fold_in(self.key, i))
            r1 = random.normal(k1, (3,)) * 2.0
            r2 = random.normal(k2, (3,)) * 2.0
            np.testing.assert_allclose(
                np.array(self.bha._compute(r1, r2, self.params)),
                np.array(self.bh._compute(r1, r2, self.params)),
                rtol=1e-8, atol=1e-8,
            )

    def test_grad_and_laplacian_match_lih(self):
        # Measured max relative error over 20 random configs (default
        # 17-term basis, LiH + (H2O)2): grad 2.4e-8, lap 2.9e-8 -- roundoff
        # from op-ordering between the vmap-based bh and broadcast-based bha
        # implementations, same as the single-type case. Tolerance below
        # matches TestBoysHandyAnalyticalMultiType's H2O case for consistency.
        for i in range(10):
            k1, k2 = random.split(random.fold_in(self.key, i + 200))
            r1 = random.normal(k1, (3,)) * 2.0
            r2 = random.normal(k2, (3,)) * 2.0
            grad_ref, lap_ref = self.bh.get_log_grads_r1(r1, r2, self.params)
            grad_new, lap_new = self.bha.get_log_grads_r1(r1, r2, self.params)
            np.testing.assert_allclose(
                np.array(grad_new), np.array(grad_ref), rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(
                np.array(lap_new), np.array(lap_ref), rtol=1e-6, atol=1e-6)

    def test_fd_grad_and_lap_r1_lih(self):
        """Central FD validation of BHA grad/lap on LiH."""
        r1 = jnp.array([0.3, -0.5, 0.7])
        r2 = jnp.array([-0.4, 0.1, -0.2])
        eps = 1e-5

        def u(r1_vec):
            return float(
                np.array(self.bha._compute(r1_vec, r2, self.params)).reshape(-1)[0])

        u0 = u(r1)

        fd_grad = np.zeros(3)
        fd_second = 0.0
        for d in range(3):
            rp = np.array(r1, dtype=np.float64)
            rm = np.array(r1, dtype=np.float64)
            rp[d] += eps
            rm[d] -= eps
            fp = u(jnp.array(rp))
            fm = u(jnp.array(rm))
            fd_grad[d] = (fp - fm) / (2 * eps)
            fd_second += (fp - 2 * u0 + fm) / eps ** 2

        grad_bha, lap_bha = self.bha.get_log_grads_r1(r1, r2, self.params)
        np.testing.assert_allclose(
            np.array(grad_bha), fd_grad, atol=1e-4,
            err_msg="BHA grad != FD grad on LiH")
        np.testing.assert_allclose(
            float(lap_bha), fd_second, atol=1e-2,
            err_msg="BHA lap != FD lap on LiH")


if __name__ == "__main__":
    unittest.main()
