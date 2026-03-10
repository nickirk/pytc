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
        self.bh = BoysHandy.create(self.mol, terms_per_nucleus=self.terms)
        self.bha = BoysHandyAnalytical.create(self.mol, terms_per_nucleus=self.terms)
        self.params = self.bh.init_params(key=self.key)

    def test_compute_matches_reference(self):
        for i in range(5):
            k1, k2 = random.split(random.fold_in(self.key, i))
            r1 = random.normal(k1, (3,))
            r2 = random.normal(k2, (3,))
            np.testing.assert_allclose(
                np.array(self.bha._compute(r1, r2, self.params)),
                np.array(self.bh._compute(r1, r2, self.params)),
                rtol=1e-12,
                atol=1e-12,
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


if __name__ == "__main__":
    unittest.main()
