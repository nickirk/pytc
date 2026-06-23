"""Finite-difference safety-net test for compute_jastrow_terms.

This test pins grad_J_over_J and lap_J_over_J against central finite
differences of the total Jastrow exponent U. It is the contract that
every Jastrow refactor must preserve.

Math:
  U(r1..rN) = 0.5 * sum_{i!=j} u(r_i, r_j)
  grad_J_over_J[k] = dU/d(r_k)
  lap_J_over_J[k]  = d²U/d(r_k)² + |dU/d(r_k)|²

Uses a SimplePoly Jastrow (u = a*|r1-r2|²) with well-defined derivatives
everywhere, exercising the folx-based base-class get_log_grads_r1 path.
Finite-difference reference computed in float64 via numpy.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from pytc.jastrow import jastrow as jastrow_mod
from pytc.vmc.hamiltonian import compute_jastrow_terms


@struct.dataclass
class SimplePolyJastrow(jastrow_mod.Jastrow):
    name: str = struct.field(pytree_node=False, default='simple_poly')

    def _compute(self, r1, r2, params):
        diff = r1 - r2
        return params['a'] * jnp.sum(diff * diff)

    def init_params(self, a=0.5):
        return {'a': jnp.array(a)}


class _MockSJ:
    def __init__(self, jastrow):
        self.jastrow = jastrow


class TestComputeJastrowTermsFiniteDiff(unittest.TestCase):

    def setUp(self):
        self.jastrow = SimplePolyJastrow()
        self.params = {'a': jnp.array(0.5)}
        self.sj = _MockSJ(self.jastrow)
        key = jax.random.PRNGKey(42)
        self.coords = jax.random.uniform(key, (4, 3), minval=-2.0, maxval=2.0)
        self.coords_np = np.asarray(self.coords, dtype=np.float64)
        self.a = 0.5

    def _total_U_np(self, coords_np):
        """Total Jastrow exponent in float64 numpy."""
        a = self.a
        N = coords_np.shape[0]
        U = 0.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = coords_np[i] - coords_np[j]
                U += 0.5 * a * np.sum(diff ** 2)
        return U

    def test_grad_finite_diff(self):
        grad_J, _ = compute_jastrow_terms(self.sj, self.coords, self.params)

        eps = 1e-5
        N = self.coords_np.shape[0]
        fd_grad = np.zeros((N, 3))
        for k in range(N):
            for d in range(3):
                plus = self.coords_np.copy()
                minus = self.coords_np.copy()
                plus[k, d] += eps
                minus[k, d] -= eps
                fd_grad[k, d] = (
                    self._total_U_np(plus) - self._total_U_np(minus)) / (2 * eps)

        np.testing.assert_allclose(
            np.asarray(grad_J), fd_grad, atol=1e-4,
            err_msg="grad_J_over_J != finite-difference dU/d(r_k)")

    def test_lap_finite_diff(self):
        _, lap_J = compute_jastrow_terms(self.sj, self.coords, self.params)

        eps = 1e-4
        N = self.coords_np.shape[0]
        U0 = self._total_U_np(self.coords_np)
        fd_second = np.zeros(N)
        fd_grad_sq = np.zeros(N)
        for k in range(N):
            grad_k = np.zeros(3)
            for d in range(3):
                plus = self.coords_np.copy()
                minus = self.coords_np.copy()
                plus[k, d] += eps
                minus[k, d] -= eps
                U_plus = self._total_U_np(plus)
                U_minus = self._total_U_np(minus)
                fd_second[k] += (U_plus - 2 * U0 + U_minus) / eps ** 2
                grad_k[d] = (U_plus - U_minus) / (2 * eps)
            fd_grad_sq[k] = np.sum(grad_k ** 2)

        fd_lap_J = fd_second + fd_grad_sq

        np.testing.assert_allclose(
            np.asarray(lap_J), fd_lap_J, atol=1e-2,
            err_msg="lap_J_over_J != finite-difference d²U/dr² + |dU/dr|²")

    def test_zero_jastrow_for_single_electron(self):
        coords = jnp.array([[0.0, 0.0, 0.0]])
        grad_J, lap_J = compute_jastrow_terms(self.sj, coords, self.params)
        np.testing.assert_allclose(np.asarray(grad_J), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(lap_J), 0.0, atol=1e-10)

    def test_analytic_poly_values(self):
        """For u=a*|r1-r2|²: grad_k=2a*Σ_{j≠k}(r_k-r_j), lap_second=6a*(N-1)."""
        a = self.a
        N = self.coords_np.shape[0]
        c = self.coords_np

        expected_grad = np.zeros((N, 3))
        expected_lap_second = np.zeros(N)
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                expected_grad[k] += 2 * a * (c[k] - c[j])
                expected_lap_second[k] += 6 * a
        expected_lap = expected_lap_second + np.sum(expected_grad ** 2, axis=1)

        grad_J, lap_J = compute_jastrow_terms(self.sj, self.coords, self.params)
        np.testing.assert_allclose(np.asarray(grad_J), expected_grad, atol=1e-4)
        np.testing.assert_allclose(np.asarray(lap_J), expected_lap, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
