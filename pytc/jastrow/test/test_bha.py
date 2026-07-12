"""Tests for analytical Boys-Handy implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.jastrow.jastrow import Jastrow
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
    """BoysHandy.create() always returns generic BoysHandy -- no implicit
    substitution to BoysHandyAnalytical, regardless of atom-type count or
    ECP status. Explicit choice over silent routing (Ke's direction,
    2026-07-10; reverts d21d7ed's original single-type auto-routing, not
    just task #5 PR-A's multi-type extension). BoysHandyAnalytical.create()
    is the only way to get the analytic-derivative implementation."""

    def test_single_type_always_generic(self):
        mol = get_h2_molecule()
        j = BoysHandy.create(mol)
        self.assertIs(type(j), BoysHandy)

    def test_multi_type_always_generic(self):
        mol = get_h2o_molecule()
        j = BoysHandy.create(mol)
        self.assertIs(type(j), BoysHandy)

    def test_single_type_explicit_bha_construction(self):
        mol = get_h2_molecule()
        j = BoysHandyAnalytical.create(mol)
        self.assertIs(type(j), BoysHandyAnalytical)

    def test_multi_type_explicit_bha_construction(self):
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
        self.bh = BoysHandy.create(self.mol)
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
        self.bh = BoysHandy.create(self.mol)
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


class TestBoysHandyAnalyticalPairGrid(unittest.TestCase):
    """Tests for get_pair_grid_grad_lap (task #5 PR-B, whole-electron-set
    contraction) against the Jastrow base class's default per-pair
    implementation it overrides -- called explicitly on the SAME
    BoysHandyAnalytical instance via ``Jastrow.get_pair_grid_grad_lap(bha,
    ...)`` (bypassing the override) so this is a genuine base-vs-override
    comparison, not two different objects.
    """

    def _check(self, mol, n_elec, key_seed):
        bh = BoysHandy.create(mol)
        bha = BoysHandyAnalytical.create(mol)
        params = bh.init_params()
        key = random.PRNGKey(key_seed)
        elec_coords = random.normal(key, (n_elec, 3)) * 1.5

        g_ref, l_ref = Jastrow.get_pair_grid_grad_lap(bha, elec_coords, params)
        g_new, l_new = bha.get_pair_grid_grad_lap(elec_coords, params)

        np.testing.assert_allclose(np.array(g_new), np.array(g_ref), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.array(l_new), np.array(l_ref), rtol=1e-10, atol=1e-10)

    def test_pair_grid_matches_reference_lih(self):
        self._check(get_lih_molecule(), n_elec=4, key_seed=5)

    def test_pair_grid_matches_reference_h2o(self):
        self._check(get_h2o_molecule(), n_elec=10, key_seed=6)

    def test_pair_grid_matches_reference_single_type(self):
        # Sanity: single-type systems (natom=1 type) must also work --
        # the scan degenerates to iterating over a single atom's type.
        self._check(get_h2_molecule(), n_elec=2, key_seed=7)


class TestBoysHandyAnalyticalNearCoalescence(unittest.TestCase):
    """Regression suite for the epsilon-default bug (2026-07-11,
    #proj-pytc-efficiency-refactor): ``BoysHandyAnalytical.create()``
    defaulted ``epsilon=1e-8`` while ``BoysHandy.create()`` defaults to
    ``1e-16``. ``_safe_norm``'s epsilon floors the e-e distance at
    ``sqrt(epsilon)`` -- with the mismatched default, BHA's Laplacian
    (which has an explicit ``2*f_d1/dist`` term) incorrectly PLATEAUED
    instead of diverging as two electrons approach coalescence, while BH's
    reference kept the correct ``1/r`` cusp growth. Diverged to a
    completely different value (including sign flips) from BH by
    separations as mild as 1e-4 bohr, with the tighter epsilon default
    now fixed. This suite locks BH-vs-BHA agreement across the whole
    near-coalescence regime, in both VALUES and PARAMETER GRADIENTS --
    the original validation gap: the pre-existing pair-grid test only
    checked forward values at random (not near-degenerate) configurations,
    and no test differentiated w.r.t. Jastrow params at all, so this
    exact bug shipped undetected.
    """

    EPS_SERIES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # Below this scale both BH (floor ~1e-8) and BHA (now matched) hit
    # their own shared numerical floor symmetrically -- that's expected
    # convergence of the safety mechanism itself, not a BH-vs-BHA gap,
    # so it's intentionally excluded from the regression tolerance below.

    def _make_pair(self, mol, n_elec, key_seed, eps, electron_idx=1):
        bh = BoysHandy.create(mol)
        bha = BoysHandyAnalytical.create(mol)
        self.assertEqual(bha.epsilon, bh.epsilon,
                          "BoysHandyAnalytical.create()'s default epsilon must "
                          "match BoysHandy.create()'s -- this is exactly the "
                          "regression this suite guards against.")
        params = bh.init_params()
        key = random.PRNGKey(key_seed)
        elec_coords = random.normal(key, (n_elec, 3)) * 1.5
        elec_coords = elec_coords.at[electron_idx].set(
            elec_coords[0] + jnp.array([eps, 0.0, 0.0])
        )
        return bh, bha, params, elec_coords

    def _check_values(self, mol, n_elec, key_seed):
        n = n_elec
        mask = 1.0 - jnp.eye(n)
        for eps in self.EPS_SERIES:
            bh, bha, params, ec = self._make_pair(mol, n, key_seed, eps)
            g_bh, l_bh = bh.get_pair_grid_grad_lap(ec, params)
            g_bha, l_bha = bha.get_pair_grid_grad_lap(ec, params)
            # Diagonal (i==j self-pair, r=0 exactly) is meaningless and
            # masked out by compute_jastrow_terms before use in
            # production -- mask it here too so the comparison matches
            # what actually reaches E_L.
            g_bh_m, g_bha_m = g_bh * mask[:, :, None], g_bha * mask[:, :, None]
            l_bh_m, l_bha_m = l_bh * mask, l_bha * mask
            np.testing.assert_allclose(
                np.array(g_bha_m), np.array(g_bh_m), rtol=1e-4, atol=1e-4,
                err_msg=f"grad mismatch at eps={eps:.0e}",
            )
            np.testing.assert_allclose(
                np.array(l_bha_m), np.array(l_bh_m), rtol=1e-4, atol=1e-4,
                err_msg=f"laplacian mismatch at eps={eps:.0e}",
            )

    def _check_param_grads(self, mol, n_elec, key_seed):
        for eps in self.EPS_SERIES:
            bh, bha, params, ec = self._make_pair(mol, n_elec, key_seed, eps)
            n = n_elec
            mask = 1.0 - jnp.eye(n)

            def scalar(get_fn, p):
                g, l = get_fn(ec, p)
                return jnp.sum((g * mask[:, :, None]) ** 2) + jnp.sum((l * mask) ** 2)

            _, grad_bh = jax.value_and_grad(lambda p: scalar(bh.get_pair_grid_grad_lap, p))(params)
            _, grad_bha = jax.value_and_grad(lambda p: scalar(bha.get_pair_grid_grad_lap, p))(params)
            for key_name in grad_bh:
                gb, ga = grad_bh[key_name], grad_bha[key_name]
                self.assertFalse(bool(jnp.any(jnp.isnan(ga))), f"{key_name} grad NaN at eps={eps:.0e}")
                self.assertFalse(bool(jnp.any(jnp.isinf(ga))), f"{key_name} grad Inf at eps={eps:.0e}")
                np.testing.assert_allclose(
                    np.array(ga), np.array(gb), rtol=1e-3, atol=1e-3,
                    err_msg=f"{key_name} param-gradient mismatch at eps={eps:.0e}",
                )

    def test_values_same_spin_pair_h2o(self):
        # electron_idx=1: both indices 0,1 fall in the same spin block for
        # H2O (5 up / 5 down) -- same-spin coalescence.
        self._check_values(get_h2o_molecule(), n_elec=10, key_seed=10)

    def test_values_opposite_spin_pair_h2o(self):
        # electron_idx=5: index 0 (alpha block) vs index 5 (beta block,
        # first beta electron for 5up/5down H2O) -- opposite-spin coalescence.
        n = 10
        bh = BoysHandy.create(get_h2o_molecule())
        bha = BoysHandyAnalytical.create(get_h2o_molecule())
        params = bh.init_params()
        mask = 1.0 - jnp.eye(n)
        for eps in self.EPS_SERIES:
            key = random.PRNGKey(11)
            elec_coords = random.normal(key, (n, 3)) * 1.5
            elec_coords = elec_coords.at[5].set(elec_coords[0] + jnp.array([eps, 0.0, 0.0]))
            g_bh, l_bh = bh.get_pair_grid_grad_lap(elec_coords, params)
            g_bha, l_bha = bha.get_pair_grid_grad_lap(elec_coords, params)
            np.testing.assert_allclose(
                np.array(g_bha * mask[:, :, None]), np.array(g_bh * mask[:, :, None]),
                rtol=1e-4, atol=1e-4, err_msg=f"opposite-spin grad mismatch at eps={eps:.0e}",
            )
            np.testing.assert_allclose(
                np.array(l_bha * mask), np.array(l_bh * mask),
                rtol=1e-4, atol=1e-4, err_msg=f"opposite-spin laplacian mismatch at eps={eps:.0e}",
            )

    def test_param_gradients_near_coalescence_h2o(self):
        self._check_param_grads(get_h2o_molecule(), n_elec=10, key_seed=10)

    def test_epsilon_defaults_match(self):
        """Direct guard on the actual bug: the two classes' create()
        defaults must agree, or _safe_norm's floor silently diverges
        between the reference and fast paths again."""
        bh = BoysHandy.create(get_h2o_molecule())
        bha = BoysHandyAnalytical.create(get_h2o_molecule())
        self.assertEqual(bh.epsilon, bha.epsilon)


if __name__ == "__main__":
    unittest.main()
