"""Tests for freezing selected Jastrow parameters during optimization."""

from types import SimpleNamespace
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from pytc.vmc.optimization import make_opt_update_step, optimize
from pytc.vmc.optimizer import apply_gradient_mask, create_gradient_mask


class FrozenJastrow:
    pass


class TrainableJastrow:
    name = "trainable"


def _ansatz():
    jastrow = SimpleNamespace(jastrows=[FrozenJastrow(), TrainableJastrow()])
    return SimpleNamespace(jastrow=jastrow)


def _params():
    return [
        [
            {"coefficient": jnp.array([1.0, 2.0])},
            {"coefficient": jnp.array([3.0])},
        ],
        jnp.array([4.0]),
    ]


class TestGradientMask(unittest.TestCase):
    def test_freezes_only_selected_jastrow(self):
        params = _params()
        grads = jax.tree_util.tree_map(jnp.ones_like, params)

        mask = create_gradient_mask(_ansatz(), params, ["FrozenJastrow"])
        masked_grads = apply_gradient_mask(grads, mask)

        np.testing.assert_array_equal(masked_grads[0][0]["coefficient"], 0.0)
        np.testing.assert_array_equal(masked_grads[0][1]["coefficient"], 1.0)
        np.testing.assert_array_equal(masked_grads[1], 1.0)

    def test_optax_step_keeps_frozen_parameters_unchanged(self):
        params = _params()
        mask = create_gradient_mask(_ansatz(), params, [0])

        def loss_fn(current_params, _):
            leaves = jax.tree_util.tree_leaves(current_params)
            return sum(jnp.sum(leaf**2) for leaf in leaves), ()

        optimizer = optax.sgd(learning_rate=0.1)
        opt_step = make_opt_update_step(loss_fn, optimizer, gradient_mask=mask)
        opt_state = optimizer.init(params)

        new_params, _, _, _ = opt_step(
            None, params, None, opt_state, jax.random.PRNGKey(0)
        )

        np.testing.assert_array_equal(
            new_params[0][0]["coefficient"], params[0][0]["coefficient"]
        )
        np.testing.assert_allclose(
            new_params[0][1]["coefficient"], jnp.array([2.4])
        )
        np.testing.assert_allclose(new_params[1], jnp.array([3.2]))

    def test_lion_weight_decay_cannot_move_frozen_parameters(self):
        params = _params()
        mask = create_gradient_mask(_ansatz(), params, [0])

        def loss_fn(current_params, _):
            leaves = jax.tree_util.tree_leaves(current_params)
            return sum(jnp.sum(leaf**2) for leaf in leaves), ()

        optimizer = optax.lion(learning_rate=0.1, weight_decay=0.01)
        opt_step = make_opt_update_step(loss_fn, optimizer, gradient_mask=mask)
        opt_state = optimizer.init(params)

        new_params, _, _, _ = opt_step(
            None, params, None, opt_state, jax.random.PRNGKey(0)
        )

        np.testing.assert_array_equal(
            new_params[0][0]["coefficient"], params[0][0]["coefficient"]
        )
        self.assertFalse(
            np.array_equal(
                new_params[0][1]["coefficient"],
                params[0][1]["coefficient"],
            )
        )

    def test_empty_frozen_params_does_not_create_mask(self):
        self.assertIsNone(create_gradient_mask(_ansatz(), _params(), []))

    def test_mask_preserves_tuple_parameter_structure(self):
        list_params = _params()
        params = (tuple(list_params[0]), list_params[1])
        grads = jax.tree_util.tree_map(jnp.ones_like, params)

        mask = create_gradient_mask(_ansatz(), params, [0])

        self.assertIsInstance(mask, tuple)
        self.assertIsInstance(mask[0], tuple)
        apply_gradient_mask(grads, mask)

    def test_unknown_frozen_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown frozen Jastrow parameter"):
            create_gradient_mask(_ansatz(), _params(), ["typo"])

    def test_newton_rejects_frozen_params_before_setup(self):
        with self.assertRaisesRegex(NotImplementedError, "Newton optimizer"):
            optimize(_ansatz(), optimizer_type="newton", frozen_params=[0])


class TestPublicOptimizeFrozenParams(unittest.TestCase):
    """Public-path regression for the original production defect: if the
    gradient_mask wiring inside ``optimize()`` is removed, the Optax path
    silently trains frozen leaves — and until this test existed, every
    other test in this file still passed.  This test exercises the public
    ``optimize(..., frozen_params=...)`` call end to end on a real ansatz
    and must fail the moment that wiring is dropped."""

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf
        from pytc.ansatz.sj import SlaterJastrow
        from pytc.ansatz.det import SlaterDet
        from pytc.jastrow import NuclearCusp, CompositeJastrow
        from pytc.jastrow.bha import BoysHandyAnalytical

        mol = gto.M(atom="H 0 0 0; H 0 0 1.8; H 0 0 3.6; H 0 0 5.4",
                    basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol).run()
        det = SlaterDet.create(mol, mf.mo_coeff)
        ncusp = NuclearCusp.create(mol, name="ncusp")
        bha = BoysHandyAnalytical.create(mol)
        jastrow = CompositeJastrow.create([ncusp, bha])
        cls.sj = SlaterJastrow.create(mol, jastrow, [det])

    def test_public_optimize_keeps_frozen_factor_fixed(self):
        initial = [self.sj.jastrow.init_params(), jnp.ones(1)]
        frozen_before = jax.tree_util.tree_map(
            lambda x: np.asarray(x).copy(), initial[0][0])
        result = optimize(
            self.sj, n_walkers=16, n_steps=8, burn_in_steps=8,
            n_opt_steps=3, learning_rate=0.05, optimizer_type="adam",
            frozen_params=[0], params=initial, adaptive_step_size=False,
        )
        final = result["params"][-1]

        # The frozen factor's parameters must be bitwise unchanged after
        # real optimization steps on the public path.
        frozen_after = jax.tree_util.tree_map(np.asarray, final[0][0])
        for key in frozen_before:
            np.testing.assert_array_equal(
                frozen_after[key], frozen_before[key],
                err_msg=f"frozen leaf {key!r} moved on the public optimize() path")

        # The trainable factor and the linear coefficients must have moved
        # (otherwise the test would also pass with optimization broken).
        bha_moved = any(
            not np.array_equal(np.asarray(final[0][1][k]),
                               np.asarray(initial[0][1][k]))
            for k in final[0][1])
        self.assertTrue(bha_moved, "trainable Jastrow factor did not update")
        self.assertFalse(
            bool(np.array_equal(np.asarray(final[1]), np.asarray(initial[1]))),
            "linear coefficients did not update")


if __name__ == "__main__":
    unittest.main()
