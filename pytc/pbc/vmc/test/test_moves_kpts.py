"""End-to-end test of the PBC ``_one_electron_move`` with a KSlaterDet.

Exercises the move's dispatch on KSlaterDet → rank1_update_one_electron_kpts
and verifies cache consistency over a sequence of MCMC proposals.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf.pbc import gto as pbcgto, scf as pbcscf, tools as pbctools

from pytc.vmc.walker import initialize_walker_state

from pytc.pbc.ansatz import create_slater_det_kpts
from pytc.pbc.vmc.moves import _one_electron_move
from pytc.pbc.vmc.walker import initialize_walkers
from pytc.pbc.ansatz.kdet import eval_kdet_value_and_grad


def _build_kpts_setup(L=4.0, nk=(2, 1, 1), n_walkers=8, key_seed=0):
    """H2 primitive + Nk supercell + KRHF + KSlaterDet + primed walker."""
    prim = pbcgto.Cell()
    prim.atom = 'H 0 0 0; H 0 0 0.7'
    prim.basis = 'sto-3g'
    prim.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    prim.unit = 'B'; prim.cart = True; prim.verbose = 0
    prim.build()

    sup = pbctools.super_cell(prim, list(nk))
    sup.cart = True; sup.verbose = 0; sup.build()

    kpts = prim.make_kpts(list(nk))
    mf = pbcscf.KRHF(prim, kpts=kpts); mf.exxdiv = None; mf.kernel()
    kdet = create_slater_det_kpts(mf, supercell=sup)

    walker = initialize_walkers(
        kdet, sup, n_walkers=n_walkers,
        key=random.PRNGKey(key_seed), log_init=False,
    )
    # Prime cache: vmap kdet over walker batch.
    batch_call = jax.vmap(lambda w: kdet(w, None))
    _, walker = batch_call(walker)
    return sup, kdet, walker


class TestKptsOneElectronMove(unittest.TestCase):
    def setUp(self):
        self.sup, self.kdet, self.walker = _build_kpts_setup()
        self.lattice = jnp.asarray(self.sup.lattice_vectors())

    def test_runs_and_produces_complex_psi(self):
        _, _, _, proposals = _one_electron_move(
            self.kdet, self.walker, step_size=0.3, key=random.PRNGKey(1),
            params=None, lattice=self.lattice,
        )
        # psi_sign must be complex (phase of det), log_psi must be real.
        self.assertTrue(jnp.iscomplexobj(proposals.psi_sign))
        self.assertEqual(proposals.log_psi.dtype, jnp.float64)
        # Positions stay in supercell.
        L = jnp.diag(self.lattice)
        self.assertTrue(bool(jnp.all(proposals.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(proposals.positions < L)))

    def test_exactly_one_electron_moves(self):
        _, _, _, proposals = _one_electron_move(
            self.kdet, self.walker, step_size=0.3, key=random.PRNGKey(2),
            params=None, lattice=self.lattice,
        )
        counts = jnp.sum(proposals.move_mask.astype(jnp.int32), axis=-1)
        np.testing.assert_array_equal(
            np.asarray(counts),
            np.full((self.walker.positions.shape[0],), 1, dtype=np.int32),
        )

    def test_rank1_psi_matches_full_recompute(self):
        """After a proposal, the cached log_psi must equal a fresh full
        eval at proposals.positions. This is the load-bearing check that
        the complex Sherman-Morrison path is consistent across batches."""
        _, new_psi_values, _, proposals = _one_electron_move(
            self.kdet, self.walker, step_size=0.3, key=random.PRNGKey(3),
            params=None, lattice=self.lattice,
        )
        sign_cached, log_cached = new_psi_values

        # Fresh full eval at the proposed positions
        fresh = initialize_walker_state(self.kdet, proposals.positions)
        _, fresh = jax.vmap(
            lambda w: eval_kdet_value_and_grad(self.kdet, w)
        )(fresh)
        np.testing.assert_allclose(
            np.asarray(log_cached), np.asarray(fresh.log_psi), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(sign_cached), np.asarray(fresh.psi_sign), atol=1e-10
        )

    def test_many_steps_stay_in_supercell(self):
        walker = self.walker
        key = random.PRNGKey(7)
        for _ in range(30):
            key, sub = random.split(key)
            _, _, _, proposals = _one_electron_move(
                self.kdet, walker, step_size=0.3, key=sub,
                params=None, lattice=self.lattice,
            )
            # Simulate acceptance of all proposals so the walker drifts.
            walker = proposals
        L = jnp.diag(self.lattice)
        self.assertTrue(bool(jnp.all(walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(walker.positions < L)))


if __name__ == '__main__':
    unittest.main()
