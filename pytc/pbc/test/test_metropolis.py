import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.ansatz.sj import SlaterJastrow
from pytc.jastrow import CompositeJastrow

from pytc.pbc.ansatz import create_slater_det
from pytc.pbc.jastrow import NuclearCusp, BoysHandy
from pytc.pbc.vmc import (
    initialize_walkers,
    make_ewald_params,
    metropolis_hastings,
    make_mcmc_step,
    compute_single_walker_energy,
)


def _build_pbc_setup(L=6.0, n_walkers=16, n_radial=200):
    """Build a small PBC H2 system: Cell, SJ, walkers (caches primed), ewald."""
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    mf = pbcscf.RHF(cell)
    mf.exxdiv = None
    mf.kernel()

    det = create_slater_det(cell, mo_coeff=mf.mo_coeff)
    ncusp = NuclearCusp.create(cell, name='ncusp', n_radial=n_radial)
    bh = BoysHandy.create(cell, name='bh')
    jastrow = CompositeJastrow.create([ncusp, bh])
    sj = SlaterJastrow.create(cell, jastrow=jastrow, dets=[det])
    params = sj.init_params(random.PRNGKey(0))

    walker = initialize_walkers(
        sj.dets[0], cell, n_walkers=n_walkers,
        key=random.PRNGKey(1), log_init=False,
    )
    # Prime the walker cache so the rank-1 move can read log_psi etc.
    batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
    _, walker = batch_ansatz(walker, params)

    ewald = make_ewald_params(cell.lattice_vectors())
    return cell, sj, params, walker, ewald


class TestMetropolisStep(unittest.TestCase):
    def setUp(self):
        self.cell, self.sj, self.params, self.walker, self.ewald = _build_pbc_setup()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_single_step_returns_walker_and_rate(self):
        new_walker, acc = metropolis_hastings(
            self.sj, self.walker, step_size=0.3, key=random.PRNGKey(2),
            params=self.params, lattice=self.lattice,
        )
        self.assertEqual(new_walker.positions.shape, self.walker.positions.shape)
        self.assertTrue(0.0 <= float(acc) <= 1.0)

    def test_walkers_stay_inside_cell_after_step(self):
        new_walker, _ = metropolis_hastings(
            self.sj, self.walker, step_size=2.0, key=random.PRNGKey(3),
            params=self.params, lattice=self.lattice,
        )
        L = jnp.diag(self.lattice)
        self.assertTrue(bool(jnp.all(new_walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(new_walker.positions < L)))

    def test_walkers_stay_inside_cell_after_many_steps(self):
        """Hundreds of MCMC steps must not drift walkers outside the cell."""
        walker = self.walker
        key = random.PRNGKey(4)
        L = jnp.diag(self.lattice)
        for _ in range(50):
            key, sub = random.split(key)
            walker, _ = metropolis_hastings(
                self.sj, walker, step_size=0.4, key=sub,
                params=self.params, lattice=self.lattice,
            )
        self.assertTrue(bool(jnp.all(walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(walker.positions < L)))

    def test_log_psi_consistent_after_step(self):
        """After a step, cached log_psi on the new walker must match a fresh
        evaluation at its positions — this catches rank-1 drift."""
        new_walker, _ = metropolis_hastings(
            self.sj, self.walker, step_size=0.3, key=random.PRNGKey(5),
            params=self.params, lattice=self.lattice,
        )
        batch_ansatz = jax.vmap(lambda w, p: self.sj(w, p), in_axes=(0, None))
        from pytc.vmc.walker import initialize_walker_state
        fresh = initialize_walker_state(self.sj.dets[0], new_walker.positions)
        (sign_fresh, logabs_fresh), _ = batch_ansatz(fresh, self.params)
        np.testing.assert_allclose(new_walker.log_psi, logabs_fresh, atol=1e-10)
        np.testing.assert_allclose(new_walker.psi_sign, sign_fresh, atol=1e-10)


class TestMakeMcmcStep(unittest.TestCase):
    def setUp(self):
        self.cell, self.sj, self.params, self.walker, self.ewald = _build_pbc_setup()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_factory_jit_compiles_and_runs(self):
        step = make_mcmc_step(self.sj, step_size=0.3, lattice=self.lattice)
        new_walker, acc = step(self.sj, self.walker, random.PRNGKey(6), self.params)
        self.assertEqual(new_walker.positions.shape, self.walker.positions.shape)
        self.assertTrue(np.isfinite(float(acc)))

    def test_factory_runs_in_loop(self):
        step = make_mcmc_step(self.sj, step_size=0.3, lattice=self.lattice)
        walker = self.walker
        key = random.PRNGKey(7)
        accs = []
        for _ in range(20):
            key, sub = random.split(key)
            walker, acc = step(self.sj, walker, sub, self.params)
            accs.append(float(acc))
        # Acceptance should hover in a sensible range for step_size=0.3.
        self.assertGreater(np.mean(accs), 0.2)
        self.assertLess(np.mean(accs), 0.95)

    def test_invalid_move_type_raises(self):
        with self.assertRaises(ValueError):
            make_mcmc_step(self.sj, step_size=0.3, lattice=self.lattice, move_type="bogus")


class TestEndToEnd(unittest.TestCase):
    """A short MCMC run + local-energy evaluation. This is the canonical
    smoke test that ties together walker init, PBC moves, Ewald, and the
    local-energy machinery."""

    def test_burnin_and_sample(self):
        cell, sj, params, walker, ewald = _build_pbc_setup(L=6.0, n_walkers=32)
        lattice = jnp.asarray(cell.lattice_vectors())
        step = make_mcmc_step(sj, step_size=0.4, lattice=lattice)

        key = random.PRNGKey(0)
        # Burn-in
        for _ in range(30):
            key, sub = random.split(key)
            walker, _ = step(sj, walker, sub, params)

        # Sample-and-measure
        energies = []
        for _ in range(20):
            key, sub = random.split(key)
            walker, _ = step(sj, walker, sub, params)
            es = jax.vmap(
                lambda w: compute_single_walker_energy(sj, w, params[0], ewald)
            )(walker)
            energies.append(np.asarray(es))

        energies = np.concatenate(energies)
        # Mean energy must be finite and within a sane window for H2 in a
        # 6 Bohr cell (HF total is ~-1 Ha; finite cell corrections and
        # partial Jastrow optimization push this around).
        mean_e = float(np.mean(energies))
        self.assertTrue(np.isfinite(mean_e))
        self.assertGreater(mean_e, -5.0)
        self.assertLess(mean_e, 5.0)


if __name__ == '__main__':
    unittest.main()
