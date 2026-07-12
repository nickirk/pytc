import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pyscf import gto, scf

from pytc.vmc.sampling import adaptive_burn_in
from pytc.vmc.walker import initialize_walkers
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
from pytc.jastrow.bha import BoysHandyAnalytical


class TestAdaptiveBurnIn(unittest.TestCase):
    """Task #12 tier-2: burn_in() should terminate on ensemble-stability
    evidence rather than a fixed, easy-to-under-provision step count."""

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; H 0 0 1.8; H 0 0 3.6; H 0 0 5.4",
                    basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol).run()
        cls.det = SlaterDet.create(mol, mf.mo_coeff)
        ncusp = NuclearCusp.create(mol, name="ncusp")
        bha = BoysHandyAnalytical.create(mol)
        jastrow = CompositeJastrow.create([ncusp, bha])
        cls.sj = SlaterJastrow.create(mol, jastrow, [cls.det])
        cls.params = [jastrow.init_params(), jnp.ones(1)]

    def _fresh_walkers(self, n_walkers=64, seed=0):
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        return initialize_walkers(self.det, n_walkers, key=subkey), key

    def test_terminates_early_when_stable(self):
        """With loose tolerances, should stop well short of max_steps and
        report the E/Var that triggered termination."""
        walkers, key = self._fresh_walkers()
        _, history, _, _, total_steps = adaptive_burn_in(
            self.det, self.sj, walkers, self.params, step_size=0.02, key=key,
            chunk_size=20, max_steps=400, acceptance_tol=0.5,
            stability_window=2, energy_stability_atol=100.0,
            variance_stability_rtol=1.0,
        )
        self.assertLess(total_steps, 400)
        self.assertGreater(len(history), 0)
        last = history[-1]
        self.assertIsNotNone(last["mean_energy"])
        self.assertIsNotNone(last["variance"])

    def test_hits_hard_cap_without_hanging(self):
        """An impossible stability tolerance must still terminate at
        max_steps rather than loop forever."""
        walkers, key = self._fresh_walkers(seed=1)
        _, history, _, _, total_steps = adaptive_burn_in(
            self.det, self.sj, walkers, self.params, step_size=0.02, key=key,
            chunk_size=20, max_steps=60, acceptance_tol=0.5,
            stability_window=2, energy_stability_atol=0.0,
            variance_stability_rtol=0.0,
        )
        self.assertEqual(total_steps, 60)
        self.assertEqual(len(history), 3)  # 60 / chunk_size(20)

    def test_pregate_skips_evaluation_when_acceptance_out_of_band(self):
        """A near-zero acceptance tolerance on freshly-initialized (highly
        accepted) walkers should never pass the pre-gate, so every chunk's
        mean_energy/variance stay None."""
        walkers, key = self._fresh_walkers(seed=2)
        _, history, _, _, total_steps = adaptive_burn_in(
            self.det, self.sj, walkers, self.params, step_size=0.02, key=key,
            chunk_size=20, max_steps=60, acceptance_target=0.5,
            acceptance_tol=1e-6, stability_window=2,
            energy_stability_atol=100.0, variance_stability_rtol=1.0,
        )
        self.assertEqual(total_steps, 60)
        self.assertTrue(all(h["mean_energy"] is None for h in history))


if __name__ == "__main__":
    unittest.main()
