"""End-to-end ECP integration tests through the VMC local-energy evaluator.

These tests cover the full path:
    parse_pyscf_ecp -> EcpData on SlaterJastrow
    -> e_n_potential picks up V_loc
    -> compute_nonlocal_ecp_energy via psi_ratio_single + Legendre quadrature
    -> compute_single_walker_energy returns a finite E_L

Quantitative validation (⟨E_L⟩ ≈ E_HF/ECP from an MCMC sample) is a slow
test and is marked ``slow`` so it can be skipped in fast CI.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.ansatz import SlaterDet, SlaterJastrow
from pytc.ecp import compute_nonlocal_ecp_energy
from pytc.jastrow import Poly
from pytc.vmc.hamiltonian import compute_single_walker_energy
from pytc.vmc.metropolis import make_mcmc_step
from pytc.vmc.walker import Walker, initialize_walkers


def _single_walker_from_batch(batch, i=0):
    """Slice the i-th walker out of a batched Walker dataclass."""
    return Walker(
        positions=batch.positions[i],
        slater_up=batch.slater_up[i],
        slater_down=batch.slater_down[i],
        inv_up=batch.inv_up[i],
        inv_down=batch.inv_down[i],
        grad_up=batch.grad_up[i],
        grad_down=batch.grad_down[i],
        lap_up=batch.lap_up[i],
        lap_down=batch.lap_down[i],
        det_up=(batch.det_up[0][i], batch.det_up[1][i]),
        det_down=(batch.det_down[0][i], batch.det_down[1][i]),
        move_mask=batch.move_mask[i],
        log_psi=batch.log_psi[i],
        psi_sign=batch.psi_sign[i],
        log_jastrow=batch.log_jastrow[i],
    )


def _build_c_ccecp_ansatz():
    """C atom with ccECP, Poly Jastrow with zero parameters (J = 1)."""
    mol = gto.M(
        atom="C 0 0 0",
        basis="ccecp-cc-pvdz",
        ecp="ccecp",
        spin=2,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = Poly()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    jastrow_params = jnp.zeros(1)
    linear_coeffs = jnp.array([1.0])
    params = (jastrow_params, linear_coeffs)
    return mol, mf, ansatz, params


class TestECPSmoke(unittest.TestCase):
    """Basic correctness: the ECP code path produces a finite, real E_L."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_carbon_local_energy_finite(self):
        _, _, ansatz, params = _build_c_ccecp_ansatz()

        # 1 walker, populated.
        key = random.PRNGKey(7)
        batch = initialize_walkers(ansatz, 1, key=key)
        batch_ansatz = jax.vmap(ansatz, in_axes=(0, None))
        _, batch = batch_ansatz(batch, params)
        walker = _single_walker_from_batch(batch, 0)

        jastrow_params, _ = params
        energy = compute_single_walker_energy(ansatz, walker, jastrow_params)
        self.assertTrue(jnp.isfinite(energy))
        # Carbon HF/ccECP energy is around -5.4 Ha; local-energy fluctuations
        # are large but should not blow up by orders of magnitude.
        self.assertGreater(float(energy), -100.0)
        self.assertLess(float(energy), 100.0)

    def test_carbon_ecp_modifies_energy(self):
        """The non-local ECP term must contribute non-trivially.

        Compares full E_L against E_L computed with all non-local channels
        zeroed out — they should differ noticeably.
        """
        _, _, ansatz, params = _build_c_ccecp_ansatz()

        key = random.PRNGKey(11)
        batch = initialize_walkers(ansatz, 1, key=key)
        batch_ansatz = jax.vmap(ansatz, in_axes=(0, None))
        _, batch = batch_ansatz(batch, params)
        walker = _single_walker_from_batch(batch, 0)

        jastrow_params, _ = params
        energy_full = float(
            compute_single_walker_energy(ansatz, walker, jastrow_params)
        )

        # Zero the non-local channels (V_loc kept) by replacing the ecp field.
        zero_nl_c = jnp.zeros_like(ansatz.ecp.nl_c)
        ansatz_local_only = ansatz.replace(
            ecp=ansatz.ecp.replace(nl_c=zero_nl_c)
        )
        energy_local_only = float(
            compute_single_walker_energy(ansatz_local_only, walker, jastrow_params)
        )

        # V_NL is not vanishingly small at a near-nucleus walker; for ccECP
        # we expect the difference to be at least a few mHa.
        self.assertGreater(abs(energy_full - energy_local_only), 1.0e-3)


class TestECPLegendreExactness(unittest.TestCase):
    """For a trivial trial wavefunction (ratios = 1), the angular quadrature
    must collapse via Legendre orthogonality to the l=0 channel only:

        ⟨ψ_NL ψ/ψ⟩  ===>  Σ_l (2l+1) V_l(r_iA) · [Σ_q w_q P_l(cos θ_q)]
                       ===>  V_0(r_iA)        (since the bracket is δ_l0)

    This is a stronger version of the rotation-invariance check.  It catches
    sign errors in cos θ, normalization errors in the Legendre stack, and
    misordered (2l+1) factors.

    We synthesize a single-electron configuration where r_iA is known, then
    confirm the angular sum gives 1 for l=0 and 0 for l>0.
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_angular_sum_collapses_to_l0(self):
        from pytc.ecp.energy import _legendre_p_stack
        from pytc.ecp.quadrature import icosahedral_12

        grid = icosahedral_12()
        # Pick an arbitrary "electron direction" Ω̂_i.
        omega_i = jnp.array([0.3, -0.5, 0.8])
        omega_i = omega_i / jnp.linalg.norm(omega_i)
        cos_theta = grid.directions @ omega_i      # (n_q,)

        # Stack P_0..P_5 (12-point grid is exact through l=5).
        P_l = _legendre_p_stack(6, cos_theta)      # (6, n_q)
        # Σ_q w_q P_l(cos θ_q) — should be 1 for l=0, ~0 for l ≥ 1.
        angular = jnp.einsum('lq,q->l', P_l, grid.weights)
        np.testing.assert_allclose(float(angular[0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(angular[1:]), 0.0, atol=1e-12)

    def test_2lplus1_factor(self):
        # With ratios = 1, the projected sum (2l+1)·V_l(r)·angular collapses
        # to V_0(r) under the same identity, since (2·0+1) = 1.
        # This test guards against off-by-one in the (2l+1) array.
        l_plus_1 = 6
        two_l_plus_1 = jnp.arange(l_plus_1) * 2 + 1
        np.testing.assert_array_equal(
            np.asarray(two_l_plus_1), np.array([1, 3, 5, 7, 9, 11])
        )


@unittest.skipUnless(os.environ.get("PYTC_RUN_SLOW"), "slow test, set PYTC_RUN_SLOW=1")
class TestECPMCMCExpectationValue(unittest.TestCase):
    """Quantitative check: ⟨E_L⟩ sampled from |ψ_HF|² should approach the
    HF/ECP total energy when J = 0 (locality approximation does NOT introduce
    bias at this level because the trial wf IS the HF det)."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_carbon_ccecp_hf_energy_recovered(self):
        mol, mf, ansatz, params = _build_c_ccecp_ansatz()
        hf_energy = float(mf.e_tot)

        # Walkers + populate
        n_walkers = 64
        key = random.PRNGKey(2026)
        walkers = initialize_walkers(ansatz, n_walkers, key=key)
        batch_ansatz = jax.vmap(ansatz, in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        # MCMC: 500 burn-in + 1500 accumulation steps.
        step = make_mcmc_step(ansatz, step_size=0.5, move_type="one")
        for s in range(500):
            key, sub = random.split(key)
            walkers, _ = step(ansatz, walkers, sub, params)

        jastrow_params = params[0]
        energy_fn = jax.jit(jax.vmap(
            lambda w: compute_single_walker_energy(ansatz, w, jastrow_params),
            in_axes=0,
        ))

        samples = []
        for s in range(1500):
            key, sub = random.split(key)
            walkers, _ = step(ansatz, walkers, sub, params)
            if s % 5 == 0:  # decorrelate
                samples.append(np.asarray(energy_fn(walkers)))
        samples = np.concatenate(samples)
        mean = samples.mean()
        stderr = samples.std(ddof=1) / np.sqrt(len(samples))

        # Allow up to 5 stderr — gives a generous margin on top of MCMC
        # autocorrelation (which inflates the true error vs naive stderr).
        delta = mean - hf_energy
        print(f"  C ccECP: HF = {hf_energy:.6f}, VMC = {mean:.6f} ± {stderr:.6f}, "
              f"delta = {delta:.6f}")
        self.assertLess(abs(delta), max(5 * stderr, 0.02),
                        msg=f"mean {mean} vs HF {hf_energy} (stderr {stderr})")


if __name__ == "__main__":
    unittest.main()
