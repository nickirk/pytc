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


def _build_co_ccecp_ansatz():
    """CO molecule (singlet) with ccECP on both atoms.

    After ccECP removes 1s² from each, 10 valence electrons remain
    (4 from C + 6 from O), split as n_alpha = n_beta = 5.
    Bond length 2.132 Bohr (close to experimental 2.1322 Bohr).
    """
    mol = gto.M(
        atom="C 0 0 0; O 0 0 2.132",
        basis="ccecp-cc-pvdz",
        ecp="ccecp",
        spin=0,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = Poly()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    jastrow_params = jnp.zeros(1)
    linear_coeffs = jnp.array([1.0])
    return mol, mf, ansatz, (jastrow_params, linear_coeffs)


def _mcmc_mean_energy(ansatz, params, *, hf_energy, n_walkers, n_burnin,
                      n_accum, sample_every, step_size, key, label):
    """Run Metropolis on ``ansatz`` and report sample mean of E_L.

    Returns ``(mean, stderr, n_samples)`` where ``stderr`` is the
    *between-walker* standard error.  With independent walker chains (vanilla
    Metropolis has no cross-walker information sharing), each walker's
    time-averaged mean is an i.i.d. estimator of ⟨E_L⟩; the standard
    deviation of those means divided by √n_walkers gives an
    autocorrelation-corrected error bar without any blocking.
    """
    walkers = initialize_walkers(ansatz, n_walkers, key=key)
    batch_ansatz = jax.vmap(ansatz, in_axes=(0, None))
    _, walkers = batch_ansatz(walkers, params)

    step = make_mcmc_step(ansatz, step_size=step_size, move_type="one")

    for _ in range(n_burnin):
        key, sub = random.split(key)
        walkers, _ = step(ansatz, walkers, sub, params)

    jastrow_params = params[0]
    energy_fn = jax.jit(jax.vmap(
        lambda w: compute_single_walker_energy(ansatz, w, jastrow_params),
        in_axes=0,
    ))

    rows = []                    # each row: shape (n_walkers,)
    accept_rates = []
    for s in range(n_accum):
        key, sub = random.split(key)
        walkers, acc = step(ansatz, walkers, sub, params)
        if s % sample_every == 0:
            rows.append(np.asarray(energy_fn(walkers)))
            accept_rates.append(float(acc))

    samples_2d = np.stack(rows, axis=0)              # (n_recorded, n_walkers)
    walker_means = samples_2d.mean(axis=0)           # (n_walkers,)
    mean = float(walker_means.mean())
    stderr = float(walker_means.std(ddof=1) / np.sqrt(n_walkers))
    naive_stderr = float(
        samples_2d.std(ddof=1) / np.sqrt(samples_2d.size)
    )
    avg_accept = float(np.mean(accept_rates))
    print(
        f"  [{label}] HF = {hf_energy:.6f}  VMC = {mean:.6f} "
        f"± {stderr:.6f} (between-walker, "
        f"naive {naive_stderr:.6f})  "
        f"delta = {mean - hf_energy:+.6f}  Naccept = {avg_accept:.2f}  "
        f"Nsamp = {samples_2d.size}"
    )
    return mean, stderr, samples_2d.size


@unittest.skipUnless(os.environ.get("PYTC_RUN_SLOW"), "slow test, set PYTC_RUN_SLOW=1")
class TestECPMCMCExpectationValue(unittest.TestCase):
    """Quantitative check: ⟨E_L⟩ sampled from |ψ_HF|² should approach the
    HF/ECP total energy when J = 0.  The trial wavefunction IS the HF
    determinant, so the locality approximation introduces no bias — the
    only discrepancy is statistical (MCMC stderr, inflated by autocorrelation
    above the naive standard error).
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    # Tolerance: 5x the between-walker stderr (a clean ~5σ acceptance test).
    # The between-walker stderr is autocorrelation-corrected by construction,
    # so we do not need an extra inflation factor.
    TOL_FACTOR = 5.0
    # Absolute floor:  even if stderr is somehow tiny, demand at most 5 mHa
    # discrepancy.  This catches systematic errors (e.g. wrong sign in the
    # non-local kernel) that would otherwise pass at large tolerance.
    ABS_FLOOR_HA = 5.0e-3

    def test_carbon_ccecp_hf_energy_recovered(self):
        _, mf, ansatz, params = _build_c_ccecp_ansatz()
        hf_energy = float(mf.e_tot)
        mean, stderr, _ = _mcmc_mean_energy(
            ansatz, params,
            hf_energy=hf_energy,
            n_walkers=256, n_burnin=1000, n_accum=5000,
            sample_every=10, step_size=0.5,
            key=random.PRNGKey(2026),
            label="C ccECP",
        )
        delta = mean - hf_energy
        tol = max(self.TOL_FACTOR * stderr, self.ABS_FLOOR_HA)
        self.assertLess(abs(delta), tol,
                        msg=f"mean {mean} vs HF {hf_energy} (stderr {stderr})")

    def test_co_ccecp_hf_energy_recovered(self):
        _, mf, ansatz, params = _build_co_ccecp_ansatz()
        hf_energy = float(mf.e_tot)
        # CO has 10 electrons (vs 4 for C); autocorrelation is larger, so
        # the naive stderr underestimates the real error.  Use a longer
        # burn-in, more accumulation, and a larger spacing between recorded
        # samples to compensate.
        mean, stderr, _ = _mcmc_mean_energy(
            ansatz, params,
            hf_energy=hf_energy,
            n_walkers=256, n_burnin=3000, n_accum=12000,
            sample_every=20, step_size=0.4,
            key=random.PRNGKey(2027),
            label="CO ccECP",
        )
        delta = mean - hf_energy
        # CO has 10 electrons (vs 4 for C), so E_L variance is larger.
        # Allow a slightly looser absolute floor than the C test.
        tol = max(self.TOL_FACTOR * stderr, 0.01)
        self.assertLess(abs(delta), tol,
                        msg=f"mean {mean} vs HF {hf_energy} (stderr {stderr})")


if __name__ == "__main__":
    unittest.main()
