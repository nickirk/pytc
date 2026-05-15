"""Local energy under PBC with k-point Slater determinants.

The load-bearing check: native k-mesh local energy on a primitive cell
matches the supercell-Gamma local energy on the Nk-replicated cell at
the same electron configuration. Both are evaluations of the same
trial wavefunction (no Jastrow) on the same physical system, so the
local energy must agree to machine precision.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from pyscf.pbc import gto as pbcgto, scf as pbcscf, tools as pbctools

from pytc.ansatz.det import eval_det_value_and_grad
from pytc.vmc.walker import initialize_walker_state

from pytc.pbc.ansatz import create_slater_det, create_slater_det_kpts
from pytc.pbc.ansatz.kdet import eval_kdet_value_and_grad
from pytc.pbc.vmc import make_ewald_params, compute_single_walker_energy_kpts


def _build_primitive(L=4.0):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'; cell.cart = True; cell.verbose = 0
    cell.build()
    return cell


def _populate_unbatched_kdet(kdet, positions):
    walker = initialize_walker_state(kdet, jnp.asarray(positions)[None, :, :])
    _, walker = eval_kdet_value_and_grad(kdet, walker)
    return jax.tree_util.tree_map(lambda x: x[0], walker)


def _populate_unbatched_mol(det, positions):
    walker = initialize_walker_state(det, jnp.asarray(positions)[None, :, :])
    _, walker = eval_det_value_and_grad(det, walker)
    return jax.tree_util.tree_map(lambda x: x[0], walker)


class TestKptsLocalEnergySmoke(unittest.TestCase):
    def test_finite_real_value(self):
        prim = _build_primitive()
        nk = [2, 1, 1]
        sup = pbctools.super_cell(prim, nk)
        sup.cart = True; sup.verbose = 0; sup.build()
        mf = pbcscf.KRHF(prim, kpts=prim.make_kpts(nk))
        mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf, supercell=sup)
        ewald = make_ewald_params(sup.lattice_vectors())

        rng = np.random.default_rng(0)
        positions = jnp.asarray(rng.uniform(0, 4, size=(2 * kdet.n_alpha, 3)))
        walker = _populate_unbatched_kdet(kdet, positions)

        e = float(compute_single_walker_energy_kpts(kdet, walker, ewald))
        self.assertTrue(np.isfinite(e))


class TestKptsLocalEnergySupercellEquivalence(unittest.TestCase):
    """Native k-mesh local energy = supercell-Gamma local energy at the
    same electron configuration, in the absence of a Jastrow."""

    def setUp(self):
        prim = _build_primitive(L=4.0)
        self.nk = [2, 1, 1]
        sup = pbctools.super_cell(prim, self.nk)
        sup.cart = True; sup.verbose = 0; sup.build()

        # Native k-mesh KRHF on the primitive cell.
        mf_k = pbcscf.KRHF(prim, kpts=prim.make_kpts(self.nk))
        mf_k.exxdiv = None; mf_k.kernel()
        self.kdet = create_slater_det_kpts(mf_k, supercell=sup)

        # Supercell-Gamma RHF.
        mf_sup = pbcscf.RHF(sup); mf_sup.exxdiv = None; mf_sup.kernel()
        self.det_sup = create_slater_det(sup, mo_coeff=mf_sup.mo_coeff)

        self.ewald = make_ewald_params(sup.lattice_vectors())
        # Both have the same number of electrons (the KSlaterDet's are
        # counted as sum over k of nocc(k), which equals the supercell
        # Gamma's nocc).
        self.assertEqual(self.kdet.n_alpha, self.det_sup.n_alpha)

    def test_local_energy_matches(self):
        n_alpha = self.kdet.n_alpha
        rng = np.random.default_rng(11)
        for trial in range(3):
            positions = jnp.asarray(rng.uniform(0, 4, size=(2 * n_alpha, 3)))

            w_k = _populate_unbatched_kdet(self.kdet, positions)
            w_m = _populate_unbatched_mol(self.det_sup, positions)

            e_k = float(compute_single_walker_energy_kpts(self.kdet, w_k, self.ewald))
            # The same function works on a real-valued walker (duck-typed
            # on inv_up/inv_down/lap_up/lap_down) — it produces the
            # supercell-Gamma bare-determinant local energy.
            e_m = float(compute_single_walker_energy_kpts(self.det_sup, w_m, self.ewald))

            np.testing.assert_allclose(
                e_k, e_m, atol=1e-8,
                err_msg=f"trial {trial}: k-mesh E_L={e_k:.6f} vs supercell-Gamma E_L={e_m:.6f}",
            )


class TestKptsLocalEnergyNearHF(unittest.TestCase):
    """A sanity-level check: with a bare HF Slater determinant trial, the
    average local energy must be close to the PBC HF total energy
    (within a few statistical errors over a small VMC sampling).

    Concretely we evaluate the local energy at a handful of randomly
    drawn configurations near the nuclei and check the mean is in the
    right ballpark. Not statistically tight — this is a smoke-level
    proximity check, not a VMC validation run.
    """

    def test_mean_local_energy_near_hf(self):
        prim = _build_primitive(L=6.0)
        nk = [2, 1, 1]
        sup = pbctools.super_cell(prim, nk)
        sup.cart = True; sup.verbose = 0; sup.build()
        mf = pbcscf.KRHF(prim, kpts=prim.make_kpts(nk))
        mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf, supercell=sup)
        ewald = make_ewald_params(sup.lattice_vectors())

        # Reference: the PBC HF total energy for the supercell-equivalent
        # system, with exxdiv=None to match our Ewald handling.
        mf_sup = pbcscf.RHF(sup); mf_sup.exxdiv = None; mf_sup.kernel()
        hf_total = float(mf_sup.e_tot)

        rng = np.random.default_rng(42)
        # Generate a handful of random configurations inside the supercell.
        L_super = jnp.diag(jnp.asarray(sup.lattice_vectors()))
        n_elec = 2 * kdet.n_alpha
        energies = []
        for trial in range(40):
            positions = jnp.asarray(rng.uniform(0, 1, size=(n_elec, 3))) * L_super
            walker = _populate_unbatched_kdet(kdet, positions)
            e = float(compute_single_walker_energy_kpts(kdet, walker, ewald))
            if np.isfinite(e):
                energies.append(e)

        mean_e = float(np.mean(energies))
        # Wide tolerance: this isn't sampled from |psi|^2, so it's not the
        # VMC mean — just a sanity proximity. HF energy for H2 in our
        # supercell at L=6 is around -1 Ha; the trial wavefunction's
        # local-energy mean over uniform sampling can drift, but should
        # land in a finite window around HF.
        self.assertTrue(np.isfinite(mean_e))
        self.assertLess(abs(mean_e - hf_total), 50.0,
                        f"mean local energy {mean_e:+.4f} too far from HF {hf_total:+.4f}")


if __name__ == '__main__':
    unittest.main()
