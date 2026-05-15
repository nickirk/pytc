"""End-to-end k-point VMC validation.

Runs a bare-HF (no Jastrow) VMC sampling two ways:
  * native primitive-cell k-mesh with KSlaterDet
  * Nk-replicated supercell at Gamma with the molecular SlaterDet

Both wavefunctions span the same one-electron space, and both walkers
sample electrons in the same supercell. The VMC mean energies must
therefore agree within combined statistical error. This is the
load-bearing 'k-point VMC works correctly' test for stage-1 k-points.

CPU-friendly tiny parameters by default; bumped via the
PYTC_KPTS_VMC_VALIDATION env var when running on a GPU.
"""

import os
import unittest

import numpy as np
import jax
from jax import random
from pyscf.pbc import gto as pbcgto, scf as pbcscf, tools as pbctools

from pytc.pbc.ansatz import create_slater_det, create_slater_det_kpts
from pytc.pbc.vmc import make_ewald_params, sample_bare


def _params_from_env():
    """Tiny by default; promotable to larger via env var.

    The CPU profile is a smoke test (runs and produces finite results).
    The GPU profile runs long enough for the 5-sigma cross-check to be
    meaningful — the bare-HF wavefunction has heavy tails near nuclei
    (no cusp correction), so a few thousand samples are needed to
    average out the tail contributions.
    """
    profile = os.environ.get('PYTC_KPTS_VMC_VALIDATION', 'tiny').lower()
    if profile == 'gpu':
        return dict(
            n_walkers=512, n_steps=2000, burn_in_steps=500, step_size=0.4,
            sigma_tolerance=5.0,
        )
    return dict(
        n_walkers=16, n_steps=60, burn_in_steps=40, step_size=0.4,
        sigma_tolerance=30.0,  # generous: tiny sample, heavy-tailed E_L
    )


class TestKptsVMCValidation(unittest.TestCase):
    """Native k-mesh VMC matches supercell-Gamma VMC on bare HF trial."""

    def test_kmesh_matches_supercell_gamma(self):
        cfg = _params_from_env()

        # Primitive H2 cell
        L = 6.0
        prim = pbcgto.Cell()
        prim.atom = 'H 0 0 0; H 0 0 1.4'
        prim.basis = 'sto-3g'
        prim.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
        prim.unit = 'B'; prim.cart = True; prim.verbose = 0
        prim.build()

        nk = [2, 1, 1]
        sup = pbctools.super_cell(prim, nk)
        sup.cart = True; sup.verbose = 0; sup.build()

        # Native k-mesh KRHF on primitive cell
        mf_k = pbcscf.KRHF(prim, kpts=prim.make_kpts(nk))
        mf_k.exxdiv = None; mf_k.kernel()
        kdet = create_slater_det_kpts(mf_k, supercell=sup)

        # Supercell-Gamma RHF
        mf_sup = pbcscf.RHF(sup); mf_sup.exxdiv = None; mf_sup.kernel()
        sup_det = create_slater_det(sup, mo_coeff=mf_sup.mo_coeff)

        ewald = make_ewald_params(sup.lattice_vectors())

        # Run both VMC samplers with independent RNG seeds.
        r_kpt = sample_bare(
            kdet, sup, ewald,
            n_walkers=cfg['n_walkers'], n_steps=cfg['n_steps'],
            burn_in_steps=cfg['burn_in_steps'], step_size=cfg['step_size'],
            key=random.PRNGKey(11), log=False,
        )
        r_sup = sample_bare(
            sup_det, sup, ewald,
            n_walkers=cfg['n_walkers'], n_steps=cfg['n_steps'],
            burn_in_steps=cfg['burn_in_steps'], step_size=cfg['step_size'],
            key=random.PRNGKey(22), log=False,
        )

        e_kpt, err_kpt = r_kpt['mean'], r_kpt['stderr']
        e_sup, err_sup = r_sup['mean'], r_sup['stderr']
        diff = abs(e_kpt - e_sup)
        combined_err = float(np.sqrt(err_kpt ** 2 + err_sup ** 2))
        tolerance = cfg['sigma_tolerance'] * combined_err

        # Reference: PBC HF total energy (Madelung-corrected). VMC of the HF
        # wavefunction without Jastrow should converge to this.
        hf_ref = float(mf_sup.e_tot)

        self.assertGreater(r_kpt['acceptance'], 0.2)
        self.assertGreater(r_sup['acceptance'], 0.2)
        self.assertLess(
            diff, tolerance,
            msg=(
                f"VMC means disagree:\n"
                f"  k-mesh:  {e_kpt:+.5f} +/- {err_kpt:.5f}  acc {r_kpt['acceptance']:.3f}\n"
                f"  super:   {e_sup:+.5f} +/- {err_sup:.5f}  acc {r_sup['acceptance']:.3f}\n"
                f"  HF ref:  {hf_ref:+.5f}\n"
                f"  |diff|:  {diff:.5f} > tolerance {tolerance:.5f}\n"
                f"  combined stderr: {combined_err:.5f}"
            ),
        )


if __name__ == '__main__':
    unittest.main()
