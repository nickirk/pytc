"""V2 reference-replay gate: compare Pi^q/eta^q/W^q against the fftisdf
external CPU oracle (task #20 baseline) on he2-cubic-cell [1,1,3], per
design v2.1 section 8's V2 spec.

fftisdf lives at ~/Work/src/fftisdf (task #20's clone) and is added to
sys.path only for THIS test -- reference/oracle only, no fftisdf code
is imported into any pytc production module (pytc.pbc.df.kpts/isdf
have zero dependency on it). Skipped cleanly if the clone is absent.

Convention note (discovered empirically while writing this test, not
assumed): fftisdf's own Pi^q/eta^q differ from pytc's own
pair_convolve-based Pi^q/eta^q by a KNOWN, exact relationship --
    fftisdf_Pi[q]  = sqrt(Nk) * conj(pytc_Pi[q])
    fftisdf_eta[q] = sqrt(Nk) * conj(pytc_eta[q])
-- a pure normalization/conjugation CONVENTION difference (fftisdf's
internal k<->supercell phase transform uses a different -- but equally
valid -- normalization than pytc's own NumPy-"backward"-FFT convention,
which design v2.1 section 4 explicitly fixes as pytc's canonical
choice). This is NOT a bug in either implementation; reference-replay
mode accounts for it explicitly rather than expecting bit-identical
raw arrays. apply_raw_kernel_and_solve's own kern_q/coulG/FFT logic
was verified independently (bit-identical to fftisdf's own intermediate
values, given the same eta input) BEFORE this convention difference was
even identified, isolating it precisely to the Pi/eta normalization
step, not the kernel-application step.
"""

import os
import sys
import unittest

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import apply_raw_kernel_and_solve, build_pi_eta
from pytc.pbc.df.kpts import canonicalize_kpts

_FFTISDF_PATH = os.path.expanduser("~/Work/src/fftisdf")
_FFTISDF_AVAILABLE = os.path.isfile(os.path.join(_FFTISDF_PATH, "fft", "__init__.py"))


@unittest.skipUnless(
    _FFTISDF_AVAILABLE, f"fftisdf reference clone not found at {_FFTISDF_PATH} (task #20)"
)
class TestV2ReferenceReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _FFTISDF_PATH not in sys.path:
            sys.path.insert(0, _FFTISDF_PATH)
        from fft.test.test_slow import setup as fftisdf_setup

        class _Holder:
            pass

        holder = _Holder()
        fftisdf_setup(
            holder, cell="he2-cubic-cell", kmesh=[1, 1, 3], wrap_around=False,
            verbose=0, cisdf=4.0,
        )
        cls.cell = holder.cell
        cls.kpts_ref = holder.isdf_obj.kpts
        cls.inpv_kpt = holder.isdf_obj.inpv_kpt
        eta_flat = holder.isdf_obj.build_eta_kpt(cls.inpv_kpt)
        nk, nip, _ = cls.inpv_kpt.shape
        ngrid = eta_flat.shape[1]
        cls.eta_ref = eta_flat.reshape(nk, nip, ngrid)
        cls.coul_kpt_ref = holder.isdf_obj.coul_kpt

        cls.mesh_obj = canonicalize_kpts(cls.cell, cls.kpts_ref)
        cls.grid_coords = cls.cell.get_uniform_grids(cls.cell.mesh)
        cls.ao_full = np.asarray(
            cls.cell.pbc_eval_gto("GTOval", cls.grid_coords, kpts=list(cls.kpts_ref)),
            dtype=np.complex128,
        )
        cls.Pi_mine, cls.eta_mine = build_pi_eta(
            cls.inpv_kpt, cls.ao_full, cls.mesh_obj.kmesh
        )

    def test_pi_matches_fftisdf_within_v2_tolerance(self):
        from fft.isdf import contract as fftisdf_contract, get_phase_factor

        phase = get_phase_factor(self.cell, self.kpts_ref)
        Pi_ref = fftisdf_contract(self.inpv_kpt, self.inpv_kpt, phase)
        n_k = self.mesh_obj.n_kpts
        sqrt_nk = np.sqrt(n_k)
        for q in range(n_k):
            expected = sqrt_nk * self.Pi_mine[q].conj()
            rel = np.abs(expected - Pi_ref[q]).max() / np.abs(Pi_ref[q]).max()
            self.assertLess(rel, 1e-6, msg=f"q={q}")

    def test_eta_matches_fftisdf_within_v2_tolerance(self):
        n_k = self.mesh_obj.n_kpts
        sqrt_nk = np.sqrt(n_k)
        for q in range(n_k):
            expected = sqrt_nk * self.eta_mine[q].conj()
            rel = np.abs(expected - self.eta_ref[q]).max() / np.abs(self.eta_ref[q]).max()
            self.assertLess(rel, 1e-6, msg=f"q={q}")

    def test_w_matches_fftisdf_within_v2_tolerance(self):
        # Reference-replay mode: inject fftisdf's own pivot order (same
        # inpv_kpt) AND account for the known Pi/eta convention
        # difference, then compare the fully-solved W^q.
        n_k = self.mesh_obj.n_kpts
        sqrt_nk = np.sqrt(n_k)
        for q in range(n_k):
            Pi_q = sqrt_nk * self.Pi_mine[q].conj()
            eta_q = sqrt_nk * self.eta_mine[q].conj()
            W_q, kern_q, info = apply_raw_kernel_and_solve(
                Pi_q, eta_q, cell=self.cell, q_kpt=self.mesh_obj.canonical_kpts[q],
                grid_coords=self.grid_coords, grid_mesh=self.cell.mesh, rtol=1e-8,
            )
            rel = np.abs(W_q - self.coul_kpt_ref[q]).max() / np.abs(self.coul_kpt_ref[q]).max()
            self.assertLess(rel, 1e-6, msg=f"q={q}: rel={rel:.3e}")

    def test_kern_q_matches_fftisdf_intermediate_exactly(self):
        # Isolates apply_raw_kernel_and_solve's own coulG/FFT/IFFT logic,
        # independent of the Pi/eta convention difference above -- feeds
        # fftisdf's OWN eta directly (no conversion needed, since this
        # only exercises the kernel-application step, not Pi).
        from fft.isdf import get_phase_factor
        from pyscf.pbc import tools as pbctools
        from pyscf import lib

        mesh = self.cell.mesh
        Gv = self.cell.get_Gv(mesh)
        coord = self.grid_coords
        n_grid = coord.shape[0]

        for q in range(self.mesh_obj.n_kpts):
            fq = np.exp(-1j * coord @ self.kpts_ref[q])
            vq = pbctools.get_coulG(self.cell, k=self.kpts_ref[q], exx=False, Gv=Gv, mesh=mesh)
            vq = vq * self.cell.vol / n_grid
            lq = self.eta_ref[q] * fq
            wq = pbctools.fft(lq, mesh)
            rq = pbctools.ifft(wq * vq, mesh).conj()
            kern_ref = lib.dot(lq, rq.T) / np.sqrt(n_grid)

            Pi_q = np.sqrt(self.mesh_obj.n_kpts) * self.Pi_mine[q].conj()
            _, kern_mine, _ = apply_raw_kernel_and_solve(
                Pi_q, self.eta_ref[q], cell=self.cell, q_kpt=self.mesh_obj.canonical_kpts[q],
                grid_coords=coord, grid_mesh=mesh, rtol=1e-8,
            )
            np.testing.assert_allclose(kern_mine, kern_ref, atol=1e-6, err_msg=f"q={q}")


if __name__ == "__main__":
    unittest.main()
