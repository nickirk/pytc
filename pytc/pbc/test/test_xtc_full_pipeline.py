"""End-to-end Gamma periodic xTC-to-factor-direct RCCSD gates."""

import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import h5py
import numpy as np
from flax import struct
from pyscf.pbc import gto, scf

from pytc.jastrow import Jastrow
from pytc.pbc.jastrow import BoysHandy
from pytc.pbc.xtc import create_isdf_xtc_fft
from pytc.solver import isdf_xtc_ccsd, jax_xtc_ccsd
from pytc.solver.xtc_ccsd import _init_df_eris


jax.config.update("jax_enable_x64", True)


@struct.dataclass
class _ZeroJastrow(Jastrow):
    """Differentiable identity Jastrow used for the pipeline identity gate."""

    def _compute(self, r1, r2, params):
        del params
        return jnp.sum((r1 - r1) + (r2 - r2))

    def init_params(self, **kwargs):
        del kwargs
        return {}


def _cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = "sto-3g"
    cell.a = np.eye(3) * 8.0
    cell.unit = "B"
    cell.cart = True
    # The SCF mesh is independent of the deliberately tiny xTC oracle mesh.
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    return cell


class TestPeriodicXTCFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell()
        cls.mf = scf.RHF(cls.cell)
        cls.mf.exxdiv = None
        cls.mf.conv_tol = 1e-10
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError("Periodic RHF fixture did not converge")

        cls.jastrow = BoysHandy.create(cls.cell)
        cls.params = cls.jastrow.init_params()
        cls.isdf_xtc = create_isdf_xtc_fft(
            cls.mf,
            cls.jastrow,
            cls.params,
            mesh=(2, 2, 2),
            n_rank=2,
            is_incore=True,
        )

    @staticmethod
    def _run_ccsd(solver_cls, mf, xtc_obj, params, *, factor_direct):
        cc = solver_cls(
            mf,
            xtc_obj,
            params,
            on_the_fly_vvvv=factor_direct,
            max_memory=2000,
            gpu_max_memory=512,
        )
        cc.conv_tol = 1e-9
        cc.max_cycle = 50
        corr, _, _ = cc.kernel()
        if not cc.converged:
            raise AssertionError("tiny-cell xTC-RCCSD did not converge")
        return float(corr), float(cc.e_tot)

    def test_fft_isdf_factor_direct_matches_materialized_solver(self):
        materialized = self._run_ccsd(
            jax_xtc_ccsd.RCCSD,
            self.mf,
            self.isdf_xtc,
            self.params,
            factor_direct=False,
        )
        factor_direct = self._run_ccsd(
            isdf_xtc_ccsd.RCCSD,
            self.mf,
            self.isdf_xtc,
            self.params,
            factor_direct=True,
        )
        np.testing.assert_allclose(factor_direct, materialized, atol=1e-12, rtol=1e-12)

    def test_zero_jastrow_has_zero_periodic_isdf_corrections(self):
        jastrow = _ZeroJastrow()
        params = jastrow.init_params()
        xtc_obj = create_isdf_xtc_fft(
            self.mf,
            jastrow,
            params,
            mesh=(2, 2, 2),
            n_rank=2,
            is_incore=True,
        )
        for name in ("K1_kernel", "K3_kernel", "D", "X"):
            np.testing.assert_allclose(
                np.asarray(xtc_obj.isdf_kernels[name]), 0.0, atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(xtc_obj.get_1b(params), 0.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(xtc_obj.get_2b(params), 0.0, atol=1e-12, rtol=0)

    def test_out_of_core_helper_persists_complete_kernel_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/periodic_xtc.h5"
            xtc_obj = create_isdf_xtc_fft(
                self.mf,
                self.jastrow,
                self.params,
                mesh=(2, 2, 2),
                n_rank=2,
                is_incore=False,
                save_path=path,
            )
            self.assertIsNone(xtc_obj.xi_phi)
            xtc_obj.isdf_kernels["X"].file.close()
            with h5py.File(path, "r") as handle:
                for name in (
                    "phi_isdf",
                    "grad_phi_isdf",
                    "pivots",
                    "K1_kernel",
                    "K3_kernel",
                    "D",
                    "X",
                ):
                    self.assertIn(name, handle)

    def test_df_bridge_rejects_incomplete_provider_before_allocation(self):
        incomplete = SimpleNamespace(blockdim=1)
        with self.assertRaisesRegex(TypeError, "callable loop"):
            _init_df_eris(
                SimpleNamespace(),
                incomplete,
                nvir=1,
                naux=1,
                nocc=1,
                nmo=2,
                mo_coeff=np.eye(2),
            )


if __name__ == "__main__":
    unittest.main()
