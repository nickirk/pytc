"""Default factorized ISDF xTC-CCSD dispatch.

The materialized implementation remains in :mod:`jax_xtc_ccsd`.  This class
inherits all of its CCSD machinery and replaces only the VVVV--T2 leg through
the narrow instance hook in that module.
"""

from __future__ import annotations

import numpy as np

from pytc.solver import factor_direct_vvvv, jax_xtc_ccsd, robust_df_thc
from pytc.solver.robust_df_thc_scalable_jax import (
    direct_df_sandwiches_panelled_jax,
    fit_panelled_lsthc_jax,
)


class RCCSD(jax_xtc_ccsd.RCCSD):
    """JAX RCCSD whose default on-the-fly VVVV path is factorized."""

    factorized_rank_panel = 128
    factorized_aux_panel = 32
    factorized_virtual_panel = 32
    factorized_rcond = 1.0e-12

    def _factorized_state(self):
        state = getattr(self, "_isdf_factorized_state", None)
        if state is not None:
            return state
        base = self.xtc_obj
        nocc = self.nocc
        kernels = base.isdf_kernels
        required = ("K1_kernel", "K3_kernel", "D", "X")
        missing = [name for name in required if name not in kernels]
        if missing:
            raise RuntimeError(f"ISDF kernels missing for factorized RCCSD: {missing}")
        tc = {
            "p": np.asarray(base.phi_isdf[nocc:], dtype=np.float64),
            "grad_p": np.asarray(base.grad_phi_isdf[nocc:], dtype=np.float64),
            "u1": np.asarray(kernels["K1_kernel"], dtype=np.float64),
            "u3": np.asarray(kernels["K3_kernel"], dtype=np.float64),
            "d": np.asarray(kernels["D"], dtype=np.float64),
            "x": np.asarray(kernels["X"], dtype=np.float64)[nocc:, nocc:, :],
        }
        with_df = getattr(self, "with_df", None) or self._scf.with_df
        b = robust_df_thc.extract_metric_applied_vv_df_factor(
            with_df, self.mo_coeff, nocc)
        nvir = tc["p"].shape[0]
        fit = fit_panelled_lsthc_jax(
            tc["p"], b, rcond=self.factorized_rcond,
            virtual_panel=min(self.factorized_virtual_panel, nvir))
        state = (tc, b, fit)
        self._isdf_factorized_state = state
        return state

    def _contract_vvvv_t2(self, cc, t2_jax, eris, t2new_host):
        """Instance hook called by the inherited JAX update path; no VVVV tile."""
        del cc
        if eris.vvvv is not None:
            raise RuntimeError(
                "factorized RCCSD refuses a materialized VVVV store; select "
                "jax_xtc_ccsd.RCCSD for the legacy materialized route")
        tc, b, fit = self._factorized_state()
        rank_panel = min(self.factorized_rank_panel, fit.p_virtual.shape[1])
        aux_panel = min(self.factorized_aux_panel, b.shape[2])
        terms = factor_direct_vvvv.contract_isdf_factor_direct_terms_t2(
            t2_jax, **tc, occupied_pair_batch_size=min(8, self.nocc * self.nocc),
            rank_panel_size=rank_panel)
        coulomb = direct_df_sandwiches_panelled_jax(
            b, fit, t2_jax, rank_panel=rank_panel, aux_panel=aux_panel)
        t2new_host += np.asarray(terms["final"] + coulomb.robust, dtype=np.float64)
