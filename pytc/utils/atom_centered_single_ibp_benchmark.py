"""Direct atom-centered-grid benchmark for single-IBP Coulomb integrals.

Unlike the uniform-grid experiment in ``single_ibp_coulomb_benchmark``, a
PySCF/TC atom-centered Becke grid is not translationally invariant.  The
``rhat`` action is therefore evaluated by blocked real-space summation, not by
FFT.  The benchmark uses the exact grid construction in ``TC.from_pyscf`` and
forms the complete occupied-virtual pair space directly (no ISDF/rank error):

    (ia|jb) = -1/2 sum_g w_g grad(phi_i phi_a)_g .
                         sum_h w_h rhat_(g-h) (phi_j phi_b)_h.

The same-grid zero separation is assigned the centered point value zero.  A
separate diagonal-omitted ``1/r`` grid sum is provided only as a diagnostic:
an atom-centered quadrature weight is not a geometric cell volume, may be zero
or negative after molecular partitioning, and therefore does not define a
controlled scalar Coulomb self-cell correction.

Task #18, #proj-isdf-coulomb-cuda, 2026-07-13.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
from pyscf import ao2mo, dft, gto, mp, scf

from pytc.df.solvers import (
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
)
from pytc.integrals.coulomb import (
    pair_collocation_at_pivots,
    reconstruct_eri_block,
    select_sector_pivots,
    weight_mo_values,
)
from pytc.utils.poisson_core_benchmark import SYSTEMS, mp2_energy_from_ovov
from pytc.utils.single_ibp_coulomb_benchmark import build_gradient_interpolation_vectors


def _validate_grid_inputs(density, coords, weights, name):
    density = np.asarray(density)
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    if density.ndim != 2:
        raise ValueError(f"{name}_density must have shape (n_function,n_grid)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name}_coords must have shape (n_grid,3)")
    if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
        raise ValueError(f"{name}_weights must have shape (n_grid,)")
    if density.shape[1] != coords.shape[0]:
        raise ValueError(f"{name}_density grid axis does not match {name}_coords")
    if not np.issubdtype(density.dtype, np.inexact):
        raise ValueError(f"{name}_density must use a floating or complex dtype")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} coordinates and weights must be finite")
    return density, coords, weights


def atom_centered_single_ibp_core(
    gradient_density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Blocked one-sided single-IBP core on one or two irregular grids.

    The returned matrix is deliberately unsymmetrized.  Coincident left/right
    points receive ``rhat=0``; ``coincident_pairs`` is returned so a benchmark
    cannot silently forget which diagonal convention was used.
    """
    gradient = np.asarray(gradient_density_left)
    density_right = np.asarray(density_right)
    if gradient.ndim != 3 or gradient.shape[1] != 3:
        raise ValueError("gradient_density_left must have shape (n_left,3,n_grid_left)")
    dummy_left = np.empty((gradient.shape[0], gradient.shape[2]), dtype=gradient.dtype)
    _, coords_left, weights_left = _validate_grid_inputs(
        dummy_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if gradient.dtype != density_right.dtype:
        raise ValueError("gradient_density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    n_left = gradient.shape[0]
    n_right = density_right.shape[0]
    result = np.zeros((n_left, n_right), dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        n_eval = i1 - i0
        vector = np.zeros((n_right, 3, n_eval), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            vector += np.einsum("nj,ijc->nci", weighted_density, rhat, optimize=True)
        result += -0.5 * np.einsum(
            "mci,nci,i->mn", gradient[:, :, i0:i1].conj(), vector,
            weights_left[i0:i1], optimize=True,
        )
    return result, coincident_pairs


def atom_centered_direct_coulomb_core_offdiagonal(
    density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Diagnostic ``1/r`` quadrature with coincident terms set to zero.

    This is *not* a controlled production self-cell prescription.  It exists
    to quantify how much the bounded single-IBP kernel helps relative to the
    naive singular grid sum on the same atom-centered points.
    """
    density_left, coords_left, weights_left = _validate_grid_inputs(
        density_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if density_left.dtype != density_right.dtype:
        raise ValueError("density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    result = np.zeros((density_left.shape[0], density_right.shape[0]),
                      dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        potential = np.zeros((density_right.shape[0], i1 - i0), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            kernel = np.divide(
                1.0, radius, out=np.zeros_like(radius), where=radius != 0.0
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            potential += weighted_density @ kernel.T
        result += (density_left[:, i0:i1].conj() * weights_left[i0:i1]) @ potential.T
    return result, coincident_pairs


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(tuple(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class AtomCenteredSingleIBPCase:
    system: str = "H2O_ccpVDZ"
    grid_level: int = 1
    pruning: str = "default"
    auxbasis: str = "weigend"
    eval_block_size: int = 128
    source_block_size: int = 4096
    include_direct_offdiagonal: bool = True
    rank_factor: float | None = None

    def __post_init__(self):
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}")
        if isinstance(self.grid_level, bool) or int(self.grid_level) < 0:
            raise ValueError("grid_level must be a non-negative integer")
        if self.pruning not in ("default", "none"):
            raise ValueError("pruning must be 'default' or 'none'")
        for name in ("eval_block_size", "source_block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rank_factor is not None:
            if not math.isfinite(self.rank_factor) or self.rank_factor <= 0.0:
                raise ValueError("rank_factor must be finite and positive when provided")


def _relative_error(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def run_atom_centered_case(case: AtomCenteredSingleIBPCase) -> dict[str, Any]:
    """Run one full occupied-virtual H2O/benzene atom-centered-grid case."""
    spec = SYSTEMS[case.system]
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=case.auxbasis).run()
    coeff = np.asarray(mf.mo_coeff)
    n_occ = mol.nelectron // 2
    n_vir = coeff.shape[1] - n_occ

    grids = dft.gen_grid.Grids(mol)
    grids.level = case.grid_level
    if case.pruning == "none":
        grids.prune = None
    grids.build()
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    mo = np.einsum("xgu,up->xgp", ao, coeff, optimize=True)
    values = mo[0].T
    gradients = mo[1:4].transpose(2, 0, 1)
    occ, vir = values[:n_occ], values[n_occ:]
    grad_occ, grad_vir = gradients[:n_occ], gradients[n_occ:]
    pair = np.einsum("ig,ag->iag", occ, vir).reshape(n_occ * n_vir, -1)
    pair_gradient = (
        np.einsum("icg,ag->iacg", grad_occ, vir)
        + np.einsum("ig,acg->iacg", occ, grad_vir)
    ).reshape(n_occ * n_vir, 3, -1)

    t0 = time.perf_counter()
    eri_ibp, ibp_coincident = atom_centered_single_ibp_core(
        pair_gradient, pair, coords, weights,
        eval_block_size=case.eval_block_size,
        source_block_size=case.source_block_size,
    )
    ibp_seconds = time.perf_counter() - t0
    eri_ibp_projected = 0.5 * (eri_ibp + eri_ibp.conj().T)

    eri_direct = None
    direct_seconds = None
    direct_coincident = None
    if case.include_direct_offdiagonal:
        t0 = time.perf_counter()
        eri_direct, direct_coincident = atom_centered_direct_coulomb_core_offdiagonal(
            pair, pair, coords, weights,
            eval_block_size=case.eval_block_size,
            source_block_size=case.source_block_size,
        )
        direct_seconds = time.perf_counter() - t0

    eri_reference_df = mf.with_df.ao2mo(
        (coeff[:, :n_occ], coeff[:, n_occ:], coeff[:, :n_occ], coeff[:, n_occ:]),
        compact=False,
    ).reshape(n_occ * n_vir, n_occ * n_vir)
    eri_reference_exact = ao2mo.kernel(
        mol,
        (coeff[:, :n_occ], coeff[:, n_occ:], coeff[:, :n_occ], coeff[:, n_occ:]),
        compact=False,
    ).reshape(n_occ * n_vir, n_occ * n_vir)
    reference_df_4 = eri_reference_df.reshape(n_occ, n_vir, n_occ, n_vir)
    reference_exact_4 = eri_reference_exact.reshape(n_occ, n_vir, n_occ, n_vir)
    e_reference_df = mp2_energy_from_ovov(reference_df_4, mf.mo_energy, n_occ)
    e_reference_exact = mp2_energy_from_ovov(reference_exact_4, mf.mo_energy, n_occ)

    def metrics(eri):
        eri4 = eri.reshape(n_occ, n_vir, n_occ, n_vir)
        return {
            "relative_eri_error_exact_4c": _relative_error(eri, eri_reference_exact),
            "relative_eri_error_analytic_df": _relative_error(eri, eri_reference_df),
            "max_abs_eri_error_exact_4c": float(
                np.max(np.abs(eri - eri_reference_exact))
            ),
            "max_abs_eri_error_analytic_df": float(
                np.max(np.abs(eri - eri_reference_df))
            ),
            "dagger_residual": _relative_error(eri, eri.conj().T),
            "mp2_delta_exact_4c_mha": 1000.0 * (
                mp2_energy_from_ovov(eri4, mf.mo_energy, n_occ) - e_reference_exact
            ),
            "mp2_delta_analytic_df_mha": 1000.0 * (
                mp2_energy_from_ovov(eri4, mf.mo_energy, n_occ) - e_reference_df
            ),
        }

    result = {
        "case": dataclasses.asdict(case),
        "n_grid": int(weights.size),
        "n_pairs": int(n_occ * n_vir),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_sum": float(weights.sum()),
        "negative_weight_count": int(np.count_nonzero(weights < 0.0)),
        "zero_weight_count": int(np.count_nonzero(weights == 0.0)),
        "coords_sha256": _array_sha256(coords),
        "weights_sha256": _array_sha256(weights),
        "exact_4c_mp2_hartree": e_reference_exact,
        "analytic_df_mp2_hartree": e_reference_df,
        "analytic_df_approximation": {
            "relative_eri_error_vs_exact_4c": _relative_error(
                eri_reference_df, eri_reference_exact
            ),
            "max_abs_eri_error_vs_exact_4c": float(
                np.max(np.abs(eri_reference_df - eri_reference_exact))
            ),
            "mp2_delta_vs_exact_4c_mha": 1000.0 * (
                e_reference_df - e_reference_exact
            ),
        },
        "pyscf_dfmp2_hartree": float(mp.dfmp2.DFMP2(mf).run().e_corr),
        "single_ibp": metrics(eri_ibp),
        "single_ibp_dagger_projected": metrics(eri_ibp_projected),
        "single_ibp_coincident_pairs": int(ibp_coincident),
        "timing_seconds": {"single_ibp": ibp_seconds},
    }
    if eri_direct is not None:
        result["direct_1_over_r_offdiagonal_only"] = metrics(eri_direct)
        result["direct_coincident_pairs_omitted"] = int(direct_coincident)
        result["timing_seconds"]["direct_1_over_r_offdiagonal_only"] = direct_seconds

    if case.rank_factor is not None:
        requested_rank = min(
            int(math.ceil(case.rank_factor * coeff.shape[1])), n_occ * n_vir
        )
        pivots = np.asarray(select_sector_pivots(
            weight_mo_values(occ, weights),
            weight_mo_values(vir, weights),
            requested_rank,
            effective_rank_rtol=1e-8,
        ))
        occ_piv = occ[:, pivots]
        vir_piv = vir[:, pivots]
        chol, lower = prepare_normal_equations_solver(
            occ_piv, vir_piv, rcond=1e-12
        )
        theta = np.asarray(solve_normal_equations_batch_prepared(
            chol, lower, occ_piv, vir_piv, occ, vir
        ))
        gradient_theta = build_gradient_interpolation_vectors(
            occ, vir,
            grad_occ.transpose(1, 0, 2),
            grad_vir.transpose(1, 0, 2),
            pivots,
            rcond=1e-12,
            grid_batch_size=4096,
        )
        t0 = time.perf_counter()
        z_isdf, isdf_coincident = atom_centered_single_ibp_core(
            gradient_theta, theta, coords, weights,
            eval_block_size=case.eval_block_size,
            source_block_size=case.source_block_size,
        )
        isdf_seconds = time.perf_counter() - t0
        p_matrix = pair_collocation_at_pivots(occ_piv, vir_piv)
        eri_isdf = reconstruct_eri_block(p_matrix, z_isdf, p_matrix)
        result["isdf_single_ibp"] = {
            **metrics(eri_isdf),
            "requested_rank": requested_rank,
            "selected_rank": int(pivots.size),
            "pivots": pivots.tolist(),
            "coincident_pairs": int(isdf_coincident),
        }
        result["timing_seconds"]["isdf_single_ibp_core"] = isdf_seconds
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="H2O_ccpVDZ")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--pruning", choices=("default", "none"), default="default")
    parser.add_argument("--eval-block-size", type=int, default=128)
    parser.add_argument("--source-block-size", type=int, default=4096)
    parser.add_argument("--skip-direct-offdiagonal", action="store_true")
    parser.add_argument("--rank-factor", type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_atom_centered_case(AtomCenteredSingleIBPCase(
        system=args.system,
        grid_level=args.grid_level,
        pruning=args.pruning,
        eval_block_size=args.eval_block_size,
        source_block_size=args.source_block_size,
        include_direct_offdiagonal=not args.skip_direct_offdiagonal,
        rank_factor=args.rank_factor,
    ))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
