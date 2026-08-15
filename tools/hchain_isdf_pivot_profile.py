#!/usr/bin/env python
"""Profile the production ISDF pivot selectors on a local H-chain CPU case."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

jax.config.update("jax_enable_x64", True)

from pytc.df.isdf import _pivoted_cholesky_grad, _pivoted_cholesky_phi
from pytc.jastrow import BoysHandy
from pytc.tc import TC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atom", type=int, default=10)
    parser.add_argument("--r-bohr", type=float, default=1.4)
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--rank", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _memory_receipt(compiled) -> dict[str, int]:
    stats = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    receipt = {name: int(getattr(stats, name)) for name in names}
    receipt["argument_output_temp_bytes"] = sum(
        receipt[name] for name in names[:3]
    )
    return receipt


def _compile_and_time(lowered, dynamic_args: tuple, repeats: int) -> tuple:
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_wall_s = time.perf_counter() - started

    started = time.perf_counter()
    result = compiled(*dynamic_args)
    jax.block_until_ready(result)
    first_execution_wall_s = time.perf_counter() - started

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compiled(*dynamic_args)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    timing = {
        "compile_wall_s": compile_wall_s,
        "first_execution_wall_s": first_execution_wall_s,
        "steady_samples_s": samples,
        "steady_median_s": float(np.median(samples)),
        "steady_min_s": float(np.min(samples)),
    }
    return compiled, result, timing


@partial(jax.jit, static_argnames=("steps",))
def _repeat_phi_columns(features: jnp.ndarray, steps: int) -> jnp.ndarray:
    accumulator = jnp.zeros(features.shape[1], dtype=features.dtype)

    def body(step, value):
        pivot = step % features.shape[1]
        overlap = features.T @ features[:, pivot]
        return value + overlap * overlap

    return jax.lax.fori_loop(0, steps, body, accumulator)


@partial(jax.jit, static_argnames=("steps",))
def _repeat_gradient_columns(
    features: jnp.ndarray, gradients: jnp.ndarray, steps: int
) -> jnp.ndarray:
    accumulator = jnp.zeros(features.shape[1], dtype=features.dtype)

    def body(step, value):
        pivot = step % features.shape[1]
        orbital = features.T @ features[:, pivot]
        gradient = sum(
            gradients[:, :, component].T @ gradients[:, pivot, component]
            for component in range(3)
        )
        return value + orbital * gradient

    return jax.lax.fori_loop(0, steps, body, accumulator)


@partial(jax.jit, static_argnames=("steps",))
def _repeat_factor_updates(factor: jnp.ndarray, steps: int) -> jnp.ndarray:
    accumulator = jnp.zeros(factor.shape[0], dtype=factor.dtype)

    def body(step, value):
        return value + factor @ factor[step % factor.shape[0]]

    return jax.lax.fori_loop(0, steps, body, accumulator)


@partial(jax.jit, static_argnames=("steps",))
def _repeat_residual_updates(
    diagonal: jnp.ndarray,
    kernel_column: jnp.ndarray,
    factor_column: jnp.ndarray,
    steps: int,
) -> jnp.ndarray:
    selected = jnp.zeros(diagonal.shape[0], dtype=bool)

    def body(_, state):
        residual, mask = state
        pivot = jnp.argmax(jnp.where(mask, -jnp.inf, residual))
        pivot_value = residual[pivot]
        column = (kernel_column - factor_column) * jax.lax.rsqrt(
            jnp.maximum(pivot_value, 1e-30)
        )
        residual = jnp.maximum(residual - 1e-12 * column * column, 0.0)
        residual = residual.at[pivot].set(0.0)
        mask = mask.at[pivot].set(True)
        return residual, mask

    return jax.lax.fori_loop(0, steps, body, (diagonal, selected))[0]


def _profile_component(function, static_args: tuple, dynamic_args: tuple, repeats: int) -> dict:
    lowered = function.lower(*static_args)
    _, _, timing = _compile_and_time(lowered, dynamic_args, repeats)
    return timing


def main() -> None:
    args = parse_args()
    if min(args.n_atom, args.r_bohr, args.rank, args.repeats) <= 0:
        raise SystemExit("all controls must be positive")

    setup_started = time.perf_counter()
    atom = "; ".join(f"H 0 0 {index * args.r_bohr}" for index in range(args.n_atom))
    mol = gto.M(atom=atom, basis=args.basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    tc = TC.from_pyscf(mf, BoysHandy.create(mol), grid_lvl=args.grid_level)
    weights_sqrt = jnp.sqrt(jnp.abs(tc.weights))
    features = tc.phi * weights_sqrt[None, :]
    gradients = tc.grad_phi * weights_sqrt[None, :, None]
    if features.dtype != jnp.float64 or gradients.dtype != jnp.float64:
        raise RuntimeError(
            f"profile requires float64 inputs, got {features.dtype}/{gradients.dtype}"
        )
    n_orb, n_grid = features.shape
    rank = min(args.rank, n_grid)
    phi_diagonal = jnp.sum(features**2, axis=0) ** 2
    grad_diagonal = jnp.sum(features**2, axis=0) * jnp.sum(
        gradients**2, axis=(0, 2)
    )
    phi_shift = 1e-12 * jnp.max(jnp.abs(phi_diagonal))
    grad_shift = 1e-12 * jnp.max(jnp.abs(grad_diagonal))
    setup_wall_s = time.perf_counter() - setup_started

    phi_compiled, phi_pivots, phi_timing = _compile_and_time(
        _pivoted_cholesky_phi.lower(features, rank, phi_shift),
        (features, phi_shift),
        args.repeats,
    )
    grad_compiled, grad_pivots, grad_timing = _compile_and_time(
        _pivoted_cholesky_grad.lower(features, gradients, rank, grad_shift),
        (features, gradients, grad_shift),
        args.repeats,
    )

    factor = jnp.sin(jnp.arange(n_grid * rank, dtype=features.dtype) * 1e-4)
    factor = factor.reshape(n_grid, rank)
    diagonal = jnp.linspace(1.0, 2.0, n_grid, dtype=features.dtype)
    kernel_column = jnp.linspace(0.1, 0.2, n_grid, dtype=features.dtype)
    factor_column = jnp.linspace(0.05, 0.1, n_grid, dtype=features.dtype)
    components = {
        "phi_kernel_columns": _profile_component(
            _repeat_phi_columns,
            (features, rank),
            (features,),
            args.repeats,
        ),
        "gradient_kernel_columns": _profile_component(
            _repeat_gradient_columns,
            (features, gradients, rank),
            (features, gradients),
            args.repeats,
        ),
        "factor_updates_per_channel": _profile_component(
            _repeat_factor_updates,
            (factor, rank),
            (factor,),
            args.repeats,
        ),
        "residual_updates_per_channel": _profile_component(
            _repeat_residual_updates,
            (diagonal, kernel_column, factor_column, rank),
            (diagonal, kernel_column, factor_column),
            args.repeats,
        ),
    }
    column_wall = (
        components["phi_kernel_columns"]["steady_median_s"]
        + components["gradient_kernel_columns"]["steady_median_s"]
    )
    factor_wall = 2 * components["factor_updates_per_channel"]["steady_median_s"]
    residual_wall = 2 * components["residual_updates_per_channel"]["steady_median_s"]
    component_wall = column_wall + factor_wall + residual_wall

    phi_indices = np.asarray(phi_pivots)
    grad_indices = np.asarray(grad_pivots)
    result = {
        "system": f"H{args.n_atom}",
        "basis": args.basis,
        "device": str(jax.devices()[0]),
        "dtype": str(features.dtype),
        "n_orb": int(n_orb),
        "n_grid": int(n_grid),
        "rank": int(rank),
        "setup_wall_s": setup_wall_s,
        "production": {
            "phi": {
                "timing": phi_timing,
                "unique_pivots": int(np.unique(phi_indices).size),
                "memory": _memory_receipt(phi_compiled),
                "xla_cost_analysis_one_loop_body": phi_compiled.cost_analysis(),
            },
            "gradient": {
                "timing": grad_timing,
                "unique_pivots": int(np.unique(grad_indices).size),
                "memory": _memory_receipt(grad_compiled),
                "xla_cost_analysis_one_loop_body": grad_compiled.cost_analysis(),
            },
            "fused_unique_pivots": int(
                np.unique(np.concatenate((phi_indices, grad_indices))).size
            ),
            "factor_storage_bytes": int(n_grid * rank * features.dtype.itemsize),
        },
        "isolated_components": components,
        "isolated_component_share": {
            "kernel_columns": column_wall / component_wall,
            "factor_updates": factor_wall / component_wall,
            "residual_selection_and_update": residual_wall / component_wall,
            "kernel_only_zero_cost_speedup_ceiling": 1.0
            / (1.0 - column_wall / component_wall),
            "warning": "isolated fused-loop microbenchmarks are attribution proxies, not additive production timers",
        },
        "scope": "CPU-only production pivot profile; no ISDF solve, cache, K1/K3, X, or CCSD",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
