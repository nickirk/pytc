#!/usr/bin/env python
"""CPU-only H-chain diagnostic for hierarchical ISDF-pivot screening.

It reports how terminal geometry-tree leaves generate local candidates and
compares a screened or leaf-refined Cholesky selection with the global
reference.  It neither writes an ISDF cache nor launches CCSD.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np
from pyscf import gto, scf

jax.config.update("jax_enable_x64", True)

from pytc.df.hierarchical_pivots import (
    global_pivoted_cholesky,
    hierarchical_pivoted_cholesky,
    orbital_product_projection_error,
)
from pytc.df.hmatrix import build_cluster_tree
from pytc.jastrow import BoysHandy
from pytc.tc import TC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atom", type=int, default=4)
    parser.add_argument("--r-bohr", type=float, default=1.4)
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--max-grid", type=int, default=256)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--local-rank", type=int, default=4)
    parser.add_argument(
        "--skip-projection-error",
        action="store_true",
        help="skip the small-grid product-space least-squares diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.n_atom, args.r_bohr, args.max_grid, args.rank, args.leaf_size, args.local_rank) <= 0:
        raise SystemExit("all controls must be positive")
    setup_start = time.perf_counter()
    atom = "; ".join(f"H 0 0 {index * args.r_bohr}" for index in range(args.n_atom))
    mol = gto.M(atom=atom, basis=args.basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    tc = TC.from_pyscf(mf, BoysHandy.create(mol), grid_lvl=args.grid_level)
    selection = np.unique(
        np.linspace(0, len(tc.grid_points) - 1, min(args.max_grid, len(tc.grid_points)), dtype=int)
    )
    points = np.asarray(tc.grid_points)[selection]
    weights = np.asarray(tc.weights)[selection]
    features = np.asarray(tc.phi)[:, selection] * np.sqrt(np.abs(weights))[None, :]
    if features.dtype != np.float64:
        raise RuntimeError(f"diagnostic requires float64 features, got {features.dtype}")
    rank = min(args.rank, features.shape[1])
    setup_wall_s = time.perf_counter() - setup_start

    tree_start = time.perf_counter()
    nodes, _ = build_cluster_tree(points, args.leaf_size)
    leaves = [node for node in nodes if node.left is None]
    tree_wall_s = time.perf_counter() - tree_start
    initial_kernel_trace = float(np.sum(np.sum(features * features, axis=0) ** 2))
    global_start = time.perf_counter()
    global_result = global_pivoted_cholesky(features, rank)
    global_wall_s = time.perf_counter() - global_start
    screened_start = time.perf_counter()
    screened = hierarchical_pivoted_cholesky(
        features,
        points,
        rank,
        leaf_size=args.leaf_size,
        local_rank=args.local_rank,
        refine_with_leaf_maxima=False,
    )
    screened_wall_s = time.perf_counter() - screened_start
    refined_start = time.perf_counter()
    refined = hierarchical_pivoted_cholesky(
        features,
        points,
        rank,
        leaf_size=args.leaf_size,
        local_rank=args.local_rank,
        refine_with_leaf_maxima=True,
    )
    refined_wall_s = time.perf_counter() - refined_start

    def quality(result: dict) -> dict:
        residual_fraction = float(
            np.sqrt(np.sum(result["residual_diagonal"]) / max(initial_kernel_trace, np.finfo(float).tiny))
        )
        if args.skip_projection_error:
            return {"residual_trace_fraction": residual_fraction}
        return {
            "residual_trace_fraction": residual_fraction,
            "projection_error": orbital_product_projection_error(features, result["pivots"]),
        }

    result = {
        "system": f"H{args.n_atom}",
        "basis": args.basis,
        "n_orb": int(features.shape[0]),
        "dtype": str(features.dtype),
        "n_grid_source": int(len(tc.grid_points)),
        "n_grid_diagnostic": int(len(selection)),
        "controls": {
            "rank_requested": args.rank,
            "rank_selected": int(len(global_result["pivots"])),
            "leaf_size_cap": args.leaf_size,
            "local_rank": args.local_rank,
        },
        "setup_wall_s": setup_wall_s,
        "tree": {
            "rule": "longest-axis stable-sort median split",
            "n_leaf": len(leaves),
            "wall_s": tree_wall_s,
            "leaf_size_min": min(len(leaf.indices) for leaf in leaves),
            "leaf_size_max": max(len(leaf.indices) for leaf in leaves),
            "leaf_diameter_min": min(float(np.linalg.norm(leaf.upper - leaf.lower)) for leaf in leaves),
            "leaf_diameter_max": max(float(np.linalg.norm(leaf.upper - leaf.lower)) for leaf in leaves),
        },
        "screened": {
            "metadata": screened["metadata"],
            "wall_s": screened_wall_s,
            **quality(screened),
            "pivot_overlap_with_global": int(
                len(set(screened["pivots"]).intersection(global_result["pivots"]))
            ),
        },
        "refined": {
            "metadata": refined["metadata"],
            "wall_s": refined_wall_s,
            **quality(refined),
            "matches_global_pivots": bool(
                np.array_equal(refined["pivots"], global_result["pivots"])
            ),
        },
        "global": {"wall_s": global_wall_s, **quality(global_result)},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
