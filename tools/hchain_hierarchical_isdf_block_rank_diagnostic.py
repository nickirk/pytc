#!/usr/bin/env python
"""Measure H-chain orbital-product kernel block ranks without writing a cache.

The diagnostic compares terminal leaves built in physical grid coordinates
with leaves built in a small randomized orbital-product feature sketch.  It
measures exact-SVD ranks of representative near and geometrically admissible
far blocks; the sketch is used only to partition the grid, never to replace
the kernel whose ranks are reported.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import numpy as np
from pyscf import gto, scf

from pytc.df.hierarchical_pivots import (
    orbital_product_feature_sketch,
    relative_block_rank_profile,
)
from pytc.df.hmatrix import ClusterNode, build_cluster_tree
from pytc.jastrow import BoysHandy
from pytc.tc import TC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atom", type=int, default=10)
    parser.add_argument("--r-bohr", type=float, default=1.4)
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--max-grid", type=int, default=100000)
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="far if box distance exceeds eta times the larger box diameter",
    )
    parser.add_argument("--sketch-dimension", type=int, default=16)
    parser.add_argument(
        "--feature-tree-dimension",
        type=int,
        default=3,
        help="principal coordinates retained from the orbital-product sketch for box tests",
    )
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--tolerances", type=float, nargs="+", default=(1e-4, 1e-6, 1e-8))
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit individual sampled-block records from the JSON receipt",
    )
    return parser.parse_args()


def _box_distance(left: ClusterNode, right: ClusterNode) -> float:
    gap = np.maximum(0.0, np.maximum(left.lower - right.upper, right.lower - left.upper))
    return float(np.linalg.norm(gap))


def _box_diameter(node: ClusterNode) -> float:
    return float(np.linalg.norm(node.upper - node.lower))


def _representative(items: Sequence[tuple[int, int, float]], count: int) -> list[tuple[int, int, float]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[2])
    positions = np.unique(np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int))
    return [ordered[position] for position in positions]


def _summarize_profiles(profiles: list[dict], tolerances: Sequence[float]) -> dict:
    summary: dict[str, dict] = {}
    for tolerance in tolerances:
        key = f"{tolerance:.0e}"
        ranks = np.asarray([profile["ranks"][key] for profile in profiles], dtype=float)
        errors = np.asarray([profile["relative_errors"][key] for profile in profiles], dtype=float)
        summary[key] = {
            "rank_min": int(np.min(ranks)),
            "rank_median": float(np.median(ranks)),
            "rank_mean": float(np.mean(ranks)),
            "rank_max": int(np.max(ranks)),
            "relative_error_max": float(np.max(errors)),
        }
    return summary


def _principal_coordinates(values: np.ndarray, dimension: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("feature tree dimension must be positive")
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    count = min(dimension, values.shape[1])
    return left[:, :count] * singular_values[:count]


def profile_tree(
    features: np.ndarray,
    coordinates: np.ndarray,
    *,
    leaf_size: int,
    eta: float,
    samples_per_class: int,
    tolerances: Sequence[float],
    include_samples: bool,
) -> dict:
    """Profile representative terminal blocks for one deterministic tree."""
    started = time.perf_counter()
    nodes, _ = build_cluster_tree(coordinates, leaf_size)
    leaves = [node for node in nodes if node.left is None]
    diagonal: list[tuple[int, int, float]] = []
    near: list[tuple[int, int, float]] = []
    far: list[tuple[int, int, float]] = []
    for left_index, left in enumerate(leaves):
        diagonal.append((left_index, left_index, 0.0))
        for right_index in range(left_index + 1, len(leaves)):
            right = leaves[right_index]
            diameter = max(_box_diameter(left), _box_diameter(right))
            ratio = _box_distance(left, right) / max(diameter, np.finfo(float).tiny)
            (far if ratio > eta else near).append((left_index, right_index, ratio))

    classes = {"diagonal": diagonal, "near": near, "far": far}
    block_classes: dict[str, dict] = {}
    for label, pairs in classes.items():
        samples = _representative(pairs, samples_per_class)
        profiles: list[dict] = []
        for left_index, right_index, ratio in samples:
            profile = relative_block_rank_profile(
                features, leaves[left_index].indices, leaves[right_index].indices, tolerances
            )
            profiles.append(
                {
                    "left_leaf": left_index,
                    "right_leaf": right_index,
                    "separation_ratio": ratio,
                    **profile,
                }
            )
        block_classes[label] = {
            "population": len(pairs),
            "sample_count": len(profiles),
            "rank_summary": _summarize_profiles(profiles, tolerances) if profiles else {},
        }
        if include_samples:
            block_classes[label]["samples"] = profiles

    return {
        "tree_wall_s": time.perf_counter() - started,
        "n_leaf": len(leaves),
        "leaf_size_min": min(len(leaf.indices) for leaf in leaves),
        "leaf_size_max": max(len(leaf.indices) for leaf in leaves),
        "admissibility": "distance > eta * max(box_diameter)",
        "eta": eta,
        "classes": block_classes,
    }


def main() -> None:
    args = parse_args()
    controls = (
        args.n_atom,
        args.r_bohr,
        args.max_grid,
        args.leaf_size,
        args.eta,
        args.sketch_dimension,
        args.feature_tree_dimension,
        args.samples_per_class,
        *args.tolerances,
    )
    if min(controls) <= 0.0:
        raise SystemExit("all controls and tolerances must be positive")

    setup_started = time.perf_counter()
    atom = "; ".join(f"H 0 0 {index * args.r_bohr}" for index in range(args.n_atom))
    mol = gto.M(atom=atom, basis=args.basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    tc = TC.from_pyscf(mf, BoysHandy.create(mol), grid_lvl=args.grid_level)
    selection = np.unique(
        np.linspace(0, len(tc.grid_points) - 1, min(args.max_grid, len(tc.grid_points)), dtype=int)
    )
    coordinates = np.asarray(tc.grid_points)[selection]
    weights = np.asarray(tc.weights)[selection]
    features = np.asarray(tc.phi)[:, selection] * np.sqrt(np.abs(weights))[None, :]
    setup_wall_s = time.perf_counter() - setup_started

    geometry = profile_tree(
        features,
        coordinates,
        leaf_size=args.leaf_size,
        eta=args.eta,
        samples_per_class=args.samples_per_class,
        tolerances=args.tolerances,
        include_samples=not args.summary_only,
    )
    sketch_started = time.perf_counter()
    feature_sketch = orbital_product_feature_sketch(
        features, dimension=args.sketch_dimension, seed=args.seed
    )
    feature_coordinates = _principal_coordinates(feature_sketch, args.feature_tree_dimension)
    feature_sketch_wall_s = time.perf_counter() - sketch_started
    feature = profile_tree(
        features,
        feature_coordinates,
        leaf_size=args.leaf_size,
        eta=args.eta,
        samples_per_class=args.samples_per_class,
        tolerances=args.tolerances,
        include_samples=not args.summary_only,
    )

    result = {
        "system": f"H{args.n_atom}",
        "basis": args.basis,
        "n_orb": int(features.shape[0]),
        "n_grid_source": int(len(tc.grid_points)),
        "n_grid_diagnostic": int(len(selection)),
        "setup_wall_s": setup_wall_s,
        "controls": {
            "leaf_size": args.leaf_size,
            "eta": args.eta,
            "sketch_dimension": args.sketch_dimension,
            "feature_tree_dimension": args.feature_tree_dimension,
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "tolerances": list(args.tolerances),
        },
        "geometry_tree": geometry,
        "feature_tree": {
            "feature_sketch_wall_s": feature_sketch_wall_s,
            "description": "principal coordinates of a random orbital-pair sketch; exact kernel ranks",
            **feature,
        },
        "scope": "CPU-only rank diagnostic; no ISDF cache, kernel build, or CCSD calculation",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
