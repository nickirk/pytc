#!/usr/bin/env python
"""Measure H-chain orbital-product kernel block ranks without writing a cache.

The diagnostic compares multilevel block partitions built in physical grid
coordinates with partitions built in a small randomized orbital-product
feature sketch.  It measures exact-SVD ranks of both production ISDF kernels
on representative direct and admissible blocks; the sketch is used only to
partition the grid, never to replace the kernels whose ranks are reported.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import jax
import numpy as np
from pyscf import gto, scf

jax.config.update("jax_enable_x64", True)

from pytc.df.hierarchical_pivots import (
    orbital_product_feature_sketch,
    relative_block_rank_profile,
    relative_gradient_block_rank_profile,
)
from pytc.df.hmatrix import (
    ClusterNode,
    admissible_blocks,
    build_cluster_tree,
)
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
    parser.add_argument(
        "--profile-all-far",
        action="store_true",
        help="profile every far block for exact partition-level storage ratios",
    )
    return parser.parse_args()


def _box_distance(left: ClusterNode, right: ClusterNode) -> float:
    gap = np.maximum(0.0, np.maximum(left.lower - right.upper, right.lower - left.upper))
    return float(np.linalg.norm(gap))


def _box_diameter(node: ClusterNode) -> float:
    return float(np.linalg.norm(node.upper - node.lower))


def _representative(
    items: Sequence[tuple[int, int, float, int]], count: int
) -> list[tuple[int, int, float, int]]:
    if not items:
        return []
    target = min(count, len(items))
    chosen: dict[tuple[int, int], tuple[int, int, float, int]] = {}
    for ordered in (
        sorted(items, key=lambda item: item[2]),
        sorted(items, key=lambda item: item[3]),
    ):
        positions = np.unique(
            np.linspace(0, len(ordered) - 1, max(1, target // 2), dtype=int)
        )
        for position in positions:
            item = ordered[position]
            chosen[(item[0], item[1])] = item
    if len(chosen) < target:
        for item in sorted(items, key=lambda value: (value[3], value[2])):
            chosen[(item[0], item[1])] = item
            if len(chosen) == target:
                break
    return list(chosen.values())[:target]


def _summarize_profiles(profiles: list[dict], tolerances: Sequence[float]) -> dict:
    summary: dict[str, dict] = {}
    for tolerance in tolerances:
        key = f"{tolerance:.0e}"
        ranks = np.asarray([profile["ranks"][key] for profile in profiles], dtype=float)
        errors = np.asarray([profile["relative_errors"][key] for profile in profiles], dtype=float)
        fractions = np.asarray(
            [
                profile["ranks"][key] / min(profile["shape"])
                for profile in profiles
            ],
            dtype=float,
        )
        summary[key] = {
            "rank_min": int(np.min(ranks)),
            "rank_median": float(np.median(ranks)),
            "rank_mean": float(np.mean(ranks)),
            "rank_max": int(np.max(ranks)),
            "rank_fraction_median": float(np.median(fractions)),
            "rank_fraction_max": float(np.max(fractions)),
            "relative_error_max": float(np.max(errors)),
        }
    return summary


def _sampled_storage_ratios(
    profiles: list[dict], kernel: str, tolerances: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Measure low-rank factor storage relative to sampled dense blocks."""
    dense_entries = sum(profile["symmetric_entry_count"] for profile in profiles)
    ratios: dict[str, dict[str, float]] = {}
    for tolerance in tolerances:
        key = f"{tolerance:.0e}"
        factor_entries = 0
        fallback_entries = 0
        for profile in profiles:
            rows, cols = profile["shape"]
            copies = profile["symmetric_entry_count"] // (rows * cols)
            rank = profile[kernel]["ranks"][key]
            block_dense_entries = copies * rows * cols
            block_factor_entries = copies * rank * (rows + cols)
            factor_entries += block_factor_entries
            fallback_entries += min(block_dense_entries, block_factor_entries)
        ratios[key] = {
            "low_rank_factor": factor_entries / dense_entries,
            "exact_fallback": fallback_entries / dense_entries,
        }
    return ratios


def _principal_coordinates(values: np.ndarray, dimension: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("feature tree dimension must be positive")
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    count = min(dimension, values.shape[1])
    return left[:, :count] * singular_values[:count]


def profile_tree(
    features: np.ndarray,
    gradient_features: np.ndarray,
    coordinates: np.ndarray,
    *,
    leaf_size: int,
    eta: float,
    samples_per_class: int,
    tolerances: Sequence[float],
    include_samples: bool,
    profile_all_far: bool,
) -> dict:
    """Profile the actual symmetric multilevel H-matrix block partition."""
    started = time.perf_counter()
    nodes, root = build_cluster_tree(coordinates, leaf_size)
    leaves = [node for node in nodes if node.left is None]
    classes: dict[str, list[tuple[int, int, float, int]]] = {
        "diagonal": [],
        "near": [],
        "far": [],
    }
    partition = admissible_blocks(nodes, root, root, eta)
    for left_index, right_index, is_far in partition:
        if left_index > right_index:
            continue
        left = nodes[left_index]
        right = nodes[right_index]
        diameter = max(_box_diameter(left), _box_diameter(right))
        ratio = _box_distance(left, right) / max(diameter, np.finfo(float).tiny)
        area = len(left.indices) * len(right.indices)
        label = "diagonal" if left_index == right_index else ("far" if is_far else "near")
        classes[label].append((left_index, right_index, ratio, area))

    block_classes: dict[str, dict] = {}
    for label, pairs in classes.items():
        samples = (
            list(pairs)
            if label == "far" and profile_all_far
            else _representative(pairs, samples_per_class)
        )
        profiles: list[dict] = []
        for left_index, right_index, ratio, area in samples:
            rows = nodes[left_index].indices
            cols = nodes[right_index].indices
            phi_profile = relative_block_rank_profile(
                features, rows, cols, tolerances
            )
            gradient_profile = relative_gradient_block_rank_profile(
                features, gradient_features, rows, cols, tolerances
            )
            profiles.append(
                {
                    "left_node": left_index,
                    "right_node": right_index,
                    "separation_ratio": ratio,
                    "symmetric_entry_count": area if left_index == right_index else 2 * area,
                    "shape": phi_profile["shape"],
                    "phi": phi_profile,
                    "gradient": gradient_profile,
                }
            )
        entry_count = sum(
            area if left_index == right_index else 2 * area
            for left_index, right_index, _, area in pairs
        )
        shapes = np.asarray(
            [
                [len(nodes[left].indices), len(nodes[right].indices)]
                for left, right, _, _ in pairs
            ],
            dtype=int,
        )
        block_classes[label] = {
            "population": len(pairs),
            "symmetric_entry_count": int(entry_count),
            "matrix_entry_fraction": entry_count / features.shape[1] ** 2,
            "sample_count": len(profiles),
            "sampled_symmetric_entry_count": int(
                sum(profile["symmetric_entry_count"] for profile in profiles)
            ),
            "sampled_entry_fraction_of_class": (
                sum(profile["symmetric_entry_count"] for profile in profiles)
                / entry_count
                if entry_count
                else 0.0
            ),
            "block_shape_min": np.min(shapes, axis=0).tolist() if len(shapes) else [],
            "block_shape_median": np.median(shapes, axis=0).tolist() if len(shapes) else [],
            "block_shape_max": np.max(shapes, axis=0).tolist() if len(shapes) else [],
            "rank_summary": {
                kernel: _summarize_profiles(
                    [profile[kernel] for profile in profiles], tolerances
                )
                for kernel in ("phi", "gradient")
            }
            if profiles
            else {},
            "sampled_storage_ratio": {
                kernel: _sampled_storage_ratios(profiles, kernel, tolerances)
                for kernel in ("phi", "gradient")
            }
            if profiles
            else {},
        }
        if include_samples:
            block_classes[label]["samples"] = profiles

    return {
        "tree_wall_s": time.perf_counter() - started,
        "n_leaf": len(leaves),
        "leaf_size_min": min(len(leaf.indices) for leaf in leaves),
        "leaf_size_max": max(len(leaf.indices) for leaf in leaves),
        "partition": "symmetric multilevel admissible_blocks",
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
    gradient_features = (
        np.asarray(tc.grad_phi)[:, selection, :]
        * np.sqrt(np.abs(weights))[None, :, None]
    )
    if features.dtype != np.float64:
        raise RuntimeError(f"diagnostic requires float64 features, got {features.dtype}")
    if gradient_features.dtype != np.float64:
        raise RuntimeError(
            f"diagnostic requires float64 gradients, got {gradient_features.dtype}"
        )
    setup_wall_s = time.perf_counter() - setup_started

    geometry = profile_tree(
        features,
        gradient_features,
        coordinates,
        leaf_size=args.leaf_size,
        eta=args.eta,
        samples_per_class=args.samples_per_class,
        tolerances=args.tolerances,
        include_samples=not args.summary_only,
        profile_all_far=args.profile_all_far,
    )
    sketch_started = time.perf_counter()
    feature_sketch = orbital_product_feature_sketch(
        features, dimension=args.sketch_dimension, seed=args.seed
    )
    feature_coordinates = _principal_coordinates(feature_sketch, args.feature_tree_dimension)
    feature_sketch_wall_s = time.perf_counter() - sketch_started
    feature = profile_tree(
        features,
        gradient_features,
        feature_coordinates,
        leaf_size=args.leaf_size,
        eta=args.eta,
        samples_per_class=args.samples_per_class,
        tolerances=args.tolerances,
        include_samples=not args.summary_only,
        profile_all_far=args.profile_all_far,
    )

    result = {
        "system": f"H{args.n_atom}",
        "basis": args.basis,
        "n_orb": int(features.shape[0]),
        "dtype": str(features.dtype),
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
            "profile_all_far": args.profile_all_far,
            "tolerances": list(args.tolerances),
        },
        "geometry_tree": geometry,
        "feature_tree": {
            "feature_sketch_wall_s": feature_sketch_wall_s,
            "description": "principal coordinates of a random orbital-pair sketch; exact kernel ranks",
            **feature,
        },
        "scope": "CPU-only exact phi/gradient multilevel rank diagnostic; no ISDF cache or CCSD",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
