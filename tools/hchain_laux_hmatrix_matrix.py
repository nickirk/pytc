#!/usr/bin/env python
"""Run the four-leg H-chain L_aux/Tucker-X energy matrix.

The matrix is deliberately built from one fresh direct full-X cache and one
static-cache clone.  Thus the direct and hierarchical L_aux legs use the
same molecular-orbital gauge and ISDF decomposition.  Each cache supplies a
full-X solve and a fixed-rank Tucker-X solve, isolating both approximations
and their compound effect:

    direct L_aux + full X       hierarchical L_aux + full X
    direct L_aux + Tucker-M X   hierarchical L_aux + Tucker-M X

The command is a measurement driver, not a recommendation of an H-matrix
tolerance.  It rejects cache/path/provenance errors and emits a single JSON
receipt with independent factor and solver timings for every leg.
"""

from __future__ import annotations

import argparse
from functools import reduce
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import BoysHandy, CompositeJastrow, NuclearCusp
from pytc.solver import jax_xtc_ccsd
from pytc.utils import cache_state
from pytc.vmc.mcmc_utils import load_optimization_history


AVERAGE_LAST_N = 500
X_SWITCHES = (
    "PYTC_XTC_DROP_X",
    "PYTC_XTC_DROP_X_NORMAL_ORDER",
    "PYTC_XTC_DROP_X_RESIDUAL",
)
DERIVED_DATASETS = {"K1_kernel", "K3_kernel", "L_aux", "H_aux", "D", "X", "X_rm"}
DERIVED_ATTRIBUTES = {
    "pytc_kmat_kernel_mode",
    "pytc_laux_gradient_mode",
    "pytc_xtc_x_mode",
    "pytc_laux_hmatrix_mode",
    "pytc_laux_hmatrix_leaf_size",
    "pytc_laux_hmatrix_eta",
    "pytc_laux_hmatrix_tolerance",
    "pytc_laux_hmatrix_max_rank",
    "pytc_laux_hmatrix_heldout_size",
    "pytc_laux_hmatrix_near_blocks",
    "pytc_laux_hmatrix_far_blocks",
    "pytc_laux_hmatrix_far_fallbacks",
    "pytc_laux_hmatrix_far_rank_max",
    "pytc_laux_hmatrix_far_rank_mean",
    "pytc_laux_hmatrix_far_factor_storage",
    "pytc_laux_hmatrix_heldout_error_max",
}


def geometry(r_bohr: float, n_atom: int) -> str:
    return "; ".join(f"H 0 0 {index * r_bohr}" for index in range(n_atom))


def eig_rhf(h, s):
    values, vectors = np.linalg.eigh(s)
    orthogonalizer = vectors[:, values > 1e-7] / np.sqrt(values[values > 1e-7])
    energies, coeff = np.linalg.eigh(
        reduce(np.dot, (orthogonalizer.T, h, orthogonalizer))
    )
    return energies, np.dot(orthogonalizer, coeff)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for panel in iter(lambda: handle.read(1 << 20), b""):
            digest.update(panel)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for panel in iter(lambda: handle.read(1 << 20), b""):
            digest.update(panel)
    return digest.hexdigest()


def json_value(value):
    """Convert scalar HDF5 metadata to an ordinary JSON value."""
    return value.item() if isinstance(value, np.generic) else value


def clean_pytc_revision(expected: str) -> str:
    import pytc

    repo = Path(pytc.__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if actual != expected or dirty:
        raise SystemExit(
            "FATAL: PyTC provenance mismatch: "
            f"expected={expected}, actual={actual}, dirty={bool(dirty)}"
        )
    return actual


def load_jastrow(path: Path):
    history = load_optimization_history(str(path))
    stacked = history["params"]
    n_saved = jax.tree_util.tree_leaves(stacked)[0].shape[0]
    n_used = min(AVERAGE_LAST_N, n_saved)
    averaged = jax.tree_util.tree_map(
        lambda value: np.mean(value[-n_used:], axis=0), stacked
    )
    return averaged[0], int(n_saved), int(n_used)


def device_receipt(expected_substring: str | None) -> dict:
    devices = jax.local_devices()
    receipt = {
        "local_device_count": len(devices),
        "platform": None if not devices else devices[0].platform,
        "kind": None if not devices else str(getattr(devices[0], "device_kind", "")),
    }
    if expected_substring is not None:
        if len(devices) != 1 or expected_substring.lower() not in receipt["kind"].lower():
            raise SystemExit(
                "FATAL: one matching device is required: "
                f"requested={expected_substring!r}, receipt={receipt}"
            )
    return receipt


def required_xla_cache_dir() -> str:
    raw = os.environ.get("PYTC_XLA_CACHE_DIR")
    if not raw:
        raise SystemExit("FATAL: set PYTC_XLA_CACHE_DIR to a job-scoped absolute path")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit("FATAL: PYTC_XLA_CACHE_DIR must be absolute")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def source_receipt(isdf, expected_path: Path) -> dict:
    x_kernel = isdf.isdf_kernels.get("X")
    if not isinstance(x_kernel, h5py.Dataset):
        raise SystemExit("FATAL: full-X leg did not retain a streamed X dataset")
    source = Path(x_kernel.file.filename).resolve()
    if source != expected_path.resolve():
        raise SystemExit(f"FATAL: full-X store mismatch: {source} != {expected_path.resolve()}")
    mode = x_kernel.file.attrs.get("pytc_xtc_x_mode", "missing")
    if isinstance(mode, bytes):
        mode = mode.decode()
    shape = (isdf.n_orb, isdf.n_orb, isdf.phi_isdf.shape[1])
    if mode != "full" or tuple(x_kernel.shape) != shape:
        raise SystemExit(
            f"FATAL: expected full X shape/mode {shape}/full, got {x_kernel.shape}/{mode!r}"
        )
    sample = np.asarray(
        x_kernel[: min(4, shape[0]), : min(4, shape[1]), : min(4, shape[2])]
    )
    if not np.any(sample):
        raise SystemExit("FATAL: full-X source sample is zero")
    hmatrix = {
        key.removeprefix("pytc_laux_hmatrix_"): json_value(value)
        for key, value in x_kernel.file.attrs.items()
        if key.startswith("pytc_laux_hmatrix_")
    }
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "mode": str(mode),
        "x_shape": list(shape),
        "x_sample_max_abs": float(np.max(np.abs(sample))),
        "isdf_static_keys": sorted(
            key for key in x_kernel.file if key not in DERIVED_DATASETS
        ),
        "hmatrix": hmatrix or None,
    }


def clone_static_isdf_cache(source: h5py.File, target: Path) -> None:
    """Copy the exact ISDF/gauge state but no derived L_aux or X kernels."""
    if target.exists():
        raise SystemExit(f"FATAL: refusing to overwrite static cache {target}")
    source.flush()
    with h5py.File(target, "x") as destination:
        for key, value in source.attrs.items():
            if key not in DERIVED_ATTRIBUTES:
                destination.attrs[key] = value
        for key in source:
            if key not in DERIVED_DATASETS:
                source.copy(key, destination)
    with h5py.File(target, "r") as check:
        leaked = sorted(DERIVED_DATASETS.intersection(check.keys()))
        required = {"xi_phi", "xi_grad", "pivots", "phi_isdf", "grad_phi_isdf"}
        missing = sorted(required.difference(check.keys()))
    if leaked or missing:
        raise SystemExit(f"FATAL: static-cache clone invalid: leaked={leaked}, missing={missing}")


def normal_order_receipt(isdf, params) -> dict:
    """Check the installed Delta-U, Fock correction, and scalar identity."""
    kernels = isdf.isdf_kernels
    all_orbitals = slice(None)
    occupied = slice(0, isdf.nocc)
    pqii = np.asarray(
        isdf._assemble_delta_u_tile(
            kernels, (all_orbitals, all_orbitals, occupied, occupied)
        )
    )
    piiq = np.asarray(
        isdf._assemble_delta_u_tile(
            kernels, (all_orbitals, occupied, occupied, all_orbitals)
        )
    )
    fock_delta = 2.0 * np.einsum("pqii->pq", pqii) - np.einsum("piiq->pq", piiq)
    delta_h = np.asarray(isdf.get_delta_h(params))
    fock_identity = float(np.max(np.abs(delta_h + 0.5 * fock_delta)))
    dm1 = np.asarray(isdf._get_mf_dm())
    e0 = float(isdf.get_const(params, delta_h=delta_h))
    scalar_identity = abs(
        (e0 - float(isdf.energy_nuc))
        - float(-2.0 / 3.0 * np.einsum("qp,pq->", delta_h, dm1))
    )
    if fock_identity > 1e-10 or scalar_identity > 1e-10:
        raise SystemExit(
            "FATAL: normal-order identity failed: "
            f"fock={fock_identity:.3e}, scalar={scalar_identity:.3e}"
        )
    return {
        "delta_h_fock_identity_max_abs": fock_identity,
        "e0_delta_h_identity_abs": scalar_identity,
    }


def solve_ccsd(mf, isdf, params, max_memory_mb: int, gpu_max_memory_mb: int, panel: int) -> dict:
    normal_order = normal_order_receipt(isdf, params)
    eris_start = time.perf_counter()
    cc = jax_xtc_ccsd.RCCSD(
        mf,
        isdf,
        params,
        max_memory=max_memory_mb,
        gpu_max_memory=gpu_max_memory_mb,
        on_the_fly_vvvv=False,
        vvvv_p_block_size=panel,
        vvvv_r_block_size=panel,
    )
    eris = cc.ao2mo()
    eris_s = time.perf_counter() - eris_start
    try:
        cc.max_cycle = 100
        cc.level_shift = 0.1
        solve_start = time.perf_counter()
        e_corr = float(cc.kernel(eris=eris)[0])
        solve_s = time.perf_counter() - solve_start
        if not cc.converged:
            raise SystemExit("FATAL: xTC-CCSD did not converge")
        return {
            "normal_order": normal_order,
            "e_corr": e_corr,
            "e_tot": float(cc.e_tot),
            "cycles": int(getattr(cc, "cycles", -1)),
            "wall_s": {"eris": eris_s, "ccsd": solve_s},
        }
    finally:
        eris.close()


def persist_tucker(path: Path, factors: dict, source: dict, revision: str) -> dict:
    if path.exists():
        raise SystemExit(f"FATAL: refusing to overwrite factor file {path}")
    u = np.asarray(factors["U"])
    z = np.asarray(factors["Z"])
    with h5py.File(path, "x") as handle:
        handle.create_dataset("U", data=u)
        handle.create_dataset("Z", data=z)
        handle.attrs["representation"] = "X[r,s,c]=U[r,a] Z[a,b,c] U[s,b]"
        handle.attrs["source_store"] = source["path"]
        handle.attrs["source_sha256"] = source["sha256"]
        handle.attrs["pytc_commit"] = revision
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "u_shape": list(u.shape),
        "z_shape": list(z.shape),
        "bytes": int(u.nbytes + z.nbytes),
    }


def tucker_view(isdf, params, source: dict, factor_path: Path, revision: str, args):
    l_aux = isdf.isdf_kernels["L_aux"]
    basis_start = time.perf_counter()
    u = isdf.select_tucker_x_orbital_basis(
        params,
        args.rank,
        oversampling=args.oversampling,
        seed=args.seed,
        batch_size=args.laux_batch_size,
        L_aux=l_aux,
        orb_block_size=args.orbital_panel,
        host_grid_block_size=args.host_grid_block_size,
    )
    basis_s = time.perf_counter() - basis_start
    core_start = time.perf_counter()
    factors = isdf.compute_tucker_x_core(
        params,
        u,
        batch_size=args.laux_batch_size,
        L_aux=l_aux,
        host_grid_block_size=args.host_grid_block_size,
    )
    core_s = time.perf_counter() - core_start
    receipt = persist_tucker(factor_path, factors, source, revision)
    kernels = dict(isdf.isdf_kernels)
    kernels.pop("X", None)
    kernels.pop("X_rm", None)
    kernels["X_tucker"] = factors
    view = isdf.replace(isdf_kernels=kernels)
    if "X" in view.isdf_kernels:
        raise SystemExit("FATAL: Tucker-X view still exposes dense X")
    receipt["wall_s"] = {
        "basis_sketch": basis_s,
        "projected_core": core_s,
        "factor_build": basis_s + core_s,
    }
    return view, receipt


def build_isdf(mf, jastrow, params, store: Path, args, *, hmatrix: bool):
    obj = xtc.ISDFXTC.from_xtc(
        xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2),
        n_rank=mf.mo_coeff.shape[1] * 12,
        save_path=str(store),
        ls_grid_batch_size=12_288,
    )
    start = time.perf_counter()
    result = obj.isdf(
        params,
        save_path=str(store),
        batch_size=args.laux_batch_size,
        orb_block_size=args.orbital_panel,
        host_grid_block_size=args.host_grid_block_size,
        reuse_aux_kernels=True,
        use_laux_fast_grad=True,
        use_laux_exact_split=True,
        use_laux_hmatrix=hmatrix,
        laux_hmatrix_leaf_size=args.hmatrix_leaf_size,
        laux_hmatrix_eta=args.hmatrix_eta,
        laux_hmatrix_tolerance=args.hmatrix_tolerance,
        laux_hmatrix_max_rank=args.hmatrix_max_rank,
        laux_hmatrix_heldout_size=args.hmatrix_heldout_size,
    )
    return result, time.perf_counter() - start


def make_mf(args, cache_path: Path):
    molecule = gto.M(
        atom=geometry(args.r, args.n_atom), basis=args.basis, unit="B", verbose=4
    )
    mf = scf.RHF(molecule).density_fit()
    mf.eig = eig_rhf
    mf.max_memory = args.max_memory_mb
    start = time.perf_counter()
    mf = cache_state.prepare_mf(mf, str(cache_path))
    return molecule, mf, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--n-atom", type=int, required=True)
    parser.add_argument("--basis", default="cc-pVTZ")
    parser.add_argument("--jastrow", type=Path, required=True)
    parser.add_argument("--jastrow-md5", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-pytc-commit", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--laux-batch-size", type=int, default=256)
    parser.add_argument("--host-grid-block-size", type=int, default=4096)
    parser.add_argument("--orbital-panel", type=int, default=128)
    parser.add_argument("--vvvv-disk-block-size", type=int, default=16)
    parser.add_argument("--max-memory-mb", type=int, default=120_800)
    parser.add_argument("--gpu-max-memory-mb", type=int, default=70_000)
    parser.add_argument("--hmatrix-leaf-size", type=int, default=128)
    parser.add_argument("--hmatrix-eta", type=float, default=0.05)
    parser.add_argument("--hmatrix-tolerance", type=float, default=1e-2)
    parser.add_argument("--hmatrix-max-rank", type=int, default=16)
    parser.add_argument("--hmatrix-heldout-size", type=int, default=16)
    parser.add_argument("--require-device-substring")
    args = parser.parse_args()

    if any(os.environ.get(name) == "1" for name in X_SWITCHES):
        raise SystemExit("FATAL: all X-drop switches must be unset")
    positive = (
        args.n_atom >= 2,
        args.n_atom % 2 == 0,
        args.rank >= 1,
        args.oversampling >= 0,
        args.laux_batch_size >= 1,
        args.host_grid_block_size >= 1,
        args.orbital_panel >= 1,
        args.vvvv_disk_block_size >= 1,
        args.max_memory_mb >= 1,
        args.gpu_max_memory_mb >= 1,
        args.hmatrix_leaf_size >= 1,
        args.hmatrix_eta >= 0.0,
        args.hmatrix_tolerance >= 0.0,
        args.hmatrix_max_rank >= 1,
        args.hmatrix_heldout_size >= 1,
    )
    if not all(positive):
        raise SystemExit("FATAL: invalid geometry, hierarchy, rank, or memory control")
    if not args.jastrow.is_file() or md5_file(args.jastrow) != args.jastrow_md5:
        raise SystemExit("FATAL: Jastrow history missing or MD5 mismatch")
    if args.output_dir.exists():
        raise SystemExit(f"FATAL: refusing to overwrite output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    revision = clean_pytc_revision(args.expect_pytc_commit)
    xla_cache_dir = required_xla_cache_dir()
    device = device_receipt(args.require_device_substring)
    params, n_saved, n_used = load_jastrow(args.jastrow)

    direct_store = args.output_dir / "direct_full_x.h5"
    hmatrix_store = args.output_dir / "hmatrix_full_x.h5"
    direct_factor = args.output_dir / f"direct_tucker_m{args.rank}.h5"
    hmatrix_factor = args.output_dir / f"hmatrix_tucker_m{args.rank}.h5"

    molecule, direct_mf, direct_scf_s = make_mf(args, direct_store)
    jastrow = CompositeJastrow.create(
        [NuclearCusp.create(molecule), BoysHandy.create(molecule)]
    )
    direct, direct_isdf_s = build_isdf(
        direct_mf, jastrow, params, direct_store, args, hmatrix=False
    )
    direct_source = source_receipt(direct, direct_store)
    if args.rank > direct.n_orb:
        raise SystemExit(f"FATAL: Tucker rank {args.rank} exceeds n_orb={direct.n_orb}")
    direct_full = solve_ccsd(
        direct_mf, direct, params, args.max_memory_mb, args.gpu_max_memory_mb,
        args.vvvv_disk_block_size,
    )
    direct_tucker_view, direct_factor_receipt = tucker_view(
        direct, params, direct_source, direct_factor, revision, args
    )
    direct_tucker = solve_ccsd(
        direct_mf, direct_tucker_view, params, args.max_memory_mb,
        args.gpu_max_memory_mb, args.vvvv_disk_block_size,
    )

    clone_static_isdf_cache(direct.isdf_kernels["X"].file, hmatrix_store)
    hmatrix_molecule, hmatrix_mf, hmatrix_scf_s = make_mf(args, hmatrix_store)
    hmatrix_jastrow = CompositeJastrow.create(
        [NuclearCusp.create(hmatrix_molecule), BoysHandy.create(hmatrix_molecule)]
    )
    hierarchy, hmatrix_isdf_s = build_isdf(
        hmatrix_mf, hmatrix_jastrow, params, hmatrix_store, args, hmatrix=True
    )
    hmatrix_source = source_receipt(hierarchy, hmatrix_store)
    hmatrix_full = solve_ccsd(
        hmatrix_mf, hierarchy, params, args.max_memory_mb, args.gpu_max_memory_mb,
        args.vvvv_disk_block_size,
    )
    hmatrix_tucker_view, hmatrix_factor_receipt = tucker_view(
        hierarchy, params, hmatrix_source, hmatrix_factor, revision, args
    )
    hmatrix_tucker = solve_ccsd(
        hmatrix_mf, hmatrix_tucker_view, params, args.max_memory_mb,
        args.gpu_max_memory_mb, args.vvvv_disk_block_size,
    )

    matrix = {
        "direct_full_x": direct_full,
        "hierarchical_full_x": hmatrix_full,
        "direct_tucker_x": direct_tucker,
        "hierarchical_tucker_x": hmatrix_tucker,
    }
    reference = matrix["direct_full_x"]["e_tot"]
    for result in matrix.values():
        result["delta_e_tot_vs_direct_full_mha"] = (result["e_tot"] - reference) * 1_000.0
    result = {
        "schema": "pytc.hchain.laux-hmatrix-x-matrix.v1",
        "pytc_commit": revision,
        "geometry": {"n_atom": args.n_atom, "r_bohr": args.r},
        "basis": args.basis,
        "jastrow": {
            "path": str(args.jastrow.resolve()),
            "md5": args.jastrow_md5,
            "n_saved": n_saved,
            "n_averaged": n_used,
        },
        "controls": {
            "rank": args.rank,
            "oversampling": args.oversampling,
            "seed": args.seed,
            "laux_batch_size": args.laux_batch_size,
            "host_grid_block_size": args.host_grid_block_size,
            "orbital_panel": args.orbital_panel,
            "vvvv_disk_block_size": args.vvvv_disk_block_size,
            "hmatrix": {
                "leaf_size": args.hmatrix_leaf_size,
                "eta": args.hmatrix_eta,
                "tolerance": args.hmatrix_tolerance,
                "max_rank": args.hmatrix_max_rank,
                "heldout_size": args.hmatrix_heldout_size,
            },
        },
        "runtime": {
            "xla_cache_dir": xla_cache_dir,
            "device": device,
            "max_memory_mb": args.max_memory_mb,
            "gpu_max_memory_mb": args.gpu_max_memory_mb,
        },
        "sources": {"direct": direct_source, "hierarchical": hmatrix_source},
        "factors": {"direct": direct_factor_receipt, "hierarchical": hmatrix_factor_receipt},
        "wall_s": {
            "direct_scf_or_cache_restore": direct_scf_s,
            "direct_full_x_build": direct_isdf_s,
            "hierarchical_scf_or_cache_restore": hmatrix_scf_s,
            "hierarchical_full_x_build": hmatrix_isdf_s,
        },
        "matrix": matrix,
    }
    output = args.output_dir / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("LAUX_HMATRIX_X_MATRIX " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
