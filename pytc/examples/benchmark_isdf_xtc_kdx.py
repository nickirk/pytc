"""Stage-wise benchmark for ISDF-XTC heavy kernels.

This script measures wall-clock times for:
1. ISDF decomposition
2. K-kernel build (K1/K3)
3. L_aux build
4. D-kernel build
5. X-kernel build (single orbital block)

Default settings are intentionally small enough for quick local runs.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.xtc import XTC, ISDFXTC
from pytc.jastrow.rexp import REXP


jax.config.update("jax_enable_x64", True)


def _sync(x: Any) -> None:
    """Force completion of JAX work and host transfer when needed."""
    if isinstance(x, dict):
        for v in x.values():
            _sync(v)
        return
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
        return
    if isinstance(x, (np.ndarray, float, int)):
        return
    try:
        np.asarray(x)
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ISDF-XTC heavy kernels")
    p.add_argument("--atom", type=str, default="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--grid-lvl", type=int, default=1)
    p.add_argument("--n-rank-factor", type=float, default=6.0,
                   help="n_rank = int(n_rank_factor * n_orb)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ls-grid-batch-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--orb-block-size", type=int, default=8)
    p.add_argument("--host-grid-block-size", type=int, default=2000)
    p.add_argument("--x-block", type=int, default=8,
                   help="orbital block size used in single X-kernel block benchmark")
    p.add_argument("--save-path", type=str, default=None,
                   help="Optional HDF5 path. Default: auto temp file name per run.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        print("JAX devices:", jax.devices(), flush=True)
    print(f"local_device_count={jax.local_device_count()}", flush=True)

    mol = gto.M(atom=args.atom, basis=args.basis, verbose=0)
    mf = scf.RHF(mol)
    t0 = time.perf_counter()
    mf.kernel()
    t_scf = time.perf_counter() - t0

    jastrow = REXP()
    jparams = {"alpha": jnp.array([args.alpha])}

    t0 = time.perf_counter()
    xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=args.grid_lvl)
    _sync(xtc.phi)
    t_xtc = time.perf_counter() - t0

    n_orb = xtc.n_orb
    n_rank = max(4, int(args.n_rank_factor * n_orb))
    print(f"n_orb={n_orb}, n_grid={xtc.grid_points.shape[0]}, n_rank={n_rank}", flush=True)

    save_path = args.save_path or f"benchmark_isdf_xtc_kdx_{uuid.uuid4().hex[:8]}.h5"
    if os.path.exists(save_path):
        os.remove(save_path)

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    isdf_xtc = ISDFXTC.from_xtc(
        xtc,
        n_rank=n_rank,
        is_incore=False,
        save_path=save_path,
        ls_grid_batch_size=args.ls_grid_batch_size,
    )
    timings["isdf_decompose"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    kmat = isdf_xtc.compute_kmat_kernels(
        jparams,
        batch_size=args.batch_size,
        host_grid_block_size=args.host_grid_block_size,
    )
    _sync(kmat)
    timings["kmat_kernels"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    l_aux = isdf_xtc._compute_L_aux(
        jparams,
        batch_size=args.batch_size,
        save_path=None,
        host_grid_block_size=args.host_grid_block_size,
    )
    _sync(l_aux)
    timings["l_aux"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = isdf_xtc._compute_D_kernel(
        jparams,
        batch_size=args.batch_size,
        L_aux=l_aux,
        host_grid_block_size=args.host_grid_block_size,
    )
    _sync(d)
    timings["d_kernel"] = time.perf_counter() - t0

    blk = min(args.x_block, n_orb)
    ranges = (slice(None), slice(None), slice(0, blk), slice(0, blk))
    t0 = time.perf_counter()
    x_blk = isdf_xtc._compute_X_kernel(
        jparams,
        ranges=ranges,
        batch_size=args.batch_size,
        L_aux=l_aux,
        host_grid_block_size=args.host_grid_block_size,
    )
    _sync(x_blk)
    timings["x_kernel_block"] = time.perf_counter() - t0

    print("\n=== Benchmark Summary (seconds) ===", flush=True)
    print(f"scf:            {t_scf:10.4f}", flush=True)
    print(f"xtc_init:       {t_xtc:10.4f}", flush=True)
    for k in ["isdf_decompose", "kmat_kernels", "l_aux", "d_kernel", "x_kernel_block"]:
        print(f"{k:15s} {timings[k]:10.4f}", flush=True)

    # Light shape checks to show the benchmark actually exercised outputs.
    print("\n=== Output Shapes ===", flush=True)
    print("K1:", np.asarray(kmat["K1_kernel"]).shape, flush=True)
    print("K3:", np.asarray(kmat["K3_kernel"]).shape, flush=True)
    print("L_aux:", np.asarray(l_aux).shape, flush=True)
    print("D:", np.asarray(d).shape, flush=True)
    print("X_block:", np.asarray(x_blk).shape, flush=True)

    if os.path.exists(save_path):
        os.remove(save_path)


if __name__ == "__main__":
    main()
