"""Performance-baseline harness for ISDF-XTC-CCSD + VMC kernels (task #16, Phase 1a).

Times the guardrail paths Felix designated, on the canonical systems:
    1. ISDF build      - ISDFXTC.from_xtc (decomposition)
    2. K integrals     - compute_kmat_kernels (K1/K3)
    3. ISDF dU kernels - L_aux / D / X (sub-steps of the ISDF delta_U path)
    4. delta_U (exact) - XTC.get_delta_U
    5. xTC-CCSD        - make_eris + cc.rccsd.RCCSD.kernel
    6. VMC step        - vmc.sample per-step (+ optimizer.step secondary)

Systems: H2O/cc-pVDZ, C2H4/cc-pVTZ, benzene/cc-pCVDZ.

Modes:
    record  (default): run all systems, write JSON of timings + sanity scalars + env.
    compare (--compare <baseline.json>): re-run, diff vs baseline, report % regression,
                exit nonzero if any guarded path exceeds --threshold (default 20%).

Methodology: warmup pass (discard JAX compile) then median of --repeats (default 5),
_sync() (block_until_ready) on all JAX outputs, lib.num_threads(1) for reproducibility.
Machine-agnostic: runs on CPU (dev) or GPU (canonical baseline via Grace's cluster).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List

import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf, cc, lib

from pytc.xtc import XTC, ISDFXTC
from pytc.jastrow.rexp import REXP

jax.config.update("jax_enable_x64", True)
# Experimental GPU config: request full-precision matmuls (avoids XLA:GPU's
# TF32-on-f32 default). Whether TF32 was actually the cause of the prior
# isdf_dU_err gap is NOT yet established -- it is characterized by the f32
# probe + per-system dtype audit in precision_probe() (Rick R2.4: characterize,
# don't assert).
jax.config.update("jax_default_matmul_precision", "highest")

# --------------------------------------------------------------------------- #
# Geometries (Angstrom). Standard experimental/optimized coordinates.
# --------------------------------------------------------------------------- #
_C2H4_GEOM = (
    "C 0 0 0.6695; C 0 0 -0.6695; "
    "H 0 0.935 1.239; H 0 -0.935 1.239; "
    "H 0 0.935 -1.239; H 0 -0.935 -1.239"
)

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "H2O_ccpVDZ": {
        "atom": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "basis": "ccpvdz",
    },
    "C2H4_ccpVDZ": {
        # ethylene (D2h): cc-pVDZ = medium-size make_eris/ccsd point (n_orb=48).
        # Added per coordinator decision: cc-pVTZ make_eris is broken (pytc bug),
        # so this system carries the full-path (incl. make_eris/ccsd) coverage.
        "atom": _C2H4_GEOM,
        "basis": "ccpvdz",
    },
    "C2H4_ccpVTZ": {
        # ethylene: D2h, C-C 1.339, C-H 1.086, HCH 117.6 deg
        "atom": _C2H4_GEOM,
        "basis": "ccpvtz",
    },
    "benzene_ccpCVDZ": {
        # benzene: D6h, C-C 1.397, C-H 1.084, centered at origin, planar (z=0)
        "atom": None,  # filled by _benzene_geom() after its definition below
        # cc-pCVDZ = core-valence DZ; exists for C but not H (H has no core-valence).
        # Standard convention: cc-pCVDZ on C, cc-pVDZ on H.
        "basis": {"C": "ccpcvdz", "H": "ccpvdz"},
    },
    # H30/minimal-basis stress vehicle (ke-liao + Woke, kernel-phase guardrail).
    # 30 H atoms in a linear chain: 30 electrons + a long-chain grid stresses
    # the grid-based ISDF/scan/tile paths (exactly the HBM-frugal code), while
    # the STO-3G basis keeps the orbital count modest. ISDF + VMC paths only
    # (make_eris/ccsd at 30 electrons would be enormous and aren't where the
    # HBM-tiling lives). Opt-in: not in the default --systems list, not in
    # REQUIRED_PATHS (would invalidate the existing canonical baseline).
    "H30_minimal": {
        "atom": None,  # filled by _h30_chain_geom() below
        "basis": "sto3g",
        # Skip make_eris/ccsd_kernel: at 30 electrons those are enormous and
        # aren't where the HBM-tiling code lives (Woke). ISDF + VMC only.
        "skip_eris": True,
    },
}


def _benzene_geom(cc: float = 1.397, ch: float = 1.084) -> str:
    """Return a benzene geometry string (D6h, planar in z=0, Angstrom)."""
    atoms: List[str] = []
    rc = cc  # C radius from center
    rh = cc + ch  # H radius from center
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"C {np.cos(theta):.6f} {np.sin(theta):.6f} 0")
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"H {rh*np.cos(theta):.6f} {rh*np.sin(theta):.6f} 0")
    return "; ".join(atoms)


def _h30_chain_geom(spacing_ang: float = 1.0, n_atoms: int = 30) -> str:
    """Linear hydrogen chain H_{n_atoms} equally spaced along x (Angstrom)."""
    return "; ".join(
        f"H {i*spacing_ang:.6f} 0 0" for i in range(n_atoms))


SYSTEMS["benzene_ccpCVDZ"]["atom"] = _benzene_geom()
SYSTEMS["H30_minimal"]["atom"] = _h30_chain_geom()

# Paths whose regression is guarded in --compare (kernel paths; excludes setup).
GUARDED_PATHS = {
    "isdf_decompose", "kmat_kernels", "l_aux", "d_kernel", "x_kernel_block",
    "delta_U_exact", "make_eris", "ccsd_kernel", "vmc_sample_step",
}

# Self-describing comparison policy, stored in every baseline JSON so that
# `--compare` carries its own tolerances (Rick #21 detail). `compare()` reads
# these from the baseline unless a CLI flag overrides.
COMPARE_POLICY: Dict[str, Any] = {
    "timing": {"guarded_threshold_pct": 20.0},
    "sanity": {
        # e_corr: absolute tolerance on |new - base| (Ha). Strict: an unchanged
        # code path is near-bit-identical; a numerics regression exceeds this.
        "e_corr_atol": 1e-8,
        # isdf_dU_err: bounded drift only (no absolute ceiling). The rank-
        # truncation floor is system-dependent (~1e-12 to ~1e-4); an absolute
        # ceiling cannot distinguish a correct rank-limited run from a precision
        # regression. The drift check below catches refactor regressions.
        "isdf_dU_err_drift_atol": 1e-9,
        # native_residual_gap: device-determinism guardrail.
        # |residual_GPU - residual_CPU| for ISDF pivots. < 1e-8 means pivot
        # selection is deterministic across devices. Optional field (present
        # when the pivot-experiment selftest runs alongside the baseline).
        "native_residual_gap_ceiling": 1e-8,
        # delta_U_maxabs: magnitude sanity (O(0.1-1)); absolute drift tolerance.
        "delta_U_maxabs_atol": 1e-8,
        # peak HBM (peak_bytes_in_use) drift tolerance per path. Relative:
        # ke-liao's "less HBM AND/OR faster" rule is enforced as "no path
        # grows peak HBM by more than this fraction" (default 10%). Only
        # enforced when both baseline and new runs reported a peak_hbm_bytes
        # for that path (i.e. a GPU run). The drift check is the load-bearing
        # memory guardrail for the kernel-phase vectorizations (PR12+).
        "peak_hbm_drift_rel_tol": 0.10,
    },
    # Experiment-defining fields: a mismatch here FAILS compare outright (Rick:
    # accelerator model/count, JAX+jaxlib, PySCF, precision/XLA flags, threads,
    # full harness config). hostname/job/partition are provenance only.
    "compat": {
        "metadata": ["jax", "jaxlib", "pyscf", "device_kinds", "n_devices",
                     "jax_enable_x64", "default_matmul_precision",
                     "env_XLA_FLAGS", "env_JAX_XLA_FLAGS",
                     "env_JAX_DEFAULT_MATMUL_PRECISION",
                     "backend_platform_version", "num_threads"],
        "config": ["grid_lvl", "n_rank_factor", "alpha", "batch_size",
                   "host_grid_block", "x_block", "warmup", "repeats",
                   "vmc_walkers", "vmc_steps", "vmc_burnin", "no_vmc",
                   "basis_override"],
    },
}

# Explicit per-system required-path matrix (Rick R2.1): the baseline contract is
# DECLARED, not inferred from whichever paths happened to succeed. A canonical
# baseline is rejected (record --canonical exits nonzero) if a required path
# errored; compare() fails if a required path is missing/errored in the new run.
# C2H4-TZ intentionally excludes make_eris/ccsd_kernel (the cc-pVTZ pytc bug is
# a known kernel-audit item; that system guards the ISDF/VMC paths only).
_FULL = set(GUARDED_PATHS)
_ISDF_ONLY = _FULL - {"make_eris", "ccsd_kernel"}
# TEMPORARY (gate G2): C2H4_ccpVDZ downgraded _FULL→_ISDF_ONLY + benzene removed
# due to make_eris GPU IndexError. RESTORE both when the make_eris GPU bug is
# fixed, before any solver/CCSD-phase refactor.
REQUIRED_PATHS: Dict[str, set] = {
    "H2O_ccpVDZ": _FULL,
    "C2H4_ccpVDZ": _ISDF_ONLY,
    "C2H4_ccpVTZ": _ISDF_ONLY,
}
# Sanity required per system, derived from the path matrix:
#   e_corr        where make_eris+ccsd_kernel are required
#   isdf_dU_err   where isdf_decompose is required
#   delta_U_maxabs where delta_U_exact is required
REQUIRED_SANITY: Dict[str, set] = {}
for _s, _paths in REQUIRED_PATHS.items():
    _san: set = set()
    if {"make_eris", "ccsd_kernel"} <= _paths:
        _san.add("e_corr")
    if "isdf_decompose" in _paths:
        _san.add("isdf_dU_err")
    if "delta_U_exact" in _paths:
        _san.add("delta_U_maxabs")
    REQUIRED_SANITY[_s] = _san


def required_paths_for(sname: str) -> set:
    """Required guarded paths for a system (defaults to all guarded if unknown)."""
    return set(REQUIRED_PATHS.get(sname, GUARDED_PATHS))


def required_sanity_for(sname: str) -> set:
    """Required sanity keys for a system (defaults to the full set if unknown)."""
    return set(REQUIRED_SANITY.get(sname, {"e_corr", "isdf_dU_err", "delta_U_maxabs"}))


def _is_finite_scalar(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _validate_self_policy(record: Dict[str, Any]) -> List[str]:
    """Validate that a record satisfies its own embedded quality policy.

    A canonical baseline must not only pass the required-path/sanity matrix;
    it must also satisfy the quality gates declared in its compare_policy
    (e.g. native_residual_gap not exceeding its determinism ceiling).  Without
    this, a baseline can advertise a strict ceiling while recording values that
    violate it — a latent false-green in every compare().

    Returns a list of violation strings (empty = valid).
    """
    violations: List[str] = []
    policy = record.get("compare_policy", COMPARE_POLICY)
    san = policy.get("sanity", {})
    nrg_ceiling = float(san.get("native_residual_gap_ceiling",
                                COMPARE_POLICY["sanity"]["native_residual_gap_ceiling"]))
    systems = record.get("systems", {})
    for sname, sysrec in systems.items():
        sanity_vals = sysrec.get("sanity", {})
        nrg = sanity_vals.get("native_residual_gap")
        if nrg is None or not _is_finite_scalar(nrg):
            continue  # optional field; absent when pivot selftest wasn't run
        nrg = float(nrg)
        if nrg > nrg_ceiling:
            violations.append(
                f"{sname}: self-policy: native_residual_gap={nrg:.2e} > "
                f"ceiling={nrg_ceiling:.0e}")
    return violations


def validate_against_matrix(record: Dict[str, Any]) -> List[str]:
    """Validate a record against the required-path/sanity matrix.

    Returns a list of violation strings (empty = valid). Used in canonical
    record mode (reject weak baselines) and conceptually mirrors what compare()
    enforces against the baseline's declared matrix. Rules (Rick R3.1/R3.2):
      * the system set must EXACTLY match REQUIRED_PATHS (no partial baselines);
      * each required path's ``med`` must be a finite scalar > 0;
      * each required sanity value must be present and finite.
    """
    violations: List[str] = []
    systems = record.get("systems", {})
    # R3.1: a canonical baseline must carry the FULL declared system set.
    if set(systems) != set(REQUIRED_PATHS):
        violations.append(
            f"system set {sorted(systems)} != required {sorted(REQUIRED_PATHS)}; "
            f"canonical mode requires all systems (no partial baselines)")
    for sname, sysrec in systems.items():
        rp = required_paths_for(sname)
        rs = required_sanity_for(sname)
        if "timings_s" not in sysrec:
            violations.append(f"{sname}: system errored ({sysrec.get('error')})")
            continue
        timings = sysrec["timings_s"]
        for path in sorted(rp):
            p = timings.get(path)
            if not (isinstance(p, dict) and "med" in p):
                violations.append(
                    f"{sname}/{path}: required path missing or errored: {p}")
                continue
            med = p["med"]
            # R3.2: a NaN/inf/<=0 med would yield pct=NaN in compare -> false-green.
            if not _is_finite_scalar(med) or float(med) <= 0:
                violations.append(
                    f"{sname}/{path}: required timing med not finite/>0: {med!r}")
        sanity = sysrec.get("sanity", {})
        for key in sorted(rs):
            v = sanity.get(key)
            if v is None:
                violations.append(f"{sname}/sanity.{key}: required sanity missing")
            elif not _is_finite_scalar(v):
                violations.append(
                    f"{sname}/sanity.{key}: required sanity non-finite: {v!r}")
    return violations

pytc_logger = logging.getLogger("pytc")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sync(x: Any) -> None:
    """Force completion of JAX work + host transfer."""
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


def _peak_hbm_now() -> int | None:
    """Best-effort peak HBM bytes for the first local device.

    Returns ``jax.devices()[0].memory_stats()['peak_bytes_in_use']`` when the
    device exposes memory stats (GPU/TPU). On CPU (no memory_stats) returns
    None and callers should treat the metric as unmeasured, not zero. The
    value is a cumulative high-water mark for the process; for relative
    comparison across versions of the same codepath the absolute value is
    directly comparable (setup is identical between baseline and new run).
    """
    try:
        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        v = stats.get("peak_bytes_in_use")
        return int(v) if v else None
    except Exception:
        return None


def _time(fn: Callable[[], Any], warmup: int, repeats: int) -> Dict[str, Any]:
    """Run fn() warmup times (discarded) then repeats times; report med/min/max.

    Also captures the peak HBM high-water mark (``peak_bytes_in_use``) before
    the warmup pass and after the final timed repeat. The post-minus-pre delta
    isolates allocations introduced *by this codepath* that exceed the
    pre-existing process high-water mark (a non-zero delta signals a path that
    pushes HBM higher than the setup state; the absolute ``peak_hbm_bytes``
    field is the canonical comparison value).
    """
    pre_peak = _peak_hbm_now()
    for _ in range(max(0, warmup)):
        out = fn()
        _sync(out)
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        _sync(out)
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    post_peak = _peak_hbm_now()
    result: Dict[str, Any] = {
        "med": float(np.median(arr)), "min": float(np.min(arr)),
        "max": float(np.max(arr)), "n": len(samples),
    }
    if post_peak is not None:
        result["peak_hbm_bytes"] = post_peak
        if pre_peak is not None:
            result["peak_hbm_delta_from_pre"] = post_peak - pre_peak
    return result


def _err(exc: BaseException) -> Dict[str, str]:
    return {"error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=2)}


def _dtype(x: Any) -> str:
    """Readable dtype for jax/numpy arrays/scalars, or the set of leaf dtypes for
    a dict/tuple/list container (e.g. kmat). Best-effort ('?' on miss)."""
    d = getattr(x, "dtype", None)
    if d is not None:
        return str(d)
    if isinstance(x, (dict, list, tuple)):
        vals = x.values() if isinstance(x, dict) else x
        leaves = sorted({_dtype(v) for v in vals} - {"?"})
        return str(leaves) if leaves else "?"
    try:
        return str(np.asarray(x).dtype)
    except Exception:
        return "?"



def collect_env() -> Dict[str, Any]:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()[:12]
    except Exception:
        pass
    import socket
    env: Dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": commit,
        "jax": jax.__version__,
        "jaxlib": _safe(lambda: __import__("jaxlib").__version__),
        "pyscf": __import__("pyscf").__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "num_threads": lib.num_threads(),
        # accelerator identity (defines the experiment per Rick's #21 detail)
        "devices": [str(d) for d in jax.devices()],
        "device_kinds": sorted({getattr(d, "device_kind", d.platform)
                                for d in jax.devices()}),
        "n_devices": jax.local_device_count(),
        # precision / x64 flags (experiment-defining)
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "env_JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "env_XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "env_JAX_XLA_FLAGS": os.environ.get("JAX_XLA_FLAGS"),
        "env_JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "env_JAX_DEFAULT_MATMUL_PRECISION": os.environ.get(
            "JAX_DEFAULT_MATMUL_PRECISION"),
        "default_matmul_precision": _safe(
            lambda: str(jax.config.read("jax_default_matmul_precision"))),
        # CUDA / backend version (GPU provenance)
        "backend_platform_version": _safe(
            lambda: getattr(jax.lib.xla_bridge.get_backend(),
                            "platform_version", None)),
        # scheduler / cluster provenance (best-effort)
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }
    return env


def _safe(fn: Callable[[], Any]) -> Any:
    """Call fn() returning None on any error (for best-effort env probes)."""
    try:
        return fn()
    except Exception:
        return None



def precision_probe() -> Dict[str, Any]:
    """Characterize matmul/dot precision on the active device. Reports findings;
    does NOT assert attribution of ISDF numerical errors (see review #21).

    Two probes, kept strictly separate:
      * f64-operand probe: float64 in -> vs NumPy float64 truth. Documents whether
        a float64 matmul runs accurately on this device. A near-zero error only
        proves *this isolated f64 op* is accurate; it CANNOT detect or rule out
        TF32, which only applies to f32 inputs (Rick finding #2).
      * f32-operand probe: float32 in -> vs NumPy float64 truth. This is what
        characterizes TF32: rel_err ~1e-3 => TF32 (10-bit mantissa); ~1e-6 =>
        true f32 accumulate; ~1e-7 => f32 with good conditioning.

    To attribute the ISDF ``isdf_dU_err`` gap, correlate these with the per-system
    dtype audit (``dtypes``): if ISDF intermediates are float64 and the gap
    persists, it is NOT TF32 (f64 matmuls don't use TF32) -> genuine GPU ISDF
    numerics / x64-not-propagating. If any intermediate is float32, the f32 probe
    quantifies the TF32 contribution.
    """
    out: Dict[str, Any] = {}
    out["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    out["default_matmul_precision"] = _safe(
        lambda: str(jax.config.read("jax_default_matmul_precision")))
    n = 512
    rng = np.random.default_rng(0)
    a64 = rng.standard_normal((n, n))
    b64 = rng.standard_normal((n, n))
    ref64 = a64 @ b64                       # NumPy float64 ground truth
    mref = float(np.max(np.abs(ref64)))
    # -- f64-operand probe (documents f64 matmul accuracy; not a TF32 test) --
    got_f64 = np.asarray(jnp.asarray(a64) @ jnp.asarray(b64))
    out["f64_matmul_n"] = n
    out["f64_matmul_rel_err"] = float(
        np.max(np.abs(got_f64 - ref64)) / max(mref, 1e-30))
    # -- f32-operand probe (isolates GPU *compute* precision from input ------
    #    quantization, per Rick R2.4): reference is the quantized operands
    #    multiplied in exact float64, so the error is the device's matmul
    #    reduction precision only (TF32 ~1e-3; true f32 accumulate ~1e-7).
    a32, b32 = a64.astype(np.float32), b64.astype(np.float32)
    ref_qf64 = a32.astype(np.float64) @ b32.astype(np.float64)
    mqref = float(np.max(np.abs(ref_qf64)))
    got_f32 = np.asarray(jnp.asarray(a32) @ jnp.asarray(b32))
    out["f32_matmul_n"] = n
    out["f32_matmul_rel_err"] = float(
        np.max(np.abs(got_f32 - ref_qf64)) / max(mqref, 1e-30))
    # -- dot accumulation probes (long sum; isolate compute precision) -------
    v = np.arange(1, 1_000_001, dtype=np.float64)
    v32 = v.astype(np.float32)
    ref_f64_dot = float(np.dot(v, v))              # exact f64 dot of v with itself
    ref_qf64_dot = float(np.dot(v32.astype(np.float64), v32.astype(np.float64)))
    out["f64_dot_rel_err"] = abs(
        float(jnp.dot(jnp.asarray(v), jnp.asarray(v))) - ref_f64_dot) / max(
        abs(ref_f64_dot), 1e-30)
    out["f32_dot_rel_err"] = abs(
        float(jnp.dot(jnp.asarray(v32), jnp.asarray(v32))) - ref_qf64_dot) / max(
        abs(ref_qf64_dot), 1e-30)
    return out



# --------------------------------------------------------------------------- #
# VMC setup (best-effort; ansatz API may vary)
# --------------------------------------------------------------------------- #
def _build_vmc_ansatz(mol, mf):
    """Build a minimal Slater-Jastrow ansatz for VMC timing. Returns (ansatz, params, key) or None."""
    try:
        from pytc.ansatz.sj import SlaterJastrow
        from pytc.ansatz.det import SlaterDet
        from pytc.jastrow import CompositeJastrow, NuclearCusp
        from jax import random
        det = SlaterDet.create(mol, mf.mo_coeff)
        jncusp = NuclearCusp.create(mol, name="ncusp")
        jastrow = CompositeJastrow.create([jncusp])
        jp = jastrow.init_params()
        lin = jnp.ones(1)
        ansatz = SlaterJastrow.create(mol, jastrow, [det])
        params = [jp, lin]
        key = random.PRNGKey(43)
        return ansatz, params, key
    except Exception as exc:
        pytc_logger.warning("VMC ansatz setup failed: %s", exc)
        return None


def _bench_vmc(mol, mf, walkers: int, steps: int, burnin: int,
               warmup: int, repeats: int) -> Dict[str, Any]:
    """Time one steady-state VMC sampling step, excluding compile + burn-in.

    ``pytc.vmc.sample`` runs burn-in and sampling in one call. We isolate the
    sampling-step cost with PAIRED samples (Rick R2.5): each repeat times a full
    call (burnin+steps) and a burn-only call (burnin) back-to-back and subtracts,
    so the per-step distribution reflects genuine per-repeat variance (not two
    independent medians). No clamping; min/max are the real extrema (a negative
    value is surfaced honestly as a too-noisy-to-isolate signal, not hidden).
    Warmup compiles both jit traces first. Falls back to diagnostic-only if
    ``sample(n_steps=0)`` is unsupported.
    """
    out: Dict[str, Any] = {}
    built = _build_vmc_ansatz(mol, mf)
    if built is None:
        return {"error": "ansatz setup unavailable"}
    ansatz, params, key = built
    try:
        from pytc.vmc import sample

        def _call(n_steps: int, burn_in_steps: int):
            return sample(ansatz, n_walkers=walkers, n_steps=n_steps,
                          step_size=0.02, burn_in_steps=burn_in_steps,
                          params=params, key=key,
                          use_importance_sampling=False)

        # warmup (compile) both traces, then paired repeats
        pre_peak = _peak_hbm_now()
        for _ in range(max(0, warmup)):
            _sync(_call(steps, burnin))
            _sync(_call(0, burnin))
        per_step: List[float] = []
        full_times: List[float] = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter(); _sync(_call(steps, burnin))
            t_full = time.perf_counter() - t0
            t0 = time.perf_counter(); _sync(_call(0, burnin))
            t_burn = time.perf_counter() - t0
            full_times.append(t_full)
            per_step.append((t_full - t_burn) / max(1, steps))
        arr = np.asarray(per_step)
        vmc_step: Dict[str, Any] = {"med": float(np.median(arr)),
                                    "min": float(np.min(arr)),
                                    "max": float(np.max(arr)),
                                    "n": int(arr.size)}
        post_peak = _peak_hbm_now()
        if post_peak is not None:
            vmc_step["peak_hbm_bytes"] = post_peak
            if pre_peak is not None:
                vmc_step["peak_hbm_delta_from_pre"] = post_peak - pre_peak
        out["vmc_sample_step"] = vmc_step
        out["vmc_sample_total_med"] = float(np.median(np.asarray(full_times)))
        out["vmc_burnin_only_med"] = None  # paired: no separate burn median
    except Exception as exc:
        out["vmc_sample_step"] = _err(exc)
    return out




# --------------------------------------------------------------------------- #
# Per-system benchmark
# --------------------------------------------------------------------------- #
def bench_system(name: str, cfg: Dict[str, Any], args) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name}
    timings: Dict[str, Any] = {}
    sanity: Dict[str, Any] = {}
    dtypes: Dict[str, str] = {}

    basis = args.basis_override or cfg["basis"]
    print(f"\n=== {name}  basis={basis} ===", flush=True)
    mol = gto.M(atom=cfg["atom"], basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    jas = REXP()
    jp = {"alpha": jnp.array([args.alpha])}
    dtypes["mo_coeff"] = _dtype(mf.mo_coeff)
    dtypes["jp_alpha"] = _dtype(jp["alpha"])

    # ---- XTC init (setup; not guarded) ----
    t = _time(lambda: XTC.from_pyscf(mf, jas, grid_lvl=args.grid_lvl), 0, 1)
    xtc = XTC.from_pyscf(mf, jas, grid_lvl=args.grid_lvl)
    _sync(xtc.phi)
    dtypes["xtc_phi"] = _dtype(xtc.phi)
    timings["xtc_init"] = t["med"]

    n_orb = xtc.n_orb
    n_grid = int(xtc.grid_points.shape[0])
    n_rank = max(4, int(args.n_rank_factor * n_orb))
    sanity["n_orb"] = int(n_orb)
    sanity["n_grid"] = n_grid
    sanity["n_rank"] = int(n_rank)
    result["shape"] = dict(sanity)

    # ---- delta_U exact ----
    def _du_exact():
        return xtc.get_delta_U(jp)
    try:
        du = np.asarray(_sync_block(_du_exact))
        timings["delta_U_exact"] = _time(_du_exact, args.warmup, args.repeats)
        sanity["delta_U_maxabs"] = float(np.max(np.abs(du)))
        dtypes["delta_U_exact"] = _dtype(du)
    except Exception as exc:
        timings["delta_U_exact"] = _err(exc)

    # ---- ISDF decompose ----
    def _new_isdf():
        return ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True)
    try:
        timings["isdf_decompose"] = _time(_new_isdf, args.warmup, args.repeats)
        ixtc = ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True)
    except Exception as exc:
        timings["isdf_decompose"] = _err(exc)
        ixtc = None

    # ---- K integrals + ISDF delta_U sub-kernels (mirror existing benchmark) ----
    # --vmc-focus skips these (l_aux dominates H30 wall at ~40 min/repeat and
    # isn't touched by jastrow/ansatz PRs; Woke + Rick agreed on VMC-focused
    # H30 profiling for the jastrow phase, reserving full sweep for kernel PRs).
    if ixtc is not None and not getattr(args, "vmc_focus", False):
        try:
            def _kmat():
                return ixtc.compute_kmat_kernels(jp, batch_size=args.batch_size,
                                                 host_grid_block_size=args.host_grid_block)
            timings["kmat_kernels"] = _time(_kmat, args.warmup, args.repeats)
            kmat = ixtc.compute_kmat_kernels(jp, batch_size=args.batch_size,
                                             host_grid_block_size=args.host_grid_block)
            _sync(kmat)
            dtypes["kmat"] = _dtype(kmat)

            def _laux():
                return ixtc._compute_L_aux(jp, batch_size=args.batch_size,
                                           save_path=None,
                                           host_grid_block_size=args.host_grid_block)
            timings["l_aux"] = _time(_laux, args.warmup, args.repeats)
            l_aux = ixtc._compute_L_aux(jp, batch_size=args.batch_size,
                                        save_path=None,
                                        host_grid_block_size=args.host_grid_block)
            _sync(l_aux)
            dtypes["l_aux"] = _dtype(l_aux)

            def _dkern():
                return ixtc._compute_D_kernel(jp, batch_size=args.batch_size,
                                              L_aux=l_aux,
                                              host_grid_block_size=args.host_grid_block)
            timings["d_kernel"] = _time(_dkern, args.warmup, args.repeats)
            d = ixtc._compute_D_kernel(jp, batch_size=args.batch_size,
                                       L_aux=l_aux,
                                       host_grid_block_size=args.host_grid_block)
            _sync(d)
            dtypes["D_kernel"] = _dtype(d)

            blk = min(args.x_block, n_orb)
            ranges = (slice(None), slice(None), slice(0, blk), slice(0, blk))

            def _xkern():
                return ixtc._compute_X_kernel(jp, ranges=ranges,
                                              batch_size=args.batch_size, L_aux=l_aux,
                                              host_grid_block_size=args.host_grid_block)
            timings["x_kernel_block"] = _time(_xkern, args.warmup, args.repeats)

            # ISDF delta_U sanity vs exact
            ixtc2 = ixtc.isdf(jp)
            du_i = np.asarray(_sync_block(lambda: ixtc2.get_delta_U(jp)))
            sanity["isdf_dU_err"] = float(np.max(np.abs(du_i - du)))
            dtypes["delta_U_isdf"] = _dtype(du_i)
        except Exception as exc:
            for k in ("kmat_kernels", "l_aux", "d_kernel", "x_kernel_block"):
                timings.setdefault(k, _err(exc))

    # ---- make_eris + CCSD kernel ----
    # Opt-out per system (H30_minimal stress vehicle skips this: at 30 electrons
    # make_eris/ccsd is enormous and isn't where the HBM-tiling lives anyway).
    if cfg.get("skip_eris", False):
        print(f"[{name}] skipping make_eris/ccsd_kernel (skip_eris=True)", flush=True)
    else:
        try:
            def _eris():
                return xtc.make_eris(mf, jp)
            timings["make_eris"] = _time(_eris, 0, min(args.repeats, 3))
            eris = xtc.make_eris(mf, jp)

            def _ccsd():
                mycc = cc.rccsd.RCCSD(mf)
                return mycc.kernel(eris=eris)
            timings["ccsd_kernel"] = _time(_ccsd, 0, min(args.repeats, 3))
            mycc = cc.rccsd.RCCSD(mf)
            ec, _, _ = mycc.kernel(eris=eris)
            sanity["e_corr"] = float(ec)
        except Exception as exc:
            timings.setdefault("make_eris", _err(exc))
            timings.setdefault("ccsd_kernel", _err(exc))

    # ---- VMC step ----
    if not args.no_vmc:
        try:
            vmc = _bench_vmc(mol, mf, args.vmc_walkers, args.vmc_steps,
                             args.vmc_burnin, warmup=args.warmup,
                             repeats=min(args.repeats, 3))
            timings.update(vmc)
        except Exception as exc:
            timings["vmc_sample_step"] = _err(exc)

    result["timings_s"] = timings
    result["sanity"] = {k: sanity.get(k) for k in sorted(sanity)}
    result["dtypes"] = {k: dtypes.get(k, "?") for k in sorted(dtypes)}
    print(f"[{name}] n_orb={n_orb} n_grid={n_grid} e_corr={sanity.get('e_corr')}", flush=True)
    return result


def _sync_block(fn: Callable):
    out = fn()
    _sync(out)
    return out


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
def compare(new: Dict[str, Any], baseline_path: str, threshold: float) -> int:
    """Fail-closed comparison vs a baseline JSON.

    Gates (any failure -> exit 1), in order:
      1. COMPAT  — experiment-defining metadata/config fields must match
                   (accelerator model/count, JAX+jaxlib, PySCF, precision/XLA
                   flags, thread count, full harness config). Per Rick #21:
                   hostname/job/partition are provenance only, not gated.
      2. COVERAGE — every baseline system and its guarded paths that were valid
                    in the baseline must be present and non-error in the new run
                    (Rick finding #1: no silent skip of missing/errored paths).
      3. SANITY   — e_corr within absolute tol; isdf_dU_err bounded drift
                    vs baseline; native_residual_gap below determinism ceiling
      4. TIMING   — guarded-path %Δ within threshold (Rick: relative is fine for
                    timings which are O(1), unlike near-zero sanity scalars).
    Policy is read from the baseline's ``compare_policy`` (self-describing);
    ``--threshold`` overrides the timing gate only.
    """
    with open(baseline_path) as f:
        base = json.load(f)

    # ---- 0. BASELINE SCHEMA VALIDATION (Rick R4: don't trust the baseline) --
    # The baseline itself must pass the same canonical schema/self-policy gates
    # that a new canonical record requires.  This closes the class where a
    # weakened/tampered baseline silently erodes the guard.
    base_violations = validate_against_matrix(base)
    base_violations.extend(_validate_self_policy(base))
    if base_violations:
        print("\n=== BASELINE REJECTED (schema/self-policy violation) ===")
        for v in base_violations:
            print(f"  - {v}")
        return 1
    # Baseline system set must match REQUIRED_PATHS exactly (prevent deleted
    # base.systems w/ intact matrices from silently dropping out of the guard).
    if set(base.get("systems", {})) != set(REQUIRED_PATHS):
        print("\n=== BASELINE REJECTED (system set mismatch) ===")
        print(f"  base.systems {sorted(base.get('systems', {}))} != required {sorted(REQUIRED_PATHS)}")
        return 1
    # Baseline's embedded compare_policy is checked additively: for every key
    # the baseline DECLARES, its value must equal the harness value (catches a
    # tampered ceiling/threshold that would weaken the guard). Keys present in
    # the harness but absent from the baseline (e.g. a newly added tolerance)
    # are permitted — the harness default applies, which is never weaker than
    # the baseline's declared policy. This keeps existing baselines valid when
    # new additive policy fields are introduced (avoids forcing a baseline
    # re-record on every additive policy change).
    base_policy = base.get("compare_policy", {})
    policy_diffs: List[str] = []

    def _policy_diff(bv: Any, hv: Any, prefix: str) -> None:
        if isinstance(bv, dict) and isinstance(hv, dict):
            for k in sorted(set(bv) | set(hv)):
                if k in bv:
                    _policy_diff(bv[k], hv.get(k), f"{prefix}.{k}")
        elif bv != hv:
            policy_diffs.append(f"  compare_policy.{prefix}: baseline={bv!r} != harness={hv!r}")

    for k in sorted(set(base_policy)):
        _policy_diff(base_policy[k], COMPARE_POLICY.get(k), k)
    if policy_diffs:
        print("\n=== BASELINE REJECTED (compare_policy tampering detected) ===")
        for d in policy_diffs:
            print(d)
        return 1
    # Surface additive policy additions (informational, never a rejection).
    added = []
    for k in sorted(set(COMPARE_POLICY) - set(base_policy)):
        added.append(k)
    if added:
        print(f"\n[info] baseline predates policy additions: {added} "
              f"(harness defaults apply; not weaker than baseline)")

    policy = base_policy
    thr = threshold
    san = policy.get("sanity", {})
    e_corr_atol = float(san.get("e_corr_atol", COMPARE_POLICY["sanity"]["e_corr_atol"]))
    nrg_ceiling = float(san.get("native_residual_gap_ceiling",
                                COMPARE_POLICY["sanity"]["native_residual_gap_ceiling"]))
    du_drift = float(san.get("isdf_dU_err_drift_atol",
                             COMPARE_POLICY["sanity"]["isdf_dU_err_drift_atol"]))
    du_maxabs_atol = float(san.get("delta_U_maxabs_atol",
                                   COMPARE_POLICY["sanity"]["delta_U_maxabs_atol"]))
    hbm_tol = float(san.get("peak_hbm_drift_rel_tol",
                            COMPARE_POLICY["sanity"]["peak_hbm_drift_rel_tol"]))
    compat_meta = policy.get("compat", {}).get("metadata",
                                               COMPARE_POLICY["compat"]["metadata"])
    compat_cfg = policy.get("compat", {}).get("config",
                                              COMPARE_POLICY["compat"]["config"])
    # Timing threshold is self-describing (Rick R2.5): if the caller did not
    # override (--threshold), read it from the baseline's own policy.
    if thr is None:
        thr = float(policy.get("timing", {}).get(
            "guarded_threshold_pct",
            COMPARE_POLICY["timing"]["guarded_threshold_pct"]))
    # Declared required-path / required-sanity matrix (Rick R2.1): coverage is
    # checked against this contract, NOT inferred from baseline's successes.
    base_req_paths = {s: set(p) for s, p in base.get(
        "required_paths", {s: sorted(required_paths_for(s))
                           for s in base.get("systems", {})}).items()}
    base_req_san = {s: set(k) for s, k in base.get(
        "required_sanity", {s: sorted(required_sanity_for(s))
                            for s in base.get("systems", {})}).items()}
    failures: List[str] = []
    bmeta, nmeta = base.get("metadata", {}), new.get("metadata", {})
    bcg, ncg = base.get("config", {}), new.get("config", {})
    base_sys, new_sys = base.get("systems", {}), new.get("systems", {})

    # R3.1: reject a baseline whose DECLARED contract was weakened vs the harness
    # constants (a tampered/deleted system+matrix entry must not erode the guard).
    for sname in REQUIRED_PATHS:
        if set(base_req_paths.get(sname, set())) != required_paths_for(sname):
            failures.append(
                f"baseline-contract: required_paths['{sname}'] "
                f"{sorted(base_req_paths.get(sname, set()))} != harness "
                f"{sorted(required_paths_for(sname))}")
        if set(base_req_san.get(sname, set())) != required_sanity_for(sname):
            failures.append(
                f"baseline-contract: required_sanity['{sname}'] "
                f"{sorted(base_req_san.get(sname, set()))} != harness "
                f"{sorted(required_sanity_for(sname))}")
    for sname in set(base_req_paths) - set(REQUIRED_PATHS):
        failures.append(
            f"baseline-contract: undeclared system '{sname}' in baseline matrix")

    print(f"\n=== PERF COMPARE vs {baseline_path} (guarded threshold +{thr:.0f}%) ===")

    # ---- 1. COMPAT ----------------------------------------------------------
    for field in compat_meta:
        if bmeta.get(field) != nmeta.get(field):
            failures.append(
                f"compat/metadata.{field}: base={bmeta.get(field)!r} new={nmeta.get(field)!r}")
    for field in compat_cfg:
        if bcg.get(field) != ncg.get(field):
            failures.append(
                f"compat/config.{field}: base={bcg.get(field)!r} new={ncg.get(field)!r}")

    # ---- 2. COVERAGE (against declared required-path matrix) ---------------
    for sname in base_sys:
        rp = base_req_paths.get(sname, required_paths_for(sname))
        if sname not in new_sys:
            failures.append(f"coverage: system '{sname}' missing in new run")
            continue
        snew = new_sys[sname]
        if "timings_s" not in snew:
            failures.append(
                f"coverage: system '{sname}' errored in new run: {snew.get('error')}")
            continue
        snt = snew.get("timings_s", {})
        for path in sorted(rp):
            pn = snt.get(path)
            if not (isinstance(pn, dict) and "med" in pn):
                failures.append(
                    f"coverage: required '{sname}/{path}' missing/errored in new "
                    f"run: {pn}")
                continue
            # R3.2: a non-finite/<=0 med would yield pct=NaN -> false-green.
            med = pn["med"]
            if not _is_finite_scalar(med) or float(med) <= 0:
                failures.append(
                    f"coverage: required '{sname}/{path}' timing med not finite/>0: "
                    f"{med!r}")

    # ---- 3. TIMING (guarded %Δ) --------------------------------------------
    print("--- timing (guarded paths) ---")
    worst = 0.0
    peak_hbm_regressions: List[str] = []
    for sname, sbase in base_sys.items():
        snew = new_sys.get(sname)
        if not snew or "timings_s" not in snew:
            continue
        sbt, snt = sbase.get("timings_s", {}), snew["timings_s"]
        for path in GUARDED_PATHS:
            pb, pn = sbt.get(path), snt.get(path)
            if not (isinstance(pb, dict) and "med" in pb
                    and isinstance(pn, dict) and "med" in pn):
                continue
            b, nn = pb["med"], pn["med"]
            if b <= 0:
                continue
            pct = (nn - b) / b * 100.0
            flag = "REGRESS" if pct > thr else ""
            worst = max(worst, pct)
            print(f"  {sname:<20} {path:<18} {b:>10.4f} {nn:>10.4f} "
                  f"{pct:>+7.1f}% {flag}")
            if pct > thr:
                failures.append(
                    f"timing: '{sname}/{path}' regressed {pct:+.1f}% "
                    f"(threshold +{thr:.0f}%)")
            # Peak-HBM drift check (only when both runs reported it; ke-liao
            # memory guardrail). Relative drift vs peak_hbm_drift_rel_tol.
            phb_b = pb.get("peak_hbm_bytes")
            phb_n = pn.get("peak_hbm_bytes")
            if (phb_b is not None and phb_n is not None
                    and phb_b > 0):
                rel = (phb_n - phb_b) / phb_b
                hbm_pct = rel * 100.0
                hbm_flag = "HBM-REGRESS" if rel > hbm_tol else ""
                print(f"  {sname:<20} {path:<18} peak_hbm "
                      f"{phb_b/1e6:>10.2f}MB {phb_n/1e6:>10.2f}MB "
                      f"{hbm_pct:>+7.1f}% {hbm_flag}")
                if rel > hbm_tol:
                    msg = (f"peak_hbm: '{sname}/{path}' grew peak HBM "
                           f"{hbm_pct:+.1f}% (rel_tol +{hbm_tol*100:.0f}%); "
                           f"baseline={phb_b/1e6:.2f}MB "
                           f"new={phb_n/1e6:.2f}MB")
                    failures.append(msg)
                    peak_hbm_regressions.append(msg)
    print(f"  worst guarded regression: {worst:+.1f}% (threshold +{thr:.0f}%)")
    if peak_hbm_regressions:
        print(f"  peak-HBM regressions: {len(peak_hbm_regressions)} path(s) "
              f"grew HBM > +{hbm_tol*100:.0f}% (ke-liao memory guardrail)")

    # ---- 4. SANITY (required keys must be present+finite; drift vs policy) --
    print("--- sanity ---")
    for sname in base_sys:
        rs = base_req_san.get(sname, required_sanity_for(sname))
        snew = new_sys.get(sname)
        if not snew:
            continue  # already failed in coverage
        bsan, nsan = base_sys[sname].get("sanity", {}), snew.get("sanity", {})
        for key in sorted(rs):
            nv = nsan.get(key)
            if nv is None or not _is_finite_scalar(nv):
                failures.append(
                    f"sanity: required '{sname}/{key}' missing/non-finite in new run")
                print(f"  {sname:<20} {key:<14} MISSING/non-finite in new [FAIL]")
                continue
            nv = float(nv)
            bv = bsan.get(key)
            # drift checks (only if baseline value is finite)
            if bv is not None and _is_finite_scalar(bv):
                bv = float(bv)
                drift = abs(nv - bv)
                if key == "e_corr":
                    flag = "FAIL" if drift > e_corr_atol else "ok"
                    print(f"  {sname:<20} e_corr        base={bv:+.12g} "
                          f"new={nv:+.12g} drift={drift:.2e} [{flag}]")
                    if drift > e_corr_atol:
                        failures.append(
                            f"sanity: '{sname}' e_corr drift {drift:.2e} > atol {e_corr_atol:.0e}")
                elif key == "isdf_dU_err":
                    flag = "FAIL" if drift > du_drift else "ok"
                    print(f"  {sname:<20} isdf_dU_err   base={bv:.2e} new={nv:.2e} "
                          f"drift={drift:.2e} [{flag}]")
                    if drift > du_drift:
                        failures.append(
                            f"sanity: '{sname}' isdf_dU_err drift {drift:.2e} "
                            f"> drift_tol {du_drift:.0e}")
                elif key == "delta_U_maxabs":
                    flag = "FAIL" if drift > du_maxabs_atol else "ok"
                    print(f"  {sname:<20} delta_U_maxabs base={bv:.6g} "
                          f"new={nv:.6g} drift={drift:.2e} [{flag}]")
                    if drift > du_maxabs_atol:
                        failures.append(
                            f"sanity: '{sname}' delta_U_maxabs drift {drift:.2e} "
                            f"> atol {du_maxabs_atol:.0e}")

    # ---- 4b. DETERMINISM (native_residual_gap, optional) -----------------
    # Device-determinism guardrail: |residual_GPU - residual_CPU| for ISDF
    # pivots. Optional field — only checked when present in the new run.
    print("--- determinism ---")
    has_nrg = False
    for sname in base_sys:
        snew = new_sys.get(sname)
        if not snew:
            continue
        nsan = snew.get("sanity", {})
        nrg = nsan.get("native_residual_gap")
        if nrg is None or not _is_finite_scalar(nrg):
            continue
        has_nrg = True
        nrg = float(nrg)
        flag = "ok" if nrg < nrg_ceiling else "FAIL"
        print(f"  {sname:<20} native_resid_gap  {nrg:.2e} "
              f"(ceiling {nrg_ceiling:.0e}) [{flag}]")
        if nrg > nrg_ceiling:
            failures.append(
                f"determinism: '{sname}' native_residual_gap {nrg:.2e} "
                f"> ceiling {nrg_ceiling:.0e}")
    if not has_nrg:
        print("  (no native_residual_gap fields — determinism check skipped)")

    # ---- verdict -----------------------------------------------------------
    print("-" * 64)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} issue(s))")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("RESULT: PASS (compat + coverage + sanity + timing within policy)")
    return 0



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEFAULT_SYSTEMS = ["H2O_ccpVDZ", "C2H4_ccpVDZ", "C2H4_ccpVTZ"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ISDF-XTC-CCSD+VMC perf baseline (task #16 Phase 1a)")
    p.add_argument("--systems", nargs="+", default=list(_DEFAULT_SYSTEMS),
                   choices=list(SYSTEMS.keys()),
                   help="subset of systems to run (H30_minimal/benzene are opt-in "
                        "stress/diagnostic vehicles, not in the canonical set)")
    p.add_argument("--with-h30", action="store_true",
                   help="append H30_minimal (STO-3G stress vehicle) to --systems "
                        "for the memory-sensitive --compare (ke-liao + Woke)")
    p.add_argument("--grid-lvl", type=int, default=2)
    p.add_argument("--n-rank-factor", type=float, default=6.0,
                   help="n_rank = int(n_rank_factor * n_orb)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--host-grid-block", type=int, default=2000)
    p.add_argument("--x-block", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-vmc", action="store_true", help="skip VMC step timing")
    p.add_argument("--vmc-focus", action="store_true",
                   help="skip ISDF sub-kernels (kmat/l_aux/d_kernel/x_kernel) "
                        "for VMC-focused profiling; l_aux dominates H30 wall "
                        "(~40 min/repeat) and isn't touched by jastrow PRs")
    p.add_argument("--vmc-walkers", type=int, default=256)
    p.add_argument("--vmc-steps", type=int, default=50)
    p.add_argument("--vmc-burnin", type=int, default=50)
    p.add_argument("--out", type=str, default="benchmarks/perf-baseline.json",
                   help="output JSON path (record mode)")
    p.add_argument("--compare", type=str, default=None,
                   help="baseline JSON to diff against (compare mode)")
    p.add_argument("--threshold", type=float, default=None,
                   help="regression %% threshold for guarded paths (default: read "
                        "from baseline compare_policy in --compare; else module default)")
    p.add_argument("--log-level", type=str, default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--basis-override", type=str, default=None,
                   help="DEV: override basis for all systems (e.g. sto3g) for fast local checks")
    p.add_argument("--canonical", action="store_true",
                   help="validate the run against the required-path/sanity matrix "
                        "before writing; exit nonzero if a required path/sanity "
                        "errored (use for the pinned canonical baseline)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pytc_logger.setLevel(getattr(logging, args.log_level))
    for h in pytc_logger.handlers:
        h.setLevel(getattr(logging, args.log_level))
    print(f"JAX devices: {jax.devices()}  local_device_count={jax.local_device_count()}", flush=True)
    lib.num_threads(1)

    # --with-h30 appends the H30_minimal STO-3G stress vehicle (opt-in; not in
    # the canonical REQUIRED_PATHS, so canonical validation is unaffected).
    selected = list(args.systems)
    if getattr(args, "with_h30", False) and "H30_minimal" not in selected:
        selected.append("H30_minimal")

    systems: Dict[str, Any] = {}
    for sname in selected:
        try:
            systems[sname] = bench_system(sname, SYSTEMS[sname], args)
        except Exception as exc:
            systems[sname] = {"name": sname, "error": str(exc),
                              "trace": traceback.format_exc(limit=3)}

    record = {"metadata": collect_env(),
              "precision_probe": precision_probe(),
              "config": {k: getattr(args, k) for k in
                         ("grid_lvl", "n_rank_factor", "alpha", "batch_size",
                          "host_grid_block", "x_block", "warmup", "repeats",
                          "vmc_walkers", "vmc_steps", "vmc_burnin", "no_vmc",
                          "basis_override", "threshold", "vmc_focus")},
              "compare_policy": COMPARE_POLICY,
              "required_paths": {s: sorted(p) for s, p in REQUIRED_PATHS.items()},
              "required_sanity": {s: sorted(k) for s, k in REQUIRED_SANITY.items()},
              "systems": systems}

    if args.compare:
        sys.exit(compare(record, args.compare, args.threshold))

    if args.canonical:
        violations = validate_against_matrix(record)
        self_policy_violations = _validate_self_policy(record)
        if self_policy_violations:
            violations.extend(self_policy_violations)
        if violations:
            print("\n=== CANONICAL VALIDATION FAILED (weak baseline rejected) ===")
            for v in violations:
                print(f"  - {v}")
            sys.exit(2)
        print("\n=== canonical validation: all required paths/sanity/self-policy checks passed ===")

    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nWrote {args.out}")
    print("\n=== SUMMARY (median seconds) ===")
    for sname, s in systems.items():
        if "timings_s" not in s:
            continue
        print(f"\n[{sname}]  n_orb={s.get('shape',{}).get('n_orb')} e_corr={s.get('sanity',{}).get('e_corr')}")
        for path, pv in s["timings_s"].items():
            if isinstance(pv, dict) and "med" in pv:
                print(f"  {path:<18} {pv['med']:10.4f}")
            elif isinstance(pv, dict) and "error" in pv:
                print(f"  {path:<18} ERROR: {pv['error']}")


if __name__ == "__main__":
    main()
