#!/usr/bin/env python
"""Task B precision experiment: ISDF pivot-divergence root-cause.

Isolates whether the GPU-vs-CPU ``isdf_dU_err`` gap (~1e-3 on GPU vs ~1e-12 on
CPU) comes from ISDF pivot SELECTION (the ``jnp.argmax`` in df._pivoted_cholesky_*
is order-dependent; GPU/CPU can break near-ties differently) or from downstream
numerics. Design per Rick's review:

  * Device is selected BEFORE importing jax/pytc (via JAX_PLATFORMS), so each leg
    runs natively on one backend.
  * Frozen inputs (phi/grad_phi/weights) are generated ONCE and saved to .npz on
    scratch (NOT git); both CPU and GPU legs load the SAME file (sha256 verified),
    so any pivot difference is attributable to the device, not the input.
  * Geometries come from benchmarks.perf_baseline.SYSTEMS (no duplication).
  * Missing grad_phi is FATAL (a zeroed gradient would silently invalidate
    selection).
  * The only library change is the narrowly-scoped ``fixed_pivots`` kwarg on
    df.isdf_decompose / ISDFXTC.from_xtc.

Subcommands:
  generate     build the frozen .npz for one system (run once, CPU)
  native       isdf decompose with native selection on --device; record pivots,
               isdf_dU_err, normal-matrix condition number, (optional) e_corr
  fixed-pivots same, but force the externally supplied --pivots onto the device
  selftest     native then fixed-to-own-pivots on the same device; assert identical
               (proves the override path is faithful before any GPU run)

Causal read: native-GPU vs native-CPU pivots differ AND fixed(GPU, cpu-pivots)
isdf_dU_err collapses to the CPU value -> selection is the cause. Gap persists ->
downstream.
"""
import argparse
import hashlib
import json
import os
import sys


def _set_device(device: str) -> None:
    """Pin the JAX backend BEFORE any jax/pytc import."""
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device in ("gpu", "cuda"):
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        raise SystemExit(f"--device must be cpu|gpu, got {device!r}")
    os.environ.setdefault("JAX_ENABLE_X64", "1")


def _validate_backends(device: str) -> None:
    """Fail-closed: confirm the requested device is available AND cpu backend is
    present (df.py requires it for xi storage). Called after jax import."""
    import jax
    if device == "cpu":
        if jax.default_backend() != "cpu":
            raise SystemExit("FATAL: --device cpu but default backend is not cpu")
    elif device in ("gpu", "cuda"):
        if jax.default_backend() not in ("gpu", "cuda"):
            raise SystemExit("FATAL: --device gpu but default backend is not gpu/cuda")
        if not jax.devices("cpu"):
            raise SystemExit("FATAL: --device gpu requires cpu backend (df.py xi storage); JAX_PLATFORMS=cuda,cpu")
    else:
        raise SystemExit(f"--device must be cpu|gpu, got {device!r}")


def _sha(arr) -> str:
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _build(system: str, grid_lvl: int, alpha: float):
    """Deterministic build of (mol, mf, xtc, jp, n_rank) for a SYSTEMS entry."""
    import numpy as np
    import jax.numpy as jnp
    from pyscf import gto, scf
    from pytc.xtc import XTC
    from pytc.jastrow.rexp import REXP
    from benchmarks.perf_baseline import SYSTEMS

    if system not in SYSTEMS:
        raise SystemExit(f"unknown system {system!r}; choices: {list(SYSTEMS)}")
    cfg = SYSTEMS[system]
    mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    jas = REXP()
    jp = {"alpha": jnp.array([alpha])}
    xtc = XTC.from_pyscf(mf, jas, grid_lvl=grid_lvl)
    return mol, mf, xtc, jp


def _n_rank(xtc, n_rank_factor: float) -> int:
    return max(4, int(n_rank_factor * int(xtc.n_orb)))


# All XTC grid/orbital state the dU + pivot pipeline consumes. Both legs must
# load EXACTLY these (not regenerate them per device) so any difference is
# attributable to the device, not to grid/SCF regeneration (Rick R1).
_FROZEN_ARRAYS = ("phi", "grad_phi", "weights", "grid_points", "mo_coeff", "mo_occ")


def cmd_generate(args) -> None:
    import numpy as np
    _, _, xtc, jp = _build(args.system, args.grid_lvl, args.alpha)
    arrays = {k: np.asarray(getattr(xtc, k)) for k in _FROZEN_ARRAYS}
    gp = arrays["grad_phi"]
    if gp is None or gp.size == 0 or not np.any(gp):
        raise SystemExit("FATAL: grad_phi missing/all-zero; refusing to write frozen input")
    n_rank = _n_rank(xtc, args.n_rank_factor)
    manifest = {
        "system": args.system, "grid_lvl": args.grid_lvl, "alpha": args.alpha,
        "n_rank_factor": args.n_rank_factor, "n_rank": n_rank,
        "n_orb": int(xtc.n_orb), "n_grid": int(arrays["phi"].shape[1]),
        "nocc": int(xtc.nocc), "energy_nuc": float(np.asarray(xtc.energy_nuc)),
        "sha": {k: _sha(v) for k, v in arrays.items()},
        "dtype": {k: str(v.dtype) for k, v in arrays.items()},
    }
    np.savez(args.npz, manifest=json.dumps(manifest), **arrays)
    print(json.dumps({"wrote": args.npz, **manifest}, indent=2))


def _load_frozen(npz_path: str, args=None):
    """Load + sha-verify every frozen array; validate manifest vs CLI args."""
    import numpy as np
    d = np.load(npz_path, allow_pickle=True)
    man = json.loads(str(d["manifest"]))
    arrays = {}
    for k in _FROZEN_ARRAYS:
        arr = d[k]
        got = _sha(arr)
        if got != man["sha"][k]:
            raise SystemExit(f"FATAL: {k} sha mismatch ({got} != {man['sha'][k]}); frozen input not intact")
        arrays[k] = arr
    if not np.any(arrays["grad_phi"]):
        raise SystemExit("FATAL: grad_phi all-zero in frozen input")
    if args is not None:
        for fld, attr in [("system", "system"), ("grid_lvl", "grid_lvl"),
                          ("alpha", "alpha"), ("n_rank_factor", "n_rank_factor")]:
            if man.get(fld) != getattr(args, attr):
                raise SystemExit(f"FATAL: manifest {fld}={man.get(fld)!r} != --{attr.replace('_','-')}={getattr(args, attr)!r}")
    return arrays, man


def _run_one(args, fixed_pivots=None):
    """Native (fixed_pivots=None) or fixed-pivot ISDF on the pinned device.

    Returns dict with pivots, isdf_dU_err, cond, (e_corr). Uses the FROZEN
    phi/grad_phi/weights injected into a fresh xtc so both legs are identical
    on input; only the device (and supplied pivots) vary.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pytc import df  # noqa: F401  (df hook used via ISDFXTC.from_xtc)
    from pytc.xtc import ISDFXTC

    arrays, man = _load_frozen(args.npz, args)
    backend = jax.default_backend()
    # Rebuild a fresh xtc/mf for the full dU + eris machinery, then inject EVERY
    # frozen grid/orbital array so CPU and GPU operate on identical input state
    # (Rick R1: regenerating grid_points/mo_coeff per device would confound).
    _, mf, xtc, jp = _build(args.system, args.grid_lvl, args.alpha)
    xtc = xtc.replace(**{k: jnp.asarray(arrays[k]) for k in _FROZEN_ARRAYS})
    mf.mo_coeff = np.asarray(arrays["mo_coeff"])
    mf.mo_occ = np.asarray(arrays["mo_occ"])
    n_rank = _n_rank(xtc, args.n_rank_factor)

    out = {"device": args.device, "backend": backend, "system": args.system,
           "n_rank": n_rank, "manifest_sha": man["sha"],
           "mode": "fixed" if fixed_pivots is not None else "native"}

    # --- ISDF object is the SINGLE authoritative selection (Rick R3): pivots come
    #     from the same ixtc that computes dU, never a second isdf_decompose call.
    ixtc = ISDFXTC.from_xtc(xtc, n_rank=n_rank, is_incore=True, fixed_pivots=fixed_pivots)
    pivots = np.asarray(jax.device_get(ixtc.pivots)).astype(int)
    out["pivots"] = pivots.tolist()
    out["n_fused"] = int(pivots.shape[0])
    if fixed_pivots is not None:
        fp = np.asarray(fixed_pivots).astype(int)
        if not np.array_equal(pivots, fp):
            raise SystemExit(f"FATAL: ixtc.pivots != supplied fixed_pivots (override not honored)")

    # --- density normal-matrix condition number at the authoritative pivots.
    #     NOTE: density only; the three gradient-channel normal matrices are NOT
    #     yet reported (Rick R5) -- treat conditioning evidence as incomplete. ---
    phi_piv = jnp.asarray(arrays["phi"])[:, jnp.asarray(pivots)]
    out["cond_phi_density"] = float(jax.device_get(jnp.linalg.cond(df._build_normal_matrix(phi_piv, phi_piv))))
    out["cond_grad_channels"] = "not_computed"  # follow-up

    # --- du_exact and du_isdf reported SEPARATELY (Rick R4) so a device delta in
    #     the exact path can't masquerade as a pivot effect. Compare across legs offline.
    du = np.asarray(jax.block_until_ready(xtc.get_delta_U(jp)))
    ixtc2 = ixtc.isdf(jp)
    du_i = np.asarray(jax.block_until_ready(ixtc2.get_delta_U(jp)))
    out["du_exact_sha"] = _sha(du)
    out["du_isdf_sha"] = _sha(du_i)
    out["du_exact_maxabs"] = float(np.max(np.abs(du)))
    out["du_isdf_maxabs"] = float(np.max(np.abs(du_i)))
    out["isdf_dU_err"] = float(np.max(np.abs(du_i - du)))

    # --- e_corr THROUGH the ISDF object (Rick R2): make_eris on ixtc2 dispatches
    #     get_2b via ISDF kernels, so e_corr actually reflects the pivots. If the
    #     ISDF object lacks make_eris, we do NOT silently fall back to exact-XTC
    #     (which would be pivot-independent and misleading). When --with-ecorr is
    #     requested, energy errors are FATAL (fail-closed, Rick R-v4).
    if args.with_ecorr:
        from pyscf import cc
        if not hasattr(ixtc2, "make_eris"):
            raise SystemExit("FATAL: --with-ecorr requested but ISDF object has no make_eris")
        eris = ixtc2.make_eris(mf, jp)
        mycc = cc.rccsd.RCCSD(mf)
        ec, _, _ = mycc.kernel(eris=eris)
        out["e_corr_isdf"] = float(ec)

    # --- serialize AFTER the energy block so result NPZs carry e_corr (Rick R-v3).
    #     du arrays + e_corr enable compare-results to compute the residual-based
    #     collapse measure and the CPU↔GPU energy delta. ---
    if getattr(args, "save_result", None):
        meta = {"system": args.system, "device": args.device,
                "mode": out["mode"], "n_rank": n_rank,
                "manifest_sha": man["sha"], "pivots": pivots.tolist(),
                "with_ecorr": bool(args.with_ecorr)}
        if args.with_ecorr:
            meta["e_corr_isdf"] = out["e_corr_isdf"]
        np.savez(args.save_result, du_exact=du, du_isdf=du_i,
                 meta=json.dumps(meta))
    return out


def _jaccard(a, b) -> float:
    import numpy as np
    sa, sb = set(np.asarray(a).astype(int).tolist()), set(np.asarray(b).astype(int).tolist())
    return (len(sa & sb) / len(sa | sb)) if (sa | sb) else 1.0


def cmd_native(args) -> None:
    import numpy as np
    out = _run_one(args, fixed_pivots=None)
    print(json.dumps(out, indent=2))
    if args.save_pivots:
        # Bind pivots to their provenance so fixed mode can reject mismatches (Rick R6).
        np.savez(args.save_pivots, pivots=np.asarray(out["pivots"], dtype=int),
                 meta=json.dumps({"system": args.system, "n_rank": out["n_rank"],
                                  "manifest_sha": out["manifest_sha"], "device": args.device}))
        print(f"# pivots saved to {args.save_pivots}", file=sys.stderr)


def cmd_fixed(args) -> None:
    import numpy as np
    pv = np.load(args.pivots, allow_pickle=True)
    if "meta" not in getattr(pv, "files", []):
        raise SystemExit("FATAL: pivot file has no provenance meta; refusing (use a file written by `native --save-pivots`)")
    meta = json.loads(str(pv["meta"]))
    pivots = np.asarray(pv["pivots"]).astype(int)
    _, man = _load_frozen(args.npz, args)
    # Provenance must bind to THIS system/frozen-input/rank, and pivots must come
    # from the CPU leg (we force CPU-selected pivots onto the device). (Rick R-v2-2)
    if meta.get("device") != "cpu":
        raise SystemExit(f"FATAL: pivot file device {meta.get('device')!r} != 'cpu'; the override must use CPU-selected pivots")
    if meta.get("system") != args.system:
        raise SystemExit(f"FATAL: pivot file system {meta.get('system')!r} != --system {args.system!r}")
    if meta.get("manifest_sha") != man["sha"]:
        raise SystemExit("FATAL: pivot file selected from a different frozen input (manifest sha mismatch)")
    if int(meta.get("n_rank", -1)) != _n_rank_from_manifest(man):
        raise SystemExit("FATAL: pivot file n_rank mismatch")
    if pivots.ndim != 1 or len(np.unique(pivots)) != len(pivots):
        raise SystemExit("FATAL: supplied pivots must be 1-D and unique")
    if pivots.size == 0 or pivots.min() < 0 or pivots.max() >= int(man["n_grid"]):
        raise SystemExit("FATAL: supplied pivots out of grid range")
    out = _run_one(args, fixed_pivots=pivots)
    out["from_pivots_device"] = meta["device"]
    print(json.dumps(out, indent=2))


def _n_rank_from_manifest(man) -> int:
    return int(man.get("n_rank", -1))


def cmd_selftest(args) -> None:
    """native then fixed-to-own-pivots on the SAME device must match exactly."""
    import numpy as np
    nat = _run_one(args, fixed_pivots=None)
    own = np.asarray(nat["pivots"], dtype=int)
    fix = _run_one(args, fixed_pivots=own)
    jac = _jaccard(nat["pivots"], fix["pivots"])
    d_err = abs(nat["isdf_dU_err"] - fix["isdf_dU_err"])
    ok = (jac == 1.0) and (nat["du_isdf_sha"] == fix["du_isdf_sha"]) and d_err < 1e-12
    res = {"selftest_ok": None, "pivot_jaccard": jac,
           "du_isdf_sha_match": nat["du_isdf_sha"] == fix["du_isdf_sha"],
           "isdf_dU_err_native": nat["isdf_dU_err"],
           "isdf_dU_err_fixed_own": fix["isdf_dU_err"], "abs_diff": d_err}
    if args.with_ecorr:
        if "e_corr_isdf" not in nat or "e_corr_isdf" not in fix:
            res["selftest_ok"] = False
            res["e_corr_isdf_match"] = False
            res["e_corr_error"] = "missing e_corr_isdf in one or both runs"
            print(json.dumps(res, indent=2))
            sys.exit(1)
        e_ok = abs(nat["e_corr_isdf"] - fix["e_corr_isdf"]) < 1e-10
        res["e_corr_isdf_match"] = e_ok
        res["e_corr_isdf_native"], res["e_corr_isdf_fixed_own"] = nat["e_corr_isdf"], fix["e_corr_isdf"]
        ok = ok and e_ok
    res["selftest_ok"] = bool(ok)
    print(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)


def cmd_compare(args) -> None:
    """Decisive cross-device comparison (Rick R-v2-1). Loads result NPZs (du arrays)
    from native-CPU, native-GPU, and fixed-GPU(cpu-pivots), and reports the deltas
    that actually establish the fork."""
    import numpy as np
    EXACT_TOL = args.exact_tol

    def _load(path):
        d = np.load(path, allow_pickle=True)
        return np.asarray(d["du_exact"]), np.asarray(d["du_isdf"]), json.loads(str(d["meta"]))

    ex_c, is_c, m_c = _load(args.cpu_native)
    ex_g, is_g, m_g = _load(args.gpu_native)
    ex_f, is_f, m_f = _load(args.gpu_fixed)

    # --- fail-closed validation (Rick R-v3): roles, system, frozen input, shapes, finiteness ---
    def _is_gpu(d): return d in ("gpu", "cuda")
    if not (m_c["manifest_sha"] == m_g["manifest_sha"] == m_f["manifest_sha"]):
        raise SystemExit("FATAL: result files come from different frozen inputs (manifest_sha mismatch)")
    if not (m_c["system"] == m_g["system"] == m_f["system"]):
        raise SystemExit("FATAL: result files are different systems")
    if not (m_c["device"] == "cpu" and m_c["mode"] == "native"):
        raise SystemExit("FATAL: --cpu-native must be a CPU native result")
    if not (_is_gpu(m_g["device"]) and m_g["mode"] == "native"):
        raise SystemExit("FATAL: --gpu-native must be a GPU native result")
    if not (_is_gpu(m_f["device"]) and m_f["mode"] == "fixed"):
        raise SystemExit("FATAL: --gpu-fixed must be a GPU fixed-pivots result")
    if m_f["pivots"] != m_c["pivots"]:
        raise SystemExit("FATAL: fixed-GPU pivots != CPU native pivots (override did not use CPU pivots)")
    if not (ex_c.shape == is_c.shape == ex_g.shape == is_g.shape == ex_f.shape == is_f.shape):
        raise SystemExit("FATAL: du array shapes differ across results")
    for nm, a in [("ex_c", ex_c), ("is_c", is_c), ("ex_g", ex_g), ("is_g", is_g), ("ex_f", ex_f), ("is_f", is_f)]:
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"FATAL: non-finite values in {nm}")

    # --- residual-based collapse measure (Rick R-v3): the metric under study is the
    #     ISDF residual r = du_isdf - du_exact (== isdf_dU_err pointwise), which removes
    #     any exact-path device confound. Raw deltas kept as diagnostics only. ---
    r_c, r_g, r_f = is_c - ex_c, is_g - ex_g, is_f - ex_f
    exact_native_delta = float(np.max(np.abs(ex_g - ex_c)))
    exact_fixed_delta = float(np.max(np.abs(ex_f - ex_c)))
    native_residual_gap = float(np.max(np.abs(r_g - r_c)))
    fixed_residual_gap = float(np.max(np.abs(r_f - r_c)))

    if exact_native_delta > EXACT_TOL or exact_fixed_delta > EXACT_TOL:
        parts = []
        if exact_native_delta > EXACT_TOL:
            parts.append(f"native exact delta {exact_native_delta:.2e} > {EXACT_TOL:.0e}")
        if exact_fixed_delta > EXACT_TOL:
            parts.append(f"fixed exact delta {exact_fixed_delta:.2e} > {EXACT_TOL:.0e}")
        verdict = ("inconclusive: exact-path control failed — " + "; ".join(parts)
                   + " -> frozen-state control failed; residual attribution unsafe")
    elif native_residual_gap <= EXACT_TOL:
        verdict = "inconclusive: no meaningful native CPU/GPU residual gap to explain"
    elif fixed_residual_gap < 0.1 * native_residual_gap:
        verdict = "SELECTION: forcing CPU pivots on GPU collapses the residual gap"
    else:
        verdict = "DOWNSTREAM: residual gap persists with CPU pivots forced -> not selection"

    res = {
        "verdict": verdict,
        "native_residual_gap": native_residual_gap,
        "fixed_residual_gap": fixed_residual_gap,
        "exact_path_native_cpu_vs_gpu_delta": exact_native_delta,
        "exact_path_fixed_cpu_vs_gpu_delta": exact_fixed_delta,
        "diag_isdf_native_cpu_vs_gpu": float(np.max(np.abs(is_g - is_c))),
        "diag_isdf_fixedGPU_vs_cpu": float(np.max(np.abs(is_f - is_c))),
        "gpu_native_pivots_match_cpu": m_g["pivots"] == m_c["pivots"],
    }
    # energy deltas when all three result NPZs carry finite e_corr_isdf (Rick R-v4).
    # If --with-ecorr was used for the runs, e_corr_isdf MUST be present and finite
    # in all three results — fail closed so the manuscript-accuracy leg can't disappear.
    if m_c.get("with_ecorr") or m_g.get("with_ecorr") or m_f.get("with_ecorr"):
        e_c, e_g, e_f = m_c.get("e_corr_isdf"), m_g.get("e_corr_isdf"), m_f.get("e_corr_isdf")
        if not all(isinstance(e, (int, float)) and not isinstance(e, bool) and np.isfinite(e)
                   for e in (e_c, e_g, e_f)):
            raise SystemExit("FATAL: --with-ecorr was used but one or more results lack finite e_corr_isdf")
        res["e_corr_isdf_cpu"], res["e_corr_isdf_gpu_native"], res["e_corr_isdf_gpu_fixed"] = e_c, e_g, e_f
        res["e_corr_isdf_native_cpu_vs_gpu_delta"] = abs(float(e_g) - float(e_c))
        res["e_corr_isdf_fixed_cpu_vs_gpu_delta"] = abs(float(e_f) - float(e_c))
    print(json.dumps(res, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["generate", "native", "fixed-pivots", "selftest", "compare-results"])
    p.add_argument("--system", default="H2O_ccpVDZ")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    p.add_argument("--grid-lvl", type=int, default=2)
    p.add_argument("--n-rank-factor", type=float, default=6.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--npz", default="frozen_inputs.npz")
    p.add_argument("--pivots", default=None, help="npz with 'pivots' (for fixed-pivots)")
    p.add_argument("--save-pivots", default=None, help="(native) write selected pivots to this npz")
    p.add_argument("--save-result", default=None, help="(native/fixed) write du arrays + meta for compare-results")
    p.add_argument("--with-ecorr", action="store_true")
    p.add_argument("--cpu-native", default=None, help="(compare-results) native CPU result npz")
    p.add_argument("--gpu-native", default=None, help="(compare-results) native GPU result npz")
    p.add_argument("--gpu-fixed", default=None, help="(compare-results) fixed-GPU(cpu-pivots) result npz")
    p.add_argument("--exact-tol", type=float, default=1e-9, help="(compare-results) exact-path threshold for frozen-state control")
    args = p.parse_args()

    if args.cmd == "compare-results":
        if not (args.cpu_native and args.gpu_native and args.gpu_fixed):
            raise SystemExit("compare-results requires --cpu-native --gpu-native --gpu-fixed")
        import numpy as np
        if not (np.isfinite(args.exact_tol) and args.exact_tol > 0):
            raise SystemExit("--exact-tol must be finite and > 0")
        cmd_compare(args)  # pure numpy, no JAX backend needed
        return

    _set_device(args.device)  # MUST precede any jax/pytc import below
    _validate_backends(args.device)
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "native":
        cmd_native(args)
    elif args.cmd == "fixed-pivots":
        if not args.pivots:
            raise SystemExit("fixed-pivots requires --pivots <cpu_native_result.npz>")
        cmd_fixed(args)
    elif args.cmd == "selftest":
        cmd_selftest(args)


if __name__ == "__main__":
    main()
