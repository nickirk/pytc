"""Run optimize_ref_var for N steps and plot the reference-variance loss
(and energy) vs optimization step, for Ke's convergence-behavior request
(task #1, #proj-pytc-efficiency-refactor) -- the per-phase timing harness
(profile_vmc_ref_var_phases.py) only measures ONE Newton step, it doesn't
track convergence across steps, so this is a separate script.

Uses optimize_ref_var's own history tracking (result["cost"]/["energies"]/
["stds"]/["acceptance"]) rather than reimplementing a training loop.

Usage:
    python plot_variance_convergence.py --system water --n-water 25 \
        --basis cc-pVTZ --n-walkers 1000 --n-opt-steps 100 \
        --jac-batch-size 64 \
        --out h2o25_convergence.json --plot h2o25_convergence.png
"""
import argparse
import json
import os
import subprocess
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.vmc import optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
from pytc.jastrow.bha import BoysHandyAnalytical


def h_chain(n, sep=1.8):
    return "; ".join(f"H 0 0 {i*sep}" for i in range(n))


def resolve_scf_max_memory(explicit_mb):
    """PySCF's Mole.max_memory defaults to 4000 MB -- at large nao (e.g.
    H300, nao~4200) the DF integral build degenerates into tiny batches
    sized to that tiny budget, becoming I/O-bound instead of BLAS-bound
    (Felix's diagnosis, 2026-07-11, H300 SCF bottleneck). Read the actual
    SLURM allocation so PySCF sizes its batches to the real node memory.
    """
    if explicit_mb is not None:
        return explicit_mb
    slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mb:
        try:
            return int(slurm_mb)
        except ValueError:
            pass
    return 64000  # 64 GB fallback for non-SLURM/local runs


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


def _to_host(x):
    """Convert a GPU-resident array (e.g. cupy, from gpu4pyscf) to host.

    ``mf.dump_chk``/``scf.chkfile.dump_scf`` write via h5py, which can't
    serialize device arrays directly. cupy arrays expose ``.get()``;
    plain numpy arrays/Python scalars don't, so they pass through
    unchanged -- this is backend-agnostic without importing cupy.
    """
    get = getattr(x, "get", None)
    return get() if callable(get) else x


def _preload_gpu4pyscf_cusolver():
    """Explicitly dlopen the real cusolver .so before gpu4pyscf imports it.

    gpu4pyscf's ``lib/cusolver.py`` resolves the library via
    ``ctypes.util.find_library('cusolver')``, which returns None for
    pip-installed ``nvidia-cusolver-cu12`` wheels (they sit in
    site-packages, never registered in ldconfig's cache). gpu4pyscf then
    calls ``ctypes.CDLL(None)``, which silently binds to the current
    process's own global symbol table instead of raising -- it "works"
    for symbols something else (e.g. jax's own CUDA libs) already
    exported, and fails opaquely on ones nothing else happens to export
    (``cusolverDnDsygvd_bufferSize`` was the one that broke here). The
    wheel itself is fine (confirmed via ``nm -D``) -- it's purely a
    dlopen-resolution miss, not a version conflict (Grace's diagnosis,
    2026-07-12, #proj-pytc-efficiency-refactor).

    Loading the real .so here with RTLD_GLOBAL beforehand makes gpu4pyscf's
    own (broken) lookup a no-op, so no launch-time LD_PRELOAD env var is
    required -- self-contained in code instead of operator-remembered
    launch state (Felix's call: same untracked-config bug class as the
    damping/LR-schedule episodes this campaign already hit twice).

    Returns the preloaded .so path, or None if the nvidia-cusolver-cu12
    package wasn't found (gpu4pyscf's own import is left to fail with its
    usual error in that case).
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia.cusolver")
    except ModuleNotFoundError:
        # find_spec raises (rather than returning None) when the parent
        # package itself isn't importable, e.g. no "nvidia" namespace
        # package present at all (a CPU-only host).
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    cusolver_dir = spec.submodule_search_locations[0]
    so_path = os.path.join(cusolver_dir, "lib", "libcusolver.so.11")
    if not os.path.exists(so_path):
        return None
    import ctypes
    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    return so_path


def _n_lindep_removed(mf, threshold):
    """Ground-truth count of overlap-matrix eigenvalues below ``threshold``,
    computed directly from the matrix itself (not parsed from any
    backend's log output)."""
    s = np.asarray(_to_host(mf.get_ovlp()))
    eigvals = np.linalg.eigvalsh(s)
    return int(np.sum(eigvals < threshold))


def _pin_lindep_threshold(mf, threshold, backend):
    """Apply canonical orthogonalization at an explicit, documented
    threshold instead of inheriting each backend's own default overlap-
    matrix conditioning behavior.

    Different backends' default eigensolvers can drop a different number
    of near-linearly-dependent overlap eigenvectors at the same system
    (observed: pyscf CPU dropped 0, gpu4pyscf dropped 7, on the same
    H160/cc-pVTZ overlap matrix, using its OWN internal
    ``overlap_zero_eigenvalue_threshold`` default of 1e-6 -- a real
    policy difference between backends, not roundoff; confirmed via
    gpu4pyscf's public source, Grace/Felix, 2026-07-12,
    #proj-pytc-efficiency-refactor). Same overlap matrix + same explicit
    threshold makes the retained space backend-independent by construction.

    ``pyscf``: pyscf.scf.addons.remove_linear_dep_ is numpy-only end to
    end (it installs a custom eigensolver used on every SCF cycle,
    including the Fock matrix), so it's only called when actually
    needed (n_removed > 0) -- the common case (nothing to remove, true
    for cc-pVTZ-scale systems at the pinned default) skips it entirely.

    ``gpu4pyscf``: it has its own module-level
    ``gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold``, read at SCF-
    cycle time by its (cupy-native) ``check_linear_dependency`` -- set
    directly to our pinned threshold so gpu4pyscf's own internal
    machinery honors it end-to-end, no pyscf/numpy interop needed.

    Returns n_lindep_removed (ground truth, computed independently of
    either backend's internal accounting).
    """
    if backend == "pyscf":
        n_removed = _n_lindep_removed(mf, threshold)
        if n_removed > 0:
            scf.addons.remove_linear_dep_(mf, threshold=threshold)
        return n_removed
    elif backend == "gpu4pyscf":
        import gpu4pyscf.scf.hf as gpu4pyscf_hf
        gpu4pyscf_hf.overlap_zero_eigenvalue_threshold = threshold
        return _n_lindep_removed(mf, threshold)
    else:
        raise ValueError(f"Unknown --scf-backend: {backend!r}")


def _apply_scf_convergence_settings(mf, max_cycle, level_shift, diis_space):
    """Plain passthrough to the mf object -- same attribute names on both
    pyscf and gpu4pyscf (confirmed against gpu4pyscf's source: max_cycle/
    diis_space/level_shift all mirror pyscf.scf.hf.SCF's). None means
    "leave at the backend's own default," not "explicitly set to a
    falsy value."
    """
    if max_cycle is not None:
        mf.max_cycle = max_cycle
    if level_shift is not None:
        mf.level_shift = level_shift
    if diis_space is not None:
        mf.diis_space = diis_space


def make_rhf(mol, backend="pyscf", lindep_threshold=1e-8,
             max_cycle=None, level_shift=None, diis_space=None):
    """Construct the density-fitted RHF object for the requested SCF backend.

    ``pyscf`` (default) is the existing CPU DF-RHF path, byte-for-byte
    unchanged. ``gpu4pyscf`` swaps in gpu4pyscf's GPU-accelerated RHF --
    Felix's spike (2026-07-11) to test whether GPU DF-SCF is the real fix
    for H300's bottleneck, since CPU max_memory sizing alone only bought
    1.39x, far short of the hoped-for 10-50x. Import is lazy so choosing
    "pyscf" never requires gpu4pyscf (a CUDA-only package) to be installed.

    max_cycle/level_shift/diis_space: None (default) leaves the
    backend's own default; explicit values are passed straight through.
    Needed for systems with near-degenerate frontier orbitals (e.g.
    H300/cc-pVTZ) where the default SCF settings oscillate instead of
    converging (Grace/Felix, 2026-07-12, #proj-pytc-efficiency-refactor).

    Returns (mf, cusolver_preload_path, n_lindep_removed). The preload
    path is None for the pyscf backend, or for gpu4pyscf if no preload
    was needed/found.
    """
    if backend == "pyscf":
        mf = scf.RHF(mol).density_fit()
        _apply_scf_convergence_settings(mf, max_cycle, level_shift, diis_space)
        n_lindep_removed = _pin_lindep_threshold(mf, lindep_threshold, backend)
        return mf, None, n_lindep_removed
    elif backend == "gpu4pyscf":
        preload_path = _preload_gpu4pyscf_cusolver()
        try:
            from gpu4pyscf import scf as gpu_scf
        except ImportError as e:
            raise ImportError(
                "--scf-backend gpu4pyscf requires the gpu4pyscf package "
                "(GPU-only, needs CUDA) -- not installed in this environment."
            ) from e
        mf = gpu_scf.RHF(mol).density_fit()
        _apply_scf_convergence_settings(mf, max_cycle, level_shift, diis_space)
        n_lindep_removed = _pin_lindep_threshold(mf, lindep_threshold, backend)
        return mf, preload_path, n_lindep_removed
    else:
        raise ValueError(f"Unknown --scf-backend: {backend!r}")


def run_scf(mol, atom, basis, unit, cache_dir, backend="pyscf", lindep_threshold=1e-8,
            max_cycle=None, level_shift=None, diis_space=None):
    """Same caching pattern as profile_vmc_ref_var_phases.py -- large
    systems shouldn't re-pay SCF on every convergence-plot rerun either.

    Returns (mf, scf_time_s, was_cached, cusolver_preload_path,
    n_lindep_removed)."""
    mf, preload_path, n_lindep_removed = make_rhf(
        mol, backend, lindep_threshold, max_cycle, level_shift, diis_space)
    if not cache_dir:
        t0 = time.time()
        mf.kernel()
        print(f"SCF: converged={mf.converged}  e_tot={mf.e_tot}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
        return mf, time.time() - t0, False, preload_path, n_lindep_removed
    os.makedirs(cache_dir, exist_ok=True)
    import hashlib
    # max_cycle/level_shift/diis_space affect the optimization trajectory,
    # not just speed -- for near-degenerate systems a different path can
    # converge to a different local solution, so they're part of the
    # cache key too (same principle as backend/lindep_threshold above).
    cache_key = hashlib.sha256(
        f"{atom}|{basis}|{unit}|{backend}|{lindep_threshold}|"
        f"{max_cycle}|{level_shift}|{diis_space}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{cache_key}.h5")
    converged_marker = cache_path + ".converged"
    t0 = time.time()
    cache_hit = os.path.exists(cache_path) and os.path.exists(converged_marker)
    print(f"SCF cache: key={cache_key}  path={cache_path}  "
          f"{'HIT' if cache_hit else 'MISS'}", flush=True)
    # PySCF writes the chkfile every SCF cycle, so a job killed mid-SCF
    # leaves a chkfile with partial, unconverged orbitals. Only trust the
    # cache if the converged-marker is present too -- it's written below
    # only after mf.kernel() actually returns with mf.converged True, so
    # its absence means "don't trust this," not "converged, load it."
    if cache_hit:
        loaded = scf.chkfile.load(cache_path, "scf")
        mf.mo_coeff = loaded["mo_coeff"]
        mf.mo_energy = loaded["mo_energy"]
        mf.mo_occ = loaded["mo_occ"]
        mf.e_tot = loaded["e_tot"]
        mf.converged = True
        return mf, time.time() - t0, True, preload_path, n_lindep_removed
    # Don't pre-set mf.chkfile -- PySCF's during-kernel() incremental
    # chkfile writes are what leaves partial files behind on interrupted
    # runs, and whether that mechanism even fires reliably through
    # density_fit()'s decorator is version-dependent. Explicitly dump the
    # final converged state ourselves, once, via scf.chkfile.dump_scf
    # directly (same call mf.dump_chk's string-path form makes
    # internally) with each array passed through _to_host first, since
    # gpu4pyscf's mo_coeff/mo_energy/mo_occ may be cupy device arrays
    # that h5py can't serialize.
    mf.kernel()
    print(f"SCF: converged={mf.converged}  e_tot={mf.e_tot}  "
          f"elapsed={time.time() - t0:.1f}s", flush=True)
    if mf.converged:
        scf.chkfile.dump_scf(
            mf.mol, cache_path,
            _to_host(mf.e_tot), _to_host(mf.mo_energy),
            _to_host(mf.mo_coeff), _to_host(mf.mo_occ),
        )
        open(converged_marker, "w").close()
    return mf, time.time() - t0, False, preload_path, n_lindep_removed


def save_figure(fig, path, meta):
    """Save a figure with a provenance footer (commit/system/params) baked
    into the image itself -- no established save_figure() helper existed
    elsewhere in the repo, so this mirrors the JSON provenance fields
    (git_commit(), system, basis, n_walkers, jac_batch_size) already used
    by profile_vmc_ref_var_phases.py, just rendered onto the PNG so the
    plot is self-describing even if separated from its JSON sidecar."""
    footer = (
        f"commit={meta['commit'][:12]}  system={meta['system']}  basis={meta['basis']}  "
        f"W={meta['n_walkers']}  burn_in={meta['burn_in_steps']}  "
        f"jac_batch={meta['jac_batch_size'] or 'full'}"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7, color="gray")
    fig.savefig(path, dpi=150)


def build_system(args):
    if args.system == "h-chain":
        return h_chain(args.n_atoms), "Bohr", f"H{args.n_atoms}"
    elif args.system == "water":
        from pytc.utils.gen_water_cluster import build_water_cluster
        return build_water_cluster(args.n_water), "Angstrom", f"(H2O){args.n_water}"
    raise ValueError(args.system)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["h-chain", "water"], required=True)
    p.add_argument("--n-atoms", type=int, default=20)
    p.add_argument("--n-water", type=int, default=2)
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=1000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--n-opt-steps", type=int, default=50,
                    help="Felix's spec for the first GPU convergence run: 50 steps, "
                         "extend later if the curve looks healthy.")
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--learning-rate", type=float, default=0.01,
                    help="Matches production's phase-B learning rate (confirmed "
                         "from the H-chain campaign's run.sh COMMON_FLAGS). The "
                         "original 0.1 default was an untracked choice from an "
                         "early small-molecule stability note that never matched "
                         "production practice -- 10x too large, part of what drove "
                         "the (H2O)25/W=5000 Newton step into the b/d-saturation "
                         "plateau (2026-07-11 NaN investigation).")
    p.add_argument("--damping", type=float, default=1e-3,
                    help="Matches production's effective damping (create_optimizer's "
                         "own default -- production's run_opt.py never overrides "
                         "it). The original 1e-6 default was 1000x less "
                         "regularization than production practice, same "
                         "untracked-default issue as learning-rate above.")
    p.add_argument("--jac-batch-size", type=int, default=0,
                    help="max_vmap_batch_size passed to optimize_ref_var -- unlike "
                         "the timing harness, optimize_ref_var exposes only ONE "
                         "combined batch-size knob shared by burn_in/MCMC/Jacobian, "
                         "so this sizes all three. Match to whatever fits per task "
                         "#6's Wave-2 per-system sizing (e.g. 64 for (H2O)25 at "
                         "moderate W, 32 for W=5000+).")
    p.add_argument("--n-mcmc-per-opt", type=int, default=20,
                    help="Wired to optimize_ref_var(n_mcmc_per_opt=..., "
                         "n_opt_per_mcmc=1) -- 20 MCMC decorrelation steps between "
                         "each Newton update, matching production's cadence "
                         "(confirmed from the H-chain campaign's run.sh). Without "
                         "this, optimize_ref_var's legacy default "
                         "(n_mcmc_per_opt unset -> n_opt_per_mcmc=n_steps=20) runs "
                         "the OPPOSITE pattern -- 20 Newton substeps chained on the "
                         "SAME frozen walkers per outer step (1000 total Jacobian "
                         "builds for 50 outer steps, ~35h+, and the exact stale-"
                         "walker pathology that caused the 2026-07-11 NaN "
                         "investigation in the first place). Felix's catch.")
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"),
                    help="Same cache as profile_vmc_ref_var_phases.py; empty string disables.")
    p.add_argument("--scf-max-memory", type=int, default=None,
                    help="PySCF Mole.max_memory in MB. Default: auto-detect from "
                         "SLURM_MEM_PER_NODE, else 64000 (64GB) fallback. PySCF's "
                         "own default is 4000MB, which at large nao (e.g. H300) "
                         "degenerates the DF integral build into tiny I/O-bound "
                         "batches (Felix's H300 SCF-bottleneck diagnosis, "
                         "2026-07-11).")
    p.add_argument("--scf-log", default=None,
                    help="Redirect PySCF's own verbose=4 SCF log to this file path "
                         "(kept off stdout to avoid flooding repeated diagnostic "
                         "runs). None = PySCF stays silent (verbose=0), matching "
                         "prior behavior.")
    p.add_argument("--scf-backend", choices=["pyscf", "gpu4pyscf"], default="pyscf",
                    help="SCF backend for the density-fitted RHF calculation. "
                         "'pyscf' (default) is the existing CPU path, unchanged. "
                         "'gpu4pyscf' uses gpu4pyscf's GPU-accelerated DF-RHF "
                         "(requires the gpu4pyscf package + CUDA GPU) -- Felix's "
                         "spike to test whether GPU SCF is the real fix for "
                         "H300's bottleneck, since CPU max_memory sizing alone "
                         "only bought 1.39x (2026-07-11). Validate smallest-"
                         "first: single H2O correctness vs CPU before trusting "
                         "larger systems.")
    p.add_argument("--scf-lindep-threshold", type=float, default=1e-8,
                    help="Overlap-matrix eigenvalue threshold below which "
                         "canonical orthogonalization discards a direction "
                         "(pyscf.scf.addons.remove_linear_dep_). Pinned "
                         "explicitly (default: PySCF's own standard 1e-8) "
                         "rather than left to each backend's own default, "
                         "since different backends' eigensolvers were "
                         "observed to drop a different number of near-"
                         "linearly-dependent directions on the same overlap "
                         "matrix (pyscf: 0, gpu4pyscf: 7, same H160 system) "
                         "-- a real difference in retained orbital space, "
                         "not roundoff (2026-07-12).")
    p.add_argument("--scf-max-cycle", type=int, default=None,
                    help="SCF max_cycle, passed straight through to the mf "
                         "object (both backends). None = backend default "
                         "(50). Needed for systems with near-degenerate "
                         "frontier orbitals (e.g. H300/cc-pVTZ) where the "
                         "default oscillates instead of converging within "
                         "50 cycles (2026-07-12).")
    p.add_argument("--scf-level-shift", type=float, default=None,
                    help="SCF level_shift (Ha), passed straight through. "
                         "None = backend default (0, no shift). A positive "
                         "shift damps oscillation from near-degenerate "
                         "HOMO/LUMO by artificially raising virtual-orbital "
                         "energies during the iteration (2026-07-12).")
    p.add_argument("--scf-diis-space", type=int, default=None,
                    help="SCF diis_space, passed straight through. None = "
                         "backend default (8).")
    p.add_argument("--out", default=None, help="Write raw history JSON here.")
    p.add_argument("--plot", default=None, help="Write convergence plot PNG here.")
    p.add_argument("--save-h5", default=None,
                    help="Write the FULL optimization history (including every "
                         "step's params, via save_optimization_history) to this "
                         "HDF5 path -- the JSON --out only saves cost/energies/"
                         "stds/acceptance, not params, so this is what a "
                         "subsequent --init-params-from run needs.")
    p.add_argument("--init-params-from", default=None,
                    help="Warm-start from a prior run's saved HDF5 history "
                         "(--save-h5 output) -- uses that run's LAST step's "
                         "params instead of jastrow.init_params(), for chaining "
                         "an aggressive phase-A run into a phase-B refinement "
                         "(Felix's two-phase production recipe, 2026-07-11). "
                         "By default also resumes the LR decay schedule from "
                         "the prior run's final step (see --reset-lr-schedule).")
    p.add_argument("--reset-lr-schedule", action="store_true",
                    help="With --init-params-from, restart the LR decay "
                         "schedule at full --learning-rate instead of "
                         "resuming from the prior run's final step. Use this "
                         "when deliberately starting a new phase at a "
                         "different learning rate (e.g. phase-A -> phase-B); "
                         "leave unset when just chaining a single long "
                         "optimization across wall-clock/job boundaries, "
                         "which is the default (Felix's continuity fix, "
                         "2026-07-11).")
    args = p.parse_args()

    atom, unit, label = build_system(args)
    result_meta = {
        "commit": git_commit(),
        "system": label,
        "basis": args.basis,
        "n_walkers": args.n_walkers,
        "burn_in_steps": args.burn_in,
        "n_opt_steps": args.n_opt_steps,
        "jac_batch_size": args.jac_batch_size,
    }

    scf_max_memory = resolve_scf_max_memory(args.scf_max_memory)
    result_meta["scf_max_memory_mb"] = scf_max_memory
    mol_kwargs = dict(atom=atom, basis=args.basis, unit=unit, max_memory=scf_max_memory)
    if args.scf_log:
        mol_kwargs.update(verbose=4, output=args.scf_log)
    else:
        mol_kwargs.update(verbose=0)
    mol = gto.M(**mol_kwargs)
    result_meta["scf_backend"] = args.scf_backend
    result_meta["scf_lindep_threshold"] = args.scf_lindep_threshold
    result_meta["scf_max_cycle"] = args.scf_max_cycle
    result_meta["scf_level_shift"] = args.scf_level_shift
    result_meta["scf_diis_space"] = args.scf_diis_space
    mf, scf_time_s, scf_cached, cusolver_preload_path, n_lindep_removed = run_scf(
        mol, atom, args.basis, unit, args.scf_cache_dir, backend=args.scf_backend,
        lindep_threshold=args.scf_lindep_threshold, max_cycle=args.scf_max_cycle,
        level_shift=args.scf_level_shift, diis_space=args.scf_diis_space)
    result_meta["scf_time_s"] = scf_time_s
    result_meta["scf_cached"] = scf_cached
    result_meta["cusolver_preload_path"] = cusolver_preload_path
    result_meta["n_lindep_removed"] = n_lindep_removed
    result_meta["scf_e_tot"] = float(mf.e_tot)
    result_meta["scf_converged"] = bool(mf.converged)
    # mf.cycles defaults to 0 (class-level) even when .kernel() was never
    # called, e.g. the cached-SCF path -- None here means "not applicable
    # this run" rather than a misleading literal 0.
    result_meta["scf_n_cycles"] = None if scf_cached else getattr(mf, "cycles", None)
    result_meta["n_orb"] = int(mol.nao)
    result_meta["n_elec"] = int(mol.nelectron)

    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandyAnalytical.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])

    initial_opt_state = None
    if args.init_params_from:
        from pytc.vmc.mcmc_utils import load_optimization_history
        prior = load_optimization_history(args.init_params_from)
        # params leaves are stacked along axis 0 (the step) -- take the last.
        params = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf[-1]), prior["params"])
        result_meta["init_params_from"] = args.init_params_from
        print(f"Warm-starting from {args.init_params_from} (last of "
              f"{jax.tree_util.tree_leaves(prior['params'])[0].shape[0]} saved steps)")
        if args.reset_lr_schedule:
            print("--reset-lr-schedule set: LR schedule restarts at full "
                  "--learning-rate despite warm-started params.")
        else:
            prior_final_state = prior.get("final_opt_state")
            if prior_final_state is not None:
                initial_opt_state = int(prior_final_state)
                result_meta["initial_opt_state"] = initial_opt_state
                print(f"Resuming LR schedule from step {initial_opt_state} "
                      "(pass --reset-lr-schedule to restart at full "
                      "--learning-rate instead).")
            else:
                print("Warning: prior history has no 'final_opt_state' "
                      "(saved before this feature existed) -- LR schedule "
                      "restarts at full --learning-rate.")
    else:
        jastrow_params = jastrow.init_params()
        linear_coeffs = jnp.ones(1)
        params = [jastrow_params, linear_coeffs]

    key = random.PRNGKey(43)

    t0 = time.time()
    opt_result = optimize_ref_var(
        sj_ansatz,
        params=params,
        n_walkers=args.n_walkers,
        n_opt_steps=args.n_opt_steps,
        burn_in_steps=args.burn_in,
        step_size=args.step_size,
        max_vmap_batch_size=args.jac_batch_size,
        optimizer_type="newton",
        learning_rate=args.learning_rate,
        opt_kwargs={"damping": args.damping, "solver": "exact"},
        key=key,
        n_mcmc_per_opt=args.n_mcmc_per_opt,
        n_opt_per_mcmc=1,
        initial_opt_state=initial_opt_state,
    )
    result_meta["total_optimize_time_s"] = time.time() - t0

    if args.save_h5:
        from pytc.vmc.mcmc_utils import save_optimization_history
        save_optimization_history(opt_result, args.save_h5)
        print(f"Wrote {args.save_h5}")

    cost = np.asarray(opt_result["cost"])
    energies = np.asarray(opt_result["energies"])
    stds = np.asarray(opt_result["stds"])
    acceptance = np.asarray(opt_result["acceptance"])

    out_payload = {
        **result_meta,
        "cost": cost.tolist(),
        "energies": energies.tolist(),
        "stds": stds.tolist(),
        "acceptance": acceptance.tolist(),
    }
    out_json = json.dumps(out_payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_json)
        print(f"Wrote {args.out}")
    else:
        print(out_json)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(len(cost))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.plot(steps, cost, marker="o", markersize=3)
        ax1.set_ylabel("Reference variance (cost)")
        ax1.set_yscale("log")
        ax1.set_title(f"{label} / {args.basis}, W={args.n_walkers}, "
                       f"jac_batch={args.jac_batch_size or 'full'}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, energies, marker="o", markersize=3, color="C1")
        if len(stds) == len(energies):
            ax2.fill_between(steps, energies - stds, energies + stds, alpha=0.2, color="C1")
        ax2.set_xlabel("Optimization step")
        ax2.set_ylabel("Energy (Ha)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0, 0.02, 1, 1))
        save_figure(fig, args.plot, result_meta)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
