"""Per-phase timing instrumentation for the optimize_ref_var VMC path.

Unlike a cProfile pass over the fused, JIT-compiled `optimize_ref_var` call
(which misattributes queued-async compute to whatever host call happens to
block first -- see the retracted 89%-in-device_get finding), this harness
calls the SAME building blocks `optimize_ref_var` uses (`burn_in`,
`make_mcmc_step`, the Newton optimizer's Jacobian-build and linear-solve
sub-steps) as separate, explicitly `jax.block_until_ready()`-bounded calls,
so each phase's wall-clock number reflects only that phase's real compute.

Emits one JSON object per run (task #1 in #pro-pytc-efficiency-refactor,
Felix's measurement spec, doc attachment b35f245c).

Usage:
    python profile_vmc_ref_var_phases.py --n-atoms 20 --basis cc-pVTZ \
        --n-walkers 1000 --burn-in 500 --n-newton-steps 10 \
        --system h-chain --out h20_w1000.json
    python profile_vmc_ref_var_phases.py --n-water 4 --basis cc-pVTZ \
        --n-walkers 1000 --system water --out h2o4_w1000.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.vmc.sampling import burn_in
from pytc.vmc.metropolis import make_mcmc_step
from pytc.vmc.hamiltonian import eval_local_energy
from pytc.vmc.walker import initialize_walkers
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow, BoysHandy
from pytc.jastrow.bha import BoysHandyAnalytical


def block(x):
    """Recursively block_until_ready() a pytree; return it unchanged."""
    jax.block_until_ready(x)
    return x


def mem_stats():
    """Return (bytes_in_use, peak_bytes_in_use) from the first local device,
    or (None, None) on CPU / if unavailable."""
    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
        if not stats:
            return None, None
        return stats.get("bytes_in_use"), stats.get("peak_bytes_in_use")
    except Exception:
        return None, None


def record_mem_checkpoint(result, phase, prev_peak):
    """Sample memory after a phase boundary (call AFTER block()-ing that
    phase's output). Records live bytes-in-use for the phase, plus how much
    the CUMULATIVE peak grew during it -- peak_bytes_in_use never resets
    between phases, so a raw peak reading conflates "what this phase used"
    with "the highest-water-mark of everything before it" (see the W=5000
    Wave-2 case: an unbatched burn-in's peak outlived and masked the later,
    smaller, properly-batched Jacobian phase's own peak). Returns the new
    prev_peak for the next call.
    """
    bytes_in_use, peak = mem_stats()
    result[f"{phase}_mem_bytes_in_use"] = bytes_in_use
    if peak is not None and prev_peak is not None:
        result[f"{phase}_mem_peak_growth_bytes"] = peak - prev_peak
    else:
        result[f"{phase}_mem_peak_growth_bytes"] = None
    return peak if peak is not None else prev_peak


def h_chain(n, sep=1.8):
    return "; ".join(f"H 0 0 {i*sep}" for i in range(n))


# Bump whenever the SCF cache key's hash-input string changes (a new
# field added/removed/reordered) -- lets a future cache-entry survive
# code updates that don't actually change the key, and gives migration
# tooling something to key off. Two undocumented schema changes already
# happened before this existed (433341a added backend/lindep_threshold,
# bf8263e added solver), both silently orphaning old entries with no
# record of why -- this is meant to end that "key archaeology" pattern
# (Felix, 2026-07-12, #proj-pytc-efficiency-refactor).
SCF_CACHE_KEY_SCHEMA_VERSION = 1


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


def dump_scf_restart(mf, path):
    """Save the current density matrix for a later --scf-init-dm-from
    restart.

    Unlike the converged-only chkfile cache, this is written
    unconditionally regardless of mf.converged -- its whole purpose is
    chaining an unconverged/partial SCF into a fresh follow-up run (the
    standard shift-then-release recipe for oscillatory near-degenerate
    systems: converge shifted to a loose tolerance, then restart
    unshifted from that density, usually finishing in tens of cycles
    instead of a marathon single run). Deliberately a separate path
    from the converged-only cache, per Felix's call, 2026-07-12,
    #proj-pytc-efficiency-refactor.
    """
    import h5py
    dm = np.asarray(_to_host(mf.make_rdm1()))
    with h5py.File(path, "w") as f:
        f.create_dataset("dm", data=dm)


def load_scf_restart(path):
    """Load a density matrix saved by dump_scf_restart, for --scf-init-
    dm-from. Returns a plain host numpy array; gpu4pyscf's own kernel()
    converts host dm0 arrays to device internally (same pattern it uses
    for its own initial-guess construction)."""
    import h5py
    with h5py.File(path, "r") as f:
        return f["dm"][()]


def write_converged_marker(path, cache_params):
    """Write the SCF cache-key schema version + the full param dict that
    produced this cache entry into the ``.converged`` marker (previously
    an empty touch-file). Makes an entry self-describing -- ``cat`` the
    marker instead of reconstructing the hash input offline to find out
    what a stale/orphaned cache entry actually was (Felix, 2026-07-12,
    #proj-pytc-efficiency-refactor, after two cache-key schema changes
    in one night needed exactly that reconstruction)."""
    with open(path, "w") as f:
        json.dump({
            "cache_key_schema_version": SCF_CACHE_KEY_SCHEMA_VERSION,
            **cache_params,
        }, f, indent=2)


def make_rhf(mol, backend="pyscf", lindep_threshold=1e-8,
             max_cycle=None, level_shift=None, diis_space=None, solver="diis"):
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

    solver: "diis" (default, unchanged) or "newton" -- wraps the mf
    object with second-order (Newton/SOSCF) convergence via the
    backend's own ``mf.newton()`` (both pyscf and gpu4pyscf expose the
    identical bound method; gpu4pyscf's is cupy-native, confirmed via
    its public source). Applied last so the wrap inherits the already-
    configured max_cycle/level_shift/diis_space/lindep state. Second-
    order SCF is the standard closer for oscillatory near-degenerate
    cases -- a real alternative to the level-shift dance, not just
    another knob.

    Returns (mf, cusolver_preload_path, n_lindep_removed). The preload
    path is None for the pyscf backend, or for gpu4pyscf if no preload
    was needed/found.
    """
    if solver not in ("diis", "newton"):
        raise ValueError(f"Unknown --scf-solver: {solver!r}")
    if backend == "pyscf":
        mf = scf.RHF(mol).density_fit()
        _apply_scf_convergence_settings(mf, max_cycle, level_shift, diis_space)
        n_lindep_removed = _pin_lindep_threshold(mf, lindep_threshold, backend)
        if solver == "newton":
            mf = mf.newton()
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
        if solver == "newton":
            mf = mf.newton()
        return mf, preload_path, n_lindep_removed
    else:
        raise ValueError(f"Unknown --scf-backend: {backend!r}")


def run_scf(mol, atom, basis, unit, cache_dir, backend="pyscf", lindep_threshold=1e-8,
            max_cycle=None, level_shift=None, diis_space=None, solver="diis",
            init_dm_from=None, dump_restart_to=None):
    """Run RHF+density-fit SCF on ``mol``, caching converged
    mo_coeff/mo_energy/mo_occ to a PySCF chkfile keyed by (atom, basis,
    unit) so repeated harness runs on the same system (Wave-2 reruns after
    a batch-size/knob change, for instance) skip the SCF entirely instead
    of re-paying it -- H300-class systems have cost multiple hours of sunk
    SCF time across reruns (Felix, #proj-pytc-efficiency-refactor).

    init_dm_from/dump_restart_to: the shift-then-release restart chain,
    separate from the converged-only cache (see dump_scf_restart's
    docstring) -- init_dm_from loads a prior run's density matrix as
    the initial guess (mf.kernel(dm0=...)); dump_restart_to saves the
    current density matrix after kernel() regardless of convergence.

    Returns (mf, scf_time_s, was_cached, cusolver_preload_path,
    n_lindep_removed). Pass cache_dir=None/"" to disable caching.
    """
    mf, preload_path, n_lindep_removed = make_rhf(
        mol, backend, lindep_threshold, max_cycle, level_shift, diis_space, solver)
    dm0 = load_scf_restart(init_dm_from) if init_dm_from else None

    if not cache_dir:
        t0 = time.time()
        mf.kernel(dm0=dm0)
        print(f"SCF: converged={mf.converged}  e_tot={mf.e_tot}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
        if dump_restart_to:
            dump_scf_restart(mf, dump_restart_to)
        return mf, time.time() - t0, False, preload_path, n_lindep_removed

    os.makedirs(cache_dir, exist_ok=True)
    # max_cycle/level_shift/diis_space/solver affect the optimization
    # trajectory, not just speed -- for near-degenerate systems a
    # different path can converge to a different local solution, so
    # they're part of the cache key too (same principle as backend/
    # lindep_threshold above).
    cache_key = hashlib.sha256(
        f"{atom}|{basis}|{unit}|{backend}|{lindep_threshold}|"
        f"{max_cycle}|{level_shift}|{diis_space}|{solver}".encode()
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
    mf.kernel(dm0=dm0)
    print(f"SCF: converged={mf.converged}  e_tot={mf.e_tot}  "
          f"elapsed={time.time() - t0:.1f}s", flush=True)
    if dump_restart_to:
        dump_scf_restart(mf, dump_restart_to)
    if mf.converged:
        scf.chkfile.dump_scf(
            mf.mol, cache_path,
            _to_host(mf.e_tot), _to_host(mf.mo_energy),
            _to_host(mf.mo_coeff), _to_host(mf.mo_occ),
        )
        write_converged_marker(converged_marker, {
            "atom": atom, "basis": basis, "unit": unit, "backend": backend,
            "lindep_threshold": lindep_threshold, "max_cycle": max_cycle,
            "level_shift": level_shift, "diis_space": diis_space,
            "solver": solver, "e_tot": float(_to_host(mf.e_tot)),
        })
    return mf, time.time() - t0, False, preload_path, n_lindep_removed


def build_system(args):
    if args.system == "h-chain":
        atom = h_chain(args.n_atoms)
        unit = "Bohr"
        label = f"H{args.n_atoms}"
    elif args.system == "water":
        from pytc.utils.gen_water_cluster import build_water_cluster
        atom = build_water_cluster(args.n_water)
        unit = "Angstrom"
        label = f"(H2O){args.n_water}"
    else:
        raise ValueError(args.system)
    return atom, unit, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["h-chain", "water"], required=True)
    p.add_argument("--n-atoms", type=int, default=20, help="H-chain: n H atoms")
    p.add_argument("--n-water", type=int, default=2, help="water: n H2O monomers")
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=1000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--n-newton-steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--damping", type=float, default=1e-6)
    p.add_argument("--jac-batch-size", type=int, default=0,
                    help="If >0, use the batched scan-accumulator Jacobian path "
                         "(optimizer.py:187-297, NewtonOptimizer's max_vmap_batch_size "
                         "branch) instead of a full-batch vmap. Needed for W/N combos "
                         "that OOM unbatched (task #4).")
    p.add_argument("--vmap-batch-size", type=int, default=256,
                    help="max_vmap_batch_size forwarded to burn_in and "
                         "make_mcmc_step (routes through folx.batched_vmap "
                         "instead of a plain full-batch vmap for the AO/"
                         "determinant eval in _warmup_ansatz and each MCMC "
                         "step). Unbatched (0, the underlying library "
                         "default) OOMs or hits an XLA autotune-reject at "
                         "large N or W (Wave-2 finding, task #6) -- this "
                         "harness defaults to 256 instead so new waves don't "
                         "hit the same wall; pass 0 explicitly to reproduce "
                         "the unbatched failure. Same knob optimize_ref_var "
                         "itself exposes; not wired into this harness until "
                         "now.")
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"),
                    help="Directory for cached SCF results (PySCF chkfiles), keyed "
                         "by (atom, basis, unit). Repeated runs on the same system "
                         "(e.g. rerunning a Wave after a batch-size change) load the "
                         "converged mo_coeff/mo_energy/mo_occ instead of re-running "
                         "SCF from scratch -- large basis systems have cost multiple "
                         "hours of sunk SCF time across reruns. Pass an empty string "
                         "to disable caching.")
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
    p.add_argument("--scf-solver", choices=["diis", "newton"], default="diis",
                    help="SCF convergence algorithm. 'diis' (default) is "
                         "the standard first-order DIIS-accelerated SCF. "
                         "'newton' wraps the mf object with second-order "
                         "(Newton/SOSCF) convergence via mf.newton() -- "
                         "the standard closer for oscillatory near-"
                         "degenerate cases (e.g. H300/cc-pVTZ), a real "
                         "alternative to level-shifting, not just another "
                         "knob. Works identically on both backends "
                         "(2026-07-12).")
    p.add_argument("--scf-init-dm-from", default=None,
                    help="Warm-start SCF from a prior run's saved density "
                         "matrix (--scf-dump-restart output). For the "
                         "shift-then-release recipe: converge shifted to a "
                         "loose tolerance, save via --scf-dump-restart, "
                         "then restart unshifted from that density -- "
                         "usually finishes in tens of cycles instead of a "
                         "marathon single run (2026-07-12). Separate "
                         "mechanism from the converged-only SCF cache.")
    p.add_argument("--scf-dump-restart", default=None,
                    help="Save the density matrix after SCF (regardless "
                         "of convergence) to this path, for a later "
                         "--scf-init-dm-from restart. Separate from the "
                         "converged-only SCF cache -- this is meant for "
                         "chaining an unconverged/partial SCF into a "
                         "follow-up run (2026-07-12).")
    p.add_argument("--jastrow-class", choices=["bh", "bha"], default="bha",
                    help="bh = original BoysHandy per-pair vmap path (pre-R1-"
                         "refactor behavior); bha = BoysHandyAnalytical, the "
                         "R1 fast contracted-tensor override (default, current "
                         "production behavior). Dispatch is pure class choice "
                         "since commit 12405eb removed the jastrow_terms_impl "
                         "flag -- this CLI flag just selects which class the "
                         "harness constructs, matching nan_stage_trace.py's "
                         "existing convention. Does not affect the SCF cache "
                         "key (implementation choice, not SCF physics) "
                         "(2026-07-12).")
    p.add_argument("--out", default=None, help="Write JSON here (default: stdout)")
    args = p.parse_args()

    atom, unit, label = build_system(args)
    result = {
        "commit": git_commit(),
        "system": label,
        "basis": args.basis,
        "n_walkers": args.n_walkers,
        "burn_in_steps": args.burn_in,
        "n_newton_steps": args.n_newton_steps,
        "vmap_batch_size": args.vmap_batch_size,
    }

    scf_max_memory = resolve_scf_max_memory(args.scf_max_memory)
    result["scf_max_memory_mb"] = scf_max_memory
    mol_kwargs = dict(atom=atom, basis=args.basis, unit=unit, max_memory=scf_max_memory)
    if args.scf_log:
        mol_kwargs.update(verbose=4, output=args.scf_log)
    else:
        mol_kwargs.update(verbose=0)
    mol = gto.M(**mol_kwargs)
    result["scf_backend"] = args.scf_backend
    result["scf_lindep_threshold"] = args.scf_lindep_threshold
    result["scf_max_cycle"] = args.scf_max_cycle
    result["scf_level_shift"] = args.scf_level_shift
    result["scf_diis_space"] = args.scf_diis_space
    result["scf_solver"] = args.scf_solver
    result["scf_init_dm_from"] = args.scf_init_dm_from
    result["scf_dump_restart"] = args.scf_dump_restart
    mf, scf_time_s, scf_cached, cusolver_preload_path, n_lindep_removed = run_scf(
        mol, atom, args.basis, unit, args.scf_cache_dir, backend=args.scf_backend,
        lindep_threshold=args.scf_lindep_threshold, max_cycle=args.scf_max_cycle,
        level_shift=args.scf_level_shift, diis_space=args.scf_diis_space,
        solver=args.scf_solver, init_dm_from=args.scf_init_dm_from,
        dump_restart_to=args.scf_dump_restart)
    result["scf_time_s"] = scf_time_s
    result["scf_cached"] = scf_cached
    result["cusolver_preload_path"] = cusolver_preload_path
    result["n_lindep_removed"] = n_lindep_removed
    result["scf_e_tot"] = float(mf.e_tot)
    result["scf_converged"] = bool(mf.converged)
    # mf.cycles defaults to 0 (class-level) even when .kernel() was never
    # called, e.g. the cached-SCF path -- None here means "not applicable
    # this run" rather than a misleading literal 0.
    result["scf_n_cycles"] = None if scf_cached else getattr(mf, "cycles", None)
    result["n_orb"] = int(mol.nao)
    result["n_elec"] = int(mol.nelectron)

    det = SlaterDet.create(mol, mf.mo_coeff)
    # Class choice is the only dispatch (task #5 PR-B; BoysHandy.create()
    # doesn't implicitly route to BoysHandyAnalytical, Ke's direction,
    # 2026-07-10). --jastrow-class selects which class this harness
    # constructs, matching nan_stage_trace.py's existing convention.
    result["jastrow_class"] = args.jastrow_class
    bh = (BoysHandy if args.jastrow_class == "bh" else BoysHandyAnalytical).create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    # Provenance: which Jastrow classes actually ran, for anyone reading
    # the JSON later without the script in front of them.
    result["jastrow_component_classes"] = [type(j).__name__ for j in jastrow.jastrows]

    key = random.PRNGKey(43)
    key, subkey = random.split(key)

    prev_peak = mem_stats()[1] or 0  # baseline before any VMC-specific allocation

    # --- Walker init ---
    t0 = time.time()
    walkers = block(initialize_walkers(sj_ansatz, args.n_walkers, key=subkey))
    result["walker_init_time_s"] = time.time() - t0
    prev_peak = record_mem_checkpoint(result, "walker_init", prev_peak)

    # --- Burn-in ---
    # NOTE: calling burn_in() twice in a row on already-warmed-up walkers hits
    # a vmap shape bug in _warmup_ansatz (sampling.py) unrelated to this
    # profiling harness -- filed separately, not blocking this PR. Reporting
    # one number (total, including one-time JIT compile) rather than a
    # compile/steady-state split until that's fixed.
    key, subkey = random.split(key)
    t0 = time.time()
    walkers, _acc_hist, key, adapted_step_size = burn_in(
        det, walkers, n_steps=args.burn_in, step_size=args.step_size,
        key=subkey, params=params, report_interval=10 ** 9,
        max_vmap_batch_size=args.vmap_batch_size,
    )
    walkers = block(walkers)
    result["burn_in_total_time_s"] = time.time() - t0
    result["burn_in_time_per_step_s"] = result["burn_in_total_time_s"] / args.burn_in
    prev_peak = record_mem_checkpoint(result, "burn_in", prev_peak)

    # --- One MCMC step (compile + steady-state, averaged over 20 steps) ---
    # Use the burn-in-adapted step_size, matching production (optimization.py:747-748),
    # so acceptance_rate here is comparable to a real run.
    mcmc_step = make_mcmc_step(det, adapted_step_size, move_type="one",
                                max_vmap_batch_size=args.vmap_batch_size)
    key, subkey = random.split(key)
    t0 = time.time()
    walkers, acc = mcmc_step(det, walkers, subkey, params)
    walkers, acc = block(walkers), block(acc)
    result["mcmc_step_compile_time_s"] = time.time() - t0

    n_mcmc_avg = 20
    accs = []
    t0 = time.time()
    for _ in range(n_mcmc_avg):
        key, subkey = random.split(key)
        walkers, acc = mcmc_step(det, walkers, subkey, params)
        walkers, acc = block(walkers), block(acc)
        accs.append(float(acc))
    result["mcmc_step_steady_time_s"] = (time.time() - t0) / n_mcmc_avg
    result["acceptance_rate"] = sum(accs) / len(accs)
    prev_peak = record_mem_checkpoint(result, "mcmc_step", prev_peak)

    # --- E_L + Jacobian build (Gauss-Newton path, matches optimizer.py) ---
    def single_local_energy_and_grad(w, p):
        return jax.value_and_grad(
            lambda pp: eval_local_energy(sj_ansatz, w, pp)[0]
        )(p)

    result["jac_batch_size"] = args.jac_batch_size

    if args.jac_batch_size > 0:
        # Batched scan-accumulator path, matching optimizer.py:187-297
        # (NewtonOptimizer's max_vmap_batch_size>0 branch): accumulates
        # sum_e/sum_e2/sum_j/sum_jte/sum_jtj per batch instead of
        # materializing the full (W,P) Jacobian + reverse-mode residuals
        # for all walkers at once -- the latter is what OOMs past ~40
        # electrons at W=1000 on a single A100 (task #4, Wave-1 diagnosis).
        batch_size = min(args.jac_batch_size, args.n_walkers)
        n_batches = (args.n_walkers + batch_size - 1) // batch_size
        padded_n = n_batches * batch_size
        pad_count = padded_n - args.n_walkers
        result["jac_n_batches"] = n_batches

        params_vec, _ = jax.flatten_util.ravel_pytree(params)
        param_dtype = params_vec.dtype

        def pad_walkers(w):
            if pad_count == 0:
                return w, jnp.ones((padded_n,), dtype=bool)
            padded = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(
                    [x, jnp.repeat(x[:1], pad_count, axis=0)], axis=0
                ),
                w,
            )
            mask = jnp.concatenate(
                [jnp.ones((args.n_walkers,), dtype=bool),
                 jnp.zeros((pad_count,), dtype=bool)]
            )
            return padded, mask

        def flatten_jacobian(jac, n):
            jac_flat, _ = jax.tree_util.tree_flatten(jac)
            return jnp.concatenate(
                [jnp.reshape(leaf, (n, -1)) for leaf in jac_flat], axis=1
            )

        def masked_energy_jacobian_batch(batch_walkers, batch_mask):
            energies_batch, jac_batch = jax.vmap(
                single_local_energy_and_grad, in_axes=(0, None)
            )(batch_walkers, params)
            energies_batch = jnp.where(batch_mask, energies_batch, 0.0)
            jac_mat_batch = flatten_jacobian(jac_batch, batch_size)
            jac_mat_batch = jnp.where(batch_mask[:, None], jac_mat_batch, 0.0)
            return energies_batch, jac_mat_batch

        def stats_scan_body(carry, xs):
            sum_e, sum_e2, sum_j, sum_jte, sum_jtj = carry
            batch_walkers, batch_mask = xs
            energies_batch, jac_mat_batch = masked_energy_jacobian_batch(
                batch_walkers, batch_mask
            )
            sum_e = sum_e + jnp.sum(energies_batch)
            sum_e2 = sum_e2 + jnp.sum(energies_batch ** 2)
            sum_j = sum_j + jnp.sum(jac_mat_batch, axis=0)
            sum_jte = sum_jte + jac_mat_batch.T @ energies_batch
            sum_jtj = sum_jtj + jac_mat_batch.T @ jac_mat_batch
            return (sum_e, sum_e2, sum_j, sum_jte, sum_jtj), None

        @jax.jit
        def batched_jacobian_pass(walkers):
            padded_walkers, mask = pad_walkers(walkers)
            batched_walkers = jax.tree_util.tree_map(
                lambda x: x.reshape((n_batches, batch_size) + x.shape[1:]),
                padded_walkers,
            )
            batched_mask = mask.reshape((n_batches, batch_size))
            init_carry = (
                jnp.array(0.0, dtype=param_dtype),
                jnp.array(0.0, dtype=param_dtype),
                jnp.zeros_like(params_vec),
                jnp.zeros_like(params_vec),
                jnp.zeros((params_vec.shape[0], params_vec.shape[0]), dtype=param_dtype),
            )
            return jax.lax.scan(
                stats_scan_body, init_carry, (batched_walkers, batched_mask)
            )[0]

        t0 = time.time()
        sum_e, sum_e2, sum_j, sum_jte, sum_jtj = block(batched_jacobian_pass(walkers))
        result["jacobian_build_compile_time_s"] = time.time() - t0

        t0 = time.time()
        sum_e, sum_e2, sum_j, sum_jte, sum_jtj = block(batched_jacobian_pass(walkers))
        result["jacobian_build_steady_time_s"] = time.time() - t0

        # Same normalization as optimizer.py:292-297 (2/(n-1) for the
        # gradient, 2/n for the curvature matrix -- not a typo, matches
        # production exactly for this path).
        n_w = args.n_walkers
        e_mean = sum_e / n_w
        mean_j = sum_j / n_w
        grads_vec = (2.0 / (n_w - 1)) * (sum_jte - n_w * mean_j * e_mean)
        curvature_mat = (2.0 / n_w) * (sum_jtj - n_w * jnp.outer(mean_j, mean_j))
    else:
        # Unbatched path: jitted as a whole -- production runs this inside
        # the jitted training_step (optimization.py:776); an eager vmap
        # would overstate this phase's cost with per-op dispatch overhead,
        # which is precisely the number R1 hinges on.
        el_grad_jit = jax.jit(jax.vmap(single_local_energy_and_grad, in_axes=(0, None)))

        t0 = time.time()
        (energies, grads) = el_grad_jit(walkers, params)
        energies, grads = block(energies), block(grads)
        result["jacobian_build_compile_time_s"] = time.time() - t0

        t0 = time.time()
        (energies, grads) = el_grad_jit(walkers, params)
        energies, grads = block(energies), block(grads)
        result["jacobian_build_steady_time_s"] = time.time() - t0

        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        jac_mat = jnp.concatenate(
            [jnp.reshape(leaf, (args.n_walkers, -1)) for leaf in grads_flat], axis=1
        )
        jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
        e_centered = energies - jnp.mean(energies)
        n_w = jac_mat.shape[0]
        curvature_mat = (2.0 / (n_w - 1)) * (jac_centered.T @ jac_centered)
        grads_vec = (2.0 / (n_w - 1)) * (jac_centered.T @ e_centered)

    # Peak device memory right after the Jacobian phase (GPU only), before
    # the (much smaller) Newton solve allocates anything more. Field name
    # kept for backward compat with existing analysis; still a cumulative
    # high-water-mark, not phase-isolated -- use jacobian_build_mem_bytes_in_use
    # / jacobian_build_mem_peak_growth_bytes below for that.
    result["jacobian_build_peak_mem_bytes"] = mem_stats()[1]
    prev_peak = record_mem_checkpoint(result, "jacobian_build", prev_peak)

    # --- Newton solve (linear solve only; curvature/grad already assembled above) ---
    def solve_delta(curvature_mat, grads_vec, damping):
        p = curvature_mat.shape[0]
        damped = curvature_mat + damping * jnp.eye(p)
        return jnp.linalg.solve(damped, -grads_vec)

    solve_delta_jit = jax.jit(solve_delta)
    t0 = time.time()
    delta = block(solve_delta_jit(curvature_mat, grads_vec, args.damping))
    result["newton_solve_compile_time_s"] = time.time() - t0

    t0 = time.time()
    delta = block(solve_delta_jit(curvature_mat, grads_vec, args.damping))
    result["newton_solve_steady_time_s"] = time.time() - t0
    record_mem_checkpoint(result, "newton_solve", prev_peak)

    # --- Peak device memory (GPU only; empty dict on CPU) ---
    dev = jax.local_devices()[0]
    result["peak_device_memory_bytes"] = mem_stats()[1]
    result["device_kind"] = dev.device_kind

    out = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"Wrote {args.out}")
    print(out)


if __name__ == "__main__":
    main()
