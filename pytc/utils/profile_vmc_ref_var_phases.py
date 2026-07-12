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
from jax import random
from pyscf import gto, scf

from pytc.vmc.sampling import burn_in
from pytc.vmc.metropolis import make_mcmc_step
from pytc.vmc.hamiltonian import eval_local_energy
from pytc.vmc.walker import initialize_walkers
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
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


def make_rhf(mol, backend="pyscf"):
    """Construct the density-fitted RHF object for the requested SCF backend.

    ``pyscf`` (default) is the existing CPU DF-RHF path, byte-for-byte
    unchanged. ``gpu4pyscf`` swaps in gpu4pyscf's GPU-accelerated RHF --
    Felix's spike (2026-07-11) to test whether GPU DF-SCF is the real fix
    for H300's bottleneck, since CPU max_memory sizing alone only bought
    1.39x, far short of the hoped-for 10-50x. Import is lazy so choosing
    "pyscf" never requires gpu4pyscf (a CUDA-only package) to be installed.

    Returns (mf, cusolver_preload_path) -- the latter is None for the
    pyscf backend, or for gpu4pyscf if no preload was needed/found.
    """
    if backend == "pyscf":
        return scf.RHF(mol).density_fit(), None
    elif backend == "gpu4pyscf":
        preload_path = _preload_gpu4pyscf_cusolver()
        try:
            from gpu4pyscf import scf as gpu_scf
        except ImportError as e:
            raise ImportError(
                "--scf-backend gpu4pyscf requires the gpu4pyscf package "
                "(GPU-only, needs CUDA) -- not installed in this environment."
            ) from e
        return gpu_scf.RHF(mol).density_fit(), preload_path
    else:
        raise ValueError(f"Unknown --scf-backend: {backend!r}")


def run_scf(mol, atom, basis, unit, cache_dir, backend="pyscf"):
    """Run RHF+density-fit SCF on ``mol``, caching converged
    mo_coeff/mo_energy/mo_occ to a PySCF chkfile keyed by (atom, basis,
    unit) so repeated harness runs on the same system (Wave-2 reruns after
    a batch-size/knob change, for instance) skip the SCF entirely instead
    of re-paying it -- H300-class systems have cost multiple hours of sunk
    SCF time across reruns (Felix, #proj-pytc-efficiency-refactor).

    Returns (mf, scf_time_s, was_cached, cusolver_preload_path). Pass
    cache_dir=None/"" to disable caching.
    """
    mf, preload_path = make_rhf(mol, backend)

    if not cache_dir:
        t0 = time.time()
        mf.kernel()
        print(f"SCF: converged={mf.converged}  e_tot={mf.e_tot}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
        return mf, time.time() - t0, False, preload_path

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.sha256(f"{atom}|{basis}|{unit}|{backend}".encode()).hexdigest()[:16]
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
        return mf, time.time() - t0, True, preload_path

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
    return mf, time.time() - t0, False, preload_path


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
    mf, scf_time_s, scf_cached, cusolver_preload_path = run_scf(
        mol, atom, args.basis, unit, args.scf_cache_dir, backend=args.scf_backend)
    result["scf_time_s"] = scf_time_s
    result["scf_cached"] = scf_cached
    result["cusolver_preload_path"] = cusolver_preload_path
    result["scf_e_tot"] = float(mf.e_tot)
    result["scf_converged"] = bool(mf.converged)
    # mf.cycles defaults to 0 (class-level) even when .kernel() was never
    # called, e.g. the cached-SCF path -- None here means "not applicable
    # this run" rather than a misleading literal 0.
    result["scf_n_cycles"] = None if scf_cached else getattr(mf, "cycles", None)
    result["n_orb"] = int(mol.nao)
    result["n_elec"] = int(mol.nelectron)

    det = SlaterDet.create(mol, mf.mo_coeff)
    # Explicit construction, not BoysHandy.create(mol): BoysHandy.create()
    # no longer implicitly routes to BoysHandyAnalytical (Ke's direction,
    # 2026-07-10 -- explicit choice over silent substitution). This harness
    # wants the analytic path (BoysHandyAnalytical's get_pair_grid_grad_lap
    # override), so it opts in directly -- there is no flag to select it,
    # class choice is the only dispatch (task #5 PR-B).
    bh = BoysHandyAnalytical.create(mol)
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
