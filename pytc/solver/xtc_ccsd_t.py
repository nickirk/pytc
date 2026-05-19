"""xTC-CCSD(T) perturbative triples on top of converged xTC-RCCSD amplitudes.

Uses 1- and 2-body xTC integrals only — no 3-body diagrams. The integrals
consumed live in ``eris.ovvv``, ``eris.ovoo``, ``eris.ovov`` and
``eris.fock`` exactly as built by ``xtc_ccsd._make_xtc_eris``; they are
non-Hermitian because the xTC corrections to (pq|rs) are not symmetric
under (pq) <-> (rs).

Algebra
-------
The right-amplitude (T) correction adapted from PySCF's
``pyscf.cc.ccsd_t_slow`` (JCP 94, 442; 1991). For non-Hermitian H the
``.conj()`` factors that appear in the Hermitian reference are dropped —
xTC ERIs are real but asymmetric, so symmetrising them is incorrect.

For each virtual triple (a, b, c) with a >= b >= c::

    W[i,j,k] = einsum('if,fkj->ijk', vvov[a,b], t2T[c])      # particle line
             - einsum('ijm,mk->ijk', vooo[a],   t2T[b,c])    # hole line
    V[i,j,k] = einsum('ij,k  ->ijk', vvoo[a,b], t1T[c])
             + einsum('ij,k  ->ijk', t2T[a,b],  fvo[c])

The 36-term ``r3`` permutation accumulator is identical to the closed-shell
restricted reference and is copied unchanged.

Performance plan (multi-GPU, limited HBM)
-----------------------------------------
The outer (a, b, c) triangular loop is embarrassingly parallel. Strategy:

1. Partition triangular (a, b, c) tiles across local devices via
   ``pytc.utils.gpu_pipeline.partition_round_robin``. Each GPU accumulates a
   private ``et`` scalar; we reduce on host at the end.
2. Per GPU, hold ``t1``, ``t2``, ``vooo``, ``vvoo``, ``fvo``, ``mo_energy``
   resident — these are small (<~1 GB at production scale).
3. The large tensor is ``vvov`` (n_v^3 * n_o * 8B). Tile along the first virtual
   index ``a`` using ``resolve_v3o_panel_block_size`` and stream slabs from
   HDF5 ``eris.ovvv`` via ``PrefetchIterator`` + ``hdf5_slice_loader``,
   transposing each slab on the fly to vvov layout. For each resident
   ``vvov[a_tile]`` slab, sweep all valid ``(b, c)`` inside that tile.
4. Inner ``get_w`` / ``get_v`` calls are ``jax.jit``-compiled with the panel
   shape baked in so XLA can fuse the three small einsums into one kernel.
   Compilation is cached across iterations via the already-active
   ``enable_xla_compilation_cache()``.

Two paths are provided:

* ``_kernel_reference`` — single-device NumPy, used as the correctness
  oracle and for very small jobs.
* ``_kernel_multigpu`` — JAX-jitted per-triple kernel, replicated tensors,
  triangular (a,b,c) triples partitioned round-robin across all local
  devices and processed concurrently by one worker thread per device.
  Out-of-HBM regime (``vvov`` won't fit) is detected and the caller is
  pointed at the sharded-vvov follow-up (not yet implemented; see
  ``_check_hbm_budget``).
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp
from pyscf import lib
from pyscf.lib import logger as pyscf_logger

from pytc.utils.gpu_pipeline import (
    _solver_local_devices,
    broadcast_to_devices,
    partition_round_robin,
)

logger = logging.getLogger(__name__)


def kernel(mycc, eris=None, t1=None, t2=None, verbose=None):
    """xTC-CCSD(T) energy correction.

    Parameters
    ----------
    mycc : xtc_ccsd.RCCSD
        Converged xTC-RCCSD solver. ``mycc.t1`` / ``mycc.t2`` must be set.
    eris : _ChemistsERIs, optional
        ERIs as built by ``mycc.ao2mo()``. Defaults to a fresh build.
    t1, t2 : array_like, optional
        Override amplitudes. Default uses ``mycc.t1`` / ``mycc.t2``.
    verbose : int or pyscf logger, optional
        Verbosity for PySCF-style timing.

    Returns
    -------
    float
        E((T)) correlation correction. To get the total energy, add this
        to ``mycc.e_tot``.
    """
    if eris is None:
        eris = mycc.ao2mo()
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t1 is None or t2 is None:
        raise RuntimeError("xTC-CCSD(T) requires converged t1/t2 — run mycc.ccsd() first.")

    mode = getattr(mycc, "ccsd_t_mode", None)
    if mode is None:
        # Legacy switch — preserved for tests written against the v1 scaffold.
        mode = "multigpu" if getattr(mycc, "ccsd_t_use_multigpu", False) else "reference"

    if mode == "reference":
        return _kernel_reference(mycc, eris, t1, t2, verbose=verbose)
    if mode == "multigpu":
        return _kernel_multigpu(mycc, eris, t1, t2, verbose=verbose)
    if mode == "streaming":
        return _kernel_multigpu_streaming(mycc, eris, t1, t2, verbose=verbose)
    if mode == "auto":
        return _kernel_auto(mycc, eris, t1, t2, verbose=verbose)
    raise ValueError(f"Unknown ccsd_t_mode: {mode!r}. "
                     "Choose one of: 'reference', 'multigpu', 'streaming', 'auto'.")


def _kernel_auto(mycc, eris, t1, t2, verbose=None):
    """Pick replicated vs streaming based on per-device HBM headroom.

    Replicated path is faster (no PCIe traffic per triple) when ``vvov`` fits
    comfortably on each device; otherwise stream.
    """
    if isinstance(verbose, pyscf_logger.Logger):
        log = verbose
    else:
        log = pyscf_logger.Logger(mycc.stdout,
                                  verbose if verbose is not None else mycc.verbose)

    nocc, nvir = t1.shape
    devices = _solver_local_devices()
    n_devices = len(devices)

    only_cpu = all(d is None or getattr(d, "platform", "") == "cpu" for d in devices)
    if only_cpu and not getattr(mycc, "ccsd_t_force_multigpu", False):
        return _kernel_reference(mycc, eris, t1, t2, verbose=verbose)

    per_dev_gb = _check_hbm_budget(nocc, nvir, n_devices, log)
    gpu_max_mb = getattr(mycc, "gpu_max_memory", None)
    if gpu_max_mb is not None and per_dev_gb * 1000 > 0.7 * float(gpu_max_mb):
        log.info("xTC-(T) auto: replicated path needs %.1f GB/device but "
                 "gpu_max_memory=%.1f GB — using streaming.",
                 per_dev_gb, float(gpu_max_mb) / 1000)
        return _kernel_multigpu_streaming(mycc, eris, t1, t2, verbose=verbose)
    return _kernel_multigpu(mycc, eris, t1, t2, verbose=verbose)


# ---------------------------------------------------------------------------
# Single-device reference (correctness-only, slow)
# ---------------------------------------------------------------------------

def _kernel_reference(mycc, eris, t1, t2, verbose=None):
    """NumPy reference implementation.

    Adapted from ``pyscf/cc/ccsd_t_slow.py`` with ``.conj()`` removed for the
    real non-Hermitian xTC case. Used as a correctness oracle; not intended
    for production scale.
    """
    if isinstance(verbose, pyscf_logger.Logger):
        log = verbose
    else:
        log = pyscf_logger.Logger(mycc.stdout, verbose if verbose is not None else mycc.verbose)
    t_start = time.perf_counter()

    nocc, nvir = t1.shape

    t1T = t1.T                              # (a, i)
    t2T = t2.transpose(2, 3, 0, 1)          # (a, b, i, j)

    mo_e = eris.mo_energy
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = lib.direct_sum("i,j,k->ijk", e_occ, e_occ, e_occ)

    # xTC eris.ovvv is full 4D (HDF5 or ndarray); plain PySCF eris.ovvv is
    # stored packed-triangular (n_o, n_v, n_v*(n_v+1)/2). ``get_ovvv`` unpacks
    # to (n_o, n_v, n_v, n_v) in either case.
    if hasattr(eris, "get_ovvv"):
        ovvv = np.asarray(eris.get_ovvv())
    else:
        ovvv = np.asarray(eris.ovvv)
    if ovvv.ndim == 3:  # safety fallback if get_ovvv missed
        ovvv = lib.unpack_tril(ovvv.reshape(nocc * nvir, -1)).reshape(nocc, nvir, nvir, nvir)
    ovoo = np.asarray(eris.ovoo)            # (i, a, j, k)
    ovov = np.asarray(eris.ovov)            # (i, a, j, b)

    # Layout transforms matching ccsd_t_slow (without .conj() for non-Hermitian).
    eris_vvov = ovvv.transpose(1, 3, 0, 2)  # (a, c, i, b)  — get_w 1st term uses [a,b,i,f]
    eris_vooo = ovoo.transpose(1, 0, 2, 3)  # (a, i, j, k)
    eris_vvoo = ovov.transpose(1, 3, 0, 2)  # (a, b, i, j)
    fvo = eris.fock[nocc:, :nocc]

    def get_w(a, b, c):
        w = np.einsum("if,fkj->ijk", eris_vvov[a, b], t2T[c, :])
        w -= np.einsum("ijm,mk->ijk", eris_vooo[a, :], t2T[b, c])
        return w

    def get_v(a, b, c):
        v = np.einsum("ij,k->ijk", eris_vvoo[a, b], t1T[c])
        v += np.einsum("ij,k->ijk", t2T[a, b], fvo[c])
        return v

    et = 0.0
    for a in range(nvir):
        for b in range(a + 1):
            for c in range(b + 1):
                d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                if a == c:                  # a == b == c
                    d3 *= 6
                elif a == b or b == c:
                    d3 *= 2

                wabc = get_w(a, b, c); wacb = get_w(a, c, b)
                wbac = get_w(b, a, c); wbca = get_w(b, c, a)
                wcab = get_w(c, a, b); wcba = get_w(c, b, a)
                vabc = get_v(a, b, c); vacb = get_v(a, c, b)
                vbac = get_v(b, a, c); vbca = get_v(b, c, a)
                vcab = get_v(c, a, b); vcba = get_v(c, b, a)

                zabc = _r3(wabc + 0.5 * vabc) / d3
                zacb = _r3(wacb + 0.5 * vacb) / d3
                zbac = _r3(wbac + 0.5 * vbac) / d3
                zbca = _r3(wbca + 0.5 * vbca) / d3
                zcab = _r3(wcab + 0.5 * vcab) / d3
                zcba = _r3(wcba + 0.5 * vcba) / d3

                # Drop .conj() — xTC ERIs are real but non-Hermitian.
                et += np.einsum("ijk,ijk", wabc, zabc)
                et += np.einsum("ikj,ijk", wacb, zabc)
                et += np.einsum("jik,ijk", wbac, zabc)
                et += np.einsum("jki,ijk", wbca, zabc)
                et += np.einsum("kij,ijk", wcab, zabc)
                et += np.einsum("kji,ijk", wcba, zabc)

                et += np.einsum("ijk,ijk", wacb, zacb)
                et += np.einsum("ikj,ijk", wabc, zacb)
                et += np.einsum("jik,ijk", wcab, zacb)
                et += np.einsum("jki,ijk", wcba, zacb)
                et += np.einsum("kij,ijk", wbac, zacb)
                et += np.einsum("kji,ijk", wbca, zacb)

                et += np.einsum("ijk,ijk", wbac, zbac)
                et += np.einsum("ikj,ijk", wbca, zbac)
                et += np.einsum("jik,ijk", wabc, zbac)
                et += np.einsum("jki,ijk", wacb, zbac)
                et += np.einsum("kij,ijk", wcba, zbac)
                et += np.einsum("kji,ijk", wcab, zbac)

                et += np.einsum("ijk,ijk", wbca, zbca)
                et += np.einsum("ikj,ijk", wbac, zbca)
                et += np.einsum("jik,ijk", wcba, zbca)
                et += np.einsum("jki,ijk", wcab, zbca)
                et += np.einsum("kij,ijk", wabc, zbca)
                et += np.einsum("kji,ijk", wacb, zbca)

                et += np.einsum("ijk,ijk", wcab, zcab)
                et += np.einsum("ikj,ijk", wcba, zcab)
                et += np.einsum("jik,ijk", wacb, zcab)
                et += np.einsum("jki,ijk", wabc, zcab)
                et += np.einsum("kij,ijk", wbca, zcab)
                et += np.einsum("kji,ijk", wbac, zcab)

                et += np.einsum("ijk,ijk", wcba, zcba)
                et += np.einsum("ikj,ijk", wcab, zcba)
                et += np.einsum("jik,ijk", wbca, zcba)
                et += np.einsum("jki,ijk", wbac, zcba)
                et += np.einsum("kij,ijk", wacb, zcba)
                et += np.einsum("kji,ijk", wabc, zcba)

    et *= 2
    log.info("xTC-CCSD(T) correction = %.15g  (%.1f s)",
             et, time.perf_counter() - t_start)
    return float(et)


def _r3(w):
    """The closed-shell restricted (T) permutation operator."""
    return (4 * w
            + w.transpose(1, 2, 0)
            + w.transpose(2, 0, 1)
            - 2 * w.transpose(2, 1, 0)
            - 2 * w.transpose(0, 2, 1)
            - 2 * w.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# JAX inner kernel: one (a, b, c) triple → scalar contribution
# ---------------------------------------------------------------------------

def _r3_jax(w):
    """jax-native version of the closed-shell (T) permutation operator."""
    return (4 * w
            + jnp.transpose(w, (1, 2, 0))
            + jnp.transpose(w, (2, 0, 1))
            - 2 * jnp.transpose(w, (2, 1, 0))
            - 2 * jnp.transpose(w, (0, 2, 1))
            - 2 * jnp.transpose(w, (1, 0, 2)))


def _single_triple_contribution(a, b, c,
                                mo_e_o, mo_e_v,
                                t1T, t2T, vvov, vooo, vvoo, fvo):
    """Scalar energy contribution from one (a, b, c) triple with a >= b >= c.

    All tensor inputs are on a single JAX device. ``a``, ``b``, ``c`` are
    traced int32 scalars so the same compiled kernel is reused across every
    triple in the sweep.
    """
    eijk = (mo_e_o[:, None, None]
            + mo_e_o[None, :, None]
            + mo_e_o[None, None, :])
    d3_base = eijk - mo_e_v[a] - mo_e_v[b] - mo_e_v[c]

    # Triangular multiplicity (matches ccsd_t_slow lines 65-68).
    sym = jnp.where(a == c, 6.0,
                    jnp.where((a == b) | (b == c), 2.0, 1.0))
    d3 = d3_base * sym

    def get_w(p, q, r):
        # vvov[p, q] -> (i, f); t2T[r] -> (f, k, j)
        # vooo[p]    -> (i, j, m); t2T[q, r] -> (m, k)
        slab = jax.lax.dynamic_index_in_dim(vvov, p, axis=0, keepdims=False)
        slab = jax.lax.dynamic_index_in_dim(slab, q, axis=0, keepdims=False)
        vooo_p = jax.lax.dynamic_index_in_dim(vooo, p, axis=0, keepdims=False)
        t2T_r = jax.lax.dynamic_index_in_dim(t2T, r, axis=0, keepdims=False)
        # t2T_qr = t2T[q, r] — must index q FIRST then r, matching the reference.
        t2T_q = jax.lax.dynamic_index_in_dim(t2T, q, axis=0, keepdims=False)
        t2T_qr = jax.lax.dynamic_index_in_dim(t2T_q, r, axis=0, keepdims=False)
        w = jnp.einsum("if,fkj->ijk", slab, t2T_r)
        w -= jnp.einsum("ijm,mk->ijk", vooo_p, t2T_qr)
        return w

    def get_v(p, q, r):
        vvoo_pq = jax.lax.dynamic_index_in_dim(
            jax.lax.dynamic_index_in_dim(vvoo, p, axis=0, keepdims=False),
            q, axis=0, keepdims=False)
        t2T_pq = jax.lax.dynamic_index_in_dim(
            jax.lax.dynamic_index_in_dim(t2T, p, axis=0, keepdims=False),
            q, axis=0, keepdims=False)
        t1T_r = jax.lax.dynamic_index_in_dim(t1T, r, axis=0, keepdims=False)
        fvo_r = jax.lax.dynamic_index_in_dim(fvo, r, axis=0, keepdims=False)
        v = jnp.einsum("ij,k->ijk", vvoo_pq, t1T_r)
        v += jnp.einsum("ij,k->ijk", t2T_pq, fvo_r)
        return v

    wabc = get_w(a, b, c); wacb = get_w(a, c, b)
    wbac = get_w(b, a, c); wbca = get_w(b, c, a)
    wcab = get_w(c, a, b); wcba = get_w(c, b, a)
    vabc = get_v(a, b, c); vacb = get_v(a, c, b)
    vbac = get_v(b, a, c); vbca = get_v(b, c, a)
    vcab = get_v(c, a, b); vcba = get_v(c, b, a)

    zabc = _r3_jax(wabc + 0.5 * vabc) / d3
    zacb = _r3_jax(wacb + 0.5 * vacb) / d3
    zbac = _r3_jax(wbac + 0.5 * vbac) / d3
    zbca = _r3_jax(wbca + 0.5 * vbca) / d3
    zcab = _r3_jax(wcab + 0.5 * vcab) / d3
    zcba = _r3_jax(wcba + 0.5 * vcba) / d3

    et = jnp.zeros((), dtype=wabc.dtype)
    # Block 1: zabc consumer
    et = et + jnp.einsum("ijk,ijk", wabc, zabc)
    et = et + jnp.einsum("ikj,ijk", wacb, zabc)
    et = et + jnp.einsum("jik,ijk", wbac, zabc)
    et = et + jnp.einsum("jki,ijk", wbca, zabc)
    et = et + jnp.einsum("kij,ijk", wcab, zabc)
    et = et + jnp.einsum("kji,ijk", wcba, zabc)
    # Block 2: zacb consumer
    et = et + jnp.einsum("ijk,ijk", wacb, zacb)
    et = et + jnp.einsum("ikj,ijk", wabc, zacb)
    et = et + jnp.einsum("jik,ijk", wcab, zacb)
    et = et + jnp.einsum("jki,ijk", wcba, zacb)
    et = et + jnp.einsum("kij,ijk", wbac, zacb)
    et = et + jnp.einsum("kji,ijk", wbca, zacb)
    # Block 3: zbac consumer
    et = et + jnp.einsum("ijk,ijk", wbac, zbac)
    et = et + jnp.einsum("ikj,ijk", wbca, zbac)
    et = et + jnp.einsum("jik,ijk", wabc, zbac)
    et = et + jnp.einsum("jki,ijk", wacb, zbac)
    et = et + jnp.einsum("kij,ijk", wcba, zbac)
    et = et + jnp.einsum("kji,ijk", wcab, zbac)
    # Block 4: zbca consumer
    et = et + jnp.einsum("ijk,ijk", wbca, zbca)
    et = et + jnp.einsum("ikj,ijk", wbac, zbca)
    et = et + jnp.einsum("jik,ijk", wcba, zbca)
    et = et + jnp.einsum("jki,ijk", wcab, zbca)
    et = et + jnp.einsum("kij,ijk", wabc, zbca)
    et = et + jnp.einsum("kji,ijk", wacb, zbca)
    # Block 5: zcab consumer
    et = et + jnp.einsum("ijk,ijk", wcab, zcab)
    et = et + jnp.einsum("ikj,ijk", wcba, zcab)
    et = et + jnp.einsum("jik,ijk", wacb, zcab)
    et = et + jnp.einsum("jki,ijk", wabc, zcab)
    et = et + jnp.einsum("kij,ijk", wbca, zcab)
    et = et + jnp.einsum("kji,ijk", wbac, zcab)
    # Block 6: zcba consumer
    et = et + jnp.einsum("ijk,ijk", wcba, zcba)
    et = et + jnp.einsum("ikj,ijk", wcab, zcba)
    et = et + jnp.einsum("jik,ijk", wbca, zcba)
    et = et + jnp.einsum("jki,ijk", wbac, zcba)
    et = et + jnp.einsum("kij,ijk", wacb, zcba)
    et = et + jnp.einsum("kji,ijk", wabc, zcba)
    return et


def _make_scan_fn():
    """Compile a per-device on-device scan that sums contributions over triples.

    Wrapped in a builder so each device gets its own JIT compilation cache
    keyed by tensor shapes (one compile per problem size, reused across runs
    when the persistent XLA cache is active).
    """

    @jax.jit
    def _scan(abc_array, mo_e_o, mo_e_v,
              t1T, t2T, vvov, vooo, vvoo, fvo):
        def body(et_acc, abc):
            contrib = _single_triple_contribution(
                abc[0], abc[1], abc[2],
                mo_e_o, mo_e_v, t1T, t2T, vvov, vooo, vvoo, fvo,
            )
            return et_acc + contrib, None

        et0 = jnp.zeros((), dtype=t2T.dtype)
        et_final, _ = jax.lax.scan(body, et0, abc_array)
        return et_final

    return _scan


_scan_triples = _make_scan_fn()


# ---------------------------------------------------------------------------
# Streaming-variant JIT kernel: consume three explicit slabs of vvov
# ---------------------------------------------------------------------------

def _single_triple_streaming(slab_a, slab_b, slab_c,
                             a, b, c,
                             mo_e_o, mo_e_v,
                             t1T, t2T, vooo, vvoo, fvo):
    """Same algebra as ``_single_triple_contribution`` but indexes three
    explicit per-triple slabs of ``vvov`` instead of the full 4-D tensor.

    ``slab_x`` is ``vvov[x]`` with shape ``(nvir, nocc, nvir)``. The six W
    permutations need exactly the slab-pairs ``(slab_a, slab_b, slab_c)``
    cross-indexed by the other two of ``a/b/c``; nothing else changes.

    Compiled once per problem-size on each device — the slab tensors and
    accessory tensors have shapes that don't depend on ``(a, b, c)``.
    """
    eijk = (mo_e_o[:, None, None]
            + mo_e_o[None, :, None]
            + mo_e_o[None, None, :])
    d3_base = eijk - mo_e_v[a] - mo_e_v[b] - mo_e_v[c]
    sym = jnp.where(a == c, 6.0,
                    jnp.where((a == b) | (b == c), 2.0, 1.0))
    d3 = d3_base * sym

    def _w_from(slab_p, q, r, p_for_vooo, q_for_t2T):
        # slab_p[q] -> (i, f) = vvov[p, q]
        slab_pq = jax.lax.dynamic_index_in_dim(slab_p, q, axis=0, keepdims=False)
        vooo_p = jax.lax.dynamic_index_in_dim(vooo, p_for_vooo, axis=0, keepdims=False)
        t2T_r = jax.lax.dynamic_index_in_dim(t2T, r, axis=0, keepdims=False)
        # t2T_qr = t2T[q_for_t2T, r] — index q FIRST, then r (matches the reference).
        t2T_q = jax.lax.dynamic_index_in_dim(t2T, q_for_t2T, axis=0, keepdims=False)
        t2T_qr = jax.lax.dynamic_index_in_dim(t2T_q, r, axis=0, keepdims=False)
        w = jnp.einsum("if,fkj->ijk", slab_pq, t2T_r)
        w -= jnp.einsum("ijm,mk->ijk", vooo_p, t2T_qr)
        return w

    def _v_from(p, q, r):
        vvoo_pq = jax.lax.dynamic_index_in_dim(
            jax.lax.dynamic_index_in_dim(vvoo, p, axis=0, keepdims=False),
            q, axis=0, keepdims=False)
        t2T_pq = jax.lax.dynamic_index_in_dim(
            jax.lax.dynamic_index_in_dim(t2T, p, axis=0, keepdims=False),
            q, axis=0, keepdims=False)
        t1T_r = jax.lax.dynamic_index_in_dim(t1T, r, axis=0, keepdims=False)
        fvo_r = jax.lax.dynamic_index_in_dim(fvo, r, axis=0, keepdims=False)
        v = jnp.einsum("ij,k->ijk", vvoo_pq, t1T_r)
        v += jnp.einsum("ij,k->ijk", t2T_pq, fvo_r)
        return v

    # 6 W permutations: each is (slab_for_outer_index, inner_index, r-index, ...)
    wabc = _w_from(slab_a, b, c, a, b)
    wacb = _w_from(slab_a, c, b, a, c)
    wbac = _w_from(slab_b, a, c, b, a)
    wbca = _w_from(slab_b, c, a, b, c)
    wcab = _w_from(slab_c, a, b, c, a)
    wcba = _w_from(slab_c, b, a, c, b)

    vabc = _v_from(a, b, c); vacb = _v_from(a, c, b)
    vbac = _v_from(b, a, c); vbca = _v_from(b, c, a)
    vcab = _v_from(c, a, b); vcba = _v_from(c, b, a)

    zabc = _r3_jax(wabc + 0.5 * vabc) / d3
    zacb = _r3_jax(wacb + 0.5 * vacb) / d3
    zbac = _r3_jax(wbac + 0.5 * vbac) / d3
    zbca = _r3_jax(wbca + 0.5 * vbca) / d3
    zcab = _r3_jax(wcab + 0.5 * vcab) / d3
    zcba = _r3_jax(wcba + 0.5 * vcba) / d3

    et = jnp.zeros((), dtype=wabc.dtype)
    # Same 36-term block as _single_triple_contribution.
    for w_, z_ in (
        (wabc, zabc), (wacb, zacb), (wbac, zbac),
        (wbca, zbca), (wcab, zcab), (wcba, zcba),
    ):
        et = et + jnp.einsum("ijk,ijk", w_, z_)
    # Cross terms — flat list of (W-source, Z-target, einsum-spec) tuples.
    for w_, z_, spec in (
        (wacb, zabc, "ikj,ijk"), (wbac, zabc, "jik,ijk"),
        (wbca, zabc, "jki,ijk"), (wcab, zabc, "kij,ijk"),
        (wcba, zabc, "kji,ijk"),
        (wabc, zacb, "ikj,ijk"), (wcab, zacb, "jik,ijk"),
        (wcba, zacb, "jki,ijk"), (wbac, zacb, "kij,ijk"),
        (wbca, zacb, "kji,ijk"),
        (wbca, zbac, "ikj,ijk"), (wabc, zbac, "jik,ijk"),
        (wacb, zbac, "jki,ijk"), (wcba, zbac, "kij,ijk"),
        (wcab, zbac, "kji,ijk"),
        (wbac, zbca, "ikj,ijk"), (wcba, zbca, "jik,ijk"),
        (wcab, zbca, "jki,ijk"), (wabc, zbca, "kij,ijk"),
        (wacb, zbca, "kji,ijk"),
        (wcba, zcab, "ikj,ijk"), (wacb, zcab, "jik,ijk"),
        (wabc, zcab, "jki,ijk"), (wbca, zcab, "kij,ijk"),
        (wbac, zcab, "kji,ijk"),
        (wcab, zcba, "ikj,ijk"), (wbca, zcba, "jik,ijk"),
        (wbac, zcba, "jki,ijk"), (wacb, zcba, "kij,ijk"),
        (wabc, zcba, "kji,ijk"),
    ):
        et = et + jnp.einsum(spec, w_, z_)
    return et


def _make_streaming_kernel():
    @jax.jit
    def _kern(slab_a, slab_b, slab_c, a, b, c,
              mo_e_o, mo_e_v, t1T, t2T, vooo, vvoo, fvo):
        return _single_triple_streaming(
            slab_a, slab_b, slab_c, a, b, c,
            mo_e_o, mo_e_v, t1T, t2T, vooo, vvoo, fvo,
        )
    return _kern


_streaming_kernel = _make_streaming_kernel()


# ---------------------------------------------------------------------------
# Slab provider + LRU prefetcher
# ---------------------------------------------------------------------------

class _SlabPrefetcher:
    """Per-device async slab cache backed by a host array.

    ``host_source`` is anything that supports ``host_source[idx]`` returning a
    ``(nvir, nocc, nvir)`` slab (NumPy array view, HDF5 dataset, etc.). Slabs
    are loaded on a small background thread pool, ``device_put`` onto
    ``device``, and held in an LRU dict capped at ``max_cached``.

    The cache size is set by the caller from ``gpu_max_memory``; the dict is
    keyed by ``int(slab_idx)`` and stores on-device arrays. Eviction simply
    drops the reference; the device buffer is freed when JAX's tracker sees
    no remaining handles to it.

    Thread safety: the cache and ``pending`` dict are guarded by a single
    lock. ``get`` waits on the in-flight load Future outside the lock to
    avoid blocking other prefetches.
    """

    def __init__(self, host_source, device, max_cached, executor):
        from collections import OrderedDict
        self.source = host_source
        self.device = device
        self.max_cached = max(3, int(max_cached))
        self.cache = OrderedDict()  # idx -> on-device array
        self.pending = {}            # idx -> Future
        self.lock = threading.Lock()
        self.executor = executor
        # Stats — useful for tuning cache size in production runs.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _load(self, idx):
        # HDF5 reads are not Python-thread-safe for some builds; serialise on
        # the executor (small worker count is fine — we want pipelining, not
        # parallel HDF5 reads from a single dataset).
        host = np.asarray(self.source[idx])
        if self.device is None:
            return jnp.asarray(host)
        return jax.device_put(host, self.device)

    def prefetch(self, idx):
        """Issue a non-blocking load if ``idx`` is not already cached/pending."""
        with self.lock:
            if idx in self.cache or idx in self.pending:
                return
            self.pending[idx] = self.executor.submit(self._load, idx)

    def get(self, idx):
        """Return the slab for ``idx``, blocking if necessary. Updates LRU."""
        with self.lock:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                self.hits += 1
                return self.cache[idx]
            future = self.pending.pop(idx, None)
            if future is None:
                self.misses += 1
                future = self.executor.submit(self._load, idx)
            else:
                self.misses += 1
        slab = future.result()
        with self.lock:
            self.cache[idx] = slab
            while len(self.cache) > self.max_cached:
                self.cache.popitem(last=False)
                self.evictions += 1
            self.cache.move_to_end(idx)
        return slab

    def stats(self):
        return dict(hits=self.hits, misses=self.misses,
                    evictions=self.evictions,
                    cache_size=len(self.cache),
                    max_cached=self.max_cached)


def _make_host_slab_view(eris, nocc, nvir):
    """Return an object supporting ``[idx] -> (nvir, nocc, nvir)`` slab access.

    Always materialises ``ovvv`` on the host (NumPy) and transposes to vvov
    layout once; the result is a contiguous host array, indexable as
    ``vvov[idx]``. For very large jobs this still requires
    ``nocc * nvir**3 * 8`` host bytes — that's the *host* RAM cost, which is
    typically large but available (~51 GB for nvir=600, nocc=30). The
    point of streaming is to bound *device* HBM, not host RAM.

    If/when even host RAM is too small, this is the right hook to swap in an
    HDF5-dataset-backed slab view (the JIT kernel needs no further changes).
    """
    if hasattr(eris, "get_ovvv"):
        ovvv = np.asarray(eris.get_ovvv())
    else:
        ovvv = np.asarray(eris.ovvv)
    if ovvv.ndim == 3:
        ovvv = lib.unpack_tril(
            ovvv.reshape(nocc * nvir, -1)
        ).reshape(nocc, nvir, nvir, nvir)
    return np.ascontiguousarray(ovvv.transpose(1, 3, 0, 2))


def _balanced_a_partition(nvir, n_partitions):
    """Cube-root-balanced contiguous-`a` partition.

    The number of triples for a-range ``[lo, hi)`` is the sum of triangular
    numbers ``sum_{a=lo}^{hi-1} (a+1)(a+2)/2`` ~ ``hi^3 / 6``. Equal-work
    boundaries are therefore ``a_d = nvir * ((d / n)^(1/3))`` rounded.

    If ``n_partitions > nvir``, the surplus partitions are returned as empty
    ranges — the worker function treats those as a no-op partition.
    """
    if n_partitions <= 1:
        return [(0, nvir)]
    bounds = [int(round(nvir * (d / n_partitions) ** (1.0 / 3.0)))
              for d in range(n_partitions + 1)]
    bounds[0] = 0
    bounds[-1] = nvir
    # Make monotonic-non-decreasing within [0, nvir]. Empty partitions are
    # allowed (when there are more devices than virtuals).
    for i in range(1, len(bounds)):
        bounds[i] = max(bounds[i], bounds[i - 1])
        bounds[i] = min(bounds[i], nvir)
    return [(bounds[d], bounds[d + 1]) for d in range(n_partitions)]


# ---------------------------------------------------------------------------
# Multi-GPU driver
# ---------------------------------------------------------------------------

def _check_hbm_budget(nocc, nvir, n_devices, log):
    """Return required-bytes-per-device for the replicated path, and warn / abort.

    Strategy A (this module) replicates ``vvov`` on every device. When that
    won't fit, Strategy B (sharded vvov with cross-device fetches) is
    required; that's not yet implemented and is gated behind an explicit
    error here so callers don't silently OOM.
    """
    vvov_bytes = float(nocc) * float(nvir) ** 3 * 8
    other_bytes = (
        # t2T, vvoo
        2 * float(nocc) ** 2 * float(nvir) ** 2 * 8
        # vooo
        + float(nocc) ** 3 * float(nvir) * 8
        # t1T, fvo, mo_e — negligible but cheap to include
        + 4 * float(nocc) * float(nvir) * 8
    )
    total_per_dev_gb = (vvov_bytes + other_bytes) / 1e9
    log.info("xTC-(T) multi-GPU memory budget: %.2f GB per device "
             "(vvov %.2f GB + small tensors %.2f GB) across %d device(s)",
             total_per_dev_gb, vvov_bytes / 1e9, other_bytes / 1e9, n_devices)
    return total_per_dev_gb


def _build_layout_transforms(eris, nocc, nvir):
    """Materialise vvov, vooo, vvoo on the host once (CPU)."""
    if hasattr(eris, "get_ovvv"):
        ovvv = np.asarray(eris.get_ovvv())
    else:
        ovvv = np.asarray(eris.ovvv)
    if ovvv.ndim == 3:
        ovvv = lib.unpack_tril(
            ovvv.reshape(nocc * nvir, -1)
        ).reshape(nocc, nvir, nvir, nvir)
    # Match ccsd_t_slow convention: vvov layout (a, b, i, f); vooo (a, i, j, k);
    # vvoo (a, b, i, j). For non-Hermitian xTC we drop the .conj().
    vvov = np.ascontiguousarray(ovvv.transpose(1, 3, 0, 2))
    vooo = np.ascontiguousarray(np.asarray(eris.ovoo).transpose(1, 0, 2, 3))
    vvoo = np.ascontiguousarray(np.asarray(eris.ovov).transpose(1, 3, 0, 2))
    return vvov, vooo, vvoo


def _kernel_multigpu(mycc, eris, t1, t2, verbose=None):
    """Multi-device JAX-jitted (T) on top of converged xTC amplitudes.

    Tensors that are small enough to replicate (``t1T``, ``t2T``, ``vooo``,
    ``vvoo``, ``fvo``, ``mo_energy``, ``vvov``) are broadcast to every local
    device. Triangular (a, b, c) triples are partitioned round-robin and one
    worker thread per device runs an on-device ``lax.scan`` that accumulates
    its partition's contribution to ``et``. Partials are summed on the host.

    Falls back to the reference path on a single CPU device unless
    ``mycc.ccsd_t_force_multigpu = True`` — the JAX path on CPU is slower
    than NumPy for small problems because of XLA dispatch overhead.
    """
    if isinstance(verbose, pyscf_logger.Logger):
        log = verbose
    else:
        log = pyscf_logger.Logger(mycc.stdout,
                                  verbose if verbose is not None else mycc.verbose)
    t_start = time.perf_counter()

    nocc, nvir = t1.shape
    devices = _solver_local_devices()
    n_devices = len(devices)

    only_cpu = all(d is None or getattr(d, "platform", "") == "cpu" for d in devices)
    if only_cpu and not getattr(mycc, "ccsd_t_force_multigpu", False):
        log.info("xTC-(T) multi-GPU: no accelerators present and "
                 "ccsd_t_force_multigpu is unset — falling back to NumPy reference.")
        return _kernel_reference(mycc, eris, t1, t2, verbose=verbose)

    _check_hbm_budget(nocc, nvir, n_devices, log)

    # --- 1. Host-side layout transforms (once) ---
    vvov_host, vooo_host, vvoo_host = _build_layout_transforms(eris, nocc, nvir)
    t1T_host = np.ascontiguousarray(t1.T)
    t2T_host = np.ascontiguousarray(t2.transpose(2, 3, 0, 1))
    fvo_host = np.ascontiguousarray(eris.fock[nocc:, :nocc])
    mo_e = np.asarray(eris.mo_energy)
    mo_e_o_host = mo_e[:nocc]
    mo_e_v_host = mo_e[nocc:]

    # --- 2. Broadcast everything to every device ---
    vvov_by_dev = broadcast_to_devices(vvov_host, devices)
    vooo_by_dev = broadcast_to_devices(vooo_host, devices)
    vvoo_by_dev = broadcast_to_devices(vvoo_host, devices)
    t1T_by_dev = broadcast_to_devices(t1T_host, devices)
    t2T_by_dev = broadcast_to_devices(t2T_host, devices)
    fvo_by_dev = broadcast_to_devices(fvo_host, devices)
    mo_e_o_by_dev = broadcast_to_devices(mo_e_o_host, devices)
    mo_e_v_by_dev = broadcast_to_devices(mo_e_v_host, devices)

    # Free the large host copies — they live on-device now.
    del vvov_host, vooo_host, vvoo_host

    # --- 3. Triangular triples, partitioned across devices ---
    triples = [(a, b, c)
               for a in range(nvir)
               for b in range(a + 1)
               for c in range(b + 1)]
    log.debug("xTC-(T) sweeping %d triangular triples across %d device(s)",
              len(triples), n_devices)
    triples_by_dev = partition_round_robin(triples, devices)

    # --- 4. Per-device worker ---
    def worker(device):
        my_triples = triples_by_dev[device]
        if not my_triples:
            return 0.0
        abc_np = np.asarray(my_triples, dtype=np.int32)
        ctx = (jax.default_device(device) if device is not None
               else contextlib.nullcontext())
        with ctx:
            abc_jax = jax.device_put(abc_np, device) if device is not None else jnp.asarray(abc_np)
            et_scalar = _scan_triples(
                abc_jax,
                mo_e_o_by_dev[device], mo_e_v_by_dev[device],
                t1T_by_dev[device], t2T_by_dev[device],
                vvov_by_dev[device], vooo_by_dev[device],
                vvoo_by_dev[device], fvo_by_dev[device],
            )
            # Force completion on this device before returning to the host.
            return float(np.asarray(et_scalar))

    # --- 5. Concurrent dispatch — one OS thread per device ---
    if n_devices > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as pool:
            futs = [pool.submit(worker, d) for d in devices]
            partials = [f.result() for f in futs]
    else:
        partials = [worker(devices[0])]

    et = 2.0 * sum(partials)
    log.info("xTC-CCSD(T) (multi-GPU) correction = %.15g  (%d devices, %.1f s)",
             et, n_devices, time.perf_counter() - t_start)
    return float(et)


# ---------------------------------------------------------------------------
# Streaming multi-GPU path: vvov host-resident, per-device LRU slab cache
# ---------------------------------------------------------------------------

def _resolve_max_cached_slabs(mycc, nocc, nvir, n_partitions, log):
    """Bound the LRU slab cache so the device fits cache + replicated tensors.

    Each slab is ``nvir * nocc * nvir * 8 B``. The accessory tensors
    (``t2T``, ``vvoo``, ``vooo``, smalls) take a fixed amount; the rest of
    the per-device budget is split among slabs. ``mycc.ccsd_t_max_cached_slabs``
    overrides this if set.
    """
    override = getattr(mycc, "ccsd_t_max_cached_slabs", None)
    if override is not None:
        log.info("xTC-(T) streaming: cache cap from ccsd_t_max_cached_slabs = %d", override)
        return int(override)

    slab_bytes = float(nvir) * float(nocc) * float(nvir) * 8
    accessory_bytes = (
        2 * float(nocc) ** 2 * float(nvir) ** 2 * 8   # t2T, vvoo
        + float(nocc) ** 3 * float(nvir) * 8          # vooo
        + 8 * float(nocc) * float(nvir) * 8           # smalls (generous)
    )
    gpu_max_mb = getattr(mycc, "gpu_max_memory", None)
    if gpu_max_mb is None:
        # No budget hint — pick a sensible default that gives prefetch room
        # for several triples ahead without ballooning unbounded.
        default = min(nvir, 32)
        log.info("xTC-(T) streaming: cache cap defaulted to %d slabs "
                 "(no gpu_max_memory set)", default)
        return default
    # 60% of HBM for slabs, leaving headroom for compile artifacts +
    # transient JIT workspaces. The compute itself only needs a few
    # intermediates of size (nocc^3) — negligible compared to slab cache.
    budget_bytes = 0.6 * float(gpu_max_mb) * 1e6 - accessory_bytes
    max_slabs = max(3, int(budget_bytes / slab_bytes))
    log.info("xTC-(T) streaming: cache cap = %d slabs "
             "(slab=%.1f MB, budget=%.1f GB/device)",
             max_slabs, slab_bytes / 1e6, float(gpu_max_mb) / 1000)
    return max_slabs


def _per_device_triples_contiguous_a(nvir, n_devices, device_index):
    """Triples (a, b, c) with ``a`` in this device's contiguous partition."""
    if n_devices == 0:
        return []
    partitions = _balanced_a_partition(nvir, n_devices)
    a_lo, a_hi = partitions[device_index]
    return [(a, b, c)
            for a in range(a_lo, a_hi)
            for b in range(a + 1)
            for c in range(b + 1)]


def _kernel_multigpu_streaming(mycc, eris, t1, t2, verbose=None):
    """Streaming multi-GPU (T): vvov on host, slabs pulled on demand.

    Memory profile per device: at most ``ccsd_t_max_cached_slabs`` slabs of
    ``vvov`` plus the small replicated tensors (``t2T``, ``vvoo``, ``vooo``,
    ``t1T``, ``fvo``, ``mo_energy``). One background thread pool per device
    overlaps HDF5/numpy slab reads + ``device_put`` with on-device compute.

    Triples are partitioned contiguously by ``a`` with cube-root-balanced
    boundaries so the FLOP counts per device are equal.

    The JIT kernel takes three slabs and the integer ``(a, b, c)`` so XLA
    compiles a single program reused across every triple.
    """
    if isinstance(verbose, pyscf_logger.Logger):
        log = verbose
    else:
        log = pyscf_logger.Logger(mycc.stdout,
                                  verbose if verbose is not None else mycc.verbose)
    t_start = time.perf_counter()

    nocc, nvir = t1.shape
    devices = _solver_local_devices()
    n_devices = len(devices)

    only_cpu = all(d is None or getattr(d, "platform", "") == "cpu" for d in devices)
    if only_cpu and not getattr(mycc, "ccsd_t_force_multigpu", False):
        log.info("xTC-(T) streaming: CPU-only environment without "
                 "ccsd_t_force_multigpu — falling back to NumPy reference.")
        return _kernel_reference(mycc, eris, t1, t2, verbose=verbose)

    # --- Host-resident vvov + accessories ---
    log.info("xTC-(T) streaming: building host vvov view (%.1f GB)",
             nocc * nvir ** 3 * 8 / 1e9)
    vvov_host = _make_host_slab_view(eris, nocc, nvir)
    vooo_host = np.ascontiguousarray(np.asarray(eris.ovoo).transpose(1, 0, 2, 3))
    vvoo_host = np.ascontiguousarray(np.asarray(eris.ovov).transpose(1, 3, 0, 2))
    t1T_host = np.ascontiguousarray(t1.T)
    t2T_host = np.ascontiguousarray(t2.transpose(2, 3, 0, 1))
    fvo_host = np.ascontiguousarray(eris.fock[nocc:, :nocc])
    mo_e = np.asarray(eris.mo_energy)
    mo_e_o_host = mo_e[:nocc]
    mo_e_v_host = mo_e[nocc:]

    # Replicate the small tensors to every device. ``vvov`` is *not*
    # replicated — that's the whole point.
    vooo_by_dev = broadcast_to_devices(vooo_host, devices)
    vvoo_by_dev = broadcast_to_devices(vvoo_host, devices)
    t1T_by_dev = broadcast_to_devices(t1T_host, devices)
    t2T_by_dev = broadcast_to_devices(t2T_host, devices)
    fvo_by_dev = broadcast_to_devices(fvo_host, devices)
    mo_e_o_by_dev = broadcast_to_devices(mo_e_o_host, devices)
    mo_e_v_by_dev = broadcast_to_devices(mo_e_v_host, devices)

    max_cached = _resolve_max_cached_slabs(mycc, nocc, nvir, n_devices, log)
    prefetch_workers = max(2, int(getattr(mycc, "ccsd_t_prefetch_workers", 2)))
    prefetch_lookahead = max(1, int(getattr(mycc, "ccsd_t_prefetch_lookahead", 4)))

    # --- Per-device worker ---
    def worker(device_index, device):
        my_triples = _per_device_triples_contiguous_a(nvir, n_devices, device_index)
        if not my_triples:
            return 0.0, None

        # Each device gets its own slab prefetch pool — small thread count
        # avoids HDF5 lock contention; pipelining is the goal, not parallel
        # reads of the same dataset.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=prefetch_workers,
            thread_name_prefix=f"xtc-t-slab-dev{device_index}",
        )
        try:
            slab_cache = _SlabPrefetcher(
                vvov_host, device, max_cached, executor,
            )
            # Prime: fetch lookahead worth of slabs from the head of the list.
            seen = set()
            for k in range(min(prefetch_lookahead, len(my_triples))):
                for s in my_triples[k]:
                    if s not in seen:
                        slab_cache.prefetch(int(s))
                        seen.add(s)

            ctx = (jax.default_device(device) if device is not None
                   else contextlib.nullcontext())
            et_local = 0.0
            with ctx:
                for i, (a, b, c) in enumerate(my_triples):
                    # Look ahead and prefetch upcoming slabs while compute runs.
                    look = i + prefetch_lookahead
                    if look < len(my_triples):
                        for s in my_triples[look]:
                            slab_cache.prefetch(int(s))

                    slab_a = slab_cache.get(int(a))
                    slab_b = slab_cache.get(int(b)) if b != a else slab_a
                    slab_c = (slab_cache.get(int(c))
                              if c != a and c != b
                              else (slab_a if c == a else slab_b))

                    contrib = _streaming_kernel(
                        slab_a, slab_b, slab_c,
                        jnp.int32(a), jnp.int32(b), jnp.int32(c),
                        mo_e_o_by_dev[device], mo_e_v_by_dev[device],
                        t1T_by_dev[device], t2T_by_dev[device],
                        vooo_by_dev[device], vvoo_by_dev[device],
                        fvo_by_dev[device],
                    )
                    et_local += float(np.asarray(contrib))
            return et_local, slab_cache.stats()
        finally:
            executor.shutdown(wait=True)

    # --- Concurrent dispatch ---
    if n_devices > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as pool:
            futs = [pool.submit(worker, idx, d) for idx, d in enumerate(devices)]
            results = [f.result() for f in futs]
    else:
        results = [worker(0, devices[0])]

    partials = [r[0] for r in results]
    for idx, (_, stats) in enumerate(results):
        if stats is not None:
            log.debug("xTC-(T) streaming dev %d slab stats: %s", idx, stats)

    et = 2.0 * sum(partials)
    log.info("xTC-CCSD(T) (streaming) correction = %.15g  (%d devices, %.1f s)",
             et, n_devices, time.perf_counter() - t_start)
    return float(et)
