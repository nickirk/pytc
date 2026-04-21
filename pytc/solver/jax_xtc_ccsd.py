import contextlib
import logging
import threading
import time
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import lib
from pyscf import ao2mo

from pytc.solver import xtc_ccsd
from pytc import xtc as xtc_mod
from pytc.utils.gpu_memory import resolve_vvvv_panel_block_sizes
from pytc.utils.gpu_memory import estimate_blksize

# JAX config
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

# --- JAX-compatible Intermediates (Micro-Kernels) ---

@jax.jit
def _jax_cc_Fov(t1, eris_fock, eris_ovov):
    fov = eris_fock[:t1.shape[0], t1.shape[0]:]
    Fkc  = 2*jnp.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   jnp.einsum('kdlc,ld->kc', eris_ovov, t1)
    Fkc += fov
    return Fkc

@jax.jit
def _jax_cc_Foo(t1, t2, eris_fock, eris_ovov):
    nocc = t1.shape[0]
    foo = eris_fock[:nocc, :nocc]
    Fki  = 2*jnp.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   jnp.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*jnp.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   jnp.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Fki += foo
    return Fki

@jax.jit
def _jax_cc_Fvv(t1, t2, eris_fock, eris_ovov):
    nocc = t1.shape[0]
    fvv = eris_fock[nocc:, nocc:]
    Fac  =-2*jnp.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   jnp.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*jnp.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   jnp.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Fac += fvv
    return Fac

@jax.jit
def _jax_Loo(t1, t2, eris_fock, eris_ovov, eris_ovoo):
    nocc = t1.shape[0]
    ki = _jax_cc_Foo(t1, t2, eris_fock, eris_ovov)
    ki += jnp.einsum('kc,ic->ki', eris_fock[:nocc, nocc:], t1)
    ki += 2*jnp.einsum('lcki,lc->ki', eris_ovoo, t1)
    ki -=   jnp.einsum('kcli,lc->ki', eris_ovoo, t1)
    return ki

@jax.jit
def _jax_Lvv(t1, t2, eris_fock, eris_ovov, eris_ovvv_all=None):
    # This Lvv includes contributions from Fvv and ovvv if provided (memory resident)
    # If ovvv is blocked (None), only the Fvv part is computed here.
    nocc = t1.shape[0]
    fov = eris_fock[:nocc, nocc:]
    ac = _jax_cc_Fvv(t1, t2, eris_fock, eris_ovov)
    ac -= jnp.einsum('kc,ka->ac', fov, t1)
    if eris_ovvv_all is not None:
        ac += 2*jnp.einsum('kdac,kd->ac', eris_ovvv_all, t1)
        ac -=   jnp.einsum('kcad,kd->ac', eris_ovvv_all, t1)
    return ac

@jax.jit
def _jax_cc_Woooo(t1, t2, eris_oooo, eris_ovov, eris_ovoo):
    Wklij  = jnp.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += jnp.einsum('kclj,ic->klij', eris_ovoo, t1)
    Wklij += jnp.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += jnp.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += eris_oooo.transpose(0,2,1,3)
    return Wklij

# Micro-kernels for Block processing

@jax.jit
def kernel_process_ovvv_block(ovvv_blk, t1, t2, tau):
    """
    JIT-compiled kernel for processing ovvv block contributions to
    t1new, Lvv, Wvoov, Wvovo, tmp_a, tmp_b.

    ``ovvv_blk`` has shape ``(nocc, nvir, blk, nvir)`` with logical axes
    ``(k, d, a, c)``; the sliced dimension is axis 2 (``a``).

    Implementation note
    -------------------
    The textbook form of this kernel uses paired einsums with ``'kdac'``
    vs ``'kcad'`` subscripts against the same ``ovvv_blk`` tensor. That
    asks XLA to produce an axis-1↔axis-3 transpose of a ~30 GB f64 block
    and then fuse the transpose into several downstream contractions
    (``input_transpose_fusion`` HLOs). For the shapes we actually hit
    at pCV5Z-class calculations (e.g. ``f64[1179, 3070116]`` ≈ 27 GB with
    a 1179-long minor axis) the XLA autotuner runs out of viable tile
    configurations and aborts with::

        INTERNAL: Autotuning failed for HLO: %input_transpose_fusion ...
                  NOT_FOUND: No valid config found!

    We eliminate every ``'kcad'``/``'kcbd'``-style binding by materializing
    ``ovvv_swap = transpose(ovvv_blk, (0, 3, 2, 1))`` once as a single
    standalone transpose HLO (which the autotuner handles comfortably),
    and rewriting every contraction in the native ``'kdac'`` form against
    either ``ovvv_blk``, ``ovvv_swap``, or the linear combination
    ``ovvv_t1sym = 2*ovvv_blk - ovvv_swap`` that folds the
    ``2*native - swap`` T1 / Lvv pattern into a single einsum.

    The output of this kernel is bit-for-bit equivalent to the textbook
    version up to floating-point reassociation inside the contractions;
    see ``pytc/test/test_kernel_process_ovvv_block.py`` for the
    regression test.
    """
    # Axes swap of (d, c) computed exactly once as a standalone HLO.
    ovvv_swap = jnp.transpose(ovvv_blk, (0, 3, 2, 1))
    # Linear combination absorbing the "2*native - swap" pattern that
    # appears in the T1 and Lvv contributions.
    ovvv_t1sym = 2 * ovvv_blk - ovvv_swap

    # 1. t1_upd (ia): fused 2*native - swap against t2 and (t1, t1).
    t1_upd  = jnp.einsum('kdac,ikcd->ia',    ovvv_t1sym, t2)
    t1_upd += jnp.einsum('kdac,kd,ic->ia',   ovvv_t1sym, t1, t1)

    # 2. Lvv_blk (ac): fused 2*native - swap against t1.
    Lvv_blk = jnp.einsum('kdac,kd->ac',      ovvv_t1sym, t1)

    # 3. Wvoov_blk (akic): the original was 'kcad,id->akic' on ovvv_blk,
    #    i.e. the pure swap branch — compute it on ovvv_swap.
    Wvoov_blk = jnp.einsum('kdac,id->akic',  ovvv_swap, t1)

    # 4. Wvovo_blk (akci): native branch, unchanged.
    Wvovo_blk = jnp.einsum('kdac,id->akci',  ovvv_blk,  t1)

    # 5. tmp_a_blk (kaij): native branch, unchanged.
    tmp_a_blk = jnp.einsum('kdac,ijcd->kaij', ovvv_blk,  tau)

    # 6. tmp_b_blk (kbij): the original was 'kcbd,ijcd->kbij' on ovvv_blk,
    #    which is the swap branch. On ovvv_swap this is the native
    #    'kdac,ijcd->kaij' pattern (the output letter is just renamed
    #    'a' -> 'b' to match the original's output labeling).
    tmp_b_blk = jnp.einsum('kdbc,ijcd->kbij', ovvv_swap, tau)

    return t1_upd, Lvv_blk, Wvoov_blk, Wvovo_blk, tmp_a_blk, tmp_b_blk

@jax.jit
def kernel_process_vovv_block(vovv_slice, eris_oovv, t1_slice, t1_full):
    """
    JIT-compiled kernel for processing vovv block contributions to t2new.
    vovv_slice: (a_blk, i, b, c)
    eris_oovv: (k, i, b, c) -- complete
    t1_slice: (k, a_blk)
    t1_full: (j, c)
    """
    # To avoid OOM, split contraction and avoid large intermediate (a_blk, b, i, c)
    # term1 = einsum('aibc,jc->ijab', vovv_slice, t1_full)
    # term2 = einsum('kibc,ka,jc->ijab', eris_oovv, -t1_slice, t1_full)
    
    term = jnp.einsum('aibc,jc->ijab', vovv_slice, t1_full)
    
    # Optimized term2 contraction order
    tmp = jnp.einsum('kibc,jc->kijb', eris_oovv, t1_full)
    term += jnp.einsum('kijb,ka->ijab', tmp, -t1_slice)
    return term

@jax.jit
def _jax_energy(t1, t2, eris_fock, eris_ovov):
    nocc = t1.shape[0]
    fov = eris_fock[:nocc, nocc:]
    e = 2 * jnp.einsum('ia,ia->', fov, t1)
    tau = t2 + jnp.einsum('ia,jb->ijab', t1, t1)
    # ovov is (i, a, j, b)
    # energy: 2*(ia|jb) - (ib|ja)
    e += 2 * jnp.einsum('ijab,iajb->', tau, eris_ovov)
    e += -jnp.einsum('ijab,ibja->', tau, eris_ovov)
    return e


class RCCSD(xtc_ccsd.RCCSD):
    """Restricted CCSD with ISDF-XTC integrals (JAX Optimized)."""
    
    def update_amps(self, t1, t2, eris):
        return _update_amps(self, t1, t2, eris)

    def energy(self, t1=None, t2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.eris
        
        # Cast to JAX for computation
        t1_jax = jnp.asarray(t1)
        t2_jax = jnp.asarray(t2)
        fock_jax = jnp.asarray(eris.fock)
        ovov_jax = jnp.asarray(eris.ovov)
        
        return np.array(_jax_energy(t1_jax, t2_jax, fock_jax, ovov_jax)).item()


def _should_force_host_accumulators(eris, n_devices_local):
    """Whether the OVVV/VOVV pipelines must use host-side accumulators.

    GPU-resident accumulators are only viable when *all three* of the
    following hold:

      * exactly one local device (the round-robin pipeline accumulates
        per-tile outputs into a single buffer; multiple devices require
        a host-side buffer guarded by a lock);
      * ``eris.ovvv`` is an in-RAM ``np.ndarray`` (the HDF5-streamed
        path always accumulates on host inside ``consume_ovvv``);
      * ``eris.vovv`` is an in-RAM ``np.ndarray`` (same reason for the
        VOVV pipeline below).

    This helper is the single source of truth for that decision so the
    inline override in :func:`_update_amps` and the regression tests in
    ``pytc/solver/test/test_jax_xtc_ccsd.py`` stay in sync.
    """
    return (
        n_devices_local > 1
        or not isinstance(eris.ovvv, np.ndarray)
        or not isinstance(eris.vovv, np.ndarray)
    )


def _update_amps(cc, t1, t2, eris):
    """
    JAX-Recovered Restricted CCSD amplitude update with hybrid memory strategy.
    
    t1, t2: Input arrays (NumPy or JAX).
    eris: _ChemistsERIs object (NumPy/HDF5 backed).
    
    Returns: t1new, t2new (NumPy arrays).
    """
    logger.debug("Starting _update_amps (JAX-Optimized)")
    t_start = time.perf_counter()
    
    # 1. Cast inputs to JAX and move to GPU
    t1_jax = jnp.asarray(t1)
    t2_jax = jnp.asarray(t2)
    
    nocc, nvir = t1_jax.shape
    
    # Load small intermediates to GPU
    fock_jax = jnp.asarray(eris.fock)
    mo_e_o = jnp.diag(fock_jax)[:nocc]
    mo_e_v = jnp.diag(fock_jax)[nocc:] + cc.level_shift
    fov = fock_jax[:nocc, nocc:]
    
    # === Phase 1: Load ERIs needed for F-intermediates, accumulator base, T1 core ===
    # ovov is used heavily throughout (F-intermediates, Wvoov/Wvovo base, T1 core)
    eris_ovov = jnp.asarray(eris.ovov)
    # ovvo needed for Wvoov base and T1 core
    eris_ovvo = jnp.asarray(eris.ovvo)
    # oovv needed for Wvovo base, T1 core, and VOVV loop
    eris_oovv = jnp.asarray(eris.oovv)
    # ovoo needed for accumulator base, T1 core, and Loo
    eris_ovoo = jnp.asarray(eris.ovoo)
    # oooo, vooo, vovo deferred to Phase 6 (basic T2 terms)
    logger.debug(f"Phase 1 data transfer to GPU took {time.perf_counter()-t_start:.4f} s")
    t_kernels = time.perf_counter()
    
    # 2. Compute intermediates (Micro-Kernels)
    # These return JAX arrays on GPU
    Fov = _jax_cc_Fov(t1_jax, fock_jax, eris_ovov)
    Foo = _jax_cc_Foo(t1_jax, t2_jax, fock_jax, eris_ovov)
    Fvv = _jax_cc_Fvv(t1_jax, t2_jax, fock_jax, eris_ovov)
    logger.debug(f"Micro-kernels (Foo/Fov/Fvv) took {time.perf_counter()-t_kernels:.4f} s")
    
    Foo_shifted = Foo.at[jnp.diag_indices(nocc)].add(-mo_e_o)
    Fvv_shifted = Fvv.at[jnp.diag_indices(nvir)].add(-mo_e_v)
    # Keep unshifted Fvv for Lvv
    Fvv_unshifted = Fvv
    jax.block_until_ready([Foo_shifted, Fvv_shifted])
    logger.debug(f"Micro-kernels (Foo/Fov/Fvv) and shift took {time.perf_counter()-t_kernels:.4f} s")
    
    size_W = nvir * nocc * nocc * nvir * 8
    size_tmp = nocc * nvir * nocc * nocc * 8
    # Total needed for full GPU accumulators: 2*size_W + 2*size_tmp + Lvv (negligible)
    total_acc_mem = 2 * size_W + 2 * size_tmp
    
    # Use centralized GPU budget utility for accumulator decision
    _gpu_max = getattr(cc, 'gpu_max_memory', None)
    use_gpu_acc_flag, _ = estimate_blksize(
        nocc, nvir, 'acc_decision', gpu_max_memory_mb=_gpu_max)
    use_gpu_acc = bool(use_gpu_acc_flag)

    # Force host accumulation whenever the OVVV/VOVV pipeline below cannot
    # keep accumulators GPU-resident.  See ``_should_force_host_accumulators``
    # for the exact predicate (multi-GPU OR HDF5-backed ovvv OR HDF5-backed
    # vovv).  Previously the override only checked the multi-GPU case, so a
    # single-GPU run with HDF5-backed ovvv (large systems where ovvv was
    # spilled to disk regardless of device count) hit a misleading
    # AssertionError further down complaining that the multi-GPU check
    # "above" had failed to set use_gpu_acc=False.
    _n_devices_local = len(xtc_ccsd._solver_local_devices())
    if _should_force_host_accumulators(eris, _n_devices_local) and use_gpu_acc:
        logger.debug(
            "Forcing host-side accumulators for OVVV/VOVV "
            "(n_devices=%d, ovvv_in_ram=%s, vovv_in_ram=%s)",
            _n_devices_local,
            isinstance(eris.ovvv, np.ndarray),
            isinstance(eris.vovv, np.ndarray),
        )
        use_gpu_acc = False

    logger.debug(f"Accumulators need {total_acc_mem/1024**3:.2f} GB. Using GPU acc: {use_gpu_acc}")

    # Initialize accumulators with non-ovvv terms
    if use_gpu_acc:
        # GPU Path
        Lvv_acc = Fvv_unshifted - jnp.einsum('kc,ka->ac', fov, t1_jax)
        
        Wvoov_acc = eris_ovvo.transpose(2,0,3,1)
        Wvoov_acc -= jnp.einsum('kcli,la->akic', eris_ovoo, t1_jax)
        Wvoov_acc -= 0.5 * jnp.einsum('ldkc,ilda->akic', eris_ovov, t2_jax)
        Wvoov_acc -= 0.5 * jnp.einsum('lckd,ilad->akic', eris_ovov, t2_jax)
        Wvoov_acc -=       jnp.einsum('ldkc,id,la->akic', eris_ovov, t1_jax, t1_jax)
        Wvoov_acc +=       jnp.einsum('ldkc,ilad->akic', eris_ovov, t2_jax)
        
        Wvovo_acc = eris_oovv.transpose(2,0,3,1)
        Wvovo_acc -= jnp.einsum('lcki,la->akci', eris_ovoo, t1_jax)
        Wvovo_acc -= 0.5 * jnp.einsum('lckd,ilda->akci', eris_ovov, t2_jax)
        Wvovo_acc -=       jnp.einsum('lckd,id,la->akci', eris_ovov, t1_jax, t1_jax)
        
        tmp_a_acc = jnp.zeros((nocc, nvir, nocc, nocc))
        tmp_b_acc = jnp.zeros((nocc, nvir, nocc, nocc))
        
    else:
        # Host Path - Compute base terms on GPU then move to Host
        Lvv_acc = np.array(Fvv_unshifted - jnp.einsum('kc,ka->ac', fov, t1_jax))
        
        # Wvoov base
        tmp = eris_ovvo.transpose(2,0,3,1)
        tmp -= jnp.einsum('kcli,la->akic', eris_ovoo, t1_jax)
        tmp -= 0.5 * jnp.einsum('ldkc,ilda->akic', eris_ovov, t2_jax)
        tmp -= 0.5 * jnp.einsum('lckd,ilad->akic', eris_ovov, t2_jax)
        tmp -=       jnp.einsum('ldkc,id,la->akic', eris_ovov, t1_jax, t1_jax)
        tmp +=       jnp.einsum('ldkc,ilad->akic', eris_ovov, t2_jax)
        Wvoov_acc = np.array(tmp)
        del tmp
        
        # Wvovo base
        tmp = eris_oovv.transpose(2,0,3,1)
        tmp -= jnp.einsum('lcki,la->akci', eris_ovoo, t1_jax)
        tmp -= 0.5 * jnp.einsum('lckd,ilda->akci', eris_ovov, t2_jax)
        tmp -=       jnp.einsum('lckd,id,la->akci', eris_ovov, t1_jax, t1_jax)
        Wvovo_acc = np.array(tmp)
        del tmp
        
        tmp_a_acc = np.zeros((nocc, nvir, nocc, nocc))
        tmp_b_acc = np.zeros((nocc, nvir, nocc, nocc))
    
    # T1/T2 accumulators (Host)
    # We accumulate T1/T2 updates on host to save GPU memory and avoid in-place updates
    t1new_host = np.zeros_like(t1)
    t2new_host = np.zeros_like(t2)
    
    # --- T1 Core Updates ---
    t1new_jax = jnp.zeros_like(t1_jax)
    t1new_jax += -2 * jnp.einsum('kc,ka,ic->ia', fov, t1_jax, t1_jax)
    t1new_jax +=      jnp.einsum('ac,ic->ia', Fvv_shifted, t1_jax)
    t1new_jax -=      jnp.einsum('ki,ka->ia', Foo_shifted, t1_jax)
    t1new_jax +=  2 * jnp.einsum('kc,kica->ia', Fov, t2_jax)
    t1new_jax -=      jnp.einsum('kc,ikca->ia', Fov, t2_jax)
    t1new_jax +=      jnp.einsum('kc,ic,ka->ia', Fov, t1_jax, t1_jax)
    t1new_jax += fock_jax[nocc:, :nocc].T
    
    t1new_jax +=  2 * jnp.einsum('kcai,kc->ia', eris_ovvo, t1_jax)
    t1new_jax -=      jnp.einsum('kiac,kc->ia', eris_oovv, t1_jax)
    
    t1new_jax -=  2 * jnp.einsum('lcki,klac->ia', eris_ovoo, t2_jax)
    t1new_jax +=      jnp.einsum('kcli,klac->ia', eris_ovoo, t2_jax)
    t1new_jax -=  2 * jnp.einsum('lcki,lc,ka->ia', eris_ovoo, t1_jax, t1_jax)
    t1new_jax +=      jnp.einsum('kcli,lc,ka->ia', eris_ovoo, t1_jax, t1_jax)
    
    t_comp = time.perf_counter()
    t1new_jax.block_until_ready()
    t_comp_dur = time.perf_counter() - t_comp
    
    t_trans = time.perf_counter()
    t1new_host += np.asarray(t1new_jax) # Move T1 partial to host
    t_trans_dur = time.perf_counter() - t_trans
    
    del t1new_jax # Free GPU memory
    logger.debug(f"T1 core updates: Comp {t_comp_dur:.4f}s, Host accum {t_trans_dur:.4f}s")
    
    t_ovvv = time.perf_counter()
    
    # Tau: (O, O, V, V) needs to be on GPU for optimal contraction
    tau_jax = t2_jax + jnp.einsum('ia,jb->ijab', t1_jax, t1_jax)

    # --- OVVV Processing (Hybrid) ---
    _gpu_max = getattr(cc, 'gpu_max_memory', None)
    _host_max = getattr(cc, 'max_memory', None)
    blksize, _ = estimate_blksize(
        nocc, nvir, 'ovvv',
        gpu_max_memory_mb=_gpu_max,
        host_max_memory_mb=_host_max,
        include_accumulators=use_gpu_acc)
    
    if isinstance(eris.ovvv, np.ndarray):
        ovvv_jax = jnp.asarray(eris.ovvv)
        t1_upd, Lvv_p, Wvoov_p, Wvovo_p, tmp_a_p, tmp_b_p = kernel_process_ovvv_block(
            ovvv_jax, t1_jax, t2_jax, tau_jax
        )
        if use_gpu_acc:
             t1new_host += np.asarray(t1_upd)
             Lvv_acc += Lvv_p
             Wvoov_acc += Wvoov_p
             Wvovo_acc += Wvovo_p
             tmp_a_acc += tmp_a_p
             tmp_b_acc += tmp_b_p
        else:
             t1new_host += np.asarray(t1_upd)
             Lvv_acc += np.asarray(Lvv_p)
             Wvoov_acc += np.asarray(Wvoov_p)
             Wvovo_acc += np.asarray(Wvovo_p)
             tmp_a_acc += np.asarray(tmp_a_p)
             tmp_b_acc += np.asarray(tmp_b_p)
    else:
        # Multi-GPU OVVV block loop via _round_robin_pipeline.
        # Each tile reads `ovvv[:, :, p0:p1, :]` from HDF5, runs the 6-output
        # kernel on the chosen device, and accumulates the per-block outputs
        # into host-side buffers under a lock.  Non-ovvv slices (t1new,
        # Lvv_acc, Wvoov_acc, Wvovo_acc, tmp_a_acc, tmp_b_acc) are all
        # non-overlapping across different p0 ranges, but we use a single
        # lock for simplicity and future-proofing.
        #
        # HDF5 reads stay on the main dispatch thread (h5py isn't safe for
        # concurrent reads) but are one-block-ahead prefetched via async_read
        # so reads overlap GPU compute.
        from pytc.utils.prefetch import async_read, await_read

        if use_gpu_acc:
            # Should be unreachable: the early ``_force_host_acc`` override
            # forces ``use_gpu_acc = False`` whenever ``eris.ovvv`` is not
            # an in-RAM ndarray (i.e. exactly when this branch runs).
            raise AssertionError(
                "use_gpu_acc must be False for the HDF5-backed OVVV pipeline "
                "(in-RAM accumulators are not supported when streaming ovvv "
                "tiles from disk); the early _force_host_acc override should "
                "have set use_gpu_acc=False."
            )

        ovvv_devices = xtc_ccsd._solver_local_devices()

        # Cache t1, t2, tau on each device so kernel launches don't pay
        # a host→device transfer for these shared inputs every tile.
        ovvv_t1_by_dev  = xtc_ccsd.broadcast_to_devices(t1_jax,  ovvv_devices)
        ovvv_t2_by_dev  = xtc_ccsd.broadcast_to_devices(t2_jax,  ovvv_devices)
        ovvv_tau_by_dev = xtc_ccsd.broadcast_to_devices(tau_jax, ovvv_devices)

        ovvv_tile_specs = [
            (p0, min(p0 + blksize, nvir))
            for p0 in range(0, nvir, blksize)
        ]

        # 1-ahead HDF5 prefetch state, shared by closure (main thread only).
        _ovvv_prefetch = {'future': None, 'spec': None, 'next_idx': 0}
        if ovvv_tile_specs:
            first_spec = ovvv_tile_specs[0]
            _ovvv_prefetch['future'] = async_read(
                xtc_ccsd._get_slice, eris.ovvv, slice(first_spec[0], first_spec[1]), 2
            )
            _ovvv_prefetch['spec'] = first_spec
            _ovvv_prefetch['next_idx'] = 1

        ovvv_acc_lock = threading.Lock()

        def issue_ovvv(spec, device):
            p0, p1 = spec
            # Pick up the HDF5 read queued by the previous call.
            assert _ovvv_prefetch['spec'] == spec, (
                f"OVVV prefetch spec mismatch: expected {spec}, got {_ovvv_prefetch['spec']}"
            )
            blk_np = await_read(_ovvv_prefetch['future'])
            # Kick off HDF5 read for the NEXT tile so I/O overlaps GPU work.
            nidx = _ovvv_prefetch['next_idx']
            if nidx < len(ovvv_tile_specs):
                next_spec = ovvv_tile_specs[nidx]
                _ovvv_prefetch['future'] = async_read(
                    xtc_ccsd._get_slice, eris.ovvv, slice(next_spec[0], next_spec[1]), 2
                )
                _ovvv_prefetch['spec'] = next_spec
                _ovvv_prefetch['next_idx'] = nidx + 1
            else:
                _ovvv_prefetch['future'] = None
                _ovvv_prefetch['spec'] = None

            dev_key = device if device is not None else None
            device_ctx = (
                jax.default_device(device)
                if device is not None else contextlib.nullcontext()
            )
            with device_ctx:
                if device is not None:
                    blk_dev = jax.device_put(blk_np, device)
                else:
                    blk_dev = jnp.asarray(blk_np)
                return kernel_process_ovvv_block(
                    blk_dev,
                    ovvv_t1_by_dev[dev_key],
                    ovvv_t2_by_dev[dev_key],
                    ovvv_tau_by_dev[dev_key],
                )

        def consume_ovvv(spec, device, result, release_gpu_slot):
            p0, p1 = spec
            device_key = getattr(device, "id", "host")
            t0_comp_blk = time.perf_counter()
            t1_upd, Lvv_blk, Wvoov_blk, Wvovo_blk, tmp_a_blk, tmp_b_blk = result
            # Batch GPU→CPU readbacks before releasing the GPU slot.
            t1_upd_np = np.asarray(t1_upd)
            Lvv_blk_np = np.asarray(Lvv_blk)
            Wvoov_blk_np = np.asarray(Wvoov_blk)
            Wvovo_blk_np = np.asarray(Wvovo_blk)
            tmp_a_blk_np = np.asarray(tmp_a_blk)
            tmp_b_blk_np = np.asarray(tmp_b_blk)
            t1_readback = time.perf_counter()
            release_gpu_slot()  # GPU pipeline is now free for the next tile
            with ovvv_acc_lock:
                t1new_host[:, p0:p1] += t1_upd_np
                Lvv_acc[p0:p1, :] += Lvv_blk_np
                Wvoov_acc[p0:p1] += Wvoov_blk_np
                Wvovo_acc[p0:p1] += Wvovo_blk_np
                tmp_a_acc[:, p0:p1] += tmp_a_blk_np
                tmp_b_acc[:, p0:p1] += tmp_b_blk_np
            logger.debug(
                "OVVV block %d:%d on device %s: readback=%.4fs accum=%.4fs",
                p0, p1, device_key,
                t1_readback - t0_comp_blk,
                time.perf_counter() - t1_readback,
            )

        xtc_ccsd._round_robin_pipeline(
            ovvv_tile_specs, issue_ovvv, consume_ovvv, devices=ovvv_devices
        )

        # Drop per-device caches now that the OVVV phase is complete.
        del ovvv_t1_by_dev, ovvv_t2_by_dev, ovvv_tau_by_dev
    t_t2 = time.perf_counter()
            
    # --- T2 Updates ---
    t2new_jax = jnp.zeros_like(t2_jax) 
    
    if isinstance(eris.vovv, np.ndarray):
        vovv = jnp.asarray(eris.vovv)
        tmp2 = jnp.einsum('kibc,ka->abic', eris_oovv, -t1_jax)
        tmp2 += vovv.transpose(0, 2, 1, 3)
        tmp = jnp.einsum('abic,jc->ijab', tmp2, t1_jax)
        t2new_jax += (tmp + tmp.transpose(1, 0, 3, 2))
    else:
        blksize_t2, _ = estimate_blksize(
            nocc, nvir, 'vovv',
            gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
            host_max_memory_mb=getattr(cc, 'max_memory', None),
            include_accumulators=use_gpu_acc)

        # Multi-GPU VOVV block loop via _round_robin_pipeline.
        # Each tile reads a slab `vovv[p0:p1, :, :, :]` from HDF5, runs the
        # kernel on the chosen device, and accumulates the (nocc, nocc, p_len,
        # nvir) result into the non-overlapping slice `t2new_host[:, :, p0:p1, :]`.
        # HDF5 reads stay on the main dispatch thread (h5py is not safe for
        # concurrent reads) but are one-block-ahead prefetched via async_read
        # so they overlap GPU compute.
        from pytc.utils.prefetch import async_read, await_read

        vovv_devices = xtc_ccsd._solver_local_devices()

        # Cache t1 and eris_oovv on each device so issue_tile does not
        # pay a host→device transfer per tile.
        vovv_t1_by_dev   = xtc_ccsd.broadcast_to_devices(t1_jax,   vovv_devices)
        vovv_oovv_by_dev = xtc_ccsd.broadcast_to_devices(eris_oovv, vovv_devices)

        vovv_tile_specs = [
            (p0, min(p0 + blksize_t2, nvir))
            for p0 in range(0, nvir, blksize_t2)
        ]

        # 1-ahead async HDF5 prefetch, shared by closure (main thread only).
        _vovv_prefetch = {'future': None, 'spec': None, 'next_idx': 0}
        if vovv_tile_specs:
            first_spec = vovv_tile_specs[0]
            _vovv_prefetch['future'] = async_read(
                xtc_ccsd._get_slice, eris.vovv, slice(first_spec[0], first_spec[1]), 0
            )
            _vovv_prefetch['spec'] = first_spec
            _vovv_prefetch['next_idx'] = 1

        vovv_acc_lock = threading.Lock()

        def issue_vovv(spec, device):
            p0, p1 = spec
            # Pick up the block whose HDF5 read was kicked off last call.
            assert _vovv_prefetch['spec'] == spec, (
                f"VOVV prefetch spec mismatch: expected {spec}, got {_vovv_prefetch['spec']}"
            )
            blk_np = await_read(_vovv_prefetch['future'])
            # Kick off HDF5 read for the NEXT tile immediately, before the
            # (potentially slower) GPU dispatch, so I/O overlaps GPU compute.
            nidx = _vovv_prefetch['next_idx']
            if nidx < len(vovv_tile_specs):
                next_spec = vovv_tile_specs[nidx]
                _vovv_prefetch['future'] = async_read(
                    xtc_ccsd._get_slice, eris.vovv, slice(next_spec[0], next_spec[1]), 0
                )
                _vovv_prefetch['spec'] = next_spec
                _vovv_prefetch['next_idx'] = nidx + 1
            else:
                _vovv_prefetch['future'] = None
                _vovv_prefetch['spec'] = None

            dev_key = device if device is not None else None
            device_ctx = (
                jax.default_device(device)
                if device is not None else contextlib.nullcontext()
            )
            with device_ctx:
                if device is not None:
                    blk_dev = jax.device_put(blk_np, device)
                else:
                    blk_dev = jnp.asarray(blk_np)
                t1_dev = vovv_t1_by_dev[dev_key]
                oovv_dev = vovv_oovv_by_dev[dev_key]
                t1_slice_dev = t1_dev[:, p0:p1]
                term = kernel_process_vovv_block(
                    blk_dev, oovv_dev, t1_slice_dev, t1_dev
                )
            return term

        def consume_vovv(spec, device, term, release_gpu_slot):
            # Capture the device tag up front; ``device`` itself is no
            # longer needed once the kernel has produced ``term``.
            # (A prior refactor accidentally introduced ``del device``
            # while the format string below still referenced it, causing
            # ``UnboundLocalError`` on every VOVV tile.)
            device_key = getattr(device, "id", "host")
            p0, p1 = spec
            t0_trans = time.perf_counter()
            term_host = np.asarray(term)  # GPU→CPU readback
            release_gpu_slot()  # GPU pipeline is now free for the next tile
            with vovv_acc_lock:
                t2new_host[:, :, p0:p1, :] += term_host
            logger.debug(
                "VOVV block %d:%d on device %s accumulated in %.4fs",
                p0, p1, device_key,
                time.perf_counter() - t0_trans,
            )

        xtc_ccsd._round_robin_pipeline(
            vovv_tile_specs, issue_vovv, consume_vovv, devices=vovv_devices
        )

        # Drop per-device caches as soon as the VOVV phase is done.
        del vovv_t1_by_dev, vovv_oovv_by_dev

        t2new_host = t2new_host + t2new_host.transpose(1, 0, 3, 2)
        
    t2new_host += np.asarray(t2new_jax)
    del t2new_jax
    logger.debug(f"T2 core updates (VOVV) took {time.perf_counter()-t_t2:.4f} s")
    
    # === Phase 5: Free eris_oovv (no longer needed after VOVV) ===
    del eris_oovv
    
    # --- Output Preparation ---
    
    # Use GPU accumulators if available, else copy from host
    if not use_gpu_acc:
        Lvv_acc = jnp.asarray(Lvv_acc)
        Wvoov_acc = jnp.asarray(Wvoov_acc)
        Wvovo_acc = jnp.asarray(Wvovo_acc)
        tmp_a_acc = jnp.asarray(tmp_a_acc)
        tmp_b_acc = jnp.asarray(tmp_b_acc)
    
    # === Phase 6: Load deferred ERIs for basic T2 terms ===
    eris_oooo = jnp.asarray(eris.oooo)   # ~3 MB (tiny)
    eris_vooo = jnp.asarray(eris.vooo)   # ~77 MB
    eris_vovo = jnp.asarray(eris.vovo)   # ~2 GB
    # Reload eris_ovvo (needed for tmp2 below; was freed implicitly or still live)
    # It was loaded in Phase 1 and not freed, so it's still available.
    logger.debug(f"Phase 6 deferred ERI load done")
        
    # Basic T2 terms
    t2new_basic = jnp.zeros_like(t2_jax)
    tmp2 = jnp.einsum('kcai,jc->akij', eris_ovvo, t1_jax)
    tmp2 += eris_vooo.transpose(0, 2, 1, 3)
    tmp = jnp.einsum('akij,kb->ijab', tmp2, t1_jax)
    t2new_basic -= (tmp + tmp.transpose(1,0,3,2))
    t2new_basic += eris_vovo.transpose(1,3,0,2)
    
    Loo_jax = _jax_Loo(t1_jax, t2_jax, fock_jax, eris_ovov, eris_ovoo)
    Loo_jax = Loo_jax.at[jnp.diag_indices(nocc)].add(-mo_e_o)
    
    Woooo_jax = _jax_cc_Woooo(t1_jax, t2_jax, eris_oooo, eris_ovov, eris_ovoo)
    t2new_basic += jnp.einsum('klij,klab->ijab', Woooo_jax, tau_jax)
    
    t2new_host += np.asarray(t2new_basic)
    del t2new_basic
    
    t2new_tmp = -jnp.einsum('kb,kaij->ijab', t1_jax, tmp_a_acc)
    t2new_tmp -= jnp.einsum('ka,kbij->ijab', t1_jax, tmp_b_acc)
    t2new_host += np.asarray(t2new_tmp)
    del t2new_tmp
    
    Lvv_acc = Lvv_acc.at[jnp.diag_indices(nvir)].add(-mo_e_v)
    
    tmp = jnp.einsum('ac,ijcb->ijab', Lvv_acc, t2_jax)
    tmp += tmp.transpose(1,0,3,2)
    
    tmp2 = jnp.einsum('ki,kjab->ijab', Loo_jax, t2_jax)
    tmp -= (tmp2 + tmp2.transpose(1,0,3,2))
    
    tmp3 = 2*jnp.einsum('akic,kjcb->ijab', Wvoov_acc, t2_jax)
    tmp3 -=  jnp.einsum('akci,kjcb->ijab', Wvovo_acc, t2_jax)
    tmp += (tmp3 + tmp3.transpose(1,0,3,2))
    
    tmp4 = jnp.einsum('akic,kjbc->ijab', Wvoov_acc, t2_jax)
    tmp -= (tmp4 + tmp4.transpose(1,0,3,2))
    
    tmp5 = jnp.einsum('bkci,kjac->ijab', Wvovo_acc, t2_jax)
    tmp -= (tmp5 + tmp5.transpose(1,0,3,2))
    
    t2new_host += np.asarray(tmp)
    del tmp, tmp2, tmp3, tmp4, tmp5

    # === Phase 7: Free all ERIs before VVVV to maximize headroom ===
    del eris_ovov, eris_ovvo, eris_ovoo
    del eris_oooo, eris_vooo, eris_vovo
    del Wvoov_acc, Wvovo_acc, tmp_a_acc, tmp_b_acc, Lvv_acc
    del Loo_jax, Woooo_jax
    logger.debug(f"Phase 7: freed ERIs/accumulators before VVVV")

    # --- VVVV Contraction ---
    _contract_vvvv_t2(cc, tau_jax, eris, t2new_host)
    
    # Final Division
    eia = mo_e_o[:,None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    
    eia_np = np.asarray(eia)
    eijab_np = np.asarray(eijab)
    
    t1new_host /= eia_np
    t2new_host /= eijab_np
    
    # Release the X slice cache at the end of each iteration.
    from pytc.xtc import invalidate_X_cache
    invalidate_X_cache()

    logger.debug("_update_amps finished in %.3f s", time.perf_counter()-t_start)
    return t1new_host, t2new_host


def _contract_vvvv_t2(cc, t2_jax, eris, t2new_host):
    """
    Hybrid contraction of (vv|vv) with t2 (or tau).
    t2_jax: (nocc, nocc, nvir, nvir) on GPU.
    eris: NumPy/HDF5 backed.
    t2new_host: Accumulation target (NumPy).
    """
    import h5py
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params
    
    if eris.vvvv is not None:
        if isinstance(eris.vvvv, np.ndarray):
            # In-memory numpy array (legacy small-molecule path)
            vvvv_jax = jnp.asarray(eris.vvvv)
            term = jnp.einsum('acbd,ijcd->ijab', vvvv_jax, t2_jax)
            t2new_host += np.asarray(term)
            return

        if isinstance(eris.vvvv, h5py.Dataset):
            # HDF5-backed: read blocks from disk, transfer to GPU, contract
            blksize, _ = estimate_blksize(
                nocc, nvir, 'vvvv_gpu',
                gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
                host_max_memory_mb=getattr(cc, 'max_memory', None))
            logger.debug(f"VVVV contraction from disk: blksize={blksize}, "
                         f"n_blocks={(nvir+blksize-1)//blksize}")

            @jax.jit
            def contract_disk_kernel(t2, vvvv_block):
                return jnp.einsum('acbd,ijcd->ijab', vvvv_block, t2)

            from pytc.utils.prefetch import async_read, await_read
            pending = None
            pending_key = None

            for p0 in range(0, nvir, blksize):
                p1 = min(p0 + blksize, nvir)
                t0 = time.perf_counter()

                # Await prefetched block or read synchronously
                if pending is not None and pending_key == (p0, p1):
                    vvvv_blk_np = await_read(pending)
                    pending = None
                else:
                    vvvv_blk_np = np.asarray(eris.vvvv[p0:p1])

                vvvv_blk_jax = jnp.asarray(vvvv_blk_np)
                del vvvv_blk_np

                # Kick off NEXT block read in background
                next_p0 = p0 + blksize
                if next_p0 < nvir:
                    next_p1 = min(next_p0 + blksize, nvir)
                    pending = async_read(
                        lambda s=slice(next_p0, next_p1): np.asarray(eris.vvvv[s]))
                    pending_key = (next_p0, next_p1)

                term = contract_disk_kernel(t2_jax, vvvv_blk_jax)
                term.block_until_ready()
                t_comp = time.perf_counter() - t0

                t0_trans = time.perf_counter()
                t2new_host[:, :, p0:p1, :] += np.asarray(term)
                t_trans = time.perf_counter() - t0_trans
                logger.debug(f"VVVV disk block {p0}:{p1}: "
                             f"Comp {t_comp:.4f}s, Host accum {t_trans:.4f}s")
            return

    # --- On-the-fly path (eris.vvvv is None) ---
    # Determine naux for DF overhead estimation and n_fused for GPU scan workspace
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
         with_df = cc._scf.with_df
    _naux = None
    if with_df is not None and hasattr(eris, 'vvL'):
        _naux = eris.vvL.shape[1]
    _n_fused = None
    if hasattr(xtc_obj, 'phi_isdf') and xtc_obj.phi_isdf is not None:
        _n_fused = xtc_obj.phi_isdf.shape[1]

    p_blksize, r_blksize = resolve_vvvv_panel_block_sizes(
        nocc, nvir,
        p_block_size=getattr(cc, 'vvvv_p_block_size', None),
        r_block_size=getattr(cc, 'vvvv_r_block_size', None),
        gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
        naux=_naux,
        n_fused=_n_fused,
        include_eris=True,
        include_accumulators=False,
    )
    panel_size = p_blksize
    logger.debug(
        "VVVV on-the-fly contraction: p_blksize=%d, r_blksize=%d, n_p_blocks=%d, n_r_blocks=%d",
        p_blksize, r_blksize,
        (nvir + p_blksize - 1) // p_blksize,
        (nvir + r_blksize - 1) // r_blksize,
    )

    L_vv_full_host = None
    if with_df is not None:
         L_vv_full_host = lib.unpack_tril(eris.vvL[:], axis=0)

    @jax.jit
    def contract_tc_tile_kernel(t2, xtc_tile):
        return jnp.einsum('abcd,ijcd->ijab', xtc_tile.transpose(0, 2, 1, 3), t2)

    @jax.jit
    def contract_df_tile_kernel(t2, xtc_tile, L_p_tile, L_r_tile):
        std_tile = jnp.tensordot(L_p_tile, L_r_tile, axes=((2,), (2,)))
        vvvv_tile = xtc_tile + std_tile
        return jnp.einsum('abcd,ijcd->ijab', vvvv_tile.transpose(0, 2, 1, 3), t2)

    devices      = xtc_ccsd._solver_local_devices()
    t2_by_device = xtc_ccsd.broadcast_to_devices(t2_jax, devices)

    mo_v = None if with_df is not None else cc.mo_coeff[:, nocc:]
    tile_specs = []
    for p0 in range(0, nvir, p_blksize):
        p1 = min(p0 + p_blksize, nvir)
        for r0 in range(0, nvir, r_blksize):
            r1 = min(r0 + r_blksize, nvir)
            tile_specs.append((p0, p1, r0, r1))

    def issue_tile(spec, device):
        p0, p1, r0, r1 = spec
        p_len = p1 - p0
        r_len = r1 - r0
        ranges = (
            slice(nocc + p0, nocc + p1),
            slice(nocc, cc.nmo),
            slice(nocc + r0, nocc + r1),
            slice(nocc, cc.nmo),
        )
        t0 = time.perf_counter()
        vvvv_tile_jax = xtc_mod.compute_2b_tile(
            xtc_obj, jastrow_params, ranges, device=device, panel_size=panel_size
        )
        logger.debug(
            "get_2b (tile p[%d:%d] r[%d:%d] on device %s) issued in %.4f s",
            p0, p1, r0, r1, getattr(device, "id", "host"), time.perf_counter() - t0,
        )
        dev_key = device if device is not None else None
        device_ctx = jax.default_device(device) if device is not None else contextlib.nullcontext()
        with device_ctx:
            if with_df is not None:
                # Slice L_vv_full_host directly.  On full tiles (p_len ==
                # panel_size) this is a zero-copy view; only the last tile
                # needs padding.  Avoids per-tile np.zeros(panel_size, nvir,
                # naux) allocations that dominated issue_tile overhead.
                L_p_raw = L_vv_full_host[p0:p1]   # (p_len, nvir, naux)
                L_r_raw = L_vv_full_host[r0:r1]   # (r_len, nvir, naux)
                if p_len < panel_size:
                    pad = np.zeros((panel_size - p_len, nvir, L_vv_full_host.shape[2]),
                                   dtype=L_vv_full_host.dtype)
                    L_p_raw = np.concatenate([L_p_raw, pad], axis=0)
                if r_len < panel_size:
                    pad = np.zeros((panel_size - r_len, nvir, L_vv_full_host.shape[2]),
                                   dtype=L_vv_full_host.dtype)
                    L_r_raw = np.concatenate([L_r_raw, pad], axis=0)
                if device is not None:
                    L_p_tile_jax = jax.device_put(L_p_raw, device)
                    L_r_tile_jax = jax.device_put(L_r_raw, device)
                else:
                    L_p_tile_jax = jnp.asarray(L_p_raw)
                    L_r_tile_jax = jnp.asarray(L_r_raw)
                term = contract_df_tile_kernel(
                    t2_by_device[dev_key], vvvv_tile_jax,
                    L_p_tile_jax, L_r_tile_jax
                )
            else:
                std_tile = ao2mo.general(
                    cc.mol,
                    (mo_v[:, p0:p1], mo_v, mo_v[:, r0:r1], mo_v),
                    compact=False,
                )
                std_tile_pad = np.zeros((panel_size, nvir, panel_size, nvir))
                std_tile_pad[:p_len, :, :r_len, :] = std_tile.reshape(p_len, nvir, r_len, nvir)
                if device is not None:
                    std_tile_jax = jax.device_put(std_tile_pad, device)
                else:
                    std_tile_jax = jnp.asarray(std_tile_pad)
                term = contract_tc_tile_kernel(
                    t2_by_device[dev_key], vvvv_tile_jax + std_tile_jax
                )
        return term

    def consume_tile(spec, device, term, release_gpu_slot):
        p0, p1, r0, r1 = spec
        p_len = p1 - p0
        r_len = r1 - r0
        t0_trans = time.perf_counter()
        term_host = np.asarray(term)[:, :, :p_len, :r_len]
        release_gpu_slot()  # GPU pipeline is now free to issue the next tile
        t2new_host[:, :, p0:p1, r0:r1] += term_host
        logger.debug(
            "VVVV tile p[%d:%d] r[%d:%d] on device %s accumulated in %.4fs",
            p0, p1, r0, r1, getattr(device, "id", "host"), time.perf_counter() - t0_trans,
        )

    xtc_ccsd._round_robin_pipeline(tile_specs, issue_tile, consume_tile, devices=devices,
                                   device_key=lambda spec: spec[0])

    if L_vv_full_host is not None:
        del L_vv_full_host
