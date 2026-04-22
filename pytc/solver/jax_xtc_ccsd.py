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

    Implementation notes
    --------------------
    This kernel has been rewritten three times in response to production
    XLA failures on pCV5Z-class (≈30 GB f64 ``ovvv_blk``) inputs:

    * v1 (textbook): paired ``'kdac'``/``'kcad'`` einsums against the
      same ``ovvv_blk``.  XLA fused an axis-1↔3 transpose of the 30 GB
      tile into several downstream contractions (``input_transpose_fusion``
      HLOs) and the autotuner aborted with ``NOT_FOUND: No valid config
      found!`` on shapes like ``f64[1179, 3070116]``.

    * v2 (commit ``eeb39b2``): precomputed ``ovvv_swap`` and
      ``ovvv_t1sym = 2*ovvv_blk - ovvv_swap`` inside this ``@jax.jit``.
      XLA's common-subexpression / fusion passes still re-fused the
      transpose into the downstream GEMMs that contract ``(c, d)``
      against ``tau`` in ``tmp_a`` and ``tmp_b``.  The natural GEMM plan
      for ``einsum('kdac,ijcd->kaij', ovvv_blk, tau)`` wants
      ``(k*a, d*c) @ (d*c, i*j)``, which requires reshape-after-transpose
      of ``ovvv_blk`` into layout ``(k, a, c, d)``.  At
      ``(nocc=21, nvir=1179, blk=124)`` that scratch is 27 GB, and the
      BFC allocator could not fit it alongside ``ovvv_blk`` +
      ``ovvv_swap`` + ``ovvv_t1sym`` + t1/t2/tau → OOM → autotuner
      reports "NOT_FOUND" (no config can allocate its scratch).

    * v3 (current): two targeted changes.

      1. Materialise ``ovvv_swap`` with an ``optimization_barrier`` so
         XLA cannot re-fuse the transpose into any downstream tile
         config.  The single standalone transpose HLO is trivially
         handled by the autotuner.
      2. Drop the separate ``ovvv_t1sym`` intermediate (saving a third
         30 GB tensor) and split the ``2*native - swap`` pattern into
         two explicit einsums at each use site.
      3. Precompute ``tau_swap = transpose(tau, (0, 1, 3, 2))`` (≈5 GB,
         also barrier-wrapped) so the contraction in ``tmp_a`` and
         ``tmp_b`` is written as ``'...dc'`` on the RHS — matching the
         ``(d, c)`` order on the LHS and removing the ordering mismatch
         that was pushing XLA toward the failing transpose plan.

    Memory budget inside this kernel at pCV5Z scale:
    ``ovvv_blk (29 GB) + ovvv_swap (29 GB) + tau (5 GB) + tau_swap (5 GB)
    + small = ~68 GB``, leaving comfortable headroom on a 80 GB H100 for
    per-einsum GEMM scratch.

    The output is numerically equivalent to v1/v2 up to floating-point
    reassociation; see ``pytc/test/test_kernel_process_ovvv_block.py``
    for the regression test.
    """
    # Pre-materialise the axis-1↔3 swap of ovvv_blk once, with a barrier
    # to prevent XLA from re-fusing the transpose into downstream GEMMs.
    ovvv_swap = jax.lax.optimization_barrier(
        jnp.transpose(ovvv_blk, (0, 3, 2, 1))
    )
    # Pre-materialise tau with its last two axes swapped.  Writing the
    # tmp_a / tmp_b contractions as ``'...dc'`` against this tensor
    # gives XLA a GEMM plan whose natural layout does not require a
    # reshape-after-transpose of the 30 GB ``ovvv_blk``.
    tau_swap = jax.lax.optimization_barrier(
        jnp.transpose(tau, (0, 1, 3, 2))
    )

    # 1. t1_upd (ia): 2*native - swap split into two einsums rather than
    #    via a pre-computed ``ovvv_t1sym`` (which would cost a third 30
    #    GB resident tensor).
    t1_upd  = 2 * jnp.einsum('kdac,ikcd->ia',    ovvv_blk,  t2)
    t1_upd -=     jnp.einsum('kdac,ikcd->ia',    ovvv_swap, t2)
    t1_upd += 2 * jnp.einsum('kdac,kd,ic->ia',   ovvv_blk,  t1, t1)
    t1_upd -=     jnp.einsum('kdac,kd,ic->ia',   ovvv_swap, t1, t1)

    # 2. Lvv_blk (ac): same 2*native - swap split.
    Lvv_blk  = 2 * jnp.einsum('kdac,kd->ac',     ovvv_blk,  t1)
    Lvv_blk -=     jnp.einsum('kdac,kd->ac',     ovvv_swap, t1)

    # 3. Wvoov_blk (akic): pure swap branch.
    Wvoov_blk = jnp.einsum('kdac,id->akic',      ovvv_swap, t1)

    # 4. Wvovo_blk (akci): native branch.
    Wvovo_blk = jnp.einsum('kdac,id->akci',      ovvv_blk,  t1)

    # 5. tmp_a_blk (kaij): rewritten against ``tau_swap`` so the RHS
    #    contraction axes appear as ``'ijdc'`` — aligned with the LHS
    #    ``'kdac'``.  Numerically identical to
    #    ``einsum('kdac,ijcd->kaij', ovvv_blk, tau)`` because
    #    ``tau_swap[i, j, d, c] == tau[i, j, c, d]``.
    tmp_a_blk = jnp.einsum('kdac,ijdc->kaij',    ovvv_blk,  tau_swap)

    # 6. tmp_b_blk (kbij): swap branch, same tau_swap rewrite.
    tmp_b_blk = jnp.einsum('kdbc,ijdc->kbij',    ovvv_swap, tau_swap)

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

        # ``_round_robin_pipeline`` runs one issue thread per device.
        # Each thread enters ``issue_ovvv`` concurrently for tiles
        # assigned to its device, so the previous "single shared
        # prefetch dict" pattern would race.  Instead, partition tiles
        # up front and give each device its own prefetch chain — each
        # issue thread only reads/writes its own state, no locks needed.
        ovvv_tiles_by_dev = xtc_ccsd.partition_round_robin(
            ovvv_tile_specs, ovvv_devices,
        )
        # Per-device 1-ahead HDF5 prefetch.  ``_specs`` is the device's
        # private tile sub-list; ``_next_idx`` is the index within that
        # sub-list (NOT within the global ``ovvv_tile_specs``).
        _ovvv_prefetch_by_dev = {}
        for _d in ovvv_devices:
            _specs = ovvv_tiles_by_dev[_d]
            _pf = {"specs": _specs, "future": None, "spec": None, "next_idx": 0}
            if _specs:
                _first = _specs[0]
                _pf["future"] = async_read(
                    xtc_ccsd._get_slice, eris.ovvv,
                    slice(_first[0], _first[1]), 2,
                )
                _pf["spec"] = _first
                _pf["next_idx"] = 1
            _ovvv_prefetch_by_dev[_d] = _pf

        ovvv_acc_lock = threading.Lock()

        # Per-stage wall-time accumulator for the OVVV issue path.  The
        # parent ``_round_robin_pipeline`` reports total ``issue_s`` but
        # that is a single number — we need to decompose it to see which
        # sub-step (HDF5 await, H→D copy, JAX kernel dispatch) is
        # actually blocking the main thread each tile.  Two issue
        # threads now write to it concurrently, so guard with a lock
        # (held only briefly to apply per-tile deltas).
        ovvv_issue_stats = {
            "n_calls":        0,
            "await_read_s":   0.0,
            "prefetch_next_s": 0.0,
            "device_put_s":   0.0,
            "kernel_s":       0.0,
            "total_s":        0.0,
        }
        ovvv_stats_lock = threading.Lock()

        def issue_ovvv(spec, device):
            p0, p1 = spec
            _t_total0 = time.perf_counter()
            pf = _ovvv_prefetch_by_dev[device]
            # Pick up the HDF5 read queued by the previous call FOR THIS DEVICE.
            assert pf['spec'] == spec, (
                f"OVVV prefetch spec mismatch on device {getattr(device, 'id', 'host')}: "
                f"expected {spec}, got {pf['spec']}"
            )
            _t_ar0 = time.perf_counter()
            blk_np = await_read(pf['future'])
            _t_ar1 = time.perf_counter()
            # Kick off HDF5 read for the NEXT tile assigned to THIS device
            # so I/O overlaps GPU work on the same device.
            nidx = pf['next_idx']
            if nidx < len(pf['specs']):
                next_spec = pf['specs'][nidx]
                pf['future'] = async_read(
                    xtc_ccsd._get_slice, eris.ovvv,
                    slice(next_spec[0], next_spec[1]), 2,
                )
                pf['spec'] = next_spec
                pf['next_idx'] = nidx + 1
            else:
                pf['future'] = None
                pf['spec'] = None
            _t_pf1 = time.perf_counter()

            dev_key = device if device is not None else None
            device_ctx = (
                jax.default_device(device)
                if device is not None else contextlib.nullcontext()
            )
            with device_ctx:
                _t_dp0 = time.perf_counter()
                if device is not None:
                    blk_dev = jax.device_put(blk_np, device)
                else:
                    blk_dev = jnp.asarray(blk_np)
                _t_dp1 = time.perf_counter()
                _t_k0 = time.perf_counter()
                result = kernel_process_ovvv_block(
                    blk_dev,
                    ovvv_t1_by_dev[dev_key],
                    ovvv_t2_by_dev[dev_key],
                    ovvv_tau_by_dev[dev_key],
                )
                _t_k1 = time.perf_counter()
            _t_total1 = time.perf_counter()

            with ovvv_stats_lock:
                ovvv_issue_stats["n_calls"]         += 1
                ovvv_issue_stats["await_read_s"]    += (_t_ar1  - _t_ar0)
                ovvv_issue_stats["prefetch_next_s"] += (_t_pf1  - _t_ar1)
                ovvv_issue_stats["device_put_s"]    += (_t_dp1  - _t_dp0)
                ovvv_issue_stats["kernel_s"]        += (_t_k1   - _t_k0)
                ovvv_issue_stats["total_s"]         += (_t_total1 - _t_total0)
            return result

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

        # --- issue-stage decomposition summary -----------------------
        # Answers the follow-on question the parent pipeline's
        # ``issue=X.Xs`` line cannot: WHICH stage inside issue_ovvv is
        # blocking the main dispatch thread.  Look at which fraction
        # dominates ``total_s``:
        #   * await_read  ≫ others → HDF5 prefetch isn't keeping up.
        #   * device_put  ≫ others → H→D copy is sync (pinned memory /
        #                             memory pressure on the target GPU).
        #   * kernel      ≫ others → the JIT call is materialising
        #                             instead of returning a future.
        _n = ovvv_issue_stats["n_calls"]
        if _n > 0:
            _tot   = ovvv_issue_stats["total_s"]
            _other = _tot - (ovvv_issue_stats["await_read_s"]
                             + ovvv_issue_stats["prefetch_next_s"]
                             + ovvv_issue_stats["device_put_s"]
                             + ovvv_issue_stats["kernel_s"])
            logger.info(
                "[issue-ovvv] n_calls=%d  total=%.2fs  "
                "await_read=%.2fs  prefetch_next=%.2fs  "
                "device_put=%.2fs  kernel_dispatch=%.2fs  (other=%.2fs)",
                _n, _tot,
                ovvv_issue_stats["await_read_s"],
                ovvv_issue_stats["prefetch_next_s"],
                ovvv_issue_stats["device_put_s"],
                ovvv_issue_stats["kernel_s"],
                _other,
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

        # Per-device prefetch chain — see the parallel OVVV block above
        # for why we partition the prefetch state per device rather than
        # sharing a single dict across the parallel issue threads.
        vovv_tiles_by_dev = xtc_ccsd.partition_round_robin(
            vovv_tile_specs, vovv_devices,
        )
        _vovv_prefetch_by_dev = {}
        for _d in vovv_devices:
            _specs = vovv_tiles_by_dev[_d]
            _pf = {"specs": _specs, "future": None, "spec": None, "next_idx": 0}
            if _specs:
                _first = _specs[0]
                _pf["future"] = async_read(
                    xtc_ccsd._get_slice, eris.vovv,
                    slice(_first[0], _first[1]), 0,
                )
                _pf["spec"] = _first
                _pf["next_idx"] = 1
            _vovv_prefetch_by_dev[_d] = _pf

        vovv_acc_lock = threading.Lock()

        # Per-stage issue-time decomposition for VOVV, mirroring the
        # one above for OVVV.  See the comment next to ``ovvv_issue_stats``
        # for the intended diagnostic use.  Locked because two issue
        # threads update concurrently.
        vovv_issue_stats = {
            "n_calls":         0,
            "await_read_s":    0.0,
            "prefetch_next_s": 0.0,
            "device_put_s":    0.0,
            "kernel_s":        0.0,
            "total_s":         0.0,
        }
        vovv_stats_lock = threading.Lock()

        def issue_vovv(spec, device):
            p0, p1 = spec
            _t_total0 = time.perf_counter()
            pf = _vovv_prefetch_by_dev[device]
            # Pick up the block whose HDF5 read was kicked off last call
            # FOR THIS DEVICE.
            assert pf['spec'] == spec, (
                f"VOVV prefetch spec mismatch on device {getattr(device, 'id', 'host')}: "
                f"expected {spec}, got {pf['spec']}"
            )
            _t_ar0 = time.perf_counter()
            blk_np = await_read(pf['future'])
            _t_ar1 = time.perf_counter()
            # Kick off HDF5 read for THIS device's next tile.
            nidx = pf['next_idx']
            if nidx < len(pf['specs']):
                next_spec = pf['specs'][nidx]
                pf['future'] = async_read(
                    xtc_ccsd._get_slice, eris.vovv,
                    slice(next_spec[0], next_spec[1]), 0,
                )
                pf['spec'] = next_spec
                pf['next_idx'] = nidx + 1
            else:
                pf['future'] = None
                pf['spec'] = None
            _t_pf1 = time.perf_counter()

            dev_key = device if device is not None else None
            device_ctx = (
                jax.default_device(device)
                if device is not None else contextlib.nullcontext()
            )
            with device_ctx:
                _t_dp0 = time.perf_counter()
                if device is not None:
                    blk_dev = jax.device_put(blk_np, device)
                else:
                    blk_dev = jnp.asarray(blk_np)
                _t_dp1 = time.perf_counter()
                t1_dev = vovv_t1_by_dev[dev_key]
                oovv_dev = vovv_oovv_by_dev[dev_key]
                t1_slice_dev = t1_dev[:, p0:p1]
                _t_k0 = time.perf_counter()
                term = kernel_process_vovv_block(
                    blk_dev, oovv_dev, t1_slice_dev, t1_dev
                )
                _t_k1 = time.perf_counter()
            _t_total1 = time.perf_counter()

            with vovv_stats_lock:
                vovv_issue_stats["n_calls"]         += 1
                vovv_issue_stats["await_read_s"]    += (_t_ar1  - _t_ar0)
                vovv_issue_stats["prefetch_next_s"] += (_t_pf1  - _t_ar1)
                vovv_issue_stats["device_put_s"]    += (_t_dp1  - _t_dp0)
                vovv_issue_stats["kernel_s"]        += (_t_k1   - _t_k0)
                vovv_issue_stats["total_s"]         += (_t_total1 - _t_total0)
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

        # Issue-stage decomposition summary (see OVVV block above for
        # the diagnostic interpretation).
        _n = vovv_issue_stats["n_calls"]
        if _n > 0:
            _tot   = vovv_issue_stats["total_s"]
            _other = _tot - (vovv_issue_stats["await_read_s"]
                             + vovv_issue_stats["prefetch_next_s"]
                             + vovv_issue_stats["device_put_s"]
                             + vovv_issue_stats["kernel_s"])
            logger.info(
                "[issue-vovv] n_calls=%d  total=%.2fs  "
                "await_read=%.2fs  prefetch_next=%.2fs  "
                "device_put=%.2fs  kernel_dispatch=%.2fs  (other=%.2fs)",
                _n, _tot,
                vovv_issue_stats["await_read_s"],
                vovv_issue_stats["prefetch_next_s"],
                vovv_issue_stats["device_put_s"],
                vovv_issue_stats["kernel_s"],
                _other,
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
                # ``vvvv_block`` is laid out (a, c, b, d); the contraction
                # axes (c, d) are at positions 1 and 3 (non-adjacent) on
                # the LHS while tau/t2 has them at 2, 3 (adjacent).  XLA's
                # natural GEMM plan for this mismatch wants to materialise
                # a ~27 GB reshape-after-transpose of vvvv_block, which
                # blows the BFC allocator at cc-pCV5Z scale.  We precompute
                # ``t2_swap`` with the last two axes swapped (≈5 GB, one-
                # shot, barrier-wrapped so XLA cannot re-fuse it) and
                # rewrite the contraction as ``'acbd,ijdc->ijab'``, which
                # aligns the RHS contraction axes with the LHS and lets
                # XLA pick a plan that keeps vvvv_block in its original
                # storage layout.  See the parallel rewrite in
                # ``kernel_process_ovvv_block`` for the full argument.
                t2_swap = jax.lax.optimization_barrier(
                    jnp.transpose(t2, (0, 1, 3, 2))
                )
                return jnp.einsum('acbd,ijdc->ijab', vvvv_block, t2_swap)

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

    # VVVV on-the-fly contraction tile kernels.
    #
    # The raw VVVV tile (from compute_2b_tile / the DF outer product)
    # lives in layout ``(p=a, q=c, r=b, s=d)`` with the two virtual-
    # contraction axes ``(c, d)`` at positions 1 and 3 — non-adjacent.
    # Writing the einsum directly as ``'acbd,ijcd->ijab'`` would put
    # XLA into the same failure mode as ``kernel_process_ovvv_block``:
    # its natural GEMM plan materialises a reshape-after-transpose of
    # the tile as scratch.  At pCV5Z-class tile sizes that scratch is
    # big enough to blow the BFC allocator and blow the autotuner
    # (``NOT_FOUND: No valid config found!``).
    #
    # The ``.transpose(0, 2, 1, 3)`` before the einsum pre-arranges the
    # tile into ``(a, b, c, d)`` so the contraction axes ``(c, d)`` are
    # adjacent and at the end on both operands — a clean GEMM pattern.
    # However XLA's fusion passes can re-fold that transpose into the
    # einsum's tile config, negating the benefit.  We wrap the
    # pre-transpose in ``jax.lax.optimization_barrier`` to pin it as a
    # standalone HLO, guaranteeing that the downstream einsum sees an
    # already-permuted input and compiles to a straight GEMM — the same
    # defensive pattern we apply to ``ovvv_swap`` / ``tau_swap`` in
    # ``kernel_process_ovvv_block``.
    @jax.jit
    def contract_tc_tile_kernel(t2, xtc_tile):
        xtc_tile_p = jax.lax.optimization_barrier(
            xtc_tile.transpose(0, 2, 1, 3)
        )
        return jnp.einsum('abcd,ijcd->ijab', xtc_tile_p, t2)

    @jax.jit
    def contract_df_tile_kernel(t2, xtc_tile, L_p_tile, L_r_tile):
        std_tile = jnp.tensordot(L_p_tile, L_r_tile, axes=((2,), (2,)))
        vvvv_tile = xtc_tile + std_tile
        vvvv_tile_p = jax.lax.optimization_barrier(
            vvvv_tile.transpose(0, 2, 1, 3)
        )
        return jnp.einsum('abcd,ijcd->ijab', vvvv_tile_p, t2)

    devices      = xtc_ccsd._solver_local_devices()
    t2_by_device = xtc_ccsd.broadcast_to_devices(t2_jax, devices)

    mo_v = None if with_df is not None else cc.mo_coeff[:, nocc:]
    tile_specs = []
    for p0 in range(0, nvir, p_blksize):
        p1 = min(p0 + p_blksize, nvir)
        for r0 in range(0, nvir, r_blksize):
            r1 = min(r0 + r_blksize, nvir)
            tile_specs.append((p0, p1, r0, r1))

    # Fine-grained per-stage timing.  Every line below is a wall-clock
    # marker that can be cross-referenced against ``nvidia-smi dmon -s pu
    # -d 1 -o DT`` (which writes ``YYYYMMDD HH:MM:SS`` per row, matching
    # the Python logger's ``%(asctime)s``).  Use:
    #
    #   grep '\[VVVV-(issue|consume) d[01]\]' ben_*.out
    #
    # to extract just the markers, then overlay with the dmon log to see
    # which sub-stage of which device's tile was running at every second.
    # Each marker is tagged ``[VVVV-issue dN]`` or ``[VVVV-consume dN]``
    # so the device producing it is unambiguous.

    def issue_tile(spec, device):
        p0, p1, r0, r1 = spec
        p_len = p1 - p0
        r_len = r1 - r0
        dev_id = getattr(device, "id", "host")
        ranges = (
            slice(nocc + p0, nocc + p1),
            slice(nocc, cc.nmo),
            slice(nocc + r0, nocc + r1),
            slice(nocc, cc.nmo),
        )
        logger.debug(
            "[VVVV-issue d%s] BEGIN tile p[%d:%d] r[%d:%d]",
            dev_id, p0, p1, r0, r1,
        )
        t_xtc_0 = time.perf_counter()
        vvvv_tile_jax = xtc_mod.compute_2b_tile(
            xtc_obj, jastrow_params, ranges, device=device, panel_size=panel_size
        )
        t_xtc_1 = time.perf_counter()
        logger.debug(
            "[VVVV-issue d%s] xtc_2b_tile RETURN +%.4fs (returned future: %s)",
            dev_id, t_xtc_1 - t_xtc_0,
            "blocking-call" if (t_xtc_1 - t_xtc_0) > 0.5 else "async",
        )
        dev_key = device if device is not None else None
        device_ctx = jax.default_device(device) if device is not None else contextlib.nullcontext()
        with device_ctx:
            if with_df is not None:
                # Slice L_vv_full_host directly.  On full tiles (p_len ==
                # panel_size) this is a zero-copy view; only the last tile
                # needs padding.  Avoids per-tile np.zeros(panel_size, nvir,
                # naux) allocations that dominated issue_tile overhead.
                t_lpr_0 = time.perf_counter()
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
                t_lpr_1 = time.perf_counter()
                logger.debug(
                    "[VVVV-issue d%s] L_p/L_r device_put RETURN +%.4fs",
                    dev_id, t_lpr_1 - t_lpr_0,
                )
                t_einsum_0 = time.perf_counter()
                term = contract_df_tile_kernel(
                    t2_by_device[dev_key], vvvv_tile_jax,
                    L_p_tile_jax, L_r_tile_jax
                )
                t_einsum_1 = time.perf_counter()
                logger.debug(
                    "[VVVV-issue d%s] contract_df_tile RETURN +%.4fs (returned future: %s)",
                    dev_id, t_einsum_1 - t_einsum_0,
                    "blocking-call" if (t_einsum_1 - t_einsum_0) > 0.5 else "async",
                )
            else:
                t_std_0 = time.perf_counter()
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
                t_std_1 = time.perf_counter()
                logger.debug(
                    "[VVVV-issue d%s] std_tile ao2mo+device_put RETURN +%.4fs",
                    dev_id, t_std_1 - t_std_0,
                )
                t_einsum_0 = time.perf_counter()
                term = contract_tc_tile_kernel(
                    t2_by_device[dev_key], vvvv_tile_jax + std_tile_jax
                )
                t_einsum_1 = time.perf_counter()
                logger.debug(
                    "[VVVV-issue d%s] contract_tc_tile RETURN +%.4fs (returned future: %s)",
                    dev_id, t_einsum_1 - t_einsum_0,
                    "blocking-call" if (t_einsum_1 - t_einsum_0) > 0.5 else "async",
                )
        logger.debug(
            "[VVVV-issue d%s] END tile p[%d:%d] r[%d:%d]",
            dev_id, p0, p1, r0, r1,
        )
        # Keep the legacy summary log so existing log-grep tools still work.
        logger.debug(
            "get_2b (tile p[%d:%d] r[%d:%d] on device %s) issued in %.4f s",
            p0, p1, r0, r1, dev_id, t_xtc_1 - t_xtc_0,
        )
        return term

    def consume_tile(spec, device, term, release_gpu_slot):
        p0, p1, r0, r1 = spec
        p_len = p1 - p0
        r_len = r1 - r0
        dev_id = getattr(device, "id", "host")
        logger.debug(
            "[VVVV-consume d%s] BEGIN tile p[%d:%d] r[%d:%d] (waiting on np.asarray)",
            dev_id, p0, p1, r0, r1,
        )
        t_asarray_0 = time.perf_counter()
        # ``np.asarray(term)`` blocks until the GPU has finished every op
        # that produced ``term``.  Wall time here = time waiting for the
        # tile's K1 / K3 / ΔU / contract kernels on this device to drain
        # PLUS the GPU→CPU DMA of the result.  This is the most direct
        # measurement of how long this device's tile actually took on the
        # silicon, independent of what ``issue_tile`` reported.
        term_host = np.asarray(term)[:, :, :p_len, :r_len]
        t_asarray_1 = time.perf_counter()
        logger.debug(
            "[VVVV-consume d%s] np.asarray RETURN +%.4fs (GPU done + DMA copied)",
            dev_id, t_asarray_1 - t_asarray_0,
        )
        release_gpu_slot()  # GPU pipeline is now free to issue the next tile
        t_accum_0 = time.perf_counter()
        t2new_host[:, :, p0:p1, r0:r1] += term_host
        t_accum_1 = time.perf_counter()
        logger.debug(
            "[VVVV-consume d%s] host accum RETURN +%.4fs",
            dev_id, t_accum_1 - t_accum_0,
        )
        logger.debug(
            "[VVVV-consume d%s] END tile p[%d:%d] r[%d:%d] total=%.4fs",
            dev_id, p0, p1, r0, r1, t_accum_1 - t_asarray_0,
        )
        # Keep the legacy summary log so existing log-grep tools still work.
        logger.debug(
            "VVVV tile p[%d:%d] r[%d:%d] on device %s accumulated in %.4fs",
            p0, p1, r0, r1, dev_id, t_accum_1 - t_asarray_0,
        )

    # Tile-id round-robin across local devices.  We intentionally do NOT
    # pass ``device_key=lambda spec: spec[0]`` (p-block locality) here:
    # the Apr-7 ``_get_isdf_device_cache`` refactor made phi_isdf /
    # grad_phi_isdf / K1 / K3 / D fully device-resident, so there is no
    # longer any locality benefit from keeping r-tiles of a given p on
    # one device.  With ``device_key`` on, ``_round_robin_pipeline``
    # would assign the first-seen p-block to device 0, the second to
    # device 1, etc.  At small ``p_blksize`` that is fine, but at the
    # degenerate ``p_blksize=1`` the scheduler must drain all ``n_r``
    # tiles of ``p=0`` before dispatching ``p=1`` to device 1, leaving
    # every other device idle for roughly ``n_r * per_tile_gpu_time``
    # (observed on QZ: device 0 finished 1306 tiles while device 1 had
    # only started 64).  Default tile-id round-robin avoids that:
    # ``(p=0,r=0)→d0, (p=0,r=1)→d1, (p=0,r=2)→d0, ...`` so every device
    # gets work on tile 0/1 and the staggering is bounded by a single
    # tile's latency.
    xtc_ccsd._round_robin_pipeline(tile_specs, issue_tile, consume_tile,
                                   devices=devices)

    if L_vv_full_host is not None:
        del L_vv_full_host
