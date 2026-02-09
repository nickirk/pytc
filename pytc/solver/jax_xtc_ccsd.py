import logging
import time
import numpy as np
from functools import reduce
import jax
import jax.numpy as jnp
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

from pytc.solver import xtc_ccsd
from pytc.autodiff.xtc import XTC, ISDFXTC

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
    JIT-compiled kernel for processing ovvv block contributions to t1new, Lvv, Wvoov, Wvovo, tmp_a, tmp_b.
    ovvv_blk: (nocc, nvir, blk, nvir) - sliced along axis 2 ('a')
    """
    # 1. Update t1new (ia) partial -> accumulating into (ia)
    # 2*einsum('kdac,ikcd->ia') - einsum('kcad,ikcd->ia')
    t1_upd =  2 * jnp.einsum('kdac,ikcd->ia', ovvv_blk, t2)
    t1_upd -=     jnp.einsum('kcad,ikcd->ia', ovvv_blk, t2)
    
    t1_upd += 2 * jnp.einsum('kdac,kd,ic->ia', ovvv_blk, t1, t1)
    t1_upd -=     jnp.einsum('kcad,kd,ic->ia', ovvv_blk, t1, t1)

    # 2. Update Lvv (ac) partial -> accumulating into (ac)
    # Note: ovvv_blk corresponds to 'ac' slice [p0:p1, :]
    Lvv_blk =  2 * jnp.einsum('kdac,kd->ac', ovvv_blk, t1)
    Lvv_blk -=     jnp.einsum('kcad,kd->ac', ovvv_blk, t1)
    
    # 3. Update Wvoov (akic) partial -> accumulating into slice [p0:p1]
    Wvoov_blk = jnp.einsum('kcad,id->akic', ovvv_blk, t1)
    
    # 4. Update Wvovo (akci) partial -> accumulating into slice [p0:p1]
    Wvovo_blk = jnp.einsum('kdac,id->akci', ovvv_blk, t1)
    
    # 5. Update tmp_a (kaij) partial -> accumulating into slice [:, p0:p1, :, :]
    tmp_a_blk = jnp.einsum('kdac,ijcd->kaij', ovvv_blk, tau)
    
    # 6. Update tmp_b (kbij) partial -> accumulating into slice [:, p0:p1, :, :]
    # 'kcbd' with ovvv (k,d,a,c) -> k=0, c=3, b=2(a), d=1
    tmp_b_blk = jnp.einsum('kcbd,ijcd->kbij', ovvv_blk, tau)
    
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
    # tmp2_blk = -oovv.ka + vovv -> (a_blk, b, i, c)
    tmp2_blk = jnp.einsum('kibc,ka->abic', eris_oovv, -t1_slice)
    tmp2_blk += vovv_slice.transpose(0, 2, 1, 3)
    
    # Contract with t1_full: (a_blk, b, i, c) * (j, c) -> (a_blk, b, i, j) -> (i, j, a_blk, b)
    term = jnp.einsum('abic,jc->ijab', tmp2_blk, t1_full)
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
    
    # ERIs (Small blocks) -> GPU
    eris_ovov = jnp.asarray(eris.ovov)
    eris_ovvo = jnp.asarray(eris.ovvo)
    eris_oovv = jnp.asarray(eris.oovv)
    eris_ovoo = jnp.asarray(eris.ovoo)
    eris_oooo = jnp.asarray(eris.oooo)
    eris_vooo = jnp.asarray(eris.vooo)
    eris_vovo = jnp.asarray(eris.vovo)
    logger.debug(f"    Data transfer to GPU took {time.perf_counter()-t_start:.4f} s")
    t_kernels = time.perf_counter()
    
    # 2. Compute intermediates (Micro-Kernels)
    # These return JAX arrays on GPU
    Fov = _jax_cc_Fov(t1_jax, fock_jax, eris_ovov)
    Foo = _jax_cc_Foo(t1_jax, t2_jax, fock_jax, eris_ovov)
    Fvv = _jax_cc_Fvv(t1_jax, t2_jax, fock_jax, eris_ovov)
    logger.debug(f"    Micro-kernels (Foo/Fov/Fvv) took {time.perf_counter()-t_kernels:.4f} s")
    
    Foo_shifted = Foo.at[jnp.diag_indices(nocc)].add(-mo_e_o)
    Fvv_shifted = Fvv.at[jnp.diag_indices(nvir)].add(-mo_e_v)
    # Keep unshifted Fvv for Lvv
    Fvv_unshifted = Fvv
    jax.block_until_ready([Foo_shifted, Fvv_shifted])
    logger.debug(f"    Micro-kernels (Foo/Fov/Fvv) and shift took {time.perf_counter()-t_kernels:.4f} s")
    
    size_W = nvir * nocc * nocc * nvir * 8
    size_tmp = nocc * nvir * nocc * nocc * 8
    # Total needed for full GPU accumulators: 2*size_W + 2*size_tmp + Lvv (negligible)
    total_acc_mem = 2 * size_W + 2 * size_tmp
    
    # Query available GPU memory
    available_mem = 0
    try:
        stats = jax.devices()[0].memory_stats()
        logger.debug(f"    Raw GPU stats: {stats}")
        # available = limit - in_use. Leave 20% buffer.
        available_mem = stats['bytes_reservable_limit'] - stats['bytes_in_use']
        use_gpu_acc = (available_mem * 0.8) > total_acc_mem
    except:
        # Fallback if stats not available (e.g. CPU or some backends)
        use_gpu_acc = False
        
    logger.debug(f"    Accumulators need {total_acc_mem/1e9:.2f} GB. GPU available: {available_mem/1e9:.2f} GB. Using GPU acc: {use_gpu_acc}")

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
    logger.debug(f"    T1 core updates: Comp {t_comp_dur:.4f}s, Host accum {t_trans_dur:.4f}s")
    
    t_ovvv = time.perf_counter()
    
    # Tau: (O, O, V, V) needs to be on GPU for optimal contraction
    tau_jax = t2_jax + jnp.einsum('ia,jb->ijab', t1_jax, t1_jax)

    # --- OVVV Processing (Hybrid) ---
    mem_host = cc.max_memory * 1e6
    stats = jax.devices()[0].memory_stats()
    logger.debug(f"    Raw GPU stats (contract_ovvv): {stats}")
    # Use (limit - in_use) to get actual free space, 
    # because bytes_reservable_limit might be equal to limit if JAX pre-allocated everything.
    mem_gpu = stats['bytes_limit'] - stats['bytes_in_use']
    max_mem = min(mem_host, mem_gpu) * 0.3
    blksize = max(4, int(max_mem / (nocc*nvir*nvir*8)))
    blksize = min(nvir, blksize)
    
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
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            ovvv_blk = xtc_ccsd._get_slice(eris.ovvv, slice(p0, p1), axis=2)
            ovvv_blk_jax = jnp.asarray(ovvv_blk)
            
            t0_comp = time.perf_counter()
            t1_upd, Lvv_blk, Wvoov_blk, Wvovo_blk, tmp_a_blk, tmp_b_blk = kernel_process_ovvv_block(
                ovvv_blk_jax, t1_jax, t2_jax, tau_jax
            )
            
            # Accumulate on Host
            t1new_host[:, p0:p1] += np.asarray(t1_upd)
            
            if use_gpu_acc:
                # In-place JAX accumulation
                Lvv_acc = Lvv_acc.at[p0:p1, :].add(Lvv_blk)
                Wvoov_acc = Wvoov_acc.at[p0:p1].add(Wvoov_blk)
                Wvovo_acc = Wvovo_acc.at[p0:p1].add(Wvovo_blk)
                tmp_a_acc = tmp_a_acc.at[:, p0:p1].add(tmp_a_blk)
                tmp_b_acc = tmp_b_acc.at[:, p0:p1].add(tmp_b_blk)
                
            else:
                # Ensure JAX arrays are ready
                jax.block_until_ready([t1_upd, Lvv_blk, Wvoov_blk, Wvovo_blk, tmp_a_blk, tmp_b_blk])
                t_comp_blk = time.perf_counter() - t0_comp
                
                t0_trans = time.perf_counter()
                # Accumulate on Host
                Lvv_acc[p0:p1, :] += np.asarray(Lvv_blk)
                Wvoov_acc[p0:p1] += np.asarray(Wvoov_blk)
                Wvovo_acc[p0:p1] += np.asarray(Wvovo_blk)
                tmp_a_acc[:, p0:p1] += np.asarray(tmp_a_blk)
                tmp_b_acc[:, p0:p1] += np.asarray(tmp_b_blk)
                t_trans_blk = time.perf_counter() - t0_trans
                logger.debug(f"      OVVV block {p0}:{p1}: Comp {t_comp_blk:.4f}s, Host accum {t_trans_blk:.4f}s")
            
            del ovvv_blk_jax, t1_upd, Lvv_blk
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
        mem_host = cc.max_memory * 1e6
        stats = jax.devices()[0].memory_stats()
        logger.debug(f"    Raw GPU stats (contract_vovv): {stats}")
        # Use (limit - in_use) to get actual free space, 
        # because bytes_reservable_limit might be equal to limit if JAX pre-allocated everything.
        mem_gpu = stats['bytes_limit'] - stats['bytes_in_use']
        max_mem = min(mem_host, mem_gpu) * 0.3
        blksize_t2 = max(4, int(max_mem / (nvir*nocc*nvir*8)))
        blksize_t2 = min(nvir, blksize_t2)
        
        for p0 in range(0, nvir, blksize_t2):
            p1 = min(p0 + blksize_t2, nvir)
            vovv_slice = xtc_ccsd._get_slice(eris.vovv, slice(p0, p1), axis=0)
            vovv_slice_jax = jnp.asarray(vovv_slice)
            t1_slice_jax = t1_jax[:, p0:p1]
            
            t0_comp = time.perf_counter()
            term = kernel_process_vovv_block(vovv_slice_jax, eris_oovv, t1_slice_jax, t1_jax)
            term.block_until_ready()
            t_comp = time.perf_counter() - t0_comp
            
            t0_trans = time.perf_counter()
            t2new_host[:, :, p0:p1, :] += np.asarray(term)
            t_trans = time.perf_counter() - t0_trans
            logger.debug(f"      VOVV block {p0}:{p1}: Comp {t_comp:.4f}s, Host accum {t_trans:.4f}s")
        
        t2new_host = t2new_host + t2new_host.transpose(1, 0, 3, 2)
        
    t2new_host += np.asarray(t2new_jax)
    del t2new_jax
    logger.debug(f"    T2 core updates (VOVV) took {time.perf_counter()-t_t2:.4f} s")
    
    # --- Output Preparation ---
    
    # Use GPU accumulators if available, else copy from host
    if not use_gpu_acc:
        Lvv_acc = jnp.asarray(Lvv_acc)
        Wvoov_acc = jnp.asarray(Wvoov_acc)
        Wvovo_acc = jnp.asarray(Wvovo_acc)
        tmp_a_acc = jnp.asarray(tmp_a_acc)
        tmp_b_acc = jnp.asarray(tmp_b_acc)
        
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

    # --- VVVV Contraction ---
    _contract_vvvv_t2(cc, tau_jax, eris, t2new_host)
    
    # Final Division
    eia = mo_e_o[:,None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    
    eia_np = np.asarray(eia)
    eijab_np = np.asarray(eijab)
    
    t1new_host /= eia_np
    t2new_host /= eijab_np
    
    logger.debug("_update_amps finished in %.3f s", time.perf_counter()-t_start)
    return t1new_host, t2new_host


def _contract_vvvv_t2(cc, t2_jax, eris, t2new_host):
    """
    Hybrid contraction of (vv|vv) with t2 (or tau).
    t2_jax: (nocc, nocc, nvir, nvir) on GPU.
    eris: NumPy/HDF5 backed.
    t2new_host: Accumulation target (NumPy).
    """
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params
    
    if eris.vvvv is not None:
        vvvv_jax = jnp.asarray(eris.vvvv)
        term = jnp.einsum('acbd,ijcd->ijab', vvvv_jax, t2_jax)
        t2new_host += np.asarray(term)
        return

    mem_host = cc.max_memory * 1e6
    try:
        stats = jax.devices()[0].memory_stats()
        logger.debug(f"    Raw GPU stats (contract_vvvv): {stats}")
        # Use (limit - in_use) to get actual free space, 
        # because bytes_reservable_limit might be equal to limit if JAX pre-allocated everything.
        mem_gpu = stats['bytes_limit'] - stats['bytes_in_use']
    except:
        mem_gpu = cc.gpu_max_memory * 1e6
        
    # Adaptive block size based on VRAM
    # Need to fit vvvv_block (blk*nvir^3), t2 (O^2 V^2), output (O^2 blk V).
    # Approx: blk * nvir^3 * 8
    avail_gpu = mem_gpu * 0.3
    # Use 80% of available free memory for the block (since we already subtracted usage)
    blksize = max(1, int(avail_gpu / (nvir**3 * 8)))
    blksize = min(nvir, blksize)
    logger.debug(f"    VVVV contraction: blksize={blksize}, n_blocks={(nvir+blksize-1)//blksize}")

    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
         with_df = cc._scf.with_df
    
    L_vv_full_jax = None
    if with_df is not None:
         L_vv_full_jax = jnp.asarray(lib.unpack_tril(eris.vvL[:], axis=0))

    @jax.jit
    def contract_block_kernel(t2, xtc_block, L_ab_sub, L_vv_full):
        if L_ab_sub is not None:
            vvvv_jax = xtc_block + jnp.tensordot(L_ab_sub, L_vv_full, axes=((2), (2)))
        else:
            vvvv_jax = xtc_block
        return jnp.einsum('acbd,ijcd->ijab', vvvv_jax, t2)

    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        ranges = (slice(nocc + p0, nocc + p1), slice(nocc, cc.nmo), slice(nocc, cc.nmo), slice(nocc, cc.nmo))
        
        t_get_2b = time.perf_counter()
        vvvv_block_jax = xtc_obj.get_2b(jastrow_params, ranges=ranges)
        # Force block to see true computation time if it returns JAX array
        if hasattr(vvvv_block_jax, 'block_until_ready'):
            vvvv_block_jax.block_until_ready()
        logger.debug(f"      get_2b (block {p0}:{p1}) took {time.perf_counter()-t_get_2b:.4f} s")
        
        L_ab_sub_jax = L_vv_full_jax[p0:p1] if with_df is not None else None
        
        if with_df is None:
             t_ao2mo = time.perf_counter()
             mo_v = cc.mo_coeff[:, nocc:]
             # CPU-bound standard integral calculation
             std_block = ao2mo.general(cc.mol, (mo_v[:, p0:p1], mo_v, mo_v, mo_v), compact=False)
             logger.debug(f"      ao2mo (std integrals) took {time.perf_counter()-t_ao2mo:.4f} s")
             
             t_transfer = time.perf_counter()
             std_jax = jnp.asarray(std_block.reshape(p1-p0, nvir, nvir, nvir))
             # Ensure transfer is complete before proceeding
             if hasattr(std_jax, 'block_until_ready'):
                 std_jax.block_until_ready()
             logger.debug(f"      Host->Device transfer of std integrals took {time.perf_counter()-t_transfer:.4f} s")
             
             t_add = time.perf_counter()
             vvvv_block_jax = vvvv_block_jax + std_jax
             if hasattr(vvvv_block_jax, 'block_until_ready'):
                 vvvv_block_jax.block_until_ready()
             logger.debug(f"      Element-wise addition (vvvv + std) took {time.perf_counter()-t_add:.4f} s")
             
        t0_comp = time.perf_counter()
        term = contract_block_kernel(t2_jax, vvvv_block_jax, L_ab_sub_jax, L_vv_full_jax)
        term.block_until_ready()
        t_comp = time.perf_counter() - t0_comp
        
        t0_trans = time.perf_counter()
        t2new_host[:, :, p0:p1, :] += np.asarray(term)
        t_trans = time.perf_counter() - t0_trans
        logger.debug(f"      VVVV block {p0}:{p1}: Comp {t_comp:.4f}s, Host accum {t_trans:.4f}s")
    
    if L_vv_full_jax is not None:
        del L_vv_full_jax