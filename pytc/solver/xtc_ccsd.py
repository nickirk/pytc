import logging
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.cc import rccsd
from pyscf.cc import rintermediates as imd
from pyscf import ao2mo

logger = logging.getLogger(__name__)

class RCCSD(rccsd.RCCSD):
    """Restricted CCSD with ISDF-XTC integrals."""
    def __init__(self, mf, xtc_obj, jastrow_params, **kwargs):
        rccsd.RCCSD.__init__(self, mf, **kwargs)
        self.xtc_obj = xtc_obj
        self.jastrow_params = jastrow_params

    def ao2mo(self, mo_coeff=None):
        return _make_xtc_eris(self, mo_coeff)
        
    def update_amps(self, t1, t2, eris):
        return _update_amps(self, t1, t2, eris)

    def energy(self, t1=None, t2=None, eris=None):
        return _energy(self, t1, t2, eris)


class _ChemistsERIs(rccsd._ChemistsERIs):
    """Custom ERIs holding XTC data."""
    def __init__(self, mol=None):
        rccsd._ChemistsERIs.__init__(self, mol)
        self.xtc_obj = None
        self.jastrow_params = None

def _make_xtc_eris(cc, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
        
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params
    
    # Use standard make_eris if it's not an ISDF object to avoid recomputation
    from pytc.autodiff.xtc import XTC, ISDFXTC
    if isinstance(xtc_obj, XTC) and not isinstance(xtc_obj, ISDFXTC):
        logger.info("Using XTC.make_eris for standard XTC object")
        return xtc_obj.make_eris(cc._scf, jastrow_params)

    eris = _ChemistsERIs(cc.mol)
    eris._common_init_(cc, mo_coeff)
    eris.xtc_obj = xtc_obj
    eris.jastrow_params = jastrow_params
    
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    mo_o = mo_coeff[:, :nocc]
    
    # 1. Fock matrix construction 
    # Start from MF Fock matrix (diagonal in MO basis if mo_coeff are mf.mo_coeff)
    # We use get_fock to ensure standard ERI part is exactly consistent with mf
    dm_std = cc._scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=cc._scf.mo_occ)
    fock_std = cc._scf.get_fock(dm=dm_std)
    fock_std = reduce(np.dot, (mo_coeff.T, fock_std, mo_coeff))

    
    h1e_corr = np.asarray(xtc_obj.get_1b(jastrow_params))
    # Corrections to Fock from TC 2-body part: (pq|ii) and (pi|iq) corrections only
    h2e_pqii_corr = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=(slice(None), slice(None), slice(0, nocc), slice(0, nocc))))
    h2e_piiq_corr = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=(slice(None), slice(0, nocc), slice(0, nocc), slice(None))))
    
    fock_corr = h1e_corr + 2 * np.einsum('pqii->pq', h2e_pqii_corr) - np.einsum('piiq->pq', h2e_piiq_corr)
    eris.fock = fock_std + fock_corr
    eris.fvo = eris.fock[nocc:, :nocc].copy()
    eris.mo_energy = np.diag(eris.fock)

    # 2. Materialize required blocks (except vvvv) using ISDF efficiently
    eri_std_full = ao2mo.kernel(cc.mol, mo_coeff, compact=False, aosym='s1', intor='int2e')
    eri_std_full = eri_std_full.reshape(nmo, nmo, nmo, nmo)
    
    def get_block(block_str):
        tc_part = np.asarray(xtc_obj.get_2b(jastrow_params, block_str=block_str))
        slices = [slice(0, nocc) if c == 'o' else slice(nocc, nmo) for c in block_str]
        return eri_std_full[tuple(slices)] + tc_part

    eris.oooo = get_block('oooo')
    eris.ovoo = get_block('ovoo')
    eris.ooov = get_block('ooov')
    eris.vooo = get_block('vooo')
    eris.ovov = get_block('ovov')
    eris.vovo = get_block('vovo')
    eris.ovvo = get_block('ovvo')
    eris.voov = get_block('voov')
    eris.oovv = get_block('oovv')
    eris.vvoo = get_block('vvoo')
    eris.ovvv = get_block('ovvv')
    eris.vvov = get_block('vvov')
    eris.vovv = get_block('vovv')
    eris.vvvv = None
    
    return eris




    
    return eris

def _contract_vvvv_t2(cc, t2, eris, out=None):
    """Contraction of (vv|vv) with t2. Handles both materialized and block-wise cases."""
    if eris.vvvv is not None:
        # Materialized case: Transpose to (a, c, b, d) and contract
        # Standard PySCF index for Wvvvv is (ab|cd) contracted with t2(ij|cd) gives (ij|ab)
        # Here we follow PySCF's rintermediates.cc_Wvvvv logic if materialized
        vvvv = np.asarray(eris.vvvv)
        return lib.einsum('abcd,ijcd->ijab', vvvv.transpose(0, 2, 1, 3), t2)

    if out is None:
        out = np.zeros_like(t2)

    
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params

    # Memory-efficient block size
    blksize = max(1, int(1e8 / (nvir**3 * 8)))
    blksize = min(nvir, blksize)
    
    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        ranges = (slice(nocc + p0, nocc + p1), slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo))
        
        vvvv_block = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))
        
        # Add Standard Integrals if not handled by with_df
        mo_v = cc.mo_coeff[:, nocc:]
        std_block = ao2mo.general(cc.mol, (mo_v[:, p0:p1], mo_v, mo_v, mo_v), compact=False)
        vvvv_block = vvvv_block + std_block.reshape(p1-p0, nvir, nvir, nvir)
        
        # Transpose to (a, c, b, d) and contract
        vvvv_trans = vvvv_block.transpose(0, 2, 1, 3)
        out[:, :, p0:p1, :] += lib.einsum('abcd,ijcd->ijab', vvvv_trans, t2)
        
    return out

def _update_amps(cc, t1, t2, eris):
    """
    Restricted CCSD amplitude update with efficient Wvvvv contraction.
    
    Modified from PySCF rccsd.py:
    Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += eris.fock[nocc:, :nocc].T
    
    t1new += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)

    eris_ovoo = np.asarray(eris.ovoo)
    t1new +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris.vovv).transpose(0, 2, 1, 3)

    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += np.asarray(eris.vooo).transpose(0, 3, 1, 2) 
    # eris.vooo is (a, i, j, k) as (ai|jk). 
    # Transpose (a:0, k:3, i:1, j:2) gives (ak|ij).
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)


    t2new -= tmp + tmp.transpose(1,0,3,2)
    
    t2new += np.asarray(eris.ovov).transpose(0, 2, 1, 3)

    
    if cc.cc2:
        raise NotImplementedError("CC2 not supported")

    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    Lvv[np.diag_indices(nvir)] -= mo_e_v

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)

    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
    
    # Efficient Wvvvv contraction
    t2new += _contract_vvvv_t2(cc, tau, eris)
    
    # Substituted ovvv with appropriate non-hermitian counterparts if necessary?
    # PySCF uses eris_ovvv which is (ia|bc).
    tmp_a = lib.einsum('kdac,ijcd->kaij', eris_ovvv, tau)
    t2new -= lib.einsum('kb,kaij->ijab', t1, tmp_a)
    tmp_b = lib.einsum('kcbd,ijcd->kbij', eris_ovvv, tau)
    t2new -= lib.einsum('ka,kbij->ijab', t1, tmp_b)



    tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

def _energy(cc, t1, t2, eris):
    """CCSD correlation energy for non-Hermitian case."""
    nocc, nvir = t1.shape
    fock = eris.fock
    # e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    # Non-Hermitian: uses fov?
    fov = fock[:nocc, nocc:]
    e = 2*np.einsum('ia,ia', fov, t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    return e.real

