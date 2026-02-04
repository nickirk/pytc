import logging
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.cc import rccsd
from pyscf.cc import rintermediates as imd
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

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

    def density_fit(self, auxbasis=None, with_df=None, n_rank_xtc=None, with_isdf_xtc=None, **kwargs):
        '''
        Update local RCCSD object to use density fitting for standard Coulomb integrals
        and/or ISDF for XTC integrals.

        Args:
            auxbasis (str): Auxiliary basis for standard Coulomb DF.
            with_df (pyscf.df.DF): Existing DF object for standard integrals.
            n_rank_xtc (int): Rank for ISDF-XTC decomposition. used if with_isdf_xtc is None.
            with_isdf_xtc (ISDFXTC): Existing ISDF-XTC object to use.
            **kwargs: Additional arguments for ISDFXTC conversion (e.g., save_path) if creating new ISDFXTC.
        
        Returns:
            A new RCCSD object with DF enabled.
        '''
        new_cc = self.copy()
        
        # 1. Setup Standard Coulomb DF
        if with_df is not None:
             new_cc.with_df = with_df
        elif getattr(self._scf, 'with_df', None):
             new_cc.with_df = self._scf.with_df.copy()
        else:
             from pyscf import df
             new_cc.with_df = df.DF(self.mol)
             new_cc.with_df.auxbasis = auxbasis or 'weigend'
        
        if auxbasis is not None and new_cc.with_df.auxbasis != auxbasis:
             new_cc.with_df = new_cc.with_df.copy()
             new_cc.with_df.auxbasis = auxbasis
             
        # 2. Setup ISDF-XTC
        from pytc.autodiff.xtc import ISDFXTC
        if with_isdf_xtc is not None:
            new_cc.xtc_obj = with_isdf_xtc
        elif n_rank_xtc is not None:
            if not isinstance(self.xtc_obj, ISDFXTC):
                 # Convert to ISDFXTC
                 new_cc.xtc_obj = ISDFXTC.from_xtc(self.xtc_obj, n_rank=n_rank_xtc, **kwargs)
                 # Ensure ISDF kernels are computed
                 if self.jastrow_params is not None:
                     new_cc.xtc_obj = new_cc.xtc_obj.isdf(self.jastrow_params)
            else:
                 logger.warn("xtc_obj is already ISDFXTC, but n_rank_xtc provided. Ignoring n_rank_xtc re-decomposition for now to avoid complexity.")
        
        return new_cc



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
    # BUT if density fitting is requested (with_df), we must use the DF path 
    # even for standard XTC objects (to use DF for standard integrals).
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
        with_df = cc._scf.with_df

    from pytc.autodiff.xtc import XTC, ISDFXTC
    if isinstance(xtc_obj, XTC) and not isinstance(xtc_obj, ISDFXTC) and with_df is None:
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
    
    # Check for density fitting
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
        with_df = cc._scf.with_df

    if with_df is not None:
        # --- Density Fitting Path ---
        logger.info("Using Density Fitting for standard Coulomb integrals in XTC-CCSD")
        
        # Prepare 3-index tensors L_pq = (L|pq)
        # We need Loo, Lov, Lvv (Lvv stored as vvL for efficiency)
        naux = with_df.get_naoaux()
        
        # Use HDF5 for large tensors if needed, typically Lvv is large
        # We follow pyscf.cc.dfccsd pattern
        if isinstance(with_df._cderi, str):
            import h5py
            eris.feri = h5py.File(with_df._cderi, 'a')
        elif isinstance(getattr(with_df, '_cderi_to_save', None), str):
            import h5py
            eris.feri = h5py.File(with_df._cderi_to_save, 'a')
        else:
            eris.feri = lib.H5TmpFile()
            
        nvir_pair = nvir * (nvir+1) // 2
        
        # Loo and Lov are usually small enough for memory
        Loo = np.empty((naux, nocc, nocc))
        Lov = np.empty((naux, nocc, nvir))
        
        # vvL stored on disk: (nvir_pair, naux)
        # chunks strategy from dfccsd
        chunks = (min(nvir_pair, int(4e8/with_df.blockdim)), min(naux, with_df.blockdim))
        eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair, naux), 'f8', chunks=chunks)
        
        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None
        
        # Determine max_memory for the loop
        # max_memory = cc.max_memory - lib.current_memory()[0]
        
        for k, eri1 in enumerate(with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1-p0, nmo, nmo)
            
            # Extract blocks
            Loo[p0:p1] = Lpq[:, :nocc, :nocc]
            Lov[p0:p1] = Lpq[:, :nocc, nocc:]
            
            # Pack vv part
            Lvv_tril = lib.pack_tril(Lpq[:, nocc:, nocc:])
            eris.vvL[:, p0:p1] = Lvv_tril.T
            
        Lpq = None
        
        # Reshape Loo, Lov
        Loo = Loo.reshape(naux, nocc*nocc)
        Lov = Lov.reshape(naux, nocc*nvir)
        
        def get_block_df(block_str):
            tc_part = np.asarray(xtc_obj.get_2b(jastrow_params, block_str=block_str))
            
            # Construct standard part from L tensors
            # Note: lib.ddot(A.T, B) computes A^T . B
            
            if block_str == 'oooo':
                std = lib.ddot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
            
            elif block_str == 'ovoo':
                std = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
                
            elif block_str == 'ovov':
                std = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
                
            elif block_str == 'ovvo':
                # ovov constructed as Lov.T @ Lov is (ia|jb)
                # ovvo is (ia|jb).transpose(0, 1, 3, 2)
                tmp = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
                std = tmp.transpose(0, 1, 3, 2)
                
            elif block_str == 'oovv':
                oovv_tril = np.empty((nocc*nocc, nvir_pair))
                blksize = max(4, int(1e8/naux)) 
                for p0, p1 in lib.prange(0, nvir_pair, blksize):
                     vvL_slice = eris.vvL[p0:p1] # (blk, naux)
                     oovv_tril[:, p0:p1] = lib.ddot(Loo.T, vvL_slice.T)
                
                std = lib.unpack_tril(oovv_tril).reshape(nocc, nocc, nvir, nvir)
                
            elif block_str == 'vvoo':
                oovv_tril = np.empty((nocc*nocc, nvir_pair))
                blksize = max(4, int(1e8/naux)) 
                for p0, p1 in lib.prange(0, nvir_pair, blksize):
                     vvL_slice = eris.vvL[p0:p1]
                     oovv_tril[:, p0:p1] = lib.ddot(Loo.T, vvL_slice.T)
                std_oovv = lib.unpack_tril(oovv_tril).reshape(nocc, nocc, nvir, nvir)
                std = std_oovv.transpose(2,3,0,1)

            elif block_str == 'ovvv':
                 ovvv_tril = np.empty((nocc*nvir, nvir_pair))
                 blksize = max(4, int(1e8/naux))
                 for p0, p1 in lib.prange(0, nvir_pair, blksize):
                     vvL_slice = eris.vvL[p0:p1]
                     ovvv_tril[:, p0:p1] = lib.ddot(Lov.T, vvL_slice.T)
                     
                 std = lib.unpack_tril(ovvv_tril).reshape(nocc, nvir, nvir, nvir)
                 
            elif block_str == 'vvov':
                ovvv_tril = np.empty((nocc*nvir, nvir_pair))
                blksize = max(4, int(1e8/naux))
                for p0, p1 in lib.prange(0, nvir_pair, blksize):
                    vvL_slice = eris.vvL[p0:p1]
                    ovvv_tril[:, p0:p1] = lib.ddot(Lov.T, vvL_slice.T)
                std_ovvv = lib.unpack_tril(ovvv_tril).reshape(nocc, nvir, nvir, nvir)
                std = std_ovvv.transpose(2,3,0,1)
                
            elif block_str == 'vovv':
                ovvv_tril = np.empty((nocc*nvir, nvir_pair))
                blksize = max(4, int(1e8/naux))
                for p0, p1 in lib.prange(0, nvir_pair, blksize):
                    vvL_slice = eris.vvL[p0:p1]
                    ovvv_tril[:, p0:p1] = lib.ddot(Lov.T, vvL_slice.T)
                std_ovvv = lib.unpack_tril(ovvv_tril).reshape(nocc, nvir, nvir, nvir)
                std = std_ovvv.transpose(1,0,2,3)

            elif block_str == 'ooov':
                 std = lib.ddot(Loo.T, Lov).reshape(nocc, nocc, nocc, nvir)

            else:
                 if block_str == 'vooo':
                     std = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc).transpose(1, 0, 2, 3)
                 elif block_str == 'voov':
                     std = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir).transpose(1, 0, 2, 3)
                 elif block_str == 'vovo':
                     # (ai|bj) from (ia|jb)
                     std = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir).transpose(1, 0, 3, 2)
                 else:
                     raise NotImplementedError(f"DF block {block_str} not implemented")

            return std + tc_part

        eris.oooo = get_block_df('oooo')
        eris.ovoo = get_block_df('ovoo')
        eris.ooov = get_block_df('ooov')
        eris.vooo = get_block_df('vooo')
        eris.ovov = get_block_df('ovov')
        eris.vovo = get_block_df('vovo')
        eris.ovvo = get_block_df('ovvo')
        eris.voov = get_block_df('voov')
        eris.oovv = get_block_df('oovv')
        eris.vvoo = get_block_df('vvoo')
        eris.ovvv = get_block_df('ovvv')
        eris.vvov = get_block_df('vvov')
        eris.vovv = get_block_df('vovv')
        eris.vvvv = None
        
        # Cleanup
        del Loo, Lov, Lpq, mo
        # Keep eris.vvL for vvvv contraction
        
        return eris

    else:
        # --- Standard Path (ao2mo) --- 
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
        
        with_df = getattr(cc, 'with_df', None)
        if with_df is None and getattr(cc._scf, 'with_df', None):
             with_df = cc._scf.with_df

        if with_df is not None:
             naux = eris.vvL.shape[1]
             L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0) # (nvir, nvir, naux)
             
             L_ab_sub = L_vv_full[p0:p1]
             std_block = np.tensordot(L_ab_sub, L_vv_full, axes=((2), (2)))
             # std_block shape is (blk, nvir, nvir, nvir)
             
             vvvv_block = vvvv_block + std_block
             
        else:
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

