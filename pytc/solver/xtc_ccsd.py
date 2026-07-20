import contextlib
import concurrent.futures
import logging
import threading
import time
from collections import deque
import numpy as np
import jax
from functools import reduce
from pyscf import lib
from pyscf.cc import rccsd
from pyscf.cc import rintermediates as imd
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

from pytc.utils.gpu_memory import (
    estimate_blksize,
    resolve_vvvv_panel_block_sizes,
    resolve_v3o_panel_block_size,
    enable_xla_compilation_cache,
)
from pytc.utils.gpu_pipeline import (
    _solver_local_devices,
    broadcast_to_devices,
    _gpu_slot_ctx,
    _round_robin_pipeline,
    partition_round_robin,
    _AsyncHDF5Writer,
)
import h5py
from pytc import xtc as xtc_mod

logger = logging.getLogger(__name__)

class RCCSD(rccsd.RCCSD):
    """Restricted CCSD with ISDF-XTC integrals."""
    def __init__(self, mf, xtc_obj=None, jastrow_params=None, **kwargs):
        self.gpu_max_memory = kwargs.pop('gpu_max_memory', 4000)
        self.on_the_fly_vvvv = kwargs.pop('on_the_fly_vvvv', False)
        self.vvvv_p_block_size = kwargs.pop('vvvv_p_block_size', None)
        self.vvvv_r_block_size = kwargs.pop('vvvv_r_block_size', None)
        self.v3o_block_size = kwargs.pop('v3o_block_size', None)
        max_memory = kwargs.pop('max_memory', None)
        rccsd.RCCSD.__init__(self, mf, **kwargs)
        self.xtc_obj = xtc_obj
        self.jastrow_params = jastrow_params

        if max_memory is not None:
            self.max_memory = max_memory
        if getattr(self, 'max_memory', None) is None:
            self.max_memory = getattr(mf, 'max_memory', 4000)

        self._keys = self._keys.union([
            'xtc_obj', 'jastrow_params', 'gpu_max_memory',
            'on_the_fly_vvvv', 'vvvv_p_block_size', 'vvvv_r_block_size',
            'v3o_block_size',
        ])

        # Enable XLA persistent compilation cache so compiled HLO programs
        # are reused across CCSD iterations and across runs.
        enable_xla_compilation_cache()

    def ao2mo(self, mo_coeff=None):
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        eris = _make_xtc_eris(self, mo_coeff)
        self.e_hf = self.get_e_hf(eris)
        return eris
        
    def update_amps(self, t1, t2, eris):
        return _update_amps(self, t1, t2, eris)

    def energy(self, t1=None, t2=None, eris=None):
        return _energy(self, t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=None):
        if eris is None:
            eris = self.ao2mo()
        self.e_hf = self.get_e_hf(eris)
        return super().ccsd(t1, t2, eris, mbpt2)

    def energy_tot(self, t1=None, t2=None, eris=None):
        return self.get_e_hf(eris) + self.energy(t1, t2, eris)

    def get_e_hf(self, eris=None):
        if eris is None:
            if getattr(self, 'e_hf', None) is not None:
                return self.e_hf
            return self._scf.e_tot
        
        no = self.nocc
        fock = eris.fock
        # E_hf = 2*sum_i F_ii - 2*sum_ij (ii|jj) + sum_ij (ij|ji) + E_core
        e_hf = 2*np.einsum('ii->', fock[:no,:no])
        if hasattr(eris, 'oooo'):
            oooo = np.asarray(eris.oooo)
            e_hf -= 2*np.einsum('iijj ->', oooo)
            e_hf += np.einsum('ijji ->', oooo)
        
        e_hf += getattr(eris, 'e_core', 0)
        
        return e_hf.real

    def _finalize(self):
        if self.converged:
            lib.logger.info(self, '%s converged', self.__class__.__name__)
        else:
            lib.logger.note(self, '%s not converged', self.__class__.__name__)
        
        lib.logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr
    
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
        from pytc.xtc import ISDFXTC
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
                logger.warning("xtc_obj is already ISDFXTC, but n_rank_xtc provided. Ignoring n_rank_xtc re-decomposition for now to avoid complexity.")
        
        return new_cc


# GPU pipeline primitives are imported at the top of the module (see
# pytc.utils.gpu_pipeline) and include _AsyncHDF5Writer used by the
# large-blocks / vvvv write paths below.


class _ChemistsERIs(rccsd._ChemistsERIs):
    """Custom ERIs holding XTC data."""
    def __init__(self, mol=None):
        rccsd._ChemistsERIs.__init__(self, mol)
        self.xtc_obj = None
        self.jastrow_params = None
        self.max_memory = None
        self.gpu_max_memory = None
        if not hasattr(self, '_keys'):
            self._keys = set()
        self._keys = self._keys.union(['xtc_obj', 'jastrow_params', 'max_memory', 'gpu_max_memory'])

    def close(self):
        if hasattr(self, 'feri') and hasattr(self.feri, 'close'):
             self.feri.close()

    def __del__(self):
        """
        Some DF/ao2mo paths store large ERI blocks in an HDF5 handle (self.feri).
        In test suites, these objects may become unreachable without explicit
        close() calls, which can leave HDF5 files/datasets open and lead to
        cross-test state leakage.
        """
        try:
            self.close()
        except Exception:
            pass

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

    from pytc.xtc import XTC, ISDFXTC
    if isinstance(xtc_obj, XTC) and not isinstance(xtc_obj, ISDFXTC) and with_df is None:
        logger.info("Using XTC.make_eris for standard XTC object")
        return xtc_obj.make_eris(cc._scf, jastrow_params)

    eris = _ChemistsERIs(cc.mol)
    eris._common_init_(cc, mo_coeff)
    eris.xtc_obj = xtc_obj
    eris.jastrow_params = jastrow_params
    eris.max_memory = cc.max_memory
    eris.gpu_max_memory = cc.gpu_max_memory
    
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    # mo_o removed as unused
    
    # 1. Fock matrix construction 
    # Start from MF Fock matrix (diagonal in MO basis if mo_coeff are mf.mo_coeff)
    # We use get_fock to ensure standard ERI part is exactly consistent with mf
    dm_std = cc._scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=cc._scf.mo_occ)
    fock_std = cc._scf.get_fock(dm=dm_std)
    fock_std = reduce(np.dot, (mo_coeff.T, fock_std, mo_coeff))

    
    # orb_block_size=None → get_delta_h auto-picks from probed GPU free memory.
    h1e_corr = np.asarray(xtc_obj.get_1b(jastrow_params))
    eris.e_core = np.asarray(xtc_obj.get_const(jastrow_params, delta_h=h1e_corr))
    # Corrections to Fock from TC 2-body part: (pq|ii) and (pi|iq) corrections.
    _fock_devices = _solver_local_devices()
    _fock_ranges = (
        (slice(None), slice(None), slice(0, nocc), slice(0, nocc)),
        (slice(None), slice(0, nocc), slice(0, nocc), slice(None)),
    )

    def _fock_worker(ranges, device):
        _ctx = jax.default_device(device) if device is not None else contextlib.nullcontext()
        with _ctx:
            return np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))

    if len(_fock_devices) >= 2:
        # Genuinely independent GPUs available — dispatch the two
        # ``get_2b`` calls to distinct devices in parallel.  Each
        # call materialises a (nmo, nmo, nocc, nocc) intermediate
        # which can be multi-GB at production scale, so on a single-
        # device run putting both in flight at once would roughly
        # double the peak device memory and trip OOM.  See the else
        # branch for the safe single-GPU fallback.
        _fock_results = [None, None]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
            futs = [
                _pool.submit(_fock_worker, _fock_ranges[i], _fock_devices[i % len(_fock_devices)])
                for i in range(2)
            ]
            for i, _f in enumerate(futs):
                _fock_results[i] = _f.result()
        h2e_pqii_corr, h2e_piiq_corr = _fock_results
    else:
        # Single-device (or CPU) run — issue serially so the two big
        # intermediates do not coexist on the same GPU.  Codex P1
        # observation: dispatching both to the same accelerator
        # concurrently doubles peak VRAM in the 1-GPU path and was
        # the dominant OOM trigger before this guard.
        _device = _fock_devices[0] if _fock_devices else None
        h2e_pqii_corr = _fock_worker(_fock_ranges[0], _device)
        h2e_piiq_corr = _fock_worker(_fock_ranges[1], _device)

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
        naux = with_df.get_naoaux()
        Loo, Lov = _init_df_eris(eris, with_df, nvir, naux, nocc, nmo, mo_coeff)
        

        # Unpack Lvv to RAM if possible (approx 5-10GB for 800 orbitals)
        L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0) # (nvir, nvir, naux)
        Lov_reshaped = Lov.reshape(naux, nocc, nvir)
        
        _n_fused = None
        if hasattr(xtc_obj, 'phi_isdf') and xtc_obj.phi_isdf is not None:
            _n_fused = xtc_obj.phi_isdf.shape[1]
        panel_blk = resolve_v3o_panel_block_size(
            nocc, nvir,
            block_size=getattr(cc, 'v3o_block_size', None),
            gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
            host_max_memory_mb=getattr(cc, 'max_memory', None),
            naux=naux,
            n_fused=_n_fused,
            include_eris=False,
            include_accumulators=False,
        )

        # Create HDF5 datasets for large blocks.
        #
        # Both ovvv and vovv are WRITTEN as [:, :, r0:r1, :] slabs in chunks
        # of panel_blk along axis 2 (the r virtual index), and READ as
        # [:, :, p0:p1, :] slabs along the same axis during CCSD iterations.
        # Chunking axis-2 at panel_blk makes every write slab exactly cover
        # an integer number of chunks on that axis — no read-modify-write
        # of boundary chunks, which was the main write amplifier that made
        # HDF5 writes the pipeline bottleneck (consume threads held the
        # acc_lock for the full RMW, blocking all other consume threads and
        # eventually stalling the main dispatch thread on host_sem).
        #
        # Axes 1 and 3 are chunked at 64 each so a single chunk is ~15 MB
        # for typical (nocc, nvir, panel_blk) — a good HDF5 compromise
        # between per-chunk overhead (favours bigger) and chunk cache hit
        # rate (favours smaller).
        _ax13 = min(64, nvir)
        _ax2  = min(panel_blk, nvir)
        eris_blocks = {
            'ovvv': ((nocc, nvir, nvir, nvir),
                     (nocc, _ax13, _ax2, _ax13)),
            'vovv': ((nvir, nocc, nvir, nvir),
                     (_ax13, nocc, _ax2, _ax13)),
        }
        for name, (shape, chunks) in eris_blocks.items():
            if name in eris.feri:
                del eris.feri[name]
            setattr(eris, name, eris.feri.create_dataset(name, shape, 'f8', chunks=chunks))

        # Preload X into RAM before the large-block / VVVV pipelines. Every
        # tile in _compute_large_blocks and _compute_vvvv_block_df calls
        # _get_delta_u_direct_tile, which slices X per tile; if X is still an
        # HDF5 dataset, those slice reads happen on the main dispatch thread
        # and serialize issue_tile, preventing multi-GPU overlap.
        # The materialized default preloads (a real optimization there); a
        # class whose contraction reads X panel-wise from the store (e.g. the
        # factorized subclass) suppresses it via `_preload_x_for_eris = False`
        # rather than paying for a full host copy it never uses.
        _kernels = xtc_obj.isdf_kernels
        _X_kernel = _kernels.get('X') if _kernels is not None else None
        if (isinstance(_X_kernel, h5py.Dataset)
                and getattr(cc, "_preload_x_for_eris", True)):
            _x_gb = _X_kernel.size * 8 / 1e9
            logger.info(
                "Preloading X into RAM before large-block build (%.2f GB) ...",
                _x_gb,
            )
            _t_x = time.perf_counter()
            _kernels = dict(_kernels)
            _kernels['X'] = _X_kernel[:]
            xtc_obj = xtc_obj.replace(isdf_kernels=_kernels)
            # Propagate to cc/eris so the downstream VVVV on-the-fly path
            # (jax_xtc_ccsd.py:953 reads ``cc.xtc_obj``) sees the preloaded
            # numpy X and does NOT fall back to HDF5 reads per tile.
            cc.xtc_obj = xtc_obj
            eris.xtc_obj = xtc_obj
            logger.info("X preload done in %.1f s", time.perf_counter() - _t_x)

        logger.info("Computing large blocks...")
        _compute_large_blocks(
            eris, xtc_obj, jastrow_params, Lov_reshaped, L_vv_full,
            nocc, nvir, nmo, panel_blk)

        # Medium blocks: tiled multi-GPU pipeline (same approach as ovvv/vovv).
        # oovv/vvoo/ovov/ovvo/vovo each have two virtual indices; we tile over
        # one virtual dimension in chunks of panel_blk and dispatch across all
        # local GPUs via _round_robin_pipeline.
        _medium_devices = _solver_local_devices()
        _medium_results = _compute_medium_blocks_tiled(
            xtc_obj, jastrow_params, Loo, Lov_reshaped, L_vv_full,
            nocc, nvir, nmo, panel_blk, _medium_devices,
        )
        eris.oovv = _medium_results['oovv']
        eris.vvoo = _medium_results['vvoo']
        
        # --- VVVV handling (must run before L_vv_full is freed) ---
        vvvv_bytes = float(nvir)**4 * 8
        on_the_fly = getattr(cc, 'on_the_fly_vvvv', False)
        if on_the_fly:
            logger.info(f"VVVV on-the-fly mode (on_the_fly_vvvv=True). "
                        f"Skipping disk storage. "
                        f"({vvvv_bytes/1e9:.2f} GB would be needed on disk)")
            eris.vvvv = None
        else:
            logger.info(f"VVVV will be saved to disk ({vvvv_bytes/1e9:.2f} GB). "
                        f"Set on_the_fly_vvvv=True to compute on-the-fly instead.")
            # Save vvvv block-wise to HDF5 (same file as ovvv/vovv)
            vvvv_shape = (nvir, nvir, nvir, nvir)
            if 'vvvv' in eris.feri:
                del eris.feri['vvvv']
            eris.vvvv = eris.feri.create_dataset('vvvv', vvvv_shape, 'f8')
            _compute_vvvv_block_df(eris, xtc_obj, jastrow_params,
                                   L_vv_full, nocc, nvir, nmo, cc)

        # Free L_vv_full after we are done with all blocks needing it
        del L_vv_full

        # Assign remaining medium blocks from the tiled pipeline results
        eris.ovvo = _medium_results['ovvo']
        eris.ovov = _medium_results['ovov']
        eris.vovo = _medium_results['vovo']
        del _medium_results

        # Small all-occupied blocks — full-block get_2b + DF, no tiling needed
        # Loo: (naux, nocc²), Lov: (naux, nocc×nvir) — both flat for lib.ddot
        for _blk_str, _std in [
            ('oooo', lib.ddot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)),
            ('ovoo', lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)),
            ('ooov', lib.ddot(Loo.T, Lov).reshape(nocc, nocc, nocc, nvir)),
            ('vooo', lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc).transpose(1, 0, 2, 3)),
        ]:
            logger.debug("Computing block %s", _blk_str)
            _tc = np.asarray(xtc_obj.get_2b(jastrow_params, block_str=_blk_str))
            setattr(eris, _blk_str, _std + _tc)

        del Loo, Lov, Lov_reshaped

        # Keep eris.vvL for on-the-fly vvvv contraction if needed
        
        return eris

    else:
        # --- Standard Path (ao2mo) --- 
        logger.info("Using standard ao2mo for Coulomb integrals")
        eri_std_full = ao2mo.kernel(cc.mol, mo_coeff, compact=False, aosym='s1', intor='int2e')
        eri_std_full = eri_std_full.reshape(nmo, nmo, nmo, nmo)
        
        def get_block(block_str):
            logger.debug(f"Computing block {block_str} for xtc")
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
        
        # --- VVVV handling ---
        vvvv_bytes = float(nvir)**4 * 8
        on_the_fly = getattr(cc, 'on_the_fly_vvvv', False)
        if on_the_fly:
            logger.info(f"VVVV on-the-fly mode (on_the_fly_vvvv=True). "
                        f"Skipping disk storage. "
                        f"({vvvv_bytes/1e9:.2f} GB would be needed on disk)")
            eris.vvvv = None
        else:
            logger.info(f"VVVV will be saved to disk ({vvvv_bytes/1e9:.2f} GB). "
                        f"Set on_the_fly_vvvv=True to compute on-the-fly instead.")
            # Create HDF5 temp file for vvvv (no DF feri available in ao2mo path)
            eris.feri = lib.H5TmpFile()
            vvvv_shape = (nvir, nvir, nvir, nvir)
            eris.vvvv = eris.feri.create_dataset('vvvv', vvvv_shape, 'f8')
            _compute_vvvv_block_ao2mo(eris, xtc_obj, jastrow_params,
                                      cc.mol, mo_coeff, nocc, nvir, nmo, cc)
        
        return eris



def _contract_vvvv_t2(cc, t2, eris, out=None):
    """Contraction of (vv|vv) with t2. Handles HDF5-backed, in-memory, and on-the-fly cases."""
    import h5py

    if eris.vvvv is not None:
        if isinstance(eris.vvvv, np.ndarray):
            # In-memory numpy array (legacy small-molecule path)
            vvvv = np.asarray(eris.vvvv)
            return lib.einsum('abcd,ijcd->ijab', vvvv.transpose(0, 2, 1, 3), t2)

        if isinstance(eris.vvvv, h5py.Dataset):
            # HDF5-backed: read blocks from disk
            if out is None:
                out = np.zeros_like(t2)
            nocc = cc.nocc
            nvir = cc.nmo - nocc

            mem_host = cc.max_memory * 1e6
            # Block size: each block loads (blk, nvir, nvir, nvir) floats
            blksize = max(4, int(mem_host / (nvir * nvir * nvir * 8)))
            blksize = min(nvir, blksize)

            from pytc.utils.prefetch import PrefetchIterator, hdf5_slice_loader
            chunks = [(p0, min(p0 + blksize, nvir))
                      for p0 in range(0, nvir, blksize)]
            loader = hdf5_slice_loader(eris.vvvv, axis=0)
            logger.debug("VVVV contraction from disk (blksize=%d, n_blocks=%d)",
                         blksize, len(chunks))
            with PrefetchIterator(chunks, loader) as pit:
                for (p0, p1), vvvv_blk in pit:
                    t0 = time.perf_counter()
                    out[:, :, p0:p1, :] += lib.einsum(
                        'abcd,ijcd->ijab', vvvv_blk.transpose(0, 2, 1, 3), t2)
                    logger.debug("VVVV disk block %d:%d done in %.3f s",
                                 p0, p1, time.perf_counter()-t0)
            return out

    # --- On-the-fly path (eris.vvvv is None) ---
    if out is None:
        out = np.zeros_like(t2)

    
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params

    # Resolve with_df early — needed by both blksize estimation and the loop
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
         with_df = cc._scf.with_df

    # Memory-efficient block size using centralized utility
    _naux = None
    if with_df is not None and hasattr(eris, 'vvL'):
        _naux = eris.vvL.shape[1]
    _n_fused = None
    if hasattr(xtc_obj, 'phi_isdf') and xtc_obj.phi_isdf is not None:
        _n_fused = xtc_obj.phi_isdf.shape[1]
    # For ISDF-XTC, _assemble_2b_tile holds a tc_tile output live while the
    # delta_u tile is computed, then a sum result while both are live.  Each is
    # (blk, nvir, blk, nvir) × f64 = blk²·nvir²·8 bytes.  Add both to the
    # per-tile budget so the setup-time estimate is conservative enough.
    _extra_tile_fn = None
    if _n_fused is not None:
        _isdf_nvir = nvir
        def _extra_tile_fn(blk, _V=_isdf_nvir):
            return 2 * blk * _V * blk * _V * 8
    p_blksize, r_blksize = resolve_vvvv_panel_block_sizes(
        nocc, nvir,
        p_block_size=getattr(cc, 'vvvv_p_block_size', None),
        r_block_size=getattr(cc, 'vvvv_r_block_size', None),
        gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
        naux=_naux,
        n_fused=_n_fused,
        include_eris=False,
        include_accumulators=False,
        extra_tile_bytes_fn=_extra_tile_fn,
    )
    panel_size = p_blksize

    # Pre-unpack L_vv_full if using density fitting to avoid repeated IO/unpacking
    
    L_vv_full = None
    if with_df is not None:
         naux = eris.vvL.shape[1]
         # This might be large (approx 5-10GB for 800 orbitals) but necessary for performance
         L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0) # (nvir, nvir, naux)

    devices = _solver_local_devices()
    logger.debug(
        "VVVV on-the-fly contraction: scheduling %d panel pipelines across %d local devices",
        ((nvir + p_blksize - 1) // p_blksize) * ((nvir + r_blksize - 1) // r_blksize),
        len(devices),
    )
    mo_v = None if with_df is not None else cc.mo_coeff[:, nocc:]

    tile_specs = []
    for p0 in range(0, nvir, p_blksize):
        p1 = min(p0 + p_blksize, nvir)
        for r0 in range(0, nvir, r_blksize):
            r1 = min(r0 + r_blksize, nvir)
            tile_specs.append((p0, p1, r0, r1))

    def issue_tile(spec, device):
        p0, p1, r0, r1 = spec
        ranges = (
            slice(nocc + p0, nocc + p1),
            slice(nocc, nmo),
            slice(nocc + r0, nocc + r1),
            slice(nocc, nmo),
        )
        return xtc_mod.compute_2b_tile(
            xtc_obj, jastrow_params, ranges, device=device, panel_size=panel_size)

    def consume_tile(spec, device, tile_handle, release_gpu_slot):
        p0, p1, r0, r1 = spec
        p_len = p1 - p0
        r_len = r1 - r0
        logger.debug(
            "Contraction tile p[%d:%d] r[%d:%d] on device %s",
            p0, p1, r0, r1, getattr(device, "id", "host"),
        )
        t0 = time.perf_counter()
        vvvv_tile = np.asarray(tile_handle)
        release_gpu_slot()  # GPU pipeline is now free to issue the next tile
        if with_df is not None:
            std_tile = np.tensordot(L_vv_full[p0:p1], L_vv_full[r0:r1], axes=((2,), (2,)))
            vvvv_tile = vvvv_tile[:p_len, :, :r_len, :] + std_tile
            del std_tile
        else:
            std_tile = ao2mo.general(
                cc.mol,
                (mo_v[:, p0:p1], mo_v, mo_v[:, r0:r1], mo_v),
                compact=False,
            )
            vvvv_tile = vvvv_tile[:p_len, :, :r_len, :] + std_tile.reshape(p_len, nvir, r_len, nvir)
            del std_tile

        out[:, :, p0:p1, r0:r1] += lib.einsum(
            'abcd,ijcd->ijab', vvvv_tile.transpose(0, 2, 1, 3), t2
        )
        logger.debug(
            "Tile p[%d:%d] r[%d:%d] done in %.3f s",
            p0, p1, r0, r1, time.perf_counter() - t0,
        )

    # Tile-id round-robin; see the parallel call site in
    # ``jax_xtc_ccsd._contract_vvvv_t2`` for why we no longer pass
    # ``device_key=lambda spec: spec[0]`` — the original locality benefit
    # is now redundant (post Apr-7 per-device ISDF kernel caching) and
    # at ``p_blksize=1`` the p-block grouping monopolises device 0.
    _round_robin_pipeline(tile_specs, issue_tile, consume_tile,
                          devices=devices)

    if L_vv_full is not None:
        del L_vv_full
        
    return out

def _update_amps(cc, t1, t2, eris):
    """
    Restricted CCSD amplitude update with efficient Wvvvv contraction.
    
    Modified from PySCF rccsd.py and ccsd.py:
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
    logger.debug("Starting _update_amps")
    t_start = time.perf_counter()
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()

    # Small intermediates using PySCF logic (safe for RAM)
    t0 = time.perf_counter()
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    logger.debug("imd Foo, Fvv, Fov done in %.3f s", time.perf_counter()-t0)

    Foo[np.diag_indices(nocc)] -= mo_e_o
    # Keep an unshifted copy of Fvv for Lvv initialization
    Fvv_unshifted = Fvv.copy()
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation - terms not involving ovvv
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += eris.fock[nocc:, :nocc].T
    logger.debug("T1 initial terms done in %.3f s", time.perf_counter()-t_start)
    
    eris_ovvo = np.asarray(eris.ovvo)
    eris_oovv = np.asarray(eris.oovv)
    
    t1new += 2*np.einsum('kcai,kc->ia', eris_ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris_oovv, t1)
    
    eris_ovoo = np.asarray(eris.ovoo)
    t1new +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # Pre-allocate large intermediates that fit in RAM (11GB)
    # Wvoov: (a, k, i, c) -> (nvir, nocc, nocc, nvir)
    Wvoov = np.zeros((nvir, nocc, nocc, nvir))
    # Wvovo: (a, k, c, i) -> (nvir, nocc, nvir, nocc)
    Wvovo = np.zeros((nvir, nocc, nvir, nocc))
    
    # Lvv: (a, c) -> (nvir, nvir)
    # Initialize with non-ovvv terms (using unshifted Fvv)
    Lvv = Fvv_unshifted - np.einsum('kc,ka->ac', fov, t1)
    
    # tmp_a: (k, a, i, j) -> (nocc, nvir, nocc, nocc)
    tmp_a = np.zeros((nocc, nvir, nocc, nocc))
    # tmp_b: (k, b, i, j) -> (nocc, nvir, nocc, nocc)
    tmp_b = np.zeros((nocc, nvir, nocc, nocc))
    
    # Prepare Tau for tmp_a/b
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    
    # Add non-ovvv contributions to Wvoov/Wvovo
    # Wvoov += eris.ovvo.transpose(...) etc.
    # Logic copied from rintermediates.py but using arrays
    # Wvoov (akic)
    Wvoov += eris_ovvo.transpose(2,0,3,1)
    Wvoov -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wvoov -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wvoov -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wvoov -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wvoov += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    
    # Wvovo (akci)
    Wvovo += eris_oovv.transpose(2,0,3,1)
    Wvovo -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
    Wvovo -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wvovo -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    logger.debug("Wvoov, Wvovo basic terms done in %.3f s", time.perf_counter()-t_start)



    mem_host = cc.max_memory * 1e6
    # Handle ovvv contributions - branch based on array type
    if isinstance(eris.ovvv, np.ndarray):
        # In-memory path: compute t1new ovvv terms and tmp_a/tmp_b
        # Lvv, Wvoov, Wvovo are computed by imd.* calls later
        eris_ovvv = np.asarray(eris.ovvv)
        t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
        t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
        t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
        t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
        
        # tmp_a, tmp_b for t2new (still needed, not in imd)
        tmp_a = lib.einsum('kdac,ijcd->kaij', eris_ovvv, tau)
        tmp_b = lib.einsum('kcbd,ijcd->kbij', eris_ovvv, tau)
    else:
        # HDF5 blocked loop path — prefetch next block while processing current one
        from pytc.utils.prefetch import PrefetchIterator, hdf5_slice_loader
        blksize = max(4, int(mem_host / (nocc*nvir*nvir*8)))
        blksize = min(nvir, blksize)
        logger.debug("Starting ovvv loop (blksize=%d, prefetched)", blksize)
        t_loop = time.perf_counter()
        chunks = [(p0, min(p0 + blksize, nvir))
                  for p0 in range(0, nvir, blksize)]
        loader = hdf5_slice_loader(eris.ovvv, axis=2)
        with PrefetchIterator(chunks, loader) as pit:
            for (p0, p1), ovvv_blk in pit:
                _process_ovvv_block_prefetched(
                    ovvv_blk, t1, t2, tau, t1new, Lvv,
                    Wvoov, Wvovo, tmp_a, tmp_b, p0, p1)
        logger.debug("ovvv loop done in %.3f s", time.perf_counter()-t_loop)

    t2new = np.zeros_like(t2)
    
    # Handle vovv contributions - branch based on array type
    if isinstance(eris.vovv, np.ndarray):
        # In-memory path (matches working code)
        tmp2  = lib.einsum('kibc,ka->abic', eris_oovv, -t1)
        tmp2 += np.asarray(eris.vovv).transpose(0, 2, 1, 3)
        tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
        t2new = tmp + tmp.transpose(1,0,3,2)
    else:
        # HDF5 blocked loop path — prefetch next block while processing current one
        from pytc.utils.prefetch import PrefetchIterator, hdf5_slice_loader
        mem_host = cc.max_memory * 1e6
        blksize_t2 = max(4, int(mem_host / (nvir*nocc*nvir*8)))
        blksize_t2 = min(nvir, blksize_t2)
        logger.debug("Starting vovv loop (blksize=%d, prefetched)", blksize_t2)
        t_loop = time.perf_counter()
        chunks = [(b0, min(b0 + blksize_t2, nvir))
                  for b0 in range(0, nvir, blksize_t2)]
        loader = hdf5_slice_loader(eris.vovv, axis=2)
        with PrefetchIterator(chunks, loader) as pit:
            for (b0, b1), vovv_slice in pit:
                _process_vovv_block_prefetched(
                    vovv_slice, eris_oovv, t1, t2, t2new, b0, b1)
        # Symmetrize the accumulated t2new from vovv blocks
        t2new = t2new + t2new.transpose(1, 0, 3, 2)
        logger.debug("vovv loop done in %.3f s", time.perf_counter()-t_loop)


    tmp2  = lib.einsum('kcai,jc->akij', eris_ovvo, t1)
    tmp2 += np.asarray(eris.vooo).transpose(0, 2, 1, 3)
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)

    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.vovo).transpose(1,3,0,2)
    logger.debug("t2new basic terms done in %.3f s", time.perf_counter()-t_start)

    # Add W loops
    Loo = imd.Loo(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    
    # Lvv, Wvoov, Wvovo depend on ovvv - use imd for in-memory, or already-computed for blocked
    if isinstance(eris.ovvv, np.ndarray):
        # In-memory: use imd functions (which internally handle ovvv)
        Lvv = imd.Lvv(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
    # else: Lvv, Wvoov, Wvovo were already computed incrementally in the blocked loop
    
    Lvv[np.diag_indices(nvir)] -= mo_e_v
    Woooo = imd.cc_Woooo(t1, t2, eris)

    t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
    t_vvvv = time.perf_counter()
    t2new += _contract_vvvv_t2(cc, tau, eris)
    logger.debug("_contract_vvvv_t2 done in %.3f s", time.perf_counter()-t_vvvv)

    # Use precomputed tmp_a, tmp_b
    t2new -= lib.einsum('kb,kaij->ijab', t1, tmp_a)
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
    logger.debug("Final t1/t2 processing done in %.3f s", time.perf_counter()-t_start)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    # Release the X slice cache at the end of each iteration to
    # free host memory between CCSD steps.
    from pytc.xtc import invalidate_X_cache
    invalidate_X_cache()

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


def _get_slice(dset, sl, axis=0):
    """Helper to slice HDF5 dataset or numpy array safely."""
    if hasattr(dset, 'shape'): 
        idx = [slice(None)] * dset.ndim
        idx[axis] = sl
        return np.asarray(dset[tuple(idx)])
    return dset

def _process_ovvv_block(eris, t1, t2, tau, t1new, Lvv, Wvoov, Wvovo, tmp_a, tmp_b, p0, p1):
    """Process a chunk of ovvv block for amplitude updates.
    
    ovvv indexing: (k, d, a, c) with shape (nocc, nvir, nvir, nvir).
    We slice along axis 2 ('a') to get ovvv_blk with shape (nocc, nvir, blk, nvir).
    
    Key PySCF formulas from rintermediates.py:
    - t1new: 2*einsum('kdac,ikcd->ia') - einsum('kcad,ikcd->ia')
    - Lvv:   2*einsum('kdac,kd->ac') - einsum('kcad,kd->ac')  
    - Wvoov: einsum('kcad,id->akic')
    - Wvovo: einsum('kdac,id->akci')
    - tmp_a: einsum('kdac,ijcd->kaij')  
    - tmp_b: einsum('kcbd,ijcd->kbij')
    """
    # ovvv_blk: (k, d, a_blk, c) - sliced along 'a' (axis 2)
    logger.debug("_process_ovvv_block chunk %d:%d", p0, p1)
    t0 = time.perf_counter()
    ovvv_blk = _get_slice(eris.ovvv, slice(p0, p1), axis=2)  # (nocc, nvir, blk, nvir)
    _process_ovvv_block_prefetched(ovvv_blk, t1, t2, tau, t1new, Lvv,
                                    Wvoov, Wvovo, tmp_a, tmp_b, p0, p1)
    logger.debug("chunk %d:%d done in %.3f s", p0, p1, time.perf_counter()-t0)


def _process_ovvv_block_prefetched(ovvv_blk, t1, t2, tau, t1new, Lvv,
                                    Wvoov, Wvovo, tmp_a, tmp_b, p0, p1):
    """Process an already-loaded ovvv block (used by PrefetchIterator path)."""
    t0 = time.perf_counter()
    
    # 1. Update t1new (ia)
    # PySCF: 2*einsum('kdac,ikcd->ia') - einsum('kcad,ikcd->ia')
    # First term: contract k,d (axes 0,1) with t2's i,k,c,d -> output (i, a_blk)
    t1new[:, p0:p1] += 2*lib.einsum('kdac,ikcd->ia', ovvv_blk, t2)
    # Second term: 'kcad' means axis1='c', axis3='d', contract k,d with t2's k,d 
    t1new[:, p0:p1] +=  -lib.einsum('kcad,ikcd->ia', ovvv_blk, t2)

    t1new[:, p0:p1] += 2*lib.einsum('kdac,kd,ic->ia', ovvv_blk, t1, t1)
    t1new[:, p0:p1] +=  -lib.einsum('kcad,kd,ic->ia', ovvv_blk, t1, t1)

    # 2. Update Lvv (ac)
    # PySCF: 2*einsum('kdac,kd->ac') - einsum('kcad,kd->ac')
    Lvv[p0:p1, :] += 2*lib.einsum('kdac,kd->ac', ovvv_blk, t1)
    Lvv[p0:p1, :] -=   lib.einsum('kcad,kd->ac', ovvv_blk, t1)

    # 3. Update Wvoov (akic)
    # PySCF: einsum('kcad,id->akic')
    Wvoov[p0:p1, :, :, :] += lib.einsum('kcad,id->akic', ovvv_blk, t1)

    # 4. Update Wvovo (akci)
    # PySCF: einsum('kdac,id->akci')
    Wvovo[p0:p1, :, :, :] += lib.einsum('kdac,id->akci', ovvv_blk, t1)
    
    # 5. Update tmp_a (kaij) for t2new
    # PySCF: einsum('kdac,ijcd->kaij')
    tmp_a[:, p0:p1, :, :] += lib.einsum('kdac,ijcd->kaij', ovvv_blk, tau) 
    
    # 6. Update tmp_b (kbij) for t2new
    # PySCF: einsum('kcbd,ijcd->kbij') - iterates over 'b', not 'a'
    # When we slice ovvv along 'a', we get contributions to tmp_b[:, a_blk, :, :]
    # But tmp_b is indexed by 'b'. The mapping is: 'kcbd' with ovvv (k,d,a,c) means
    # k=axis0, c=axis1, b=axis2 (which is our 'a'), d=axis3 (which is our 'c')
    # So we're computing contributions to tmp_b[:, p0:p1, :, :] 
    tmp_b[:, p0:p1, :, :] += lib.einsum('kcbd,ijcd->kbij', ovvv_blk, tau)
    logger.debug("chunk %d:%d done in %.3f s", p0, p1, time.perf_counter()-t0)

def _process_vovv_block(eris, eris_oovv, t1, t2, t2new, b0, b1):
    """Process a chunk of vovv block for t2 updates."""
    # vovv shape is (a, i, b, c) - slice along axis 2 (b)
    logger.debug("_process_vovv_block chunk %d:%d", b0, b1)
    t0 = time.perf_counter()
    vovv_slice = _get_slice(eris.vovv, slice(b0, b1), axis=2)  # (a, i, b_blk, c)
    _process_vovv_block_prefetched(vovv_slice, eris_oovv, t1, t2, t2new, b0, b1)
    logger.debug("chunk %d:%d done in %.3f s", b0, b1, time.perf_counter()-t0)


def _process_vovv_block_prefetched(vovv_slice, eris_oovv, t1, t2, t2new, b0, b1):
    """Process an already-loaded vovv block (used by PrefetchIterator path).

    vovv_slice shape: (nvir, nocc, b_blk, nvir)  — vovv[:, :, b0:b1, :]
    Accumulates into t2new[:, :, :, b0:b1].
    """
    t0 = time.perf_counter()

    # oovv is (k, i, b_blk, c) for this b-slice, t1 is (k, a)
    # einsum('kibc,ka->abic') gives (a, b_blk, i, c)
    oovv_b_blk = eris_oovv[:, :, b0:b1, :]
    tmp2_blk = lib.einsum('kibc,ka->abic', oovv_b_blk, -t1)

    # vovv_slice is (a, i, b_blk, c), transpose to (a, b_blk, i, c)
    tmp2_blk += vovv_slice.transpose(0, 2, 1, 3)

    # Contract: tmp2(a, b_blk, i, c) * t1(j, c) -> (i, j, a, b_blk)
    term = lib.einsum('abic,jc->ijab', tmp2_blk, t1)

    # Accumulate into b_blk slice; caller symmetrizes after full loop
    t2new[:, :, :, b0:b1] += term

    logger.debug("chunk %d:%d done in %.3f s", b0, b1, time.perf_counter()-t0)

def _init_df_eris(eris, with_df, nvir, naux, nocc, nmo, mo_coeff):
    """Initialize DF tensors and HDF5 file."""
    if isinstance(with_df._cderi, str):
        import h5py
        eris.feri = h5py.File(with_df._cderi, 'a')
    elif isinstance(getattr(with_df, '_cderi_to_save', None), str):
        import h5py
        eris.feri = h5py.File(with_df._cderi_to_save, 'a')
    else:
        eris.feri = lib.H5TmpFile()
        
    nvir_pair = nvir * (nvir+1) // 2
    
    Loo = np.empty((naux, nocc, nocc))
    Lov = np.empty((naux, nocc, nvir))
    
    # Use max_memory to estimate chunk size (approximate)
    mem_elements = int(eris.max_memory * 1e6 / 8)
    # Ensure reasonable lower bound for chunking (e.g. 1% of memory or at least some blocks)
    # 4e8 in original code likely meant ~3GB. We replace it with mem_elements.
    chunks = (min(nvir_pair, int(mem_elements/with_df.blockdim)), min(naux, with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair, naux), 'f8', chunks=chunks)
    
    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0, nmo, nmo)
        
        Loo[p0:p1] = Lpq[:, :nocc, :nocc]
        Lov[p0:p1] = Lpq[:, :nocc, nocc:]
        
        Lvv_tril = lib.pack_tril(Lpq[:, nocc:, nocc:])
        eris.vvL[:, p0:p1] = Lvv_tril.T
        
    Loo = Loo.reshape(naux, nocc*nocc)
    Lov = Lov.reshape(naux, nocc*nvir)
    return Loo, Lov


# ---------------------------------------------------------------------------
# Generic tiled ERI block builder
# ---------------------------------------------------------------------------

class _TiledBlockSpec:
    """Describes one ERI block built tile-by-tile via the multi-GPU pipeline.

    All per-block logic (index ranges, padding trim, DF contribution, output
    accumulation) lives here so ``_run_tiled_block_pipeline`` contains only
    the device-dispatch boilerplate.

    Parameters
    ----------
    name : str
        Human-readable block name used in log messages.
    ranges_fn : callable (i0, i1) -> tuple[slice, ...]
        Returns the four MO-index slices for tile ``[i0:i1]``.
    panel_layout : str
        ``"pr"`` or ``"qr"`` — which axes are JIT-padded to ``panel_size``.
    trim_fn : callable (tc_raw, i_len) -> ndarray
        Strips XTC padding to the actual tile shape.
    df_fn : callable (i0, i1) -> ndarray or None
        CPU DF contribution for the tile (same shape as trimmed XTC result).
        Pass ``None`` to skip the DF addition.
    write_fn : callable (i0, i1, tile) -> None
        Accumulates / writes the finished tile.  Called under ``acc_lock``.
    """
    __slots__ = ("name", "ranges_fn", "panel_layout", "trim_fn", "df_fn", "write_fn")

    def __init__(self, name, ranges_fn, panel_layout, trim_fn, df_fn, write_fn):
        self.name        = name
        self.ranges_fn   = ranges_fn
        self.panel_layout = panel_layout
        self.trim_fn     = trim_fn
        self.df_fn       = df_fn
        self.write_fn    = write_fn


def _run_tiled_block_pipeline(blocks, nvir, panel_blk, nocc,
                               xtc_obj, jastrow_params, devices,
                               writer=None):
    """Run a single multi-GPU tiled pipeline over one or more ERI blocks.

    All ``blocks`` are interleaved in one ``_round_robin_pipeline`` call so
    GPUs stay continuously occupied across block boundaries.

    Parameters
    ----------
    blocks : list[_TiledBlockSpec]
        Blocks to compute, in the order their tiles should be scheduled.
    nvir, panel_blk, nocc : int
        Virtual dimension size, tile size, and occupied dimension size.
    xtc_obj, jastrow_params : XTC object and parameters
        Passed through to ``compute_2b_tile``.
    devices : sequence
        Local devices for round-robin dispatch.
    writer : _AsyncHDF5Writer, optional
        When provided, tile writes are submitted to this background writer
        instead of running on the pipeline consumer thread under a local
        lock.  This is the desired mode for HDF5-backed ``write_fn`` (e.g.
        ovvv/vovv in ``_compute_large_blocks``) because:

          * a single writer thread inherently serialises HDF5 access
            (the library is not thread-safe), replacing the previous
            ``acc_lock`` without adding contention;
          * the consume thread returns to the ``host_sem`` pool in
            microseconds, so the dispatch thread never stalls on slot
            availability, keeping all GPUs continuously fed even when
            the backing store is slow.

        When ``writer`` is ``None`` (the default — used by the in-memory
        medium-blocks path) writes run inline under a per-pipeline lock;
        for NumPy target arrays this is fast and avoids spawning an
        unnecessary background thread.
    """
    panel_size = max(nocc, panel_blk)
    acc_lock   = threading.Lock() if writer is None else None

    tile_specs = [
        (blk, i0, min(i0 + panel_blk, nvir))
        for blk in blocks
        for i0 in range(0, nvir, panel_blk)
    ]

    # --- Per-stage consume timers ---------------------------------------
    # Accumulated across all tiles (and all consume threads).  Together
    # with ``_round_robin_pipeline``'s main-thread stats and the
    # ``_AsyncHDF5Writer`` stats this gives a full picture of where each
    # tile's wall-clock seconds went:
    #
    #   gpu2cpu_s : np.asarray(handle) — GPU→CPU DMA
    #   df_s      : blk.df_fn()        — CPU DF contraction (tensordot)
    #   add_s     : np.add(tile, tc_view, out=tile)  — in-place accumulate
    #               (or np.ascontiguousarray(tc_view) on the no-DF branch)
    #   submit_s  : writer.submit(...) — should be ~0; nonzero means the
    #               writer queue is full (writer is the bottleneck)
    #   writefn_s : blk.write_fn(...)  — inline-write path only (writer=None)
    stage_stats = {
        "gpu2cpu_s": 0.0,
        "df_s":      0.0,
        "add_s":     0.0,
        "submit_s":  0.0,
        "writefn_s": 0.0,
        "n_tiles":   0,
        "bytes":     0,
    }
    stage_lock = threading.Lock()

    def issue(spec, device):
        blk, i0, i1 = spec
        return xtc_mod.compute_2b_tile(
            xtc_obj, jastrow_params, blk.ranges_fn(i0, i1),
            device=device, panel_size=panel_size, panel_layout=blk.panel_layout,
        )

    def consume(spec, device, handle, release_gpu_slot):
        blk, i0, i1 = spec
        i_len     = i1 - i0
        t0 = time.perf_counter()
        tc_raw    = np.asarray(handle)             # GPU→CPU transfer
        t1 = time.perf_counter()
        release_gpu_slot()                          # free GPU slot immediately
        tc_view   = blk.trim_fn(tc_raw, i_len)     # strip JIT padding (view)
        if blk.df_fn is not None:
            # IN-PLACE add: allocate the DF buffer once, then accumulate the
            # TC view into it.  This avoids a 3rd host-side 5 GB allocation
            # per tile (which was the main cause of the 50 % host RAM blow-up
            # vs. the pre-refactoring path).
            tile = blk.df_fn(i0, i1)               # fresh contiguous alloc
            t2 = time.perf_counter()
            np.add(tile, tc_view, out=tile)        # tile += tc_view
            t3 = time.perf_counter()
            df_s  = t2 - t1
            add_s = t3 - t2
        else:
            # No DF contribution: materialise a contiguous copy of the view
            # so the padded tc_raw can be freed before the write.
            tile = np.ascontiguousarray(tc_view)
            t3 = time.perf_counter()
            df_s  = 0.0
            add_s = t3 - t1
        del tc_view, tc_raw                        # release padded buffer ASAP
        tile_bytes = tile.nbytes
        if writer is not None:
            # Hand the finished tile off to the background writer and
            # return immediately — the HDF5 write then runs in parallel
            # with the next tiles' GPU→CPU readbacks and DF contractions.
            t_sub0 = time.perf_counter()
            writer.submit(blk.write_fn, i0, i1, tile, _bytes=tile_bytes)
            t_sub1 = time.perf_counter()
            submit_s  = t_sub1 - t_sub0
            writefn_s = 0.0
        else:
            submit_s = 0.0
            t_wf0 = time.perf_counter()
            with acc_lock:
                blk.write_fn(i0, i1, tile)
            t_wf1 = time.perf_counter()
            writefn_s = t_wf1 - t_wf0
        with stage_lock:
            stage_stats["n_tiles"]   += 1
            stage_stats["gpu2cpu_s"] += (t1 - t0)
            stage_stats["df_s"]      += df_s
            stage_stats["add_s"]     += add_s
            stage_stats["submit_s"]  += submit_s
            stage_stats["writefn_s"] += writefn_s
            stage_stats["bytes"]     += tile_bytes

    # Open a pipeline-level issue-stage accumulator so ``_assemble_2b_tile``
    # records its tc_assemble / delta_u_assemble / final_sum per-tile
    # timings.  The scheduler's ``issue_s`` is already wall-time-total for
    # the ``issue(...)`` call itself; this decomposition tells us WHICH
    # sub-call inside the assembler is blocking the main thread tile
    # after tile (the signature that drives ``issue_s ≈ wall``).
    with xtc_mod.issue_stage_stats_scope() as _issue_stage_stats:
        pipeline_stats = _round_robin_pipeline(
            tile_specs, issue, consume, devices=devices, return_stats=True)

    # --- Summary log -----------------------------------------------------
    # Everything needed to answer the three diagnostic questions:
    #   Q1 (is the writer saturated?)     → writer.log_summary() below
    #   Q2 (is HDF5 the bandwidth cap?)   → busy_BW / effective_BW fields
    #   Q3 (where do tile seconds go?)    → gpu2cpu/df/add/submit/writefn
    #                                       + main-thread host_wait/gpu_wait/issue
    block_names = "/".join(b.name for b in blocks)
    n_tiles = stage_stats["n_tiles"]
    total_gb = stage_stats["bytes"] / 1e9
    wall = pipeline_stats["wall_s"]
    logger.info(
        "[tiled-pipeline:%s] n_tiles=%d  wall=%.2fs  total=%.2f GB  "
        "main:{host_wait=%.2fs gpu_wait=%.2fs issue=%.2fs}  "
        "consume_sum:{gpu2cpu=%.2fs df=%.2fs add=%.2fs submit=%.2fs writefn=%.2fs}",
        block_names, n_tiles, wall, total_gb,
        pipeline_stats["host_wait_s"], pipeline_stats["gpu_wait_s"],
        pipeline_stats["issue_s"],
        stage_stats["gpu2cpu_s"], stage_stats["df_s"], stage_stats["add_s"],
        stage_stats["submit_s"], stage_stats["writefn_s"],
    )
    # Issue-stage decomposition (only present when ``_assemble_2b_tile`` is
    # the body of ``issue`` — i.e. medium blocks or any future caller that
    # routes through the assembler).  If every stage is tiny the scheduler's
    # issue cost came from somewhere OTHER than the assembler (e.g. a
    # ``compute_2b_tile`` setup cost, host-side kernel prep) — that
    # discrepancy is itself diagnostic.
    tc_s  = _issue_stage_stats.get("tc_assemble_s",     0.0)
    du_s  = _issue_stage_stats.get("delta_u_assemble_s", 0.0)
    sum_s = _issue_stage_stats.get("final_sum_s",        0.0)
    n_ta  = int(_issue_stage_stats.get("n_tiles",        0))
    if n_ta > 0:
        other_s = pipeline_stats["issue_s"] - (tc_s + du_s + sum_s)
        logger.info(
            "[tiled-pipeline:%s] issue-stage: n_tiles_timed=%d  "
            "tc_assemble=%.2fs  delta_u_assemble=%.2fs  final_sum=%.2fs  "
            "(other_in_issue=%.2fs)",
            block_names, n_ta, tc_s, du_s, sum_s, other_s,
        )
    # host_wait ≈ the smoking gun: if it's a big fraction of ``wall``,
    # the main thread spent most of its time waiting for host_sem slots,
    # which means a downstream stage (writer / disk) is the bottleneck.
    if wall > 0 and pipeline_stats["host_wait_s"] / wall > 0.20:
        logger.info(
            "[tiled-pipeline:%s] host_wait is %.0f%% of wall — downstream "
            "stage (writer/disk) is throttling GPU dispatch",
            block_names, 100.0 * pipeline_stats["host_wait_s"] / wall,
        )


def _compute_large_blocks(eris, xtc_obj, jastrow_params, Lov_reshaped,
                          L_vv_full, nocc, nvir, nmo, panel_blk):
    """Compute and write balanced tiled ovvv / vovv blocks to HDF5.

    The r virtual index is sliced in chunks of ``panel_blk`` (GPU-memory-aware,
    from ``resolve_v3o_panel_block_size``).  Both tensors are tiled in a single
    interleaved ``_run_tiled_block_pipeline`` call so GPUs stay occupied across
    block boundaries.

    Both tensors are written as [:, :, r0:r1, :] slabs on axis 2.

      ovvv tile: (occ_all, vir_all, vir_r_blk, vir_all)  panel_layout="pr"
      vovv tile: (vir_all, occ_all, vir_r_blk, vir_all)  panel_layout="qr"

    panel_size=max(nocc, panel_blk) pads the variable-sized dimensions (p=occ
    for ovvv, q=occ for vovv, and r=vir_r_blk) to a fixed size for JIT shape
    stability.  The large vir_all dimensions remain fixed across all tiles.
    """
    devices  = _solver_local_devices()
    n_tiles  = -(-nvir // panel_blk)
    logger.info(
        "Computing large blocks (blk=%d, panel_size=%d, n_tiles=%d per tensor)...",
        panel_blk, max(nocc, panel_blk), n_tiles,
    )

    vovv_ds = eris.vovv
    ovvv_ds = eris.ovvv

    # vovv: shape (nvir, nocc, nvir, nvir) = (a, i, b, c)
    # panel_layout="qr": q=occ and r=vir_r_blk padded to panel_size
    # DF: vovv[a,i,b,c] = sum_L Lov[L,i,a] * Lvv[L,b,c]
    #   tensordot (naux,nocc,nvir) x (i_len,nvir,naux) -> (nocc,nvir,i_len,nvir)
    #   .transpose(1,0,2,3) -> (nvir, nocc, i_len, nvir)
    def _vovv_df(i0, i1):
        return (np.tensordot(Lov_reshaped, L_vv_full[i0:i1], axes=((0,), (2,)))
                .transpose(1, 0, 2, 3))

    vovv_spec = _TiledBlockSpec(
        name="vovv",
        ranges_fn=lambda i0, i1: (
            slice(nocc, nmo),
            slice(0, nocc),
            slice(nocc + i0, nocc + i1),
            slice(nocc, nmo),
        ),
        panel_layout="qr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "qr", nocc, i_len),
        df_fn=_vovv_df,
        write_fn=lambda i0, i1, tile: vovv_ds.__setitem__(
            (slice(None), slice(None), slice(i0, i1), slice(None)), tile),
    )

    # ovvv: shape (nocc, nvir, nvir, nvir) = (k, d, a, c)
    # panel_layout="pr": p=occ and r=vir_r_blk padded to panel_size
    # DF: ovvv[k,d,a,c] = sum_L Lov[L,k,d] * Lvv[L,a,c]
    #   tensordot (naux,nocc,nvir) x (i_len,nvir,naux) -> (nocc,nvir,i_len,nvir)
    def _ovvv_df(i0, i1):
        return np.tensordot(Lov_reshaped, L_vv_full[i0:i1], axes=((0,), (2,)))

    ovvv_spec = _TiledBlockSpec(
        name="ovvv",
        ranges_fn=lambda i0, i1: (
            slice(0, nocc),
            slice(nocc, nmo),
            slice(nocc + i0, nocc + i1),
            slice(nocc, nmo),
        ),
        panel_layout="pr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "pr", nocc, i_len),
        df_fn=_ovvv_df,
        write_fn=lambda i0, i1, tile: ovvv_ds.__setitem__(
            (slice(None), slice(None), slice(i0, i1), slice(None)), tile),
    )

    # Offload all ovvv/vovv tile writes to a dedicated background thread.
    # The queue depth is intentionally generous (larger than host_sem) so
    # back-pressure comes from host_sem in _round_robin_pipeline, not from
    # the writer; this keeps the GPU pipeline saturated whenever the disk
    # can keep up, and degrades gracefully to backpressure-limited mode
    # only when the disk is the true bottleneck.  See _AsyncHDF5Writer
    # docstring for the full motivation.
    with _AsyncHDF5Writer(max_pending=16, name="ovvv-vovv-writer") as writer:
        _run_tiled_block_pipeline(
            [vovv_spec, ovvv_spec], nvir, panel_blk, nocc,
            xtc_obj, jastrow_params, devices,
            writer=writer,
        )
        # Explicit drain here (even though __exit__ does it) so the
        # writer summary reflects the full run, including any writes
        # still in flight when the pipeline returned.
        writer.drain()
        writer.log_summary(logger.info)


def _compute_medium_blocks_tiled(
        xtc_obj, jastrow_params, Loo, Lov_reshaped, L_vv_full,
        nocc, nvir, nmo, panel_blk, devices):
    """Compute oovv/vvoo/ovov/ovvo/vovo via a single tiled multi-GPU pipeline.

    Each block is sliced over one virtual dimension in chunks of ``panel_blk``.
    All five blocks are represented as ``_TiledBlockSpec`` objects and passed
    to ``_run_tiled_block_pipeline``, which interleaves their tiles in one
    ``_round_robin_pipeline`` call so GPUs stay occupied across block boundaries.

    Panel layout and tile shape per block (``ps = max(nocc, panel_blk)``):
      oovv (nocc,nocc,nvir,nvir) — tile dim-2 (vir): layout="pr" → JIT (ps,nocc,ps,nvir)
      vvoo (nvir,nvir,nocc,nocc) — tile dim-0 (vir): layout="pr" → JIT (ps,nvir,ps,nocc)
      ovov (nocc,nvir,nocc,nvir) — tile dim-1 (vir): layout="qr" → JIT (nocc,ps,ps,nvir)
      ovvo (nocc,nvir,nvir,nocc) — tile dim-2 (vir): layout="pr" → JIT (ps,nvir,ps,nocc)
      vovo (nvir,nocc,nvir,nocc) — tile dim-2 (vir): layout="qr" → JIT (nvir,ps,ps,nocc)
    """
    naux     = Loo.shape[0]
    Lov_flat = Lov_reshaped.reshape(naux, nocc * nvir)  # (naux, nocc*nvir) for ddot

    # Pre-allocate host result arrays (tiles are non-overlapping; write_fn uses =)
    results = {
        'oovv': np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64),
        'vvoo': np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64),
        'ovov': np.zeros((nocc, nvir, nocc, nvir), dtype=np.float64),
        'ovvo': np.zeros((nocc, nvir, nvir, nocc), dtype=np.float64),
        'vovo': np.zeros((nvir, nocc, nvir, nocc), dtype=np.float64),
    }

    n_tiles_per_block = -(-nvir // panel_blk)
    logger.info(
        "Computing medium blocks (oovv/vvoo/ovov/ovvo/vovo): "
        "blk=%d, panel_size=%d, %d tiles/block × 5 blocks = %d tiles total",
        panel_blk, max(nocc, panel_blk), n_tiles_per_block, 5 * n_tiles_per_block,
    )

    # ---- oovv: tile r (dim 2), layout="pr" → JIT shape (ps, nocc, ps, nvir) ----
    # DF: (ij|a_tile b) = Loo[L,ij] ⋅ L_vv[L,a_tile,b]
    def _oovv_df(i0, i1):
        i_len  = i1 - i0
        L_tile = L_vv_full[i0:i1].reshape(i_len * nvir, naux)
        return lib.ddot(Loo.T, L_tile.T).reshape(nocc, nocc, i_len, nvir)

    oovv_spec = _TiledBlockSpec(
        name="oovv",
        ranges_fn=lambda i0, i1: (
            slice(0, nocc), slice(0, nocc),
            slice(nocc + i0, nocc + i1), slice(nocc, nmo),
        ),
        panel_layout="pr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "pr", nocc, i_len),
        df_fn=_oovv_df,
        write_fn=lambda i0, i1, tile: results['oovv'].__setitem__(
            (slice(None), slice(None), slice(i0, i1), slice(None)), tile),
    )

    # ---- vvoo: tile p (dim 0), layout="pr" → JIT shape (ps, nvir, ps, nocc) ----
    # DF: (a_tile b|ij) = L_vv[L,a_tile,b] ⋅ Loo[L,ij]
    def _vvoo_df(i0, i1):
        i_len  = i1 - i0
        L_tile = L_vv_full[i0:i1].reshape(i_len * nvir, naux)
        return lib.ddot(L_tile, Loo).reshape(i_len, nvir, nocc, nocc)

    vvoo_spec = _TiledBlockSpec(
        name="vvoo",
        ranges_fn=lambda i0, i1: (
            slice(nocc + i0, nocc + i1), slice(nocc, nmo),
            slice(0, nocc), slice(0, nocc),
        ),
        panel_layout="pr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "pr", i_len, nocc),
        df_fn=_vvoo_df,
        write_fn=lambda i0, i1, tile: results['vvoo'].__setitem__(
            (slice(i0, i1), slice(None), slice(None), slice(None)), tile),
    )

    # ---- ovov: tile q (dim 1), layout="qr" → JIT shape (nocc, ps, ps, nvir) ----
    # DF: (i a_tile|j b) = Lov[L,i,a_tile] ⋅ Lov[L,j,b]
    def _ovov_df(i0, i1):
        i_len    = i1 - i0
        Lov_tile = Lov_reshaped[:, :, i0:i1].reshape(naux, nocc * i_len)
        return lib.ddot(Lov_tile.T, Lov_flat).reshape(nocc, i_len, nocc, nvir)

    ovov_spec = _TiledBlockSpec(
        name="ovov",
        ranges_fn=lambda i0, i1: (
            slice(0, nocc), slice(nocc + i0, nocc + i1),
            slice(0, nocc), slice(nocc, nmo),
        ),
        panel_layout="qr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "qr", i_len, nocc),
        df_fn=_ovov_df,
        write_fn=lambda i0, i1, tile: results['ovov'].__setitem__(
            (slice(None), slice(i0, i1), slice(None), slice(None)), tile),
    )

    # ---- ovvo: tile r (dim 2), layout="pr" → JIT shape (ps, nvir, ps, nocc) ----
    # DF: (i a|b_tile j) = Lov[L,i,a] ⋅ Lov[L,j,b_tile]
    #   ddot → (nocc,nvir,nocc,i_len) → .transpose(0,1,3,2) → (nocc,nvir,i_len,nocc)
    def _ovvo_df(i0, i1):
        i_len    = i1 - i0
        Lov_tile = Lov_reshaped[:, :, i0:i1].reshape(naux, nocc * i_len)
        return (lib.ddot(Lov_flat.T, Lov_tile)
                .reshape(nocc, nvir, nocc, i_len)
                .transpose(0, 1, 3, 2))

    ovvo_spec = _TiledBlockSpec(
        name="ovvo",
        ranges_fn=lambda i0, i1: (
            slice(0, nocc), slice(nocc, nmo),
            slice(nocc + i0, nocc + i1), slice(0, nocc),
        ),
        panel_layout="pr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "pr", nocc, i_len),
        df_fn=_ovvo_df,
        write_fn=lambda i0, i1, tile: results['ovvo'].__setitem__(
            (slice(None), slice(None), slice(i0, i1), slice(None)), tile),
    )

    # ---- vovo: tile r (dim 2), layout="qr" → JIT shape (nvir, ps, ps, nocc) ----
    # DF: (a i|b_tile j) = Lov[L,i,a] ⋅ Lov[L,j,b_tile]
    #   ddot → (nocc,nvir,nocc,i_len) → .transpose(1,0,3,2) → (nvir,nocc,i_len,nocc)
    def _vovo_df(i0, i1):
        i_len    = i1 - i0
        Lov_tile = Lov_reshaped[:, :, i0:i1].reshape(naux, nocc * i_len)
        return (lib.ddot(Lov_flat.T, Lov_tile)
                .reshape(nocc, nvir, nocc, i_len)
                .transpose(1, 0, 3, 2))

    vovo_spec = _TiledBlockSpec(
        name="vovo",
        ranges_fn=lambda i0, i1: (
            slice(nocc, nmo), slice(0, nocc),
            slice(nocc + i0, nocc + i1), slice(0, nocc),
        ),
        panel_layout="qr",
        trim_fn=lambda tc_raw, i_len: xtc_mod.trim_panel(tc_raw, "qr", nocc, i_len),
        df_fn=_vovo_df,
        write_fn=lambda i0, i1, tile: results['vovo'].__setitem__(
            (slice(None), slice(None), slice(i0, i1), slice(None)), tile),
    )

    _run_tiled_block_pipeline(
        [oovv_spec, vvoo_spec, ovov_spec, ovvo_spec, vovo_spec],
        nvir, panel_blk, nocc, xtc_obj, jastrow_params, devices,
    )
    logger.info("Medium blocks done.")
    return results


def _compute_vvvv_block_df(eris, xtc_obj, jastrow_params, L_vv_full, nocc, nvir, nmo, cc):
    """Compute and write vvvv block to HDF5, block-wise (DF path).

    vvvv shape: (a, b, c, d) = (nvir, nvir, nvir, nvir).
    We iterate over the first index 'a' in blocks to limit memory usage.
    """
    _n_fused = None
    if hasattr(xtc_obj, 'phi_isdf') and xtc_obj.phi_isdf is not None:
        _n_fused = xtc_obj.phi_isdf.shape[1]
    _naux = L_vv_full.shape[2] if L_vv_full is not None else None

    p_blksize, r_blksize = resolve_vvvv_panel_block_sizes(
        nocc, nvir,
        p_block_size=getattr(cc, 'vvvv_p_block_size', None),
        r_block_size=getattr(cc, 'vvvv_r_block_size', None),
        gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
        naux=_naux,
        n_fused=_n_fused,
        include_eris=False,
        include_accumulators=False,
    )
    panel_size = p_blksize
    logger.info(
        "    Writing VVVV to disk (p_blksize=%d, r_blksize=%d, n_p_blocks=%d, n_r_blocks=%d)",
        p_blksize, r_blksize,
        (nvir + p_blksize - 1) // p_blksize,
        (nvir + r_blksize - 1) // r_blksize,
    )

    # If X is an HDF5 dataset, preload it into RAM to avoid 15k+ per-tile
    # GPFS reads (outer-p × inner-r loop would re-read every r-slice per p-slab).
    kernels = xtc_obj.isdf_kernels
    X = kernels.get('X')
    if isinstance(X, h5py.Dataset):
        x_gb = X.size * 8 / 1e9
        logger.info("Preloading X into RAM before VVVV write (%.2f GB) ...", x_gb)
        t_x = time.perf_counter()
        kernels = dict(kernels)
        kernels['X'] = X[:]
        xtc_obj = xtc_obj.replace(isdf_kernels=kernels)
        # Propagate the preloaded X to cc/eris — the same reason as in
        # _make_xtc_eris above (see note there).  Without this, downstream
        # consumers that read ``cc.xtc_obj`` or ``eris.xtc_obj`` would still
        # see the HDF5-backed X and fall back to GPFS reads per tile.
        cc.xtc_obj = xtc_obj
        eris.xtc_obj = xtc_obj
        logger.info("X preload done in %.1f s", time.perf_counter() - t_x)

    ds = eris.vvvv
    devices = _solver_local_devices()

    # --- Async HDF5 slab writer ----------------------------------------
    # Each vvvv slab is up to tens of GB, so we cannot afford the
    # "queue a bunch of slabs and let them drain later" pattern used for
    # the ovvv/vovv small-tile path.  We instead cap the number of slabs
    # alive at any instant to 2 (one being written, one being built) via
    # a dedicated semaphore, and route the write itself through
    # ``_AsyncHDF5Writer`` so the main loop can immediately start
    # allocating + computing the NEXT slab while the previous one is
    # streaming to disk.  The writer thread releases ``slab_sem`` in a
    # ``finally`` after the ``ds[...] = slab`` assignment, so the next
    # ``slab_sem.acquire()`` unblocks as soon as the prior write lands.
    #
    # ``max_pending=2`` for the writer queue is a safety margin — the
    # real back-pressure is ``slab_sem``, which caps total live slabs at
    # 2 (one queued/writing + one being built).
    slab_sem = threading.Semaphore(2)

    def _write_vvvv_slab(slab, p0_local, p1_local, sem=slab_sem):
        try:
            ds[p0_local:p1_local, :, :, :] = slab
        finally:
            sem.release()

    with _AsyncHDF5Writer(max_pending=2, name="vvvv-writer") as writer:
        for p0, p1 in lib.prange(0, nvir, p_blksize):
            # Bound live slabs at 2: blocks here if the previous slab is
            # still waiting in (or being drained from) the writer queue.
            slab_sem.acquire()
            try:
                L_p = L_vv_full[p0:p1]
                vvvv_slab = np.empty((p1 - p0, nvir, nvir, nvir), dtype=np.float64)

                tile_specs = [
                    (p0, p1, r0, min(r0 + r_blksize, nvir))
                    for r0 in range(0, nvir, r_blksize)
                ]

                _t_issue = [0.0]
                _t_gpu_wait = [0.0]
                _t_tensordot = [0.0]
                _t_assign = [0.0]
                _timing_lock = threading.Lock()
                _issued_devices = set()  # only mutated from main thread (issue_tile)
                _logged_devices = set()  # guarded by _timing_lock (consume_tile threads)

                def issue_tile(spec, device, _t=_t_issue,
                               _p0=p0, _p1=p1):
                    _, _, r0, r1 = spec
                    ranges = (
                        slice(nocc + _p0, nocc + _p1),
                        slice(nocc, nmo),
                        slice(nocc + r0, nocc + r1),
                        slice(nocc, nmo),
                    )
                    device_key = getattr(device, "id", "host")
                    if device_key not in _issued_devices:
                        _issued_devices.add(device_key)
                        logger.debug(
                            "VVVV first issue on device %s: p=%d:%d r=%d:%d panel=%d",
                            device_key, _p0, _p1, r0, r1, panel_size,
                        )
                    t0 = time.perf_counter()
                    result = xtc_mod.compute_2b_tile(
                        xtc_obj, jastrow_params, ranges, device=device, panel_size=panel_size)
                    _t[0] += time.perf_counter() - t0
                    return result

                def consume_tile(spec, device, tile_handle, release_gpu_slot,
                                 _slab=vvvv_slab, _L_p=L_p, _p0=p0, _p1=p1):
                    _, _, r0, r1 = spec
                    p_len = _p1 - _p0
                    r_len = r1 - r0
                    device_key = getattr(device, "id", "host")
                    t0 = time.perf_counter()
                    tc_tile = np.asarray(tile_handle)[:p_len, :, :r_len, :]  # GPU→CPU transfer
                    t1 = time.perf_counter()
                    release_gpu_slot()  # GPU pipeline is now free to issue the next tile
                    std_tile = np.tensordot(_L_p, L_vv_full[r0:r1], axes=((2,), (2,)))
                    t2 = time.perf_counter()
                    # vvvv_slab writes are non-overlapping (different r-ranges) — no lock needed
                    _slab[:, :, r0:r1, :] = std_tile + tc_tile
                    t3 = time.perf_counter()
                    with _timing_lock:
                        _t_gpu_wait[0] += t1 - t0
                        _t_tensordot[0] += t2 - t1
                        _t_assign[0] += t3 - t2
                        if device_key not in _logged_devices:
                            _logged_devices.add(device_key)
                            logger.debug(
                                "VVVV first consume on device %s: gpu_wait=%.3fs tensordot=%.3fs assign=%.3fs",
                                device_key, t1 - t0, t2 - t1, t3 - t2,
                            )

                _round_robin_pipeline(tile_specs, issue_tile, consume_tile, devices=devices)
            except BaseException:
                # If tile compute fails before we hand ownership of the
                # slab to the writer, release the semaphore ourselves so
                # we do not wedge the next iteration (or drain) forever.
                slab_sem.release()
                raise

            # Hand the fully-built slab to the writer thread; the writer
            # owns it from here and will release ``slab_sem`` in its
            # finally after the HDF5 assignment lands.  ``_bytes`` feeds
            # the writer's throughput counters so the end-of-run
            # ``log_summary`` can report effective GB/s.
            writer.submit(
                _write_vvvv_slab, vvvv_slab, p0, p1,
                _bytes=vvvv_slab.nbytes,
            )
            # Drop our local reference so only the writer queue pins the
            # slab — otherwise the next-iteration np.empty allocation
            # would transiently hold 3 slabs.
            del vvvv_slab, L_p

            logger.debug(
                "VVVV slab %3d:%3d | dispatch=%.2fs  gpu_wait=%.2fs  "
                "tensordot=%.2fs  assign=%.2fs  (write queued)",
                p0, p1,
                _t_issue[0], _t_gpu_wait[0], _t_tensordot[0], _t_assign[0],
            )
        # Flush any in-flight writes before closing the context so that
        # the "write complete" log line is truthful and so any latched
        # writer error is re-raised on the main thread.
        writer.drain()
        writer.log_summary(logger.info)
    logger.info("    VVVV disk write complete")


def _compute_vvvv_block_ao2mo(eris, xtc_obj, jastrow_params, mol, mo_coeff, nocc, nvir, nmo, cc):
    """Compute and write vvvv block to HDF5, block-wise (ao2mo path).

    vvvv shape: (a, b, c, d) = (nvir, nvir, nvir, nvir).
    We iterate over the first index 'a' in blocks to limit memory usage.
    """
    _n_fused = None
    if hasattr(xtc_obj, 'phi_isdf') and xtc_obj.phi_isdf is not None:
        _n_fused = xtc_obj.phi_isdf.shape[1]

    blksize, _ = estimate_blksize(
        nocc, nvir, 'vvvv',
        gpu_max_memory_mb=getattr(cc, 'gpu_max_memory', None),
        host_max_memory_mb=getattr(cc, 'max_memory', None),
        n_fused=_n_fused)
    blksize = max(4, blksize)
    logger.info(f"    Writing VVVV to disk (blksize={blksize}, "
                f"n_blocks={(nvir+blksize-1)//blksize})")

    from pytc.utils.prefetch import async_read, await_read
    ds = eris.vvvv
    mo_v = mo_coeff[:, nocc:]
    pending_tc = None
    pending_key = None

    # --- Async HDF5 slab writer (see _compute_vvvv_block_df for rationale) -
    # Each slab is up to tens of GB; bound alive slabs at 2 so that the
    # next-iteration ao2mo + TC compute can overlap with the previous
    # slab's HDF5 write without ballooning host memory.
    slab_sem = threading.Semaphore(2)

    def _write_vvvv_slab(slab, p0_local, p1_local, sem=slab_sem):
        try:
            ds[p0_local:p1_local, :, :, :] = slab
        finally:
            sem.release()

    with _AsyncHDF5Writer(max_pending=2, name="vvvv-writer") as writer:
        for p0, p1 in lib.prange(0, nvir, blksize):
            slab_sem.acquire()
            try:
                # Compute only the needed block of standard integrals
                std_blk = ao2mo.general(
                    mol, (mo_v[:, p0:p1], mo_v, mo_v, mo_v), compact=False)
                std_blk = std_blk.reshape(p1-p0, nvir, nvir, nvir)

                ranges = (slice(nocc+p0, nocc+p1), slice(nocc, nmo),
                          slice(nocc, nmo), slice(nocc, nmo))
                if pending_tc is not None and pending_key == (p0, p1):
                    tc_blk = await_read(pending_tc)
                    pending_tc = None
                else:
                    tc_blk = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))

                # Kick off NEXT block's get_2b in background
                next_p0 = p0 + blksize
                if next_p0 < nvir:
                    next_p1 = min(next_p0 + blksize, nvir)
                    next_ranges = (slice(nocc+next_p0, nocc+next_p1),
                                   slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo))
                    pending_tc = async_read(
                        lambda r=next_ranges: np.asarray(
                            xtc_obj.get_2b(jastrow_params, ranges=r)))
                    pending_key = (next_p0, next_p1)

                # Fuse std + tc into the slab that will be handed to the writer.
                # Use += on std_blk to avoid one extra slab-sized allocation.
                std_blk += tc_blk
                del tc_blk
                slab = std_blk
                del std_blk
            except BaseException:
                slab_sem.release()
                raise

            writer.submit(
                _write_vvvv_slab, slab, p0, p1,
                _bytes=slab.nbytes,
            )
            del slab
            logger.debug(f"VVVV block {p0}:{p1} write queued")
        writer.drain()
        writer.log_summary(logger.info)
    logger.info("    VVVV disk write complete")
