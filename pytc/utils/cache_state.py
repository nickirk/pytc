"""Persist HF orbital state (``mo_coeff`` / ``mo_energy`` / ``mo_occ`` / ``e_tot``)
in the ISDF cache so that a cached XTC computation is reproducible across
processes.

Background
----------
Fresh SCF runs can return unitarily-equivalent ``mo_coeff`` with different
column signs or subspace mixings for (near-)degenerate eigenvalues, because
multithreaded LAPACK ``dsyev`` has no canonical tie-breaking.  The SCF
*energy* is invariant under this gauge, but the ISDF cache stores
``xi_phi`` / ``phi_isdf`` built from one *specific* ``mo_coeff``.  Combining
cached kernels with a freshly-computed (different-gauge) ``mo_coeff`` silently
produces ~mHa-scale errors in transcorrelated CCSD.

Recommended usage (driver scripts)::

    from pytc.utils.cache_state import cache_has_mf_state, sync_mf_from_cache

    mf = scf.RHF(mol).density_fit()
    if cache_has_mf_state(save_path):
        mf = sync_mf_from_cache(mf, save_path)   # skip mf.kernel()
    else:
        mf.kernel()                              # first compute
    my_xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)
    isdf_xtc = ISDFXTC.from_xtc(my_xtc, n_rank=n_rank, save_path=save_path)
    isdf_xtc = isdf_xtc.isdf(jastrow_params, ...)

The first (compute) run populates the cached mf-state datasets inside
``ISDFXTC.from_xtc``, right after ``df.isdf_decompose`` returns.  Subsequent
(load) runs adopt that state and become bit-reproducible regardless of BLAS
thread count.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_KEY_MO_COEFF = "mo_coeff_cached"
_KEY_MO_ENERGY = "mo_energy_cached"
_KEY_MO_OCC = "mo_occ_cached"
_KEY_E_TOT = "e_tot_cached"


def cache_has_mf_state(save_path: Optional[str]) -> bool:
    """Return True iff *save_path* has a usable cached mf state.

    A usable state must include at least both cached ``mo_coeff`` and
    ``mo_occ`` — otherwise callers like :func:`prepare_mf` could skip SCF
    based on a partial orbital cache and leave ``mf.mo_occ`` unset, which
    would crash downstream ``XTC.from_pyscf`` when it computes ``nocc``.
    """
    if not save_path or not os.path.exists(save_path):
        return False
    try:
        with h5py.File(save_path, "r") as f:
            return _KEY_MO_COEFF in f and _KEY_MO_OCC in f
    except Exception:
        return False


def cache_has_isdf_kernels(save_path: Optional[str]) -> bool:
    """Return True iff *save_path* already holds ISDF decomposition tensors.

    Used to detect "legacy" caches that were written before this module
    existed: they contain ``xi_phi`` / ``phi_isdf`` / etc. built from some
    specific ``mo_coeff`` gauge, but lack the ``mo_coeff_cached`` dataset
    that pins that gauge down.  In that situation we must NOT overwrite
    the cache with a fresh ``mo_coeff`` — that could lock later reloads to
    the wrong orbital gauge.
    """
    if not save_path or not os.path.exists(save_path):
        return False
    try:
        with h5py.File(save_path, "r") as f:
            return "xi_phi" in f
    except Exception:
        return False


def save_mf_state_to_cache(save_path: Optional[str], mf) -> bool:
    """Persist ``mf.mo_coeff`` / ``mo_energy`` / ``mo_occ`` / ``e_tot`` to *save_path*.

    No-op (returns False) when *save_path* is falsy or when the mf object
    has not been solved yet (``mo_coeff is None``).
    """
    if not save_path:
        return False
    if getattr(mf, "mo_coeff", None) is None:
        return False
    result = save_orbital_state_to_cache(
        save_path,
        mo_coeff=getattr(mf, "mo_coeff", None),
        mo_energy=getattr(mf, "mo_energy", None),
        mo_occ=getattr(mf, "mo_occ", None),
        e_tot=getattr(mf, "e_tot", None),
    )
    if result:
        logger.info("Persisted mf orbital state (mo_coeff etc.) to %s", save_path)
    return result


def save_orbital_state_to_cache(
    save_path: Optional[str],
    mo_coeff=None,
    mo_energy=None,
    mo_occ=None,
    e_tot=None,
) -> bool:
    """Like :func:`save_mf_state_to_cache` but takes raw arrays instead of an mf."""
    if not save_path:
        return False
    if mo_coeff is None:
        return False
    with h5py.File(save_path, "a") as f:
        for key, val in (
            (_KEY_MO_COEFF, mo_coeff),
            (_KEY_MO_ENERGY, mo_energy),
            (_KEY_MO_OCC, mo_occ),
        ):
            if val is None:
                continue
            if key in f:
                del f[key]
            f.create_dataset(key, data=np.asarray(val))
        if e_tot is not None:
            if _KEY_E_TOT in f:
                del f[_KEY_E_TOT]
            f.create_dataset(_KEY_E_TOT, data=float(e_tot))
    return True


def sync_mf_from_cache(mf, save_path: Optional[str]):
    """Overwrite mf's orbital state with cached values (if the cache has them)
    and return the same mf object for explicit chaining.

    Typical usage — skip SCF entirely on reload runs::

        mf = scf.RHF(mol).density_fit()
        if cache_has_mf_state(save_path):
            mf = sync_mf_from_cache(mf, save_path)   # no mf.kernel() needed
        else:
            mf.kernel()                              # first compute

    Returns
    -------
    mf
        The *same* object that was passed in, with ``mo_coeff`` / ``mo_energy``
        / ``mo_occ`` / ``e_tot`` overwritten from the cache when present.
        When the cache has no stored mf state the mf is returned unchanged;
        callers must have run ``mf.kernel()`` themselves in that case.

    Notes
    -----
    * For shape sanity we check that the cached ``mo_coeff`` shape matches the
      number of AOs implied by ``mf.mol.nao_nr()``; on mismatch the sync is
      *not* performed (e.g. the user switched basis sets between runs).
    * Sets ``mf.converged = True`` when a sync happens so pyscf-side callers
      that check convergence do not assume an uninitialised SCF.
    """
    if not cache_has_mf_state(save_path):
        logger.info(
            "sync_mf_from_cache: no cached mf state at %s — returning mf unchanged. "
            "Caller must run mf.kernel() before using mf.",
            save_path,
        )
        return mf

    with h5py.File(save_path, "r") as f:
        cached_mo_coeff = np.array(f[_KEY_MO_COEFF][:])
        cached_mo_energy = (
            np.array(f[_KEY_MO_ENERGY][:]) if _KEY_MO_ENERGY in f else None
        )
        cached_mo_occ = (
            np.array(f[_KEY_MO_OCC][:]) if _KEY_MO_OCC in f else None
        )
        cached_e_tot = float(f[_KEY_E_TOT][()]) if _KEY_E_TOT in f else None

    mol = getattr(mf, "mol", None)
    if mol is not None:
        try:
            expected_nao = int(mol.nao_nr())
        except Exception:
            expected_nao = None
        if expected_nao is not None and cached_mo_coeff.shape[0] != expected_nao:
            logger.warning(
                "sync_mf_from_cache: cached mo_coeff AO-count %d != mol.nao_nr() %d "
                "— basis mismatch? Not syncing; returning mf unchanged.",
                cached_mo_coeff.shape[0], expected_nao,
            )
            return mf

    mf.mo_coeff = cached_mo_coeff
    if cached_mo_energy is not None:
        mf.mo_energy = cached_mo_energy
    if cached_mo_occ is not None:
        mf.mo_occ = cached_mo_occ
    if cached_e_tot is not None:
        mf.e_tot = cached_e_tot
    mf.converged = True
    logger.info(
        "sync_mf_from_cache: loaded cached orbital state from %s "
        "(mo_coeff gauge frozen; SCF skipped)",
        save_path,
    )
    return mf


def prepare_mf(mf, save_path: Optional[str]):
    """One-shot convenience: restore cached mf state when possible, otherwise
    run ``mf.kernel()`` normally.

    Equivalent to::

        if cache_has_mf_state(save_path):
            mf = sync_mf_from_cache(mf, save_path)
            if getattr(mf, "mo_coeff", None) is not None:
                return mf
        mf.kernel()
        return mf

    Useful for driver scripts that want a single call doing the right thing
    on both the first (compute) run and all subsequent (reload) runs.  If the
    cache exists but cannot be applied (e.g. basis/AO-count mismatch caught
    by :func:`sync_mf_from_cache`), this falls back to a normal SCF solve
    rather than returning an unsolved mf.
    """
    if cache_has_mf_state(save_path):
        mf = sync_mf_from_cache(mf, save_path)
        if getattr(mf, "mo_coeff", None) is not None:
            return mf
        logger.info(
            "prepare_mf: sync_mf_from_cache did not populate mo_coeff "
            "(cache present but rejected); falling back to mf.kernel()."
        )
    mf.kernel()
    return mf


def mo_coeff_fingerprint(mo_coeff) -> str:
    """Short printable summary for diagnostics."""
    a = np.asarray(mo_coeff)
    return (
        f"shape={a.shape} sum={a.sum():.12e} norm={np.linalg.norm(a):.12e}"
    )


def check_mo_coeff_matches_cache(
    mo_coeff, save_path: Optional[str], atol: float = 1e-12
) -> bool:
    """Return True if *mo_coeff* is elementwise close to the cached one.

    Emits a warning (not an error) if the cache has a mo_coeff but it doesn't
    match the current one, pointing the user at :func:`sync_mf_from_cache`.
    """
    if not cache_has_mf_state(save_path):
        return True
    with h5py.File(save_path, "r") as f:
        cached = np.array(f[_KEY_MO_COEFF][:])
    fresh = np.asarray(mo_coeff)
    if cached.shape != fresh.shape:
        logger.warning(
            "cached mo_coeff shape %s != fresh shape %s (different basis?)",
            cached.shape, fresh.shape,
        )
        return False
    # rtol=0 so the tolerance behaviour matches the advertised ``atol``;
    # np.allclose defaults to rtol=1e-5 which would let a visibly different
    # mo_coeff pass as "matching" and silently suppress the warning.
    if np.allclose(cached, fresh, atol=atol, rtol=0.0):
        return True
    diff = float(np.linalg.norm(cached - fresh))
    logger.warning(
        "Fresh mo_coeff does not match the mo_coeff stored in the ISDF cache "
        "(|cached - fresh|=%.3e). This is usually SCF gauge non-determinism "
        "(LAPACK eigvec sign/subspace mixing). Call "
        "pytc.utils.cache_state.sync_mf_from_cache(mf, save_path) BEFORE "
        "XTC.from_pyscf to adopt the cached gauge; otherwise expect "
        "~mHa-scale errors in transcorrelated results.",
        diff,
    )
    return False
