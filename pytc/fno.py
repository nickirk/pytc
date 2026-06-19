"""Frozen-natural-orbital (FNO) / MP2-natural-orbital truncation.

``make_fno_mo_coeff`` builds MP2 natural orbitals from an RHF mean field,
keeps only the most important virtuals, and returns a *truncated*
``mf`` copy (plus the matching ``mo_coeff`` / ``mo_occ`` / ``mo_energy``)
that drops straight into the existing ``ISDFXTC.from_pyscf(mf, ...)`` →
``RCCSD(mf, xtc_obj=..., ...)`` pipeline with ``n_orb < n_ao``.

Design note — why a truncated ``mf`` rather than bare arrays: the XTC and
solver code paths read ``mo_coeff`` and ``mo_occ`` from the same source
(``XTC.from_pyscf`` pulls ``mf.mo_coeff``/``mf.mo_occ``;
``solver.xtc_ccsd.RCCSD`` inherits ``cc.mo_coeff``/``cc._scf.mo_occ``
from the ``mf`` passed to it).  Returning one self-consistent truncated
``mf`` keeps those two read-sites in agreement (``len(mo_occ) ==
mo_coeff.shape[1] == n_orb``) with **no API change** anywhere downstream.

The occupied orbitals are kept untouched and first; only the virtual
space is rotated into natural-orbital form and truncated.  The kept
virtual block is then semicanonicalized (diagonalizing the Fock within
it) so the downstream CCSD sees a canonical-ish reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FNOResult:
    """Outcome of an MP2-natural-orbital truncation.

    Attributes:
        mf: shallow copy of the input mean-field with ``mo_coeff`` /
            ``mo_occ`` / ``mo_energy`` replaced by the truncated
            natural-orbital set.  Drop-in for ``ISDFXTC.from_pyscf`` and
            ``solver.xtc_ccsd.RCCSD``.
        mo_coeff: ``(n_ao, nocc + n_keep)`` truncated MO coefficients
            ``[occupied | kept-semicanonical-virtuals]``.
        mo_occ: ``(nocc + n_keep,)`` occupation numbers (occupieds keep
            their input occupations; kept virtuals are 0).
        mo_energy: ``(nocc + n_keep,)`` orbital energies in the
            semicanonical basis (occupieds untouched; kept virtuals are
            the eigenvalues of the in-block Fock).
        nocc: number of occupied orbitals (read from ``mf.mo_occ``).
        n_keep: number of virtuals retained.
        nvir_full: number of virtuals in the full MO space.
        no_occ: MP2 natural-orbital occupations of *all* virtuals,
            sorted most-important-first (descending).  Use this to pick
            ``occ_threshold`` values when scanning.
        kept_mask: boolean mask of length ``nvir_full`` over the
            descending-occupation virtual NOs, True for retained.
    """
    mf: object
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    mo_energy: np.ndarray
    nocc: int
    n_keep: int
    nvir_full: int
    no_occ: np.ndarray
    kept_mask: np.ndarray
    extras: dict = field(default_factory=dict)


def make_fno_mo_coeff(mf, *, n_keep=None, occ_threshold=None):
    """Build MP2 natural orbitals and truncate the virtual space.

    Args:
        mf: PySCF restricted mean-field (``scf.rhf.RHF``); must have been
            converged.  Used for the MP2 calculation, the reference Fock,
            and as the template for the returned truncated copy.
        n_keep: number of virtual natural orbitals to retain.  Ignored if
            ``occ_threshold`` is given.  If both are ``None`` all virtuals
            are kept (the result is just a rotation to the MP2-NO / full
            semicanonical basis — useful as the ``n_keep → all-virtuals``
            limit for validation).
        occ_threshold: retain every virtual whose MP2 natural-orbital
            occupation is ``>= occ_threshold`` (ke-liao's threshold scan).
            Takes precedence over ``n_keep`` when set.

    Returns:
        :class:`FNOResult` with the truncated ``mf`` and matching arrays.

    Raises:
        ValueError: if the mean-field is not restricted, if neither
            truncation argument is compatible, or if ``n_keep`` exceeds
            the virtual count.
    """
    from pyscf import scf as _pyscf_scf

    if not isinstance(mf, _pyscf_scf.rhf.RHF):
        raise ValueError(
            f"make_fno_mo_coeff expects a restricted (RHF) mean-field, "
            f"got {type(mf).__name__}."
        )

    mo_coeff = np.asarray(mf.mo_coeff, dtype=float)
    mo_occ_in = np.asarray(mf.mo_occ, dtype=float)
    nocc = int(np.sum(mo_occ_in > 0))
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    if nvir <= 0:
        raise ValueError("mean-field has no virtual orbitals to truncate.")

    # AO overlap metric.  MO coefficients are orthonormal in *this* metric
    # (C^T S C = I), not the Euclidean one, so density-matrix transforms must
    # carry S on both sides:  gamma_mo = C^T S P_ao S C   (P_ao = C gamma C^T).
    # Operator matrices like the Fock transform one-sidedly: F_mo = C^T F C.
    s1 = np.asarray(mf.get_ovlp(), dtype=float)

    # --- 1. MP2 1-RDM → virtual-virtual block in the MO basis ----------
    from pyscf import mp as _pyscf_mp
    pt = _pyscf_mp.MP2(mf).run()
    rdm1_ao = np.asarray(pt.make_rdm1(ao_repr=True), dtype=float)
    rdm1_mo = mo_coeff.T @ s1 @ rdm1_ao @ s1 @ mo_coeff
    rdm1_vv = rdm1_mo[nocc:, nocc:]
    # Symmetrize to kill any tiny asymmetry from the RDM assembly before
    # diagonalizing.
    rdm1_vv = 0.5 * (rdm1_vv + rdm1_vv.T)

    # --- 2. diagonalize → virtual natural orbitals --------------------
    # Eigenvectors are the NO rotation in the input virtual basis;
    # eigenvalues are the NO occupations (descending = most important).
    occ_no, rot_v = np.linalg.eigh(rdm1_vv)
    order = np.argsort(occ_no)[::-1]
    occ_no = occ_no[order]
    rot_v = rot_v[:, order]

    # --- 3. select the kept virtuals ----------------------------------
    if occ_threshold is not None:
        kept_mask = occ_no >= float(occ_threshold)
    elif n_keep is not None:
        n_keep = int(n_keep)
        if n_keep < 0:
            raise ValueError(f"n_keep must be >= 0, got {n_keep}.")
        if n_keep > nvir:
            raise ValueError(f"n_keep={n_keep} exceeds nvir={nvir}.")
        kept_mask = np.zeros(nvir, dtype=bool)
        kept_mask[:n_keep] = True
    else:
        # No truncation: rotate to the full MP2-NO basis (validation limit).
        kept_mask = np.ones(nvir, dtype=bool)
    n_keep = int(np.sum(kept_mask))

    rot_keep = rot_v[:, kept_mask]                       # (nvir, n_keep), orthonormal cols

    # --- 4. semicanonicalize the kept virtual block -------------------
    # Diagonalize the Fock within the kept-virtual subspace so downstream
    # CCSD starts from a canonical-like reference.  The occupieds are left
    # in the input (canonical) basis and placed first, untouched.
    fock_ao = np.asarray(mf.get_fock(), dtype=float)
    fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
    fock_vv = fock_mo[nocc:, nocc:]
    fock_keep = rot_keep.T @ fock_vv @ rot_keep          # (n_keep, n_keep)
    fock_keep = 0.5 * (fock_keep + fock_keep.T)
    mo_energy_keep, sc_rot = np.linalg.eigh(fock_keep)   # semicanonical NOs

    # MOs: kept virtuals in the input AO basis, rotated to semicanonical.
    mo_virt_in = mo_coeff[:, nocc:] @ rot_keep           # (n_ao, n_keep)
    mo_virt_sc = mo_virt_in @ sc_rot                     # (n_ao, n_keep)

    # --- 5. assemble the truncated orbital set ------------------------
    mo_coeff_fno = np.hstack([mo_coeff[:, :nocc], mo_virt_sc])
    mo_occ_fno = np.concatenate([mo_occ_in[:nocc], np.zeros(n_keep)])
    mo_energy_fno = np.concatenate([np.diag(fock_mo)[:nocc], mo_energy_keep])

    # Sanity: truncated MOs stay orthonormal in the AO-overlap metric.
    orth = mo_coeff_fno.T @ s1 @ mo_coeff_fno
    if not np.allclose(orth, np.eye(mo_coeff_fno.shape[1]), atol=1e-8):
        max_dev = float(np.max(np.abs(orth - np.eye(mo_coeff_fno.shape[1]))))
        raise RuntimeError(
            f"FNO mo_coeff lost orthonormality (max |S - I| = {max_dev:.3e}); "
            "refusing to return a non-orthonormal orbital set."
        )

    # --- 6. truncated mf copy (drop-in for the existing pipeline) -----
    # Use pyscf's own ``mf.copy()``: the generic ``copy.copy`` trips pyscf's
    # ``__getstate__``/``__setstate__``, which nulls the cached in-core ERI
    # tensor (``mf._eri``) to keep pickling light — and that breaks the
    # solver's ``ao2mo`` path.  ``mf.copy()`` shares ``mol`` / ``_eri`` /
    # ``excount`` read-only while letting us override the MO attributes.
    mf_fno = mf.copy()
    mf_fno.mo_coeff = mo_coeff_fno
    mf_fno.mo_occ = mo_occ_fno
    mf_fno.mo_energy = mo_energy_fno

    logger.info(
        "make_fno_mo_coeff: nocc=%d nvir_full=%d n_keep=%d (n_orb %d → %d, %.1f%% of virtuals kept)",
        nocc, nvir, n_keep, nmo, nocc + n_keep, 100.0 * n_keep / max(nvir, 1),
    )

    return FNOResult(
        mf=mf_fno,
        mo_coeff=mo_coeff_fno,
        mo_occ=mo_occ_fno,
        mo_energy=mo_energy_fno,
        nocc=nocc,
        n_keep=n_keep,
        nvir_full=nvir,
        no_occ=occ_no,
        kept_mask=kept_mask,
        extras={"mp2_correlation": float(getattr(pt, "e_corr", float("nan")))},
    )
