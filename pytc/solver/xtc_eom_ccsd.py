"""
JAX-accelerated EOM-EE-CCSD on top of the transcorrelated (XTC) reference.

Strategy mirrors ``nickirk/pyscf@tc-ccsd``: pyscf already provides everything
non-σ-vector — Davidson driver, Koopmans initial guess, vector packing,
diagonal preconditioner. We subclass :class:`pyscf.cc.eom_rccsd.EOMEESinglet`
and override only

  * :meth:`make_imds` — builds the EE H̄ intermediates (Foo, Fov, Fvv,
    woOoO, woVoO, woVvO, woVVo, woOoV, wvOvV) with our JAX kernels reading
    pytc's TC-aware ERI blocks (``eris.vovv``, ``eris.vooo`` etc. enter
    automatically because pytc stores those blocks separately).
  * :meth:`matvec` — JAX σ-vector port of
    :func:`pyscf.cc.eom_rccsd.eeccsd_matvec_singlet`.

The (vv|vv)·tau2 contraction is delegated to pytc's existing
``_contract_vvvv_t2`` helper, which already handles in-memory, HDF5-backed,
and on-the-fly modes for the ground-state code.

For the closed-shell 1¹A_g reference of cyclobutadiene at D₄ₕ, EOM-EE here
can return *negative* excitation energies — the 1¹B₁g open-shell singlet
sits below the closed-shell reference and is accessed as a "de-excitation".
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from pyscf.cc import eom_rccsd

from pytc.solver import jax_xtc_ccsd, xtc_ccsd  # noqa: F401  (kept for callers)
from pytc.solver.jax_xtc_rintermediates import (
    _jax_make_ee_imds,
    _jax_eeccsd_matvec_singlet,
    _jax_eeccsd_diag_singlet,
)

logger = logging.getLogger(__name__)


def _materialize_ovvv(eris) -> jnp.ndarray:
    """Pull ``eris.ovvv`` fully into a JAX array.

    For HDF5-backed blocks this allocates O(no·nv³) memory. Fine for the
    M5 validation path (H₂O/sto-3g) and the M6 CBD/avdz target; production
    pCV5Z-scale runs will need a tiled variant analogous to
    ``jax_xtc_ccsd.kernel_process_ovvv_block``.
    """
    return jnp.asarray(np.asarray(eris.ovvv))


class _JAXIMDS:
    """Container for JAX-array H̄ intermediates used by the singlet σ-vector.

    Holds the same attribute names as :class:`pyscf.cc.eom_rccsd._IMDS` so
    pyscf's :func:`eeccsd_diag` and downstream helpers can consume it
    without modification.

    Populated by :meth:`EOMEE.make_imds` — left empty by default so the
    pyscf scaffolding can probe ``.eris`` / ``.t1`` / ``.t2`` early.
    """

    def __init__(self, cc, eris):
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = eris
        self.max_memory = cc.max_memory
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.made_ee_imds = False
        # F-blocks, populated by make_ee
        self.Foo = self.Fov = self.Fvv = None
        # 2e blocks
        self.woOoO = self.woVoO = None
        self.woVvO = self.woVVo = self.woOoV = self.wvOvV = None


def _build_ee_imds(imds: "_JAXIMDS") -> None:
    """Populate ``imds`` with the EE H̄ intermediates (JAX, in-memory).

    Calls :func:`_jax_make_ee_imds` and stores its outputs on ``imds`` under
    pyscf-compatible attribute names so :func:`pyscf.cc.eom_rccsd.eeccsd_diag`
    can consume the result without modification.
    """
    eris = imds.eris
    t1 = jnp.asarray(imds.t1)
    t2 = jnp.asarray(imds.t2)

    fock = jnp.asarray(eris.fock)
    ovov = jnp.asarray(eris.ovov)
    ovoo = jnp.asarray(eris.ovoo)
    oooo = jnp.asarray(eris.oooo)
    ovvo = jnp.asarray(eris.ovvo)
    oovv = jnp.asarray(eris.oovv)
    ovvv = _materialize_ovvv(eris)

    # eris.vvov: pytc stores this TC block explicitly. For pyscf-only eris
    # objects (used in M3d numerical validation), reconstruct it from ovvv
    # via the Hermitian-limit identity vvov = ovvv.conj().transpose(2,0,3,1).
    # pytc's vvov has layout (a, b, c, d) = (ab|cd) with c occupied — shape
    # (nv, nv, no, nv). For pyscf-only eris (M3d validation), reconstruct
    # via Hermitian-limit identity (ab|cd) = (cd|ab) = ovvv[c, d, a, b].
    if getattr(eris, "vvov", None) is not None:
        vvov = jnp.asarray(np.asarray(eris.vvov))
    else:
        vvov = ovvv.transpose(2, 3, 0, 1).conj()

    Foo, Fov, Fvv, woOoO, woVoO, woVvO, woVVo, woOoV, wvOvV = _jax_make_ee_imds(
        t1, t2, fock, ovov, ovoo, oooo, ovvo, oovv, ovvv, vvov,
    )

    imds.Foo = Foo
    imds.Fov = Fov
    imds.Fvv = Fvv
    imds.woOoO = woOoO
    imds.woVoO = woVoO
    imds.woVvO = woVvO
    imds.woVVo = woVVo
    imds.woOoV = woOoV
    imds.wvOvV = wvOvV

    # Cache JAX-array views the σ-vector needs on every Davidson iteration.
    imds._t1_jax = t1
    imds._t2_jax = t2
    imds._eris_ovov = ovov
    imds._eris_ovvv = ovvv
    imds._eris_vvvv = jnp.asarray(_load_full_vvvv(eris))
    imds.made_ee_imds = True


def _load_full_vvvv(eris) -> np.ndarray:
    """Return ``eris.vvvv`` as a dense ``(nv, nv, nv, nv)`` numpy array.

    Handles three storage layouts: pyscf's triangular-packed
    ``(npair, npair)``, an already-dense 4D array (pytc), and an HDF5
    dataset of either shape. The full block is loaded fully into memory —
    OK for the M5 validation target and CBD/avdz; on-the-fly tiling for
    pCV5Z-scale runs is a later concern (matches the ground-state pattern).
    """
    from pyscf import ao2mo

    vvvv = np.asarray(eris.vvvv)
    if vvvv.ndim == 4:
        return vvvv
    nvir = int(np.sqrt(vvvv.shape[0] * 2))
    return ao2mo.restore(1, vvvv, nvir)


def _matvec_eeccsd_singlet(r1, r2, imds):
    """Apply H̄ to the singlet trial vector (r1, r2). Returns (Hr1, Hr2)."""
    return _jax_eeccsd_matvec_singlet(
        r1, r2,
        imds._t1_jax, imds._t2_jax,
        imds.Foo, imds.Fov, imds.Fvv,
        imds.woOoO, imds.woVoO, imds.woVvO, imds.woVVo, imds.woOoV, imds.wvOvV,
        imds._eris_ovov, imds._eris_ovvv, imds._eris_vvvv,
    )


class EOMEE(eom_rccsd.EOMEESinglet):
    """Singlet EOM-EE-CCSD on top of a converged XTC-CCSD reference.

    Subclasses :class:`pyscf.cc.eom_rccsd.EOMEESinglet`, inheriting Davidson,
    initial-guess generation, diagonal preconditioner, and vector packing.
    Only :meth:`make_imds` and :meth:`matvec` are TC-aware.
    """

    def __init__(self, cc):
        if getattr(cc, "t1", None) is None or getattr(cc, "t2", None) is None:
            raise RuntimeError(
                "EOMEE requires a converged XTC-CCSD reference; "
                "call cc.kernel() first."
            )
        super().__init__(cc)

    def make_imds(self, eris=None):
        if eris is None:
            eris = getattr(self._cc, "eris", None)
            if eris is None:
                eris = self._cc.ao2mo()
        imds = _JAXIMDS(self._cc, eris)
        _build_ee_imds(imds)  # M3b — populates imds in place
        return imds

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        r1, r2 = self.vector_to_amplitudes(vector, self.nmo, self.nocc)
        r1j = jnp.asarray(r1)
        r2j = jnp.asarray(r2)
        s1, s2 = _matvec_eeccsd_singlet(r1j, r2j, imds)
        return self.amplitudes_to_vector(np.asarray(s1), np.asarray(s2))

    def get_diag(self, imds=None):
        """Full singlet EOM-EE diagonal preconditioner.

        Direct JAX port of pyscf's :func:`eeccsd_diag` (singlet path only),
        reading pytc's natively-stored 4D ``ovvv`` / ``vvvv`` blocks instead
        of pyscf's triangular-packed layout. Davidson with this preconditioner
        finds the same subset of roots as stock pyscf EOMEE — important when
        the spectrum has near-degenerate or doubles-dominant states the
        simpler F-only diagonal would miss.
        """
        if imds is None:
            imds = self.make_imds()
        Hr1, Hr2 = _jax_eeccsd_diag_singlet(
            imds._t1_jax, imds._t2_jax,
            imds.Foo, imds.Fvv,
            imds.woOoO, imds.woVVo, imds.woVvO,
            imds._eris_ovov, imds._eris_ovvv, imds._eris_vvvv,
        )
        return self.amplitudes_to_vector(np.asarray(Hr1), np.asarray(Hr2))


def eomee(cc, nroots=1, koopmans=False):
    """Shortcut: ``e, r = eomee(cc, nroots=4)``."""
    return EOMEE(cc).kernel(nroots=nroots, koopmans=koopmans)
