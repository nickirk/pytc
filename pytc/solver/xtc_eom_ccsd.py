"""
JAX-accelerated EOM-CCSD on top of the transcorrelated (XTC) reference.

Strategy mirrors ``nickirk/pyscf@tc-ccsd`` (https://github.com/nickirk/pyscf):
the EOM-CC σ-vector algebra is unchanged; only the ERI access switches to
the non-Hermitian-aware blocks that ``pytc.solver.xtc_ccsd`` already stores
(``vovv``, ``vooo``, ``vovo``, ``vvoo`` separate from ``ovvv``, ``ovoo``,
``ovov``, ``oovv``; and ``fvo`` separate from ``fov``).

The driver uses ``pyscf.lib.davidson_nosym1`` (non-symmetric Davidson) on
the σ-vector callable, which can return **negative** eigenvalues when the
EOM root sits below the CCSD reference energy — exactly what's needed for
"de-excitation" states like the 1¹B₁g singlet of CBD at D₄ₕ relative to a
closed-shell 1¹A_g reference.

This is a SKELETON. Milestones:
    M1 = current file (skeleton + imports + class signatures, no σ-vector yet)
    M2 = build F̄ / W̄ intermediates with TC ERI access (port from rintermediates)
    M3 = port singlet σ from pyscf.cc.eom_rccsd.eeccsd_matvec_singlet
    M4 = wire Davidson driver, get *a* root out
    M5 = validate vs pyscf EOM-EE for zero-Jastrow case
    M6 = CBD diradical test

See ``docs/xtc_eom_ccsd_design.md`` for the full plan.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import lib

from pytc.solver import jax_xtc_ccsd, xtc_ccsd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# H̄ intermediates (TC-aware)
# ---------------------------------------------------------------------------
#
# All builders below mirror ``pyscf.cc.rintermediates`` line-for-line,
# substituting the Hermitian conjugate of an ERI block with the corresponding
# pre-built TC block, e.g.::
#
#     pyscf hermitian:     eris.ovvv.conj().transpose(2, 0, 3, 1)
#     pytc TC:             eris.vovv.transpose(0, 2, 1, 3)         # vovv ≠ ovvv* for TC
#
# This is the *entire* substance of the ``nickirk/pyscf@tc-ccsd`` rccsd.py /
# rintermediates.py diff. The EOM σ-vector then uses these in the same way
# as Hermitian EOM-EE.


def _cc_Foo(t1, t2, eris):
    """F̄_oo intermediate. Mirrors rintermediates.cc_Foo."""
    raise NotImplementedError("M2: port from pyscf.cc.rintermediates")


def _cc_Fvv(t1, t2, eris):
    """F̄_vv intermediate. Mirrors rintermediates.cc_Fvv."""
    raise NotImplementedError("M2: port from pyscf.cc.rintermediates")


def _cc_Fov(t1, t2, eris):
    """F̄_ov intermediate. Mirrors rintermediates.cc_Fov."""
    raise NotImplementedError("M2: port from pyscf.cc.rintermediates")


def _Woooo(t1, t2, eris):
    """W_oooo. Uses eris.oooo, eris.ovoo, eris.ovov."""
    raise NotImplementedError("M2")


def _Wvoov(t1, t2, eris):
    """W_voov (= W1ovvo + W2ovvo). Uses eris.ovvo, eris.ovvv, eris.ovoo, eris.ovov."""
    raise NotImplementedError("M2")


def _Wvovo(t1, t2, eris):
    """W_vovo. Uses eris.oovv, eris.ovvv, eris.ovoo, eris.ovov."""
    raise NotImplementedError("M2")


def _Wvvvo(t1, t2, eris):
    """W_vvvo.

    TC substitution: instead of eris.ovvv.conj().transpose(2,0,3,1) use
    eris.vovv.transpose(0,2,1,3).
    """
    raise NotImplementedError("M2")


def _Wovoo(t1, t2, eris):
    """W_ovoo.

    TC substitution: instead of eris.ovoo.conj().transpose(2,0,3,1) use
    eris.vooo.transpose(0,2,1,3).
    """
    raise NotImplementedError("M2")


def _Wvovv(t1, t2, eris):
    """W_vovv. Uses eris.vovv (TC block)."""
    raise NotImplementedError("M2")


def _Wooov(t1, t2, eris):
    """W_ooov. Uses eris.ovov, eris.ovoo."""
    raise NotImplementedError("M2")


def _build_imds(cc, t1, t2, eris):
    """Pack all H̄ intermediates into a dict and return."""
    imds = {
        "Foo":  _cc_Foo(t1, t2, eris),
        "Fvv":  _cc_Fvv(t1, t2, eris),
        "Fov":  _cc_Fov(t1, t2, eris),
        "Woooo": _Woooo(t1, t2, eris),
        "Wvoov": _Wvoov(t1, t2, eris),
        "Wvovo": _Wvovo(t1, t2, eris),
        "Wvvvo": _Wvvvo(t1, t2, eris),
        "Wovoo": _Wovoo(t1, t2, eris),
        "Wvovv": _Wvovv(t1, t2, eris),
        "Wooov": _Wooov(t1, t2, eris),
    }
    return imds


# ---------------------------------------------------------------------------
# σ-vector for EOM-EE singlet
# ---------------------------------------------------------------------------


def _matvec_eeccsd_singlet(r1, r2, imds, eris):
    """Apply H̄ to the singlet trial vector (r1, r2).

    Direct port of pyscf.cc.eom_rccsd.eeccsd_matvec_singlet, with TC-aware
    ERI access through ``imds`` and ``eris``. Returns (σ1, σ2).
    """
    raise NotImplementedError("M3: port from pyscf.cc.eom_rccsd")


def _diag_eeccsd(imds, eris):
    """Diagonal of H̄ in the (r1, r2) basis, for Davidson preconditioner."""
    raise NotImplementedError("M3")


def _initial_guess(nroots, ediag, koopmans=True):
    """Build initial trial vectors. Koopmans = HOMO→LUMO singlet, else random."""
    raise NotImplementedError("M3")


# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------


class EOMEE:
    """EOM-EE-CCSD on top of a converged XTC-CCSD reference.

    Parameters
    ----------
    cc : pytc.solver.jax_xtc_ccsd.RCCSD
        Converged XTC-CCSD calculation (must have ``t1``, ``t2``, ``eris``
        populated).
    """

    def __init__(self, cc: "jax_xtc_ccsd.RCCSD"):
        if getattr(cc, "t1", None) is None or getattr(cc, "t2", None) is None:
            raise RuntimeError(
                "EOMEE requires a converged XTC-CCSD reference; "
                "call cc.kernel() first."
            )
        self.cc = cc
        self.eris = cc.eris
        self.t1 = cc.t1
        self.t2 = cc.t2

        # Davidson convergence controls
        self.conv_tol = 1e-6
        self.max_cycle = 100
        self.max_space = 20

        self._imds = None
        self.e = None              # excitation energies (relative to E_CC)
        self.r = None              # right eigenvectors

    # ---- intermediates ---------------------------------------------------
    def make_imds(self):
        if self._imds is None:
            logger.info("EOMEE: building H̄ intermediates (singlet EE)")
            t0 = time.time()
            self._imds = _build_imds(self.cc, self.t1, self.t2, self.eris)
            logger.info("EOMEE: imds built in %.2f s", time.time() - t0)
        return self._imds

    # ---- σ-vector --------------------------------------------------------
    def matvec(self, r_flat):
        """One H̄ × R application. Used by Davidson."""
        r1, r2 = self._unpack(r_flat)
        s1, s2 = _matvec_eeccsd_singlet(r1, r2, self.make_imds(), self.eris)
        return self._pack(s1, s2)

    def _pack(self, r1, r2) -> np.ndarray:
        return np.concatenate([np.asarray(r1).ravel(), np.asarray(r2).ravel()])

    def _unpack(self, r_flat) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nocc, nvir = self.t1.shape
        n1 = nocc * nvir
        r1 = jnp.asarray(r_flat[:n1]).reshape(nocc, nvir)
        r2 = jnp.asarray(r_flat[n1:]).reshape(nocc, nocc, nvir, nvir)
        return r1, r2

    @property
    def vector_size(self) -> int:
        nocc, nvir = self.t1.shape
        return nocc * nvir + nocc**2 * nvir**2

    # ---- driver ----------------------------------------------------------
    def kernel(self, nroots: int = 1, koopmans: bool = True,
               guess: Optional[np.ndarray] = None):
        """Run Davidson and return (e, R).

        Returns
        -------
        e : np.ndarray, shape (nroots,)
            Excitation energies in Hartree, relative to E_CC. Can be NEGATIVE
            for de-excitation states (e.g. 1¹B₁g of CBD D₄ₕ).
        r : list of (r1, r2) tuples
            Right eigenvectors. r2 normalized so ||(r1, r2)|| = 1.
        """
        self.make_imds()

        if guess is None:
            ediag = _diag_eeccsd(self._imds, self.eris)
            guess = _initial_guess(nroots, ediag, koopmans=koopmans)
        else:
            ediag = _diag_eeccsd(self._imds, self.eris)

        def matvec_batch(xs):
            return [self.matvec(x) for x in xs]

        conv, e, r = lib.davidson_nosym1(
            matvec_batch,
            guess,
            ediag,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=nroots,
        )
        if not all(conv):
            logger.warning("EOMEE: %d/%d roots did not converge",
                           sum(not c for c in conv), nroots)
        self.e = np.asarray(e)
        self.r = [self._unpack(rv) for rv in r]
        return self.e, self.r


# ---------------------------------------------------------------------------
# Convenience top-level
# ---------------------------------------------------------------------------


def eomee(cc, nroots=1, koopmans=True):
    """Shortcut: ``e, r = eomee(cc, nroots=4)``."""
    return EOMEE(cc).kernel(nroots=nroots, koopmans=koopmans)
