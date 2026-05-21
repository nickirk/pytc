"""JAX H̄ intermediates for (XTC-)CCSD and EOM-CCSD.

Mirrors ``pyscf.cc.rintermediates``: a shared library of contraction kernels
used by both the ground-state amplitude update (``jax_xtc_ccsd``) and the
EOM-CCSD σ-vector (``xtc_eom_ccsd``). Keeping a single source of truth here
avoids drift between the two callers.

For the transcorrelated (XTC) reference, intermediates that reference the
non-Hermitian ERI blocks (``vovv``, ``vooo``, ...) need TC-aware substitutions
relative to stock pyscf — see ``xtc_eom_ccsd`` for the Wvvvo/Wovoo/Wvovv/Wooov
ports where this matters. The kernels collected here use only the blocks
that are identical between the Hermitian and TC paths (``ovov``, ``ovoo``,
``ovvv``, ``ovvo``, ``oovv``, ``oooo``), so no substitution is needed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


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
    Wklij += eris_oooo.transpose(0, 2, 1, 3)
    return Wklij


@jax.jit
def _jax_cc_Wvoov(t1, t2, eris_ovvv, eris_ovoo, eris_ovvo, eris_ovov):
    Wakic  = jnp.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= jnp.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += eris_ovvo.transpose(2, 0, 3, 1)
    Wakic -= 0.5 * jnp.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5 * jnp.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -=       jnp.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic +=       jnp.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic


@jax.jit
def _jax_cc_Wvovo(t1, t2, eris_ovvv, eris_ovoo, eris_oovv, eris_ovov):
    Wakci  = jnp.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= jnp.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += eris_oovv.transpose(2, 0, 3, 1)
    Wakci -= 0.5 * jnp.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -=       jnp.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci


# ---------------------------------------------------------------------------
# EE-EOM intermediates (pyscf._IMDS.make_ee port)
# ---------------------------------------------------------------------------
#
# Direct JAX transcription of ``pyscf.cc.eom_rccsd._IMDS.make_ee``
# (pyscf/cc/eom_rccsd.py:1910–2099) with two adaptations:
#
#   1. Untiled — whole-array contractions instead of pyscf's HDF5-backed
#      blocked accumulators. Memory cost is O(no·nv³) for the largest
#      intermediate (``wvOvV``). Tiling is deferred.
#
#   2. TC-aware ERI access — the one conjugate-transpose pattern in pyscf's
#      reference (``eris_ovvv.transpose(2,0,3,1).conj()`` at the wvOvV
#      Hermitian-symmetry line) is replaced by ``eris.vvov.transpose(0,2,1,3)``
#      per the nickirk/pyscf@tc-ccsd convention. All other ERI blocks are
#      used directly (pytc stores ``oovv``, ``ovvo``, ``ovoo``, ``ovov``,
#      ``ovvv``, ``oooo`` as separate TC blocks).


def _make_tau(t2, t1a, t1b, fac=1.0):
    """tau = t2 + fac/2 * (t1a ⊗ t1b + t1b ⊗ t1a)  (pyscf _make_tau)."""
    tau = jnp.einsum('ia,jb->ijab', t1a, t1b)
    tau = tau + tau.transpose(1, 0, 3, 2)
    tau = tau * (fac * 0.5)
    tau = tau + t2
    return tau


@jax.jit
def _jax_make_ee_imds(
    t1, t2,
    eris_fock,
    eris_ovov,
    eris_ovoo,
    eris_oooo,
    eris_ovvo,
    eris_oovv,
    eris_ovvv,
    eris_vvov,
):
    """JAX port of ``_IMDS.make_ee``. Returns the 9 EE H̄ intermediates.

    Parameters
    ----------
    t1, t2 : jnp.ndarray
        Converged CCSD amplitudes.
    eris_fock : jnp.ndarray, shape (nmo, nmo)
    eris_ovov, eris_ovoo, eris_oooo, eris_ovvo, eris_oovv : jnp.ndarray
        Standard chemist-notation ERI blocks (pytc-stored TC versions).
    eris_ovvv : jnp.ndarray, shape (no, nv, nv, nv)
        Full ovvv block (caller materializes from HDF5 if needed).
    eris_vvov : jnp.ndarray, shape (nv, nv, no, nv)
        TC-aware vvov block (= ``ovvv.conj().transpose(2,0,3,1)`` for the
        Hermitian limit; pytc stores it separately for TC).

    Returns
    -------
    (Foo, Fov, Fvv, woOoO, woVoO, woVvO, woVVo, woOoV, wvOvV)
        All as JAX arrays, matching pyscf attribute names on ``_IMDS``.
    """
    nocc = t1.shape[0]
    foo = eris_fock[:nocc, :nocc]
    fov = eris_fock[:nocc, nocc:]
    fvv = eris_fock[nocc:, nocc:]

    tau = _make_tau(t2, t1, t1)
    theta = 2 * t2 - t2.transpose(0, 1, 3, 2)

    # --- ovvv-driven contributions (one-shot, untiled) ------------------
    ovvv = eris_ovvv  # axes (m, e, b, f), value (me|bf)
    Fvv  = 2 * jnp.einsum('mf,mfae->ae', t1, ovvv)
    Fvv -=     jnp.einsum('mf,meaf->ae', t1, ovvv)

    woVoO  = jnp.einsum('mebf,ijef->mbij', ovvv, tau)
    woVvO  = jnp.einsum('jf,mebf->mbej',  t1, ovvv)
    woVVo  = jnp.einsum('jf,mfbe->mbej', -t1, ovvv)

    # --- ovov-driven contributions --------------------------------------
    woOoV  = jnp.einsum('if,mfne->mnie', t1, eris_ovov)
    woOoV += eris_ovoo.transpose(2, 0, 3, 1)

    tmp = jnp.einsum('njbf,mfne->mbej', t2, eris_ovov)
    woVvO -= tmp * 0.5
    woVVo += tmp

    tmp_ovoo = jnp.einsum('menf,jf->menj', eris_ovov, t1)
    woVvO -= jnp.einsum('nb,menj->mbej', t1, tmp_ovoo)
    tmp_ovoo = jnp.einsum('mfne,jf->menj', eris_ovov, t1)
    woVVo += jnp.einsum('nb,menj->mbej', t1, tmp_ovoo)

    ovov_anti = 2 * eris_ovov - eris_ovov.transpose(0, 3, 2, 1)
    woVvO += jnp.einsum('njfb,menf->mbej', theta, ovov_anti) * 0.5

    Fov = jnp.einsum('nf,menf->me', t1, ovov_anti)
    tilab = 0.5 * jnp.einsum('ia,jb->ijab', t1, t1) + t2
    Foo  = jnp.einsum('mief,menf->ni', tilab, ovov_anti)
    Fvv -= jnp.einsum('mnaf,menf->ae', tilab, ovov_anti)

    # --- remaining ovoo / ovvo / oovv contributions to woVvO/woVVo -------
    woVvO -= jnp.einsum('nb,menj->mbej', t1, eris_ovoo)
    woVVo += jnp.einsum('nb,nemj->mbej', t1, eris_ovoo)

    woVvO += eris_ovvo.transpose(0, 2, 1, 3)
    woVVo -= eris_oovv.transpose(0, 2, 3, 1)

    # --- finish F-blocks ------------------------------------------------
    Fov = Fov + fov
    Foo = Foo + foo + 0.5 * jnp.einsum('me,ie->mi', Fov + fov, t1)
    Fvv = Fvv + fvv - 0.5 * jnp.einsum('me,ma->ae', Fov + fov, t1)

    # --- woOoO ----------------------------------------------------------
    woOoO  = jnp.einsum('je,nemi->mnij', t1, eris_ovoo)
    woOoO  = woOoO + woOoO.transpose(1, 0, 3, 2)
    woOoO += eris_oooo.transpose(0, 2, 1, 3)

    # --- woVoO continued ------------------------------------------------
    tmp = jnp.einsum('meni,jneb->mbji', eris_ovoo, t2)
    woVoO -= tmp.transpose(0, 1, 3, 2) * 0.5
    woVoO -= tmp

    ovoo_anti = 2 * eris_ovoo - eris_ovoo.transpose(2, 1, 0, 3)
    woVoO += jnp.einsum('nemi,njeb->mbij', ovoo_anti, theta) * 0.5
    Foo = Foo + jnp.einsum('ne,nemi->mi', t1, ovoo_anti)

    woOoO += jnp.einsum('ijef,menf->mnij', tau, eris_ovov)
    woVoO -= jnp.einsum('nb,mnij->mbij', t1, woOoO)

    tmpoovv = jnp.einsum('njbf,nemf->ejmb', t2, eris_ovov)
    tmpovvo = jnp.einsum('nifb,menf->eimb', theta, ovov_anti)
    tmpovvo = tmpovvo * -0.5 + tmpoovv * 0.5

    woVoO -= jnp.einsum('ie,ejmb->mbij', t1, tmpovvo)
    woVoO -= jnp.einsum('ie,ejmb->mbji', t1, tmpoovv)
    woVoO += eris_ovoo.transpose(3, 1, 2, 0)

    tmpovvo = tmpovvo - eris_ovvo.transpose(1, 3, 0, 2)
    tmpoovv = tmpoovv - eris_oovv.transpose(3, 1, 0, 2)

    woVoO += jnp.einsum('mebj,ie->mbij', eris_ovvo, t1)
    woVoO += jnp.einsum('mjbe,ie->mbji', eris_oovv, t1)
    woVoO += jnp.einsum('me,ijeb->mbij', Fov, t2)

    # --- wvOvV ----------------------------------------------------------
    # ebmf[e,b,m,f] = ovvv[m,e,b,f] = (me|bf)
    ebmf = ovvv.transpose(1, 2, 0, 3)
    wvOvV = jnp.einsum('ebmf,miaf->eiab', ebmf, t2)
    wvOvV = -0.5 * wvOvV.transpose(0, 1, 3, 2) - wvOvV

    # Hermitian-symmetry term: pyscf uses einsum('ebmf->bmfe', ebmf.conj()),
    # i.e. ovvv.transpose(2, 0, 3, 1).conj() — shape (nv, no, nv, nv), value
    # (me|bf)* at element [b, m, f, e]. The TC substitution swaps this with
    # the directly-stored (vv|ov) block. With pytc's layout
    # eris_vvov[a, b, c, d] = (ab|cd) (a, b, d vir; c occ), the equivalent
    # is eris_vvov.transpose(0, 2, 1, 3) — shape (nv, no, nv, nv), value
    # (ab|cd) at element [a, c, b, d], matching the Hermitian form's
    # element-wise content via (me|bf)* = (bf|me) for real integrals.
    wvOvV += eris_vvov.transpose(0, 2, 1, 3)

    tmp = -0.5 * ebmf + ebmf.transpose(1, 0, 2, 3)
    wvOvV += jnp.einsum('efmb,mifa->eiba', tmp, theta)

    wvOvV += jnp.einsum('meni,mnab->eiab', eris_ovoo, tau)
    wvOvV -= jnp.einsum('me,miab->eiab', Fov, t2)
    wvOvV += jnp.einsum('ma,eimb->eiab', t1, tmpovvo)
    wvOvV += jnp.einsum('ma,eimb->eiba', t1, tmpoovv)

    return Foo, Fov, Fvv, woOoO, woVoO, woVvO, woVVo, woOoV, wvOvV


# ---------------------------------------------------------------------------
# Singlet EOM-EE σ-vector
# ---------------------------------------------------------------------------
#
# Direct JAX port of ``pyscf.cc.eom_rccsd.eeccsd_matvec_singlet``
# (pyscf/cc/eom_rccsd.py:1187–1282). The algebra is identical to stock
# pyscf — all TC-awareness lives in the H̄ intermediates above. We accept
# the (vv|vv) block as a separate argument so the caller can choose how to
# materialize it (in-memory ndarray vs. on-the-fly tiled contraction).


@jax.jit
def _jax_eeccsd_matvec_singlet(
    r1, r2,
    t1, t2,
    Foo, Fov, Fvv,
    woOoO, woVoO, woVvO, woVVo, woOoV, wvOvV,
    eris_ovov, eris_ovvv, eris_vvvv,
):
    """Apply the singlet EOM-EE H̄ to (r1, r2). Returns (Hr1, Hr2).

    All inputs are JAX arrays. ``eris_vvvv`` must be the full ``(nv,nv,nv,nv)``
    chemist-notation block as a JAX array (caller's responsibility — for
    HDF5 / on-the-fly modes, materialize before calling).
    """
    # Hr1 from F-blocks
    Hr1  = jnp.einsum('ae,ie->ia', Fvv, r1)
    Hr1 -= jnp.einsum('mi,ma->ia', Foo, r1)
    Hr1 += jnp.einsum('me,imae->ia', Fov, r2) * 2
    Hr1 -= jnp.einsum('me,imea->ia', Fov, r2)

    # tau2 = r2 + (t1 ⊗ r1 + r1 ⊗ t1)
    tau2 = _make_tau(r2, r1, t1, fac=2)

    # (vv|vv) · tau2 contribution to Hr2:  Hr2 += 0.5 · einsum('ijef,aebf->ijab', tau2, vvvv)
    Hr2 = jnp.einsum('ijef,aebf->ijab', tau2, eris_vvvv)

    Hr2 += jnp.einsum('mnij,mnab->ijab', woOoO, r2)
    Hr2 = Hr2 * 0.5

    Hr2 += jnp.einsum('be,ijae->ijab', Fvv, r2)
    Hr2 -= jnp.einsum('mj,imab->ijab', Foo, r2)

    # ovvv-driven contributions (untiled — whole array)
    theta = 2 * r2 - r2.transpose(0, 1, 3, 2)
    Hr1 += jnp.einsum('mfae,mife->ia', eris_ovvv, theta)

    tmp = jnp.einsum('meaf,ijef->maij', eris_ovvv, tau2)
    Hr2 -= jnp.einsum('ma,mbij->ijab', t1, tmp)

    tmp  = jnp.einsum('meaf,me->af', eris_ovvv, r1) * 2
    tmp -= jnp.einsum('mfae,me->af', eris_ovvv, r1)
    Hr2 += jnp.einsum('af,ijfb->ijab', tmp, t2)

    Hr2 -= jnp.einsum('mbij,ma->ijab', woVoO, r1)
    Hr2 += jnp.einsum('ejab,ie->ijab', wvOvV, r1)

    # woVVo / woVvO blocks
    tmp = jnp.einsum('mbej,imea->jiab', woVVo, r2)
    Hr2 += tmp
    Hr2 += 0.5 * tmp.transpose(0, 1, 3, 2)

    woVvO_eff = 0.5 * woVVo + woVvO
    Hr1 += jnp.einsum('maei,me->ia', woVvO_eff, r1) * 2
    Hr2 += jnp.einsum('mbej,imae->ijab', woVvO_eff, theta)

    # woOoV
    Hr1 -= jnp.einsum('mnie,mnae->ia', woOoV, theta)
    tmp = jnp.einsum('nmie,me->ni', woOoV, r1) * 2
    tmp -= jnp.einsum('mnie,me->ni', woOoV, r1)
    Hr2 -= jnp.einsum('ni,njab->ijab', tmp, t2)

    # ovov-driven contributions
    tmp  = jnp.einsum('mfne,mf->en', eris_ovov, r1) * 2
    tmp -= jnp.einsum('menf,mf->en', eris_ovov, r1)
    tmp  = jnp.einsum('en,nb->eb', tmp, t1)
    tmp += jnp.einsum('menf,mnbf->eb', eris_ovov, theta)
    Hr2 -= jnp.einsum('eb,ijea->jiab', tmp, t2)

    tmp = jnp.einsum('nemf,imef->ni', eris_ovov, theta)
    Hr1 -= jnp.einsum('na,ni->ia', t1, tmp)
    Hr2 -= jnp.einsum('mj,miab->ijba', tmp, t2)

    # tau2 / tau coupling
    tau2 = _make_tau(r2, r1, t1, fac=2)
    tmp = jnp.einsum('menf,ijef->mnij', eris_ovov, tau2)
    tau = _make_tau(t2, t1, t1) * 0.5
    Hr2 += jnp.einsum('mnij,mnab->ijab', tmp, tau)

    Hr2 = Hr2 + Hr2.transpose(1, 0, 3, 2)
    return Hr1, Hr2


# ---------------------------------------------------------------------------
# Singlet EOM-EE diagonal preconditioner
# ---------------------------------------------------------------------------
#
# Direct JAX port of the singlet portion of
# ``pyscf.cc.eom_rccsd.eeccsd_diag`` (pyscf/cc/eom_rccsd.py:1559–1662),
# stripped of the triplet and SF outputs. The pyscf reference uses
# ``eris.get_ovvv(slice(p0,p1))`` which assumes triangular-packed storage;
# pytc stores ``ovvv`` and ``vvvv`` as dense 4D arrays, so the corresponding
# einsums work directly without the prange/unpack_tril dance.


@jax.jit
def _jax_eeccsd_diag_singlet(
    t1, t2,
    Foo, Fvv,
    woOoO, woVVo, woVvO,
    eris_ovov, eris_ovvv, eris_vvvv,
):
    """Diagonal of H̄ in the singlet (r1, r2) basis.

    Returns ``(Hr1aa, Hr2ab)``; pack with
    ``EOMEESinglet.amplitudes_to_vector`` to obtain the flat preconditioner
    vector Davidson expects.
    """
    nocc = t1.shape[0]
    nvir = t1.shape[1]

    tau = _make_tau(t2, t1, t1)

    # F-block diagonals (orbital energies dressed by t1,t2)
    Fo = jnp.diag(Foo)
    Fv = jnp.diag(Fvv)

    # 2e Wov corrections to the singles diagonal
    Wovab = jnp.einsum('iaai->ia', woVVo)
    Wovaa = Wovab + jnp.einsum('iaai->ia', woVvO)

    # eia[i,a] = Fv[a] − Fo[i]
    eia = -Fo[:, None] + Fv[None, :]
    Hr1aa = eia + Wovaa

    # Doubles diagonal — ovov-derived blocks
    ijb = jnp.einsum('iejb,ijeb->ijb', eris_ovov, t2)
    jab = jnp.einsum('kajb,kjab->jab', eris_ovov, t2)

    # Hr2ab[i,j,a,b] = −ijb[i,j,b] + Fv[a] − Fo[i] − jab[j,a,b]
    Hr2ab = (-ijb)[:, :, None, :] + Fv[None, None, :, None]
    Hr2ab = Hr2ab + (-Fo)[:, None, None, None] + (-jab)[None, :, :, :]

    # Wov contributions to doubles diagonal
    Hr2ab = Hr2ab + Wovaa[None, :, None, :]   # j,b axis
    Hr2ab = Hr2ab + Wovab[:, None, None, :]   # i,b axis
    # Singlet symmetry: r2[i,j,a,b] = r2[j,i,b,a]
    Hr2ab = Hr2ab + Hr2ab.transpose(1, 0, 3, 2)

    # Woooo contribution
    Wooab = jnp.einsum('ijij->ij', woOoO)
    Hr2ab = Hr2ab + Wooab[:, :, None, None]

    # Wvvab contributions
    Wvvab = jnp.einsum('mnab,manb->ab', tau, eris_ovov)
    # ovvv contribution: tmp[a,b] = sum_m t1[m,b] * ovvv[m,b,a,a]
    tmp = jnp.einsum('mb,mbaa->ab', t1, eris_ovvv)
    Wvvab = Wvvab - tmp - tmp.T
    # vvvv contribution: diagonal in the (a,b) sense
    Wvvab = Wvvab + jnp.einsum('aabb->ab', eris_vvvv)

    Hr2ab = Hr2ab + Wvvab[None, None, :, :]

    return Hr1aa, Hr2ab
