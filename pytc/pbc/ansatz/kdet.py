"""k-point Slater determinant.

A :class:`KSlaterDet` is mathematically a Slater determinant of Bloch
orbitals at a k-point mesh. It is equivalent to a supercell-Gamma
Slater determinant on the Nk-replicated supercell — same trial
wavefunction values at the same electron coordinates — but parameterised
by primitive-cell orbitals to avoid the Nk-fold blow-up of the AO
basis.

For a closed-shell system at a uniform k-mesh:
  * total alpha electrons in the cell = sum_k nocc(k)
  * Slater matrix shape: (n_alpha, n_alpha) complex
  * each column labels one occupied (band_idx, k_idx) pair
  * row i evaluates that orbital at electron position r_i

This module exposes the :class:`KSlaterDet` dataclass and a
:func:`create_slater_det_kpts` factory that takes a PySCF
``pbc.scf.KRHF`` (or ``RHF`` — wrapped to list form for uniformity).
Walker, moves, Jastrow, and local-energy integration come in subsequent
steps; this slice is value-evaluation only, validated against the
existing Gamma-only path at ``Nk = 1``.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from .kgto import KGTO, eval_gto

logger = logging.getLogger(__name__)


@struct.dataclass
class KSlaterDet:
    """Slater determinant on a k-point mesh of Bloch orbitals.

    Attributes:
        mo_coeff_kpts_alpha: ``(Nk, n_ao, n_mo)`` complex stacked MO
            coefficients. Same for beta in the closed-shell case.
        kgto: k-aware Bloch GTO evaluator.
        alpha_occ_bands: ``(n_alpha,)`` band index for each occupied
            alpha orbital.
        alpha_occ_kidx: ``(n_alpha,)`` k-point index for each occupied
            alpha orbital.
        beta_occ_bands, beta_occ_kidx: analogous for beta.
        n_alpha, n_beta: total occupied per spin (= sum_k nocc_per_k).
        n_kpts: number of k-points.
        n_orb: number of MO bands per k-point (uniform across k).
        unrestricted: True if alpha != beta MO coefficients.
    """

    mo_coeff_kpts_alpha: jax.Array
    mo_coeff_kpts_beta: jax.Array
    kgto: KGTO
    alpha_occ_bands: jax.Array
    alpha_occ_kidx: jax.Array
    beta_occ_bands: jax.Array
    beta_occ_kidx: jax.Array
    n_alpha: int = struct.field(pytree_node=False)
    n_beta: int = struct.field(pytree_node=False)
    n_kpts: int = struct.field(pytree_node=False)
    n_orb: int = struct.field(pytree_node=False)
    unrestricted: bool = struct.field(pytree_node=False, default=False)

    @property
    def n_electrons(self):
        return self.n_alpha + self.n_beta


def _normalize_mf_inputs(mf, mo_coeff, mo_occ, kpts):
    """Coerce both single-k (RHF) and multi-k (KRHF) inputs into list form."""
    mo_coeff = mo_coeff if mo_coeff is not None else mf.mo_coeff
    mo_occ = mo_occ if mo_occ is not None else mf.mo_occ
    kpts = kpts if kpts is not None else getattr(mf, 'kpts', None)

    if not isinstance(mo_coeff, list):
        # Gamma-only RHF case
        mo_coeff = [mo_coeff]
        mo_occ = [mo_occ]
        if kpts is None:
            kpts = np.zeros((1, 3))

    kpts_arr = np.atleast_2d(np.asarray(kpts))
    if kpts_arr.shape[0] != len(mo_coeff):
        raise ValueError(
            f"Mismatch: {len(mo_coeff)} mo_coeff blocks but "
            f"{kpts_arr.shape[0]} k-points"
        )
    return mo_coeff, mo_occ, kpts_arr


def _build_occupation_lists(mo_occ):
    """Flatten per-k occupation arrays into (band_idx, k_idx) lists.

    Returns:
        bands, kidx — int32 1D arrays of equal length.
    """
    bands = []
    kidx = []
    for k_idx, occ_k in enumerate(mo_occ):
        occ_k = np.asarray(occ_k)
        for n in np.where(occ_k > 0)[0]:
            bands.append(int(n))
            kidx.append(k_idx)
    return (
        jnp.asarray(bands, dtype=jnp.int32),
        jnp.asarray(kidx, dtype=jnp.int32),
    )


def create_slater_det_kpts(
    mf,
    mo_coeff=None,
    mo_occ=None,
    kpts=None,
    rcut: float = None,
    precision: float = 1e-8,
) -> KSlaterDet:
    """Build a :class:`KSlaterDet` from a PySCF mean-field.

    Accepts both ``pyscf.pbc.scf.RHF`` (single Gamma) and
    ``pyscf.pbc.scf.KRHF`` (multi-k). Single-k inputs are wrapped to
    list form so the rest of the pipeline is uniform.

    Closed-shell only for now: alpha and beta MOs share the same
    coefficients and occupation pattern. Open-shell support is a
    follow-up.

    Args:
        mf: A built mean-field with ``cell``, ``mo_coeff``, ``mo_occ``,
            and (for KRHF) ``kpts``.
        mo_coeff, mo_occ, kpts: Optional overrides.
        rcut: Image-summation cutoff for the underlying KGTO.
        precision: Tolerance used to derive ``rcut``.

    Returns:
        A :class:`KSlaterDet`.
    """
    cell = mf.cell
    if not cell.cart:
        raise ValueError(
            "create_slater_det_kpts currently requires a Cartesian basis "
            "(set cell.cart = True before cell.build())."
        )

    mo_coeff, mo_occ, kpts_arr = _normalize_mf_inputs(mf, mo_coeff, mo_occ, kpts)
    Nk = len(mo_coeff)
    n_ao, n_mo = mo_coeff[0].shape
    for c in mo_coeff:
        if c.shape != (n_ao, n_mo):
            raise ValueError(
                "All k-point mo_coeff blocks must share shape; got "
                f"{[a.shape for a in mo_coeff]}"
            )

    mo_coeff_stack = jnp.asarray(
        np.stack([np.asarray(c) for c in mo_coeff], axis=0)
    ).astype(jnp.complex128)
    # PySCF's KRHF normalises each k-block's MOs so the orbital is unit-norm
    # over the *primitive* cell. For VMC the wavefunction is sampled in the
    # supercell (volume Nk * V_prim), so we scale by 1/sqrt(Nk) to get
    # supercell normalisation. This makes |det| match the supercell-Gamma
    # convention exactly (and is a uniform rescaling, so it commutes with
    # all downstream determinant operations).
    mo_coeff_stack = mo_coeff_stack / jnp.sqrt(jnp.asarray(Nk, dtype=jnp.float64))

    alpha_bands, alpha_kidx = _build_occupation_lists(mo_occ)
    # Closed-shell: beta == alpha
    beta_bands, beta_kidx = alpha_bands, alpha_kidx

    n_alpha = int(alpha_bands.shape[0])
    n_beta = n_alpha
    kgto = KGTO.from_cell(cell, kpts=kpts_arr, rcut=rcut, precision=precision)

    return KSlaterDet(
        mo_coeff_kpts_alpha=mo_coeff_stack,
        mo_coeff_kpts_beta=mo_coeff_stack,
        kgto=kgto,
        alpha_occ_bands=alpha_bands,
        alpha_occ_kidx=alpha_kidx,
        beta_occ_bands=beta_bands,
        beta_occ_kidx=beta_kidx,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_kpts=Nk,
        n_orb=n_mo,
        unrestricted=False,
    )


def eval_slater_matrix(kdet: KSlaterDet, positions: jax.Array, spin: str = 'alpha') -> jax.Array:
    """Evaluate the spin-block Slater matrix at a set of positions.

    Args:
        kdet: KSlaterDet
        positions: ``(n_spin, 3)`` electron coordinates (real-valued).
        spin: ``"alpha"`` or ``"beta"``.

    Returns:
        ``(n_spin, n_spin)`` complex Slater matrix where row i is the
        i-th electron's orbital values across the n_spin occupied
        (band, k) labels.
    """
    if spin == 'alpha':
        C = kdet.mo_coeff_kpts_alpha
        occ_kidx = kdet.alpha_occ_kidx
        occ_bands = kdet.alpha_occ_bands
    else:
        C = kdet.mo_coeff_kpts_beta
        occ_kidx = kdet.beta_occ_kidx
        occ_bands = kdet.beta_occ_bands

    # chi[i, k, a] = chi_{a,k}(r_i): (n_spin, Nk, n_ao) complex
    chi = jax.vmap(lambda r: eval_gto(kdet.kgto, r))(positions)

    # phi_full[i, k, n] = sum_a chi[i, k, a] * C[k, a, n]
    phi_full = jnp.einsum('ika,kan->ikn', chi, C)

    # Pick out occupied (k_idx, band_idx) per orbital column j:
    # slater[i, j] = phi_full[i, occ_kidx[j], occ_bands[j]]
    slater = phi_full[:, occ_kidx, occ_bands]
    return slater


def eval_kdet_value(kdet: KSlaterDet, positions_up: jax.Array,
                    positions_down: jax.Array = None):
    """Slater determinant value(s).

    Args:
        kdet: KSlaterDet
        positions_up: ``(n_alpha, 3)`` alpha-electron positions.
        positions_down: ``(n_beta, 3)`` beta-electron positions. If None,
            only the alpha block is evaluated.

    Returns:
        ``(sign_up, logabs_up, sign_down, logabs_down)`` — complex sign
        and real log magnitude for each spin block. If ``positions_down``
        is None, the down block returns trivial ``(1+0j, 0.0)``.
    """
    s_up = eval_slater_matrix(kdet, positions_up, spin='alpha')
    sign_up, logabs_up = jnp.linalg.slogdet(s_up)

    if positions_down is not None:
        s_down = eval_slater_matrix(kdet, positions_down, spin='beta')
        sign_down, logabs_down = jnp.linalg.slogdet(s_down)
    else:
        sign_down = jnp.asarray(1.0 + 0j)
        logabs_down = jnp.asarray(0.0)

    return sign_up, logabs_up, sign_down, logabs_down
