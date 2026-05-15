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

from .kgto import KGTO, eval_gto, eval_gto_grad, eval_gto_lap

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


def _eval_one_pos(kdet: KSlaterDet, r_i: jax.Array, spin_is_alpha: bool):
    """Slater row + grad row + lap row at one electron position.

    Returns three complex arrays of shapes ``(n_spin,)``, ``(n_spin, 3)``,
    ``(n_spin,)``.
    """
    chi_v = eval_gto(kdet.kgto, r_i)           # (Nk, n_ao)
    chi_g = eval_gto_grad(kdet.kgto, r_i)      # (Nk, n_ao, 3)
    chi_l = eval_gto_lap(kdet.kgto, r_i)       # (Nk, n_ao)

    if spin_is_alpha:
        C = kdet.mo_coeff_kpts_alpha
        occ_k = kdet.alpha_occ_kidx
        occ_n = kdet.alpha_occ_bands
    else:
        C = kdet.mo_coeff_kpts_beta
        occ_k = kdet.beta_occ_kidx
        occ_n = kdet.beta_occ_bands

    phi_v = jnp.einsum('ka,kan->kn', chi_v, C)        # (Nk, n_mo)
    phi_g = jnp.einsum('kad,kan->knd', chi_g, C)      # (Nk, n_mo, 3)
    phi_l = jnp.einsum('ka,kan->kn', chi_l, C)        # (Nk, n_mo)

    row = phi_v[occ_k, occ_n]
    grad_row = phi_g[occ_k, occ_n]
    lap_row = phi_l[occ_k, occ_n]
    return row, grad_row, lap_row


def _build_spin_block(kdet: KSlaterDet, positions: jax.Array, spin_is_alpha: bool):
    """Build the spin-block Slater matrix + gradient + Laplacian arrays.

    Args:
        kdet: KSlaterDet
        positions: ``(n_spin, 3)`` electron positions (real).
        spin_is_alpha: True for the alpha block, False for beta.

    Returns:
        slater: ``(n_spin, n_spin)`` complex Slater matrix.
        grad:   ``(n_spin, n_spin, 3)`` complex.
        lap:    ``(n_spin, n_spin)`` complex.
    """
    rows, grad_rows, lap_rows = jax.vmap(
        lambda r: _eval_one_pos(kdet, r, spin_is_alpha)
    )(positions)
    return rows, grad_rows, lap_rows


def eval_kdet_value_and_grad(kdet: KSlaterDet, walker):
    """Full evaluation populating Slater, inverse, det, grad, Laplacian on walker.

    The walker carries complex-typed Slater quantities for KSlaterDet —
    the dataclass fields are dtype-polymorphic, so the same Walker
    class is reused. log_psi stays real (it's log|det|); psi_sign is a
    complex unit-modulus phase.

    Args:
        kdet: KSlaterDet
        walker: Walker (unbatched or batched on axis 0)

    Returns:
        ``((det_sign, det_logabs), updated_walker)``. det_sign is complex,
        det_logabs is real.
    """
    positions = walker.positions
    is_batched = positions.ndim == 3
    n_alpha = kdet.n_alpha

    def _eval_unbatched(positions_u):
        pos_a = positions_u[:n_alpha]
        pos_b = positions_u[n_alpha:]
        slater_up, grad_up, lap_up = _build_spin_block(kdet, pos_a, True)
        slater_dn, grad_dn, lap_dn = _build_spin_block(kdet, pos_b, False)
        return slater_up, grad_up, lap_up, slater_dn, grad_dn, lap_dn

    if is_batched:
        (slater_up, grad_up, lap_up,
         slater_dn, grad_dn, lap_dn) = jax.vmap(_eval_unbatched)(positions)
    else:
        (slater_up, grad_up, lap_up,
         slater_dn, grad_dn, lap_dn) = _eval_unbatched(positions)

    sign_up, logdet_up = jnp.linalg.slogdet(slater_up)
    sign_dn, logdet_dn = jnp.linalg.slogdet(slater_dn)
    inv_up = jnp.linalg.inv(slater_up)
    inv_dn = jnp.linalg.inv(slater_dn)

    det_sign = sign_up * sign_dn
    det_logabs = logdet_up + logdet_dn

    updated_walker = walker.replace(
        slater_up=slater_up,
        slater_down=slater_dn,
        inv_up=inv_up,
        inv_down=inv_dn,
        det_up=(sign_up, logdet_up),
        det_down=(sign_dn, logdet_dn),
        grad_up=grad_up,
        grad_down=grad_dn,
        lap_up=lap_up,
        lap_down=lap_dn,
        log_psi=det_logabs,
        psi_sign=det_sign,
    )
    return (det_sign, det_logabs), updated_walker


def _compute_det_ratio_from_row(new_row, inv, row_idx):
    """det(S')/det(S) for a rank-1 row replacement (complex- or real-safe)."""
    return new_row @ inv[:, row_idx]


def _update_inverse_sherman_morrison(inv, new_row, row_idx, ratio):
    """Sherman-Morrison inverse update after replacing row ``row_idx``.

    Identical to the molecular helper but written here to make the
    complex-dtype contract explicit.
    """
    col_k = inv[:, row_idx]
    row_update = new_row @ inv
    row_update = row_update.at[row_idx].add(-1.0)
    return inv - jnp.outer(col_k, row_update) / ratio


def rank1_update_one_electron_kpts(kdet: KSlaterDet, walker, electron_idx):
    """Sherman-Morrison rank-1 update for KSlaterDet after a one-electron move.

    Mirrors :func:`pytc.ansatz.det.rank1_update_one_electron`. The
    walker's ``positions[electron_idx]`` is assumed to already point at
    the proposed location; this function reads it, computes the new
    Slater row for the appropriate spin block via :func:`_eval_one_pos`,
    and updates the cached Slater / inverse / grad / Laplacian / det
    quantities.

    Args:
        kdet: KSlaterDet
        walker: Walker (unbatched).
        electron_idx: Index of the moved electron (0 <= idx < n_alpha + n_beta).

    Returns:
        ``(total_ratio, det_logabs_new, det_sign_new, updated_walker)``.
        ``total_ratio = det(S')/det(S)`` is complex.
    """
    n_alpha = kdet.n_alpha
    new_pos = walker.positions[electron_idx]
    is_alpha = electron_idx < n_alpha
    local_idx = jnp.where(is_alpha, electron_idx, electron_idx - n_alpha)

    # Compute the candidate new row for BOTH spin blocks; pick later with
    # jnp.where. JAX traces both, which avoids data-dependent branching.
    row_up, grad_row_up, lap_row_up = _eval_one_pos(kdet, new_pos, True)
    row_dn, grad_row_dn, lap_row_dn = _eval_one_pos(kdet, new_pos, False)

    ratio_up = jnp.where(
        is_alpha,
        _compute_det_ratio_from_row(row_up, walker.inv_up, local_idx),
        jnp.asarray(1.0 + 0j),
    )
    ratio_dn = jnp.where(
        is_alpha,
        jnp.asarray(1.0 + 0j),
        _compute_det_ratio_from_row(row_dn, walker.inv_down, local_idx),
    )

    inv_up_new = jnp.where(
        is_alpha,
        _update_inverse_sherman_morrison(walker.inv_up, row_up, local_idx, ratio_up),
        walker.inv_up,
    )
    inv_dn_new = jnp.where(
        is_alpha,
        walker.inv_down,
        _update_inverse_sherman_morrison(walker.inv_down, row_dn, local_idx, ratio_dn),
    )

    slater_up_new = jnp.where(
        is_alpha, walker.slater_up.at[local_idx].set(row_up), walker.slater_up
    )
    slater_dn_new = jnp.where(
        is_alpha, walker.slater_down, walker.slater_down.at[local_idx].set(row_dn)
    )
    grad_up_new = jnp.where(
        is_alpha, walker.grad_up.at[local_idx].set(grad_row_up), walker.grad_up
    )
    grad_dn_new = jnp.where(
        is_alpha, walker.grad_down, walker.grad_down.at[local_idx].set(grad_row_dn)
    )
    lap_up_new = jnp.where(
        is_alpha, walker.lap_up.at[local_idx].set(lap_row_up), walker.lap_up
    )
    lap_dn_new = jnp.where(
        is_alpha, walker.lap_down, walker.lap_down.at[local_idx].set(lap_row_dn)
    )

    sign_up_old, logdet_up_old = walker.det_up
    sign_dn_old, logdet_dn_old = walker.det_down

    # Complex "sign" of a ratio = ratio / |ratio| (the phase). jnp.sign on
    # complex inputs already returns this.
    sign_up_new = jnp.where(
        is_alpha, sign_up_old * jnp.sign(ratio_up), sign_up_old
    )
    logdet_up_new = jnp.where(
        is_alpha, logdet_up_old + jnp.log(jnp.abs(ratio_up)), logdet_up_old
    )
    sign_dn_new = jnp.where(
        is_alpha, sign_dn_old, sign_dn_old * jnp.sign(ratio_dn)
    )
    logdet_dn_new = jnp.where(
        is_alpha, logdet_dn_old, logdet_dn_old + jnp.log(jnp.abs(ratio_dn))
    )

    det_sign_new = sign_up_new * sign_dn_new
    det_logabs_new = logdet_up_new + logdet_dn_new

    updated_walker = walker.replace(
        slater_up=slater_up_new,
        slater_down=slater_dn_new,
        inv_up=inv_up_new,
        inv_down=inv_dn_new,
        det_up=(sign_up_new, logdet_up_new),
        det_down=(sign_dn_new, logdet_dn_new),
        grad_up=grad_up_new,
        grad_down=grad_dn_new,
        lap_up=lap_up_new,
        lap_down=lap_dn_new,
    )

    total_ratio = ratio_up * ratio_dn
    return total_ratio, det_logabs_new, det_sign_new, updated_walker


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
