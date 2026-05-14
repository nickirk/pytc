"""Radial part of semi-local effective core potentials.

Each radial channel is a sum of Gaussian primitives,

    V(r) = sum_k c_k * r^(n_k - 2) * exp(-zeta_k * r^2),

with integer powers n_k (>= 0). This is the PySCF / GAMESS convention:
the parsed dictionary `mol._ecp[symbol]` stores terms indexed by n_k, and
the radial power applied to r is n_k - 2.

The arrays passed in here are padded across primitives (and across atoms by
the caller). Padding terms have `c = 0` so they contribute nothing to the
sum; we still evaluate them, which is the standard tradeoff for JIT-friendly
kernels.
"""

from __future__ import annotations

import jax.numpy as jnp


# Small floor used when an electron coincides with a nucleus. The r^(n-2)
# polynomial term is regularized as max(r, _R_EPS)^(n-2). For physical
# QMC sampling the probability of r < 1e-8 is vanishing, so the value of
# the floor does not affect statistics; it only prevents nan/inf when JIT
# traces the kernel.
_R_EPS = 1.0e-8


def eval_radial_channel(
    r: jnp.ndarray,
    n_powers: jnp.ndarray,
    zetas: jnp.ndarray,
    coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate one radial channel V(r) at a batch of radii.

    Args:
        r: (...) array of electron-atom distances.
        n_powers: (K,) integer powers n_k.
        zetas: (K,) Gaussian exponents zeta_k.
        coeffs: (K,) linear coefficients c_k. Padding entries have c_k = 0.

    Returns:
        V(r) with shape r.shape.
    """
    r_safe = jnp.maximum(r, _R_EPS)
    # Shape broadcast: r[..., None] vs (K,) -> (..., K)
    r_exp = r_safe[..., None]
    powers = r_exp ** (n_powers - 2)
    gauss = jnp.exp(-zetas * r_exp ** 2)
    terms = coeffs * powers * gauss
    return jnp.sum(terms, axis=-1)


def eval_v_loc(
    r_iA: jnp.ndarray,
    loc_n: jnp.ndarray,
    loc_zeta: jnp.ndarray,
    loc_c: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the local ECP channel V_loc per (electron, atom) pair.

    Args:
        r_iA: (N_e, N_atoms) electron-atom distances.
        loc_n: (N_atoms, K_loc) integer powers, padded with zeros.
        loc_zeta: (N_atoms, K_loc) exponents.
        loc_c: (N_atoms, K_loc) coefficients, padded with zeros.

    Returns:
        V_loc with shape (N_e, N_atoms). Non-ECP atoms (all c = 0) yield 0.
    """
    # Broadcast over electrons:
    # r_iA[..., None]: (N_e, N_atoms, 1); loc_*[None, ...]: (1, N_atoms, K)
    r_exp = jnp.maximum(r_iA, _R_EPS)[..., None]
    powers = r_exp ** (loc_n[None, ...] - 2)
    gauss = jnp.exp(-loc_zeta[None, ...] * r_exp ** 2)
    terms = loc_c[None, ...] * powers * gauss
    return jnp.sum(terms, axis=-1)


def eval_v_nl(
    r_iA: jnp.ndarray,
    nl_n: jnp.ndarray,
    nl_zeta: jnp.ndarray,
    nl_c: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the non-local ECP channels V_l per (electron, atom, l).

    Args:
        r_iA: (N_e, N_atoms) electron-atom distances.
        nl_n: (N_atoms, L_plus_1, K_nl) integer powers, padded with zeros.
        nl_zeta: (N_atoms, L_plus_1, K_nl) exponents.
        nl_c: (N_atoms, L_plus_1, K_nl) coefficients, padded with zeros.

    Returns:
        V_l with shape (N_e, N_atoms, L_plus_1). Channels that do not exist
        on a given atom (all c = 0 along the primitive axis) yield 0.
    """
    # r_iA[..., None, None]: (N_e, N_atoms, 1, 1)
    # nl_*[None, ...]:       (1,  N_atoms, L_plus_1, K_nl)
    r_exp = jnp.maximum(r_iA, _R_EPS)[..., None, None]
    powers = r_exp ** (nl_n[None, ...] - 2)
    gauss = jnp.exp(-nl_zeta[None, ...] * r_exp ** 2)
    terms = nl_c[None, ...] * powers * gauss
    return jnp.sum(terms, axis=-1)


def find_nonlocal_cutoff(
    nl_n: jnp.ndarray,
    nl_zeta: jnp.ndarray,
    nl_c: jnp.ndarray,
    tol: float = 1.0e-5,
    r_max: float = 10.0,
    n_grid: int = 4096,
) -> jnp.ndarray:
    """Per-atom radius r_c^A beyond which all non-local channels are < tol.

    Scans V_l(r) for r in [0, r_max] on a uniform grid and returns the
    smallest r such that max_l |V_l(r)| < tol for all r' > r.

    Args:
        nl_n, nl_zeta, nl_c: (N_atoms, L_plus_1, K_nl) padded ECP arrays.
        tol: threshold in Hartree (default 1e-5, matching QMCPACK).
        r_max: upper end of the scan. Atoms whose channels are still above
            tol at r_max get r_c = r_max (and a warning would be appropriate
            at the call site; we just return r_max here).
        n_grid: number of grid points for the scan.

    Returns:
        (N_atoms,) array of r_c values. Atoms with no non-local channels
        (all c = 0) return 0.0.
    """
    # Build a fake "electron" axis of length n_grid where r_iA[e, A] = r_grid[e].
    r_grid = jnp.linspace(0.0, r_max, n_grid)
    # eval_v_nl expects (N_e, N_atoms) distances. Broadcast r_grid across atoms.
    n_atoms = nl_n.shape[0]
    r_iA = jnp.broadcast_to(r_grid[:, None], (n_grid, n_atoms))
    v_nl = eval_v_nl(r_iA, nl_n, nl_zeta, nl_c)  # (n_grid, N_atoms, L_plus_1)
    max_over_l = jnp.max(jnp.abs(v_nl), axis=-1)  # (n_grid, N_atoms)

    # For each atom, find the largest grid index where max_over_l >= tol.
    # r_c is r_grid at (index + 1), clipped to r_max.
    above = max_over_l >= tol  # (n_grid, N_atoms) bool
    # Index of last True per atom: if no True, treat as -1 -> r_c = 0.
    # We use cummax-style logic via reverse search:
    # for each atom, last_idx = max(i where above[i, A]) or -1.
    grid_idx = jnp.arange(n_grid)[:, None]  # (n_grid, 1)
    masked_idx = jnp.where(above, grid_idx, -1)
    last_idx = jnp.max(masked_idx, axis=0)  # (N_atoms,)
    # r_c = r_grid[last_idx + 1] if last_idx >= 0 else 0
    # (using last_idx + 1 ensures r_c is past the last point where V_l >= tol)
    next_idx = jnp.clip(last_idx + 1, 0, n_grid - 1)
    r_c = jnp.where(last_idx < 0, 0.0, r_grid[next_idx])
    return r_c
