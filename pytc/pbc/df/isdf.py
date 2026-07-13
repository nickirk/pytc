"""Periodic ISDF fit machinery (task #21, #proj-isdf-periodic, design
v2.1 section 3): a generic, matrix-free Hermitian-PSD pivoted Cholesky
selector. The Pi^q/eta^q builders and per-q kernel application land in
a later commit (V2).

Design decision (Alice's option (a), design v2.1 section 3): the
existing pytc.df.pivots molecular pair core
(_pivoted_cholesky_pair_pivots_core) is one JIT with semantics far
richer than a (diag, col_eval) skeleton -- dual Cholesky states,
normalized/legacy tie-break ramps, a latched effective-rank prefix, and
dtype/rtol-coupled safety thresholds. Extract-and-rewrap cannot be
byte-identical in behavior or execution placement, so it is not
attempted. This module is a genuinely SEPARATE, simpler generic
primitive; pytc/df/pivots.py is not imported and not modified. The
argmax tie-break RULE (a tiny monotonically increasing ramp added to
the score before argmax, biasing ties toward the higher index) is
copied by inspection from pytc.df.pivots's own selector, not shared by
import -- see pivoted_cholesky_hermitian's docstring.
"""

from __future__ import annotations

import numpy as np


def pivoted_cholesky_hermitian(diag, col_eval, rank, *, rcond=1e-12, ramp_scale=1e-12):
    """Matrix-free pivoted (partial) Cholesky for an implicit N x N
    Hermitian PSD matrix M, given only its diagonal and an on-demand
    column oracle col_eval(j) -> M[:, j] (shape (N,), complex128). M
    itself is never materialized.

    Standard greedy pivoted-Cholesky / low-rank PSD approximation
    algorithm: at each step, select the largest remaining Schur-
    complement diagonal entry as the next pivot, fetch that column of
    the ORIGINAL matrix via col_eval, subtract off the already-selected
    pivots' contribution, and normalize by sqrt(pivot diagonal) to get
    the next column of the Cholesky factor L. The Schur-complement
    diagonal is updated by subtracting |l_t|^2 after each step and
    guarded to stay >= 0 (real, per the design spec) despite roundoff.

    Tie-break rule (copied from pytc.df.pivots's molecular selector, by
    inspection -- not shared code, not an import): a tiny monotonically
    increasing ramp `ramp_scale * arange(n) * max(diag)` is added to the
    score used for argmax, biasing an exact numerical tie toward the
    HIGHER index -- deterministic and reproducible, matching the
    molecular selector's own convention, rather than depending on
    argmax's otherwise implementation-defined first-max behavior.

    Args:
        diag: (n,) real, non-negative (guarded) diagonal of M.
        col_eval: callable, col_eval(j) -> (n,) complex128 array, the
            j-th column of the ORIGINAL M (not the Schur complement).
        rank: requested number of pivots (upper bound; may return fewer
            if the Schur-complement diagonal is numerically exhausted
            first).
        rcond: relative threshold (vs max(diag)) below which a
            candidate pivot's Schur-complement diagonal is treated as
            numerically zero -- selection stops there.
        ramp_scale: tie-break ramp coefficient (see above).

    Returns:
        (pivots, L, n_selected):
            pivots: (n_selected,) int64 array of selected column indices.
            L: (n, n_selected) complex128 array, the partial Cholesky
                factor restricted to selected columns (M[pivots,pivots]
                block satisfies L[pivots,:] @ L[pivots,:].conj().T ==
                M[pivots,pivots] to numerical precision; full
                reconstruction is L @ L.conj().T approx M when rank is
                sufficient).
            n_selected: int, <= rank.

    Raises:
        ValueError: rank > n, diag has a materially negative entry
            (M is not PSD within rcond), or col_eval returns a
            malformed shape/dtype.
    """
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1:
        raise ValueError(f"diag must be 1-D, got shape {diag.shape}.")
    n = diag.shape[0]
    if n == 0:
        raise ValueError("diag must be nonempty.")
    if not np.all(np.isfinite(diag)):
        raise ValueError("diag must be finite.")

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if rank > n:
        raise ValueError(f"rank={rank} exceeds n={n} -- cannot select more pivots than rows.")

    max_diag = float(np.max(diag)) if n > 0 else 0.0
    if max_diag <= 0.0:
        raise ValueError("diag is entirely non-positive -- M appears to be the zero matrix.")

    neg_floor = -rcond * max_diag
    if np.any(diag < neg_floor):
        raise ValueError(
            f"diag contains an entry below -{rcond:.1e}*max(diag)={neg_floor:.3e} -- "
            f"M does not appear to be PSD within the declared rcond."
        )
    diag = np.maximum(diag, 0.0)

    ramp = ramp_scale * np.arange(n, dtype=np.float64) * max_diag
    threshold = rcond * max_diag

    L = np.zeros((n, rank), dtype=np.complex128)
    pivots = np.zeros(rank, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)

    n_selected = 0
    for t in range(rank):
        score = np.where(selected, -np.inf, diag + ramp)
        j = int(np.argmax(score))
        if diag[j] <= threshold:
            break

        col = np.asarray(col_eval(j))
        if col.shape != (n,):
            raise ValueError(f"col_eval({j}) must return shape ({n},), got {col.shape}.")
        col = col.astype(np.complex128)

        if t > 0:
            update = L[:, :t] @ L[j, :t].conj()
        else:
            update = 0.0
        l_t = (col - update) / np.sqrt(diag[j])
        L[:, t] = l_t

        diag = diag - np.abs(l_t) ** 2
        diag = np.maximum(diag, 0.0)

        pivots[t] = j
        selected[j] = True
        n_selected += 1

    return pivots[:n_selected], L[:, :n_selected], n_selected
