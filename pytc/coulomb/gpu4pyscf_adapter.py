"""Adapter extracting SCF quantities from a converged mean-field object
for the ISDF/LS-THC Coulomb-integral-factorization pipeline (task #4,
isdf-coulomb-cuda decision 001: "gpu4pyscf owns SCF; this project
consumes mo_coeff and AO values/weights on a grid as its upstream
interface").

Backend-agnostic by design: every function here accepts either a plain
pyscf.scf.hf.RHF (CPU, testable without CUDA) or a gpu4pyscf.scf.hf.RHF
(GPU) and returns host numpy regardless. gpu4pyscf keeps mo_coeff/grid
data on-device as cupy arrays until explicitly converted (gpu4pyscf's
own gpu4pyscf/lib/utils.py:to_cpu uses the same isinstance(val,
cupy.ndarray) + .get() idiom _to_host mirrors here, without importing
cupy so this module stays importable on CPU-only hosts).
"""

import numpy as np
from pyscf import dft, df


def _to_host(x):
    """Convert a possibly-cupy array to host numpy; numpy arrays and
    plain Python scalars pass through via np.asarray unchanged."""
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def get_mo_coeff(mf):
    """Host-numpy MO coefficients, shape (n_ao, n_mo).

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).

    Raises:
        ValueError: if mf.mo_coeff is None (mf.kernel() not yet run).
    """
    if mf.mo_coeff is None:
        raise ValueError("mf.mo_coeff is None -- run mf.kernel() first.")
    return _to_host(mf.mo_coeff)


def get_grid_ao_values_and_weights(mf, grid_lvl=2, deriv=0):
    """DFT-grid AO values, weights, and coordinates.

    Follows pytc's own from_pyscf convention exactly (pytc/tc.py:395-409):
    dft.gen_grid.Grids(mol) built at `grid_lvl`, dft.numint.eval_ao at
    the resulting coords.

    Deliberately uses PLAIN pyscf.dft here, never gpu4pyscf.dft, even
    when `mf` came from gpu4pyscf -- gpu4pyscf.dft.gen_grid pads the
    grid with zero-weight ghost points and re-sorts points into atomic
    groups for its own GPU integration scheme (an internal performance
    detail), so its grid point count/order does not match plain
    pyscf's Grids at the same level. Using pyscf.dft keeps the grid
    backend-independent and directly comparable to pytc's existing
    dense/DF reference paths (this step is CPU-bound regardless of
    which backend ran the SCF, so there's no performance reason to use
    the GPU grid here).

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).
            Only mf.mol is used.
        grid_lvl: dft.gen_grid.Grids level (pytc's default: 2).
        deriv: AO derivative order forwarded to dft.numint.eval_ao.
            0 = values only, shape (n_grid, n_ao). >0 = shape
            (comp, n_grid, n_ao) (see pyscf's eval_ao for the component
            ordering convention).

    Returns:
        (ao_values, weights, coords).
    """
    mol = mf.mol
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_lvl
    grids.build()
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    ao_values = np.asarray(dft.numint.eval_ao(mol, coords, deriv=deriv))
    return ao_values, weights, coords


def get_naux(with_df):
    """Number of auxiliary (density-fitting) basis functions, backend-
    agnostic.

    gpu4pyscf.df.df.DF exposes a `.naux` attribute (set during
    `.build()`); plain pyscf.df.df.DF exposes `.get_naoaux()` instead
    and has no `.naux` attribute -- this checks for the attribute
    first so a gpu4pyscf DF is never made to instantiate a method it
    doesn't have.
    """
    naux = getattr(with_df, "naux", None)
    if naux is not None:
        return int(naux)
    return int(with_df.get_naoaux())


def stream_df_cderi_blocks(mf, auxbasis="weigend", blksize=None):
    """Yield Cholesky-factorized density-fitting blocks L_P^{pq} one
    aux-index block at a time, instead of materializing the full
    (naux, nao_pair) tensor.

    Backend dispatch: gpu4pyscf.df.df.DF.loop() yields cupy
    (unpacked_tensor, packed_slab) tuples; plain pyscf.df.df.DF.loop()
    yields a packed numpy block directly. Both are converted through
    _to_host so callers see numpy regardless of backend -- the same
    "loop over aux blocks, consume packed slab" shape pytc's own DF-CCSD
    path already relies on (pytc/solver/xtc_ccsd.py's _init_df_eris).

    Reuses mf.with_df if the mean-field already carries one (same
    fallback convention as pytc/solver/xtc_ccsd.py's density_fit());
    otherwise builds a fresh DF object with `auxbasis`, dispatched to
    the SAME backend as `mf` -- a gpu4pyscf mf gets a
    gpu4pyscf.df.df.DF, not pyscf's, since gpu4pyscf's DF assumes a
    cupy-resident integral setup and isn't a drop-in substitute.

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).
        auxbasis: Auxiliary basis name, used only when `mf` has no
            existing with_df.
        blksize: Forwarded to with_df.loop(); None lets the backend's
            own memory-based heuristic choose.

    Yields:
        (naux_block, nao_pair) host-numpy packed lower-triangular
        cderi blocks.
    """
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        if type(mf).__module__.startswith("gpu4pyscf"):
            from gpu4pyscf import df as gpu4pyscf_df
            with_df = gpu4pyscf_df.df.DF(mf.mol, auxbasis=auxbasis)
        else:
            with_df = df.DF(mf.mol, auxbasis=auxbasis)
        with_df.build()

    for block in with_df.loop(blksize):
        if isinstance(block, tuple):
            # gpu4pyscf.df.df.DF.loop(): (unpacked_cupy, packed_slab)
            _, packed = block
            yield _to_host(packed)
        else:
            # pyscf.df.df.DF.loop(): packed numpy block directly
            yield _to_host(block)
