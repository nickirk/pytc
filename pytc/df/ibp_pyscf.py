"""PySCF-compatible density-fitting provider for the atom-centered single-IBP
Coulomb metric.

``IBPISDF`` subclasses PySCF's molecular ``DF`` so it can be assigned to
``mf.with_df`` or passed to a post-HF density-fitting helper. This module is
the high-level adapter: it turns a molecule into the low-level IBP artifacts
in ``pytc.df.ibp`` and exposes the streamed packed AO-pair factor through the
standard ``get_naoaux``/``loop``/``ao2mo`` surface.

``get_jk`` raises ``NotImplementedError`` so the inherited analytic-DF SCF
path can never silently run against a different Coulomb metric.

Version policy: the public DF surface this adapter overrides was measured on
PySCF 2.10.0 and only ``>=2.10,<2.11`` is claimed.
"""

import dataclasses

import numpy as np
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.df.df import DF
from pyscf.dft import numint
from pyscf.lib import pack_tril, unpack_tril
from scipy.linalg.blas import dgemm

from pytc.df.ibp import (
    build_ibp_grid,
    build_ibp_interpolation_sector,
    build_ibp_operator_plan,
    ibp_core,
)

_IBPISDF_METRIC = "atom_centered_single_ibp"
_SUPPORTED_IBPISDF_BACKENDS = ("numpy",)
_SUPPORTED_ON_OVER_RANK = ("truncate", "raise")


def _validate_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive int, got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _validate_positive_float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive real number, got {value!r}.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
    return value


def _mol_fingerprint(mol):
    """A lightweight identity for the stale-molecule guard: an in-place
    mutation (set_geom_, basis exponent/coefficient change, ECP change, cart
    flip) after build() must not silently reuse stale factors. Binds PySCF's
    realized numeric molecule state -- the internal integral tables _atm/_bas/
    _env (which encode the actual basis exponents and contraction coefficients,
    so a same-nao basis change is caught) and _ecpbas -- plus charge/spin/cart.
    No hashing: the raw bytes/tuples are compared directly."""
    def _tobytes(name):
        arr = getattr(mol, name, None)
        return np.ascontiguousarray(arr).tobytes() if arr is not None else b""
    return (
        bool(mol.cart),
        int(mol.charge),
        int(mol.spin),
        _tobytes("_atm"),
        _tobytes("_bas"),
        _tobytes("_env"),
        _tobytes("_ecpbas"),
    )


@dataclasses.dataclass(frozen=True)
class IBPISDFConfig:
    """Immutable configuration carrying every algorithmic knob that affects
    the realized artifacts. Rank is explicit -- there is no silent rank
    heuristic."""
    rank: int
    grid_level: int = 2
    backend: str = "numpy"
    psd_rtol: float = 1e-10
    packed_pair_tol: float = 1e-3
    pivot_effective_rank_rtol: float = 1e-6
    pivot_on_over_rank: str = "truncate"
    grid_batch_size: object = None      # None -> whole grid in one batch
    eval_block_size: int = 128
    source_block_size: int = 4096
    core_mu_block_size: object = None   # None -> full pivot axis
    core_nu_block_size: object = None

    def __post_init__(self):
        object.__setattr__(self, "rank", _validate_positive_int("rank", self.rank))
        object.__setattr__(self, "grid_level", self._validate_grid_level())
        if self.backend not in _SUPPORTED_IBPISDF_BACKENDS:
            raise ValueError(
                f"Unsupported backend={self.backend!r}; only {_SUPPORTED_IBPISDF_BACKENDS} "
                f"is implemented (JAX is a later milestone)."
            )
        object.__setattr__(self, "psd_rtol", self._nonneg_float("psd_rtol", self.psd_rtol))
        object.__setattr__(self, "packed_pair_tol",
                           _validate_positive_float("packed_pair_tol", self.packed_pair_tol))
        object.__setattr__(self, "pivot_effective_rank_rtol",
                           self._validate_open_unit_interval(
                               "pivot_effective_rank_rtol", self.pivot_effective_rank_rtol))
        if self.pivot_on_over_rank not in _SUPPORTED_ON_OVER_RANK:
            raise ValueError(
                f"pivot_on_over_rank must be one of {_SUPPORTED_ON_OVER_RANK}, got "
                f"{self.pivot_on_over_rank!r}."
            )
        for name in ("grid_batch_size", "core_mu_block_size", "core_nu_block_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_positive_int(name, value))
        object.__setattr__(self, "eval_block_size",
                           _validate_positive_int("eval_block_size", self.eval_block_size))
        object.__setattr__(self, "source_block_size",
                           _validate_positive_int("source_block_size", self.source_block_size))

    def _validate_grid_level(self):
        if isinstance(self.grid_level, bool) or not isinstance(self.grid_level, (int, np.integer)):
            raise ValueError(f"grid_level must be a non-negative int, got {self.grid_level!r}.")
        level = int(self.grid_level)
        if level < 0:
            raise ValueError(f"grid_level must be non-negative, got {level}.")
        return level

    @staticmethod
    def _validate_open_unit_interval(name, value):
        """The selector's effective-rank rtol contract is 0 < rtol < 1; reject
        anything outside the open unit interval so failure is early and closed."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number in (0, 1), got {value!r}.")
        value = float(value)
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError(f"{name} must satisfy 0 < {name} < 1, got {value!r}.")
        return value

    @staticmethod
    def _nonneg_float(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a non-negative real number, got {value!r}.")
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
        return value


@dataclasses.dataclass(frozen=True)
class _IBPDiagnostics:
    """Compact rank/PSD/residual record kept after the factor is published,
    so the intermediate grid/sector/plan/core artifacts need not be retained."""
    psd_status: str
    psd_retained_rank: int
    raw_packed_pair_metric_dagger_residual: float
    Z: object


class IBPISDF(DF):
    """Atom-centered single-IBP Coulomb density-fitting provider."""

    metric = _IBPISDF_METRIC

    def __init__(self, mol, auxbasis=None, *, config=None, **knobs):
        if auxbasis is not None:
            raise ValueError(
                "IBPISDF does not fabricate an auxiliary basis; auxbasis must be None "
                "(the atom-centered single-IBP metric replaces the DF metric)."
            )
        if config is not None:
            if knobs:
                raise ValueError(
                    "Pass either an explicit IBPISDFConfig or direct keyword knobs, "
                    "never both."
                )
            if not isinstance(config, IBPISDFConfig):
                raise TypeError(f"config must be an IBPISDFConfig, got {type(config).__name__}.")
        else:
            if "rank" not in knobs:
                raise ValueError("rank is required (no rank heuristic); pass rank=... .")
            config = IBPISDFConfig(**knobs)

        DF.__init__(self, mol, auxbasis=None)
        self._config = config
        self._clear_custom_cache()

    # -- lifecycle -----------------------------------------------------------

    def _clear_custom_cache(self):
        self._ibp_built = False
        self._ibp_mol_fingerprint = None
        self._ibp_diagnostics = None
        # _cderi is the SINGLE authoritative packed-AO factor buffer
        # (naoaux, nao_pair), aosym='s2', float64 -- the genuine cderi that
        # unchanged PySCF DF consumers (e.g. DFMP2) read. loop()/get_naoaux()/
        # ao2mo() all consume this one state, so a perturbation of the factor
        # propagates. No separate W/P copy is retained.
        self._cderi = None

    @property
    def config(self):
        return self._config

    @staticmethod
    def _require_supported_molecule(mol):
        """v1 supports all-electron and ECP molecules. A pseudopotential
        (GTH/pseudo) molecule changes the realized artifacts and has not been
        validated here, so it is rejected explicitly."""
        if getattr(mol, "_pseudo", None):
            raise NotImplementedError(
                "IBPISDF v1 does not support pseudopotential (pseudo/GTH) molecules; "
                "only all-electron and ECP molecules are validated. Remove the "
                "pseudopotential or use a supported provider."
            )

    def build(self):
        # Idempotent: a second build validates that the molecule identity has
        # not changed under the built cache. An in-place mutation after the
        # first build must not silently reuse stale factors -- it fails loudly
        # and demands reset().
        self._require_supported_molecule(self.mol)
        current = _mol_fingerprint(self.mol)
        if self._ibp_built:
            if current != self._ibp_mol_fingerprint:
                raise RuntimeError(
                    "IBPISDF: the molecule identity changed after build() (geometry, "
                    "basis, or AO representation differs from the built artifact); call "
                    "reset() before reusing this provider."
                )
            return self
        self._ibp_build(current)
        return self

    def reset(self, mol=None):
        # Base reset clears the inherited DF caches (including any _cderi the
        # base constructor allocated); then clear all custom state so no built
        # artifact survives a molecule replacement.
        super().reset(mol)
        self._clear_custom_cache()
        return self

    def copy(self):
        # An unbuilt copy with the same immutable configuration; it must not
        # share this provider's mutable build cache. PySCF runtime settings are
        # preserved so the copy behaves like a fresh, unbuilt clone.
        new = IBPISDF(self.mol, config=self._config)
        new.blockdim = self.blockdim
        new.max_memory = self.max_memory
        new.verbose = self.verbose
        new.stdout = self.stdout
        return new

    def _resolve_blksize(self, blksize):
        """Resolve and type-check the effective block size (including the
        blockdim default) before any streaming so an invalid block size or a
        corrupted blockdim can never be silently swallowed by an early
        (e.g. zero-rank) return."""
        if blksize is None:
            blksize = self.blockdim
        if isinstance(blksize, bool) or not isinstance(blksize, (int, np.integer)):
            raise ValueError(f"blksize must be a positive int or None, got {blksize!r}.")
        blksize = int(blksize)
        if blksize <= 0:
            raise ValueError(f"blksize must be positive, got {blksize}.")
        return blksize

    def get_naoaux(self):
        self.build()
        return self._cderi.shape[0]

    def loop(self, blksize=None):
        self.build()
        # Validate the block size first -- before the zero-rank short circuit --
        # so an invalid blksize/blockdim is rejected regardless of rank.
        blksize = self._resolve_blksize(blksize)
        cderi = self._cderi
        naoaux = cderi.shape[0]
        if naoaux == 0:
            return
        # Stream row blocks of the single authoritative packed-AO factor.
        for start in range(0, naoaux, blksize):
            end = min(start + blksize, naoaux)
            yield np.ascontiguousarray(cderi[start:end])

    def _normalize_mo_coeffs(self, mo_coeffs):
        """Accept the two public PySCF forms -- one 2-D coefficient matrix
        (used for all four indices) or a length-4 sequence of 2-D matrices --
        with strict validation and NO silent coercion. The pinned PySCF 2.10
        DF ao2mo accepts float64 and rejects float32/integer/bool with an
        assertion, so v1 requires the coefficient dtype to be exactly float64
        (plus real/finite/2-D/``nao``-row checks) and rejects the others
        explicitly. Complex coefficients are a real-only NotImplementedError."""
        nao = int(self.mol.nao)
        if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
            seq = (mo_coeffs,) * 4
        elif isinstance(mo_coeffs, (tuple, list)) and len(mo_coeffs) == 4:
            seq = tuple(mo_coeffs)
        else:
            raise ValueError(
                "mo_coeffs must be a single 2-D coefficient matrix or a length-4 "
                "sequence of 2-D matrices."
            )
        normalized = []
        for k, c in enumerate(seq):
            c = np.asarray(c)
            if np.iscomplexobj(c):
                raise NotImplementedError(
                    "IBPISDF.ao2mo is real-only in v1 (the pinned PySCF 2.10 DF ao2mo "
                    "rejects complex coefficients); a complex-capable transform is a "
                    "separately versioned extension."
                )
            if c.dtype != np.float64:
                raise TypeError(
                    f"mo_coeffs[{k}] must be exactly float64 (PySCF 2.10 DF ao2mo "
                    f"rejects float32/integer/bool/object; no silent coercion), got "
                    f"dtype {c.dtype}."
                )
            if c.ndim != 2:
                raise ValueError(f"mo_coeffs[{k}] must be 2-D, got ndim={c.ndim}.")
            if c.shape[0] != nao:
                raise ValueError(
                    f"mo_coeffs[{k}] has {c.shape[0]} AO rows, expected nao={nao}."
                )
            if not np.all(np.isfinite(c)):
                raise ValueError(f"mo_coeffs[{k}] contains non-finite values.")
            normalized.append(np.ascontiguousarray(c))
        return tuple(normalized)

    @staticmethod
    def _half_transform(b_ao, ca, cb, pack, na, nb):
        """Half-transform one L block of symmetric AO factors to an MO pair
        axis: ``(L|pq) = Ca^T B_L Cb``. Returns ``(n_L, na_pair)`` packed
        lower-triangular when ``pack``, else ``(n_L, na*nb)`` full. Peak
        transients are bounded by blockdim x nmo x nao; no AO four-index tensor
        is formed."""
        tmp = np.einsum("Lab,ai->Lib", b_ao, ca, optimize=True)   # (n_L, na, nao)
        m = np.einsum("Lib,bj->Lij", tmp, cb, optimize=True)      # (n_L, na, nb)
        if pack:
            return pack_tril(m)                                   # (n_L, na_pair)
        return m.reshape(m.shape[0], na * nb)

    def ao2mo(self, mo_coeffs, compact=True):
        """Full four-index MO ERIs ``(pq|rs)`` for the atom-centered single-IBP
        metric, streamed from the packed pseudo-auxiliary factor. ``compact``
        follows PySCF's truth-value convention: a pair axis is packed
        lower-triangular only when its two coefficient matrices are identical
        (PySCF ``iden_coeffs``) and ``compact`` is truthy. Routes through
        ``build()`` so lazy build and the stale-molecule guard are inherited;
        derives solely from W/P via ``loop()`` and never reads the inherited
        ``_cderi``.

        Coefficients are validated (dtype/shape/finite/real) BEFORE the
        expensive lazy build. Per L block the peak transients are bounded by
        blockdim; ``dgemm`` accumulates into the ``(bra_dim, ket_dim)`` output
        in place (beta=1), so there is no output-sized GEMM temporary."""
        c1, c2, c3, c4 = self._normalize_mo_coeffs(mo_coeffs)   # validate before build
        self.build()
        pack_bra = bool(compact) and iden_coeffs(c1, c2)
        pack_ket = bool(compact) and iden_coeffs(c3, c4)
        n1, n2 = c1.shape[1], c2.shape[1]
        n3, n4 = c3.shape[1], c4.shape[1]
        bra_dim = n1 * (n1 + 1) // 2 if pack_bra else n1 * n2
        ket_dim = n3 * (n3 + 1) // 2 if pack_ket else n3 * n4
        out = np.zeros((bra_dim, ket_dim), order="F")
        # Iterate loop() unconditionally: zero retained rank yields zero blocks
        # (leaving the zero output), and the block-size validation inside loop()
        # still fires -- no zero-rank short circuit to bypass the contract.
        for block in self.loop():                     # (n_L, nao_pair) packed-lower
            b_ao = unpack_tril(np.ascontiguousarray(block))   # (n_L, nao, nao) symmetric
            l12 = self._half_transform(b_ao, c1, c2, pack_bra, n1, n2)
            l34 = self._half_transform(b_ao, c3, c4, pack_ket, n3, n4)
            out = dgemm(1.0, l12, l34, beta=1.0, c=out, trans_a=1, overwrite_c=1)
        return out

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13,
               omega=None):
        raise NotImplementedError(
            "IBPISDF does not implement SCF get_jk in this phase; it must not silently "
            "fall back to a different Coulomb metric."
        )

    # -- build pipeline ------------------------------------------------------

    def _ibp_build(self, mol_fingerprint):
        cfg = self._config
        mol = self.mol

        # Realized atom grid.
        from pyscf.dft import gen_grid
        g = gen_grid.Grids(mol)
        g.level = cfg.grid_level
        g.build()
        coords = np.ascontiguousarray(g.coords)
        weights = np.ascontiguousarray(g.weights)

        # AO values and gradients on the grid: eval_ao(deriv=1) -> (4, n_grid,
        # n_ao) as [value, d/dx, d/dy, d/dz]. Sector wants (n_ao, n_grid) values
        # and (3, n_ao, n_grid) gradients.
        ao = numint.eval_ao(mol, coords, deriv=1)
        ao_values = np.ascontiguousarray(ao[0].T)                     # (n_ao, n_grid)
        ao_gradients = np.ascontiguousarray(ao[1:4].transpose(0, 2, 1))  # (3, n_ao, n_grid)

        grid = build_ibp_grid(coords, weights)

        # Canonical pivot selection uses the WEIGHTED AO values; the sector is
        # built from the RAW values/gradients.
        from pytc.integrals.coulomb import select_sector_pivots, weight_mo_values
        weighted = weight_mo_values(ao_values, grid.weights)
        pivots, record = select_sector_pivots(
            weighted, weighted, cfg.rank,
            on_over_rank=cfg.pivot_on_over_rank,
            same_factor=True,
            effective_rank_rtol=cfg.pivot_effective_rank_rtol,
            return_provenance=True,
        )

        sector = build_ibp_interpolation_sector(
            ao_values, ao_values, ao_gradients, ao_gradients, pivots, grid,
            pivot_provenance=record, same_factor=True,
            grid_batch_size=cfg.grid_batch_size,
        )
        plan = build_ibp_operator_plan(
            grid, method="direct",
            eval_block_size=cfg.eval_block_size, source_block_size=cfg.source_block_size,
        )
        # Same-sector, two-sided-averaged core. ibp_core hard-fails a materially
        # indefinite core; a within-band PSD factor is produced here.
        core = ibp_core(
            sector, operator=plan, symmetry_mode="two_sided_average",
            mu_block_size=cfg.core_mu_block_size, nu_block_size=cfg.core_nu_block_size,
            psd_rtol=cfg.psd_rtol,
        )

        # Gates: the packed AO-pair metric must be Hermitian to tolerance and
        # the core must have passed PSD before the provider is published.
        if core.raw_packed_pair_metric_dagger_residual > cfg.packed_pair_tol:
            raise ValueError(
                f"packed-pair metric dagger residual "
                f"{core.raw_packed_pair_metric_dagger_residual!r} exceeds the tolerance "
                f"{cfg.packed_pair_tol!r}; the AO-pair Coulomb metric is not sufficiently "
                f"Hermitian on this grid."
            )
        if core.psd_status != "factorized":
            raise ValueError(
                f"core psd_status={core.psd_status!r}; a factorized (PSD) core is required "
                f"before the provider becomes built."
            )

        # Materialize the single authoritative packed-AO DF factor
        # _cderi = W^dagger P, shape (naoaux, nao_pair), aosym='s2', float64 --
        # the genuine cderi that unchanged PySCF DF consumers read (loop()
        # streams from it; DFMP2 hits its non-analytic path).
        cderi = np.ascontiguousarray(core.psd_factor.conj().T @ sector.P, dtype=np.float64)
        nao = int(self.mol.nao)
        nao_pair = nao * (nao + 1) // 2
        naoaux = int(core.psd_retained_rank)
        if cderi.shape != (naoaux, nao_pair):
            raise ValueError(
                f"_cderi shape {cderi.shape} must be (naoaux, nao_pair)="
                f"{(naoaux, nao_pair)} (packed s2 lower-triangular AO pairs).")
        if cderi.dtype != np.float64 or not cderi.flags["C_CONTIGUOUS"]:
            raise ValueError("_cderi must be a C-contiguous float64 array.")
        if not np.all(np.isfinite(cderi)):
            raise ValueError("_cderi has non-finite entries.")
        self._cderi = cderi
        self._ibp_diagnostics = _IBPDiagnostics(
            psd_status=core.psd_status,
            psd_retained_rank=int(core.psd_retained_rank),
            raw_packed_pair_metric_dagger_residual=core.raw_packed_pair_metric_dagger_residual,
            Z=core.Z,
        )
        self._ibp_mol_fingerprint = mol_fingerprint
        self._ibp_built = True
