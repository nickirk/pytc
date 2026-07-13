"""PySCF-compatible density-fitting provider for the atom-centered single-IBP
Coulomb metric.

``IBPISDF`` subclasses PySCF's molecular ``DF`` so it can be assigned to
``mf.with_df`` or passed to a post-HF density-fitting helper. This module is
the high-level adapter: it turns a molecule into the low-level IBP artifacts
in ``pytc.df.ibp`` and exposes the streamed packed AO-pair factor through the
standard ``get_naoaux``/``loop`` surface.

This milestone implements the lifecycle, ``get_naoaux``, and the packed
three-index ``loop`` only. ``ao2mo`` and the unchanged-consumer energy gates
are a separate step and raise ``NotImplementedError`` here so the inherited
analytic-DF behavior can never run.

Version policy: the public DF surface this adapter overrides was measured on
PySCF 2.10.0 and only ``>=2.10,<2.11`` is claimed. The compatibility matrix
records the surface-probe parity observed on newer PySCF; support beyond
2.10.x is not claimed without the full acceptance matrix.
"""

import dataclasses

import numpy as np
import pyscf
from pyscf.df.df import DF
from pyscf.dft import numint

from pytc.df.ibp import (
    _canonical_sha256,
    _canonical_spec_sha256,
    build_ibp_grid,
    build_ibp_interpolation_sector,
    build_ibp_operator_plan,
    ibp_core,
)

_IBPISDF_VERSION = "1"
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


def _plain_python(obj):
    """Recursively coerce a PySCF parsed-basis / ECP structure into plain
    Python dict/list/str/int/float/bool/None so the canonical TLV encoder
    (which rejects object-dtype arrays and unknown types) can hash it. Numpy
    scalars become their Python items; numpy arrays become nested lists."""
    if isinstance(obj, dict):
        return {str(k): _plain_python(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _plain_python(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_plain_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bool, int, float, str, bytes)) or obj is None:
        return obj
    # Anything else (unexpected) is stringified rather than silently dropped.
    return str(obj)


def _normalized_basis(mol):
    """The parsed numeric basis (``mol._basis``: per-element list of angular
    momentum + primitive exponent/coefficient blocks), as plain Python. This
    is the realized numeric basis, not a repr of the requested basis name, so
    two molecules that resolve to different contractions are distinguished."""
    return _plain_python(dict(getattr(mol, "_basis", {}) or {}))


def _normalized_ecp(mol):
    """The parsed ECP (``mol._ecp``), plain Python, or None if none is set."""
    ecp = getattr(mol, "_ecp", None)
    if not ecp:
        return None
    return _plain_python(dict(ecp))


def _canonical_molecule_digest(mol):
    """A reproducible, canonical identity digest for a molecule, built from
    normalized numeric/string fields via the closed TLV encoder (never
    ``repr``): element charges, coordinates (bohr), total charge, spin, the
    AO representation (``cart`` vs spherical -- it changes ``nao`` for the
    same basis), the realized AO count, the parsed numeric basis, and any
    ECP. A different molecule, geometry, basis, or AO representation always
    yields a different digest, deterministically across processes."""
    return _canonical_spec_sha256({
        "atom_charges": np.ascontiguousarray(np.asarray(mol.atom_charges())),
        "atom_coords_bohr": np.ascontiguousarray(mol.atom_coords()),
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "cart": bool(mol.cart),
        "nao": int(mol.nao),
        "basis": _normalized_basis(mol),
        "ecp": _normalized_ecp(mol),
    })


@dataclasses.dataclass(frozen=True)
class IBPISDFConfig:
    """Immutable configuration carrying every algorithmic knob that affects
    the realized artifacts, so the provider's cache can bind and reject a
    stale build. Rank is explicit -- there is no silent rank heuristic."""
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
    config_spec_sha256: str = dataclasses.field(default="", compare=False)

    def __post_init__(self):
        rank = _validate_positive_int("rank", self.rank)
        object.__setattr__(self, "rank", rank)
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
        spec = _canonical_spec_sha256({
            "rank": self.rank, "grid_level": self.grid_level, "backend": self.backend,
            "psd_rtol": self.psd_rtol, "packed_pair_tol": self.packed_pair_tol,
            "pivot_effective_rank_rtol": self.pivot_effective_rank_rtol,
            "pivot_on_over_rank": self.pivot_on_over_rank,
            "grid_batch_size": self.grid_batch_size,
            "eval_block_size": self.eval_block_size,
            "source_block_size": self.source_block_size,
            "core_mu_block_size": self.core_mu_block_size,
            "core_nu_block_size": self.core_nu_block_size,
            "metric": _IBPISDF_METRIC, "version": _IBPISDF_VERSION,
        })
        object.__setattr__(self, "config_spec_sha256", spec)

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
        self._ibp_mol_digest = None
        self._ibp_provenance = None
        self._ibp_grid = None
        self._ibp_sector = None
        self._ibp_plan = None
        self._ibp_core = None
        self._ibp_factor = None       # W, (n_mu, retained_rank)
        self._ibp_pair = None         # P, (rank, n_pair)
        self._ibp_naoaux = None

    @property
    def config(self):
        return self._config

    @property
    def provenance(self):
        return self._ibp_provenance

    def build(self):
        # Idempotent: a second build validates that the molecule identity has
        # not changed under the built cache. An in-place mutation (set_geom_,
        # basis change, cart flip) after the first build must not silently
        # reuse stale factors -- it fails loudly and demands reset().
        current = _canonical_molecule_digest(self.mol)
        if self._ibp_built:
            if current != self._ibp_mol_digest:
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
        # An unbuilt copy with the same immutable configuration -- the simplest
        # honest contract; it must not share this provider's mutable build
        # cache. PySCF runtime settings are preserved so the copy behaves like
        # a fresh, unbuilt clone of this provider rather than a defaulted one.
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
        return self._ibp_naoaux

    def loop(self, blksize=None):
        self.build()
        # Validate the block size first -- before the zero-rank short circuit --
        # so an invalid blksize/blockdim is rejected regardless of rank.
        blksize = self._resolve_blksize(blksize)
        naoaux = self._ibp_naoaux
        if naoaux == 0:
            return
        W = self._ibp_factor
        P = self._ibp_pair
        for start in range(0, naoaux, blksize):
            end = min(start + blksize, naoaux)
            yield np.ascontiguousarray(W[:, start:end].conj().T @ P)

    def ao2mo(self, mo_coeffs, compact=True):
        raise NotImplementedError(
            "IBPISDF.ao2mo is implemented in a later milestone; the inherited "
            "analytic-DF ao2mo must not run for this Coulomb metric."
        )

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13,
               omega=None):
        raise NotImplementedError(
            "IBPISDF does not implement SCF get_jk in this phase; it must not silently "
            "fall back to a different Coulomb metric."
        )

    # -- build pipeline ------------------------------------------------------

    def _provider_provenance(self, mol_digest):
        """One closed provenance record identifying exactly which molecule,
        configuration, adapter, and PySCF produced the artifacts. It is bound
        into every downstream artifact so the identity travels with the data."""
        return {
            "provider": "IBPISDF",
            "adapter_version": _IBPISDF_VERSION,
            "metric": _IBPISDF_METRIC,
            "mol_digest": mol_digest,
            "config_spec_sha256": self._config.config_spec_sha256,
            "pyscf_version": str(pyscf.__version__),
        }

    def _ibp_build(self, mol_digest):
        cfg = self._config
        mol = self.mol
        provenance = self._provider_provenance(mol_digest)

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

        grid = build_ibp_grid(
            coords, weights,
            construction_metadata={
                "source": "pyscf_atom_grid", "grid_level": cfg.grid_level,
                "provider_provenance": provenance,
            },
        )

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
            upstream_provenance=provenance,
        )
        plan = build_ibp_operator_plan(
            grid, method="direct",
            eval_block_size=cfg.eval_block_size, source_block_size=cfg.source_block_size,
            upstream_provenance=provenance,
        )
        # Same-sector, two-sided-averaged core. ibp_core hard-fails a materially
        # indefinite core; a within-band PSD factor is produced here.
        core = ibp_core(
            sector, operator=plan, symmetry_mode="two_sided_average",
            mu_block_size=cfg.core_mu_block_size, nu_block_size=cfg.core_nu_block_size,
            psd_rtol=cfg.psd_rtol,
            upstream_provenance=provenance,
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

        # Atomic publish: only now assign the cache references.
        self._ibp_grid = grid
        self._ibp_sector = sector
        self._ibp_plan = plan
        self._ibp_core = core
        self._ibp_factor = core.psd_factor            # W
        self._ibp_pair = sector.P                      # P
        self._ibp_naoaux = int(core.psd_retained_rank)
        self._ibp_provenance = provenance
        self._ibp_mol_digest = mol_digest
        self._ibp_built = True
