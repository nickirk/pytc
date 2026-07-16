"""CPU-only feasibility helpers for reciprocal primitive-AO translation.

These helpers are deliberately standalone: they are not used by the periodic
selector or any production build path.  They preserve the sampled-grid
approximation explicitly so a pilot can compare it with direct AO values.
"""

from __future__ import annotations

import dataclasses
import weakref
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import tools as pbc_tools

from pytc.pbc.df.isdf import pivoted_cholesky_hermitian


class ReciprocalSameGridCapacityError(RuntimeError):
    """Raised when the reciprocal selector exceeds its declared byte policy."""

    condition = "RECIPROCAL_SAME_GRID_SELECTION_PEAK_EXCEEDS_POLICY"


class ReciprocalOrbitPartitionError(ValueError):
    """Raised when a requested AO reuse orbit is not physically validated."""

    condition = "RECIPROCAL_SAME_GRID_INVALID_ORBIT_PARTITION"


@dataclasses.dataclass(frozen=True)
class ReciprocalOrbitPartition:
    """Explicit seed, translated-replica, and directly streamed AO groups."""

    seed_shell_slice: tuple[int, int]
    replica_shell_slices: tuple[tuple[int, int], ...]
    replica_translations: np.ndarray
    unique_shell_slices: tuple[tuple[int, int], ...] = ()
    supercell_matrix: np.ndarray | None = None


def _shell_ao_count(cell, shell_slice):
    start, stop = shell_slice
    ao_loc = np.asarray(cell.ao_loc_nr(), dtype=np.int64)
    if not (0 <= start < stop <= cell.nbas):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: invalid shell slice {shell_slice}."
        )
    return int(ao_loc[stop] - ao_loc[start])


def _validate_shell_slice(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: {name} must be a (start, stop) tuple."
        )
    start, stop = value
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: {name} entries must be integers."
        )
    return int(start), int(stop)


def validate_reciprocal_orbit_partition(cell, partition, *, coordinate_tol=1e-10):
    """Validate that every reused shell block is a translated seed orbit."""
    if not isinstance(partition, ReciprocalOrbitPartition):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: partition must be ReciprocalOrbitPartition."
        )
    if not isinstance(coordinate_tol, (int, float)) or coordinate_tol <= 0:
        raise ValueError("coordinate_tol must be positive.")
    seed_slice = _validate_shell_slice(partition.seed_shell_slice, "seed_shell_slice")
    replica_slices = tuple(
        _validate_shell_slice(value, "replica_shell_slices")
        for value in partition.replica_shell_slices
    )
    unique_slices = tuple(
        _validate_shell_slice(value, "unique_shell_slices")
        for value in partition.unique_shell_slices
    )
    translations = np.asarray(partition.replica_translations, dtype=np.float64)
    if translations.shape != (len(replica_slices), 3):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: replica_translations must have shape "
            f"({len(replica_slices)},3), got {translations.shape}."
        )
    if not np.all(np.isfinite(translations)):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: replica_translations must be finite."
        )
    if len(replica_slices) == 0:
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: at least one translated replica is required."
        )
    if np.unique(translations, axis=0).shape[0] != translations.shape[0]:
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: replica translations must be unique."
        )

    seed_ao = _shell_ao_count(cell, seed_slice)
    n_seed_shells = seed_slice[1] - seed_slice[0]
    coordinates = np.asarray(cell.atom_coords(), dtype=np.float64)
    base = np.asarray(cell._bas, dtype=np.int32)
    grouped_shells = list(range(*seed_slice))
    residuals = []
    for replica_slice, translation in zip(replica_slices, translations, strict=True):
        if replica_slice[1] - replica_slice[0] != n_seed_shells:
            raise ReciprocalOrbitPartitionError(
                f"{ReciprocalOrbitPartitionError.condition}: translated shell count differs from seed."
            )
        if _shell_ao_count(cell, replica_slice) != seed_ao:
            raise ReciprocalOrbitPartitionError(
                f"{ReciprocalOrbitPartitionError.condition}: translated AO count differs from seed."
            )
        for seed_shell, replica_shell in zip(
            range(*seed_slice), range(*replica_slice), strict=True,
        ):
            if not np.array_equal(base[seed_shell, 1:], base[replica_shell, 1:]):
                raise ReciprocalOrbitPartitionError(
                    f"{ReciprocalOrbitPartitionError.condition}: species or basis differs in a reuse orbit."
                )
            seed_atom, replica_atom = int(base[seed_shell, 0]), int(base[replica_shell, 0])
            residuals.append(float(np.max(np.abs(
                coordinates[replica_atom] - coordinates[seed_atom] - translation,
            ))))
        grouped_shells.extend(range(*replica_slice))
    for unique_slice in unique_slices:
        grouped_shells.extend(range(*unique_slice))
    if sorted(grouped_shells) != list(range(cell.nbas)):
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: shell groups must partition the full cell."
        )
    residual = max(residuals, default=0.0)
    if residual > coordinate_tol:
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: orbit coordinate residual "
            f"{residual:.3e} exceeds {coordinate_tol:.3e}."
        )
    supercell_matrix = None
    if partition.supercell_matrix is not None:
        supercell_matrix = np.asarray(partition.supercell_matrix, dtype=np.int64)
        if supercell_matrix.shape != (3, 3) or round(abs(np.linalg.det(supercell_matrix))) != len(replica_slices) + 1:
            raise ReciprocalOrbitPartitionError(
                f"{ReciprocalOrbitPartitionError.condition}: supercell_matrix does not match replica count."
            )
    unique_ao = sum(_shell_ao_count(cell, value) for value in unique_slices)
    realized_factor = ((len(replica_slices) + 1) * seed_ao + unique_ao) / (seed_ao + unique_ao)
    return {
        "seed_shell_slice": seed_slice,
        "replica_shell_slices": replica_slices,
        "replica_translations": translations,
        "unique_shell_slices": unique_slices,
        "supercell_matrix": None if supercell_matrix is None else supercell_matrix.tolist(),
        "NAO_reused": seed_ao,
        "NAO_unique": unique_ao,
        "n_replicas": len(replica_slices) + 1,
        "orbit_validation_residual": residual,
        "realized_cache_factor": realized_factor,
    }


@partial(jax.jit, static_argnames=("mesh",))
def _reciprocal_same_grid_translate_kernel(
    seed, remove_phase, restore_phase, reciprocal_phase, *, mesh,
):
    """Translate one primitive AO group with a cached same-grid FFT kernel."""
    n_kpts, n_grid, n_ao = seed.shape
    values = seed * remove_phase[:, :, None]
    fourier = jnp.fft.fftn(
        values.transpose(0, 2, 1).reshape(n_kpts, n_ao, *mesh),
        axes=(-3, -2, -1),
    )
    translated = jnp.fft.ifftn(
        fourier * reciprocal_phase.reshape((1, 1, *mesh)), axes=(-3, -2, -1),
    ).reshape(n_kpts, n_ao, n_grid).transpose(0, 2, 1)
    return restore_phase[:, :, None] * translated


def reciprocal_same_grid_byte_model(
    n_kpts, n_grid, nao_reused, nao_unique, n_replicas, rank, *,
    selection_peak_max_bytes, peak_safety_factor=1.10,
    process_pipeline_allowance_bytes=None,
):
    """Fail-closed host/device accounting for primitive-seed selection.

    The explicit process allowance covers interpreter/JAX runtime residency not
    attributable to a tensor below.  It is mandatory: a tensor-only estimate
    cannot safely stand in for a process peak.
    """
    if process_pipeline_allowance_bytes is None:
        raise ReciprocalSameGridCapacityError(
            "RECIPROCAL_SAME_GRID_PROCESS_PIPELINE_ALLOWANCE_REQUIRED: "
            "an explicit positive process_pipeline_allowance_bytes is required."
        )
    if (
        isinstance(process_pipeline_allowance_bytes, bool)
        or not isinstance(process_pipeline_allowance_bytes, (int, np.integer))
        or int(process_pipeline_allowance_bytes) <= 0
    ):
        raise ReciprocalSameGridCapacityError(
            "RECIPROCAL_SAME_GRID_PROCESS_PIPELINE_ALLOWANCE_REQUIRED: "
            "process_pipeline_allowance_bytes must be a positive integer."
        )
    values = {
        "n_kpts": n_kpts, "n_grid": n_grid, "nao_reused": nao_reused,
        "nao_unique": nao_unique, "n_replicas": n_replicas, "rank": rank,
        "selection_peak_max_bytes": selection_peak_max_bytes,
        "process_pipeline_allowance_bytes": process_pipeline_allowance_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    if min(int(n_kpts), int(n_grid), int(nao_reused), int(n_replicas), int(rank), int(selection_peak_max_bytes)) <= 0:
        raise ValueError("n_kpts, n_grid, nao_reused, n_replicas, rank, and policy must be positive.")
    if not isinstance(peak_safety_factor, (int, float)) or peak_safety_factor < 1.0:
        raise ValueError("peak_safety_factor must be a real number >= 1.")
    c128 = np.dtype(np.complex128).itemsize
    f64 = np.dtype(np.float64).itemsize
    seed_bytes = int(n_kpts) * int(n_grid) * int(nao_reused) * c128
    generated_bytes = seed_bytes
    fft_workspace_bytes = seed_bytes
    unique_group_bytes = int(n_kpts) * int(n_grid) * int(nao_unique) * c128
    factor_bytes = int(n_grid) * int(rank) * f64
    residual_bytes = int(n_grid) * f64
    metric_column_bytes = int(n_grid) * c128
    selector_bytes = int(n_grid) * np.dtype(np.bool_).itemsize + int(rank) * np.dtype(np.int64).itemsize
    work_bytes = residual_bytes + metric_column_bytes + selector_bytes
    phase_plane_bytes = int(n_kpts) * int(n_grid) * c128
    reciprocal_phase_bytes = int(n_grid) * c128
    # Host keeps only the NumPy seed, a single emitted group, the host-side
    # pivot state, and the caller-declared process allowance.  Device FFT
    # execution also retains the JAX seed and phase planes plus input/Fourier/
    # translated/output buffers.  These are intentionally separate peaks.
    host_group_bytes = max(generated_bytes, unique_group_bytes)
    host_tensor_peak = seed_bytes + host_group_bytes + factor_bytes + work_bytes
    device_tensor_peak = (
        seed_bytes
        + 2 * phase_plane_bytes
        + reciprocal_phase_bytes
        + 4 * fft_workspace_bytes
    )
    host_peak = host_tensor_peak + int(process_pipeline_allowance_bytes)
    device_peak = device_tensor_peak + int(process_pipeline_allowance_bytes)
    host_with_safety = int(np.ceil(host_peak * float(peak_safety_factor)))
    device_with_safety = int(np.ceil(device_peak * float(peak_safety_factor)))
    condition = None
    if max(host_with_safety, device_with_safety) > int(selection_peak_max_bytes):
        condition = ReciprocalSameGridCapacityError.condition
    return {
        "mode": "reciprocal_same_grid",
        "persistent_seed_complex128_bytes": seed_bytes,
        "one_generated_group_complex128_bytes": generated_bytes,
        "one_generated_group_fft_workspace_complex128_bytes": fft_workspace_bytes,
        "unique_direct_group_max_complex128_bytes": unique_group_bytes,
        "cholesky_real_float64_bytes": factor_bytes,
        "pivot_work_bytes": work_bytes,
        "host_tensor_peak_bytes": host_tensor_peak,
        "device_jax_seed_complex128_bytes": seed_bytes,
        "device_remove_restore_phase_complex128_bytes_each": phase_plane_bytes,
        "device_reciprocal_phase_complex128_bytes": reciprocal_phase_bytes,
        "device_fft_value_fourier_translated_output_complex128_bytes": 4 * fft_workspace_bytes,
        "device_tensor_peak_bytes": device_tensor_peak,
        "process_pipeline_allowance_bytes": int(process_pipeline_allowance_bytes),
        "selection_peak_host_bytes": host_peak,
        "selection_peak_device_bytes": device_peak,
        "selection_peak_host_with_safety_bytes": host_with_safety,
        "selection_peak_device_with_safety_bytes": device_with_safety,
        "selection_peak_max_bytes": int(selection_peak_max_bytes),
        "selection_peak_safety_factor": float(peak_safety_factor),
        "capacity_condition": condition,
        "within_cache_policy": condition is None,
    }


def build_reciprocal_same_grid_seed(
    cell, grid_coords, canonical_kpts, partition, *, stats=None,
):
    """Evaluate only the validated primitive seed shells on the full grid."""
    validation = validate_reciprocal_orbit_partition(cell, partition)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpts = np.asarray(canonical_kpts, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3 or grid_coords.shape[0] == 0:
        raise ValueError("grid_coords must have nonempty shape (Ng,3).")
    if kpts.ndim != 2 or kpts.shape[1] != 3 or kpts.shape[0] == 0:
        raise ValueError("canonical_kpts must have nonempty shape (Nk,3).")
    seed = np.asarray(cell.pbc_eval_gto(
        "GTOval", grid_coords, kpts=list(kpts),
        shls_slice=validation["seed_shell_slice"],
    ), dtype=np.complex128)
    expected = (kpts.shape[0], grid_coords.shape[0], validation["NAO_reused"])
    if seed.shape != expected:
        raise ReciprocalOrbitPartitionError(
            f"{ReciprocalOrbitPartitionError.condition}: seed AO shape {seed.shape}, "
            f"expected {expected}."
        )
    if stats is not None:
        stats["seed_evaluations"] = stats.get("seed_evaluations", 0) + 1
        stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
        stats["grid_points"] = stats.get("grid_points", 0) + int(grid_coords.shape[0])
        stats["full_supercell_ao_allocation"] = False
    return seed, validation


def reciprocal_same_grid_ao_groups(
    cell, grid_coords, canonical_kpts, seed, validation, *, stats=None,
):
    """Yield seed, one regenerated replica at a time, then unique direct groups."""
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpts = np.asarray(canonical_kpts, dtype=np.float64)
    seed = np.asarray(seed, dtype=np.complex128)
    mesh = tuple(int(value) for value in np.asarray(cell.mesh, dtype=np.int64))
    n_kpts, n_grid, n_ao = seed.shape
    if seed.shape != (kpts.shape[0], grid_coords.shape[0], validation["NAO_reused"]):
        raise ValueError("seed does not match the validated reciprocal partition.")
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("reciprocal_same_grid requires jax_enable_x64=True for complex128 FFTs.")
    seed_device = jnp.asarray(seed, dtype=jnp.complex128)
    remove_phase = jnp.asarray(
        np.exp(-1j * (kpts @ grid_coords.T)), dtype=jnp.complex128,
    )
    g_vectors = np.asarray(cell.get_Gv(cell.mesh), dtype=np.float64)
    if g_vectors.shape != grid_coords.shape:
        raise ValueError("cell G-vectors must have shape (Ng,3) for the selected grid.")
    yield seed
    # The metric consumers explicitly drop their previous group before
    # advancing this generator.  Delete the generator's reference too before
    # constructing another replica, so a multi-replica orbit never retains two
    # generated NumPy groups across a ``next()`` transition.
    generated = None
    restore_phase = None
    reciprocal_phase = None

    def release_generated():
        nonlocal generated
        if generated is None:
            return
        prior_ref = weakref.ref(generated)
        del generated
        generated = None
        if stats is not None:
            stats["generated_group_release_checks"] = (
                stats.get("generated_group_release_checks", 0) + 1
            )
            stats["generated_group_release_failures"] = (
                stats.get("generated_group_release_failures", 0)
                + int(prior_ref() is not None)
            )

    for target_slice, translation in zip(
        validation["replica_shell_slices"], validation["replica_translations"], strict=True,
    ):
        release_generated()
        if restore_phase is not None:
            del restore_phase
            restore_phase = None
        if reciprocal_phase is not None:
            del reciprocal_phase
            reciprocal_phase = None
        restore_phase = jnp.asarray(
            np.exp(-1j * (kpts @ translation))[:, None]
            * np.exp(1j * (kpts @ grid_coords.T)),
            dtype=jnp.complex128,
        )
        reciprocal_phase = jnp.asarray(
            np.exp(-1j * (g_vectors @ translation)), dtype=jnp.complex128,
        )
        generated = np.asarray(_reciprocal_same_grid_translate_kernel(
            seed_device, remove_phase, restore_phase, reciprocal_phase, mesh=mesh,
        ))
        expected = (n_kpts, n_grid, n_ao)
        if generated.shape != expected:
            raise RuntimeError(f"reciprocal FFT generator returned {generated.shape}, expected {expected}.")
        if stats is not None:
            stats["reconstruction_count"] = stats.get("reconstruction_count", 0) + 1
            stats["fft_count"] = stats.get("fft_count", 0) + 2 * n_kpts
            stats["generated_group_peak_count"] = max(
                stats.get("generated_group_peak_count", 0), 1,
            )
            stats["full_supercell_ao_allocation"] = False
            stats.setdefault("generated_target_shell_slices", []).append(list(target_slice))
        yield generated
    release_generated()
    unique = None

    def release_unique():
        nonlocal unique
        if unique is None:
            return
        prior_ref = weakref.ref(unique)
        del unique
        unique = None
        if stats is not None:
            stats["unique_group_release_checks"] = (
                stats.get("unique_group_release_checks", 0) + 1
            )
            stats["unique_group_release_failures"] = (
                stats.get("unique_group_release_failures", 0)
                + int(prior_ref() is not None)
            )

    for unique_slice in validation["unique_shell_slices"]:
        release_unique()
        unique = np.asarray(cell.pbc_eval_gto(
            "GTOval", grid_coords, kpts=list(kpts), shls_slice=unique_slice,
        ), dtype=np.complex128)
        if stats is not None:
            stats["unique_direct_evaluations"] = stats.get("unique_direct_evaluations", 0) + 1
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + int(n_grid)
            stats["full_supercell_ao_allocation"] = False
        yield unique
    release_unique()


def select_reciprocal_same_grid(
    cell, canonical_kpts, grid_coords, rank, partition, *,
    selection_peak_max_bytes, peak_safety_factor=1.10,
    process_pipeline_allowance_bytes=None, return_factor=False,
):
    """Select non-default pivots from seed-plus-one-group reciprocal AOs."""
    validation = validate_reciprocal_orbit_partition(cell, partition)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError("rank must be a positive integer.")
    rank = int(rank)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if rank > grid_coords.shape[0]:
        raise ValueError("rank cannot exceed the number of grid points.")
    byte_model = reciprocal_same_grid_byte_model(
        len(canonical_kpts), grid_coords.shape[0], validation["NAO_reused"],
        validation["NAO_unique"], validation["n_replicas"], rank,
        selection_peak_max_bytes=selection_peak_max_bytes,
        peak_safety_factor=peak_safety_factor,
        process_pipeline_allowance_bytes=process_pipeline_allowance_bytes,
    )
    if byte_model["capacity_condition"] is not None:
        raise ReciprocalSameGridCapacityError(
            f"{byte_model['capacity_condition']}: selector peak with safety requires "
            f"{max(byte_model['selection_peak_host_with_safety_bytes'], byte_model['selection_peak_device_with_safety_bytes'])} "
            f"bytes, policy allows {byte_model['selection_peak_max_bytes']} bytes."
        )
    stats = {"full_supercell_ao_allocation": False}
    seed, validation = build_reciprocal_same_grid_seed(
        cell, grid_coords, canonical_kpts, partition, stats=stats,
    )

    def groups():
        return reciprocal_same_grid_ao_groups(
            cell, grid_coords, canonical_kpts, seed, validation, stats=stats,
        )

    diagonal = metric_diagonal_from_ao_groups(groups(), len(canonical_kpts))

    def column(pivot):
        return metric_column_from_ao_groups(groups(), pivot, len(canonical_kpts))

    pivots, factor, n_selected = pivoted_cholesky_hermitian(diagonal, column, rank=rank)
    factor = np.asarray(factor.real, dtype=np.float64)
    provenance = {
        **byte_model,
        **validation,
        "pivot_executor": "host matrix-free pivoted_cholesky_hermitian",
        "fft_executor": "jax.jit/jnp.fft.fftn+ifftn",
        "persistent_seed_shape": list(seed.shape),
        "persistent_seed_dtype": str(seed.dtype),
        "factor_dtype": str(factor.dtype),
        "reconstruction_count": int(stats.get("reconstruction_count", 0)),
        "fft_count": int(stats.get("fft_count", 0)),
        "unique_direct_evaluations": int(stats.get("unique_direct_evaluations", 0)),
        "seed_evaluations": int(stats.get("seed_evaluations", 0)),
        "pbc_eval_calls": int(stats.get("pbc_eval_calls", 0)),
        "ao_grid_points": int(stats.get("grid_points", 0)),
        "hidden_full_supercell_ao_allocation": bool(
            stats.get("full_supercell_ao_allocation", False)
        ),
        "generated_group_live_limit": 1,
        "generated_group_release_checks": int(stats.get("generated_group_release_checks", 0)),
        "generated_group_release_failures": int(stats.get("generated_group_release_failures", 0)),
        "unique_group_release_checks": int(stats.get("unique_group_release_checks", 0)),
        "unique_group_release_failures": int(stats.get("unique_group_release_failures", 0)),
        "selector_ao_layout": "persistent primitive seed + one generated replica or direct unique group",
    }
    return pivots, factor if return_factor else None, n_selected, provenance


def periodic_part_from_bloch(bloch_ao, grid_coords, kpt):
    """Return sampled ``u(r)`` for ``phi(r) = exp(i k.r) u(r)``."""
    bloch_ao = np.asarray(bloch_ao, dtype=np.complex128)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpt = np.asarray(kpt, dtype=np.float64)
    if bloch_ao.ndim != 2 or bloch_ao.shape[0] != grid_coords.shape[0]:
        raise ValueError("bloch_ao must have shape (Ng,Nao) for grid_coords.")
    return np.exp(-1j * (grid_coords @ kpt))[:, None] * bloch_ao


def reciprocal_translate_bloch_ao(
    seed_bloch_ao, grid_coords, mesh, kpt, translation, g_vectors,
):
    """Generate translated sampled Bloch AOs from a periodic-part FFT.

    This is a same-grid spectral translation.  It is not exact unless the
    periodic part is represented without aliasing on ``mesh``.
    """
    mesh = np.asarray(mesh, dtype=np.int64)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpt = np.asarray(kpt, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    g_vectors = np.asarray(g_vectors, dtype=np.float64)
    u_seed = periodic_part_from_bloch(seed_bloch_ao, grid_coords, kpt)
    if g_vectors.shape != grid_coords.shape:
        raise ValueError("g_vectors must have the same shape as grid_coords.")
    u_g = pbc_tools.fft(u_seed.T, mesh)
    translated_u = pbc_tools.ifft(
        u_g * np.exp(-1j * (g_vectors @ translation)), mesh,
    ).T
    return (
        np.exp(-1j * (kpt @ translation))
        * np.exp(1j * (grid_coords @ kpt))[:, None]
        * translated_u
    )


def relative_frobenius_error(reference, candidate):
    """Return max-abs and relative Frobenius error for matching arrays."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes.")
    difference = reference - candidate
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / denominator),
    }


def uniform_grid_downsample_indices(coarse_mesh, fine_mesh):
    """Map a commensurate fine uniform grid back to its coarse-grid points."""
    coarse_mesh = np.asarray(coarse_mesh, dtype=np.int64)
    fine_mesh = np.asarray(fine_mesh, dtype=np.int64)
    if coarse_mesh.shape != (3,) or fine_mesh.shape != (3,):
        raise ValueError("meshes must have shape (3,).")
    if np.any(coarse_mesh <= 0) or np.any(fine_mesh <= 0):
        raise ValueError("mesh entries must be positive.")
    if np.any(fine_mesh % coarse_mesh):
        raise ValueError("fine_mesh must be an integer multiple of coarse_mesh.")
    factor = fine_mesh // coarse_mesh
    coarse_indices = np.array(np.unravel_index(np.arange(np.prod(coarse_mesh)), coarse_mesh)).T
    return np.ravel_multi_index((coarse_indices * factor).T, fine_mesh)


def downsample_uniform_grid_values(values, coarse_mesh, fine_mesh):
    """Select matching coarse-grid coordinates from a fine-grid AO array."""
    values = np.asarray(values)
    if values.shape[0] != int(np.prod(fine_mesh)):
        raise ValueError("values first axis must equal the fine-grid size.")
    return values[uniform_grid_downsample_indices(coarse_mesh, fine_mesh)]


def metric_column_from_ao_groups(groups, pivot, n_kpts):
    """Form one exact metric column after accumulating every AO group."""
    accumulator = None
    for group in groups:
        group = np.asarray(group, dtype=np.complex128)
        if group.ndim != 3:
            raise ValueError("each AO group must have shape (Nk,Ng,Nao).")
        contribution = np.einsum(
            "ka,kga->g", group[:, pivot, :].conj(), group, optimize=True,
        )
        accumulator = contribution if accumulator is None else accumulator + contribution
        # See reciprocal_same_grid_ao_groups: releasing this consumer-side
        # reference before the next ``next()`` prevents two replica groups
        # being live during a multi-replica transition.
        del group
    if accumulator is None:
        raise ValueError("at least one AO group is required.")
    return np.abs(accumulator) ** 2 / n_kpts


def metric_diagonal_from_ao_groups(groups, n_kpts):
    """Form the exact metric diagonal while holding one AO group at a time."""
    density = None
    for group in groups:
        group = np.asarray(group, dtype=np.complex128)
        if group.ndim != 3:
            raise ValueError("each AO group must have shape (Nk,Ng,Nao).")
        contribution = np.sum(np.abs(group) ** 2, axis=(0, 2))
        density = contribution if density is None else density + contribution
        del group
    if density is None:
        raise ValueError("at least one AO group is required.")
    return density ** 2 / n_kpts


def pivot_prefix_from_ao_groups(group_factory, n_grid, n_kpts, rank):
    """Select a short exact prefix through streamed AO-group contributions."""
    diagonal = metric_diagonal_from_ao_groups(group_factory(), n_kpts)

    def column(pivot):
        return metric_column_from_ao_groups(group_factory(), pivot, n_kpts)

    return pivoted_cholesky_hermitian(diagonal, column, rank=rank)
