"""Standalone CPU comparator for periodic spherical AOs in JAX.

This module is a feasibility pilot only.  It deliberately yields bounded AO
blocks and is not connected to the periodic selector or a production default.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from pytc.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical


@dataclasses.dataclass(frozen=True)
class PeriodicSphericalAOPilot:
    """Static spherical basis data and PySCF-matched lattice images."""

    spherical: MolGTO_Spherical
    lattice_vectors: np.ndarray

    @classmethod
    def create(cls, cell):
        return cls(
            spherical=MolGTO_Spherical.create(cell),
            lattice_vectors=np.asarray(cell.get_lattice_Ls(), dtype=np.float64),
        )

    @property
    def nao(self):
        return self.spherical.nao


def compile_spherical_block_evaluator(pilot, grid_block_size, *, image_block_size=1):
    """Compile the bounded batched-image real spherical AO evaluator."""
    if isinstance(grid_block_size, bool) or not isinstance(grid_block_size, (int, np.integer)):
        raise ValueError("grid_block_size must be a positive integer.")
    if grid_block_size <= 0:
        raise ValueError("grid_block_size must be a positive integer.")
    if isinstance(image_block_size, bool) or not isinstance(image_block_size, (int, np.integer)):
        raise ValueError("image_block_size must be a positive integer.")
    if image_block_size <= 0:
        raise ValueError("image_block_size must be a positive integer.")

    @jax.jit
    def evaluate(image_coords):
        return jax.vmap(lambda coords: eval_ao_spherical(pilot.spherical, coords))(image_coords)

    zero_coords = jnp.zeros(
        (int(image_block_size), int(grid_block_size), 3), dtype=jnp.float64,
    )
    evaluate.lower(zero_coords).compile()
    return evaluate


def stream_periodic_spherical_ao_blocks(
    pilot, compiled_block_evaluator, grid_coords, kpt, *, grid_block_size,
    image_block_size=1, stats=None,
):
    """Yield periodic Bloch AO blocks without an image-axis tensor.

    The inner image loop bounds live state to one ``(grid_block,AO)``
    contribution plus the output accumulator.  It does not materialize a
    grid-by-shell-by-image-by-primitive object.
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpt = np.asarray(kpt, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError("grid_coords must have shape (Ng,3).")
    if isinstance(grid_block_size, bool) or not isinstance(grid_block_size, (int, np.integer)):
        raise ValueError("grid_block_size must be a positive integer.")
    if isinstance(image_block_size, bool) or not isinstance(image_block_size, (int, np.integer)):
        raise ValueError("image_block_size must be a positive integer.")
    grid_block_size = int(grid_block_size)
    image_block_size = int(image_block_size)
    if grid_block_size <= 0 or image_block_size <= 0:
        raise ValueError("block sizes must be positive.")

    lattice_vectors = pilot.lattice_vectors
    phase_weights = np.exp(1j * (lattice_vectors @ kpt))
    for g0 in range(0, grid_coords.shape[0], grid_block_size):
        g1 = min(g0 + grid_block_size, grid_coords.shape[0])
        block = grid_coords[g0:g1]
        accumulator = np.zeros((g1 - g0, pilot.nao), dtype=np.complex128)
        for l0 in range(0, lattice_vectors.shape[0], image_block_size):
            l1 = min(l0 + image_block_size, lattice_vectors.shape[0])
            images = np.zeros((image_block_size, 3), dtype=np.float64)
            weights = np.zeros((image_block_size,), dtype=np.complex128)
            count = l1 - l0
            images[:count] = lattice_vectors[l0:l1]
            weights[:count] = phase_weights[l0:l1]
            contribution = np.asarray(
                compiled_block_evaluator(jnp.asarray(block[None, :, :] - images[:, None, :])),
                dtype=np.float64,
            )
            accumulator += np.einsum("i,iga->ga", weights, contribution, optimize=True)
            if stats is not None:
                stats["image_blocks"] = stats.get("image_blocks", 0) + 1
                stats["jax_image_evaluations"] = (
                    stats.get("jax_image_evaluations", 0) + count
                )
                stats["jax_compiled_calls"] = stats.get("jax_compiled_calls", 0) + 1
        if stats is not None:
            stats["grid_blocks"] = stats.get("grid_blocks", 0) + 1
        yield g0, g1, accumulator
