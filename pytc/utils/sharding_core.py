"""Common multi-device sharding helpers shared across modules."""

from __future__ import annotations

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def n_local_devices() -> int:
    """Return number of local devices visible to this process."""
    return jax.local_device_count()


def is_multi_device() -> bool:
    """Return True if more than one local device is available."""
    return n_local_devices() > 1


def create_1d_mesh(devices=None, axis_name: str = "devices") -> Mesh:
    """Create a 1D mesh across all provided devices."""
    if devices is None:
        devices = jax.devices()
    return Mesh(devices, axis_names=(axis_name,))


def create_2d_mesh(devices, shape, axis_names=("k_ax", "g_ax")) -> Mesh:
    """Create a 2D mesh of the given shape over ``devices``.

    ``shape`` is ``(m_0, m_1)`` with ``m_0 * m_1 == len(devices)``.
    """
    m0, m1 = shape
    if m0 * m1 != len(devices):
        raise ValueError(
            f"create_2d_mesh: shape {shape} does not match device count "
            f"{len(devices)} (m0*m1={m0 * m1})."
        )
    device_grid = np.asarray(devices).reshape(m0, m1)
    return Mesh(device_grid, axis_names=axis_names)



def get_partitioned_sharding(mesh: Mesh, axis_name: str = "devices") -> NamedSharding:
    """Return sharding that partitions along one leading logical axis."""
    return NamedSharding(mesh, P(axis_name))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return fully replicated sharding."""
    return NamedSharding(mesh, P())



