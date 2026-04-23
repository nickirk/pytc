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
    assert m0 * m1 == len(devices), (m0, m1, len(devices))
    device_grid = np.asarray(devices).reshape(m0, m1)
    return Mesh(device_grid, axis_names=axis_names)


def choose_2d_mesh_shape(n_devices: int, n_fused: int, n_grid: int):
    """Pick (m_k, m_g) factorization of ``n_devices`` that minimises
    the dominant per-device memory in K-kernel construction.

    Peak per-device cost balances replicated K1 (~3*n_fused^2 / m_k)
    against replicated-along-k xi_phi_r2 (~n_fused*n_grid / m_g).

    Optimal continuous m_k is sqrt(3*n_fused*n_devices / n_grid); we
    snap to the nearest divisor of n_devices.
    """
    if n_devices <= 1:
        return (1, 1)
    divisors = [d for d in range(1, n_devices + 1) if n_devices % d == 0]
    m_k_opt = (3.0 * float(n_fused) * n_devices / max(float(n_grid), 1.0)) ** 0.5
    m_k = min(divisors, key=lambda d: (abs(d - m_k_opt), d))
    return (m_k, n_devices // m_k)


def get_partitioned_sharding(mesh: Mesh, axis_name: str = "devices") -> NamedSharding:
    """Return sharding that partitions along one leading logical axis."""
    return NamedSharding(mesh, P(axis_name))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return fully replicated sharding."""
    return NamedSharding(mesh, P())


def pad_axis_to_multiple(arr, multiple: int, axis: int = 0, pad_value=0):
    """Pad array along one axis to be divisible by ``multiple``."""
    arr_np = np.asarray(arr)
    n = arr_np.shape[axis]
    rem = n % multiple
    if rem == 0:
        return arr_np, 0

    pad = multiple - rem
    pad_width = [(0, 0)] * arr_np.ndim
    pad_width[axis] = (0, pad)
    arr_pad = np.pad(arr_np, pad_width, mode="constant", constant_values=pad_value)
    return arr_pad, pad


def device_put_sharded_along_axis(arr, devices, axis: int = 0, pad_value=0):
    """Shard an array across devices along the selected axis.

    Returns
    -------
    sharded : jax.Array
        Result of ``jax.device_put_sharded``.
    original_len : int
        Original size on the sharded axis before padding.
    per_device : int
        Per-device chunk length after padding.
    """
    arr_pad, _ = pad_axis_to_multiple(arr, len(devices), axis=axis, pad_value=pad_value)
    axis = axis % arr_pad.ndim
    original_len = np.asarray(arr).shape[axis]
    chunks = np.split(arr_pad, len(devices), axis=axis)
    per_device = chunks[0].shape[axis]
    per_dev_arrays = [jax.device_put(chunks[i], devices[i]) for i in range(len(devices))]
    sharded = jax.device_put_sharded(per_dev_arrays, devices)
    return sharded, original_len, per_device
