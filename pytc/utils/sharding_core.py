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
