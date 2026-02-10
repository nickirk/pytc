"""Multi-GPU sharding utilities for VMC.

This module provides helpers for distributing VMC walkers across multiple
devices (GPUs/TPUs) using JAX's built-in sharding.

The strategy is **data-parallel over walkers**: each device holds N/D walkers
and performs per-walker computations independently.  Only lightweight
reductions (mean energy, J^T @ δE, J^T @ J) trigger cross-device
communication, which JAX inserts automatically.

Usage
-----
::

    from pytc.autodiff.vmc.sharding import create_mesh, shard_walker, get_sharding

    mesh = create_mesh()                    # auto-detect devices
    walkers = shard_walker(walkers, mesh)   # shard along walker axis
    params  = replicate(params, mesh)       # replicate on all devices

After sharding, the existing ``jax.vmap`` and ``jax.jit`` code works
unchanged — JAX's SPMD compiler infers the communication.

Notes
-----
* ``folx.batched_vmap`` does **not** preserve sharding (it gathers results
  to all devices).  When ``multi_gpu=True`` the code should fall back to
  ``jax.vmap`` so that per-walker outputs stay distributed.
* Walkers must have ``n_walkers`` divisible by the number of devices.
  ``pad_walkers`` can add padding walkers to satisfy this.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Optional


# ---------------------------------------------------------------------------
# Mesh creation
# ---------------------------------------------------------------------------

def create_mesh(devices=None, axis_name: str = "walkers"):
    """Create a 1-D device mesh for walker-parallel sharding.

    Parameters
    ----------
    devices : sequence of jax.Device, optional
        Devices to include.  Defaults to ``jax.devices()``.
    axis_name : str
        Name for the walker-parallel axis (default ``"walkers"``).

    Returns
    -------
    jax.sharding.Mesh
    """
    if devices is None:
        devices = jax.devices()
    return Mesh(devices, axis_names=(axis_name,))


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def get_walker_sharding(mesh: Mesh, axis_name: str = "walkers"):
    """NamedSharding that partitions the leading (walker) dimension."""
    return NamedSharding(mesh, P(axis_name))


def get_replicated_sharding(mesh: Mesh):
    """NamedSharding that replicates data on every device."""
    return NamedSharding(mesh, P())


def shard_walker(walker, mesh: Mesh, axis_name: str = "walkers"):
    """Place a Walker (or any pytree) with sharding along axis 0.

    Every leaf whose leading dimension equals ``n_walkers`` is split
    across devices.  Scalar leaves are replicated automatically.
    """
    sharding = get_walker_sharding(mesh, axis_name)
    return jax.device_put(walker, sharding)


def replicate(pytree, mesh: Mesh):
    """Replicate a pytree on every device."""
    sharding = get_replicated_sharding(mesh)
    return jax.device_put(pytree, sharding)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad_n_walkers(n_walkers: int, n_devices: int) -> int:
    """Return the smallest multiple of *n_devices* >= *n_walkers*."""
    remainder = n_walkers % n_devices
    if remainder == 0:
        return n_walkers
    return n_walkers + (n_devices - remainder)


def pad_walker(walker, target_n_walkers: int):
    """Pad a Walker pytree along axis 0 to *target_n_walkers*.

    Extra walkers are copies of the first walker (so they have valid
    shapes/dtypes for JIT tracing).  They should be excluded from
    statistics after the training step.
    """
    current = walker.positions.shape[0]
    if current >= target_n_walkers:
        return walker, current  # no padding needed

    pad_count = target_n_walkers - current

    def _pad_leaf(x):
        if x.ndim == 0:
            return x
        # Repeat the first element `pad_count` times
        tile = jnp.repeat(x[:1], pad_count, axis=0)
        return jnp.concatenate([x, tile], axis=0)

    # Handle tuple fields (det_up, det_down) via tree_map
    padded = jax.tree_util.tree_map(_pad_leaf, walker)
    return padded, current  # return original count for later un-padding


# ---------------------------------------------------------------------------
# Multi-GPU aware vmap selection
# ---------------------------------------------------------------------------

def get_vmap_fn(multi_gpu: bool, max_vmap_batch_size: int = 0):
    """Return the appropriate vmap implementation.

    When ``multi_gpu=True``, always use ``jax.vmap`` (preserves sharding).
    When ``multi_gpu=False``, honour ``max_vmap_batch_size`` as before
    (``folx.batched_vmap`` for memory efficiency on a single device).

    Returns
    -------
    callable
        A vmap-like function with ``(fn, in_axes, ...)`` signature.
    """
    if multi_gpu:
        # jax.vmap preserves sharding; folx.batched_vmap does not.
        return jax.vmap
    else:
        if max_vmap_batch_size > 0:
            import folx
            import functools
            return functools.partial(
                folx.batched_vmap, max_batch_size=max_vmap_batch_size
            )
        else:
            return jax.vmap


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def is_multi_gpu() -> bool:
    """True if more than one device is available."""
    return jax.device_count() > 1


def n_devices() -> int:
    """Number of local devices."""
    return jax.local_device_count()
