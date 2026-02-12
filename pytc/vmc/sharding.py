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

    from pytc.vmc.sharding import create_mesh, shard_walker, get_sharding

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

try:
    from jax.shard_map import shard_map
except ImportError:
    try:
        from jax.experimental.shard_map import shard_map
    except ImportError:
        shard_map = None

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def n_devices() -> int:
    """Number of local devices available to this process."""
    return jax.local_device_count()


def is_multi_gpu() -> bool:
    """True if more than one local device is available."""
    return n_devices() > 1


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

def sharded_batched_vmap(fn, max_batch_size, mesh=None, in_axes=0, out_axes=0):
    """A version of folx.batched_vmap that works across multiple devices.

    It uses ``jax.shard_map`` to distribute the computation across devices,
    and then ``folx.batched_vmap`` on each device to process the local shard
    in memory-efficient chunks.
    """
    import folx
    import functools

    if shard_map is None:
        import warnings
        warnings.warn("shard_map not found, falling back to jax.vmap")
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    if mesh is None:
        mesh = create_mesh()

    # Determine sharding specs based on in_axes/out_axes
    # We assume 'walkers' is the axis name
    def _axis_to_spec(ax):
        if ax == 0: return P("walkers")
        return P(None)

    if isinstance(in_axes, (list, tuple)):
        in_specs = tuple(_axis_to_spec(ax) for ax in in_axes)
    else:
        in_specs = _axis_to_spec(in_axes)

    if isinstance(out_axes, (list, tuple)):
        out_specs = tuple(_axis_to_spec(ax) for ax in out_axes)
    else:
        out_specs = _axis_to_spec(out_axes)

    # The inner function used by shard_map
    def local_batched_fn(*args, **kwargs):
        return folx.batched_vmap(
            fn, max_batch_size=max_batch_size,
            in_axes=in_axes, out_axes=out_axes
        )(*args, **kwargs)

    return shard_map(
        local_batched_fn, mesh=mesh,
        in_specs=in_specs, out_specs=out_specs
    )


def shard_vmap(fn, mesh=None, in_axes=0, out_axes=0):
    """A version of jax.vmap that works across multiple devices using shard_map."""
    if shard_map is None:
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    if mesh is None:
        mesh = create_mesh()

    # Determine sharding specs based on in_axes/out_axes
    def _axis_to_spec(ax):
        if ax == 0: return P("walkers")
        return P(None)

    if isinstance(in_axes, (list, tuple)):
        in_specs = tuple(_axis_to_spec(ax) for ax in in_axes)
    else:
        in_specs = _axis_to_spec(in_axes)

    if isinstance(out_axes, (list, tuple)):
        out_specs = tuple(_axis_to_spec(ax) for ax in out_axes)
    else:
        out_specs = _axis_to_spec(out_axes)

    return shard_map(
        jax.vmap(fn, in_axes=in_axes, out_axes=out_axes),
        mesh=mesh,
        in_specs=in_specs, out_specs=out_specs
    )


def get_vmap_fn(max_vmap_batch_size: int = 0, mesh: Optional[Mesh] = None):
    """Return the appropriate vmap implementation.

    Automatically detects if multiple GPUs are available. If so, and
    ``max_vmap_batch_size > 0``, uses ``sharded_batched_vmap`` which
    combines sharding and batching. If ``max_vmap_batch_size == 0``,
    uses ``shard_vmap`` for sharded data-parallel execution.

    Returns
    -------
    callable
        A vmap-like function with ``(fn, in_axes, ...)`` signature.
    """
    if is_multi_gpu():
        if max_vmap_batch_size > 0:
            import functools
            return functools.partial(
                sharded_batched_vmap,
                max_batch_size=max_vmap_batch_size,
                mesh=mesh
            )
        else:
            import functools
            return functools.partial(
                shard_vmap,
                mesh=mesh
            )
    else:
        if max_vmap_batch_size > 0:
            import folx
            import functools
            return functools.partial(
                folx.batched_vmap, max_batch_size=max_vmap_batch_size
            )
        else:
            return jax.vmap


