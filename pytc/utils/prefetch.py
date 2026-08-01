"""
Asynchronous prefetch utilities for overlapping HDF5 I/O with compute.

The main building blocks are:

1. :class:`PrefetchIterator` — wraps a block-iteration loop so that the
   *next* block's data is read from HDF5 (or sliced from a host array)
   in a background thread while the *current* block is being processed
   on the CPU or GPU.

2. :func:`async_read` / :func:`await_read` — low-level helpers for
   issuing a single read in a background thread and awaiting the result.

These helpers are intentionally **thread-based** (not process-based)
because h5py releases the GIL during I/O and numpy slice operations
are also GIL-free, so a single background thread is sufficient to fully
overlap disk reads with CPU/GPU compute.

Usage example (OVVV consume loop)::

    from pytc.utils.prefetch import PrefetchIterator

    def _load_ovvv_block(p0p1):
        p0, p1 = p0p1
        return _get_slice(eris.ovvv, slice(p0, p1), axis=2)

    chunks = [(p0, min(p0 + blksize, nvir))
              for p0 in range(0, nvir, blksize)]

    for (p0, p1), ovvv_blk in PrefetchIterator(chunks, _load_ovvv_block):
        _process_ovvv_block_from_array(ovvv_blk, ...)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

# HDF5 is NOT thread-safe by default — serialize all HDF5 reads
_HDF5_LOCK = threading.Lock()


def safe_hdf5_read(dataset: Any, idx: Any) -> Any:
    """Safely read from HDF5 dataset with thread serialization.
    
    If dataset is an HDF5 dataset (has .file attribute), the read is
    protected by _HDF5_LOCK to prevent deadlocks. For numpy arrays or
    other objects, reads directly without locking.
    
    Parameters
    ----------
    dataset : h5py.Dataset or array-like
        The dataset to read from.
    idx : tuple or slice
        The index/slice to read.
        
    Returns
    -------
    np.ndarray
        The read data as a numpy array.
    """
    import numpy as np
    
    is_hdf5 = hasattr(dataset, "file")
    if is_hdf5:
        with _HDF5_LOCK:
            return np.asarray(dataset[idx])
    else:
        return np.asarray(dataset[idx])

K = TypeVar("K")   # chunk key  (e.g. (p0, p1) tuple)
V = TypeVar("V")   # loaded value (e.g. numpy array)

# ---------------------------------------------------------------------------
# High-level iterator
# ---------------------------------------------------------------------------


class PrefetchIterator(Generic[K, V]):
    """Iterate over chunk keys, prefetching one block ahead in a background thread.

    Parameters
    ----------
    keys : Sequence[K]
        Ordered collection of chunk keys (e.g. ``[(0, 64), (64, 128), ...]``).
    load_fn : Callable[[K], V]
        Function that takes a key and returns a loaded data block.
        Called in a background thread; must be thread-safe and should
        *not* hold the GIL during the heavy I/O portion (h5py and
        numpy slice both release GIL).
    prefetch_depth : int
        How many blocks to read ahead.  ``1`` is almost always optimal
        (one read overlapping one compute) and avoids memory pressure.

    Yields
    ------
    (key, value) : Tuple[K, V]
        The chunk key and the loaded data block, in order.

    Notes
    -----
    The iterator uses ``ThreadPoolExecutor(max_workers=1)`` to keep
    overhead minimal.  ``__del__`` / context-manager protocol shuts
    the pool down, but explicit ``with`` usage is recommended for
    deterministic cleanup.
    """

    def __init__(
        self,
        keys: Sequence[K],
        load_fn: Callable[[K], V],
        prefetch_depth: int = 1,
    ) -> None:
        self._keys = keys
        self._load_fn = load_fn
        self._depth = max(1, prefetch_depth)
        self._pool = ThreadPoolExecutor(max_workers=self._depth)

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "PrefetchIterator[K, V]":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        self._pool.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # -- iteration ---------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[K, V]]:
        keys = self._keys
        n = len(keys)
        if n == 0:
            return

        futures: list[Future[V]] = []
        submit_idx = 0
        for _ in range(min(self._depth, n)):
            futures.append(self._pool.submit(self._load_fn, keys[submit_idx]))
            submit_idx += 1

        for i in range(n):
            value = futures.pop(0).result()

            if submit_idx < n:
                futures.append(
                    self._pool.submit(self._load_fn, keys[submit_idx])
                )
                submit_idx += 1

            yield keys[i], value


# ---------------------------------------------------------------------------
# Low-level one-shot async read
# ---------------------------------------------------------------------------

# Module-level single-thread pool for one-shot reads.
# Lazily initialised on first use.
_ONE_SHOT_POOL: ThreadPoolExecutor | None = None


def _get_one_shot_pool() -> ThreadPoolExecutor:
    global _ONE_SHOT_POOL
    if _ONE_SHOT_POOL is None:
        _ONE_SHOT_POOL = ThreadPoolExecutor(max_workers=1)
    return _ONE_SHOT_POOL


def async_read(fn: Callable[..., V], *args: Any, **kwargs: Any) -> Future[V]:
    """Submit *fn(*args, **kwargs)* to a background thread and return a :class:`Future`.

    Example::

        future = async_read(np.asarray, hdf5_dataset[slice_obj])
        # … do GPU work …
        data = future.result()   # blocks until read finishes
    """
    pool = _get_one_shot_pool()
    return pool.submit(fn, *args, **kwargs)


def await_read(future: Future[V]) -> V:
    """Block until a :func:`async_read` future completes and return the value."""
    return future.result()


# ---------------------------------------------------------------------------
# Convenience: HDF5 dataset slice reader
# ---------------------------------------------------------------------------


def hdf5_slice_loader(
    dataset: Any,
    axis: int = 0,
) -> Callable[[Tuple[int, int]], Any]:
    """Return a ``load_fn`` suitable for :class:`PrefetchIterator`.

    The returned callable takes a ``(start, stop)`` tuple and reads
    ``dataset[..., start:stop, ...]`` along *axis*, converting to a
    numpy array via ``np.asarray``.

    **Thread-safety**: All HDF5 reads are serialized via a module-level
    lock to avoid deadlocks (HDF5 is not thread-safe by default).

    Parameters
    ----------
    dataset : h5py.Dataset or numpy array
        The data source.
    axis : int
        The axis to slice along.

    Returns
    -------
    Callable[[Tuple[int, int]], np.ndarray]
    """
    ndim = len(dataset.shape) if hasattr(dataset, "shape") else None

    def _load(key: Tuple[int, int]) -> Any:
        p0, p1 = key
        if ndim is not None:
            idx = [slice(None)] * ndim
            idx[axis] = slice(p0, p1)
            return safe_hdf5_read(dataset, tuple(idx))
        return dataset

    return _load
