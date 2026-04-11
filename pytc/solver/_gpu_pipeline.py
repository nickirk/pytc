"""Multi-GPU tile-pipeline primitives shared by xtc_ccsd and jax_xtc_ccsd.

Public API
----------
_solver_local_devices()
broadcast_to_devices(arr, devices)
_gpu_slot_ctx(gpu_sem)
_round_robin_pipeline(tile_specs, issue_tile, consume_tile, ...)
"""

import contextlib
import concurrent.futures
import threading

import numpy as np
import jax


def _solver_local_devices():
    """Return local accelerator devices for solver-side round-robin scheduling."""
    devices = tuple(jax.local_devices())
    return devices if devices else (None,)


def broadcast_to_devices(arr, devices):
    """Copy *arr* to every device in *devices*; return a ``{device: copy}`` dict.

    For ``device=None`` (CPU-only / no local accelerator) the original *arr*
    is stored unchanged.  For each real device, *arr* is materialised to
    NumPy once and sent with ``jax.device_put``.  A single NumPy copy is
    reused across all ``device_put`` calls so there is no redundant host
    allocation.

    Parameters
    ----------
    arr : array-like
        Array to broadcast.  May be a NumPy array, a JAX array, or any
        object accepted by ``np.asarray``.
    devices : iterable
        Local devices returned by ``_solver_local_devices()``.

    Returns
    -------
    dict
        ``{device: arr_on_device}`` — values are on-device JAX arrays
        for real devices, or *arr* unchanged for ``None``.

    Examples
    --------
    >>> t1_by_dev = broadcast_to_devices(t1_jax, _solver_local_devices())
    >>> t1_on_this_device = t1_by_dev[device]
    """
    arr_np = np.asarray(arr)   # materialise once; no-op if already NumPy
    return {
        device: arr if device is None else jax.device_put(arr_np, device)
        for device in devices
    }


@contextlib.contextmanager
def _gpu_slot_ctx(gpu_sem):
    """Context manager that yields an idempotent GPU-slot release callable.

    The yielded ``release()`` function releases *gpu_sem* exactly once no
    matter how many times it is called (subsequent calls are no-ops).  On
    exit the context manager calls ``release()`` itself as a safety net, so
    the slot is never leaked even if the consumer forgets to call it.

    Typical usage inside a consume callback::

        with _gpu_slot_ctx(gpu_sem) as release_gpu_slot:
            data = np.asarray(handle)   # GPU→CPU readback
            release_gpu_slot()          # free GPU slot immediately
            ... CPU post-processing ...
    """
    _released = False
    _lock = threading.Lock()

    def release():
        nonlocal _released
        with _lock:
            if _released:
                return
            _released = True
        gpu_sem.release()

    try:
        yield release
    finally:
        release()  # safety net — idempotent, so harmless if already called


def _round_robin_pipeline(tile_specs, issue_tile, consume_tile, devices=None,
                          device_key=None, gpu_slots=None, host_slots=None):
    """Issue tiles to devices in round-robin; run consume callbacks in a thread pool.

    Parameters
    ----------
    device_key : callable(spec) -> hashable, optional
        When provided, all tiles that share the same key are sent to the same
        device.  Keys are assigned to devices in first-seen order round-robin.
        Default (None) assigns tiles by sequential tile index.
    gpu_slots : int, optional
        Number of tiles allowed to be in flight on the GPU pipeline before the
        main dispatch thread must wait.  Defaults to ``2 * n_devices``.  A
        GPU slot is released the moment ``consume_tile`` calls
        ``release_gpu_slot`` (typically right after the GPU→CPU readback),
        freeing the main thread to issue the next tile even while CPU post-
        processing and disk writes for earlier tiles are still in flight.
    host_slots : int, optional
        Number of tiles allowed to hold host-side buffers concurrently.
        Defaults to ``4 * n_devices``.  Bounds peak host memory for in-flight
        CPU contractions and HDF5 writes.  Must be ``>= gpu_slots``.

    GPU dispatch (issue_tile) runs on the main thread so JAX sees a clean
    per-device dispatch order.  consume_tile callbacks run off the main thread
    so that GPU→CPU transfers, CPU contractions, and I/O all overlap with the
    next round of GPU dispatches, keeping all GPUs continuously fed.

    ``consume_tile`` is called as ``consume_tile(spec, device, handle,
    release_gpu_slot)``.  It MUST call ``release_gpu_slot()`` as soon as the
    tile is drained from the GPU (typically immediately after
    ``np.asarray(handle)``) so the main thread can issue the next GPU tile
    while CPU-only work continues.  ``release_gpu_slot`` is idempotent and
    guaranteed to be called on exit even if consume_tile raises.

    consume_tile is responsible for its own thread safety — callers that write
    to shared state (e.g. HDF5 datasets) should guard those writes with a
    threading.Lock in their closure.
    """
    devices = devices or _solver_local_devices()
    n_devices = len(devices)
    if gpu_slots is None:
        gpu_slots = 2 * n_devices
    if host_slots is None:
        host_slots = max(gpu_slots, 4 * n_devices)
    if host_slots < gpu_slots:
        host_slots = gpu_slots
    _key_to_device = {}

    gpu_sem  = threading.Semaphore(gpu_slots)
    host_sem = threading.Semaphore(host_slots)

    def _run_consume(spec, device, handle):
        try:
            with _gpu_slot_ctx(gpu_sem) as release_gpu_slot:
                consume_tile(spec, device, handle, release_gpu_slot)
        finally:
            host_sem.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=host_slots) as pool:
        futures = []
        for tile_id, spec in enumerate(tile_specs):
            if device_key is not None:
                k = device_key(spec)
                if k not in _key_to_device:
                    _key_to_device[k] = devices[len(_key_to_device) % n_devices]
                device = _key_to_device[k]
            else:
                device = devices[tile_id % n_devices]
            host_sem.acquire()  # bound host-side memory
            gpu_sem.acquire()   # bound GPU pipeline depth
            handle = issue_tile(spec, device)
            futures.append(pool.submit(_run_consume, spec, device, handle))
        for f in futures:
            f.result()  # re-raises first consume-thread exception on the main thread
