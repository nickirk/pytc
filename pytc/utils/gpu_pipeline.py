"""Multi-GPU tile-pipeline primitives shared by xtc_ccsd and jax_xtc_ccsd.

Public API
----------
_solver_local_devices()
broadcast_to_devices(arr, devices)
_gpu_slot_ctx(gpu_sem)
_round_robin_pipeline(tile_specs, issue_tile, consume_tile, ...)
_AsyncHDF5Writer(max_pending=..., name=...)
"""

import contextlib
import concurrent.futures
import queue as _queue
import threading
import time

import numpy as np
import jax


class _AsyncHDF5Writer:
    """Drain HDF5 write jobs on a single dedicated background thread.

    Why a dedicated writer thread?
    ------------------------------
    HDF5 is not thread-safe, and in the default (non-threadsafe) build of
    libhdf5 concurrent ``dataset[...] = array`` calls from multiple Python
    threads can corrupt the file or crash the process.  Even the threadsafe
    build serialises every I/O call behind a single global lock, so running
    writes from multiple consume threads buys nothing but lock contention.

    Meanwhile, on the pipeline-consumer hot path, every second a consume
    thread spends inside HDF5 is a second it cannot return to the
    ``ThreadPoolExecutor`` pool — so ``host_sem`` (and eventually
    ``gpu_sem``) run out of slots and the main dispatch thread stalls on
    issuing the next GPU tile.

    Routing every write through one dedicated background thread fixes both
    problems in one move:

    * All HDF5 calls are serialised by construction (one writer, one file
      descriptor) — no lock needed on the caller side.
    * Consume threads submit-and-return in microseconds, so the pipeline
      stays saturated and the GPUs keep getting fed.

    Semantics
    ---------
    * FIFO: writes are executed in submit order.
    * Back-pressure: ``submit()`` blocks when the queue hits
      ``max_pending`` so callers cannot outrun the writer without bound.
      For the per-tile large-blocks path a few entries are plenty; for
      per-slab vvvv writes callers should add their own slab-alive
      semaphore on top (see ``_compute_vvvv_block_df``) to also bound
      peak host memory, since a single vvvv slab can be tens of GB.
    * Error propagation: the first exception raised by a submitted job is
      latched and re-raised from the next ``submit()``, ``drain()`` or
      ``close()`` call.  After an error the worker drops any remaining
      queued jobs without running them, so subsequent submits return
      quickly (they hit the latched error).
    * Context manager: ``with _AsyncHDF5Writer(...) as w: ...`` drains
      on normal exit and always joins the worker thread on exit.

    Typical usage
    -------------
    >>> with _AsyncHDF5Writer(max_pending=4) as writer:
    ...     for tile in tiles:
    ...         writer.submit(dataset.__setitem__, tile.slice, tile.data)
    ...     # on normal __exit__: drain() then close()
    """

    _SENTINEL = object()

    def __init__(self, max_pending=4, name="hdf5-writer"):
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}")
        self._name = name
        self._q = _queue.Queue(maxsize=max_pending)
        self._err = None
        self._err_lock = threading.Lock()
        self._closed = False
        # --- instrumentation state (guarded by _stats_lock) -----------------
        # These let callers answer diagnostic questions like
        #   "is the writer thread saturated?"  (busy_time / active_time)
        #   "is HDF5 the bandwidth bottleneck?" (bytes_written / busy_time)
        #   "how much did the pipeline actually see?" (bytes_written / active_time)
        # None of them affect behaviour — they are pure counters updated
        # in the worker hot-path under a single lock.
        self._stats_lock = threading.Lock()
        self._n_jobs = 0
        self._busy_time_s = 0.0
        self._bytes_written = 0
        self._first_job_start = None  # perf_counter stamp of first fn() start
        self._last_job_end = None     # perf_counter stamp of most recent fn() end
        self._thread = threading.Thread(
            target=self._worker, name=name, daemon=True,
        )
        self._thread.start()

    # -- public API --------------------------------------------------------

    def submit(self, fn, *args, **kwargs):
        """Enqueue a write job.  Blocks if too many are already pending.

        Special ``_bytes`` kwarg: if present, it is popped from ``kwargs``
        before dispatching and used purely as a size hint for the writer's
        throughput counters — it never reaches ``fn``.  Callers that want
        effective-bandwidth stats (ovvv/vovv tiles, vvvv slabs) should
        pass ``_bytes=arr.nbytes`` at submit time; callers that don't
        care can omit it and the bandwidth fields in ``stats()`` will
        simply be zero.
        """
        # Re-raise any latched worker error eagerly so callers see the
        # failure at the earliest opportunity (and don't block forever on
        # a full queue whose writer has died).
        self._check_error()
        if self._closed:
            raise RuntimeError("_AsyncHDF5Writer is closed")
        bytes_hint = kwargs.pop("_bytes", None)
        self._q.put((fn, args, kwargs, bytes_hint))

    def drain(self):
        """Block until all currently queued writes have completed."""
        self._q.join()
        self._check_error()

    def close(self):
        """Signal the worker to stop, join it, and re-raise any error."""
        if self._closed:
            self._check_error()
            return
        self._closed = True
        self._q.put(self._SENTINEL)
        self._thread.join()
        self._check_error()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On clean exit, drain pending writes before closing.  If the
        # caller is already propagating an exception, we still close the
        # writer (to join the thread) but swallow any follow-up writer
        # error so we do not mask the original traceback.
        try:
            if exc_type is None:
                self.drain()
        finally:
            try:
                self.close()
            except Exception:
                if exc_type is None:
                    raise

    # -- internals ---------------------------------------------------------

    def _check_error(self):
        with self._err_lock:
            err = self._err
        if err is not None:
            raise err

    def _latch_error(self, exc):
        with self._err_lock:
            if self._err is None:
                self._err = exc

    def _worker(self):
        errored = False
        while True:
            item = self._q.get()
            try:
                if item is self._SENTINEL:
                    return
                if errored:
                    # Drop remaining work after the first failure so the
                    # main thread's submit() calls do not block on a full
                    # queue.  The latched error is re-raised to callers.
                    continue
                fn, args, kwargs, bytes_hint = item
                t0 = time.perf_counter()
                try:
                    fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    self._latch_error(exc)
                    errored = True
                t1 = time.perf_counter()
                # Update stats even on errored jobs: we still want the
                # saturation/bandwidth snapshot to reflect actual wall
                # time spent in fn() up to the point of failure.
                with self._stats_lock:
                    self._n_jobs += 1
                    self._busy_time_s += (t1 - t0)
                    if bytes_hint:
                        self._bytes_written += int(bytes_hint)
                    if self._first_job_start is None:
                        self._first_job_start = t0
                    self._last_job_end = t1
            finally:
                self._q.task_done()

    # -- stats -------------------------------------------------------------

    def stats(self):
        """Return a snapshot of writer throughput counters.

        Keys
        ----
        name : str
            Writer's thread name (passed at construction).
        n_jobs : int
            Completed jobs (including errored ones).
        busy_time_s : float
            Cumulative wall time spent inside ``fn(*args, **kwargs)``.
        active_time_s : float
            Wall time from the FIRST job's start to the LAST job's end.
            This is the writer's effective "on-duty" window.
        saturation : float
            ``busy_time_s / active_time_s`` (0.0 if no jobs).  A value
            near 1.0 means the writer was constantly busy → HDF5 calls
            are the pipeline bottleneck.  A value near 0.0 means the
            writer mostly sat idle waiting for submits → the pipeline
            producer side is the bottleneck, not HDF5.
        bytes_written : int
            Sum of ``_bytes`` hints passed to ``submit()`` (0 if callers
            did not opt in).
        busy_throughput_GBps : float
            ``bytes_written / busy_time_s / 1e9`` — the raw HDF5-API
            write bandwidth while the writer was actually writing.
            Compare against ``dd`` / ``fio`` on the target disk to see
            whether HDF5 overhead or the disk itself is the cap.
        effective_throughput_GBps : float
            ``bytes_written / active_time_s / 1e9`` — the bandwidth the
            pipeline actually saw.  Equals ``busy_throughput_GBps`` at
            full saturation; lower when the writer sat idle.
        """
        with self._stats_lock:
            t_busy = self._busy_time_s
            t_active = (
                (self._last_job_end - self._first_job_start)
                if self._first_job_start is not None else 0.0
            )
            n = self._n_jobs
            bytes_w = self._bytes_written
        saturation = (t_busy / t_active) if t_active > 0 else 0.0
        busy_bw = (bytes_w / t_busy / 1e9) if (t_busy > 0 and bytes_w) else 0.0
        eff_bw  = (bytes_w / t_active / 1e9) if (t_active > 0 and bytes_w) else 0.0
        return {
            "name": self._name,
            "n_jobs": n,
            "busy_time_s": t_busy,
            "active_time_s": t_active,
            "saturation": saturation,
            "bytes_written": bytes_w,
            "busy_throughput_GBps": busy_bw,
            "effective_throughput_GBps": eff_bw,
        }

    def log_summary(self, log_fn):
        """Pretty-print the current :meth:`stats` via the given ``log_fn``.

        ``log_fn`` is typically ``logger.info`` (or ``print`` in tests).
        Safe to call multiple times — each call emits a fresh snapshot.
        Emits nothing if no jobs have run yet.
        """
        s = self.stats()
        if s["n_jobs"] == 0:
            return
        log_fn(
            "[%s] writer stats: n_jobs=%d  busy=%.2fs  active=%.2fs  "
            "saturation=%.1f%%  bytes=%.2f GB  "
            "busy_BW=%.2f GB/s  effective_BW=%.2f GB/s",
            s["name"], s["n_jobs"], s["busy_time_s"], s["active_time_s"],
            100.0 * s["saturation"], s["bytes_written"] / 1e9,
            s["busy_throughput_GBps"], s["effective_throughput_GBps"],
        )


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


def partition_round_robin(tile_specs, devices, device_key=None):
    """Partition ``tile_specs`` across ``devices`` round-robin.

    Returns a dict ``{device: [specs assigned to this device]}`` in the same
    assignment order ``_round_robin_pipeline`` will use.  Useful for callers
    that need to set up per-device state (e.g. HDF5 prefetch chains) before
    handing off to the pipeline.

    Parameters
    ----------
    device_key : callable(spec) -> hashable, optional
        When provided, all tiles sharing the same key go to the same device
        (first-seen-key round-robin), exactly matching the pipeline's
        internal assignment.  Default: tile-index round-robin.
    """
    n_devices = len(devices)
    out = {d: [] for d in devices}
    if device_key is None:
        for tile_id, spec in enumerate(tile_specs):
            out[devices[tile_id % n_devices]].append(spec)
    else:
        key_to_device = {}
        for spec in tile_specs:
            k = device_key(spec)
            if k not in key_to_device:
                key_to_device[k] = devices[len(key_to_device) % n_devices]
            out[key_to_device[k]].append(spec)
    return out


def _round_robin_pipeline(tile_specs, issue_tile, consume_tile, devices=None,
                          device_key=None, gpu_slots=None, host_slots=None,
                          return_stats=False):
    """Per-device parallel issue + thread-pool consume.

    Each device gets its own *issue thread* that drains its assigned tile
    sub-list.  Both threads call ``issue_tile`` concurrently — so even if
    ``issue_tile`` blocks synchronously on GPU work for the duration of
    the tile (which is empirically what ``compute_2b_tile`` does — see the
    ``[VVVV-issue dN] xtc_2b_tile RETURN ... blocking-call`` markers in
    ``jax_xtc_ccsd._contract_vvvv_t2``), both GPUs can be busy at the
    same time instead of alternating.

    Tile assignment
    ---------------
    * Default (``device_key=None``): tile-id round-robin.  Tile ``k``
      goes to ``devices[k % n_devices]``.
    * ``device_key=callable(spec) -> hashable``: tiles sharing a key
      go to the same device, keys assigned first-seen round-robin.

    Parameters
    ----------
    issue_tile : callable(spec, device) -> jax.Array (future)
        **Must be thread-safe.**  Two threads will call it concurrently
        with different ``device`` arguments.  Closures that mutate
        shared state (e.g. an HDF5 prefetch chain) must either lock or,
        cleaner, use ``partition_round_robin`` to set up per-device
        state up front.
    consume_tile : callable(spec, device, handle, release_gpu_slot)
        Called from a separate consume thread pool.  See bottom for the
        ``release_gpu_slot`` contract.
    devices : sequence
        Local devices.  Default: ``_solver_local_devices()``.
    gpu_slots : int, optional
        Total in-flight tiles allowed across all devices.  Default
        ``2 * n_devices``.  Each ``issue_tile`` call acquires one slot;
        ``release_gpu_slot`` (called inside ``consume_tile``) releases
        it.  This bounds peak GPU memory residency.
    host_slots : int, optional
        Total in-flight tiles holding host-side buffers.  Default
        ``max(gpu_slots, 4 * n_devices)``.  Bounds peak host memory
        for tiles whose GPU work has finished but whose CPU
        accumulate / write hasn't completed yet.
    return_stats : bool, optional
        When True, return a stats dict:

        * ``n_tiles`` — total tiles processed.
        * ``wall_s`` — wall time from first issue dispatch to last
          consume completion.
        * ``host_wait_s`` — cumulative seconds *summed across issue
          threads* spent in ``host_sem.acquire()``.  Nonzero means
          host-side back-pressure (writer / host memory) is throttling.
        * ``gpu_wait_s`` — cumulative seconds in ``gpu_sem.acquire()``.
          Nonzero means the GPU-residency cap is binding.
        * ``issue_s`` — cumulative seconds spent inside
          ``issue_tile()``.  In the parallel-issue model this is the
          *sum across issue threads* — divide by ``n_devices`` to
          estimate per-device average.

    consume_tile contract
    ---------------------
    ``consume_tile(spec, device, handle, release_gpu_slot)`` MUST call
    ``release_gpu_slot()`` as soon as the tile is drained from the GPU
    (typically right after ``np.asarray(handle)``), freeing one issue
    thread to dispatch its next tile while CPU work continues.
    ``release_gpu_slot`` is idempotent and guaranteed to be called on
    exit even if consume raises.  Consume callbacks must be thread-safe
    — protect any shared mutable state (HDF5 datasets, host
    accumulators, etc.) with a ``threading.Lock`` in the closure.
    """
    devices = devices or _solver_local_devices()
    n_devices = len(devices)
    if gpu_slots is None:
        gpu_slots = 2 * n_devices
    if host_slots is None:
        host_slots = max(gpu_slots, 4 * n_devices)
    if host_slots < gpu_slots:
        host_slots = gpu_slots

    # Pre-partition tile_specs across devices so each issue thread has a
    # clean private work-list.  This is the same partition exposed
    # publicly via ``partition_round_robin``.
    tiles_by_device = partition_round_robin(tile_specs, devices, device_key)

    gpu_sem  = threading.Semaphore(gpu_slots)
    host_sem = threading.Semaphore(host_slots)

    def _run_consume(spec, device, handle):
        try:
            with _gpu_slot_ctx(gpu_sem) as release_gpu_slot:
                consume_tile(spec, device, handle, release_gpu_slot)
        finally:
            host_sem.release()

    stats = {
        "n_tiles": 0,
        "wall_s": 0.0,
        "host_wait_s": 0.0,
        "gpu_wait_s": 0.0,
        "issue_s": 0.0,
    }
    stats_lock = threading.Lock()

    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=host_slots) as consume_pool, \
         concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as issue_pool:

        consume_futures = []
        consume_futures_lock = threading.Lock()

        def _issue_worker(device, my_specs):
            """One per device.  Drains ``my_specs`` calling ``issue_tile``
            and submitting the resulting handle to the consume pool."""
            local = {"host_wait_s": 0.0, "gpu_wait_s": 0.0,
                     "issue_s": 0.0, "n_tiles": 0}
            for spec in my_specs:
                t_a = time.perf_counter()
                host_sem.acquire()
                t_b = time.perf_counter()
                gpu_sem.acquire()
                t_c = time.perf_counter()
                handle = issue_tile(spec, device)
                t_d = time.perf_counter()
                local["host_wait_s"] += (t_b - t_a)
                local["gpu_wait_s"]  += (t_c - t_b)
                local["issue_s"]     += (t_d - t_c)
                local["n_tiles"]     += 1
                fut = consume_pool.submit(_run_consume, spec, device, handle)
                with consume_futures_lock:
                    consume_futures.append(fut)
            with stats_lock:
                for k, v in local.items():
                    stats[k] += v

        # Launch one issue thread per device.  Skip devices with no work
        # (e.g. n_tiles < n_devices).
        issue_futures = []
        for device in devices:
            specs = tiles_by_device.get(device, [])
            if specs:
                issue_futures.append(issue_pool.submit(_issue_worker, device, specs))

        # Wait for all issue threads to finish dispatching.  This will
        # re-raise the first issue-thread exception (e.g. an XLA error
        # during compute_2b_tile) on the main thread.
        for f in issue_futures:
            f.result()

        # Then wait for all consume futures.  Re-raises the first
        # consume-thread exception.
        for f in consume_futures:
            f.result()

    stats["wall_s"] = time.perf_counter() - wall_t0
    return stats if return_stats else None
