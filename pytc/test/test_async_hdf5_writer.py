"""Unit tests for ``pytc.utils.gpu_pipeline._AsyncHDF5Writer``.

These tests exercise the writer purely against in-memory callables, with
no real HDF5 file I/O — the contract we care about is:

* FIFO ordering of submitted jobs.
* ``drain()`` blocks until the queue is empty.
* Errors raised inside the worker are latched and re-raised on the next
  ``submit()`` / ``drain()`` / ``close()``.
* After the first error the worker drops any remaining queued work so
  the main thread cannot deadlock on a full queue.
* Back-pressure: with ``max_pending=1`` and a slow worker, ``submit()``
  blocks once the queue is full.
* The context-manager protocol drains on clean exit and always joins
  the worker thread.
"""

import threading
import time
import unittest

from pytc.utils.gpu_pipeline import _AsyncHDF5Writer


class TestAsyncHDF5Writer(unittest.TestCase):
    def test_preserves_submit_order(self):
        results = []
        with _AsyncHDF5Writer(max_pending=8) as w:
            for i in range(16):
                w.submit(results.append, i)
            w.drain()
        self.assertEqual(results, list(range(16)))

    def test_drain_blocks_until_empty(self):
        lock = threading.Lock()
        counter = [0]

        def slow_job():
            time.sleep(0.005)
            with lock:
                counter[0] += 1

        with _AsyncHDF5Writer(max_pending=8) as w:
            for _ in range(10):
                w.submit(slow_job)
            w.drain()
            # After drain() returns, every submitted job MUST have run.
            with lock:
                self.assertEqual(counter[0], 10)

    def test_error_latched_and_reraised_on_next_submit(self):
        class Boom(RuntimeError):
            pass

        def explode():
            raise Boom("kapow")

        w = _AsyncHDF5Writer(max_pending=4)
        try:
            w.submit(explode)
            # Give the worker a moment to latch the error.
            for _ in range(200):
                with w._err_lock:          # pylint: disable=protected-access
                    if w._err is not None:  # pylint: disable=protected-access
                        break
                time.sleep(0.005)
            else:
                self.fail("worker did not latch error within 1s")

            with self.assertRaises(Boom):
                w.submit(lambda: None)
            with self.assertRaises(Boom):
                w.drain()
        finally:
            # close() should also re-raise the latched error once, then
            # become idempotent and the thread must be joined.
            with self.assertRaises(Boom):
                w.close()
            self.assertFalse(w._thread.is_alive())  # pylint: disable=protected-access

    def test_error_does_not_block_subsequent_submits(self):
        """After the first failure the worker drops remaining jobs so
        submit() does not block on a full queue (regression guard for
        the 'error + full queue + dead worker' deadlock)."""

        started_second = threading.Event()
        release_first = threading.Event()

        def first():
            release_first.wait(timeout=2.0)
            raise RuntimeError("first failed")

        def second():
            started_second.set()

        w = _AsyncHDF5Writer(max_pending=2)
        try:
            w.submit(first)       # occupies the worker
            w.submit(second)      # sits in the queue
            release_first.set()   # let first() raise → latches error
            # Give the worker a moment to drain `second` without running it.
            for _ in range(200):
                with w._err_lock:          # pylint: disable=protected-access
                    if w._err is not None:  # pylint: disable=protected-access
                        break
                time.sleep(0.005)
            else:
                self.fail("error was not latched in time")
            self.assertFalse(started_second.is_set(),
                             "second job must be dropped after first failure")
        finally:
            with self.assertRaises(RuntimeError):
                w.close()

    def test_backpressure_blocks_when_queue_full(self):
        """With max_pending=1 and a blocked worker, the 2nd submit must
        wait for the worker to free a slot."""
        gate = threading.Event()

        def blocked():
            gate.wait(timeout=2.0)

        with _AsyncHDF5Writer(max_pending=1) as w:
            # First submit occupies the worker and pops the queue to 0.
            w.submit(blocked)
            # Second submit occupies the 1-deep queue.
            w.submit(lambda: None)

            submitted = threading.Event()

            def try_third():
                w.submit(lambda: None)
                submitted.set()

            t = threading.Thread(target=try_third)
            t.start()
            # Without a free slot the third submit must block.
            self.assertFalse(
                submitted.wait(timeout=0.2),
                "submit should block when queue is full")
            gate.set()  # let blocked() return → worker frees a slot
            self.assertTrue(
                submitted.wait(timeout=2.0),
                "submit should unblock once the worker drains a slot")
            t.join()
            w.drain()

    def test_context_manager_drains_on_clean_exit(self):
        results = []
        with _AsyncHDF5Writer(max_pending=4) as w:
            for i in range(5):
                w.submit(results.append, i)
            # Intentionally do NOT call drain() — __exit__ should do it.
        self.assertEqual(results, [0, 1, 2, 3, 4])
        # Worker thread should be joined.
        self.assertFalse(w._thread.is_alive())  # pylint: disable=protected-access

    def test_context_manager_joins_thread_on_exception(self):
        """If the with-block body raises, __exit__ must still join the
        worker thread (no leaked daemon) and must not swallow the
        original exception."""
        class UserError(ValueError):
            pass

        w = None
        with self.assertRaises(UserError):
            with _AsyncHDF5Writer(max_pending=2) as w:
                w.submit(lambda: None)
                raise UserError("bang")
        self.assertIsNotNone(w)
        self.assertFalse(w._thread.is_alive())  # pylint: disable=protected-access

    def test_submit_after_close_raises(self):
        w = _AsyncHDF5Writer(max_pending=2)
        w.close()
        with self.assertRaises(RuntimeError):
            w.submit(lambda: None)

    def test_invalid_max_pending_rejected(self):
        with self.assertRaises(ValueError):
            _AsyncHDF5Writer(max_pending=0)

    def test_stats_tracks_busy_time_bytes_and_saturation(self):
        """``stats()`` must report reasonable busy/active/bytes totals.

        Uses a sleep-based fake job so busy_time is large enough to
        compare against real wall time without flakiness.
        """
        def fake_write(payload):
            # Simulate ~30ms of HDF5 work; we don't actually touch
            # a file — the writer doesn't care what fn() does.
            time.sleep(0.03)

        with _AsyncHDF5Writer(max_pending=4, name="test-writer") as w:
            for i in range(5):
                w.submit(fake_write, f"chunk-{i}", _bytes=1_000_000)
            w.drain()
            s = w.stats()

        self.assertEqual(s["n_jobs"], 5)
        self.assertEqual(s["bytes_written"], 5 * 1_000_000)
        # Each job slept ~30ms, so busy_time should be near 0.15s.
        self.assertGreater(s["busy_time_s"], 0.10)
        self.assertLess(s["busy_time_s"], 0.50)
        # Single-threaded serial execution → saturation near 1.0.
        self.assertGreater(s["saturation"], 0.80)
        self.assertLessEqual(s["saturation"], 1.0 + 1e-9)
        # Bandwidth numbers should be positive and finite.
        self.assertGreater(s["busy_throughput_GBps"], 0.0)
        self.assertGreater(s["effective_throughput_GBps"], 0.0)
        # And busy_BW should be >= effective_BW by construction.
        self.assertGreaterEqual(
            s["busy_throughput_GBps"] + 1e-9,
            s["effective_throughput_GBps"],
        )

    def test_stats_bytes_zero_when_hint_omitted(self):
        with _AsyncHDF5Writer(max_pending=4) as w:
            for _ in range(3):
                w.submit(lambda: None)
            w.drain()
            s = w.stats()
        self.assertEqual(s["n_jobs"], 3)
        self.assertEqual(s["bytes_written"], 0)
        self.assertEqual(s["busy_throughput_GBps"], 0.0)
        self.assertEqual(s["effective_throughput_GBps"], 0.0)

    def test_log_summary_noop_before_any_job(self):
        """log_summary must emit nothing if n_jobs == 0 (don't spam
        callers with meaningless zeros at shutdown of an unused writer)."""
        calls = []
        w = _AsyncHDF5Writer(max_pending=2)
        try:
            w.log_summary(lambda *a, **k: calls.append(a))
            self.assertEqual(calls, [])
        finally:
            w.close()

    def test_bytes_kwarg_is_stripped_before_calling_fn(self):
        """The ``_bytes`` hint is consumed by the writer and must not be
        forwarded to the user's callable."""
        received = {}

        def record(**kwargs):
            received.update(kwargs)

        with _AsyncHDF5Writer(max_pending=2) as w:
            w.submit(record, foo=1, _bytes=42)
            w.drain()

        self.assertEqual(received, {"foo": 1})
        # The writer saw the size hint though:
        with _AsyncHDF5Writer(max_pending=2) as w:
            w.submit(record, _bytes=99)
            w.drain()
            self.assertEqual(w.stats()["bytes_written"], 99)


if __name__ == "__main__":
    unittest.main()
