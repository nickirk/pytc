"""Opt-in per-term wall timers for the TC/ΔU tile assembly path.

Motivation: the eris build's tile assembly is the dominant phase cost at
production decks, but the phase is one opaque number.  Deciding between
reshaping the panels and reformulating the assembly requires the split
into the individual contraction legs (K1/K2, K3, ΔU, final sum) over ALL
tiles, not just a first-tile sample.

Discipline (mirrors the issue-stage-stats mechanism in ``pytc.xtc``):

- Strictly opt-in via the ``PYTC_TILE_TIMERS`` environment variable.
  When unset (the production default), every entry point costs a single
  module-level flag read and the assembly path is bit-identical to an
  instrumented build of the code.
- When enabled, ``Timer.sync()`` inserts ``jax.block_until_ready`` so the
  asynchronous JAX dispatch cannot smear one term's compute into the
  next term's measurement.  These barriers exist ONLY in the enabled
  path — enabling the timers is a measurement run, never production.
- The accumulator is process-wide (not ``threading.local()``) so the
  issue worker threads spawned by the tiled pipelines all land in the
  same dict; the lock window is one dict lookup + add.

Reporting: at process exit (atexit, registered on first enabled
accumulation) a summary is logged at INFO and, when
``PYTC_TILE_TIMERS_JSON`` is set, written to that path as JSON.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time

import jax

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("PYTC_TILE_TIMERS", "").lower() in ("1", "true", "yes")
_JSON_PATH = os.environ.get("PYTC_TILE_TIMERS_JSON") or None

_STATE = {
    "terms": {},          # name -> [total_s, count]
    "counters": {},       # name -> int
    "lock": threading.Lock(),
    "atexit_registered": False,
}


def incr(name: str, n: int = 1):
    """Increment a named counter (e.g. cache hits/misses).  Cheap always;
    reported alongside the timed terms in the exit dump."""
    with _STATE["lock"]:
        _STATE["counters"][name] = _STATE["counters"].get(name, 0) + n
        if _ENABLED and not _STATE["atexit_registered"]:
            atexit.register(_dump_at_exit)
            _STATE["atexit_registered"] = True


def enabled() -> bool:
    return _ENABLED


class _TermTimer(contextlib.AbstractContextManager):
    """Times one assembly term; syncs the produced value only when enabled."""

    __slots__ = ("_name", "_t0")

    def __init__(self, name):
        self._name = name
        self._t0 = None

    def __enter__(self):
        if _ENABLED:
            self._t0 = time.perf_counter()
        return self

    def sync(self, value):
        """Block on ``value`` (JAX async) so the term's compute is inside
        the measured window.  No-op when disabled, and returns ``value``
        either way so call sites read naturally."""
        if _ENABLED:
            jax.block_until_ready(value)
        return value

    def __exit__(self, exc_type, exc, tb):
        if _ENABLED and exc_type is None:
            dt = time.perf_counter() - self._t0
            with _STATE["lock"]:
                slot = _STATE["terms"].setdefault(self._name, [0.0, 0])
                slot[0] += dt
                slot[1] += 1
                if not _STATE["atexit_registered"]:
                    atexit.register(_dump_at_exit)
                    _STATE["atexit_registered"] = True
        return False


def term(name: str) -> _TermTimer:
    """Time an assembly term (no measurable cost when disabled)."""
    return _TermTimer(name)


def report() -> dict:
    """Snapshot of accumulated timers as {name: {total_s, count, mean_s}}."""
    with _STATE["lock"]:
        return {
            name: {"total_s": total, "count": count,
                   "mean_s": total / count if count else 0.0}
            for name, (total, count) in sorted(_STATE["terms"].items())
        }


def counters() -> dict:
    """Snapshot of plain counters (e.g. cache hits/misses)."""
    with _STATE["lock"]:
        return dict(_STATE["counters"])


def _dump_at_exit():
    rep = report()
    cts = counters()
    if not rep and not cts:
        return
    grand = sum(v["total_s"] for v in rep.values())
    for name, v in rep.items():
        logger.info(
            "tile_timers: %-24s %10.3fs total  x%-6d mean %.4fs  (%5.1f%%)",
            name, v["total_s"], v["count"], v["mean_s"],
            100.0 * v["total_s"] / grand if grand else 0.0,
        )
    if rep:
        logger.info("tile_timers: %-24s %10.3fs total", "ALL_TRACKED", grand)
    for name, n in sorted(cts.items()):
        logger.info("tile_timers: counter %-20s %d", name, n)
    if _JSON_PATH:
        payload = {"terms": rep, "counters": cts, "total_tracked_s": grand}
        try:
            with open(_JSON_PATH, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("tile_timers: wrote %s", _JSON_PATH)
        except OSError as exc:
            logger.warning("tile_timers: could not write %s: %s",
                           _JSON_PATH, exc)


def _reset_for_tests():
    with _STATE["lock"]:
        _STATE["terms"].clear()
        _STATE["counters"].clear()
