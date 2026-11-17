##############################################################################
#
# Copyright (c) 2026 Zope Foundation and Contributors.
#
# This software is subject to the provisions of the Zope Public License,
# Version 2.1 (ZPL).  A copy of the ZPL should accompany this distribution.
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
# WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
# FOR A PARTICULAR PURPOSE
#
##############################################################################
"""Layer 2 -- per-execution resource quotas.

Each script run gets a fresh :class:`ResourceQuota` covering three
independent dimensions:

* ``wall_clock``   -- elapsed real time in seconds (watchdog enforced);
* ``memory``       -- peak Python allocations in bytes (``tracemalloc``);
* ``concurrency``  -- number of runs allowed at the same time (semaphore).

Exceeding *any* dimension terminates the current script with
:class:`QuotaExceeded` whose ``dimension`` names the trigger.  Once a quota
has tripped the quota enters a ``tripped`` state and every remaining guarded
call stops immediately instead of continuing to make progress.
"""

from __future__ import annotations

import contextvars
import threading
import time
import tracemalloc
import typing

from RestrictedPython.runtime.errors import QuotaExceeded


DIMENSION_WALL_CLOCK = "wall_clock"
DIMENSION_MEMORY = "memory"
DIMENSION_CONCURRENCY = "concurrency"

# Stable quota rule numbers surfaced in diagnostics.
_RULE_WALL_CLOCK = "RP-QM-0001"
_RULE_MEMORY = "RP-QM-0002"
_RULE_CONCURRENCY = "RP-QM-0003"

_RULE_IDS = {
    DIMENSION_WALL_CLOCK: _RULE_WALL_CLOCK,
    DIMENSION_MEMORY: _RULE_MEMORY,
    DIMENSION_CONCURRENCY: _RULE_CONCURRENCY,
}


class QuotaConfig(typing.NamedTuple):
    """Limits applied to one execution; ``None`` means unlimited."""

    wall_clock_seconds: float | None = 5.0
    memory_bytes: int | None = 16 * 1024 * 1024
    max_concurrency: int | None = 4
    # How long to wait for a concurrency slot before treating the wait as a
    # quota breach rather than blocking the host indefinitely.
    acquire_timeout_seconds: float = 30.0

    def validated(self) -> "QuotaConfig":
        if self.wall_clock_seconds is not None \
                and self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive or None")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive or None")
        if self.max_concurrency is not None and self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive or None")
        return self


# The quota for the currently executing run, or None outside an engine run.
_current: contextvars.ContextVar["ResourceQuota | None"] = \
    contextvars.ContextVar("restricted_runtime_quota", default=None)


@typing.overload
def active_quota(required: typing.Literal[True]) -> "ResourceQuota":
    ...  # pragma: no cover


@typing.overload
def active_quota(required: bool = False) -> "ResourceQuota | None":
    ...  # pragma: no cover


def active_quota(required: bool = False) -> "ResourceQuota | None":
    quota = _current.get()
    if required and quota is None:
        # Guarded builtins are only callable inside an engine-run context;
        # reaching this outside means a wiring error, not a user script.
        raise RuntimeError("guarded callable used outside a runtime run")
    return quota


class ResourceQuota:
    """Tracks and enforces the three quota dimensions for one run."""

    def __init__(self, config: QuotaConfig,
                 clock: typing.Callable[[], float] = time.monotonic) -> None:
        self.config = config.validated()
        self._clock = clock
        self._tripped: str | None = None
        self.tripped_dimension: str | None = None
        self.tripped_limit: int | float | None = None
        self._tripped_lock = threading.Lock()
        self._element_allowance = self.config.memory_bytes
        self._element_charged = 0
        # A rough per-element cost used for pre-allocation accounting so a
        # giant list/range trips the quota before it is built.
        self._element_cost_bytes = 8

        self._concurrency_semaphore = _global_concurrency_semaphore(
            self.config.max_concurrency)
        self._acquired_slot = False
        self._tracemalloc_started_here = False
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._target_ident: int | None = None
        self._start_time: float | None = None
        self.peak_memory_bytes = 0

    # -- lifecycle

    def __enter__(self) -> "ResourceQuota":
        # Dimension: concurrency.  Refuse rather than queue forever.
        if self._concurrency_semaphore is not None:
            if not self._concurrency_semaphore.acquire(
                    timeout=self.config.acquire_timeout_seconds):
                concurrency_limit = self.config.max_concurrency
                assert concurrency_limit is not None
                self._trip(DIMENSION_CONCURRENCY, concurrency_limit)
                raise QuotaExceeded(
                    DIMENSION_CONCURRENCY,
                    concurrency_limit,
                    rule_id=_RULE_IDS[DIMENSION_CONCURRENCY])
            self._acquired_slot = True

        self._start_time = self._clock()
        self._token = _current.set(self)

        # Dimension: memory.  Share a pre-existing tracemalloc session if the
        # host already started one; otherwise own the session lifetime.
        if self.config.memory_bytes is not None:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._tracemalloc_started_here = True
            self._baseline_memory = \
                tracemalloc.get_traced_memory()[0]

        # Dimension: wall clock.  A daemon watchdog can interrupt a pure-Python
        # infinite loop via ctypes on CPython; I/O syscalls are already
        # unavailable through the whitelist, so no blocking call can hide.
        if self.config.wall_clock_seconds is not None:
            import _thread
            self._target_ident = threading.get_ident()
            self._thread_raise = _thread.interrupt_main
            self._watchdog = threading.Thread(
                target=self._watchdog_loop,
                name="restricted-runtime-watchdog",
                daemon=True)
            self._watchdog.start()
        return self

    def __exit__(self, exc_type: typing.Any, exc: typing.Any,
                 tb: typing.Any) -> None:
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
            self._watchdog = None
        if self.config.memory_bytes is not None:
            current, peak = tracemalloc.get_traced_memory()
            self.peak_memory_bytes = max(
                self.peak_memory_bytes,
                peak - getattr(self, "_baseline_memory", 0))
            if self._tracemalloc_started_here:
                tracemalloc.stop()
        _current.reset(self._token)
        if self._acquired_slot and self._concurrency_semaphore is not None:
            self._concurrency_semaphore.release()
            self._acquired_slot = False

    # -- watchdog

    def _watchdog_loop(self) -> None:
        assert self.config.wall_clock_seconds is not None
        deadline = (self._start_time or self._clock()) \
            + self.config.wall_clock_seconds
        while not self._watchdog_stop.wait(timeout=0.02):
            if self._clock() >= deadline:
                self._trip(DIMENSION_WALL_CLOCK,
                           self.config.wall_clock_seconds)
                # Inject KeyboardInterrupt into the run thread so a tight
                # pure-Python loop returns control to the engine boundary,
                # where it is translated into QuotaExceeded.
                try:
                    self._interrupt_run_thread()
                except Exception:  # pragma: no cover - best effort
                    pass
                return

    def _interrupt_run_thread(self) -> None:
        if self._target_ident == threading.get_ident():
            return  # pragma: no cover - watchdog never runs the script
        import ctypes
        if self._target_ident is None:  # pragma: no cover
            return
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self._target_ident),
            ctypes.py_object(KeyboardInterrupt))

    # -- accounting hooks called by guarded builtins

    def check_alive(self) -> None:
        """Raise immediately once any dimension has tripped."""
        if self._tripped is not None:
            limit = self._limit_for(self._tripped)
            assert limit is not None
            raise QuotaExceeded(
                self._tripped, limit,
                rule_id=_RULE_IDS[self._tripped])
        self._sample_memory()

    def charge_allocation_hint(self) -> None:
        self.check_alive()

    def charge_elements(self, count: int) -> None:
        """Account for an allocation of ``count`` elements up front."""
        self.check_alive()
        if count <= 0 or self._element_allowance is None:
            return
        self._element_charged += count * self._element_cost_bytes
        if self._element_charged >= self._element_allowance:
            memory_limit = self.config.memory_bytes
            assert memory_limit is not None
            self._trip(DIMENSION_MEMORY, memory_limit)
            raise QuotaExceeded(
                DIMENSION_MEMORY, memory_limit,
                rule_id=_RULE_MEMORY)

    def _sample_memory(self) -> None:
        if self.config.memory_bytes is None or not tracemalloc.is_tracing():
            return
        current, peak = tracemalloc.get_traced_memory()
        baseline = getattr(self, "_baseline_memory", 0)
        used = current - baseline
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak - baseline)
        if used >= self.config.memory_bytes:
            self._trip(DIMENSION_MEMORY, self.config.memory_bytes)
            raise QuotaExceeded(
                DIMENSION_MEMORY, self.config.memory_bytes,
                rule_id=_RULE_MEMORY)

    # -- shared state

    def _limit_for(self, dimension: str) -> int | float | None:
        return {
            DIMENSION_WALL_CLOCK: self.config.wall_clock_seconds,
            DIMENSION_MEMORY: self.config.memory_bytes,
            DIMENSION_CONCURRENCY: self.config.max_concurrency,
        }.get(dimension)

    def _trip(self, dimension: str, limit: int | float | None) -> None:
        with self._tripped_lock:
            if self._tripped is None:
                self._tripped = dimension
                self.tripped_dimension = dimension
                self.tripped_limit = limit

    @property
    def tripped(self) -> bool:
        return self._tripped is not None


# A process-wide semaphore enforces the concurrency ceiling across runs.
_global_semaphore: threading.Semaphore | None = None
_global_semaphore_value: int | None = None
_global_semaphore_lock = threading.Lock()


def _global_concurrency_semaphore(
        value: int | None) -> threading.Semaphore | None:
    global _global_semaphore, _global_semaphore_value
    if value is None:
        return None
    with _global_semaphore_lock:
        if _global_semaphore is None or _global_semaphore_value != value:
            _global_semaphore = threading.Semaphore(value)
            _global_semaphore_value = value
        return _global_semaphore


def reset_global_concurrency() -> None:
    """Test helper: drop the process-wide concurrency semaphore."""
    global _global_semaphore, _global_semaphore_value
    with _global_semaphore_lock:
        _global_semaphore = None
        _global_semaphore_value = None
