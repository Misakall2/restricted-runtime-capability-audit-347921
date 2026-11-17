"""Tests for layer 2 -- per-execution resource quotas.

Each dimension (wall clock, memory, concurrency) is exercised independently so
a failure can be attributed to a specific quota rather than a generic error.
Quotas are disabled on the other dimensions where they are not under test.
"""

import threading
import time

import pytest

from RestrictedPython.runtime import DIMENSION_CONCURRENCY
from RestrictedPython.runtime import DIMENSION_MEMORY
from RestrictedPython.runtime import DIMENSION_WALL_CLOCK
from RestrictedPython.runtime import QuotaConfig
from RestrictedPython.runtime import Reason
from RestrictedPython.runtime import RestrictedRuntime
from RestrictedPython.runtime import reset_global_concurrency
from RestrictedPython.runtime.quota import ResourceQuota


def _no_limits(**overrides):
    base = dict(wall_clock_seconds=None, memory_bytes=None,
                max_concurrency=None)
    base.update(overrides)
    return QuotaConfig(**base)


# ---------------------------------------------------------------------------
# Wall clock
# ---------------------------------------------------------------------------

def test_wall_clock_quota_terminates_infinite_loop():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(wall_clock_seconds=0.4))
    start = time.monotonic()
    result = runtime.run("while True:\n    pass")
    elapsed = time.monotonic() - start
    assert not result.ok
    assert result.exit_code == 30
    assert result.reason == Reason.RESOURCE_LIMIT.value
    assert DIMENSION_WALL_CLOCK in result.diagnostics[0]
    # The watchdog must stop the script promptly, not run to suite timeout.
    assert elapsed < 3.0


def test_wall_clock_diagnostic_names_rule_number():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(wall_clock_seconds=0.3))
    result = runtime.run("while True:\n    pass")
    assert "RP-QM-0001" in result.diagnostics[0]


def test_fast_script_is_not_killed_by_wall_clock():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(wall_clock_seconds=5.0))
    result = runtime.run("value = sum(range(100))",
                         result_variable="value")
    assert result.ok and result.value == sum(range(100))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_memory_quota_trips_on_large_materialized_list():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(memory_bytes=128 * 1024))
    result = runtime.run("data = list(range(1_000_000))")
    assert not result.ok
    assert result.exit_code == 30
    assert result.reason == Reason.RESOURCE_LIMIT.value
    assert DIMENSION_MEMORY in result.diagnostics[0]
    assert "RP-QM-0002" in result.diagnostics[0]


def test_memory_quota_trips_on_growing_allocation():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(memory_bytes=128 * 1024))
    result = runtime.run(
        "chunks = []\n"
        "for i in range(100000):\n"
        "    chunks.append('x' * 200)")
    assert not result.ok
    assert result.reason == Reason.RESOURCE_LIMIT.value


def test_small_allocation_stays_within_memory_quota():
    runtime = RestrictedRuntime(
        quota_config=_no_limits(memory_bytes=8 * 1024 * 1024))
    result = runtime.run("value = len(list(range(100)))",
                         result_variable="value")
    assert result.ok and result.value == 100


def test_tripped_quota_blocks_further_guarded_calls():
    # Once any dimension trips, remaining guarded calls must stop rather than
    # keep attempting allocations.
    config = _no_limits(memory_bytes=64 * 1024)
    quota = ResourceQuota(config)
    from RestrictedPython.runtime.errors import QuotaExceeded
    with quota:
        with pytest.raises(QuotaExceeded):
            quota.charge_elements(10_000_000)
        assert quota.tripped
        # Subsequent checks raise immediately, named after the dimension.
        with pytest.raises(QuotaExceeded) as exc:
            quota.check_alive()
        assert exc.value.dimension == DIMENSION_MEMORY


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrency_quota_refuses_when_no_slot_available():
    reset_global_concurrency()
    config = QuotaConfig(
        wall_clock_seconds=10, memory_bytes=None,
        max_concurrency=1, acquire_timeout_seconds=0.5)
    runtime = RestrictedRuntime(quota_config=config)
    # Hold the single slot externally, simulating an in-flight execution.
    from RestrictedPython.runtime import quota as quota_module
    semaphore = quota_module._global_concurrency_semaphore(1)
    assert semaphore.acquire(timeout=0.1)
    try:
        result = runtime.run("value = 1", result_variable="value")
        assert not result.ok
        assert result.exit_code == 30
        assert result.reason == Reason.RESOURCE_LIMIT.value
        assert DIMENSION_CONCURRENCY in result.diagnostics[0]
        assert "RP-QM-0003" in result.diagnostics[0]
    finally:
        semaphore.release()
    reset_global_concurrency()


def test_concurrent_runs_are_serialized_under_ceiling():
    reset_global_concurrency()
    config = QuotaConfig(
        wall_clock_seconds=10, memory_bytes=None,
        max_concurrency=2, acquire_timeout_seconds=5)
    runtime = RestrictedRuntime(quota_config=config)
    outcomes = []
    lock = threading.Lock()

    def worker():
        result = runtime.run("value = sum(range(500))",
                             result_variable="value")
        with lock:
            outcomes.append(result.ok)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == 6
    assert all(outcomes)
    reset_global_concurrency()


def test_invalid_quota_configuration_is_rejected_upfront():
    with pytest.raises(ValueError):
        QuotaConfig(wall_clock_seconds=0).validated()
    with pytest.raises(ValueError):
        QuotaConfig(memory_bytes=-1).validated()
    with pytest.raises(ValueError):
        QuotaConfig(max_concurrency=0).validated()


def test_each_run_gets_an_independent_quota_instance():
    runtime = RestrictedRuntime(quota_config=_no_limits())
    first = runtime.run("a = list(range(10))")
    second = runtime.run("b = list(range(10))")
    assert first.ok and second.ok
    # A memory breach in one run must not poison the quota of the next run.
    breaching = RestrictedRuntime(
        quota_config=_no_limits(memory_bytes=64 * 1024))
    bad = breaching.run("data = list(range(1_000_000))")
    assert not bad.ok
    healthy = breaching.run("value = 1", result_variable="value")
    assert healthy.ok and healthy.value == 1
