"""Tests for layer 3 -- post-execution resource reclamation.

Cleanup is tested independently of policy/quota and is exercised on both the
success and failure paths.  Tests deliberately reserve handles and spawn
child processes, then assert that no residual object survives the run.
"""

import os
import sys

import pytest

from RestrictedPython.runtime import Reason
from RestrictedPython.runtime import ResourceCleanup
from RestrictedPython.runtime import RestrictedRuntime
from RestrictedPython.runtime.errors import CleanupFailure
from RestrictedPython.runtime.quota import QuotaConfig


def _runtime():
    # Resource factories are opt-in; the cleanup tests exercise the audited,
    # auto-reclaimed handles/directories/processes they expose.
    return RestrictedRuntime(
        quota_config=QuotaConfig(
            wall_clock_seconds=15, memory_bytes=None,
            max_concurrency=None),
        enable_resource_factories=True)


# ---------------------------------------------------------------------------
# Direct ResourceCleanup unit tests (layer in isolation)
# ---------------------------------------------------------------------------

def test_handle_is_closed_on_reclaim():
    cleanup = ResourceCleanup()
    handle = cleanup.temporary_file()
    assert not handle.closed
    cleanup.reclaim()
    assert handle.closed
    assert not cleanup.has_residuals()


def test_resource_factories_are_absent_by_default():
    # The default sandbox must expose no file-system/process capability,
    # even the audited, auto-reclaimed kind.
    default_runtime = RestrictedRuntime(quota_config=QuotaConfig(
        wall_clock_seconds=15, memory_bytes=None, max_concurrency=None))
    for factory in ("create_temp_file", "create_temp_dir",
                    "spawn_process"):
        result = default_runtime.run(f"{factory}()")
        assert not result.ok
        assert result.reason in (
            Reason.POLICY_DENIED.value, Reason.SCRIPT_ERROR.value)
    # Opting in makes exactly those factories available.
    enabled = RestrictedRuntime(
        quota_config=QuotaConfig(
            wall_clock_seconds=15, memory_bytes=None,
            max_concurrency=None),
        enable_resource_factories=True)
    assert enabled.run("f = create_temp_file()").ok


def test_temp_directory_is_removed_on_reclaim():
    cleanup = ResourceCleanup()
    path = cleanup.temporary_directory()
    assert os.path.isdir(path)
    cleanup.reclaim()
    assert not os.path.exists(path)


def test_temp_file_path_is_removed_on_reclaim():
    cleanup = ResourceCleanup()
    cleanup.temporary_file()
    # On Windows the wrapper's ``name`` may be the integer fd, so read the
    # path from the cleanup registry instead of relying on ``handle.name``.
    path = cleanup._temp_paths[-1]
    assert os.path.exists(path)
    cleanup.reclaim()
    assert not os.path.exists(path)


def test_subprocess_is_terminated_on_reclaim():
    cleanup = ResourceCleanup()
    process = cleanup.spawn(
        sys.executable, "-c", "import time; time.sleep(60)")
    assert process.poll() is None
    cleanup.reclaim()
    assert process.poll() is not None


def test_reclaim_runs_idempotently():
    cleanup = ResourceCleanup()
    handle = cleanup.temporary_file()
    cleanup.reclaim()
    cleanup.reclaim()  # must not raise or double-count
    assert handle.closed


def test_stubborn_handle_is_reported_as_residual():
    cleanup = ResourceCleanup()

    class BrokenHandle:
        closed = False

        def close(self):
            raise OSError("refused")

    cleanup.track_handle(BrokenHandle())
    with pytest.raises(CleanupFailure) as exc:
        cleanup.reclaim()
    assert exc.value.residuals == ("handle",)
    assert exc.value.diagnostic.startswith("stage=cleanup")
    # Only the residual *kind* is reported, never a host path.
    assert os.sep not in exc.value.diagnostic


def test_reclaim_continues_after_one_failure_and_reports_all_kinds():
    cleanup = ResourceCleanup()

    class BrokenHandle:
        closed = False

        def close(self):
            raise OSError("refused")

    cleanup.track_handle(BrokenHandle())
    good = cleanup.temporary_file()
    with pytest.raises(CleanupFailure) as exc:
        cleanup.reclaim()
    assert "handle" in exc.value.residuals
    # The good resource is still reclaimed even though one step failed.
    assert good.closed


# ---------------------------------------------------------------------------
# Engine integration: cleanup runs on BOTH paths
# ---------------------------------------------------------------------------

def test_cleanup_runs_on_success_path_with_no_residuals():
    runtime = _runtime()
    result = runtime.run(
        "f = create_temp_file()\n"
        "f.write('payload')\n"
        "d = create_temp_dir()\n"
        "value = 7",
        result_variable="value")
    assert result.ok
    assert result.value == 7
    assert result.exit_code == 0


def test_cleanup_runs_on_failure_path():
    runtime = _runtime()
    result = runtime.run(
        "f = create_temp_file()\n"
        "f.write('payload')\n"
        "raise RuntimeError('intentional failure')")
    # The script error is reported, but cleanup still ran (no cleanup error).
    assert not result.ok
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert result.exit_code == 10


def test_reserved_handle_is_reclaimed_after_failed_script():
    runtime = _runtime()
    result = runtime.run(
        "f = create_temp_file()\n"
        "f.write('kept-open until cleanup')\n"
        "raise ValueError('boom')")
    assert result.reason == Reason.SCRIPT_ERROR.value
    # No cleanup-layer failure is layered on top of the script failure.
    assert result.reason != Reason.CLEANUP_ERROR.value


def test_spawned_child_process_does_not_survive_the_run():
    runtime = _runtime()
    source = (
        "p = spawn_process(%r, '-c', "
        "'import time; time.sleep(60)')\n"
        "pid = p.pid" % sys.executable)
    result = runtime.run(source, result_variable="pid")
    assert result.ok
    pid = result.value
    # The child must have been terminated and reaped by cleanup.
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_child_process_spawned_in_failing_script_is_still_reclaimed():
    runtime = _runtime()
    # Record the child pid through a mutable container the engine exposes so
    # it remains observable even though the script later raises.
    source = (
        "p = spawn_process(%r, '-c', "
        "'import time; time.sleep(60)')\n"
        "record['pid'] = p.pid\n"
        "raise RuntimeError('after spawn')" % sys.executable)
    record = {}
    result = runtime.run(source, names={"record": record})
    # Script failed, but the process reclamation still happened.
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert record["pid"]
    with pytest.raises(OSError):
        os.kill(record["pid"], 0)


def test_state_does_not_leak_between_consecutive_runs():
    runtime = _runtime()
    first = runtime.run(
        "f = create_temp_file()\npath = f.name", result_variable="path")
    assert first.ok
    assert not os.path.exists(first.value)
    second = runtime.run("value = 2", result_variable="value")
    assert second.ok and second.value == 2


def test_cleanup_failure_is_distinguishable_from_other_layers():
    # A reclamation that always fails maps to the dedicated exit
    # code/reason, distinct from policy and quota failures.
    cleanup = ResourceCleanup()

    class BrokenHandle:
        closed = False

        def close(self):
            raise OSError("refused")

    cleanup.track_handle(BrokenHandle())
    with pytest.raises(CleanupFailure) as exc:
        cleanup.reclaim()
    assert exc.value.exit_code == 40
    assert exc.value.reason == Reason.CLEANUP_ERROR
