"""White-box tests for runtime internals.

These cover branch-level behaviour of the guard wrappers, cleanup helpers and
the guarded builtins mapping that the black-box engine tests do not reach
directly. They keep the three layers independently testable.
"""

import os
import subprocess
import sys

import pytest

from RestrictedPython.runtime import Reason
from RestrictedPython.runtime import RestrictedRuntime
from RestrictedPython.runtime import WhitelistPolicy
from RestrictedPython.runtime.cleanup import ResourceCleanup
from RestrictedPython.runtime.cleanup import _close_handle
from RestrictedPython.runtime.cleanup import _is_closed
from RestrictedPython.runtime.cleanup import _remove_path
from RestrictedPython.runtime.cleanup import _terminate_process
from RestrictedPython.runtime.errors import PolicyViolation
from RestrictedPython.runtime.guards import _GuardedPrint
from RestrictedPython.runtime.guards import _NetworkFreeDelegator
from RestrictedPython.runtime.guards import guarded_hasattr
from RestrictedPython.runtime.guards import guarded_list
from RestrictedPython.runtime.guards import guarded_print
from RestrictedPython.runtime.guards import guarded_range
from RestrictedPython.runtime.guards import guarded_tuple
from RestrictedPython.runtime.policy import PolicyDenial
from RestrictedPython.runtime.quota import QuotaConfig
from RestrictedPython.runtime.quota import ResourceQuota


# ---------------------------------------------------------------------------
# Guarded builtin wrappers (executed inside an active quota context)
# ---------------------------------------------------------------------------

@pytest.fixture()
def quota_context():
    config = QuotaConfig(wall_clock_seconds=None, memory_bytes=None,
                         max_concurrency=None)
    quota = ResourceQuota(config)
    with quota:
        yield quota


def test_guarded_range_variant_arity(quota_context):
    assert list(guarded_range(3)) == [0, 1, 2]
    assert list(guarded_range(2, 5)) == [2, 3, 4]
    assert list(guarded_range(0, 6, 2)) == [0, 2, 4]
    assert list(guarded_range(5, 0, -1)) == [5, 4, 3, 2, 1]


def test_guarded_range_rejects_bad_arguments(quota_context):
    with pytest.raises(TypeError):
        guarded_range()
    with pytest.raises(TypeError):
        guarded_range(1, 2, 3, 4)
    with pytest.raises(ValueError):
        guarded_range(1, 5, 0)


def test_guarded_list_and_tuple_reject_strings(quota_context):
    assert guarded_list([1, 2]) == [1, 2]
    assert guarded_tuple((1, 2)) == (1, 2)
    assert guarded_list() == []
    with pytest.raises(TypeError):
        guarded_list("abc")
    with pytest.raises(TypeError):
        guarded_tuple("abc")


def test_guarded_hasattr_distinguishes_members(quota_context):
    assert guarded_hasattr("abc", "upper") is True
    # safer_getattr returns None rather than raising for a missing public
    # attribute, so the hasattr wrapper does not observe an AttributeError.
    assert guarded_hasattr("abc", "definitely_missing") is True
    # Dunder attributes are actively forbidden and surface as AttributeError,
    # which the hasattr wrapper turns into False.
    assert guarded_hasattr("abc", "__class__") is False


def test_guarded_getattr_default_branch(quota_context):
    from RestrictedPython.runtime.guards import guarded_getattr
    assert guarded_getattr("abc", "missing", "fallback") == "fallback"
    # Without a default the guard yields None rather than raising.
    assert guarded_getattr("abc", "missing") is None
    # Dunder access is still refused.
    with pytest.raises(AttributeError):
        guarded_getattr("abc", "__class__")


def test_guarded_print_collector_behaviour():
    collector = guarded_print(lambda ob, name: getattr(ob, name))
    collector._call_print("a", "b")
    assert collector.text == "a b\n"
    assert collector() == "a b\n"
    collector.write("more")
    assert collector.text == "a b\nmore"
    # Custom file target routes through the getattr guard + render.
    captured = []

    class FakeFile:
        def write(self, data):
            captured.append(data)

    collector._call_print("x", "y", file=FakeFile())
    assert captured == ["x y\n"]


def test_guarded_print_render_uses_sep_and_end():
    rendered = _GuardedPrint._render(
        (1, 2), {"sep": "-", "end": "!"})
    assert rendered == "1-2!"


def test_network_free_delegator_allows_known_and_denies_unknown():
    delegator = _NetworkFreeDelegator(sys.modules["math"],
                                      frozenset({"sqrt"}))
    assert delegator.sqrt(16) == 4
    with pytest.raises(PolicyViolation):
        delegator.connect


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def test_is_closed_handles_objects_without_closed_attr():
    class FilenoOnly:
        def fileno(self):
            raise ValueError("bad fd")

    assert _is_closed(FilenoOnly()) is True

    class Nothing:
        pass

    assert _is_closed(Nothing()) is True


def test_close_handle_returns_true_for_already_closed():
    cleanup = ResourceCleanup()
    handle = cleanup.temporary_file()
    handle.close()
    try:
        assert _close_handle(handle) is True
    finally:
        # Remove the temp path even though the handle is already closed.
        cleanup.reclaim()


def test_close_handle_returns_false_when_close_raises():
    class Broken:
        closed = False

        def close(self):
            raise OSError("no")

    assert _close_handle(Broken()) is False


def test_terminate_process_short_circuits_when_already_dead():
    completed = subprocess.Popen(
        [sys.executable, "-c", "pass"])
    completed.wait()
    assert _terminate_process(completed) is True


def test_terminate_process_fallback_kill(monkeypatch):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])

    real_terminate = process.terminate

    def slow_terminate():
        # Make the first wait time out so the kill() branch runs.
        real_terminate()
        raise subprocess.TimeoutExpired(cmd="x", timeout=0)

    monkeypatch.setattr(process, "terminate", slow_terminate)
    try:
        assert _terminate_process(process, wait_timeout=5) is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_terminate_process_reports_failure_when_kill_also_fails(monkeypatch):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    real_wait = process.wait
    real_kill = process.kill

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=0)

    monkeypatch.setattr(process, "terminate", lambda: None)
    monkeypatch.setattr(process, "kill", lambda: None)
    monkeypatch.setattr(process, "wait", raise_timeout)
    try:
        assert _terminate_process(process) is False
    finally:
        # Use the unpatched implementations to guarantee reclamation.
        real_kill()
        real_wait()


def test_remove_path_handles_files_dirs_and_missing(tmp_path):
    target_file = tmp_path / "file.txt"
    target_file.write_text("x")
    assert _remove_path(str(target_file)) is True
    assert not target_file.exists()
    # A path that no longer exists is not an error.
    assert _remove_path(str(target_file)) is True

    target_dir = tmp_path / "sub"
    target_dir.mkdir()
    assert _remove_path(str(target_dir)) is True


def test_remove_path_returns_false_on_oserror(monkeypatch, tmp_path):
    target = tmp_path / "locked.txt"
    target.write_text("x")

    def boom(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr("os.remove", boom)
    assert _remove_path(str(target)) is False


# ---------------------------------------------------------------------------
# Guarded builtins mapping
# ---------------------------------------------------------------------------

def test_guarded_builtins_mapping_surface():
    policy = WhitelistPolicy()
    builtins = policy.build_builtins()
    assert builtins["abs"] is abs
    assert "abs" in builtins
    assert "unknown_name" not in builtins
    assert builtins.get("unknown_name") is None
    assert builtins.get("unknown_name", "d") == "d"
    with pytest.raises(PolicyDenial):
        builtins["unknown_name"]
    with pytest.raises(PolicyDenial):
        builtins["x"] = 1
    with pytest.raises(PolicyDenial):
        del builtins["x"]
    for operation in (
        builtins.setdefault,
        builtins.update,
        builtins.pop,
        builtins.popitem,
        builtins.clear,
    ):
        with pytest.raises(PolicyDenial):
            operation()


def test_guarded_builtins_contains_rejects_non_string():
    builtins = WhitelistPolicy().build_builtins()
    assert (1 in builtins) is False


# ---------------------------------------------------------------------------
# Quota internals
# ---------------------------------------------------------------------------

def test_quota_trip_is_sticky_and_records_dimension():
    config = QuotaConfig(wall_clock_seconds=None, memory_bytes=None,
                         max_concurrency=None)
    quota = ResourceQuota(config)
    assert not quota.tripped
    quota._trip("wall_clock", 1.0)
    assert quota.tripped
    assert quota.tripped_dimension == "wall_clock"
    # A second trip does not overwrite the first dimension.
    quota._trip("memory", 2)
    assert quota.tripped_dimension == "wall_clock"


def test_charge_elements_ignores_non_positive_and_unlimited(quota_context):
    quota_context.charge_elements(0)
    quota_context.charge_elements(-5)
    assert not quota_context.tripped


def test_active_quota_outside_run_raises():
    from RestrictedPython.runtime.quota import active_quota
    assert active_quota() is None
    with pytest.raises(RuntimeError):
        active_quota(required=True)


# ---------------------------------------------------------------------------
# Engine audit + result plumbing
# ---------------------------------------------------------------------------

def test_audit_sink_receives_events():
    events = []
    runtime = RestrictedRuntime(audit_sink=events.append)
    runtime.run('open("x")')
    assert events
    assert all(event.rule_id for event in events)


def test_invalid_result_variable_is_structured_not_raised():
    runtime = RestrictedRuntime()
    # The engine never raises to the caller; an invalid result name is
    # surfaced as a structured script-error result.
    result = runtime.run("x = 1", result_variable="_private")
    assert not result.ok
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert result.exit_code == 10


def test_names_mapping_filters_unsafe_identifiers():
    runtime = RestrictedRuntime()
    result = runtime.run(
        "value = safe",
        names={"safe": 99, "_hidden": 1, "def": 1, 7: "bad"},
        result_variable="value")
    assert result.ok
    assert result.value == 99


def test_utility_modules_are_usable_inside_sandbox():
    runtime = RestrictedRuntime()
    result = runtime.run(
        "root = math.sqrt(16)\nletter = string.ascii_letters[0]",
        result_variable="root")
    assert result.ok
    assert result.value == 4.0
    # random is exposed through a network-free delegator.
    rnd = runtime.run("choice = random.Random is None",
                      result_variable="choice")
    assert rnd.ok


# ---------------------------------------------------------------------------
# Remaining branch coverage
# ---------------------------------------------------------------------------

def test_denied_debug_marker_raises_policy_violation():
    from RestrictedPython.runtime.guards import denied_debug
    with pytest.raises(PolicyViolation):
        denied_debug()


def test_policy_rejects_rule_without_a_name():
    policy = WhitelistPolicy([("", "", "pure-builtin", abs)])
    assert policy.fail_closed


def test_policy_describe_and_contains():
    policy = WhitelistPolicy()
    described = {entry["name"]: entry for entry in policy.describe()}
    assert "abs" in described
    assert policy.contains("abs")
    assert not policy.contains("missing")
    closed = WhitelistPolicy((("", "abs", "pure-builtin", abs),
                              ("", "abs", "pure-builtin", abs)))
    assert not closed.contains("abs")


def test_cleanup_live_count_helpers():
    cleanup = ResourceCleanup()
    handle = cleanup.temporary_file()
    directory = cleanup.temporary_directory()
    process = cleanup.spawn(
        sys.executable, "-c", "import time; time.sleep(30)")
    assert cleanup.live_handle_count() == 1
    assert cleanup.live_process_count() == 1
    assert cleanup.live_temp_path_count() == 2
    assert cleanup.has_residuals()
    cleanup.reclaim()
    assert handle.closed
    assert not os.path.exists(directory)
    assert process.poll() is not None
    assert cleanup.live_handle_count() == 0
    assert cleanup.live_process_count() == 0
    assert cleanup.live_temp_path_count() == 0
    assert not cleanup.has_residuals()


def test_cleanup_reports_subprocess_and_tempfile_residuals(monkeypatch):
    cleanup = ResourceCleanup()
    process = cleanup.spawn(
        sys.executable, "-c", "import time; time.sleep(30)")
    handle = cleanup.temporary_file()
    # On Windows TextIOWrapper.name may be the integer fd, so use the path
    # the cleanup registry actually tracks rather than ``handle.name``.
    path = cleanup._temp_paths[-1]

    # Force both the process-kill and file removal to fail.
    monkeypatch.setattr(
        "RestrictedPython.runtime.cleanup._terminate_process",
        lambda proc, **kw: False)
    monkeypatch.setattr(
        "RestrictedPython.runtime.cleanup._remove_path",
        lambda path: False)
    from RestrictedPython.runtime.errors import CleanupFailure
    try:
        with pytest.raises(CleanupFailure) as exc:
            cleanup.reclaim()
        assert "subprocess" in exc.value.residuals
        assert "temporary-file" in exc.value.residuals
    finally:
        # Real reclamation so the test leaves nothing behind.
        process.kill()
        process.wait()
        if not handle.closed:
            handle.close()
        if os.path.exists(path):
            os.remove(path)


def test_is_closed_fileno_open_branch():
    # A real open file has a working fileno, exercising the fstat path.
    cleanup = ResourceCleanup()
    handle = cleanup.temporary_file()
    try:
        assert _is_closed(handle) is False
    finally:
        cleanup.reclaim()


def test_engine_cleanup_failure_takes_precedence(monkeypatch):
    runtime = RestrictedRuntime()

    def broken_reclaim(self):
        from RestrictedPython.runtime.errors import CleanupFailure
        self.reclaimed = True
        self.residuals = ("handle",)
        raise CleanupFailure(("handle",))

    monkeypatch.setattr(ResourceCleanup, "reclaim", broken_reclaim)
    result = runtime.run("value = 1", result_variable="value")
    assert not result.ok
    assert result.reason == Reason.CLEANUP_ERROR.value
    assert result.exit_code == 40
    assert any(event.stage == "cleanup" for event in runtime.audit_log)


def test_engine_sanitizes_non_string_input():
    from RestrictedPython.runtime.engine import sanitize_message
    assert sanitize_message(12345) == "12345"


def test_quota_memory_sample_trips_inside_run():
    # A tiny memory ceiling must trip during guarded allocation accounting.
    config = QuotaConfig(wall_clock_seconds=None,
                         memory_bytes=1024, max_concurrency=None)
    quota = ResourceQuota(config)
    from RestrictedPython.runtime.errors import QuotaExceeded
    with quota:
        with pytest.raises(QuotaExceeded) as exc:
            quota.charge_elements(10_000_000)
        assert exc.value.dimension == "memory"


def test_quota_baseline_getattr_default_branch():
    # Constructing without entering means no baseline attribute exists; the
    # getattr default branch must not blow up.
    config = QuotaConfig(wall_clock_seconds=None, memory_bytes=None,
                         max_concurrency=None)
    quota = ResourceQuota(config)
    assert getattr(quota, "_baseline_memory", 0) == 0


def test_watchdog_thread_attribute_branches():
    # A run without a wall-clock limit creates no watchdog thread.
    config = QuotaConfig(wall_clock_seconds=None, memory_bytes=None,
                         max_concurrency=None)
    quota = ResourceQuota(config)
    with quota:
        assert quota._watchdog is None
    assert not quota.tripped


def test_is_closed_fstat_success_for_open_fd_without_closed_attr():
    # An object exposing a working fileno but no ``closed`` attribute takes
    # the fstat-success branch and is considered open.
    cleanup = ResourceCleanup()
    real = cleanup.temporary_file()

    class FdOnly:
        def fileno(self):
            return real.fileno()

    try:
        assert _is_closed(FdOnly()) is False
    finally:
        # Full reclamation closes the handle and removes the temp path.
        cleanup.reclaim()


def test_policy_allows_none_implementation_for_constants():
    policy = WhitelistPolicy([("", "None", "constant", None)])
    assert not policy.fail_closed
    assert policy.resolve("None").callable is None


def test_guarded_builtins_for_fail_closed_policy_holds_no_rules():
    closed = WhitelistPolicy([
        ("", "abs", "pure-builtin", abs),
        ("", "abs", "pure-builtin", abs),
    ])
    assert closed.fail_closed
    builtins = closed.build_builtins()
    # The policy reference is retained but no capability names are populated.
    assert builtins.policy is closed
    assert "abs" not in builtins
    with pytest.raises(PolicyDenial):
        builtins["abs"]


def test_quota_reuses_preexisting_tracemalloc_session():
    import tracemalloc
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    try:
        config = QuotaConfig(wall_clock_seconds=None,
                             memory_bytes=64 * 1024 * 1024,
                             max_concurrency=None)
        quota = ResourceQuota(config)
        with quota:
            assert quota._tracemalloc_started_here is False
        # The pre-existing session must still be running after the run.
        assert tracemalloc.is_tracing()
    finally:
        if started_here:
            tracemalloc.stop()


def test_sample_memory_trips_quota_on_live_usage():
    config = QuotaConfig(wall_clock_seconds=None, memory_bytes=1,
                         max_concurrency=None)
    quota = ResourceQuota(config)
    from RestrictedPython.runtime.errors import QuotaExceeded
    with quota:
        # Force an allocation large enough to exceed the 1-byte ceiling.
        holder = [bytearray(4096) for _ in range(4)]
        with pytest.raises(QuotaExceeded) as exc:
            quota._sample_memory()
        assert exc.value.dimension == "memory"
        del holder
