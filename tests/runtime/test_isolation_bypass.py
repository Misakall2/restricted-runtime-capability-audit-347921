"""Integration tests: bypass scripts against every isolation point.

These scripts emulate hostile input trying to reach host file system,
process, network and introspection mechanisms.  Each must be blocked, while a
representative benign script keeps working.  The structured result must always
name the real cause (rule denial / quota / cleanup) rather than a vague error.
"""

import pytest

from RestrictedPython.runtime import Reason
from RestrictedPython.runtime import RestrictedRuntime
from RestrictedPython.runtime import sanitize_message
from RestrictedPython.runtime.quota import QuotaConfig


@pytest.fixture()
def runtime():
    return RestrictedRuntime(quota_config=QuotaConfig(
        wall_clock_seconds=5, memory_bytes=4 * 1024 * 1024,
        max_concurrency=None))


# Host capability -> bypass script.  Every entry must be refused.
ISOLATION_BYPASSES = [
    # File system
    ("fs-open", 'open("/etc/passwd")'),
    ("fs-input", "input()"),
    ("fs-import-os", "import os"),
    ("fs-from-os", "from os import remove"),
    # Process
    ("process-exit", "exit()"),
    ("process-quit", "quit()"),
    ("process-subprocess-import", "import subprocess"),
    ("process-breakpoint", "breakpoint()"),
    # Network
    ("network-socket-import", "import socket"),
    ("network-urllib-import", "import urllib.request"),
    ("network-http-import", "import http.client"),
    # Code generation / dynamic escape
    ("codegen-eval", 'eval("1+1")'),
    ("codegen-exec", 'exec("x=1")'),
    # Introspection / namespace escape
    ("introspect-globals", "globals()"),
    ("introspect-locals", "locals()"),
    ("introspect-vars", "vars()"),
    ("introspect-dir", "dir()"),
    ("attr-dunder-class", "(1).__class__"),
    ("attr-dunder-mro", "(1).__class__.__mro__"),
    ("attr-subclasses", "().__class__.__base__.__subclasses__()"),
    ("attr-generator-frame",
     "def g():\n    yield 1\ng().gi_frame"),
    ("attr-traceback-frame",
     "try:\n    1/0\nexcept Exception as e:\n    e.__traceback__"),
    ("format-string-leak", '"{0.__class__}".format(1)'),
    # Builtins table tampering
    ("tamper-builtins-set", '__builtins__["evil"] = 1'),
    ("tamper-builtins-update", "__builtins__.update({})"),
]


@pytest.mark.parametrize("label,script", ISOLATION_BYPASSES)
def test_every_isolation_point_blocks_bypass(runtime, label, script):
    result = runtime.run(script)
    assert not result.ok, f"{label} was allowed: {script}"
    # Blocked either at compile time or by the runtime whitelist, but never
    # reported as a successful execution and never as an opaque error.
    assert result.reason in (
        Reason.POLICY_DENIED.value,
        Reason.SCRIPT_ERROR.value,
    )
    assert result.exit_code in (10, 20)
    # No host path may appear anywhere in the surfaced result.
    assert "\\" not in result.message
    assert "//" not in result.message


def test_benign_script_remains_usable(runtime):
    result = runtime.run(
        "squares = [n * n for n in range(5)]\n"
        "total = sum(squares)\n"
        "print('done')",
        result_variable="total")
    assert result.ok
    assert result.exit_code == 0
    assert result.value == sum(n * n for n in range(5))
    assert result.output == "done\n"
    assert result.reason == "ok"


def test_structured_result_distinguishes_the_three_failure_layers(runtime):
    # Layer 1 -- policy denial.
    denied = runtime.run('open("x")')
    assert denied.reason == Reason.POLICY_DENIED.value
    assert denied.exit_code == 20
    assert denied.diagnostics[0].startswith("RP-WL-")

    # Layer 2 -- resource limit.
    tight_memory = RestrictedRuntime(quota_config=QuotaConfig(
        wall_clock_seconds=None, memory_bytes=64 * 1024,
        max_concurrency=None))
    limited = tight_memory.run("data = list(range(10_000_000))")
    assert limited.reason == Reason.RESOURCE_LIMIT.value
    assert limited.exit_code == 30
    assert "memory" in limited.diagnostics[0]

    # Layer 3 -- cleanup anomaly (unit-level).
    from RestrictedPython.runtime import ResourceCleanup
    from RestrictedPython.runtime.errors import CleanupFailure

    class Broken:
        closed = False

        def close(self):
            raise OSError("no")

    cleanup = ResourceCleanup()
    cleanup.track_handle(Broken())
    with pytest.raises(CleanupFailure) as exc:
        cleanup.reclaim()
    assert exc.value.reason == Reason.CLEANUP_ERROR
    assert exc.value.exit_code == 40


def test_script_error_is_reported_with_type_but_without_host_path(runtime):
    result = runtime.run(
        'raise RuntimeError("failed at E:/secret/host/path.txt")')
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert result.exit_code == 10
    assert "RuntimeError" in result.message
    assert "E:/secret" not in result.message
    assert "<host-path>" in result.message


def test_system_exit_inside_script_is_contained(runtime):
    result = runtime.run("raise SystemExit(2)")
    # The engine must not itself exit; it captures the script's exception.
    assert not result.ok
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert result.exit_code == 10


def test_audit_record_keeps_only_stage_and_rule_id(runtime):
    runtime.run('open("E:/hidden/secret.txt")')
    assert runtime.audit_log
    for event in runtime.audit_log:
        payload = repr(event.as_dict())
        assert "E:/" not in payload
        assert event.stage
        assert event.rule_id.startswith(("RP-WL", "RP-QM", "RP-CL"))


def test_sanitize_message_strips_posix_and_windows_paths():
    assert sanitize_message("open /var/secret/key.pem failed").find(
        "/var/secret") == -1
    cleaned = sanitize_message(
        r"cannot read C:\Users\admin\key.pem")
    assert "key.pem" not in cleaned
    assert "<host-path>" in cleaned
    # Ordinary text is left untouched.
    assert sanitize_message("plain error") == "plain error"


def test_syntax_error_does_not_leak_paths_and_is_structured(runtime):
    result = runtime.run("def broken(:")
    assert not result.ok
    assert result.reason == Reason.SCRIPT_ERROR.value
    assert result.exit_code == 10
    assert result.diagnostics


def test_result_as_dict_is_json_like_and_complete(runtime):
    result = runtime.run("value = 1", result_variable="value")
    payload = result.as_dict()
    assert set(payload) == {
        "ok", "exit_code", "reason", "message",
        "diagnostics", "output",
    }
    assert payload["ok"] is True
    assert isinstance(payload["diagnostics"], list)


def test_network_capability_is_absent_from_whitelist(runtime):
    names = runtime.policy.allowed_names()
    # No socket/network primitive may be enumerable as a capability.
    for forbidden in ("socket", "urllib", "requests", "http",
                      "connect", "subprocess", "os", "sys"):
        assert forbidden not in names
