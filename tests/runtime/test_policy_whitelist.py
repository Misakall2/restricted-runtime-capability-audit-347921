"""Tests for layer 1 -- the enumerable whitelist policy.

These tests exercise the policy in isolation (no quota, no cleanup), so a
failure here is unambiguously a rule/policy failure rather than a resource or
reclamation problem.
"""

import pytest

from RestrictedPython.runtime import Reason
from RestrictedPython.runtime import RestrictedRuntime
from RestrictedPython.runtime import WhitelistPolicy
from RestrictedPython.runtime.policy import STAGE_RESOLVE
from RestrictedPython.runtime.policy import PolicyDenial


# ---------------------------------------------------------------------------
# Enumerable set
# ---------------------------------------------------------------------------

def test_default_policy_loads_with_stable_rule_ids():
    policy = WhitelistPolicy()
    assert not policy.fail_closed
    ids = policy.rule_ids()
    assert len(ids) > 50
    # Rule ids are unique and follow the RP-WL-xxxx scheme.
    assert len(set(ids)) == len(ids)
    assert all(rule_id.startswith("RP-WL-") for rule_id in ids)


def test_allowed_names_is_an_enumerable_set():
    policy = WhitelistPolicy()
    names = policy.allowed_names()
    assert isinstance(names, frozenset)
    assert "abs" in names
    # Host capabilities are represented explicitly (as denial markers), so
    # the set is fully enumerable rather than relying on NameError.
    assert {"open", "__import__", "compile", "eval"} <= names


def test_describe_exposes_no_host_callables():
    policy = WhitelistPolicy()
    for entry in policy.describe():
        assert set(entry) == {"rule_id", "name", "category"}
        assert all(isinstance(value, str) for value in entry.values())


def test_resolve_returns_rule_and_backing_callable():
    policy = WhitelistPolicy()
    rule = policy.resolve("abs")
    assert rule.name == "abs"
    assert rule.rule_id.startswith("RP-WL-")
    assert callable(rule.callable) or rule.category == "constant"


# ---------------------------------------------------------------------------
# Deny-unknown behaviour
# ---------------------------------------------------------------------------

def test_resolve_unknown_name_is_denied_with_rule_number():
    policy = WhitelistPolicy()
    with pytest.raises(PolicyDenial) as exc:
        policy.resolve("definitely_not_whitelisted")
    assert exc.value.rule_id == "RP-WL-MISSING"
    assert exc.value.stage == STAGE_RESOLVE
    # The diagnostic never carries a host path.
    assert "/" not in str(exc.value) and "\\" not in str(exc.value)


def test_runtime_rejects_call_to_non_whitelisted_builtin():
    runtime = RestrictedRuntime()
    result = runtime.run("value = definitely_not_whitelisted_fn(1)")
    assert not result.ok
    assert result.exit_code == 20
    assert result.reason == Reason.POLICY_DENIED.value
    assert result.diagnostics[0] == "RP-WL-MISSING"


def test_filesystem_process_network_and_codegen_are_denied():
    runtime = RestrictedRuntime()
    scripts = {
        "open": 'open("whatever")',
        "input": "input()",
        "compile": 'compile("1", "<x>", "eval")',
        "exit": "exit()",
        "quit": "quit()",
        "globals": "globals()",
        "locals": "locals()",
        "vars": "vars(1)",
        "dir": "dir(1)",
    }
    for label, source in scripts.items():
        result = runtime.run(source)
        assert not result.ok, label
        assert result.reason == Reason.POLICY_DENIED.value, label
        assert result.exit_code == 20, label


def test_imports_are_refused():
    runtime = RestrictedRuntime()
    # A direct dunder import call is rejected at compile time, while an
    # ``import`` statement that survives compilation hits the runtime
    # ``__import__`` denial marker.  Both must be refused, never executed.
    direct = runtime.run("__import__('os')")
    assert not direct.ok
    assert direct.reason in (
        Reason.POLICY_DENIED.value, Reason.SCRIPT_ERROR.value)
    statement = runtime.run("import os")
    assert not statement.ok
    assert statement.reason == Reason.POLICY_DENIED.value


# ---------------------------------------------------------------------------
# Fail-closed loading
# ---------------------------------------------------------------------------

def test_duplicate_rule_name_fails_closed():
    bad_spec = [
        ("", "abs", "pure-builtin", abs),
        ("", "abs", "pure-builtin", abs),
    ]
    policy = WhitelistPolicy(bad_spec)
    assert policy.fail_closed
    assert policy.load_error is not None
    assert policy.rule_ids() == ()
    with pytest.raises(PolicyDenial):
        policy.resolve("abs")


def test_rule_without_implementation_fails_closed():
    # A spec is a *collection* of entry tuples, so the single entry is
    # wrapped in a list rather than passed as a bare tuple.
    policy = WhitelistPolicy([("", "broken", "pure-builtin", None)])
    assert policy.fail_closed
    assert "no impl" in (policy.load_error or "")


def test_malformed_spec_entry_fails_closed():
    policy = WhitelistPolicy([("only-three", "fields")])
    assert policy.fail_closed


def test_fail_closed_policy_refuses_every_script_before_execution():
    policy = WhitelistPolicy([
        ("", "abs", "pure-builtin", abs),
        ("", "abs", "pure-builtin", abs),
    ])
    runtime = RestrictedRuntime(policy=policy)
    result = runtime.run("x = 1 + 1")
    assert not result.ok
    assert result.exit_code == 50
    assert result.reason == Reason.LOAD_ERROR.value
    # The audit trail records the load-stage refusal with a rule number.
    assert any(event.rule_id == "RP-WL-0000"
               and event.stage == "load" for event in runtime.audit_log)


# ---------------------------------------------------------------------------
# Immutable capability table
# ---------------------------------------------------------------------------

def test_builtins_mapping_cannot_be_mutated_by_script():
    runtime = RestrictedRuntime()
    tamper_scripts = [
        '__builtins__["evil"] = 1',
        'builtins["open"] = open',
        "__builtins__.update({'x': 1})",
    ]
    for source in tamper_scripts:
        result = runtime.run(source)
        assert not result.ok
    # The table is still intact afterwards.
    healthy = runtime.run("value = abs(-4)", result_variable="value")
    assert healthy.ok and healthy.value == 4


def test_builtins_copy_keeps_policy_enforcement():
    policy = WhitelistPolicy()
    builtins = policy.build_builtins()
    clone = builtins.copy()
    with pytest.raises(PolicyDenial):
        clone["anything"] = 1
    assert clone["abs"] is abs
    with pytest.raises(PolicyDenial):
        clone["unknown"]


# ---------------------------------------------------------------------------
# Benign functionality stays available
# ---------------------------------------------------------------------------

def test_normal_scripts_remain_fully_usable():
    runtime = RestrictedRuntime()
    result = runtime.run(
        "total = sum(x * x for x in range(10))\n"
        "names = [str(n) for n in list(range(3))]\n"
        "mapping = dict(a=max(1, 2), b=min(1, 2))",
        result_variable="total")
    assert result.ok
    assert result.exit_code == 0
    assert result.value == sum(x * x for x in range(10))


def test_whitelist_can_be_tested_independently_of_other_layers():
    # A bespoke minimal policy proves the layer is independently testable.
    policy = WhitelistPolicy([
        ("", "int", "pure-builtin", int),
        ("", "str", "pure-builtin", str),
    ])
    runtime = RestrictedRuntime(policy=policy)
    allowed = runtime.run("value = int(str(42))", result_variable="value")
    assert allowed.ok and allowed.value == 42
    denied = runtime.run("value = abs(-1)")
    assert denied.reason == Reason.POLICY_DENIED.value
