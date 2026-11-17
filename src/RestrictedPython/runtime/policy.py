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
"""Layer 1 -- enumerable whitelist policy for the restricted runtime.

Every callable exposed to untrusted code is represented by a :class:`Rule`
carrying a stable rule id (``RP-WL-xxxx``).  A :class:`WhitelistPolicy` must be
fully loaded *before* any execution takes place; if loading fails the policy
enters ``fail_closed`` mode and resolves *no* names at all, so a missing rule
can never silently open a capability.

The policy only answers two questions -- "is this name enumerable/allowed?"
and "which implementation backs it?" -- it never performs I/O itself.  The
diagnostic surface deliberately contains rule ids and policy stages, never
host file system paths.
"""

from __future__ import annotations

import collections.abc
import typing


class PolicyError(Exception):
    """Raised when a whitelist policy cannot be loaded consistently.

    A policy that raises during construction fails closed: the runtime
    refuses every subsequent script rather than executing with partial
    rules.
    """


# Stages reported in diagnostics.  Only these stable identifiers are ever
# written to audit records or surfaced to callers -- never host paths.
STAGE_LOAD = "load"
STAGE_RESOLVE = "resolve"


class Rule(typing.NamedTuple):
    """One enumerable whitelist entry."""

    rule_id: str
    name: str
    callable: typing.Any
    category: str

    def as_dict(self) -> dict[str, str]:
        # Intentionally excludes ``callable`` so the enumerable view never
        # carries host objects, only stable identifiers.
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category,
        }


def _default_spec() -> tuple[tuple[str, str, str, typing.Any], ...]:
    """Build the default rule specification.

    Imported lazily inside the function so that importing this module never
    hard-depends on the guard implementations and the table stays the single
    enumerable source of truth for the default policy.
    """
    import builtins

    from RestrictedPython import Guards
    from RestrictedPython import Utilities
    from RestrictedPython.runtime import guards as _guards

    spec: list[tuple[str, str, str, typing.Any]] = []

    # Pure, side-effect free builtins taken straight from the interpreter.
    pure_builtins: tuple[tuple[str, typing.Any], ...] = (
        ("abs", builtins.abs),
        ("bool", builtins.bool),
        ("callable", builtins.callable),
        ("chr", builtins.chr),
        ("complex", builtins.complex),
        ("divmod", builtins.divmod),
        ("float", builtins.float),
        ("hash", builtins.hash),
        ("hex", builtins.hex),
        ("id", builtins.id),
        ("int", builtins.int),
        ("isinstance", builtins.isinstance),
        ("issubclass", builtins.issubclass),
        ("len", builtins.len),
        ("oct", builtins.oct),
        ("ord", builtins.ord),
        ("pow", builtins.pow),
        ("repr", builtins.repr),
        ("round", builtins.round),
        ("slice", builtins.slice),
        ("str", builtins.str),
    )
    for name, impl in pure_builtins:
        spec.append(("", name, "pure-builtin", impl))

    # Exception classes are inert types, safe to expose.
    for name in Guards._safe_exceptions:
        spec.append(("", name, "exception", getattr(builtins, name)))

    # Constants / sentinels.  These are keywords, not attributes of the
    # ``builtins`` module, so reference their values directly.
    constants = (
        ("None", None),
        ("False", False),
        ("True", True),
        ("Ellipsis", Ellipsis),
    )
    for name, value in constants:
        spec.append(("", name, "constant", value))
    spec.append(("", "__build_class__", "language",
                 builtins.__build_class__))

    # Guarded wrappers provided by RestrictedPython itself.
    # Guarded wrappers provided by RestrictedPython itself.
    guarded: tuple[tuple[str, typing.Any], ...] = (
        ("setattr", Guards.guarded_setattr),
        ("delattr", Guards.guarded_delattr),
        ("_getattr_", Guards.safer_getattr),
    )
    for name, impl in guarded:
        spec.append(("", name, "guarded", impl))

    # Runtime capability guards: every host-touching operation funnels
    # through these so the quota/cleanup layers stay in control.
    runtime_guarded: tuple[tuple[str, typing.Any], ...] = (
        ("range", _guards.guarded_range),
        ("list", _guards.guarded_list),
        ("tuple", _guards.guarded_tuple),
        ("dict", builtins.dict),
        ("enumerate", builtins.enumerate),
        ("sum", builtins.sum),
        ("min", builtins.min),
        ("max", builtins.max),
        ("all", builtins.all),
        ("any", builtins.any),
        ("map", builtins.map),
        ("filter", builtins.filter),
        ("zip", builtins.zip),
        ("sorted", builtins.sorted),
        ("reversed", builtins.reversed),
        ("iter", builtins.iter),
        ("next", builtins.next),
        ("getattr", _guards.guarded_getattr),
        ("hasattr", _guards.guarded_hasattr),
        ("set", builtins.set),
        ("frozenset", builtins.frozenset),
        ("bytes", builtins.bytes),
        ("bytearray", builtins.bytearray),
        ("memoryview", builtins.memoryview),
        ("object", builtins.object),
        ("super", builtins.super),
        ("property", builtins.property),
        ("staticmethod", builtins.staticmethod),
        ("classmethod", builtins.classmethod),
        ("type", builtins.type),
        # Namespace hooks consumed by the transformed code.
        ("_getitem_", _guards.guarded_getitem),
        ("_getiter_", _guards.guarded_getiter),
        ("_write_", _guards.guarded_write),
        ("_print_", _guards.guarded_print),
        # Explicit, audited denial markers for host capabilities.  Present in
        # the enumerable set so a denial is a policy decision, not an
        # accidental NameError.
        ("open", _guards.denied_filesystem),
        ("input", _guards.denied_filesystem),
        ("compile", _guards.denied_codegen),
        ("eval", _guards.denied_codegen),
        ("exec", _guards.denied_codegen),
        ("__import__", _guards.denied_import),
        ("breakpoint", _guards.denied_debug),
        ("exit", _guards.denied_process),
        ("quit", _guards.denied_process),
        ("globals", _guards.denied_introspection),
        ("locals", _guards.denied_introspection),
        ("vars", _guards.denied_introspection),
        ("dir", _guards.denied_introspection),
    )
    for name, impl in runtime_guarded:
        spec.append(("", name, "runtime-guard", impl))

    # Side-effect free utility modules (network/fs-free).
    spec.append(("", "math", "utility-module", _guards.guarded_math()))
    spec.append(("", "string", "utility-module",
                 Utilities._AttributeDelegator(
                     _guards._load_stdlib("string"), "Formatter")))
    spec.append(("", "random", "utility-module",
                 _guards.guarded_random()))

    return tuple(spec)


class WhitelistPolicy:
    """An immutable, enumerable set of allowed builtin operations.

    Construction either succeeds with a fully populated rule table or fails
    closed.  The :attr:`fail_closed` flag is sticky: once set, :meth:`resolve`
    rejects every name for the lifetime of the policy.
    """

    RULE_PREFIX = "RP-WL-"

    def __init__(
        self,
        spec: collections.abc.Collection[
            tuple[str, str, str, typing.Any]] | None = None,
    ) -> None:
        self._rules: dict[str, Rule] = {}
        self.fail_closed = False
        self.load_error: str | None = None
        try:
            self._load(_default_spec() if spec is None else spec)
        except PolicyError as exc:
            # Fail closed: drop anything partially loaded and refuse all.
            self._rules = {}
            self.fail_closed = True
            self.load_error = str(exc)

    def _load(
        self,
        spec: collections.abc.Collection[
            tuple[str, str, str, typing.Any]],
    ) -> None:
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for index, entry in enumerate(spec):
            if len(entry) != 4:
                raise PolicyError(
                    f"{self.RULE_PREFIX}0000: malformed rule specification")
            _ignored_id, name, category, impl = entry
            if not isinstance(name, str) or not name:
                raise PolicyError(
                    f"{self.RULE_PREFIX}0000: rule without a name")
            if name in seen_names:
                raise PolicyError(
                    f"{self.RULE_PREFIX}0000: duplicate rule name {name!r}")
            if category != "constant" and impl is None:
                msg = f"{self.RULE_PREFIX}0000: rule {name!r} no impl"
                raise PolicyError(msg)
            rule_id = f"{self.RULE_PREFIX}{index + 1:04d}"
            if rule_id in seen_ids:  # pragma: no cover - deterministic ids
                raise PolicyError(f"duplicate rule id {rule_id}")
            seen_ids.add(rule_id)
            seen_names.add(name)
            self._rules[name] = Rule(rule_id, name, impl, category)

    # -- enumerable introspection

    def rule_ids(self) -> tuple[str, ...]:
        """Return all loaded rule ids in a stable order."""
        return tuple(rule.rule_id for rule in self._rules.values())

    def allowed_names(self) -> frozenset[str]:
        """Return the enumerable set of allowed builtin names."""
        return frozenset(self._rules)

    def describe(self) -> tuple[dict[str, str], ...]:
        """Return a host-object-free description of every rule."""
        return tuple(rule.as_dict() for rule in self._rules.values())

    def contains(self, name: str) -> bool:
        return (not self.fail_closed) and name in self._rules

    # -- resolution

    def resolve(self, name: str) -> Rule:
        """Resolve a name to its rule or raise :class:`PolicyDenial`.

        A fail-closed policy denies *everything*, including names that would
        normally be present.
        """
        if self.fail_closed or name not in self._rules:
            raise PolicyDenial(name, stage=STAGE_RESOLVE,
                               fail_closed=self.fail_closed)
        return self._rules[name]

    def build_builtins(self) -> "_GuardedBuiltins":
        """Construct the mapping object installed as ``__builtins__``."""
        return _GuardedBuiltins(self)


class PolicyDenial(Exception):
    """Raised when a call is rejected because no rule allows it.

    The message intentionally contains only the policy stage and a rule
    marker -- never a host path or internal object representation.
    """

    def __init__(self, name: str, *, stage: str,
                 fail_closed: bool = False) -> None:
        # Do not echo arbitrary caller-provided text verbatim beyond the
        # bare name; names are identifiers produced by the bytecode loader.
        safe_name = name if isinstance(name, str) and name.isidentifier() \
            else "<non-identifier>"
        if fail_closed:
            rule_id = "RP-WL-0000"
            detail = "policy failed closed; all scripts are refused"
        else:
            rule_id = "RP-WL-MISSING"
            detail = "no whitelist rule matches the requested operation"
        self.name = safe_name
        self.stage = stage
        self.rule_id = rule_id
        self.fail_closed = fail_closed
        super().__init__(
            f"[{rule_id}] stage={stage} name={safe_name}: {detail}")


class _GuardedBuiltins(dict[str, typing.Any]):
    """Mapping installed as ``__builtins__`` for restricted code.

    Lookups go through the policy so a rebinding trick cannot smuggle in a
    host builtin: only names present in the loaded, non-fail-closed policy
    resolve, and every other name raises :class:`PolicyDenial`.
    """

    def __init__(self, policy: WhitelistPolicy) -> None:
        super().__init__()
        # Store under a mangled, non-identifier-ish key so restricted code,
        # which cannot reference underscore names, cannot reach the policy.
        dict.__setitem__(self, "<restricted-policy>", policy)
        if not policy.fail_closed:
            for name, rule in policy._rules.items():
                dict.__setitem__(self, name, rule.callable)

    @property
    def policy(self) -> WhitelistPolicy:
        value = dict.__getitem__(self, "<restricted-policy>")
        return typing.cast(WhitelistPolicy, value)

    def __getitem__(self, key: str) -> typing.Any:
        policy = self.policy
        rule = policy.resolve(key)  # raises PolicyDenial when absent
        return rule.callable

    def get(self, key: typing.Any, default: typing.Any = None) -> typing.Any:
        try:
            return self[key]
        except PolicyDenial:
            return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.policy.contains(key)

    # Restricted code must not be able to mutate the capability table.
    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        raise PolicyDenial("__setitem__", stage=STAGE_RESOLVE)

    def __delitem__(self, key: typing.Any) -> None:
        raise PolicyDenial("__delitem__", stage=STAGE_RESOLVE)

    def setdefault(self, *args: typing.Any) -> typing.Any:
        raise PolicyDenial("setdefault", stage=STAGE_RESOLVE)

    def update(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        raise PolicyDenial("update", stage=STAGE_RESOLVE)

    def pop(self, *args: typing.Any) -> typing.Any:
        raise PolicyDenial("pop", stage=STAGE_RESOLVE)

    def popitem(self) -> typing.Any:
        raise PolicyDenial("popitem", stage=STAGE_RESOLVE)

    def clear(self) -> None:
        raise PolicyDenial("clear", stage=STAGE_RESOLVE)

    def copy(self) -> "_GuardedBuiltins":
        # A copy keeps the same policy instead of becoming a plain dict that
        # could be mutated into bypassing the whitelist.
        return _GuardedBuiltins(self.policy)
