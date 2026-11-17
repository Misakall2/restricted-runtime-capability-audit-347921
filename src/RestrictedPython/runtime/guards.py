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
"""Guarded callables exposed by the default whitelist.

Two kinds of callable live here:

* ``denied_*`` sentinels -- explicit, audited refusals of host capabilities
  (file system, process, network, introspection, code generation);
* ``guarded_*`` wrappers -- safe equivalents that route allocations through
  the active :class:`~RestrictedPython.runtime.quota.ResourceQuota`.

Every denial raises :class:`~RestrictedPython.runtime.errors.PolicyViolation`
with a stable rule number and **no** host path in the message.  Uncaught
exceptions raised inside the sandbox are sanitized at the engine boundary, so
host paths can never leak through arbitrary tracebacks either.
"""

from __future__ import annotations

import importlib
import math
import random
import string
import types
import typing

from RestrictedPython import Guards
from RestrictedPython.runtime.errors import PolicyViolation
from RestrictedPython.runtime.quota import active_quota


# Rule numbers for explicitly denied host capabilities.  These are stable
# identifiers used in audit records; they mirror the policy table ordering.
_RULE_OPEN = "RP-WL-DENY-FS"
_RULE_CODEGEN = "RP-WL-DENY-CODEGEN"
_RULE_IMPORT = "RP-WL-DENY-IMPORT"
_RULE_DEBUG = "RP-WL-DENY-DEBUG"
_RULE_PROCESS = "RP-WL-DENY-PROCESS"
_RULE_INTROSPECT = "RP-WL-DENY-INTROSPECT"


def _deny(rule_id: str, operation: str) -> typing.NoReturn:
    raise PolicyViolation(
        f"operation is forbidden inside the restricted runtime: {operation}",
        diagnostic=f"stage=resolve rule={rule_id} operation={operation}",
        rule_id=rule_id)


# --- explicit host-capability refusals ---------------------------------------

def denied_filesystem(*args: typing.Any,
                      **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse file system access (``open``/``input``)."""
    _deny(_RULE_OPEN, "filesystem")


def denied_codegen(*args: typing.Any,
                   **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse dynamic code generation (``compile``/``eval``/``exec``)."""
    _deny(_RULE_CODEGEN, "code-generation")


def denied_import(*args: typing.Any,
                  **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse module imports, which could reach socket/subprocess/os."""
    _deny(_RULE_IMPORT, "import")


def denied_debug(*args: typing.Any,
                 **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse debugger entry points."""
    _deny(_RULE_DEBUG, "debugger")


def denied_process(*args: typing.Any,
                   **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse process control (``exit``/``quit``)."""
    _deny(_RULE_PROCESS, "process-control")


def denied_introspection(*args: typing.Any,
                         **kwargs: typing.Any) -> typing.NoReturn:
    """Refuse namespace/frame introspection."""
    _deny(_RULE_INTROSPECT, "introspection")


# --- quota-aware safe wrappers -----------------------------------------------

def guarded_range(*args: int) -> range:
    quota = active_quota(required=True)
    # Delegate argument validation to the standard limited_range semantics,
    # but enforce the per-script quota before materializing anything.
    quota.charge_allocation_hint()
    if not args:
        raise TypeError("range expected 1 argument, got 0")
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError("range expected at most 3 arguments, got %d"
                        % len(args))
    if step == 0:
        raise ValueError("range() arg 3 must not be zero")
    length = max(0, (stop - start + step - (1 if step > 0 else -1)) // step)
    quota.charge_elements(length)
    return range(start, stop, step)


def guarded_list(seq: typing.Any = ()) -> list[typing.Any]:
    quota = active_quota(required=True)
    quota.charge_allocation_hint()
    if isinstance(seq, str):
        raise TypeError("cannot convert string to list")
    result = list(seq)
    quota.charge_elements(len(result))
    return result


def guarded_tuple(seq: typing.Any = ()) -> tuple[typing.Any, ...]:
    quota = active_quota(required=True)
    quota.charge_allocation_hint()
    if isinstance(seq, str):
        raise TypeError("cannot convert string to tuple")
    result = tuple(seq)
    quota.charge_elements(len(result))
    return result


def guarded_getattr(object: typing.Any, name: str,
                    default: typing.Any = Guards.raise_) -> typing.Any:
    safe_getattr = typing.cast(typing.Any, Guards.safer_getattr)
    if default is Guards.raise_:
        return safe_getattr(object, name)
    return safe_getattr(object, name, default)


def guarded_hasattr(object: typing.Any, name: str) -> bool:
    try:
        guarded_getattr(object, name)
    except AttributeError:
        return False
    return True


def guarded_getitem(ob: typing.Any, index: typing.Any) -> typing.Any:
    return ob[index]


def guarded_getiter(ob: typing.Any) -> typing.Any:
    return ob


def guarded_write(ob: typing.Any) -> typing.Any:
    """Reuse RestrictedPython's write guard so writes hit guarded types."""
    return Guards.full_write_guard(ob)


class _GuardedPrint:
    """Print collector bound to one execution; mirrors PrintCollector.

    The transformer rewrites ``print(...)`` into
    ``_print._call_print(...)`` and reads output via ``printed`` (an
    invocation of the collector), so the public surface matches
    :class:`RestrictedPython.PrintCollector.PrintCollector`.
    """

    def __init__(self, getattr_guard: typing.Any = None) -> None:
        self.txt: list[str] = []
        self._getattr_ = getattr_guard

    def write(self, text: str) -> None:
        self.txt.append(str(text))

    def __call__(self) -> str:
        return "".join(self.txt)

    def _call_print(self, *objects: typing.Any,
                    **kwargs: typing.Any) -> None:
        target = kwargs.get("file", None)
        if target is None:
            kwargs["file"] = self
        else:
            # A custom file target must expose a guarded ``write``.
            write = self._getattr_(target, "write")
            write(self._render(objects, kwargs))
            return
        # ``print`` writes through the file-like object's ``write``.
        print(*objects, **kwargs)

    @staticmethod
    def _render(objects: typing.Any, kwargs: typing.Any) -> str:
        sep = typing.cast(str, kwargs.get("sep", " "))
        end = typing.cast(str, kwargs.get("end", "\n"))
        return sep.join(str(obj) for obj in objects) + end

    @property
    def text(self) -> str:
        return "".join(self.txt)


def guarded_print(getattr_guard: typing.Any) -> _GuardedPrint:
    # The transformer injects ``_print_(_getattr_)``; output is fully
    # captured instead of touching the host's stdout.
    return _GuardedPrint(getattr_guard)


# --- side-effect free stdlib modules -----------------------------------------

def _load_stdlib(name: str) -> types.ModuleType:
    """Import a vetted stdlib module at policy-load time only.

    Untrusted code never gets an import hook: these modules are pre-loaded
    here and handed out as already-resolved objects.
    """
    return importlib.import_module(name)


class _NetworkFreeDelegator:
    """Wrap a module so only whitelisted, side-effect-free attributes show."""

    def __init__(self, module: typing.Any,
                 allowed: frozenset[str]) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_allowed", allowed)

    def __getattr__(self, attr: str) -> typing.Any:
        allowed = object.__getattribute__(self, "_allowed")
        module = object.__getattribute__(self, "_module")
        if attr not in allowed:
            _deny("RP-WL-DENY-MODULE-ATTR", f"module-attribute:{attr}")
        return getattr(module, attr)


def guarded_math() -> _NetworkFreeDelegator:
    # math has no I/O surface; expose its callables/constants wholesale but
    # keep the delegator as a choke point for future attribute policy.
    allowed = frozenset(
        name for name in dir(math) if not name.startswith("_"))
    return _NetworkFreeDelegator(math, allowed)


def guarded_random() -> _NetworkFreeDelegator:
    # Exclude the seeding-from-system-entropy/os-touching entry points and
    # anything stateful that could reach the host; keep pure PRNG helpers.
    excluded = frozenset((
        "SystemRandom", "seed", "getstate", "setstate",
        "getrandbits",  # fine in itself but keep the surface minimal
    ))
    allowed = frozenset(
        name for name in dir(random)
        if not name.startswith("_") and name not in excluded)
    return _NetworkFreeDelegator(random.Random(), allowed)


# ``string`` is loaded via _load_stdlib by the policy table; keep the name
# imported here so the module graph is explicit for static analysis.
_ = string
