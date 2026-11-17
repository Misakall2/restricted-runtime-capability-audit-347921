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
"""Layer-independent diagnostics for the restricted runtime.

Failure causes are partitioned into three stable ``reason`` values so the
whitelist, quota and cleanup layers can be tested independently:

* ``policy_denied``  -- a call was refused by the whitelist layer;
* ``resource_limit`` -- a wall-clock / memory / concurrency quota tripped;
* ``cleanup_error``  -- post-run reclamation left residual resources;
* ``script_error``   -- the untrusted script itself raised.

Exception text and structured results never contain host paths.  Audit
records keep only the rejection stage and a rule/quota identifier.
"""

from __future__ import annotations

import enum
import typing


class Reason(str, enum.Enum):
    POLICY_DENIED = "policy_denied"
    RESOURCE_LIMIT = "resource_limit"
    CLEANUP_ERROR = "cleanup_error"
    SCRIPT_ERROR = "script_error"
    LOAD_ERROR = "load_error"


# Process exit codes are grouped by layer so an external caller can tell the
# cause apart without parsing prose.
EXIT_OK = 0
EXIT_SCRIPT_ERROR = 10
EXIT_POLICY_DENIED = 20
EXIT_RESOURCE_LIMIT = 30
EXIT_CLEANUP_ERROR = 40
EXIT_LOAD_ERROR = 50


class RuntimeErrorBase(Exception):
    """Base class carrying a stable reason and a host-path-free message."""

    reason: Reason = Reason.SCRIPT_ERROR
    exit_code: int = EXIT_SCRIPT_ERROR

    def __init__(self, message: str, *, diagnostic: str | None = None,
                 rule_id: str | None = None) -> None:
        super().__init__(message)
        # ``diagnostic`` is a stable, path-free identifier pair such as
        # "stage=resolve rule=RP-WL-0007".
        self.diagnostic = diagnostic or message
        self.rule_id = rule_id


class PolicyViolation(RuntimeErrorBase):
    """A whitelist rule refused the operation (layer 1)."""

    reason = Reason.POLICY_DENIED
    exit_code = EXIT_POLICY_DENIED


class QuotaExceeded(RuntimeErrorBase):
    """A resource quota was exceeded (layer 2).

    ``dimension`` records which of wall-clock time, memory or concurrency
    triggered the termination so the structured result names the cause.
    """

    reason = Reason.RESOURCE_LIMIT
    exit_code = EXIT_RESOURCE_LIMIT

    def __init__(self, dimension: str, limit: int | float, *,
                 rule_id: str | None = None) -> None:
        self.dimension = dimension
        self.limit = limit
        diagnostic = (
            f"stage=execute dimension={dimension} "
            f"quota={rule_id or 'RP-QM'} limit={limit!r}")
        super().__init__(
            f"resource quota exceeded on dimension {dimension!r}",
            diagnostic=diagnostic, rule_id=rule_id)


class CleanupFailure(RuntimeErrorBase):
    """Reclamation after a run left residual objects (layer 3)."""

    reason = Reason.CLEANUP_ERROR
    exit_code = EXIT_CLEANUP_ERROR

    def __init__(self, residuals: typing.Collection[str]) -> None:
        self.residuals = tuple(residuals)
        diagnostic = (
            "stage=cleanup residual-kinds="
            + ",".join(self.residuals))
        super().__init__(
            "cleanup failed to reclaim all resources",
            diagnostic=diagnostic, rule_id="RP-CL-0001")


class AuditEvent(typing.NamedTuple):
    """One path-free diagnostic record."""

    stage: str
    rule_id: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "rule_id": self.rule_id,
            "detail": self.detail,
        }


class ExecutionResult(typing.NamedTuple):
    """Structured outcome returned for every script run."""

    ok: bool
    exit_code: int
    reason: str
    # Human-readable but path-free summary.
    message: str
    # Stable diagnostic identifier(s), e.g. rule numbers / quota dimension.
    diagnostics: tuple[str, ...]
    # The script's own return value when it produced one.
    value: typing.Any = None
    # Printed output collected through the guarded print hook.
    output: str = ""

    def as_dict(self) -> dict[str, typing.Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "message": self.message,
            "diagnostics": list(self.diagnostics),
            "output": self.output,
        }
