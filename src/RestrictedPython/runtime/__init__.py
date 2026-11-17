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
"""Hardened, in-process restricted execution runtime.

The runtime adds three independently testable defence layers on top of the
compile-time restrictions of :mod:`RestrictedPython`:

* :mod:`~RestrictedPython.runtime.policy` -- an enumerable whitelist of
  builtin operations that must fully load before any execution;
* :mod:`~RestrictedPython.runtime.quota` -- per-run wall-clock, memory and
  concurrency quotas;
* :mod:`~RestrictedPython.runtime.cleanup` -- handle/temp-file/subprocess
  reclamation on both success and failure paths.

Typical use::

    from RestrictedPython.runtime import RestrictedRuntime

    runtime = RestrictedRuntime()
    result = runtime.run("answer = 6 * 7", result_variable="answer")
    assert result.ok and result.value == 42
"""

from RestrictedPython.runtime.cleanup import ResourceCleanup
from RestrictedPython.runtime.engine import RestrictedRuntime
from RestrictedPython.runtime.engine import sanitize_message
from RestrictedPython.runtime.errors import AuditEvent
from RestrictedPython.runtime.errors import CleanupFailure
from RestrictedPython.runtime.errors import ExecutionResult
from RestrictedPython.runtime.errors import PolicyViolation
from RestrictedPython.runtime.errors import QuotaExceeded
from RestrictedPython.runtime.errors import Reason
from RestrictedPython.runtime.policy import STAGE_LOAD
from RestrictedPython.runtime.policy import STAGE_RESOLVE
from RestrictedPython.runtime.policy import PolicyDenial
from RestrictedPython.runtime.policy import Rule
from RestrictedPython.runtime.policy import WhitelistPolicy
from RestrictedPython.runtime.quota import DIMENSION_CONCURRENCY
from RestrictedPython.runtime.quota import DIMENSION_MEMORY
from RestrictedPython.runtime.quota import DIMENSION_WALL_CLOCK
from RestrictedPython.runtime.quota import QuotaConfig
from RestrictedPython.runtime.quota import ResourceQuota
from RestrictedPython.runtime.quota import reset_global_concurrency


__all__ = [
    "AuditEvent",
    "CleanupFailure",
    "DIMENSION_CONCURRENCY",
    "DIMENSION_MEMORY",
    "DIMENSION_WALL_CLOCK",
    "ExecutionResult",
    "PolicyDenial",
    "PolicyViolation",
    "QuotaConfig",
    "QuotaExceeded",
    "Reason",
    "ResourceCleanup",
    "ResourceQuota",
    "RestrictedRuntime",
    "Rule",
    "STAGE_LOAD",
    "STAGE_RESOLVE",
    "WhitelistPolicy",
    "reset_global_concurrency",
    "sanitize_message",
]
