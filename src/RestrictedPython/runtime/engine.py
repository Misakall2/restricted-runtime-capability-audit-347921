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
"""Restricted runtime engine -- orchestrates the three defence layers.

Execution order is fixed and enforced in code:

1. **Load** the whitelist policy completely.  A load failure fails closed and
   every script is refused before any execution action.
2. **Enter** the per-run :class:`ResourceQuota` (wall clock / memory /
   concurrency).
3. **Execute** the compiled restricted code with an immutable guarded
   builtins mapping.
4. **Reclaim** resources in a ``finally`` block, on both the success and
   failure paths.

All exceptions crossing the engine boundary are converted to a structured
:class:`ExecutionResult`.  Host paths are stripped from any message; audit
records retain only the rejection stage and a rule/quota identifier.
"""

from __future__ import annotations

import keyword
import re
import typing

from RestrictedPython.compile import compile_restricted_exec
from RestrictedPython.runtime import guards as _guards
from RestrictedPython.runtime.cleanup import ResourceCleanup
from RestrictedPython.runtime.errors import EXIT_CLEANUP_ERROR
from RestrictedPython.runtime.errors import EXIT_LOAD_ERROR
from RestrictedPython.runtime.errors import EXIT_OK
from RestrictedPython.runtime.errors import EXIT_POLICY_DENIED
from RestrictedPython.runtime.errors import EXIT_RESOURCE_LIMIT
from RestrictedPython.runtime.errors import EXIT_SCRIPT_ERROR
from RestrictedPython.runtime.errors import AuditEvent
from RestrictedPython.runtime.errors import CleanupFailure
from RestrictedPython.runtime.errors import ExecutionResult
from RestrictedPython.runtime.errors import PolicyViolation
from RestrictedPython.runtime.errors import QuotaExceeded
from RestrictedPython.runtime.errors import Reason
from RestrictedPython.runtime.policy import STAGE_LOAD
from RestrictedPython.runtime.policy import PolicyDenial
from RestrictedPython.runtime.policy import WhitelistPolicy
from RestrictedPython.runtime.quota import QuotaConfig
from RestrictedPython.runtime.quota import ResourceQuota


# Matches Windows drive paths, UNC paths and POSIX absolute paths.
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s'\"]*)"
    r"|(?:\\\\[^\s'\"]+)"
    r"|(?:/[^\s'\"]+)")


def sanitize_message(text: object) -> str:
    """Remove host paths and frame locations from an exception message."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _PATH_RE.sub("<host-path>", text)
    # Drop "File \"...\", line n" style traceback fragments entirely.
    cleaned = re.sub(
        r'File "<host-path>"[^,\n]*(?:, line \d+)?',
        'File "<restricted>"', cleaned)
    return cleaned


class RestrictedRuntime:
    """Entry point for running untrusted scripts inside the sandbox."""

    def __init__(
        self,
        policy: WhitelistPolicy | None = None,
        quota_config: QuotaConfig | None = None,
        audit_sink: typing.Callable[[AuditEvent], None] | None = None,
        *,
        enable_resource_factories: bool = False,
    ) -> None:
        # Layer 1 must be fully loaded before anything can execute.
        self.policy = policy if policy is not None else WhitelistPolicy()
        self.quota_config = quota_config or QuotaConfig()
        self._audit_sink = audit_sink
        # Host resource factories (temp files/dirs, child processes) are
        # withheld by default so the sandbox exposes no file-system or
        # process capability at all.  A host that explicitly wants scripts
        # to hold audited, auto-reclaimed resources opts in.
        self.enable_resource_factories = enable_resource_factories
        self._audit_log: list[AuditEvent] = []
        if self.policy.fail_closed:
            self._audit(AuditEvent(
                stage=STAGE_LOAD,
                rule_id="RP-WL-0000",
                detail="whitelist load failed; failing closed"))

    # -- audit

    def _audit(self, event: AuditEvent) -> None:
        self._audit_log.append(event)
        if self._audit_sink is not None:
            self._audit_sink(event)

    @property
    def audit_log(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit_log)

    # -- public API

    def run(
        self,
        source: str,
        *,
        names: typing.Mapping[str, typing.Any] | None = None,
        result_variable: str | None = None,
    ) -> ExecutionResult:
        """Compile and execute ``source``; never raises to the caller."""
        # Fail-closed policy: refuse before compiling or executing.
        if self.policy.fail_closed:
            event = AuditEvent(
                stage=STAGE_LOAD, rule_id="RP-WL-0000",
                detail="script refused: policy is in fail-closed state")
            self._audit(event)
            return ExecutionResult(
                ok=False, exit_code=EXIT_LOAD_ERROR,
                reason=Reason.LOAD_ERROR.value,
                message="whitelist failed to load; all scripts refused",
                diagnostics=(event.rule_id,))

        compiled = compile_restricted_exec(source, filename="<restricted>")
        if compiled.errors:
            # Syntax errors come from the parser; sanitize defensively.
            messages = tuple(sanitize_message(e) for e in compiled.errors)
            return ExecutionResult(
                ok=False, exit_code=EXIT_SCRIPT_ERROR,
                reason=Reason.SCRIPT_ERROR.value,
                message="script failed to compile",
                diagnostics=messages)

        cleanup = ResourceCleanup()
        print_collector = _guards._GuardedPrint()
        builtins = self.policy.build_builtins()
        sandbox_globals = self._build_globals(
            builtins, cleanup, print_collector, names)

        quota = ResourceQuota(self.quota_config)
        exit_code = EXIT_OK
        reason = Reason.SCRIPT_ERROR.value
        message = "ok"
        diagnostics: list[str] = []
        value: typing.Any = None
        ok = False
        cleanup_error: CleanupFailure | None = None

        code = compiled.code
        assert code is not None  # errors were empty, so compilation succeeded
        try:
            with quota:
                exec(code, sandbox_globals, sandbox_globals)
            if result_variable is not None:
                self._validate_result_name(result_variable)
                value = sandbox_globals.get(result_variable)
            ok = True
            reason = "ok"
            exit_code = EXIT_OK
        except PolicyDenial as exc:
            exit_code = EXIT_POLICY_DENIED
            reason = Reason.POLICY_DENIED.value
            message = sanitize_message(str(exc))
            diagnostics.append(exc.rule_id)
            self._audit(AuditEvent(
                stage=exc.stage, rule_id=exc.rule_id,
                detail=f"denied name={exc.name}"))
        except PolicyViolation as exc:
            exit_code = EXIT_POLICY_DENIED
            reason = Reason.POLICY_DENIED.value
            message = sanitize_message(str(exc))
            diagnostics.append(exc.rule_id or "RP-WL-DENY")
            self._audit(AuditEvent(
                stage="resolve",
                rule_id=exc.rule_id or "RP-WL-DENY",
                detail=sanitize_message(exc.diagnostic)))
        except QuotaExceeded as exc:
            exit_code = EXIT_RESOURCE_LIMIT
            reason = Reason.RESOURCE_LIMIT.value
            message = sanitize_message(str(exc))
            diagnostics.append(exc.diagnostic)
            self._audit(AuditEvent(
                stage="execute",
                rule_id=exc.rule_id or "RP-QM",
                detail=f"dimension={exc.dimension}"))
        except KeyboardInterrupt:
            # The wall-clock watchdog injects this; translate it into the
            # structured quota result naming the wall_clock dimension.
            exit_code = EXIT_RESOURCE_LIMIT
            reason = Reason.RESOURCE_LIMIT.value
            message = "resource quota exceeded on dimension 'wall_clock'"
            diagnostics.append("stage=execute dimension=wall_clock "
                               "quota=RP-QM-0001")
            self._audit(AuditEvent(
                stage="execute", rule_id="RP-QM-0001",
                detail="dimension=wall_clock"))
        except BaseException as exc:  # includes SystemExit from scripts
            exit_code = EXIT_SCRIPT_ERROR
            reason = Reason.SCRIPT_ERROR.value
            message = sanitize_message(
                f"{type(exc).__name__}: {exc}") or (
                f"{type(exc).__name__}: <redacted>")
            diagnostics.append("stage=execute rule=RP-SCRIPT")
        finally:
            # Layer 3 runs on BOTH paths, even when the script was killed.
            try:
                cleanup.reclaim()
            except CleanupFailure as exc:
                cleanup_error = exc

        if cleanup_error is not None:
            # Reclamation problems take precedence in the exit status so the
            # caller cannot mistake a leaky run for a clean failure.
            exit_code = EXIT_CLEANUP_ERROR
            reason = Reason.CLEANUP_ERROR.value
            message = sanitize_message(str(cleanup_error))
            diagnostics.append(cleanup_error.diagnostic)
            self._audit(AuditEvent(
                stage="cleanup", rule_id=cleanup_error.rule_id or "RP-CL",
                detail="residual-kinds="
                       + ",".join(cleanup_error.residuals)))
            ok = False

        return ExecutionResult(
            ok=ok,
            exit_code=exit_code,
            reason=reason,
            message=message,
            diagnostics=tuple(diagnostics),
            value=value,
            output=print_collector.text)

    # -- helpers

    def _build_globals(
        self,
        builtins: typing.Any,
        cleanup: ResourceCleanup,
        print_collector: _guards._GuardedPrint,
        names: typing.Mapping[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        glb: dict[str, typing.Any] = {"__builtins__": builtins}
        # The transformer emits calls to these underscore hooks; provide the
        # same guarded objects through globals (user code cannot name them).
        glb["_getattr_"] = _guards.guarded_getattr
        glb["_getitem_"] = _guards.guarded_getitem
        glb["_getiter_"] = _guards.guarded_getiter
        glb["_write_"] = _guards.guarded_write
        glb["_print_"] = lambda _getattr: print_collector
        # Audited resource helpers are withheld unless the host explicitly
        # opts in; the default sandbox therefore has no FS/process surface.
        # When enabled they register with ``cleanup`` so every resource they
        # hand out is deterministically reclaimed after the run.
        if self.enable_resource_factories:
            glb["create_temp_file"] = cleanup.temporary_file
            glb["create_temp_dir"] = cleanup.temporary_directory
            glb["spawn_process"] = cleanup.spawn
        if names:
            for key, val in names.items():
                if isinstance(key, str) and key.isidentifier() \
                        and not keyword.iskeyword(key) \
                        and not key.startswith("_"):
                    glb[key] = val
        return glb

    @staticmethod
    def _validate_result_name(name: str) -> None:
        if not (isinstance(name, str) and name.isidentifier()
                and not keyword.iskeyword(name)
                and not name.startswith("_")):
            raise ValueError("result_variable must be a public identifier")
