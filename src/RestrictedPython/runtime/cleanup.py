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
"""Layer 3 -- deterministic resource reclamation after every script run.

A :class:`ResourceCleanup` tracks three kinds of resource a script is allowed
to hold through guarded helpers: open handles, temporary files/directories and
child processes.  :meth:`reclaim` runs on **both** the success and failure
paths (the engine always calls it from a ``finally`` block) and reports any
residual kind instead of silently carrying state into the next run.

Host paths are kept internal to the cleanup bookkeeping; diagnostics only name
the *kind* of residual (``handle`` / ``temporary-file`` / ``subprocess``),
never the path itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import typing

from RestrictedPython.runtime.errors import CleanupFailure


class TrackedTempFile:
    """File-like wrapper exposing the real temp path as ``name``.

    On some platforms (notably Windows/CPython) a file object built with
    :func:`os.fdopen` reports the integer file descriptor as ``name`` rather
    than the path.  This wrapper restores the path so callers can reason
    about the file while every other attribute/method is delegated to the
    underlying handle.
    """

    def __init__(self, handle: typing.Any, path: str) -> None:
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "name", path)

    def __getattr__(self, attr: str) -> typing.Any:
        return getattr(object.__getattribute__(self, "_handle"), attr)

    def close(self) -> None:
        object.__getattribute__(self, "_handle").close()

    @property
    def closed(self) -> bool:
        return bool(object.__getattribute__(self, "_handle").closed)


class ResourceCleanup:
    """Registry of resources owned by one execution."""

    def __init__(self) -> None:
        self._handles: list[typing.Any] = []
        self._temp_paths: list[str] = []
        self._processes: list[subprocess.Popen[typing.Any]] = []
        self.reclaimed = False
        # Residual kinds after reclaim(), e.g. ("handle", "subprocess").
        self.residuals: tuple[str, ...] = ()

    # -- registration hooks (used by guarded resource helpers)

    def track_handle(self, handle: typing.Any) -> typing.Any:
        self._handles.append(handle)
        return handle

    def track_temp_path(self, path: str) -> str:
        self._temp_paths.append(path)
        return path

    def track_process(
            self, process: subprocess.Popen[typing.Any]
    ) -> subprocess.Popen[typing.Any]:
        self._processes.append(process)
        return process

    # -- guarded resource factories
    # Scripts cannot import os/subprocess, so any temp file or child process
    # must come through these audited helpers, which auto-register for
    # reclamation.

    def temporary_file(self, *, suffix: str = "",
                       prefix: str = "rp-",
                       text: bool = True) -> typing.Any:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix,
                                    text=text)
        self.track_temp_path(path)
        handle = os.fdopen(fd, "w+" if text else "wb+")
        return self.track_handle(TrackedTempFile(handle, path))

    def temporary_directory(self, *, prefix: str = "rp-") -> str:
        return self.track_temp_path(tempfile.mkdtemp(prefix=prefix))

    def spawn(self, *args: str, **kwargs: typing.Any
              ) -> subprocess.Popen[typing.Any]:
        # No shell, no inherited host environment beyond an explicit empty
        # mapping; the engine's network/FS policy still applies to the parent.
        kwargs.setdefault("shell", False)
        kwargs.setdefault("env", {})
        process = subprocess.Popen(args, **kwargs)
        return self.track_process(process)

    # -- reclamation

    def reclaim(self) -> None:
        """Close/terminate/remove every tracked resource.

        Always runs to completion even if an individual reclamation step
        raises; afterwards :attr:`residuals` lists whatever could not be
        reclaimed and a :class:`CleanupFailure` is raised when non-empty.
        """
        if self.reclaimed:
            return
        failures: list[str] = []

        for handle in list(self._handles):
            if not _close_handle(handle):
                failures.append("handle")
        self._handles = [h for h in self._handles if not _is_closed(h)]

        for process in list(self._processes):
            if not _terminate_process(process):
                failures.append("subprocess")
        self._processes = [p for p in self._processes if p.poll() is None]

        for path in list(self._temp_paths):
            if not _remove_path(path):
                failures.append("temporary-file")
        self._temp_paths = [p for p in self._temp_paths if os.path.exists(p)]

        # Stable, de-duplicated ordering for deterministic diagnostics.
        self.residuals = tuple(sorted(set(failures)))
        self.reclaimed = True
        if self.residuals:
            raise CleanupFailure(self.residuals)

    # -- introspection for tests

    def has_residuals(self) -> bool:
        return bool(
            self._handles or self._temp_paths or self._processes)

    def live_handle_count(self) -> int:
        return sum(1 for h in self._handles if not _is_closed(h))

    def live_process_count(self) -> int:
        return sum(1 for p in self._processes if p.poll() is None)

    def live_temp_path_count(self) -> int:
        return sum(1 for p in self._temp_paths if os.path.exists(p))


def _is_closed(handle: typing.Any) -> bool:
    closed = getattr(handle, "closed", None)
    if closed is not None:
        return bool(closed)
    # File-descriptor based objects without a ``closed`` attribute.
    fileno = getattr(handle, "fileno", None)
    if fileno is None:
        return True
    try:
        os.fstat(fileno())
    except (OSError, ValueError):
        return True
    return False


def _close_handle(handle: typing.Any) -> bool:
    if _is_closed(handle):
        return True
    try:
        handle.close()
    except (OSError, ValueError):
        return False
    return _is_closed(handle)


def _terminate_process(
        process: subprocess.Popen[typing.Any], *,
        wait_timeout: float = 2.0) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.terminate()
        process.wait(timeout=wait_timeout)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try:
            process.kill()
            process.wait(timeout=wait_timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return False
    return process.poll() is not None


def _remove_path(path: str) -> bool:
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=False)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        return False
    return not os.path.exists(path)
