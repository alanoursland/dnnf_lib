"""Cooperative controls and diagnostics for in-process compilation.

Compilation is intentionally allowed to be expensive: the resulting circuit
turns that work into a measurable, provisionable online artifact.  These
controls let callers bound an exploratory compile, cancel it cooperatively,
and surface progress without changing the compiler's search semantics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class CompilationStats:
    """Snapshot of an in-progress or completed compilation."""

    elapsed_seconds: float
    decisions: int
    cache_hits: int
    components: int
    nodes: int
    cache_entries: int
    complete: bool = False


class CompilationInterrupted(RuntimeError):
    """Base class for cooperative compilation termination."""

    def __init__(self, message: str, stats: CompilationStats):
        super().__init__(message)
        self.stats = stats


class CompilationCancelled(CompilationInterrupted):
    """Raised when a compile's cancellation callback requests a stop."""


class CompilationBudgetExceeded(CompilationInterrupted):
    """Raised when a time, node, or cache budget is exceeded."""


@dataclass(frozen=True)
class CompileControl:
    """Optional limits and callbacks for a compilation.

    Parameters
    ----------
    timeout_seconds:
        Maximum elapsed wall-clock time for this compile.
    deadline:
        Absolute :func:`time.monotonic` deadline.  When both deadline and
        timeout are supplied, the earlier limit wins.
    max_nodes, max_cache_entries:
        Cooperative resource limits.  A compiler may overshoot by the small
        number of nodes or cache entries created between safe points.
    cancel:
        Zero-argument callback returning true when cancellation is requested.
    progress:
        Callback receiving :class:`CompilationStats` snapshots.  A final
        snapshot with ``complete=True`` is always emitted on success.
    progress_interval:
        Minimum seconds between non-final progress callbacks.
    """

    timeout_seconds: Optional[float] = None
    deadline: Optional[float] = None
    max_nodes: Optional[int] = None
    max_cache_entries: Optional[int] = None
    cancel: Optional[Callable[[], bool]] = None
    progress: Optional[Callable[[CompilationStats], None]] = None
    progress_interval: float = 0.25

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and non-negative")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        for name in ("max_nodes", "max_cache_entries"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            not math.isfinite(self.progress_interval)
            or self.progress_interval < 0
        ):
            raise ValueError("progress_interval must be finite and non-negative")

    def _start(self) -> "_CompileSession":
        return _CompileSession(self)


class _CompileSession:
    """Mutable per-invocation state owned by a compiler."""

    def __init__(self, control: CompileControl):
        self.control = control
        self.started = time.monotonic()
        timeout_deadline = (
            self.started + control.timeout_seconds
            if control.timeout_seconds is not None
            else None
        )
        if timeout_deadline is None:
            self.deadline = control.deadline
        elif control.deadline is None:
            self.deadline = timeout_deadline
        else:
            self.deadline = min(timeout_deadline, control.deadline)
        self.last_progress = self.started
        self.decisions = 0
        self.cache_hits = 0
        self.components = 0

    def snapshot(
        self, nodes: int, cache_entries: int, complete: bool = False
    ) -> CompilationStats:
        return CompilationStats(
            elapsed_seconds=time.monotonic() - self.started,
            decisions=self.decisions,
            cache_hits=self.cache_hits,
            components=self.components,
            nodes=nodes,
            cache_entries=cache_entries,
            complete=complete,
        )

    def check(self, nodes: int, cache_entries: int, force: bool = False) -> None:
        now = time.monotonic()
        stats = self.snapshot(nodes, cache_entries)
        if self.control.cancel is not None and self.control.cancel():
            raise CompilationCancelled("compilation cancelled", stats)
        if self.deadline is not None and now >= self.deadline:
            raise CompilationBudgetExceeded(
                "compilation time budget exceeded", stats
            )
        if (
            self.control.max_nodes is not None
            and nodes > self.control.max_nodes
        ):
            raise CompilationBudgetExceeded(
                f"compilation node budget exceeded "
                f"({nodes} > {self.control.max_nodes})",
                stats,
            )
        if (
            self.control.max_cache_entries is not None
            and cache_entries > self.control.max_cache_entries
        ):
            raise CompilationBudgetExceeded(
                f"compilation cache budget exceeded "
                f"({cache_entries} > {self.control.max_cache_entries})",
                stats,
            )
        if self.control.progress is not None and (
            force
            or now - self.last_progress >= self.control.progress_interval
        ):
            self.control.progress(stats)
            self.last_progress = now

    def finish(self, nodes: int, cache_entries: int) -> CompilationStats:
        stats = self.snapshot(nodes, cache_entries, complete=True)
        if self.control.progress is not None:
            self.control.progress(stats)
        return stats
