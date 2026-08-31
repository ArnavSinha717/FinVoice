"""Record what silently did not run during a pipeline call.

Most of this pipeline degrades gracefully: a missing HF token skips diarization,
an uninstalled package disables toxicity analysis, an LLM timeout falls back to
keyword matching. That is the right behaviour — but until now it left no trace in
the output, so a CallRecord produced with half its analyzers disabled looked
identical to a complete one.

That is not just an observability problem. It makes evaluation meaningless: a
harness scoring emotion accuracy cannot tell "the model was wrong" from "the model
never ran". Every component that degrades should say so here.

Thread-safety: process_call() is serialised by _PIPELINE_LOCK, so exactly one call
is in flight at a time and a module-level list is safe. The lock below guards the
Stage 4 worker threads, which report concurrently within a single call.
"""

import threading
from dataclasses import dataclass, asdict

_lock = threading.Lock()
_entries: list["Degradation"] = []


@dataclass(frozen=True)
class Degradation:
    component: str   # e.g. "emotion2vec", "diarization"
    reason: str      # why it did not run
    impact: str      # which output fields are affected
    severity: str    # "info" | "partial" | "disabled"

    def as_dict(self) -> dict:
        return asdict(self)


def reset() -> None:
    """Clear state at the start of a pipeline run."""
    with _lock:
        _entries.clear()


def report(component: str, reason: str, impact: str, severity: str = "disabled") -> None:
    """Record that a component did not run, or ran in a reduced mode."""
    entry = Degradation(component=component, reason=reason, impact=impact, severity=severity)
    with _lock:
        if entry not in _entries:      # idempotent — per-segment callers may repeat
            _entries.append(entry)


def snapshot() -> list[dict]:
    """Everything reported during the current run, oldest first."""
    with _lock:
        return [e.as_dict() for e in _entries]


def summary() -> str:
    """One-line human summary, for logging at the end of a run."""
    with _lock:
        if not _entries:
            return "all analyzers ran"
        return ", ".join(f"{e.component}({e.severity})" for e in _entries)
