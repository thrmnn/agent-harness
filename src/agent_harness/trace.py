"""Minimal tracing: one JSON line per tool call, to a file and/or memory."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_harness.tools import ToolResult


def _hash_arguments(arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, sort_keys=True, default=repr)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Tracer:
    """Records one event per tool call.

    Events are always kept in ``.events``; if ``path`` is given, each event is
    also appended to that file as a JSON line, so a crashed process still
    leaves a readable trace.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        tool_name: str,
        version: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "version": version,
            "input_hash": _hash_arguments(arguments),
            "ok": result.ok,
            "duration_ms": result.duration_ms,
        }
        if result.error is not None:
            event["error"] = result.error
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        return event


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL trace file back into a list of event dicts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
