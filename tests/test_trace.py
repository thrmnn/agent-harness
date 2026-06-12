from __future__ import annotations

import json
from pathlib import Path

from agent_harness import ToolRegistry, Tracer, read_trace, tool
from agent_harness.trace import _hash_arguments


@tool(version="3.0.0")
def divide(a: float, b: float) -> float:
    return a / b


def make_registry(tracer: Tracer) -> ToolRegistry:
    registry = ToolRegistry(tracer=tracer)
    registry.register(divide)
    return registry


def test_in_memory_event_has_expected_fields() -> None:
    tracer = Tracer()
    registry = make_registry(tracer)
    registry.call("divide", a=6, b=2)

    assert len(tracer.events) == 1
    event = tracer.events[0]
    assert event["tool"] == "divide"
    assert event["version"] == "3.0.0"
    assert event["ok"] is True
    assert event["duration_ms"] >= 0
    assert "ts" in event
    assert "error" not in event


def test_error_calls_are_traced_with_message() -> None:
    tracer = Tracer()
    registry = make_registry(tracer)
    registry.call("divide", a=1, b=0)

    event = tracer.events[0]
    assert event["ok"] is False
    assert "ZeroDivisionError" in event["error"]


def test_jsonl_file_round_trips(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    registry = make_registry(Tracer(trace_file))
    registry.call("divide", a=6, b=2)
    registry.call("divide", a=1, b=0)

    raw_lines = trace_file.read_text().splitlines()
    assert len(raw_lines) == 2
    for line in raw_lines:
        json.loads(line)

    events = read_trace(trace_file)
    assert [e["ok"] for e in events] == [True, False]


def test_input_hash_is_stable_and_discriminating() -> None:
    assert _hash_arguments({"a": 1, "b": 2}) == _hash_arguments({"b": 2, "a": 1})
    assert _hash_arguments({"a": 1}) != _hash_arguments({"a": 2})


def test_registry_without_tracer_does_not_fail() -> None:
    registry = ToolRegistry()
    registry.register(divide)
    assert registry.call("divide", a=4, b=2).value == 2
