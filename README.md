# agent-harness

A small harness that makes LLM-agent tool use testable: typed tools, readable
traces, regression evals. No LLM calls, no network; one dependency (pydantic).
You bring the model — this is the part around it that should behave like
ordinary software.

## Why

Most agent failures in production are not model failures. They are harness
failures: a tool call arrives with malformed input and the exception vanishes
into a retry loop; something breaks in the field and nobody can say which tool
version was live or what the input was; retrieval quality drifts and nothing
measures it; a prompt or tool change ships and the only regression test is the
users. This library is the minimum structure that addresses those four
problems — validated inputs, structured results, one trace line per call, and
an eval suite that fails CI before a regression reaches anyone.

## Install

```
pip install -e ".[dev]"
```

## Quickstart

```python
from agent_harness import ToolRegistry, Tracer, tool, EvalCase, EvalSuite, equals

@tool(version="1.0.0")
def temperature(city: str, unit: str = "celsius") -> dict:
    """Look up the current temperature for a city."""
    readings = {"rio": 31, "boston": 12}
    return {"city": city, "value": readings[city.lower()], "unit": unit}

registry = ToolRegistry(tracer=Tracer("trace.jsonl"))
registry.register(temperature)

registry.spec()                # JSON-schema specs, ready for an LLM API
result = registry.call("temperature", city="Rio")
result.ok, result.value        # True, {'city': 'Rio', 'value': 31, ...}

bad = registry.call("temperature", city=42)
bad.ok, bad.error              # False, "invalid input: city: ..." -- never raises

suite = EvalSuite("weather", registry, [
    EvalCase("rio-known", "temperature", {"city": "Rio"}, [equals("value", 31)]),
])
report = suite.run()
report.exit_code               # 0 when everything passes
```

Every call through the registry is validated against a schema derived from the
function's type hints, timed, traced, and returned as a `ToolResult` — tool
exceptions become error results instead of crashing the agent loop.

## Traces

One JSON line per call: timestamp, tool, version, input hash, outcome,
duration. Enough to reconstruct most incidents, small enough to grep.

```json
{"ts": "...", "tool": "kb_search", "version": "1.2.0", "input_hash": "9f31c27eb",
 "ok": true, "duration_ms": 0.41}
```

Read it back with `read_trace("trace.jsonl")`.

## Wiring evals into CI

Put cases in a file that exposes `SUITE`, then run it with the module CLI:

```
python -m agent_harness.evals examples/eval_cases.py
```

The process exits with the report's exit code — 0 when green, 1 on any
failure — so a single CI line guards against regressions:

```yaml
- run: python -m agent_harness.evals examples/eval_cases.py
```

This repository's own CI does exactly that. The example suite in
`examples/eval_cases.py` includes a regression guard over a deterministic
knowledge-base search tool; the comment on `title-match-outranks-body-match`
shows how to break it deliberately and watch CI catch it.

## Example

```
python examples/kb_search.py
```

Runs a small in-memory knowledge-base search end to end: registers two tools,
makes good and bad calls, and prints the resulting trace.

## Status

v0.1, deliberately minimal. The core ideas are stable; the API surface may
still move. Python 3.10+.

## License

MIT
