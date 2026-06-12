"""Regression evals: declarative cases run against a tool registry.

Run a cases file from the command line:

    python -m agent_harness.evals path/to/cases.py

The file must expose ``SUITE`` (an EvalSuite). The process exits with the
report's exit code, so a failing eval fails the CI job that runs it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from agent_harness.tools import ToolRegistry


@dataclass(frozen=True)
class Check:
    """A named assertion over a tool's output value.

    ``fn`` receives the ToolResult value and returns None on pass or a
    human-readable failure reason.
    """

    label: str
    fn: Callable[[Any], str | None]


def _resolve(value: Any, path: str) -> Any:
    if not path:
        return value
    for part in path.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise LookupError(f"key {part!r} not found at {path!r}")
            value = value[part]
        elif isinstance(value, (list, tuple)):
            if not part.lstrip("-").isdigit():
                raise LookupError(f"expected an index at {part!r} in {path!r}")
            index = int(part)
            if not -len(value) <= index < len(value):
                raise LookupError(f"index {index} out of range at {path!r}")
            value = value[index]
        else:
            if not hasattr(value, part):
                raise LookupError(f"attribute {part!r} not found at {path!r}")
            value = getattr(value, part)
    return value


def equals(path: str, expected: Any) -> Check:
    def fn(value: Any) -> str | None:
        actual = _resolve(value, path)
        if actual != expected:
            return f"expected {expected!r}, got {actual!r}"
        return None

    return Check(f"equals({path!r}, {expected!r})", fn)


def contains(path: str, member: Any) -> Check:
    def fn(value: Any) -> str | None:
        actual = _resolve(value, path)
        if member not in actual:
            return f"{member!r} not in {actual!r}"
        return None

    return Check(f"contains({path!r}, {member!r})", fn)


def predicate(fn: Callable[[Any], bool], path: str = "", label: str | None = None) -> Check:
    name = label or getattr(fn, "__name__", "predicate")

    def check(value: Any) -> str | None:
        actual = _resolve(value, path)
        if not fn(actual):
            return f"predicate failed on {actual!r}"
        return None

    return Check(f"predicate({name}, path={path!r})", check)


def _apply(check: Check, value: Any) -> str | None:
    try:
        reason = check.fn(value)
    except Exception as exc:
        return f"{check.label}: raised {type(exc).__name__}: {exc}"
    if reason is not None:
        return f"{check.label}: {reason}"
    return None


@dataclass(frozen=True)
class EvalCase:
    name: str
    tool: str
    input: dict[str, Any]
    checks: Sequence[Check]

    def run(self, registry: ToolRegistry) -> "CaseResult":
        try:
            result = registry.call(self.tool, **self.input)
        except KeyError as exc:
            return CaseResult(self.name, False, [str(exc)])
        if not result.ok:
            return CaseResult(self.name, False, [f"tool error: {result.error}"])
        reasons = [r for check in self.checks if (r := _apply(check, result.value)) is not None]
        return CaseResult(self.name, not reasons, reasons)


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalReport:
    suite: str
    results: Sequence[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def exit_code(self) -> int:
        return 0 if self.failed == 0 else 1

    def render(self) -> str:
        lines = [f"suite: {self.suite}"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  {status}  {r.name}")
            for reason in r.reasons:
                lines.append(f"        {reason}")
        lines.append(f"{self.passed} passed, {self.failed} failed")
        return "\n".join(lines)


class EvalSuite:
    def __init__(self, name: str, registry: ToolRegistry, cases: Sequence[EvalCase]) -> None:
        self.name = name
        self.registry = registry
        self.cases = list(cases)

    def run(self, stream: TextIO | None = None) -> EvalReport:
        report = EvalReport(self.name, [case.run(self.registry) for case in self.cases])
        if stream is not None:
            stream.write(report.render() + "\n")
        return report


def _load_cases_module(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"cases file not found: {path}")
    # The cases file's directory goes on sys.path so it can import sibling
    # modules (e.g. the module that defines and registers the tools).
    parent = str(path.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent_harness.evals",
        description="Run an eval suite and exit non-zero on any failure.",
    )
    parser.add_argument("cases", help="Python file exposing SUITE (an EvalSuite)")
    args = parser.parse_args(argv)

    module = _load_cases_module(Path(args.cases))
    suite = getattr(module, "SUITE", None)
    if not isinstance(suite, EvalSuite):
        print(f"error: {args.cases} does not expose SUITE as an EvalSuite", file=sys.stderr)
        return 2
    return suite.run(stream=sys.stdout).exit_code


if __name__ == "__main__":
    # runpy executes this file as a separate ``__main__`` module instance, so
    # re-import through the canonical name; otherwise the EvalSuite class used
    # by main() differs from the one the cases file imports.
    from agent_harness.evals import main as canonical_main

    raise SystemExit(canonical_main())
