from __future__ import annotations

import io
from pathlib import Path

from agent_harness import (
    EvalCase,
    EvalSuite,
    ToolRegistry,
    contains,
    equals,
    predicate,
    tool,
)
from agent_harness.evals import main

EXAMPLES = Path(__file__).parent.parent / "examples"


@tool(version="1.0.0")
def lookup(key: str) -> dict:
    data = {"city": {"name": "Rio", "tags": ["favela", "lidar"]}}
    if key not in data:
        raise KeyError(key)
    return data[key]


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(lookup)
    return registry


def run_case(case: EvalCase):
    return case.run(make_registry())


def test_equals_passes_and_fails_with_reason() -> None:
    ok = EvalCase("eq", "lookup", {"key": "city"}, [equals("name", "Rio")])
    assert run_case(ok).passed is True

    bad = run_case(EvalCase("eq", "lookup", {"key": "city"}, [equals("name", "Lima")]))
    assert bad.passed is False
    assert "expected 'Lima', got 'Rio'" in bad.reasons[0]


def test_contains_checks_membership_and_substrings() -> None:
    ok = EvalCase(
        "in",
        "lookup",
        {"key": "city"},
        [contains("tags", "lidar"), contains("name", "Ri")],
    )
    bad = EvalCase("in", "lookup", {"key": "city"}, [contains("tags", "robot")])
    assert run_case(ok).passed is True
    assert run_case(bad).passed is False


def test_predicate_passes_and_fails() -> None:
    ok = EvalCase(
        "pred", "lookup", {"key": "city"}, [predicate(lambda tags: len(tags) == 2, path="tags")]
    )
    bad = EvalCase(
        "pred", "lookup", {"key": "city"}, [predicate(lambda tags: len(tags) == 5, path="tags")]
    )
    assert run_case(ok).passed is True
    result = run_case(bad)
    assert result.passed is False
    assert "predicate failed" in result.reasons[0]


def test_dotted_path_traverses_lists_and_reports_missing_keys() -> None:
    indexed = EvalCase("path", "lookup", {"key": "city"}, [equals("tags.0", "favela")])
    assert run_case(indexed).passed is True

    missing = run_case(
        EvalCase("path", "lookup", {"key": "city"}, [equals("population", 6_000_000)])
    )
    assert missing.passed is False
    assert "not found" in missing.reasons[0]


def test_tool_errors_and_unknown_tools_fail_the_case() -> None:
    errored = run_case(EvalCase("err", "lookup", {"key": "nowhere"}, [equals("name", "Rio")]))
    assert errored.passed is False
    assert "tool error" in errored.reasons[0]
    assert "KeyError" in errored.reasons[0]

    unknown = run_case(EvalCase("missing", "nope", {}, []))
    assert unknown.passed is False
    assert "unknown tool" in unknown.reasons[0]


def test_report_counts_and_exit_codes() -> None:
    passing = EvalCase("ok", "lookup", {"key": "city"}, [equals("name", "Rio")])
    failing = EvalCase("bad", "lookup", {"key": "city"}, [equals("name", "Lima")])

    all_green = EvalSuite("green", make_registry(), [passing]).run()
    assert (all_green.passed, all_green.failed, all_green.exit_code) == (1, 0, 0)

    mixed = EvalSuite("mixed", make_registry(), [passing, failing]).run()
    assert (mixed.passed, mixed.failed, mixed.exit_code) == (1, 1, 1)


def test_run_renders_compact_table_to_stream() -> None:
    stream = io.StringIO()
    suite = EvalSuite(
        "table",
        make_registry(),
        [
            EvalCase("ok", "lookup", {"key": "city"}, [equals("name", "Rio")]),
            EvalCase("bad", "lookup", {"key": "city"}, [equals("name", "Lima")]),
        ],
    )
    suite.run(stream=stream)
    output = stream.getvalue()
    assert "PASS  ok" in output
    assert "FAIL  bad" in output
    assert "1 passed, 1 failed" in output


def test_cli_exits_zero_on_example_suite(capsys) -> None:
    code = main([str(EXAMPLES / "eval_cases.py")])
    assert code == 0
    assert "6 passed, 0 failed" in capsys.readouterr().out


def test_cli_exit_codes_for_failing_and_invalid_suites(tmp_path: Path, capsys) -> None:
    failing = tmp_path / "failing_cases.py"
    failing.write_text(
        "from agent_harness import EvalCase, EvalSuite, ToolRegistry, equals, tool\n"
        "\n"
        "@tool(version='1.0.0')\n"
        "def shout(text: str) -> str:\n"
        "    return text.upper()\n"
        "\n"
        "registry = ToolRegistry()\n"
        "registry.register(shout)\n"
        "SUITE = EvalSuite('failing', registry, [\n"
        "    EvalCase('wrong', 'shout', {'text': 'hi'}, [equals('', 'hi')]),\n"
        "])\n"
    )
    assert main([str(failing)]) == 1
    assert "1 failed" in capsys.readouterr().out

    no_suite = tmp_path / "empty_cases.py"
    no_suite.write_text("x = 1\n")
    assert main([str(no_suite)]) == 2
