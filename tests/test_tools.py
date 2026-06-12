from __future__ import annotations

import pytest

from agent_harness import ToolRegistry, ToolResult, tool


@tool(version="1.0.0")
def add(a: int, b: int = 10) -> int:
    """Add two integers."""
    return a + b


@tool(version="2.1.0", description="Always fails.")
def explode(message: str) -> None:
    raise RuntimeError(message)


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(add)
    r.register(explode)
    return r


def test_valid_call_returns_ok_result_with_duration() -> None:
    result = add(a=2, b=3)
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.value == 5
    assert result.error is None
    assert result.duration_ms >= 0


def test_default_parameter_is_honoured() -> None:
    result = add(a=2)
    assert result.ok is True
    assert result.value == 12


def test_schema_validation_rejects_bad_input() -> None:
    missing = add(b=3)
    assert missing.ok is False
    assert "invalid input" in missing.error
    assert "a" in missing.error

    wrong_type = add(a="not a number")
    assert wrong_type.ok is False
    assert "invalid input" in wrong_type.error

    unexpected = add(a=1, c=99)
    assert unexpected.ok is False
    assert "c" in unexpected.error


def test_exception_becomes_error_result_not_raise() -> None:
    result = explode(message="boom")
    assert result.ok is False
    assert result.error == "RuntimeError: boom"
    assert result.value is None


def test_underlying_function_stays_directly_callable() -> None:
    assert add.fn(2, 3) == 5
    with pytest.raises(RuntimeError):
        explode.fn("boom")


def test_spec_surfaces_version_description_and_schema() -> None:
    spec = explode.spec()
    assert spec["name"] == "explode"
    assert spec["version"] == "2.1.0"
    assert spec["description"] == "Always fails."

    schema = add.spec()["input_schema"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["required"] == ["a"]
    assert add.spec()["description"] == "Add two integers."


def test_registry_routes_calls_by_name(registry: ToolRegistry) -> None:
    result = registry.call("add", a=1, b=2)
    assert result.ok is True
    assert result.value == 3


def test_registry_unknown_tool_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        registry.call("does_not_exist", a=1)


def test_registry_rejects_duplicate_names(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register(add)


def test_registry_spec_lists_all_tools(registry: ToolRegistry) -> None:
    specs = registry.spec()
    assert [s["name"] for s in specs] == ["add", "explode"]


def test_var_args_are_rejected_at_decoration_time() -> None:
    with pytest.raises(TypeError, match="named parameters"):

        @tool(version="1.0.0")
        def bad(*args: int) -> int:
            return sum(args)
