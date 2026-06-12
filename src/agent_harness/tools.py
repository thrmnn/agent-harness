"""Typed tool layer: plain functions wrapped with validated schemas and structured results."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

if TYPE_CHECKING:
    from agent_harness.trace import Tracer


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a single tool call. Tools never raise; failures land here."""

    ok: bool
    value: Any = None
    error: str | None = None
    duration_ms: float = 0.0


def _input_model(fn: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"tool {tool_name!r}: *args/**kwargs parameters cannot be expressed "
                "as a JSON schema; use explicit named parameters"
            )
        annotation = hints.get(name, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    return create_model(
        f"{tool_name}_input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(piece) for piece in err["loc"]) or "input"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid input: " + "; ".join(parts)


class Tool:
    """A function plus its validated input schema, version, and metadata.

    Calling the Tool validates input, executes, and returns a ToolResult.
    The raw function stays available as ``.fn`` for direct unit testing.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        version: str,
        description: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.version = version
        doc = inspect.getdoc(fn) or ""
        self.description = description if description is not None else (doc.splitlines()[0] if doc else "")
        self.input_model = _input_model(fn, self.name)

    def __call__(self, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()

        def elapsed_ms() -> float:
            return round((time.perf_counter() - start) * 1000, 3)

        try:
            validated = self.input_model(**kwargs)
        except ValidationError as exc:
            return ToolResult(ok=False, error=_format_validation_error(exc), duration_ms=elapsed_ms())

        arguments = {field: getattr(validated, field) for field in self.input_model.model_fields}
        try:
            value = self.fn(**arguments)
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms(),
            )
        return ToolResult(ok=True, value=value, duration_ms=elapsed_ms())

    def spec(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": schema,
        }


def tool(
    *,
    name: str | None = None,
    version: str,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator: turn a plain typed function into a Tool.

    ``version`` is mandatory so that tool changes are visible in specs and traces.
    """

    def decorate(fn: Callable[..., Any]) -> Tool:
        return Tool(fn, name=name, version=version, description=description)

    return decorate


class ToolRegistry:
    """Holds tools by name; validates, executes, and traces every call."""

    def __init__(self, tracer: "Tracer | None" = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.tracer = tracer

    def register(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ValueError(f"tool {t.name!r} is already registered")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; registered: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def spec(self) -> list[dict[str, Any]]:
        return [self._tools[name].spec() for name in sorted(self._tools)]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        t = self.get(name)
        result = t(**kwargs)
        if self.tracer is not None:
            self.tracer.record(
                tool_name=t.name,
                version=t.version,
                arguments=kwargs,
                result=result,
            )
        return result
