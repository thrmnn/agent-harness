"""agent-harness: typed tools, readable traces, regression evals for LLM agents."""

from typing import TYPE_CHECKING, Any

from agent_harness.tools import Tool, ToolRegistry, ToolResult, tool
from agent_harness.trace import Tracer, read_trace

if TYPE_CHECKING:
    from agent_harness.evals import (
        CaseResult,
        Check,
        EvalCase,
        EvalReport,
        EvalSuite,
        contains,
        equals,
        predicate,
    )

__version__ = "0.1.0"

_EVALS_EXPORTS = (
    "CaseResult",
    "Check",
    "EvalCase",
    "EvalReport",
    "EvalSuite",
    "contains",
    "equals",
    "predicate",
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "Tracer",
    "read_trace",
    "tool",
    "__version__",
    *_EVALS_EXPORTS,
]


# evals is imported lazily so that `python -m agent_harness.evals` executes the
# module exactly once; an eager import here would trigger runpy's
# double-import RuntimeWarning.
def __getattr__(name: str) -> Any:
    if name in _EVALS_EXPORTS:
        from agent_harness import evals

        return getattr(evals, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
