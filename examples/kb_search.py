"""Example: a deterministic knowledge-base search tool, no network, no LLM.

Run it directly to see tool calls, captured failures, and the trace:

    python examples/kb_search.py
"""

from __future__ import annotations

import re
from typing import Any

from agent_harness import ToolRegistry, Tracer, read_trace, tool

CORPUS: list[dict[str, str]] = [
    {
        "id": "doc-lidar",
        "title": "Aligning LiDAR scans across drone flights",
        "body": "Point cloud registration drifts when GPS is degraded. We anchor "
        "scans to building footprints and refine with ICP.",
    },
    {
        "id": "doc-retrieval",
        "title": "Measuring retrieval quality before shipping",
        "body": "Run evals over a frozen query set: a ranking regression "
        "surfaces as failing evals in CI instead of as user reports.",
    },
    {
        "id": "doc-tracing",
        "title": "Tracing agent tool calls in production",
        "body": "One JSON line per call: tool, version, input hash, outcome, "
        "latency. Enough to reconstruct most incidents.",
    },
    {
        "id": "doc-grasping",
        "title": "Grasp pose estimation from a single depth image",
        "body": "A small network over depth patches beats analytic samplers on "
        "cluttered bins, and degrades predictably.",
    },
    {
        "id": "doc-evals",
        "title": "Regression evals for tool-using agents",
        "body": "Pin known-good outputs as eval cases. A prompt or tool change "
        "that breaks one fails CI before it reaches users.",
    },
]

TITLE_WEIGHT = 2


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@tool(version="1.2.0")
def kb_search(query: str, top_k: int = 3) -> dict[str, Any]:
    """Search the knowledge base; returns ranked hits with scores."""
    terms = set(_tokens(query))
    scored = []
    for doc in CORPUS:
        title_hits = sum(1 for t in _tokens(doc["title"]) if t in terms)
        body_hits = sum(1 for t in _tokens(doc["body"]) if t in terms)
        score = TITLE_WEIGHT * title_hits + body_hits
        if score > 0:
            scored.append({"id": doc["id"], "title": doc["title"], "score": score})
    scored.sort(key=lambda hit: (-hit["score"], hit["id"]))
    hits = scored[:top_k]
    return {"hits": hits, "count": len(hits)}


@tool(version="1.0.0")
def kb_fetch(doc_id: str) -> dict[str, str]:
    """Fetch a single document by id."""
    for doc in CORPUS:
        if doc["id"] == doc_id:
            return doc
    raise KeyError(f"no document with id {doc_id!r}")


def build_registry(tracer: Tracer | None = None) -> ToolRegistry:
    registry = ToolRegistry(tracer=tracer)
    registry.register(kb_search)
    registry.register(kb_fetch)
    return registry


def main() -> None:
    trace_path = "trace.jsonl"
    registry = build_registry(Tracer(trace_path))

    print("tool specs exposed to the LLM API:")
    for spec in registry.spec():
        print(f"  {spec['name']} v{spec['version']}: {spec['description']}")

    print("\nsearch 'lidar point cloud registration':")
    result = registry.call("kb_search", query="lidar point cloud registration", top_k=2)
    for hit in result.value["hits"]:
        print(f"  {hit['score']:>2}  {hit['id']}  {hit['title']}")

    print("\nbad input is captured, not raised:")
    bad = registry.call("kb_search", query="lidar", top_k="three")
    print(f"  ok={bad.ok}  error={bad.error}")

    print("\ntool exceptions are captured, not raised:")
    missing = registry.call("kb_fetch", doc_id="doc-999")
    print(f"  ok={missing.ok}  error={missing.error}")

    print(f"\ntrace written to {trace_path}:")
    for event in read_trace(trace_path)[-3:]:
        print(f"  {event['tool']} v{event['version']}  ok={event['ok']}  {event['duration_ms']}ms")


if __name__ == "__main__":
    main()
