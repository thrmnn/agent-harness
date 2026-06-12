"""Eval cases for the knowledge-base example.

Run with:

    python -m agent_harness.evals examples/eval_cases.py
"""

from __future__ import annotations

from agent_harness import EvalCase, EvalSuite, contains, equals, predicate
from kb_search import build_registry

CASES = [
    EvalCase(
        name="lidar-query-finds-lidar-doc",
        tool="kb_search",
        input={"query": "lidar drone scans"},
        checks=[equals("hits.0.id", "doc-lidar")],
    ),
    EvalCase(
        name="hit-carries-title",
        tool="kb_search",
        input={"query": "retrieval quality"},
        checks=[contains("hits.0.title", "retrieval quality")],
    ),
    EvalCase(
        name="top-k-is-respected",
        tool="kb_search",
        input={"query": "tool agents calls", "top_k": 2},
        checks=[equals("count", 2)],
    ),
    EvalCase(
        name="unknown-terms-return-nothing",
        tool="kb_search",
        input={"query": "sourdough fermentation schedule"},
        checks=[equals("hits", []), equals("count", 0)],
    ),
    # Regression guard: a title match must outrank a body-only match. To see
    # the suite catch a regression, change TITLE_WEIGHT from 2 to 1 in
    # kb_search.py and rerun -- this case fails and the CLI exits 1.
    EvalCase(
        name="title-match-outranks-body-match",
        tool="kb_search",
        input={"query": "regression evals"},
        checks=[
            equals("hits.0.id", "doc-evals"),
            predicate(
                lambda hits: all(
                    hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)
                ),
                path="hits",
                label="scores_descending",
            ),
        ],
    ),
    EvalCase(
        name="fetch-returns-full-document",
        tool="kb_fetch",
        input={"doc_id": "doc-tracing"},
        checks=[
            equals("title", "Tracing agent tool calls in production"),
            contains("body", "input hash"),
        ],
    ),
]

SUITE = EvalSuite("kb-example", build_registry(), CASES)
