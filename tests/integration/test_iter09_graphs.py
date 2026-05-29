"""Iter 9 — LangGraph wiring: build all 3 graphs, inspect nodes."""
from __future__ import annotations


def test_build_ingest_graph():
    from workflows.ingest_workflow import build_ingest_graph

    g = build_ingest_graph()
    assert g is not None
    nodes = set(g.get_graph().nodes.keys())
    # Node names are renamed (e.g. mindmap_node) to avoid LangGraph
    # state-key collisions with the IngestState TypedDict.
    expected = {"scrape", "summarize", "tag_node", "link_node", "mindmap_node", "persist"}
    assert expected.issubset(nodes), f"missing: {expected - nodes}"


def test_build_chat_graph():
    from workflows.chat_workflow import build_chat_graph

    g = build_chat_graph()
    nodes = set(g.get_graph().nodes.keys())
    assert {"retrieve", "answer_node"}.issubset(nodes), f"got: {nodes}"


def test_build_review_graph():
    from workflows.review_workflow import build_review_graph

    g = build_review_graph()
    nodes = set(g.get_graph().nodes.keys())
    assert {"collect", "summarise", "persist"}.issubset(nodes), f"got: {nodes}"


def test_graph_singletons():
    from workflows.ingest_workflow import get_ingest_graph
    from workflows.chat_workflow import get_chat_graph
    from workflows.review_workflow import get_review_graph

    assert get_ingest_graph() is get_ingest_graph()
    assert get_chat_graph() is get_chat_graph()
    assert get_review_graph() is get_review_graph()
