"""Tests for the knowledge-graph feature (tools/graph_builder + /api/graph)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from models.schemas import GraphScope


# --------------------------------------------------------------------- helpers
def _row(
    id_: str,
    title: str,
    *,
    tags: str = "",
    source_type: str = "web",
    extra: dict | None = None,
    created_at: datetime | None = None,
):
    """Build a minimal stand-in for a Knowledge ORM row."""
    return SimpleNamespace(
        id=id_,
        title=title,
        tags=tags,
        source_type=source_type,
        source=f"https://example.com/{id_}",
        summary="",
        extra=extra or {},
        chunks_indexed=0,
        created_at=created_at or datetime.utcnow(),
    )


# --------------------------------------------------------------------- tag edges
def test_g1_split_tags_basic():
    from tools.graph_builder import _split_tags

    assert _split_tags("a, b , c") == ["a", "b", "c"]
    assert _split_tags("") == []
    assert _split_tags(None) == []


def test_g2_tag_edges_share_one_tag():
    from tools.graph_builder import _node_from_row, _tag_edges

    rows = [
        _row("k1", "A", tags="ai, ml"),
        _row("k2", "B", tags="ml, python"),
        _row("k3", "C", tags="rust"),
    ]
    nodes = [_node_from_row(r) for r in rows]
    edges = _tag_edges(nodes, min_overlap=1)
    pairs = {tuple(sorted((e.source, e.target))) for e in edges}
    # k1 and k2 share "ml"; k3 isolated
    assert ("k1", "k2") in pairs
    assert ("k1", "k3") not in pairs
    assert ("k2", "k3") not in pairs


def test_g3_tag_edges_min_overlap_filters_low_overlap():
    from tools.graph_builder import _node_from_row, _tag_edges

    rows = [
        _row("a", "A", tags="x, y, z"),
        _row("b", "B", tags="x"),         # 1 overlap (x)
        _row("c", "C", tags="x, y"),      # 2 overlap (x, y)
    ]
    nodes = [_node_from_row(r) for r in rows]
    edges = _tag_edges(nodes, min_overlap=2)
    pairs = {tuple(sorted((e.source, e.target))) for e in edges}
    assert ("a", "c") in pairs  # passes min_overlap=2
    assert ("a", "b") not in pairs  # only 1 overlap
    assert ("b", "c") not in pairs  # only 1 overlap


def test_g4_tag_edges_weight_equals_shared_count():
    from tools.graph_builder import _node_from_row, _tag_edges

    rows = [
        _row("a", "A", tags="x, y, z"),
        _row("b", "B", tags="x, y, z"),
    ]
    nodes = [_node_from_row(r) for r in rows]
    edges = _tag_edges(nodes)
    assert len(edges) == 1
    assert edges[0].weight == 3.0
    assert edges[0].edge_type == "tag"


def test_g5_tag_edges_case_insensitive():
    from tools.graph_builder import _node_from_row, _tag_edges

    rows = [
        _row("a", "A", tags="Python, AI"),
        _row("b", "B", tags="python, ai"),
    ]
    nodes = [_node_from_row(r) for r in rows]
    edges = _tag_edges(nodes)
    assert len(edges) == 1
    assert edges[0].weight == 2.0  # both tags match case-insensitively


# --------------------------------------------------------------------- wikilink edges
def test_g6_wikilink_edges_by_title():
    from tools.graph_builder import _wikilink_edges

    rows = [
        _row("k1", "Alpha", extra={"related": ["Beta"]}),
        _row("k2", "Beta", extra={"related": []}),
        _row("k3", "Gamma", extra={"related": ["Beta", "Unknown"]}),
    ]
    id_set = {r.id for r in rows}
    edges = _wikilink_edges(rows, id_set)
    edge_pairs = {(e.source, e.target) for e in edges}
    assert ("k1", "k2") in edge_pairs
    assert ("k3", "k2") in edge_pairs
    # Unknown title doesn't create an edge
    assert all(e.target in id_set for e in edges)


def test_g7_wikilink_edges_by_id():
    from tools.graph_builder import _wikilink_edges

    rows = [
        _row("k1", "Alpha", extra={"related": ["k2"]}),
        _row("k2", "Beta"),
    ]
    edges = _wikilink_edges(rows, {"k1", "k2"})
    assert any(e.source == "k1" and e.target == "k2" for e in edges)


def test_g8_wikilink_edges_skip_self_and_invalid():
    from tools.graph_builder import _wikilink_edges

    rows = [
        _row("k1", "Alpha", extra={"related": ["Alpha", "", None, "k1"]}),
        _row("k2", "Beta", extra={"related": "not-a-list"}),
    ]
    edges = _wikilink_edges(rows, {"k1", "k2"})
    # Self-link, empty, None, non-list all filtered out
    assert edges == []


# --------------------------------------------------------------------- node weights
def test_g9_apply_weights_degree_counting():
    from tools.graph_builder import _apply_weights
    from models.schemas import GraphEdge, GraphNode

    nodes = [GraphNode(id=str(i), title=f"n{i}", source_type="web") for i in range(4)]
    edges = [
        GraphEdge(source="0", target="1"),
        GraphEdge(source="0", target="2"),
        GraphEdge(source="0", target="3"),
        GraphEdge(source="1", target="2"),
    ]
    _apply_weights(nodes, edges)
    by_id = {n.id: n.weight for n in nodes}
    assert by_id["0"] == 3
    assert by_id["1"] == 2
    assert by_id["2"] == 2
    assert by_id["3"] == 1


def test_g10_apply_weights_isolated_node_min_one():
    from tools.graph_builder import _apply_weights
    from models.schemas import GraphNode

    nodes = [GraphNode(id="lone", title="x", source_type="web")]
    _apply_weights(nodes, [])
    assert nodes[0].weight == 1  # isolated nodes still have minimum weight 1


# --------------------------------------------------------------------- build_graph
def test_g11_build_graph_all_scope(monkeypatch):
    from tools import graph_builder as gb

    rows = [
        _row("k1", "A", tags="ml, ai"),
        _row("k2", "B", tags="ml"),
        _row("k3", "C", extra={"related": ["A"]}),
    ]

    async def fake_fetch(*, scope, limit, tag=None):
        return rows

    monkeypatch.setattr(gb, "_fetch_rows", fake_fetch)
    resp = asyncio.run(gb.build_graph(scope=GraphScope.ALL, limit=100))
    assert resp.stats["node_count"] == 3
    assert resp.stats["tag_edges"] >= 1
    assert resp.stats["wikilink_edges"] >= 1
    assert resp.stats["similarity_edges"] == 0


def test_g12_build_graph_recent_scope_limit_passed(monkeypatch):
    from tools import graph_builder as gb

    captured = {}

    async def fake_fetch(*, scope, limit, tag=None):
        captured["scope"] = scope
        captured["limit"] = limit
        captured["tag"] = tag
        return []

    monkeypatch.setattr(gb, "_fetch_rows", fake_fetch)
    asyncio.run(gb.build_graph(scope=GraphScope.RECENT, limit=42))
    assert captured == {"scope": GraphScope.RECENT, "limit": 42, "tag": None}


def test_g13_build_graph_tag_scope_passes_tag(monkeypatch):
    from tools import graph_builder as gb

    captured = {}

    async def fake_fetch(*, scope, limit, tag=None):
        captured["tag"] = tag
        return []

    monkeypatch.setattr(gb, "_fetch_rows", fake_fetch)
    asyncio.run(gb.build_graph(scope=GraphScope.TAG, limit=50, tag="ml"))
    assert captured["tag"] == "ml"


def test_g14_build_graph_empty_db(monkeypatch):
    from tools import graph_builder as gb

    async def fake_fetch(*, scope, limit, tag=None):
        return []

    monkeypatch.setattr(gb, "_fetch_rows", fake_fetch)
    resp = asyncio.run(gb.build_graph())
    assert resp.nodes == []
    assert resp.edges == []
    assert resp.stats["node_count"] == 0


def test_g15_build_graph_with_similarity(monkeypatch):
    from tools import graph_builder as gb

    rows = [_row(f"k{i}", f"Note {i}", tags="x") for i in range(3)]

    async def fake_fetch(*, scope, limit, tag=None):
        return rows

    async def fake_sim(nodes, *, threshold, top_k_per_node=5):
        from models.schemas import GraphEdge

        return [
            GraphEdge(source="k0", target="k1", weight=0.9, edge_type="similarity")
        ]

    monkeypatch.setattr(gb, "_fetch_rows", fake_fetch)
    monkeypatch.setattr(gb, "_similarity_edges", fake_sim)
    resp = asyncio.run(gb.build_graph(include_similarity=True))
    assert resp.stats["similarity_edges"] == 1
    sim_edges = [e for e in resp.edges if e.edge_type == "similarity"]
    assert sim_edges[0].weight == 0.9


def test_g16_similarity_edges_threshold_and_dedup(monkeypatch):
    from tools import graph_builder as gb
    from models.schemas import GraphNode

    nodes = [
        GraphNode(id="a", title="A", source_type="web"),
        GraphNode(id="b", title="B", source_type="web"),
        GraphNode(id="c", title="C", source_type="web"),
    ]

    class FakeStore:
        async def similarity_search(self, query, k=5):
            # Always return: high-score hit to b + low-score hit to c + self-hit + unknown
            return [
                {"knowledge_id": "b", "score": 0.9},
                {"knowledge_id": "c", "score": 0.5},  # below threshold
                {"knowledge_id": query[-1], "score": 1.0},  # might self-match
                {"knowledge_id": "ghost", "score": 0.95},  # not in id_set
                {"knowledge_id": "b", "score": 0.92},  # dup
            ]

    monkeypatch.setattr(gb, "get_vector_store", lambda: FakeStore(), raising=False)
    import core.vector_store as vs
    monkeypatch.setattr(vs, "get_vector_store", lambda: FakeStore())

    edges = asyncio.run(gb._similarity_edges(nodes, threshold=0.75))
    pairs = {tuple(sorted((e.source, e.target))) for e in edges}
    # Should only emit ab edge (one direction, deduped); below-threshold + self + ghost excluded
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs
    assert all(e.weight >= 0.75 for e in edges)


def test_g17_similarity_edges_search_failure_swallowed(monkeypatch):
    from tools import graph_builder as gb
    from models.schemas import GraphNode

    class FakeStore:
        async def similarity_search(self, query, k=5):
            raise RuntimeError("vector store down")

    import core.vector_store as vs
    monkeypatch.setattr(vs, "get_vector_store", lambda: FakeStore())

    nodes = [GraphNode(id="a", title="A", source_type="web")]
    edges = asyncio.run(gb._similarity_edges(nodes, threshold=0.5))
    assert edges == []


def test_g18_similarity_edges_no_vector_store(monkeypatch):
    from tools import graph_builder as gb
    from models.schemas import GraphNode

    import core.vector_store as vs

    def boom():
        raise RuntimeError("not configured")

    monkeypatch.setattr(vs, "get_vector_store", boom)
    edges = asyncio.run(
        gb._similarity_edges(
            [GraphNode(id="a", title="A", source_type="web")], threshold=0.5
        )
    )
    assert edges == []


# --------------------------------------------------------------------- _fetch_rows DB integration
def test_g19_fetch_rows_real_sqlite_all_scope(tmp_path, monkeypatch):
    """Round-trip through real SQLite using an in-memory async engine."""
    import asyncio as aio

    from core import config as cfg
    from models import database as db
    from tools.graph_builder import _fetch_rows

    # Use a temp sqlite file
    monkeypatch.setattr(cfg.settings, "database_url", f"sqlite+aiosqlite:///{tmp_path}/g.db", raising=False)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_session_factory", None)

    async def setup_and_query():
        await db.init_db()
        async with db.session_scope() as s:
            s.add(
                db.Knowledge(
                    id="x1",
                    title="X One",
                    source_type="web",
                    source="https://x",
                    summary="",
                    tags="a,b",
                    extra={},
                )
            )
            s.add(
                db.Knowledge(
                    id="x2",
                    title="X Two",
                    source_type="pdf",
                    source="/tmp/y.pdf",
                    summary="",
                    tags="b,c",
                    extra={},
                )
            )
        rows = await _fetch_rows(scope=GraphScope.ALL, limit=10)
        return rows

    rows = aio.run(setup_and_query())
    assert len(rows) == 2
    titles = sorted(r.title for r in rows)
    assert titles == ["X One", "X Two"]
    # Cleanup singletons
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_session_factory", None)


def test_g20_fetch_rows_tag_scope_with_filter(tmp_path, monkeypatch):
    import asyncio as aio

    from core import config as cfg
    from models import database as db
    from tools.graph_builder import _fetch_rows

    monkeypatch.setattr(cfg.settings, "database_url", f"sqlite+aiosqlite:///{tmp_path}/g2.db", raising=False)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_session_factory", None)

    async def setup_and_query():
        await db.init_db()
        async with db.session_scope() as s:
            s.add(
                db.Knowledge(
                    id="r1", title="R1", source_type="web", source="x",
                    summary="", tags="ml,ai", extra={},
                )
            )
            s.add(
                db.Knowledge(
                    id="r2", title="R2", source_type="web", source="y",
                    summary="", tags="rust", extra={},
                )
            )
        return await _fetch_rows(scope=GraphScope.TAG, tag="ml", limit=10)

    rows = aio.run(setup_and_query())
    ids = [r.id for r in rows]
    assert "r1" in ids and "r2" not in ids
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_session_factory", None)


# --------------------------------------------------------------------- API route
def test_g21_api_graph_route_returns_data(monkeypatch):
    """End-to-end test for GET /api/graph with build_graph stubbed."""
    from api import server as srv
    import api.routes.graph as graph_route
    from models.schemas import GraphEdge, GraphNode, GraphResponse

    fake_resp = GraphResponse(
        scope="all",
        nodes=[GraphNode(id="n1", title="N1", source_type="web", tags=["x"])],
        edges=[GraphEdge(source="n1", target="n1", weight=1.0, edge_type="tag")],
        stats={
            "node_count": 1,
            "edge_count": 1,
            "tag_edges": 1,
            "wikilink_edges": 0,
            "similarity_edges": 0,
            "scope": "all",
        },
    )

    async def fake_build(**kwargs):
        return fake_resp

    monkeypatch.setattr(graph_route, "build_graph", fake_build)

    with TestClient(srv.app) as c:
        r = c.get("/api/graph?scope=all&limit=50")
        assert r.status_code == 200
        body = r.json()
        assert body["stats"]["node_count"] == 1
        assert body["nodes"][0]["id"] == "n1"


def test_g22_api_graph_route_500_on_exception(monkeypatch):
    from api import server as srv
    import api.routes.graph as graph_route

    async def boom(**kwargs):
        raise RuntimeError("graph blew up")

    monkeypatch.setattr(graph_route, "build_graph", boom)
    with TestClient(srv.app) as c:
        r = c.get("/api/graph")
        assert r.status_code == 500
        assert "graph blew up" in r.json()["detail"]


def test_g23_api_graph_route_validates_limit():
    from api import server as srv

    with TestClient(srv.app) as c:
        r = c.get("/api/graph?limit=0")
        assert r.status_code == 422  # FastAPI validation


def test_g24_graph_page_html_served():
    """`/graph` should serve the D3.js page bundled in static/."""
    from api import server as srv

    with TestClient(srv.app) as c:
        r = c.get("/graph")
        assert r.status_code == 200
        assert "PKM Knowledge Graph" in r.text or "graph" in r.text.lower()


# --------------------------------------------------------------------- Schemas
def test_g25_graph_request_defaults_and_validation():
    from models.schemas import GraphRequest

    req = GraphRequest()
    assert req.scope == GraphScope.ALL
    assert req.limit == 200
    assert req.include_similarity is False

    with pytest.raises(Exception):
        GraphRequest(limit=0)  # ge=1
    with pytest.raises(Exception):
        GraphRequest(similarity_threshold=1.5)  # le=1.0
