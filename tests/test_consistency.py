"""Database, index and files stay in agreement through re-analysis, upgrades, uploads and conversions."""
import socket
from datetime import datetime, timedelta, timezone

import pytest

from src.research.agent import ResearchAgent
from src.research.gaps import GapEngine
from src.research.topic_ops import upgrade_to_full_text
from src.ingest import vectorstore as vs
from tests.conftest import FakeProvider
from tests.test_autopilot import NOW, activate, env  # noqa: F401  (fixture)
import src.research.content as content_mod
from tests.test_fulltext import pdf_bytes
from tests.test_library import make_pdf

REAL_IS_PUBLIC_HOST = content_mod._is_public_host  # before any test patches it


def _paper(store, sid, **extra):
    fields = {"title": "A study of LLM unit test generation", "selection_status": "selected",
              "raw_content": "LLM unit test generation. " * 200, "content_status": "full_text", **extra}
    return store.add_paper(sid, fields, assign_ref=True)


def test_reanalysis_retracts_exactly_the_papers_own_evidence(env):
    store = env["store"]
    sid = store.create_session("t")["id"]
    p2, p3 = _paper(store, sid), _paper(store, sid, title="Another study")
    gap = store.create_gap(sid, {"title": "Gap", "description": "d", "confidence": "low", "status": "open"})
    store.add_evidence(gap["id"], p3["id"], "P2's analysis says P3 lacks X", "supports", "stated", source_paper_id=p2["id"])
    store.add_evidence(gap["id"], p2["id"], "P3's analysis says P2 lacks X", "supports", "stated", source_paper_id=p3["id"])
    GapEngine(store, sid).retract_paper(p2["id"])                # P2 is analysed again
    left = [e["evidence"] for e in store.evidence_for_gap(gap["id"])]
    assert left == ["P3's analysis says P2 lacks X"]           # P3's own contribution survives


def test_reanalysis_drops_the_papers_old_synthesis_claims(env):
    store = env["store"]
    sid = store.create_session("LLM unit test generation")["id"]
    paper = _paper(store, sid)
    store.update_session(sid, synthesis={"overview": "o", "known_findings": [
        {"statement": "Old abstract-level claim", "paper_refs": [paper["ref"]], "paper_ids": [paper["id"]], "strength": "emerging"}]})
    ResearchAgent(store, sid, provider=FakeProvider()).process_paper(paper["id"])   # re-analysis of its full text
    statements = [f["statement"] for f in (store.get_session(sid)["synthesis"] or {}).get("known_findings", [])]
    assert "Old abstract-level claim" not in statements


def test_dns_failure_counts_as_unreachable(monkeypatch):
    content = content_mod
    monkeypatch.setattr(content, "_is_public_host", REAL_IS_PUBLIC_HOST)

    def no_dns(*a, **kw):
        raise socket.gaierror("Temporary failure in name resolution")
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    res = content.fetch_content({"title": "T", "url": "https://publisher.example.com/p", "pdf_url": "https://cdn.example.com/p.pdf"})
    assert res.unreachable                                        # so "no internet" is recognised, not "abstract only"


def test_html_full_text_reaches_the_library_after_republish(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = _paper(store, sid, content_status="abstract", raw_content="Only the abstract. " * 20, processing_status="done",
                   abstract="Only the abstract. " * 20)
    store.update_paper(paper["id"], processed_at=NOW)
    ap.step()                                                     # published with its abstract
    lib = store.get_paper(paper["id"])["library_paper_id"]
    assert (root / "Testing" / lib / "text.txt").read_text().startswith("Only the abstract")
    article = "Introduction. Method. Results. Discussion. Conclusion. The full HTML article text. " * 80
    upgrade_to_full_text(store, "Testing", store.get_paper(paper["id"]), article, None, source="html")
    for _ in range(3):
        ap.step()
    assert (root / "Testing" / lib / "text.txt").read_text().startswith("Introduction")
    assert any("full HTML article" in c.page_content for c in vs.get_topic_chunks("Testing", lib))
    assert not list((root / "Testing" / "_incoming").glob("*"))  # staged file consumed


def test_deleting_a_paper_removes_its_staged_full_text(env):
    from src.research.publish import incoming_pdf_path
    from src.research.topic_ops import delete_paper_everywhere
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = _paper(store, sid, content_status="abstract")
    upgrade_to_full_text(store, "Testing", store.get_paper(paper["id"]), "x" * 5000, pdf_bytes(), source="upload")
    assert incoming_pdf_path("Testing", paper["id"]).is_file()
    delete_paper_everywhere(store, "Testing", paper_id=paper["id"])
    assert not incoming_pdf_path("Testing", paper["id"]).exists()


def test_converting_a_one_off_session_never_takes_over_an_existing_topic(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    store, root = env["store"], env["root"]
    (root / "Old_search" / "my_upload").mkdir(parents=True)       # somebody's Library topic with the same name
    legacy = store.create_session("Old search", "")
    with TestClient(app) as c:
        r = c.post(f"/api/research/{legacy['id']}/search", json={})
    assert r.json()["topic_name"] == "Old_search_2"
    assert store.get_topic("Old_search") is None


def test_the_same_pdf_cannot_be_uploaded_twice(env, tmp_path):
    from fastapi.testclient import TestClient
    from src.api.server import app
    pdf = make_pdf(tmp_path / "a.pdf", ["A paper about toxicity classifiers and their evaluation."])
    with TestClient(app) as c:
        first = c.post("/api/library/upload", params={"topic": "Testing"}, files={"file": ("paper.pdf", pdf, "application/pdf")})
        again = c.post("/api/library/upload", params={"topic": "Testing"}, files={"file": ("renamed.pdf", pdf, "application/pdf")})
    assert first.status_code == 201 and again.status_code == 409 and "already in the topic" in again.json()["detail"]


def test_switching_topics_keeps_a_quota_pause(env):
    from src.api.autopilot import request_topic
    store = env["store"]
    until = datetime.now(timezone.utc) + timedelta(hours=5)
    store.update_autopilot(active_topic="Testing", state="paused", paused_until=until)
    assert request_topic("Other") == "active"
    assert store.get_autopilot()["paused_until"] is not None


def test_worker_heartbeat_and_retention(env):
    from src.api.server import _autopilot_health
    import src.api.server as server
    store = env["store"]
    server.AUTOPILOT_ENABLED = True
    try:
        store.heartbeat()
        assert _autopilot_health()[0] is True
        with __import__("src.db", fromlist=["connection"]).connection() as conn:
            conn.execute("UPDATE autopilot SET heartbeat_at = now() - interval '2 hours'")
        ok, msg = _autopilot_health()
        assert ok is False and "stuck" in msg
    finally:
        server.AUTOPILOT_ENABLED = False
    sid = store.create_session("t")["id"]
    store.add_event(sid, "info", "old")
    store.add_event(sid, "error", "old error")
    with __import__("src.db", fromlist=["connection"]).connection() as conn:
        conn.execute("UPDATE events SET created_at = now() - interval '200 days'")
    store.add_event(sid, "info", "new")
    assert store.prune()["events"] == 1                               # errors are kept for a year
    assert [e["message"] for e in store.list_events(sid)] == ["old error", "new"]


def test_long_papers_are_condensed_into_notes(monkeypatch):
    """prepare_content must really produce LLM notes (an error inside it silently falls back to truncation)."""
    from src.research import llm_tasks
    monkeypatch.setattr(llm_tasks, "MAX_CONTENT_CHUNKS", 3)

    class Notes:
        name, model, calls = "fake", "m", 0

        def complete(self, prompt, **kw):
            Notes.calls += 1
            return "- key finding with 0.91 F1 on the benchmark"
    text = ("Introduction. Method. Results. " * 400 + "\n\n") * 6
    out = llm_tasks.prepare_content("A long paper", text, budget=2000, provider=Notes())
    assert out.startswith("Condensed notes from the full text") and Notes.calls >= 2
