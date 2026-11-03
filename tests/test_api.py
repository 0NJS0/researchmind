"""Research API: starting research creates a Library topic that the autopilot researches continuously."""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import PAPERS_DIR
from src.ingest import embeddings as emb_mod
from src.research.agent import ResearchAgent
from src.research.autopilot import Autopilot
from src.research.enrich import OpenAlexClient
from src.research.jobs import jobs
from src.research.searxng import SearXNGClient
from tests.conftest import FakeProvider, make_searx_transport, not_found_transport, truncate
from tests.test_library import FakeEmbeddings

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def _factory(store_, session_id, cancel_event=None):
    return ResearchAgent(store_, session_id, provider=FakeProvider(), cancel_event=cancel_event,
                         searx=SearXNGClient("http://searx.test", transport=make_searx_transport()),
                         fetch_transport=not_found_transport())


@pytest.fixture
def worker(store):
    """The autopilot, driven step by step (the background thread is disabled in tests)."""
    root = Path(PAPERS_DIR)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    truncate("chunks", "embedding_meta")
    emb_mod.set_embeddings(FakeEmbeddings())
    ap = Autopilot(store, agent_factory=_factory, quota_check=lambda: 900,
                   enricher=OpenAlexClient(enabled=False), clock=lambda: NOW)
    yield ap
    emb_mod.set_embeddings(None)
    for sid in list(jobs.claimed()):
        jobs.release(sid)
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def client(store, worker, monkeypatch):
    monkeypatch.setattr(jobs, "agent_factory", _factory)
    from src.api.server import app
    with TestClient(app) as c:
        yield c


def run_until_idle(worker, steps=30):
    """Step the autopilot until its queue is empty and it waits for the next search."""
    for _ in range(steps):
        worker.step()
        if worker.store.get_autopilot()["state"] == "waiting":
            return
    raise AssertionError("autopilot did not settle")


def test_new_research_creates_a_topic_on_autopilot(client, worker, store):
    r = client.post("/api/research", json={"topic": "  LLMs for   software testing ", "description": "empirical only"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["topic"] == "LLMs for software testing" and body["topic_name"] == "LLMs_for_software_testing"
    assert body["autopilot_status"] == "active" and body["autopilot_role"] == "active"
    assert (Path(PAPERS_DIR) / "LLMs_for_software_testing").is_dir()
    ap = store.get_autopilot()
    assert ap["active_topic"] == "LLMs_for_software_testing" and ap["enabled"]
    assert store.get_topic("LLMs_for_software_testing")["description"] == "empirical only"
    assert store.get_topic("LLMs_for_software_testing")["session_id"] == body["id"]

    run_until_idle(worker)
    sid = body["id"]
    session = client.get(f"/api/research/{sid}").json()
    assert session["counts"]["papers"] >= 1 and session["counts"]["summarized"] >= 1

    papers = client.get(f"/api/research/{sid}/papers").json()["papers"]
    assert papers and "raw_content" not in papers[0]
    detail = client.get(f"/api/papers/{papers[0]['id']}").json()
    assert detail["summary"]["sections"]["research_problem"] and "raw_content" not in detail["paper"]
    # ...and the papers are in the Library of the new topic (so Ask, Reports and the Papers page see them).
    library = client.get("/api/library/papers", params={"topic": "LLMs_for_software_testing"}).json()["papers"]
    assert library and all(p["origin"] == "agent" for p in library)

    listing = client.get("/api/research").json()["sessions"]
    assert listing[0]["id"] == sid and listing[0]["topic_name"] == "LLMs_for_software_testing"
    assert listing[0]["autopilot_role"] == "active"


def test_second_topic_waits_for_the_current_paper(client, store):
    client.post("/api/research", json={"topic": "LLMs for software testing"})
    store.update_autopilot(state="processing")                     # a paper is being processed
    r = client.post("/api/research", json={"topic": "Mutation testing"})
    assert r.json()["autopilot_status"] == "pending" and r.json()["autopilot_role"] == "next"
    ap = store.get_autopilot()
    assert ap["active_topic"] == "LLMs_for_software_testing" and ap["requested_topic"] == "Mutation_testing"


def test_research_on_an_existing_library_topic_reuses_it(client, store):
    client.post("/api/library/topics", json={"name": "My Topic"})
    r = client.post("/api/research", json={"topic": "My Topic"})
    assert r.status_code == 201 and r.json()["topic_name"] == "My_Topic"
    again = client.post("/api/research", json={"topic": "My Topic"})
    assert again.json()["id"] == r.json()["id"]                    # one topic, one research


def test_continue_research_moves_a_one_off_session_to_the_autopilot(client, store):
    legacy = store.create_session("Old one-off search", "")           # created before the merge
    r = client.post(f"/api/research/{legacy['id']}/search", json={})
    assert r.status_code == 202 and r.json()["topic_name"] == "Old_one-off_search"
    assert store.get_autopilot()["active_topic"] == "Old_one-off_search"
    assert store.get_topic("Old_one-off_search")["session_id"] == legacy["id"]
    assert (Path(PAPERS_DIR) / "Old_one-off_search").is_dir()


def test_added_paper_is_queued_for_the_autopilot(client, worker, store):
    sid = client.post("/api/research", json={"topic": "LLMs for software testing"}).json()["id"]
    r = client.post(f"/api/research/{sid}/papers", json={
        "title": "Mutation Testing with Large Language Models", "url": "https://example.org/mutation",
        "abstract": "We apply large language models to mutation testing of software. " * 5})
    assert r.status_code == 201 and r.json()["queued"] is True and r.json()["processing"] is False
    assert not jobs.is_running(sid) or sid in jobs.claimed()          # no separate manual job was started
    run_until_idle(worker)
    paper = client.get(f"/api/papers/{r.json()['paper']['id']}").json()
    assert paper["paper"]["processing_status"] == "done" and paper["summary"]["analysis_level"] == "abstract"
    dup = client.post(f"/api/research/{sid}/papers", json={"url": "https://example.org/mutation"})
    assert dup.json()["status"] == "duplicate"


def test_event_stream_ends_when_idle(client, worker):
    sid = client.post("/api/research", json={"topic": "LLMs for software testing"}).json()["id"]
    run_until_idle(worker)
    worker._release()                                              # the stream stays open while the autopilot works
    with client.stream("GET", f"/api/research/{sid}/events/stream") as resp:
        body = "".join(resp.iter_text())
    assert "event: activity" in body and "event: status" in body


def test_validation_and_not_found(client):
    assert client.post("/api/research", json={"topic": "ab"}).status_code == 422
    assert client.post("/api/research", json={"topic": "x" * 301}).status_code == 422
    assert client.get("/api/research/not-an-id").status_code == 404
    assert client.get("/api/research/" + "0" * 32).status_code == 404
    r = client.get("/api/papers/../../etc/passwd")  # normalised away from /api: never serves the file
    assert r.status_code in (404, 200) and "root:" not in r.text
    sid = client.post("/api/research", json={"topic": "valid topic"}).json()["id"]
    assert client.post(f"/api/research/{sid}/papers", json={"url": "javascript:alert(1)"}).status_code == 422
    assert client.post(f"/api/research/{sid}/papers", json={}).status_code == 422
    assert client.post(f"/api/research/{sid}/search", json={"queries": ["q" * 201]}).status_code == 422
    assert client.get(f"/api/research/{sid}/papers", params={"selection": "bogus"}).status_code == 422


def test_topic_research_is_deleted_with_its_topic(client, store):
    sid = client.post("/api/research", json={"topic": "valid topic"}).json()["id"]
    assert client.delete(f"/api/research/{sid}").status_code == 409   # it belongs to a topic
    client.post("/api/autopilot/pause")
    client.post("/api/library/topics", json={"name": "Other"})
    client.post("/api/autopilot/topic", json={"topic": "Other"})     # move the autopilot off it first
    assert client.delete("/api/library/topics/valid_topic").status_code == 200
    assert client.get(f"/api/research/{sid}").status_code == 404
    legacy = store.create_session("A one-off search", "")
    assert client.delete(f"/api/research/{legacy['id']}").json()["status"] == "deleted"
