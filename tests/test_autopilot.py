"""The continuous autopilot: one task per step, best paper first, topic switches between tasks,
papers published to the Library and the living report, quota-aware pausing."""
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document

from src.config import AUTOPILOT_RESEARCH_INTERVAL_HOURS, PAPERS_DIR
from src.ingest import embeddings as emb_mod
from src.ingest import vectorstore as vs
from src.llm.llm import QuotaExhausted
from src.paths import analysis_path, living_report_path
from src.research.agent import ResearchAgent
from src.research.autopilot import Autopilot, next_quota_reset
from src.research.enrich import OpenAlexClient, enrich_fields, parse_work
from src.research.jobs import jobs
from src.research.ranking import priority
from src.research.searxng import SearXNGClient
from tests.conftest import FakeProvider, make_searx_transport, not_found_transport, truncate
from tests.test_library import FakeEmbeddings

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)

# OpenAlex answers keyed by DOI / title: the peer-reviewed, highly cited 2024 paper must be processed first.
WORKS = {
    "10.1145/1234567.7654321": {"doi": "https://doi.org/10.1145/1234567.7654321", "title": "ChatGPT versus Search-Based Software Testing Tools",
                                "publication_year": 2024, "cited_by_count": 240, "type": "article",
                                "primary_location": {"source": {"display_name": "ICSE", "type": "conference"}}},
    "my upload": {"title": "My Upload", "publication_year": 2025, "cited_by_count": 12,
                  "type": "article", "doi": "https://doi.org/10.1000/mine",
                  "authorships": [{"author": {"display_name": "Najmus Saquib"}}],
                  "primary_location": {"source": {"display_name": "TOSEM", "type": "journal"}}},
    "Large Language Models for Unit Test Generation: An Empirical Study": {
        "title": "Large Language Models for Unit Test Generation: An Empirical Study", "publication_year": 2023,
        "cited_by_count": 3, "type": "preprint", "primary_location": {"source": {"display_name": "arXiv", "type": "repository"}}},
}


def openalex_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/works/doi:"):
            work = WORKS.get(path.removeprefix("/works/doi:"))
            return httpx.Response(200, json=work) if work else httpx.Response(404)
        if path == "/works":
            q = request.url.params.get("search", "")
            work = WORKS.get(q)
            return httpx.Response(200, json={"results": [work] if work else []})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


@pytest.fixture
def env(store, monkeypatch):
    root = Path(PAPERS_DIR)
    shutil.rmtree(root, ignore_errors=True)
    (root / "Testing").mkdir(parents=True)
    (root / "Other").mkdir(parents=True)
    truncate("chunks", "embedding_meta")
    emb_mod.set_embeddings(FakeEmbeddings())
    for sid in list(jobs.claimed()):
        jobs.release(sid)
    providers = []

    def agent_factory(store_, session_id, cancel_event=None):
        provider = FakeProvider()
        providers.append(provider)
        return ResearchAgent(store_, session_id, provider=provider, cancel_event=cancel_event,
                             searx=SearXNGClient("http://searx.test", transport=make_searx_transport()),
                             fetch_transport=not_found_transport())

    quota = {"remaining": 900}
    ap = Autopilot(store, agent_factory=agent_factory, quota_check=lambda: quota["remaining"],
                   enricher=OpenAlexClient(transport=openalex_transport(), enabled=True), clock=lambda: NOW)
    yield {"ap": ap, "store": store, "root": root, "quota": quota, "providers": providers}
    emb_mod.set_embeddings(None)
    for sid in list(jobs.claimed()):
        jobs.release(sid)
    shutil.rmtree(root, ignore_errors=True)


def upload(root: Path, topic: str, paper_id: str, text: str):
    (root / topic / paper_id).mkdir(parents=True, exist_ok=True)
    vs.add_documents([Document(page_content=text, metadata={"paper_id": paper_id, "topic": topic,
                                                            "source": f"{paper_id}.pdf", "page": 1})])


def activate(store, topic):
    store.update_autopilot(active_topic=topic, enabled=True)


def session_of(store, topic):
    return store.get_topic(topic)["session_id"]


# ── ranking & enrichment ──

def test_priority_prefers_recent_well_cited_peer_reviewed():
    today = date(2026, 9, 1)
    new_strong = {"relevance_score": 0.8, "year": 2025, "citation_count": 120, "peer_reviewed": True, "venue": "ACL"}
    old_weak = {"relevance_score": 0.8, "year": 2015, "citation_count": 4, "peer_reviewed": False}
    classic = {"relevance_score": 0.8, "year": 2017, "citation_count": 90000, "peer_reviewed": True, "venue": "NeurIPS"}
    s_new, detail = priority(new_strong, today)
    assert s_new > priority(old_weak, today)[0]
    assert priority(classic, today)[0] > priority(old_weak, today)[0]   # impact keeps landmark papers competitive
    assert set(detail) == {"relevance", "recency", "impact", "venue"}
    assert priority({**old_weak, "added_by": "user"}, today)[0] > s_new  # the user's own uploads go first


def test_openalex_enrichment_by_doi_and_title():
    client = OpenAlexClient(transport=openalex_transport(), enabled=True)
    fields = enrich_fields({"doi": "10.1145/1234567.7654321", "title": "x"}, client)
    assert fields["citation_count"] == 240 and fields["peer_reviewed"] is True and fields["year"] == 2024
    fields = enrich_fields({"title": "Large Language Models for Unit Test Generation: An Empirical Study"}, client)
    assert fields["peer_reviewed"] is False and fields["publication_type"] == "preprint"
    assert enrich_fields({"title": "A completely different paper about protein folding"}, client) == {}
    assert parse_work({"type": "article", "primary_location": {"source": {"display_name": "Nature", "type": "journal"}}}).peer_reviewed


# ── the loop ──

def test_continuous_loop_one_paper_per_step_best_first(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "my_upload", "We study LLM-based unit test generation on Java projects. " * 20)
    activate(store, "Testing")

    # Step 1: the user's upload is seeded and processed first; nothing else is processed.
    assert ap.step() == 0
    sid = session_of(store, "Testing")
    assert store.count_processed(sid) == 1
    mine = store.paper_by_library_id(sid, "my_upload")
    assert mine["processing_status"] == "done" and mine["added_by"] == "user"
    assert analysis_path("Testing", "my_upload").exists()
    assert living_report_path("Testing").exists()
    assert jobs.is_running(sid)  # the autopilot owns the session: manual jobs are refused
    assert not jobs.start(store, sid, "process")

    # Step 2: queue empty → search, screen, enrich, rank. Nothing processed in a search step.
    ap.step()
    queue = store.queued_papers(sid, ("pending",), limit=10)
    assert len(queue) == 2 and store.count_processed(sid) == 1
    assert queue[0]["title"] == "ChatGPT versus Search-Based Software Testing Tools"   # peer reviewed, 240 citations
    assert queue[0]["priority_score"] > queue[1]["priority_score"]
    assert queue[0]["priority_detail"]["impact"] > queue[1]["priority_detail"]["impact"]

    # Step 3: exactly the best paper is processed and published to the Library.
    ap.step()
    assert store.count_processed(sid) == 2
    best = store.get_paper(queue[0]["id"])
    assert best["processing_status"] == "done" and best["library_paper_id"]
    lib = {p["paper_id"]: p for p in vs.list_papers("Testing")}
    assert best["library_paper_id"] in lib                               # indexed → searchable from Ask
    meta = json.loads((root / "Testing" / best["library_paper_id"] / "paper.json").read_text())
    assert meta["origin"] == "agent" and meta["citation_count"] == 240
    assert store.get_paper(queue[1]["id"])["processing_status"] == "pending"

    # The living report now lists the new paper, its gaps and the LLM narrative (every 2 papers in tests).
    report = living_report_path("Testing").read_text()
    assert "ChatGPT versus Search-Based Software Testing Tools" in report
    assert "## Research gaps" in report and "## Executive summary" in report

    # Keep going: the expansion search finds just one more paper, so the next round waits a few hours
    # (a trickle of results doesn't justify an immediate new screening round).
    from src.research.autopilot import DRIP_WAIT
    for _ in range(8):
        ap.step()
    state = store.get_autopilot()
    assert state["state"] == "waiting"
    assert datetime.fromisoformat(state["next_search_at"]) == NOW + DRIP_WAIT
    titles = {p["title"] for p in store.list_papers(sid, selection="selected")}
    assert "Test Oracle Generation with Large Language Models" in titles
    assert all(p["processing_status"] == "done" for p in store.list_papers(sid, selection="selected"))
    # While waiting, steps do nothing and sleep.
    assert ap.step() > 0 and store.get_autopilot()["state"] == "waiting"


def test_topic_switch_waits_for_the_current_paper(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "a_upload", "LLM unit test generation study. " * 30)
    upload(root, "Other", "b_upload", "LLM test oracle study. " * 30)
    activate(store, "Testing")

    class SwitchDuringSummary(FakeProvider):
        def complete(self, prompt, **kw):
            if self._kind(prompt) == "summary":
                store.update_autopilot(requested_topic="Other")   # user switches mid-paper
            return super().complete(prompt, **kw)

    original = ap.agent_factory
    ap.agent_factory = lambda s, sid, cancel_event=None: ResearchAgent(
        s, sid, provider=SwitchDuringSummary(), searx=SearXNGClient("http://searx.test", transport=make_searx_transport()),
        fetch_transport=not_found_transport())
    ap.step()
    state = store.get_autopilot()
    assert state["active_topic"] == "Testing" and state["requested_topic"] == "Other"
    first = store.paper_by_library_id(session_of(store, "Testing"), "a_upload")
    assert first["processing_status"] == "done"          # the paper in progress was finished

    ap.agent_factory = original
    ap.step()                                             # next task: switch, then work on the new topic
    state = store.get_autopilot()
    assert state["active_topic"] == "Other" and state["requested_topic"] is None
    other_sid = session_of(store, "Other")
    assert store.paper_by_library_id(other_sid, "b_upload")["processing_status"] == "done"
    assert not jobs.is_running(session_of(store, "Testing"))  # the old session is released


def test_pauses_when_free_quota_is_low(env):
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    env["quota"]["remaining"] = 100
    ap.step()
    state = store.get_autopilot()
    assert state["state"] == "paused"
    # The /key figure is re-read every half hour (a late reset must not cost a whole day); the UI shows the reset.
    assert datetime.fromisoformat(state["paused_until"]) == NOW + timedelta(minutes=30)
    assert next_quota_reset(NOW).strftime("%H:%M") in state["state_message"]
    assert store.get_topic("Testing") is None             # no work started


def test_quota_exhausted_mid_paper_keeps_it_queued(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "q_upload", "LLM unit test generation study. " * 30)
    activate(store, "Testing")

    class Quota(FakeProvider):
        def complete(self, prompt, **kw):
            if self._kind(prompt) == "summary":
                raise QuotaExhausted("free-models-per-day")
            return super().complete(prompt, **kw)

    ap.agent_factory = lambda s, sid, cancel_event=None: ResearchAgent(s, sid, provider=Quota())
    ap.step()
    sid = session_of(store, "Testing")
    assert store.paper_by_library_id(sid, "q_upload")["processing_status"] == "pending"
    assert store.get_autopilot()["state"] == "paused"


def test_disabled_and_no_topic_states(env):
    ap, store = env["ap"], env["store"]
    assert ap.step() > 0 and store.get_autopilot()["state"] == "idle"
    activate(store, "Testing")
    store.update_autopilot(enabled=False)
    assert ap.step() > 0 and store.get_autopilot()["state"] == "stopped"


# ── API ──

def test_autopilot_api_topic_switching(env, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.autopilot as api_mod
    monkeypatch.setattr(api_mod, "free_requests_remaining", lambda: 800)
    from src.api.server import app
    store = env["store"]
    with TestClient(app) as c:
        r = c.post("/api/autopilot/topic", json={"topic": "Testing", "focus": "only empirical studies"})
        assert r.status_code == 200 and r.json()["status"] == "active" and r.json()["active_topic"] == "Testing"
        assert store.get_topic("Testing")["description"] == "only empirical studies"
        store.update_autopilot(state="processing")          # a task is in progress
        r = c.post("/api/autopilot/topic", json={"topic": "Other"})
        assert r.json()["status"] == "pending" and r.json()["requested_topic"] == "Other"
        assert r.json()["active_topic"] == "Testing"
        assert c.post("/api/autopilot/topic", json={"topic": "Missing"}).status_code == 404
        assert c.post("/api/autopilot/pause").json()["enabled"] is False
        assert c.post("/api/autopilot/resume").json()["enabled"] is True


def test_deleting_a_library_paper_removes_it_from_the_research(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "gone_upload", "LLM unit test generation study. " * 30)
    activate(store, "Testing")
    ap.step()
    sid = session_of(store, "Testing")
    assert store.list_gaps(sid)
    with TestClient(app) as c:
        assert c.delete("/api/library/papers/Testing/gone_upload").status_code == 200
    assert not [p for p in store.list_papers(sid, selection=None) if p["title"] == "gone upload"]  # deleted
    assert store.list_gaps(sid) == []                     # its only evidence was retracted
    ap.step()
    assert not store.paper_by_library_id(sid, "gone_upload")   # not re-seeded


def test_preprint_and_published_versions_keep_only_the_published_one(env):
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    title = "Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph"
    pre = store.add_paper(sid, {"title": title, "doi": "10.48550/arxiv.2412.15268", "selection_status": "selected",
                                "peer_reviewed": False, "citation_count": 1}, assign_ref=True)
    pub = store.add_paper(sid, {"title": title, "doi": "10.18653/v1/2025.findings-acl.1270", "selection_status": "selected",
                                "peer_reviewed": True, "citation_count": 5}, assign_ref=True)
    kept = ap._drop_duplicate_versions(sid, [pre["id"], pub["id"]])
    assert kept == [pub["id"]]
    assert store.get_paper(pre["id"])["selection_status"] == "rejected"
    assert "Another version" in store.get_paper(pre["id"])["relevance_reason"]


def test_topic_papers_endpoint_merges_research_and_library(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "mine_upload", "LLM unit test generation study. " * 30)
    activate(store, "Testing")
    ap.step()          # process the upload
    ap.step()          # search → queue discovered papers
    ap.step()          # process the best discovered paper
    with TestClient(app) as c:
        body = c.get("/api/library/topics/Testing/papers").json()
        rows = {p["title"]: p for p in body["papers"]}
        assert rows["mine upload"]["origin"] == "upload" and rows["mine upload"]["status"] == "processed"
        best = rows["ChatGPT versus Search-Based Software Testing Tools"]
        assert best["status"] == "processed" and best["citation_count"] == 240 and best["peer_reviewed"] is True
        assert best["url"] and best["library_paper_id"] and best["has_analysis"] and best["gap_count"] >= 1
        assert any(p["status"] == "queued" for p in body["papers"])
        assert all(p["status"] != "rejected" for p in body["papers"])
        with_rejected = c.get("/api/library/topics/Testing/papers", params={"include_rejected": True}).json()
        assert any(p["status"] == "rejected" for p in with_rejected["papers"])
        # A topic that was never researched still lists its Library papers.
        upload(root, "Other", "solo", "Something about testing. " * 20)
        other = c.get("/api/library/topics/Other/papers").json()
        assert [p["status"] for p in other["papers"]] == ["library"]
        assert c.get("/api/library/topics/Missing/papers").status_code == 404


def test_title_matching_accepts_shortened_file_names_only():
    from src.research.enrich import titles_match
    real = "Holistic Evaluation of Large Language Models: Assessing Robustness, Accuracy, and Toxicity for Real-World Applications"
    assert titles_match("Holistic Evaluation of Large Language Models Robustness Accuracy and Toxicity", real)
    assert titles_match("When LLMs Meet Acoustic Landmarks Speech for Depression Detection",
                        "When LLMs Meets Acoustic Landmarks: An Efficient Approach to Integrate Speech into Large Language "
                        "Models for Depression Detection")
    assert not titles_match("Toxicity detection with large language models", "Toxicity detection with large vision models")
    assert not titles_match("LLM toxicity", "A survey on LLM toxicity detection methods and benchmarks")



def _run_until_two_processed(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "keep_upload", "LLM unit test generation study. " * 30)
    activate(store, "Testing")
    ap.step(); ap.step(); ap.step()        # upload processed, search, best discovered paper processed
    sid = session_of(store, "Testing")
    best = next(p for p in store.list_papers(sid) if p["title"] == "ChatGPT versus Search-Based Software Testing Tools")
    return ap, store, root, sid, best


def test_delete_paper_everywhere(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store, root, sid, best = _run_until_two_processed(env)
    lib = best["library_paper_id"]
    assert (root / "Testing" / lib).is_dir() and vs.paper_exists(lib, "Testing")
    assert best["ref"] in living_report_path("Testing").read_text()
    with TestClient(app) as c:
        r = c.delete("/api/library/topics/Testing/papers", params={"paper_id": best["id"]})
        assert r.status_code == 200, r.text
        assert r.json()["database"] and r.json()["files"] and r.json()["chunks"] > 0
        assert c.delete("/api/library/topics/Testing/papers", params={"paper_id": best["id"]}).status_code == 404
        assert c.delete("/api/library/topics/Testing/papers").status_code == 400
    assert store.get_paper(best["id"]) is None and store.get_summary(best["id"]) is None
    assert not (root / "Testing" / lib).exists() and not vs.paper_exists(lib, "Testing")
    for g in store.list_gaps(sid):
        assert best["ref"] not in {p["ref"] for p in g["related_papers"]}
    synthesis = store.get_session(sid)["synthesis"] or {}
    assert all(best["ref"] not in it["paper_refs"] for k in ("known_findings", "common_methods")
               for it in synthesis.get(k) or [])
    assert "ChatGPT versus Search-Based" not in living_report_path("Testing").read_text()

    # A new search round must not bring the deleted paper back.
    store.update_autopilot(next_search_at=None)
    for _ in range(4):
        ap.step()
    assert not [p for p in store.list_papers(sid, selection=None) if p["title"] == best["title"]]


def test_cannot_delete_the_paper_being_processed(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store, root, sid, best = _run_until_two_processed(env)
    queued = store.queued_papers(sid, ("pending",), limit=1)[0]
    with TestClient(app) as c:
        store.update_autopilot(current_paper_id=queued["id"])  # after startup, which clears stale state
        assert c.delete("/api/library/topics/Testing/papers", params={"paper_id": queued["id"]}).status_code == 409


def test_pdf_download_and_topic_export(env):
    import io
    import zipfile
    import fitz
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.research.publish import publish_to_library
    ap, store, root, sid, best = _run_until_two_processed(env)
    doc = fitz.open(); doc.new_page().insert_text((72, 72), "A stored paper PDF about LLM testing."); pdf = doc.tobytes(); doc.close()
    paper = store.queued_papers(sid, ("pending",), limit=1)[0]
    full = store.get_paper(paper["id"])
    lib = publish_to_library("Testing", full, pdf)
    store.update_paper(paper["id"], library_paper_id=lib)
    with TestClient(app) as c:
        rows = {p["library_paper_id"]: p for p in c.get("/api/library/topics/Testing/papers").json()["papers"]}
        assert rows[lib]["has_pdf"] is True
        r = c.get(f"/api/library/topics/Testing/papers/{lib}/pdf")
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
        assert "attachment" in r.headers["content-disposition"]
        assert c.get("/api/library/topics/Testing/papers/keep_upload/pdf").status_code == 404
        assert c.get("/api/library/topics/Testing/papers/..%2F..%2Fetc/pdf").status_code in (400, 404, 405)

        r = c.get("/api/library/topics/Testing/export")
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
        for expected in ("Testing/README.md", "Testing/papers.csv", "Testing/papers.json", "Testing/gaps.md",
                         "Testing/gaps.json", "Testing/synthesis.md", "Testing/reports/living_report.md"):
            assert expected in names, (expected, sorted(names))
        assert any(n.startswith("Testing/analyses/") for n in names)
        assert any(n.startswith("Testing/pdfs/") and n.endswith(".pdf") for n in names)
        csv_text = zipfile.ZipFile(io.BytesIO(r.content)).read("Testing/papers.csv").decode()
        assert "ChatGPT versus Search-Based Software Testing Tools" in csv_text
        light = c.get("/api/library/topics/Testing/export", params={"include_pdfs": False})
        assert not any(n.startswith("Testing/pdfs/") for n in zipfile.ZipFile(io.BytesIO(light.content)).namelist())


def test_pdf_is_downloaded_at_publish_when_content_was_fetched_earlier(env):
    import fitz
    ap, store, root = env["ap"], env["store"], env["root"]
    doc = fitz.open(); doc.new_page().insert_text((72, 72), "Stored PDF body text about LLM testing " * 5); pdf = doc.tobytes(); doc.close()
    pdf_transport = httpx.MockTransport(lambda r: httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
                                        if r.url.path.endswith(".pdf") else httpx.Response(404))
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = store.add_paper(sid, {"title": "A previously fetched paper on LLM unit test generation",
                                  "pdf_url": "https://papers.example.org/p.pdf", "raw_content": "LLM unit test generation. " * 200,
                                  "content_status": "full_text", "content_source": "pdf", "selection_status": "selected",
                                  "priority_score": 5.0}, assign_ref=True)
    ap.agent_factory = lambda s, sid_, cancel_event=None: ResearchAgent(s, sid_, provider=FakeProvider(), fetch_transport=pdf_transport)
    ap.step()
    lib = store.get_paper(paper["id"])["library_paper_id"]
    assert lib and (root / "Testing" / lib / f"{lib}.pdf").read_bytes().startswith(b"%PDF")
    assert all(c.metadata["page"] == 1 for c in vs.get_topic_chunks("Testing", lib))   # indexed from the PDF


def test_autopilot_reclaims_an_interrupted_session(env):
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "r_upload", "LLM unit test generation study. " * 30)
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    store.update_session(sid, status="processing")
    store.mark_interrupted_sessions()                     # what a server restart does
    assert store.get_session(sid)["status"] == "failed"
    ap.step()
    assert store.get_session(sid)["status"] == "autopilot"


def test_restart_between_analysis_and_publish_publishes_on_next_start(env):
    import json
    from src.paths import analysis_path
    from src.research.autopilot import Autopilot
    ap, store, root = env["ap"], env["store"], env["root"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = store.add_paper(sid, {"title": "An analysed but unpublished LLM testing paper",
                                  "raw_content": "LLM unit test generation. " * 200, "content_status": "full_text",
                                  "selection_status": "selected", "processing_status": "done",
                                  "priority_score": 5.0}, assign_ref=True)
    store.update_paper(paper["id"], processed_at=NOW)
    # The server stopped part-way through the publish: the folder and metadata exist, nothing is indexed.
    half = root / "Testing" / "half_written"
    half.mkdir()
    (half / "paper.json").write_text(json.dumps({"origin": "agent", "research_paper_id": paper["id"]}))

    restarted = Autopilot(store, agent_factory=ap.agent_factory, quota_check=lambda: 900,
                          enricher=ap.enricher, clock=lambda: NOW)
    restarted.step()
    row = store.get_paper(paper["id"])
    assert row["library_paper_id"] == "half_written" and row["published_at"] is not None
    assert vs.get_topic_chunks("Testing", "half_written")                      # indexed for Ask
    assert analysis_path("Testing", "half_written").exists()
    assert "unpublished LLM testing paper" in (root / "Testing" / "_reports" / "living_report.md").read_text()
    assert [p.name for p in (root / "Testing").iterdir() if not p.name.startswith("_")] == ["half_written"]
    assert store.unpublished_papers(sid, ("done", "analysis_failed")) == []


def test_startup_keeps_retrying_until_the_database_is_up(monkeypatch):
    import asyncio
    import src.api.server as server
    attempts, started = [], []

    def prepare():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("connection refused")

    monkeypatch.setattr(server, "_prepare_database", prepare)
    monkeypatch.setattr(server, "DB_RETRY_SECONDS", 0)
    monkeypatch.setattr(server, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(server.autopilot, "start", lambda: started.append(1))
    asyncio.run(server._start_when_database_ready())
    assert len(attempts) == 3 and started == [1]


# ── outages, retries and concurrency ──

class _FailingProvider(FakeProvider):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def complete(self, prompt, **kw):
        raise self.error


def _agent_with(error):
    return lambda s, sid, cancel_event=None: ResearchAgent(
        s, sid, provider=_FailingProvider(error), cancel_event=cancel_event,
        searx=SearXNGClient("http://searx.test", transport=make_searx_transport()), fetch_transport=not_found_transport())


def _queued_paper(ap, store):
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    return sid, store.add_paper(sid, {"title": "A queued paper on LLM unit test generation",
                                      "raw_content": "LLM unit test generation. " * 200, "content_status": "full_text",
                                      "selection_status": "selected", "priority_score": 5.0}, assign_ref=True)


def test_expired_key_pauses_and_keeps_the_queue(env):
    from src.llm.llm import LLMHTTPError
    ap, store = env["ap"], env["store"]
    sid, paper = _queued_paper(ap, store)
    ap.agent_factory = _agent_with(LLMHTTPError(401, "User not found."))
    ap.step()
    row = store.get_paper(paper["id"])
    assert row["processing_status"] == "pending" and row["attempts"] == 0      # not the paper's fault
    state = store.get_autopilot()
    assert state["state"] == "paused" and "API key" in state["state_message"]
    assert datetime.fromisoformat(state["paused_until"]) == NOW + timedelta(minutes=30)
    ap.step()                                                                  # still paused: nothing touched
    assert store.get_paper(paper["id"])["processing_status"] == "pending"


def test_rate_limited_everywhere_pauses_briefly(env):
    from src.llm.llm import RateLimited
    ap, store = env["ap"], env["store"]
    sid, paper = _queued_paper(ap, store)
    ap.agent_factory = _agent_with(RateLimited("All configured models are rate limited"))
    ap.step()
    assert store.get_paper(paper["id"])["processing_status"] == "pending"
    assert datetime.fromisoformat(store.get_autopilot()["paused_until"]) == NOW + timedelta(minutes=3)


def test_failed_paper_is_retried_later_then_given_up(env):
    from src.research.autopilot import MAX_ATTEMPTS
    ap, store = env["ap"], env["store"]
    sid, paper = _queued_paper(ap, store)
    ap.agent_factory = _agent_with(ValueError("parser crashed"))
    clock = {"now": NOW}
    ap.clock = lambda: clock["now"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ap._process(ap.agent_factory(store, sid), "Testing", sid, store.get_paper(paper["id"]))
        row = store.get_paper(paper["id"])
        assert row["attempts"] == attempt
        if attempt < MAX_ATTEMPTS:
            assert row["processing_status"] == "pending"
            retry_at = datetime.fromisoformat(row["retry_at"])
            assert retry_at > clock["now"]
            assert store.next_queued_paper(sid, ("pending",), clock["now"]) is None   # waits for its retry time
            clock["now"] = retry_at
            assert store.next_queued_paper(sid, ("pending",), clock["now"])["id"] == paper["id"]
    assert store.get_paper(paper["id"])["processing_status"] == "failed"      # gave up after MAX_ATTEMPTS


def test_autopilot_waits_for_a_manual_job_on_the_same_session(env):
    import threading
    ap, store = env["ap"], env["store"]
    sid, paper = _queued_paper(ap, store)
    release = threading.Event()
    jobs._threads[sid] = threading.Thread(target=release.wait)
    jobs._threads[sid].start()
    try:
        ap.step()
        assert store.get_autopilot()["state"] == "waiting" and sid not in jobs.claimed()
        assert store.get_paper(paper["id"])["attempts"] == 0
    finally:
        release.set()
        jobs._threads[sid].join()
        jobs._threads.pop(sid, None)


def test_lost_lock_connection_is_detected():
    from src.research.autopilot import _lock_alive

    class Dead:
        def execute(self, *a):
            raise OSError("server closed the connection")
    assert _lock_alive(Dead()) is False


def test_gap_check_is_postponed_when_search_is_down(env):
    from src.research.searxng import SearchUnavailable
    from src.research.verify import verify_gap

    class DownLiterature:
        def search(self, queries, per_query=10):
            return [], {"searxng_failed": True, "openalex_failed": False}
    ap, store = env["ap"], env["store"]
    sid, _ = _queued_paper(ap, store)
    gap = {"id": "g", "ref": "G1", "title": "No studies of X", "description": "", "confidence": "high"}
    with pytest.raises(SearchUnavailable):
        verify_gap(store, sid, gap, "Testing", provider=FakeProvider(), literature=DownLiterature())


def test_quota_reset_time_just_after_midnight():
    assert next_quota_reset(datetime(2026, 9, 27, 0, 2, tzinfo=timezone.utc)) == datetime(2026, 9, 27, 0, 5, tzinfo=timezone.utc)
    assert next_quota_reset(datetime(2026, 9, 27, 0, 6, tzinfo=timezone.utc)) == datetime(2026, 9, 28, 0, 5, tzinfo=timezone.utc)


def test_topic_and_session_the_autopilot_uses_cannot_be_deleted(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store = env["ap"], env["store"]
    sid, _ = _queued_paper(ap, store)
    ap.step()                                                  # claims the session
    with TestClient(app) as c:
        assert c.delete("/api/library/topics/Testing").status_code == 409
        assert c.delete(f"/api/research/{sid}").status_code == 409
        assert (env["root"] / "Testing").is_dir() and store.get_session(sid)
        assert c.delete("/api/library/topics/Other").status_code == 200


# ── internet connection lost ──

def _offline_transport():
    def handler(request):
        raise httpx.ConnectError("Network is unreachable")
    return httpx.MockTransport(handler)


def _offline_paper(ap, store):
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    return sid, store.add_paper(sid, {"title": "A paper whose PDF must be downloaded", "abstract": "LLM unit test generation. " * 20,
                                      "url": "https://papers.example.org/landing", "pdf_url": "https://papers.example.org/p.pdf",
                                      "selection_status": "selected", "priority_score": 5.0}, assign_ref=True)


def test_no_internet_while_downloading_keeps_the_paper_for_later(env, monkeypatch):
    import src.research.agent as agent_mod
    ap, store = env["ap"], env["store"]
    sid, paper = _offline_paper(ap, store)
    monkeypatch.setattr(agent_mod, "internet_reachable", lambda: False)
    ap.agent_factory = lambda s, sid_, cancel_event=None: ResearchAgent(s, sid_, provider=FakeProvider(),
                                                                        fetch_transport=_offline_transport())
    ap.step()
    row = store.get_paper(paper["id"])
    assert row["processing_status"] == "pending" and row["content_status"] == "pending" and not row["raw_content"]
    assert row["attempts"] == 0 and store.get_summary(paper["id"]) is None      # nothing downgraded or spent
    state = store.get_autopilot()
    assert state["state"] == "paused" and "internet" in state["state_message"].lower()
    assert datetime.fromisoformat(state["paused_until"]) == NOW + timedelta(minutes=10)


def test_a_dead_website_is_not_mistaken_for_no_internet(env, monkeypatch):
    import src.research.agent as agent_mod
    ap, store = env["ap"], env["store"]
    sid, paper = _offline_paper(ap, store)
    monkeypatch.setattr(agent_mod, "internet_reachable", lambda: True)          # we are online; the site is not
    ap.agent_factory = lambda s, sid_, cancel_event=None: ResearchAgent(s, sid_, provider=FakeProvider(),
                                                                        fetch_transport=_offline_transport())
    ap.step()
    row = store.get_paper(paper["id"])
    assert row["content_status"] == "abstract" and row["processing_status"] in ("done", "analysis_failed")


def test_searxng_without_internet_is_a_failed_search_not_an_empty_one():
    from src.research.searxng import SearchUnavailable
    body = {"results": [], "unresponsive_engines": [["arxiv", "timeout"], ["crossref", "connection error"]]}
    client = SearXNGClient("http://searx.test", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    with pytest.raises(SearchUnavailable):
        client.search("anything")
    assert client.check()[0] is False


def test_openalex_outage_does_not_mark_papers_as_enriched(env):
    ap, store = env["ap"], env["store"]
    ap.enricher = OpenAlexClient(transport=_offline_transport(), enabled=True)
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = store.add_paper(sid, {"title": "An uploaded paper about LLM unit test generation", "added_by": "user",
                                  "selection_status": "selected"}, assign_ref=True)
    ap._enrich_uploads(sid)
    assert store.get_paper(paper["id"])["enriched_at"] is None                  # looked up again later


def test_failed_publish_is_retried_after_the_connection_returns(env, monkeypatch):
    import src.research.autopilot as ap_mod
    ap, store, root = env["ap"], env["store"], env["root"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = store.add_paper(sid, {"title": "An analysed paper waiting for its publish", "raw_content": "LLM testing. " * 200,
                                  "content_status": "full_text", "selection_status": "selected",
                                  "processing_status": "done"}, assign_ref=True)
    store.update_paper(paper["id"], processed_at=NOW)
    real_publish = ap.publish
    offline = {"on": True}

    def publish(*a, **kw):
        if offline["on"]:
            raise OSError("Network is unreachable")
        return real_publish(*a, **kw)
    monkeypatch.setattr(ap, "publish", publish)
    clock = {"now": NOW}
    ap.clock = lambda: clock["now"]
    ap.step()
    assert store.get_paper(paper["id"])["published_at"] is None
    offline["on"] = False
    ap.step()                                                                   # too soon: not retried yet
    assert store.get_paper(paper["id"])["published_at"] is None
    clock["now"] = NOW + ap_mod.PUBLISH_RETRY
    ap.step()
    assert store.get_paper(paper["id"])["published_at"] is not None
