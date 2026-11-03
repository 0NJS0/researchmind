"""Finding a free full text when a paper's own page only has the abstract; weekly re-checks; attached PDFs."""
import fitz
import httpx
import pytest

import src.research.content as content
from src.research.agent import ResearchAgent
from src.research.content import fetch_content
from src.research.fulltext import FullTextFinder, landing_pdf_links, pattern_pdf_urls
from src.research.enrich import OpenAlexClient
from src.ingest import vectorstore as vs
from src.paths import analysis_path
from tests.conftest import FakeProvider
from tests.test_autopilot import NOW, activate, env  # noqa: F401  (fixture)

BODY = ("Introduction. We study industrial anomaly detection with frozen visual features. Method. Results. "
        "Discussion. Limitations. Conclusion. References. ") * 60


def pdf_bytes(text: str = BODY) -> bytes:
    doc = fitz.open()
    for i in range(0, len(text), 2500):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text[i:i + 2500], fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def public_hosts(monkeypatch):
    monkeypatch.setattr(content, "_is_public_host", lambda host: True)  # no DNS lookups in tests


def serve(routes: dict[str, tuple[int, bytes, str]]):
    """MockTransport answering exact URLs; anything else is a 404."""
    def handler(request: httpx.Request):
        status, body, ctype = routes.get(str(request.url), (404, b"not found", "text/plain"))
        return httpx.Response(status, content=body, headers={"content-type": ctype})
    return httpx.MockTransport(handler)


class StubFinder:
    def __init__(self, urls):
        self.urls, self.calls = urls, 0

    def candidates(self, paper, exclude=frozenset()):
        self.calls += 1
        return [u for u in self.urls if u not in exclude]


# ── link discovery ──

def test_known_sites_map_to_their_pdf():
    assert pattern_pdf_urls("https://arxiv.org/abs/2310.19070v2") == ["https://arxiv.org/pdf/2310.19070"]  # latest version
    assert pattern_pdf_urls("https://aclanthology.org/N19-1423/") == ["https://aclanthology.org/N19-1423.pdf"]
    assert pattern_pdf_urls("https://openreview.net/forum?id=abc123") == ["https://openreview.net/pdf?id=abc123"]
    assert pattern_pdf_urls("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/") == [
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC123456&blobtype=pdf"]
    assert pattern_pdf_urls("https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1") == [
        "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1.full.pdf"]
    assert pattern_pdf_urls("https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5e-Abstract.html") == [
        "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5e-Paper.pdf"]
    assert pattern_pdf_urls("https://www.mdpi.com/2076-3417/13/5/3052") == ["https://www.mdpi.com/2076-3417/13/5/3052/pdf"]
    assert pattern_pdf_urls("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1") == []


def test_download_links_on_a_landing_page():
    html = """<html><head><meta name="citation_pdf_url" content="/content/paper.pdf"></head><body>
      <a href="/about">About</a><a href="/files/full.pdf">Paper</a><a href="/view/42">Download PDF</a>
      <a href="javascript:void(0)">PDF</a></body></html>"""
    assert landing_pdf_links(html, "https://repo.example.org/record/42") == [
        "https://repo.example.org/content/paper.pdf", "https://repo.example.org/files/full.pdf",
        "https://repo.example.org/view/42"]


def test_openalex_locations_become_candidates():
    work = {"title": "Calibrated anomaly detection", "doi": "https://doi.org/10.1/x", "ids": {},
            "best_oa_location": {"pdf_url": None, "landing_page_url": "https://doi.org/10.1/x", "is_oa": True},
            "locations": [{"pdf_url": "https://repo.example.org/files/x.pdf", "landing_page_url": "https://repo.example.org/x", "is_oa": True},
                          {"pdf_url": None, "landing_page_url": "http://arxiv.org/abs/2310.19070", "is_oa": True},
                          {"pdf_url": None, "landing_page_url": "https://publisher.example.com/x", "is_oa": False}]}

    class OA(OpenAlexClient):
        def find(self, paper):
            return work
    urls = FullTextFinder(openalex=OA(enabled=True), transport=serve({})).openalex_urls({"title": "x"})
    assert urls[:2] == ["https://repo.example.org/files/x.pdf", "https://arxiv.org/pdf/2310.19070"]
    assert "https://repo.example.org/x" in urls and "https://publisher.example.com/x" not in urls  # closed copy skipped


# ── fetching ──

def test_other_free_copy_is_used_when_the_page_only_has_the_abstract():
    landing = b"<html><head><meta name='citation_title' content='T'><meta name='citation_abstract' content='" + \
              b"An abstract. " * 30 + b"'></head><body>Abstract only.</body></html>"
    transport = serve({"https://publisher.example.com/paper": (200, landing, "text/html"),
                       "https://arxiv.org/pdf/2310.19070": (200, pdf_bytes(), "application/pdf")})
    finder = StubFinder(["https://arxiv.org/pdf/2310.19070"])
    res = fetch_content({"title": "T", "url": "https://publisher.example.com/paper"}, transport=transport, finder=finder)
    assert res.status == "full_text" and res.source == "pdf" and res.pdf_bytes
    assert res.metadata["pdf_url"] == "https://arxiv.org/pdf/2310.19070" and finder.calls == 1


def test_repository_page_is_followed_to_its_pdf():
    repo = b"<html><body><a href='/bitstream/1/paper.pdf'>Download PDF</a></body></html>"
    transport = serve({"https://repo.example.org/record/1": (200, repo, "text/html"),
                       "https://repo.example.org/bitstream/1/paper.pdf": (200, pdf_bytes(), "application/pdf")})
    res = fetch_content({"title": "T", "abstract": "x"}, transport=transport,
                        finder=StubFinder(["https://repo.example.org/record/1"]))
    assert res.status == "full_text" and res.metadata["pdf_url"] == "https://repo.example.org/bitstream/1/paper.pdf"


def test_blocked_site_falls_back_to_the_abstract():
    blocked = serve({"https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1": (403, b"Just a moment...", "text/html")})
    res = fetch_content({"title": "A paper", "abstract": "An abstract that is long enough. " * 10,
                         "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1"},
                        transport=blocked, finder=StubFinder([]))
    assert res.status == "abstract" and not res.unreachable


# ── autopilot: weekly re-check ──

def _abstract_only(store, sid, **extra):
    paper = store.add_paper(sid, {"title": "Calibrated few-shot industrial anomaly detection", "selection_status": "selected",
                                  "abstract": "An abstract about industrial anomaly detection. " * 10,
                                  "raw_content": "An abstract about industrial anomaly detection. " * 10,
                                  "content_status": "abstract", "processing_status": "done",
                                  "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1", **extra}, assign_ref=True)
    store.save_summary(paper["id"], "abstract", {"research_problem": "abstract-level"}, "m")
    store.update_paper(paper["id"], processed_at=NOW, published_at=NOW)
    return store.get_paper(paper["id"])


def test_weekly_recheck_finds_a_new_free_copy_and_reanalyses(env):
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = _abstract_only(store, sid)
    store.update_autopilot(next_search_at=NOW.replace(year=2027))          # nothing to search right now
    transport = serve({"https://arxiv.org/pdf/2310.19070": (200, pdf_bytes(), "application/pdf")})
    finder = StubFinder(["https://arxiv.org/pdf/2310.19070"])
    ap.agent_factory = lambda s, sid_, cancel_event=None: ResearchAgent(s, sid_, provider=FakeProvider(),
                                                                        fetch_transport=transport, fulltext=finder)
    ap.step()                                                              # the re-check
    row = store.get_paper(paper["id"])
    assert row["content_status"] == "full_text" and row["processing_status"] == "pending"
    assert store.get_summary(paper["id"]) is None                          # abstract-level summary dropped
    ap.step()                                                              # re-analysed from the full text
    assert store.get_summary(paper["id"])["analysis_level"] == "full_text"
    lib = store.get_paper(paper["id"])["library_paper_id"]
    assert lib and (env["root"] / "Testing" / lib / f"{lib}.pdf").is_file()
    assert all(c.metadata["page"] >= 1 for c in vs.get_topic_chunks("Testing", lib))   # indexed from the PDF pages


def test_recheck_without_a_free_copy_waits_a_week(env):
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = _abstract_only(store, sid)
    store.update_autopilot(next_search_at=NOW.replace(year=2027))
    ap.agent_factory = lambda s, sid_, cancel_event=None: ResearchAgent(
        s, sid_, provider=FakeProvider(), fetch_transport=serve({}), fulltext=StubFinder([]))
    ap.step()
    row = store.get_paper(paper["id"])
    assert row["content_status"] == "abstract" and row["fulltext_checked_at"] is not None
    assert store.next_fulltext_recheck(sid, NOW) is None                   # not checked again this week


# ── attaching a PDF by hand ──

def test_attached_pdf_replaces_the_abstract_analysis(env):
    from fastapi.testclient import TestClient
    from src.api.server import app
    ap, store = env["ap"], env["store"]
    activate(store, "Testing")
    sid = ap.ensure_session("Testing")
    paper = _abstract_only(store, sid)
    with TestClient(app) as c:
        listing = c.get("/api/library/topics/Testing/papers").json()["papers"]
        assert [p["needs_full_text"] for p in listing if p["id"] == paper["id"]] == [True]
        blank = fitz.open(); blank.new_page(); empty = blank.tobytes(); blank.close()
        assert c.post(f"/api/papers/{paper['id']}/pdf", files={"file": ("x.pdf", empty, "application/pdf")}).status_code == 422
        assert c.post(f"/api/papers/{paper['id']}/pdf", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 400
        store.update_autopilot(current_paper_id=paper["id"])
        assert c.post(f"/api/papers/{paper['id']}/pdf", files={"file": ("x.pdf", pdf_bytes(), "application/pdf")}).status_code == 409
        store.update_autopilot(current_paper_id=None)
        r = c.post(f"/api/papers/{paper['id']}/pdf", files={"file": ("ssrn.pdf", pdf_bytes(), "application/pdf")})
        assert r.status_code == 200 and r.json()["status"] == "queued" and r.json()["autopilot_role"] == "active"
    row = store.get_paper(paper["id"])
    assert row["content_status"] == "full_text" and row["content_source"] == "upload" and row["processing_status"] == "pending"
    ap.step()                                                              # processed next, from the full text
    assert store.get_summary(paper["id"])["analysis_level"] == "full_text"
    lib = store.get_paper(paper["id"])["library_paper_id"]
    assert (env["root"] / "Testing" / lib / f"{lib}.pdf").is_file()
    assert not (env["root"] / "Testing" / "_incoming" / f"{paper['id']}.pdf").exists()
    assert analysis_path("Testing", lib).exists()
