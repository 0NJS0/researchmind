"""pgvector store, library API (upload → ask → report) and path-safety, with fake embeddings and LLM."""
import hashlib
import math
import re
import shutil
import time
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from src.config import PAPERS_DIR
from src.ingest import embeddings as emb_mod
from src.ingest import vectorstore as vs
from tests.conftest import truncate

DIM = 16


class FakeEmbeddings:
    """Deterministic bag-of-words hashing embeddings."""

    model = "test-embed"

    def __init__(self, dim=DIM):
        self.dim = dim
        self.calls = 0

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts, progress=None):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class PipelineLLM:
    name = "fake"
    model = "fake"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, *, system=None, max_tokens=None, json_mode=False):
        self.prompts.append(prompt)
        if "Classify the academic field" in prompt:
            return '{"domain": "Computer Science", "subfield": "NLP", "research_type": "empirical", "methodology": "deep learning", "key_quantities": ["F1 on named benchmarks"], "quality_signals": ["held-out test set"]}'
        if prompt.startswith("<question>") and "<excerpts>" in prompt:
            return "### toxicity\n- **Findings:** RoBERTa reaches 0.91 F1 [toxicity, p.1]"
        if "Task: write a structured analysis of this single paper" in prompt:
            pid = re.search(r"Paper ID: (\S+)", prompt).group(1)
            return f"## {pid}\n**One-line summary:** analysis of {pid}"
        if "Task: compare these papers" in prompt:
            return "## Comparison table\n| Paper | ... |"
        if "Task: identify the most important research gaps" in prompt:
            return "## Research gaps\n### 1. No evaluation on code-mixed text"
        if "Task: answer the question" in prompt:
            return "## Answer\nRoBERTa works best [toxicity, p.1]"
        if "Task: write a literature-review report" in prompt:
            return "# report\n## Executive summary\nDone."
        return "{}"


@pytest.fixture
def embeddings():
    truncate("chunks", "embedding_meta")
    fake = FakeEmbeddings()
    emb_mod.set_embeddings(fake)
    yield fake
    emb_mod.set_embeddings(None)


@pytest.fixture
def papers_dir():
    root = Path(PAPERS_DIR)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = PipelineLLM()
    import src.agents.common as common
    import src.llm.structured as structured
    monkeypatch.setattr(common, "get_provider", lambda purpose="default": llm)
    monkeypatch.setattr(structured, "get_provider", lambda purpose="default": llm)
    return llm


@pytest.fixture
def client(embeddings, papers_dir, fake_llm, store):
    from src.api.server import app
    with TestClient(app) as c:
        yield c


def fake_llm_prompts(client):
    import src.agents.common as common
    return common.get_provider().prompts


def make_pdf(path: Path, pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def doc(text, paper="p1", topic="t1", page=1):
    from langchain_core.documents import Document
    return Document(page_content=text, metadata={"paper_id": paper, "topic": topic, "source": f"{paper}.pdf",
                                                 "page": page})


# ── vector store ──

def test_vectorstore_roundtrip(embeddings):
    stored = vs.add_documents([
        doc("toxicity detection with transformers", "a"), doc("roberta classifier toxic comments", "a", page=2),
        doc("protein folding with diffusion", "b"), doc("toxicity in another topic", "c", topic="t2"),
    ])
    assert stored == 4
    assert vs.get_embedding_meta() == {"model": "test-embed", "dim": DIM}
    papers = {p["paper_id"]: p for p in vs.list_papers("t1")}
    assert papers["a"]["chunks"] == 2 and papers["a"]["pages"] == 2 and set(papers) == {"a", "b"}

    hits = vs.similarity_search("toxic comments classifier", k=2, topic="t1")
    assert hits[0].metadata["paper_id"] == "a" and hits[0].metadata["topic"] == "t1"
    assert all(h.metadata["topic"] == "t1" for h in hits)
    assert vs.similarity_search("toxicity", k=10)  # unfiltered search uses the HNSW expression

    ordered = vs.get_topic_chunks("t1", "a")
    assert [d.metadata["page"] for d in ordered] == [1, 2]
    assert vs.get_chunk(ordered[0].metadata["chunk_id"])["content"].startswith("toxicity detection")


def test_reindexing_a_paper_replaces_its_chunks(embeddings):
    vs.add_documents([doc("first version", "a"), doc("first version page two", "a", page=2)])
    vs.add_documents([doc("second version", "a")])
    chunks = vs.get_topic_chunks("t1", "a")
    assert [c.page_content for c in chunks] == ["second version"]


def test_changing_embedding_model_requires_reindex(embeddings):
    vs.add_documents([doc("hello world", "a")])
    emb_mod.set_embeddings(FakeEmbeddings(dim=DIM * 2))
    with pytest.raises(vs.EmbeddingMismatch):
        vs.add_documents([doc("hello again", "b")])
    vs.reset_index()
    assert vs.add_documents([doc("hello again", "b")]) == 1
    assert vs.get_embedding_meta()["dim"] == DIM * 2


def test_delete_paper_and_topic(embeddings):
    vs.add_documents([doc("x one", "a"), doc("y two", "b"), doc("z three", "c", topic="t2")])
    assert vs.delete_paper("a", "t1") == 1
    assert vs.paper_exists("b", "t1") and not vs.paper_exists("a", "t1")
    assert vs.delete_topic_papers("t1") == 1
    assert [p["paper_id"] for p in vs.list_papers()] == ["c"]


# ── library API ──

def wait_job(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/library/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_ask_and_report(client, papers_dir):
    assert client.post("/api/library/topics", json={"name": "LLM Toxicity"}).json() == {"name": "LLM_Toxicity"}
    assert client.post("/api/library/topics", json={"name": "LLM Toxicity"}).status_code == 409

    pdf = make_pdf(papers_dir / "x.pdf", ["Toxicity detection with RoBERTa reaches 0.91 F1.",
                                          "Limitations: English only."])
    r = client.post("/api/library/upload", params={"topic": "LLM_Toxicity"},
                    files={"file": ("toxicity.pdf", pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    assert r.json()["pages"] == 2 and r.json()["paper_id"] == "toxicity"
    pdf2 = make_pdf(papers_dir / "y.pdf", ["Bias in hate speech datasets across dialects."])
    client.post("/api/library/upload", params={"topic": "LLM_Toxicity"},
                files={"file": ("bias study.pdf", pdf2, "application/pdf")})

    topics = client.get("/api/library/topics").json()["topics"]
    assert topics == [{"name": "LLM_Toxicity", "papers": 2, "chunks": 3}]

    r = client.post("/api/library/query", json={"question": "Which model works best?", "topic": "LLM_Toxicity",
                                                "session_id": "s1"})
    body = r.json()
    assert body["answer"].startswith("## Answer") and body["errors"] == []
    assert body["citation_check"] == {"total": 1, "verified": 1, "unverified": []}
    analysis_prompt = next(p for p in fake_llm_prompts(client) if p.startswith("<question>") and "<excerpts>" in p)
    assert "Results this field reports (capture exactly): F1 on named benchmarks" in analysis_prompt
    assert {s["paper_id"] for s in body["sources"]} == {"toxicity", "bias_study"}
    assert all(s["chunk_id"] and s["page"] for s in body["sources"])

    with client.stream("POST", "/api/library/query/stream",
                       json={"question": "And the limitations?", "topic": "LLM_Toxicity", "session_id": "s1"}) as resp:
        lines = [l for l in resp.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert '"type": "answer"' in lines[-2]
    assert any('"stage": "report", "status": "done"' in l for l in lines)

    job = client.post("/api/library/reports", json={"topic": "LLM_Toxicity"}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "completed", job
    reports = client.get("/api/library/reports", params={"topic": "LLM_Toxicity"}).json()["reports"]
    assert {r["type"] for r in reports} == {"combined", "paper"}
    combined = next(r for r in reports if r["type"] == "combined")
    content = client.get(f"/api/library/reports/LLM_Toxicity/{combined['filename']}").json()["content"]
    assert content.startswith("# report")
    analysis = client.get("/api/library/papers/LLM_Toxicity/toxicity/analysis").json()["content"]
    assert "analysis of toxicity" in analysis
    papers = client.get("/api/library/papers", params={"topic": "LLM_Toxicity"}).json()["papers"]
    assert all(p["has_analysis"] for p in papers)
    assert client.get("/api/library/papers").status_code == 200    # no topic: used to crash

    assert client.delete("/api/library/papers/LLM_Toxicity/toxicity").status_code == 200
    assert client.delete("/api/library/topics/LLM_Toxicity").status_code == 200
    assert not (papers_dir / "LLM_Toxicity").exists()


def test_rejects_non_pdf_uploads(client):
    client.post("/api/library/topics", json={"name": "t"})
    r = client.post("/api/library/upload", params={"topic": "t"}, files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert r.status_code == 400
    r = client.post("/api/library/upload", params={"topic": "t"}, files={"file": ("x.txt", b"%PDF-1", "text/plain")})
    assert r.status_code == 400


@pytest.mark.parametrize("method,url", [
    ("delete", "/api/library/topics/%2E%2E"),
    ("delete", "/api/library/topics/..%2F..%2Ftmp"),
    ("get", "/api/library/reports/t/..%2F..%2F.env"),
    ("get", "/api/library/reports/%2E%2E/report_20240101_000000.md"),
    ("get", "/api/library/papers/%2E%2E/x/analysis"),
    ("delete", "/api/library/papers/t/%2E%2E"),
    ("post", "/api/library/upload?topic=..%2Fescape"),
])
def test_path_traversal_is_rejected(client, papers_dir, method, url):
    sentinel = papers_dir.parent / "researchmind-sentinel"
    sentinel.mkdir(exist_ok=True)
    kwargs = {"files": {"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}} if method == "post" else {}
    r = getattr(client, method)(url, **kwargs)
    # 405/200 happen when the client normalises ".." out of /api and the SPA mount answers instead.
    assert r.status_code in (400, 404, 405, 422) or (r.status_code == 200 and "<!doctype html>" in r.text.lower()), \
        (url, r.status_code, r.text)
    assert papers_dir.exists() and sentinel.exists()
    sentinel.rmdir()


def test_health_and_config_do_not_leak_secrets(client):
    health = client.get("/api/health").json()
    assert set(health) >= {"status", "database", "llm", "searxng", "vectorstore", "config"}
    assert health["database"]["ok"] is True
    text = str(health) + str(client.get("/api/config").json())
    assert "postgresql://" not in text and "password" not in text.lower()


# ── SSRF guard ──

@pytest.mark.parametrize("ip,public", [
    ("8.8.8.8", True), ("127.0.0.1", False), ("10.1.2.3", False), ("169.254.169.254", False),
    ("100.64.0.1", False), ("::1", False), ("::ffff:127.0.0.1", False), ("fd00::1", False),
    ("2606:4700:4700::1111", True),
])
def test_public_ip_classification(ip, public):
    from src.research.content import _is_public_ip
    assert _is_public_ip(ip) is public


def test_citation_check():
    from langchain_core.documents import Document
    from src.agents.citations import check_citations
    docs = [Document(page_content="x", metadata={"paper_id": "toxicity", "page": 1}),
            Document(page_content="y", metadata={"paper_id": "bias_study", "page": 3})]
    r = check_citations("A [toxicity, p.1]. B [toxicity, p.7; bias_study, p.2-3]. C [ghost, p.1]. [link](x)", docs)
    assert r["total"] == 4 and r["verified"] == 2
    assert {(u["paper_id"], u["reason"]) for u in r["unverified"]} == {
        ("toxicity", "page not among retrieved passages"), ("ghost", "paper not retrieved")}


def test_uploads_with_the_same_file_name_never_overwrite(client, papers_dir):
    a = make_pdf(papers_dir / "a.pdf", ["First paper about toxicity classifiers."])
    b = make_pdf(papers_dir / "b.pdf", ["Second, different paper about dialect bias."])
    r1 = client.post("/api/library/upload", params={"topic": "Dup"}, files={"file": ("paper.pdf", a, "application/pdf")})
    r2 = client.post("/api/library/upload", params={"topic": "Dup"}, files={"file": ("paper.pdf", b, "application/pdf")})
    assert r1.status_code == r2.status_code == 201
    assert (r1.json()["paper_id"], r2.json()["paper_id"]) == ("paper", "paper-2")
    assert {p["paper_id"] for p in vs.list_papers("Dup")} == {"paper", "paper-2"}
    assert not list((papers_dir / "Dup").glob(".upload_*"))          # no temp files left behind


def test_upload_without_text_leaves_nothing_behind(client, papers_dir):
    doc = fitz.open(); doc.new_page(); blank = doc.tobytes(); doc.close()
    r = client.post("/api/library/upload", params={"topic": "Blank"}, files={"file": ("scan.pdf", blank, "application/pdf")})
    assert r.status_code == 422
    assert not [p for p in (papers_dir / "Blank").iterdir()]


def test_reindex_keeps_agent_papers_without_a_pdf(client, papers_dir):
    from src.research.publish import publish_to_library
    (papers_dir / "Agent").mkdir(parents=True, exist_ok=True)
    lib = publish_to_library("Agent", {"id": "r1", "title": "A web-only paper", "raw_content": "Text of a web-only paper. " * 20,
                                       "content_status": "full_text", "url": "https://example.org/p"})
    assert (papers_dir / "Agent" / lib / "text.txt").is_file()
    r = client.post("/api/library/ingest", params={"topic": "Agent"})
    job = r.json()["id"]
    for _ in range(100):
        if client.get(f"/api/library/jobs/{job}").json()["status"] != "running":
            break
        time.sleep(0.05)
    assert client.get(f"/api/library/jobs/{job}").json()["status"] == "completed"
    assert [p["paper_id"] for p in vs.list_papers("Agent")] == [lib]


def test_compressed_bombs_are_refused():
    import httpx
    from src.research.content import _download, make_http_client
    t = httpx.MockTransport(lambda r: httpx.Response(200, stream=httpx.ByteStream(b"x"), headers={"content-encoding": "zstd"}))
    with make_http_client(t) as c, pytest.raises(ValueError, match="encoding"):
        _download(c, "https://example.org/p.pdf")
    seen = {}
    def capture(r):
        seen.update(r.headers)
        return httpx.Response(200, content=b"%PDF")
    with make_http_client(httpx.MockTransport(capture)) as c:
        _download(c, "https://example.org/p.pdf")
    assert "zstd" not in seen["accept-encoding"] and "br" not in seen["accept-encoding"]
