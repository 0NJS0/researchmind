"""Paper library API: topics, PDF upload/indexing, question answering and topic reports (under /api/library)."""
import asyncio
import hashlib
import json
import logging
import re
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask
from starlette.responses import FileResponse, StreamingResponse

from src.agents.citations import check_citations
from src.config import PAPERS_DIR
from src.graph.report_graph import report_graph
from src.graph.workflow import research_graph
from src.ingest.chunker import chunk_documents
from src.ingest.loader import load_papers, load_pdf
from src.ingest.vectorstore import (
    add_documents, delete_paper, delete_topic_papers, get_chunk, list_papers, list_topics, reset_index,
)
from src.paths import inside_root, is_safe_name, papers_root, to_name  # noqa: F401
from src.reports import report_jobs
from src.research.normalize import canonical_url, clean_arxiv_id, clean_doi
from src.research.jobs import jobs
from src.research.publish import library_paper_exists, read_meta
from src.research.topic_ops import pdf_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/library", tags=["library"])
run = asyncio.to_thread

MAX_FILE_SIZE = 100 * 1024 * 1024
REPORT_MIN_REQUESTS = 40  # a report reuses the autopilot's summaries, but new papers and the write-up still cost requests
MAX_HISTORY = 5
MAX_CONVERSATIONS = 200
STAGES = ("retriever", "classifier", "analysis", "comparison", "gap", "report")

# Recent Q&A per conversation, bounded (least recently used conversations are dropped).
_conversations: OrderedDict[str, list[dict]] = OrderedDict()
_conversations_lock = threading.Lock()
_background: set[asyncio.Task] = set()


# ── path safety ──

def safe_name(value: str, what: str = "name") -> str:
    """Topic / paper / file names become path components: allow only a conservative character set."""
    value = (value or "").strip()
    if not is_safe_name(value):
        raise HTTPException(400, f"Invalid {what}: use letters, numbers, '.', '_' or '-' (max 120 characters)")
    return value


def safe_path(*parts: str) -> Path:
    path = inside_root(*parts)
    if path is None:
        raise HTTPException(400, "Invalid path")
    return path


def topic_dir(topic: str) -> Path:
    return safe_path(safe_name(topic, "topic"))


def analysis_file(topic: str, paper_id: str) -> Path:
    paper_id = safe_name(paper_id, "paper id")
    return safe_path(safe_name(topic, "topic"), paper_id, "_reports", f"{paper_id}_analysis.md")


# ── topics ──

class TopicIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@router.get("/topics")
async def get_topics():
    names = await run(list_topics)
    papers = await run(list_papers)
    counts: dict[str, dict] = {}
    for p in papers:
        c = counts.setdefault(p["topic"], {"papers": 0, "chunks": 0})
        c["papers"] += 1
        c["chunks"] += p["chunks"]
    all_names = sorted(set(names) | set(counts))
    return {"topics": [{"name": n, **counts.get(n, {"papers": 0, "chunks": 0})} for n in all_names]}


@router.post("/topics", status_code=201)
async def create_topic(body: TopicIn):
    name = to_name(body.name)
    safe_name(name, "topic name")
    path = topic_dir(name)
    if path.exists():
        raise HTTPException(409, f"Topic '{name}' already exists")
    await run(lambda: path.mkdir(parents=True, exist_ok=True))
    return {"name": name}


@router.delete("/topics/{topic}")
async def delete_topic(topic: str):
    topic = safe_name(topic, "topic")
    path = topic_dir(topic)
    await run(_ensure_topic_idle, topic)
    await run(delete_topic_papers, topic)
    if path.exists():
        await run(shutil.rmtree, path, True)
    await run(_forget_topic, topic)
    conversations_clear(topic)
    return {"status": "deleted", "topic": topic}


def _ensure_topic_idle(topic: str) -> None:
    """Deleting a topic that is being worked on would let the running task write it back to disk."""
    from src.research.store import get_store
    store = get_store()
    ap = store.get_autopilot()
    if topic in (ap.get("active_topic"), ap.get("requested_topic")):
        raise HTTPException(409, "The autopilot is working on this topic. Switch it to another topic "
                                 "(or pause it and wait for the current paper to finish) before deleting.")
    if report_jobs.busy(topic):
        raise HTTPException(409, "A report or re-index job is running for this topic; try again when it finishes")
    row = store.get_topic(topic)
    if row and row.get("session_id") and jobs.is_running(row["session_id"]):
        raise HTTPException(409, "A research job is running for this topic; cancel it first")


def _forget_topic(topic: str) -> None:
    """Drop the topic's research session and autopilot references."""
    from src.research.store import get_store
    store = get_store()
    row = store.get_topic(topic)
    if row and row.get("session_id"):
        jobs.release(row["session_id"])
    store.delete_topic(topic)





# ── papers ──

@router.get("/papers")
async def get_papers(topic: str | None = None):
    if topic:
        safe_name(topic, "topic")
    papers = await run(list_papers, topic)
    for p in papers:
        try:
            p["has_analysis"] = analysis_file(p["topic"], p["paper_id"]).exists()
        except HTTPException:
            p["has_analysis"] = False
        meta = read_meta(p["topic"], p["paper_id"])
        p["title"] = meta.get("title") or p["paper_id"].replace("_", " ")
        p["origin"] = meta.get("origin") or "upload"
        for key in ("year", "venue", "citation_count", "url", "doi", "authors"):
            p[key] = meta.get(key)
    return {"papers": papers}


@router.delete("/papers/{topic}/{paper_id}")
async def delete_paper_endpoint(topic: str, paper_id: str):
    topic, paper_id = safe_name(topic, "topic"), safe_name(paper_id, "paper id")
    safe_path(topic, paper_id)
    return await _delete(topic, library_id=paper_id)


async def _delete(topic: str, paper_id: str | None = None, library_id: str | None = None) -> dict:
    from src.research.store import get_store
    from src.research.topic_ops import PaperBusy, delete_paper_everywhere
    try:
        return await run(delete_paper_everywhere, get_store(), topic, paper_id, library_id)
    except PaperBusy as e:
        raise HTTPException(409, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))


QUEUED = ("pending", "fetching", "summarizing", "analyzing")


def _paper_status(p: dict) -> str:
    if p["selection_status"] == "rejected":
        return "rejected"
    if p["selection_status"] == "candidate":
        return "candidate"
    if p["processing_status"] in ("done",):
        return "processed"
    if p["processing_status"] == "analysis_failed":
        return "partial"
    if p["processing_status"] in ("failed", "skipped"):
        return "failed"
    return "processing" if p["processing_status"] in ("fetching", "summarizing", "analyzing") else "queued"


def _topic_papers(topic: str, include_rejected: bool) -> dict:
    """Every paper of a topic: the research session's papers plus Library papers it doesn't know yet."""
    from src.research.store import get_store
    store = get_store()
    row = store.get_topic(topic) or {}
    sid = row.get("session_id") if row.get("session_id") and store.get_session(row["session_id"]) else None
    library = {p["paper_id"]: p for p in list_papers(topic)}
    out: list[dict] = []
    if sid:
        gap_counts = store.paper_gap_counts(sid)
        for p in store.list_papers(sid, selection=None):
            if p["selection_status"] == "candidate" or (p["selection_status"] == "rejected" and not include_rejected):
                continue
            lib_id = p.get("library_paper_id") or ""
            doi = p.get("doi") or ""
            out.append({
                "id": p["id"], "ref": p["ref"], "title": p["title"], "authors": p.get("authors") or [],
                "year": p.get("year"), "venue": p.get("venue") or "", "citation_count": p.get("citation_count"),
                "peer_reviewed": p.get("peer_reviewed"), "publication_type": p.get("publication_type") or "",
                "doi": doi, "url": p.get("url") or (f"https://doi.org/{doi}" if doi else ""), "pdf_url": p.get("pdf_url") or "",
                "status": _paper_status(p), "content_status": p.get("content_status") or "",
                "relevance_score": p.get("relevance_score"), "priority_score": p.get("priority_score"),
                "relevance_reason": p.get("relevance_reason") or "",
                "origin": "upload" if p.get("added_by") == "user" else "agent",
                "library_paper_id": lib_id if lib_id and (lib_id in library or library_paper_exists(topic, lib_id)) else "",
                "has_analysis": bool(lib_id) and analysis_file(topic, lib_id).exists() if lib_id and is_safe_name(lib_id) else False,
                "has_summary": p.get("has_summary", False), "gap_count": gap_counts.get(p["id"], 0),
                "has_pdf": bool(lib_id in library and pdf_path(topic, lib_id)),
                "added_at": p.get("created_at"), "processed_at": p.get("processed_at"), "error": p.get("error") or "",
                "needs_full_text": p["selection_status"] == "selected"
                and (p.get("content_status") or "") in ("abstract", "metadata_only"),
                "fulltext_checked_at": p.get("fulltext_checked_at"),
            })
        known = {p["library_paper_id"] for p in out if p["library_paper_id"]}
    else:
        known = set()
    for pid, lib in library.items():
        if pid in known:
            continue
        meta = read_meta(topic, pid)
        doi = meta.get("doi") or ""
        out.append({
            "id": None, "ref": "", "title": meta.get("title") or pid.replace("_", " "), "authors": meta.get("authors") or [],
            "year": meta.get("year"), "venue": meta.get("venue") or "", "citation_count": meta.get("citation_count"),
            "peer_reviewed": None, "publication_type": "", "doi": doi,
            "url": meta.get("url") or (f"https://doi.org/{doi}" if doi else ""), "pdf_url": meta.get("pdf_url") or "",
            "status": "library", "content_status": "full_text", "relevance_score": None, "priority_score": None,
            "relevance_reason": "", "origin": meta.get("origin") or "upload", "library_paper_id": pid,
            "has_analysis": analysis_file(topic, pid).exists(), "has_summary": False, "gap_count": 0,
            "has_pdf": bool(pdf_path(topic, pid)),
            "added_at": lib.get("indexed_at"), "processed_at": None, "error": "",
            "needs_full_text": False, "fulltext_checked_at": None,
        })
    return {"topic": topic, "session_id": sid, "autopilot_managed": bool(sid and sid in jobs.claimed()), "papers": out}


@router.get("/topics/{topic}/papers")
async def get_topic_papers(topic: str, include_rejected: bool = False):
    if not topic_dir(topic).is_dir():
        raise HTTPException(404, "Topic not found")
    return await run(_topic_papers, topic, include_rejected)


@router.delete("/topics/{topic}/papers")
async def delete_topic_paper(topic: str, paper_id: str | None = None, library_id: str | None = None):
    """Delete a paper everywhere: database, vector index, PDF/metadata/analysis files, gaps, reports."""
    topic = safe_name(topic, "topic")
    if not topic_dir(topic).is_dir():
        raise HTTPException(404, "Topic not found")
    if paper_id and not re.fullmatch(r"[0-9a-f]{32}", paper_id):
        raise HTTPException(404, "Paper not found")
    if library_id:
        safe_name(library_id, "paper id")
    if not (paper_id or library_id):
        raise HTTPException(400, "paper_id or library_id is required")
    return await _delete(topic, paper_id, library_id)


@router.get("/topics/{topic}/papers/{library_id}/pdf")
async def download_paper_pdf(topic: str, library_id: str):
    topic_dir(topic)
    path = pdf_path(safe_name(topic, "topic"), safe_name(library_id, "paper id"))
    if not path:
        raise HTTPException(404, "No PDF stored for this paper")
    meta = read_meta(topic, library_id)
    name = (to_name(meta.get("title") or library_id, max_len=100) or library_id) + ".pdf"
    return FileResponse(path, media_type="application/pdf", filename=name)


@router.get("/topics/{topic}/export")
async def export_topic_zip(topic: str, include_pdfs: bool = True):
    """Everything about a topic in one ZIP: papers (CSV/JSON + PDFs), analyses, reports, gaps, synthesis."""
    if not topic_dir(topic).is_dir():
        raise HTTPException(404, "Topic not found")
    from src.research.store import get_store
    from src.research.topic_ops import export_topic

    def build():
        papers = _topic_papers(topic, include_rejected=False)["papers"]
        return export_topic(get_store(), topic, papers, include_pdfs)

    path = await run(build)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return FileResponse(path, media_type="application/zip", filename=f"{topic}_{stamp}.zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


@router.get("/papers/{topic}/{paper_id}/analysis")
async def get_paper_analysis(topic: str, paper_id: str):
    path = analysis_file(topic, paper_id)
    if not path.exists():
        raise HTTPException(404, "No analysis yet — generate a topic report first")
    return {"paper_id": paper_id, "topic": topic, "content": await run(path.read_text, "utf-8")}


@router.get("/chunks/{chunk_id}")
async def get_chunk_endpoint(chunk_id: int):
    chunk = await run(get_chunk, chunk_id)
    if not chunk:
        raise HTTPException(404, "Chunk not found")
    return chunk


@router.post("/upload", status_code=201)
async def upload_pdf(file: UploadFile, topic: str, request: Request, url: str = Form(""), doi: str = Form(""),
                     title: str = Form("")):
    """Upload a PDF into a topic, optionally with the paper's link / DOI and title. If the topic's research already
    has that paper with only its abstract, the PDF becomes its full text instead of a duplicate paper."""
    topic = safe_name(topic, "topic")
    url, doi, title = url.strip()[:1000], doi.strip()[:200], " ".join(title.split())[:500]
    if url and not canonical_url(url):
        raise HTTPException(422, "The link must be an http(s) URL")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_FILE_SIZE + 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)} MB)")
    raw_name = Path(file.filename or "").name
    if not raw_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files can be uploaded")
    base_id = to_name(Path(raw_name).stem) or "paper"  # names in non-Latin scripts reduce to nothing
    safe_name(base_id, "file name")
    root = topic_dir(topic)
    await run(lambda: root.mkdir(parents=True, exist_ok=True))

    header = await file.read(5)
    if not header.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")
    # A private temp file per upload: concurrent uploads never share (or overwrite) each other's bytes.
    fd, tmp_name = await run(lambda: tempfile.mkstemp(prefix=".upload_", suffix=".pdf.part", dir=root))
    tmp_path = Path(tmp_name)
    size = len(header)
    digest = hashlib.sha256(header)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)} MB)")
                digest.update(chunk)
                await run(f.write, chunk)
        sha256 = digest.hexdigest()
        duplicate = await run(_upload_with_hash, topic, sha256)
        if duplicate:
            raise HTTPException(409, f"This PDF is already in the topic as '{duplicate}'")
        docs = await run(load_pdf, tmp_path, base_id, topic, raw_name)
        if not docs:
            raise HTTPException(422, "No extractable text in this PDF (scanned documents need OCR first)")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    known = None
    try:
        known = await run(_known_research_paper, topic, url, doi, title)
        if known:
            if known["content_status"] == "full_text":
                raise HTTPException(409, f"This paper is already in the topic as {known['ref'] or known['title'][:80]} "
                                         "(with its full text)")
            data = await run(tmp_path.read_bytes)
            known = await run(_attach_upload, topic, known, "\n".join(d.page_content for d in docs), data, sha256)
            return {"status": "attached", "paper_id": known.get("library_paper_id") or "", "ref": known["ref"],
                    "topic": topic, "pages": len(docs), "chunks": 0}
    finally:
        if known:
            tmp_path.unlink(missing_ok=True)   # the attached PDF was staged elsewhere (or refused)

    # A new folder for every upload: a second "paper.pdf" must not replace an existing paper.
    paper_id, paper_dir = await run(_claim_paper_folder, topic, base_id)
    pdf_file = paper_dir / f"{paper_id}.pdf"
    try:
        await run(tmp_path.replace, pdf_file)
        for d in docs:
            d.metadata["paper_id"] = paper_id
        chunks = await run(chunk_documents, docs)
        stored = await run(add_documents, chunks)
        meta = {"origin": "upload", "sha256": sha256, "file_name": raw_name}
        paper_doi = clean_doi(doi) or clean_doi(url)
        meta.update({k: v for k, v in (("title", title), ("url", url), ("doi", paper_doi),
                                       ("arxiv_id", clean_arxiv_id(url))) if v})
        await run((paper_dir / "paper.json").write_text, json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    except Exception as e:
        logger.exception("Indexing failed for %s", raw_name)
        tmp_path.unlink(missing_ok=True)
        await run(shutil.rmtree, paper_dir, True)
        raise HTTPException(502, f"Indexing failed, nothing was saved: {str(e)[:300]}")
    return {"status": "indexed", "paper_id": paper_id, "topic": topic, "pages": len(docs), "chunks": stored}


def _known_research_paper(topic: str, url: str, doi: str, title: str) -> dict | None:
    """The topic's research paper this upload is (matched on DOI, arXiv id or title), if any."""
    if not (url or doi or title):
        return None
    from src.research.dedup import find_duplicate
    from src.research.store import get_store
    store = get_store()
    sid = (store.get_topic(topic) or {}).get("session_id")
    if not sid:
        return None
    cand = {"title": title, "doi": clean_doi(doi) or clean_doi(url), "arxiv_id": clean_arxiv_id(url), "url": url}
    papers = store.list_papers(sid, selection=None)
    # Prefer the version the research keeps (selected), then candidates; never a version rejected as a duplicate.
    rank = {"selected": 0, "candidate": 1, "rejected": 2}
    papers = [p for p in papers if not (p["selection_status"] == "rejected"
                                        and (p.get("relevance_reason") or "").startswith("Another version of"))]
    papers.sort(key=lambda p: rank.get(p["selection_status"], 3))
    return find_duplicate(cand, papers)


def _attach_upload(topic: str, paper: dict, text: str, pdf: bytes, sha256: str) -> dict:
    from src.research.autopilot import autopilot
    from src.research.store import get_store
    from src.research.topic_ops import upgrade_to_full_text
    store = get_store()
    if store.get_autopilot().get("current_paper_id") == paper["id"] or (
            jobs.is_running(paper["session_id"]) and paper["session_id"] not in jobs.claimed()):
        raise HTTPException(409, "This paper is being processed right now; try again in a minute")
    upgrade_to_full_text(store, topic, store.get_paper(paper["id"]), text, pdf, source="upload")
    store.update_paper(paper["id"], pdf_sha256=sha256)   # so the same file is recognised if uploaded again
    autopilot.notify()
    return store.get_paper(paper["id"])   # re-read: a candidate has just been given its ref


def _upload_with_hash(topic: str, sha256: str) -> str | None:
    """The paper of this topic that was uploaded (or attached) from exactly the same file, if any."""
    from src.research.store import get_store
    store = get_store()
    sid = (store.get_topic(topic) or {}).get("session_id")
    if sid:
        row = store.paper_with_pdf_hash(sid, sha256)
        if row:
            return row["ref"] or row["title"][:80]
    root = topic_dir(topic)
    for folder in root.iterdir() if root.is_dir() else []:
        meta = read_meta(topic, folder.name) if folder.is_dir() and not folder.name.startswith(("_", ".")) else {}
        if meta.get("sha256") == sha256:
            return meta.get("title") or folder.name.replace("_", " ")
    return None


def _claim_paper_folder(topic: str, base_id: str) -> tuple[str, Path]:
    """Atomically create an unused paper folder (mkdir fails if another request got there first)."""
    for n in range(1, 1000):
        candidate = base_id if n == 1 else f"{base_id[:112]}-{n}"
        folder = safe_path(topic, candidate)
        try:
            folder.mkdir()
            return candidate, folder
        except FileExistsError:
            continue
    raise HTTPException(409, "Too many papers with this file name")


@router.post("/ingest", status_code=202)
async def ingest(topic: str | None = None, reset: bool = False):
    if topic:
        safe_name(topic, "topic")
    job = report_jobs.create(topic or "*", kind="ingest")
    if job is None:
        raise HTTPException(409, "A job is already running for this topic")

    def work():
        jid = job["id"]
        try:
            if reset:
                report_jobs.update(jid, stage="clearing index")
                if topic:
                    delete_topic_papers(topic)   # only this topic is reloaded below
                else:
                    reset_index()
            report_jobs.update(jid, stage="loading PDFs")
            docs = load_papers(PAPERS_DIR, topic=topic)
            chunks = chunk_documents(docs)
            report_jobs.update(jid, stage="embedding", total=len(chunks), done=0)
            # Each paper's chunks are replaced atomically, so a failure part-way keeps the old index.
            stored = add_documents(chunks, progress=lambda done, total: report_jobs.update(jid, done=done))
            if topic and not reset:
                on_disk = {d.metadata["paper_id"] for d in docs}
                for p in list_papers(topic):
                    if p["paper_id"] not in on_disk and not library_paper_exists(topic, p["paper_id"]):
                        delete_paper(p["paper_id"], topic)  # folder removed outside the app
            report_jobs.update(jid, status="completed", stage="done", done=stored,
                               current=f"{len(docs)} pages · {stored} chunks")
        except Exception as e:
            logger.exception("Ingest failed")
            report_jobs.update(jid, status="failed", error=str(e)[:500])

    _spawn(run(work))
    return job


# ── Q&A ──

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    topic: str = ""
    session_id: str | None = Field(default=None, max_length=64)
    k: int = Field(default=10, ge=3, le=30)

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        return " ".join(v.split())


def conversation_key(req: QueryRequest) -> str | None:
    return f"{req.topic}:{req.session_id}" if req.session_id else None


def history_for(key: str | None) -> str:
    if not key:
        return ""
    with _conversations_lock:
        prev = list(_conversations.get(key, []))
    return "\n".join(f"Q: {h['question']}\nA (excerpt): {h['answer'][:600]}" for h in prev[-MAX_HISTORY:])


def remember(key: str | None, question: str, answer: str) -> None:
    if not key:
        return
    with _conversations_lock:
        items = _conversations.setdefault(key, [])
        items.append({"question": question, "answer": answer})
        del items[:-MAX_HISTORY]
        _conversations.move_to_end(key)
        while len(_conversations) > MAX_CONVERSATIONS:
            _conversations.popitem(last=False)


def conversations_clear(topic: str) -> None:
    with _conversations_lock:
        for key in [k for k in _conversations if k.startswith(f"{topic}:")]:
            _conversations.pop(key, None)


def initial_state(req: QueryRequest) -> dict:
    if req.topic:
        safe_name(req.topic, "topic")
    return {"question": req.question, "history": history_for(conversation_key(req)), "topic": req.topic,
            "k": req.k, "mode": "query", "documents": [], "errors": []}


def sources_of(docs) -> list[dict]:
    return [{"chunk_id": d.metadata.get("chunk_id"), "paper_id": d.metadata.get("paper_id"),
             "page": d.metadata.get("page"), "score": d.metadata.get("score"), "source": d.metadata.get("source"),
             "snippet": d.page_content[:400]} for d in docs]


@router.post("/query")
async def query(req: QueryRequest):
    state = initial_state(req)
    result = await run(research_graph.invoke, state)
    answer = result.get("report", "")
    remember(conversation_key(req), req.question, answer)
    return {"answer": answer, "sources": sources_of(result.get("documents", [])),
            "citation_check": check_citations(answer, result.get("documents", [])),
            "errors": list(dict.fromkeys(result.get("errors") or []))}


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    state = initial_state(req)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, default=str)}\n\n"

    async def events():
        final: dict = {"documents": [], "errors": [], "report": ""}
        yield sse({"type": "stage", "stage": STAGES[0], "status": "running"})
        try:
            async for update in research_graph.astream(state, stream_mode="updates"):
                for node, output in update.items():
                    output = output or {}
                    if "documents" in output:
                        final["documents"] = output["documents"]
                    final["errors"] += output.get("errors") or []
                    if "report" in output:
                        final["report"] = output["report"]
                    domain = output.get("domain") or {}
                    detail = {"retriever": f"{len(output.get('documents') or [])} passages",
                              "classifier": " · ".join(domain.get(k) for k in ("domain", "subfield", "research_type")
                                                       if domain.get(k))}
                    yield sse({"type": "stage", "stage": node, "status": "done", "detail": detail.get(node, ""),
                               "errors": output.get("errors") or []})
                    if node in STAGES and STAGES.index(node) + 1 < len(STAGES):
                        yield sse({"type": "stage", "stage": STAGES[STAGES.index(node) + 1], "status": "running"})
        except Exception as e:
            logger.exception("Streaming pipeline failed")
            final["errors"].append(f"Pipeline failed: {e.__class__.__name__}")
        remember(conversation_key(req), req.question, final["report"])
        yield sse({"type": "answer", "answer": final["report"], "sources": sources_of(final["documents"]),
                   "citation_check": check_citations(final["report"], final["documents"]),
                   "errors": list(dict.fromkeys(final["errors"]))})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── topic reports ──

class ReportRequest(BaseModel):
    topic: str


@router.post("/reports", status_code=202)
async def create_report(body: ReportRequest):
    topic = safe_name(body.topic, "topic")
    from src.llm.llm import free_requests_remaining
    remaining = await run(free_requests_remaining)
    if remaining is not None and remaining < REPORT_MIN_REQUESTS:
        raise HTTPException(429, f"Only {remaining} free LLM requests are left today; a report needs more. "
                                 "Try again after the daily reset (00:00 UTC).")
    job = report_jobs.create(topic, kind="report")
    if job is None:
        raise HTTPException(409, "A job is already running for this topic")

    def work():
        jid = job["id"]
        try:
            result = report_graph.invoke({
                "question": f"Literature report for {topic}", "topic": topic, "mode": "report", "job_id": jid,
                "documents": [], "errors": [], "history": "",
            })
            errors = list(dict.fromkeys(result.get("errors") or []))
            if not result.get("analysis") or result.get("report_failed"):
                # Nothing is written: a "could not be generated" page must not look like a finished report.
                report_jobs.update(jid, status="failed", error="; ".join(errors)[:800] or "No analysis produced")
                return
            reports = safe_path(topic, "_reports")
            reports.mkdir(parents=True, exist_ok=True)
            path = reports / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            path.write_text(result.get("report", ""), encoding="utf-8")
            report_jobs.update(jid, status="completed", stage="done", file=path.name,
                               **({"warning": "; ".join(errors)[:500]} if errors else {}))
        except Exception as e:
            logger.exception("Report generation failed")
            report_jobs.update(jid, status="failed", error=str(e)[:500])

    _spawn(run(work))
    return job


@router.get("/jobs")
async def list_jobs(topic: str | None = None):
    return {"jobs": report_jobs.list(topic)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = report_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/reports")
async def list_reports(topic: str):
    tdir = topic_dir(topic)
    if not tdir.is_dir():
        raise HTTPException(404, "Topic not found")

    def scan():
        out = []
        combined = tdir / "_reports"
        living = combined / "living_report.md"
        if living.exists():
            out.append({"type": "living", "filename": living.name, "paper_id": None,
                        "created_at": datetime.fromtimestamp(living.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
        if combined.is_dir():
            for f in sorted(combined.glob("report_*.md"), reverse=True):
                out.append({"type": "combined", "filename": f.name, "paper_id": None,
                            "created_at": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
        for paper in sorted(tdir.iterdir()):
            f = paper / "_reports" / f"{paper.name}_analysis.md"
            if paper.is_dir() and not paper.name.startswith("_") and f.exists():
                out.append({"type": "paper", "filename": f.name, "paper_id": paper.name,
                            "created_at": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
        return out
    return {"reports": await run(scan)}


def report_file(topic: str, filename: str) -> Path:
    filename = safe_name(filename, "file name")
    if not re.fullmatch(r"report_\d{8}_\d{6}\.md|living_report\.md", filename):
        raise HTTPException(400, "Invalid report file name")
    return safe_path(safe_name(topic, "topic"), "_reports", filename)


@router.get("/reports/{topic}/{filename}")
async def get_report(topic: str, filename: str):
    path = report_file(topic, filename)
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return {"topic": topic, "filename": filename, "content": await run(path.read_text, "utf-8")}


@router.delete("/reports/{topic}/{filename}")
async def delete_report(topic: str, filename: str):
    path = report_file(topic, filename)
    if not path.exists():
        raise HTTPException(404, "Report not found")
    await run(path.unlink)
    return {"status": "deleted"}


def _spawn(coro) -> None:
    """Run a background coroutine, keeping a reference so it is not garbage-collected mid-flight."""
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
