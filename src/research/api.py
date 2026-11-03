"""REST + SSE API for research sessions, mounted under /api."""
import asyncio
import hashlib
import json
import re

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from src.paths import to_name
from src.research.dedup import find_duplicate, merge_metadata
from src.research.jobs import jobs
from src.research.normalize import canonical_url, clean_arxiv_id, clean_doi, clean_text
from src.research.store import ACTIVE_STATUSES, get_store

router = APIRouter(prefix="/api", tags=["research"])

ID_RE = re.compile(r"^[0-9a-f]{32}$")
ACTIVE = ACTIVE_STATUSES
run = asyncio.to_thread  # every store call is blocking I/O: keep it off the event loop


# ── request models ──

class CreateResearch(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    description: str = Field(default="", max_length=2000)
    auto_start: bool = True

    @field_validator("topic", "description")
    @classmethod
    def _clean(cls, v: str) -> str:
        return " ".join(v.split())


class SearchRequest(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=10)
    iterations: int = Field(default=1, ge=1, le=5)

    @field_validator("queries")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        out = []
        for q in v:
            q = " ".join(str(q).split())
            if q:
                if len(q) > 200:
                    raise ValueError("queries must be at most 200 characters")
                out.append(q)
        return out


class AddPaper(BaseModel):
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    doi: str = Field(default="", max_length=200)
    abstract: str = Field(default="", max_length=20000)
    authors: list[str] = Field(default_factory=list, max_length=50)
    year: int | None = Field(default=None, ge=1800, le=2100)

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.strip()
        if v and not canonical_url(v):
            raise ValueError("url must be an http(s) URL")
        return v


# ── helpers ──

def _check_id(value: str) -> str:
    if not ID_RE.match(value or ""):
        raise HTTPException(404, "Not found")
    return value


async def _session_or_404(session_id: str) -> dict:
    session = await run(get_store().get_session, _check_id(session_id))
    if not session:
        raise HTTPException(404, "Research session not found")
    return session


async def _paper_or_404(paper_id: str) -> dict:
    paper = await run(get_store().get_paper, _check_id(paper_id))
    if not paper:
        raise HTTPException(404, "Paper not found")
    return paper


def _autopilot_role(topic: str | None, ap: dict) -> str:
    """How the autopilot relates to a topic: "active", "next" (switches after the current task) or "idle"."""
    if not topic:
        return "idle"
    if ap.get("requested_topic") == topic:
        return "next"
    return "active" if ap.get("active_topic") == topic and ap.get("enabled") else "idle"


def _public_session(session: dict) -> dict:
    store = get_store()
    topic = store.topic_for_session(session["id"])
    return {
        "topic_name": topic,
        "autopilot_role": _autopilot_role(topic, store.get_autopilot()),
        "id": session["id"],
        "topic": session["topic"],
        "description": session["description"],
        "status": session["status"],
        "status_message": session["status_message"],
        "topic_analysis": session.get("topic_analysis"),
        "iteration": session["iteration"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "synthesis_updated_at": session.get("synthesis_updated_at"),
        "counts": store.session_counts(session["id"]),
        "running": jobs.is_running(session["id"]),
        "autopilot": session["id"] in jobs.claimed(),
    }


def _public_paper(paper: dict) -> dict:
    paper = {k: v for k, v in paper.items() if k != "raw_content"}
    paper.setdefault("content_length", None)
    return paper


async def _start(session_id: str, job: str, **kwargs) -> None:
    if jobs.is_running(session_id):
        raise HTTPException(409, "The agent is already working on this session")
    await run(get_store().update_session, session_id,
              status="searching" if job == "research" else "processing", status_message="Starting")
    if not jobs.start(get_store(), session_id, job, **kwargs):
        raise HTTPException(409, "The agent is already working on this session")


# ── sessions ──

@router.get("/research")
async def list_research():
    store = get_store()
    sessions = await run(store.list_sessions)
    topics = await run(store.topics_by_session)
    ap = await run(store.get_autopilot)
    for s in sessions:
        s["running"] = jobs.is_running(s["id"])
        s["topic_name"] = topics.get(s["id"])
        s["autopilot_role"] = _autopilot_role(s["topic_name"], ap)
    return {"sessions": sessions}


@router.post("/research", status_code=201)
async def create_research(body: CreateResearch):
    """Starting research on a topic = a Library topic researched continuously by the autopilot.

    An existing topic (e.g. one with uploaded PDFs) is reused. If the autopilot is busy with another topic it
    switches after the current paper."""
    from src.api.library import safe_name, topic_dir
    name = safe_name(to_name(body.topic), "topic name")
    folder = topic_dir(name)
    existed = folder.is_dir()
    return await run(_start_topic_research, name, folder, body.topic, body.description, existed)


def _start_topic_research(name: str, folder, title: str, focus: str, existed: bool) -> dict:
    from src.api.autopilot import _set_focus, request_topic
    from src.research.autopilot import autopilot
    store = get_store()
    folder.mkdir(parents=True, exist_ok=True)
    _set_focus(name, focus)
    sid = autopilot.ensure_session(name)
    session = store.get_session(sid)
    if not existed and session["topic"] != title:
        store.update_session(sid, topic=title)       # keep the wording the user typed
    status = request_topic(name)
    store.add_event(sid, "info", "Research started on the autopilot" if status == "active"
                    else "Queued on the autopilot: starts after its current paper")
    return {**_public_session(store.get_session(sid)), "autopilot_status": status}


def _topic_of(session: dict) -> str:
    """The Library topic behind a research session; older one-off sessions become a topic now."""
    from src.api.library import safe_name, topic_dir
    store = get_store()
    name = store.topic_for_session(session["id"])
    if name:
        return name
    base = to_name(session["topic"]) or "research"
    name, n = base, 2
    # A new, unused name: never attach the session to an existing Library topic (uploads, another research).
    while store.get_topic(name) or topic_dir(name).exists():
        name, n = f"{base[:110]}_{n}", n + 1
    safe_name(name, "topic name")
    topic_dir(name).mkdir(parents=True, exist_ok=True)
    store.update_topic(name, session_id=session["id"], description=session.get("description") or "")
    store.update_session(session["id"], status="autopilot", status_message="Moved to the autopilot")
    return name


@router.get("/research/{session_id}")
async def get_research(session_id: str):
    session = await _session_or_404(session_id)
    return await run(_public_session, session)


@router.delete("/research/{session_id}")
async def delete_research(session_id: str):
    session = await _session_or_404(session_id)
    if session["id"] in jobs.claimed():
        raise HTTPException(409, "The autopilot is working on this research. Switch the autopilot to another "
                                 "topic first (or delete the topic in the Library).")
    topic = await run(get_store().topic_for_session, session["id"])
    if topic:
        raise HTTPException(409, f"This research belongs to the topic '{topic}'. Delete the topic instead: "
                                 "that removes its papers, reports and research together.")
    if jobs.is_running(session["id"]):
        jobs.cancel(session["id"])
        await jobs.wait(session["id"])
    await run(get_store().delete_session, session["id"])
    return {"status": "deleted"}


@router.post("/research/{session_id}/search", status_code=202)
async def continue_search(session_id: str, body: SearchRequest):
    """Research continues on the autopilot: put the topic on it (switching after the current paper) and,
    if it is already the active topic, search for new papers now."""
    session = await _session_or_404(session_id)
    if jobs.is_running(session["id"]) and session["id"] not in jobs.claimed():
        raise HTTPException(409, "A job is still running on this research; wait for it or stop it first")

    def apply():
        from src.api.autopilot import request_topic
        store = get_store()
        status = request_topic(_topic_of(session))
        if status == "active":
            store.update_autopilot(next_search_at=None)
        return {**_public_session(store.get_session(session["id"])), "autopilot_status": status}
    return await run(apply)


@router.post("/research/{session_id}/cancel")
async def cancel_research(session_id: str):
    session = await _session_or_404(session_id)
    if not jobs.cancel(session["id"]):
        if session["status"] in ACTIVE:  # stale state, e.g. after a crash
            await run(get_store().update_session, session["id"], status="cancelled", status_message="Stopped")
        return {"status": "not_running"}
    return {"status": "cancelling"}


@router.get("/research/{session_id}/queries")
async def list_queries(session_id: str):
    session = await _session_or_404(session_id)
    return {"queries": await run(get_store().list_queries, session["id"])}


# ── papers ──

@router.get("/research/{session_id}/papers")
async def list_papers(session_id: str, selection: str = "selected"):
    session = await _session_or_404(session_id)
    if selection not in ("selected", "rejected", "candidate", "all"):
        raise HTTPException(422, "selection must be selected, rejected, candidate or all")
    papers = await run(get_store().list_papers, session["id"],
                                     None if selection == "all" else selection)
    return {"papers": [_public_paper(p) for p in papers]}


@router.post("/research/{session_id}/papers", status_code=201)
async def add_paper(session_id: str, body: AddPaper):
    """Manually add a paper by URL / DOI / title. It is de-duplicated and then processed."""
    session = await _session_or_404(session_id)
    return await run(_add_paper_sync, session, body)


def _add_paper_sync(session: dict, body: AddPaper) -> dict:
    store = get_store()
    doi = clean_doi(body.doi) or clean_doi(body.url)
    arxiv_id = clean_arxiv_id(body.url)
    title = clean_text(body.title)
    if not (title or body.url or doi):
        raise HTTPException(422, "Provide at least a title, URL or DOI")
    url = body.url or (f"https://doi.org/{doi}" if doi else "")
    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    elif url.lower().split("?", 1)[0].endswith(".pdf"):
        pdf_url = url  # a direct PDF link: fetch it as the paper's full text
    else:
        pdf_url = ""
    candidate = {
        "title": title or url, "url": url, "doi": doi, "arxiv_id": arxiv_id,
        "abstract": clean_text(body.abstract), "authors": [clean_text(a) for a in body.authors if clean_text(a)],
        "year": body.year, "pdf_url": pdf_url,
        "source": canonical_url(url).split("/")[2] if canonical_url(url) else "manual",
    }
    existing = store.list_papers(session["id"], selection=None)
    dup = find_duplicate(candidate, existing)
    if dup:
        updates = merge_metadata(dup, candidate)
        if updates:
            store.update_paper(dup["id"], **updates)
        if dup["selection_status"] == "selected":
            return {"status": "duplicate", "paper": _public_paper(store.get_paper(dup["id"]))}
        paper = store.select_paper(dup["id"], relevance_reason="Added manually by the user",
                                   relevance_method="manual", processing_status="pending")
    else:
        paper = store.add_paper(session["id"], {**candidate, "selection_status": "selected",
                                                "relevance_reason": "Added manually by the user",
                                                "relevance_method": "manual", "added_by": "user"},
                                assign_ref=True)
    store.add_event(session["id"], "paper", f"{paper['ref']} added manually: {paper['title']}",
                    {"paper_id": paper["id"], "ref": paper["ref"]})
    started = False
    if store.topic_for_session(session["id"]):
        # The autopilot processes it: next if this topic is active, otherwise when the topic is next on autopilot.
        from src.research.autopilot import autopilot
        autopilot.notify()
        return {"status": "added", "paper": _public_paper(store.get_paper(paper["id"])), "processing": False,
                "queued": True}
    if jobs.is_running(session["id"]):
        # The running job may already be past its processing step: process leftovers when it ends.
        jobs.request_followup(session["id"])
    else:
        store.update_session(session["id"], status="processing", status_message="Processing added paper")
        started = jobs.start(store, session["id"], "process", paper_ids=None)
    return {"status": "added", "paper": _public_paper(store.get_paper(paper["id"])), "processing": started}


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    paper = await _paper_or_404(paper_id)
    store = get_store()

    def load():
        return {
            "paper": {**_public_paper(paper), "content_length": len(paper["raw_content"] or "")},
            "summary": store.get_summary(paper["id"]),
            "relations": store.relations_for_paper(paper["id"]),
            "gaps": store.gaps_for_paper(paper["id"]),
        }
    return await run(load)


@router.get("/papers/{paper_id}/summary")
async def get_paper_summary(paper_id: str):
    paper = await _paper_or_404(paper_id)
    summary = await run(get_store().get_summary, paper["id"])
    if not summary:
        raise HTTPException(404, "No summary yet")
    return summary


@router.post("/papers/{paper_id}/analyze", status_code=202)
async def analyze_paper(paper_id: str, force: bool = True):
    paper = await _paper_or_404(paper_id)
    if jobs.is_running(paper["session_id"]):  # refuse before touching the paper's state
        raise HTTPException(409, "The agent is already working on this session")
    store = get_store()
    if paper["selection_status"] != "selected":
        await run(store.select_paper, paper["id"], relevance_reason="Selected manually by the user",
                  relevance_method="manual")
    await run(lambda: store.update_paper(paper["id"], processing_status="pending", published_at=None))
    await _start(paper["session_id"], "process", paper_ids=[paper["id"]], force=force)
    return {"status": "started"}


MAX_ATTACH_BYTES = 100 * 1024 * 1024


@router.post("/papers/{paper_id}/pdf")
async def attach_paper_pdf(paper_id: str, file: UploadFile, request: Request):
    """Attach the PDF of a paper the agent could only read the abstract of (e.g. downloaded from SSRN in the
    browser). The paper is re-analysed from the full text by the autopilot, ahead of new papers."""
    from src.research.content import MIN_FULL_TEXT_CHARS, extract_pdf_text
    from src.research.topic_ops import upgrade_to_full_text

    paper = await _paper_or_404(paper_id)
    store = get_store()
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_ATTACH_BYTES + 1024 * 1024:
        raise HTTPException(413, "File too large (max 100 MB)")
    data = await file.read(MAX_ATTACH_BYTES + 1)
    await file.close()
    if len(data) > MAX_ATTACH_BYTES:
        raise HTTPException(413, "File too large (max 100 MB)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")
    try:
        text = await run(extract_pdf_text, data)
    except Exception:
        raise HTTPException(400, "This PDF could not be read")
    if len(text) < MIN_FULL_TEXT_CHARS:
        raise HTTPException(422, "No extractable text in this PDF (scanned documents need OCR first)")

    def apply():
        ap = store.get_autopilot()
        if ap.get("current_paper_id") == paper["id"] or (jobs.is_running(paper["session_id"])
                                                         and paper["session_id"] not in jobs.claimed()):
            raise HTTPException(409, "This paper is being processed right now; try again in a minute")
        topic = store.topic_for_session(paper["session_id"]) or _topic_of(store.get_session(paper["session_id"]))
        upgrade_to_full_text(store, topic, store.get_paper(paper["id"]), text, data, source="upload")
        store.update_paper(paper["id"], pdf_sha256=hashlib.sha256(data).hexdigest())
        return {"status": "queued", "topic": topic,
                "autopilot_role": _autopilot_role(topic, store.get_autopilot()),
                "paper": _public_paper(store.get_paper(paper["id"]))}

    result = await run(apply)
    from src.research.autopilot import autopilot
    autopilot.notify()
    return result


# ── gaps & synthesis ──

@router.get("/research/{session_id}/gaps")
async def list_gaps(session_id: str, include_merged: bool = False):
    session = await _session_or_404(session_id)
    return {"gaps": await run(get_store().list_gaps, session["id"], include_merged)}


@router.get("/gaps/{gap_id}")
async def get_gap(gap_id: str):
    store = get_store()
    gap = await run(store.get_gap, _check_id(gap_id))
    if not gap:
        raise HTTPException(404, "Gap not found")

    def load():
        merged_into = store.get_gap(gap["merged_into"]) if gap.get("merged_into") else None
        return {
            "gap": gap,
            "evidence": store.evidence_for_gap(gap["id"]),
            "history": store.gap_history(gap["id"]),
            "merged_into": {"id": merged_into["id"], "ref": merged_into["ref"], "title": merged_into["title"]}
            if merged_into else None,
        }
    return await run(load)


@router.get("/research/{session_id}/synthesis")
async def get_synthesis(session_id: str):
    session = await _session_or_404(session_id)
    return {"synthesis": session.get("synthesis"), "updated_at": session.get("synthesis_updated_at")}


@router.post("/research/{session_id}/analyze-gaps", status_code=202)
async def analyze_gaps(session_id: str):
    session = await _session_or_404(session_id)
    await _start(session["id"], "gaps")
    return {"status": "started"}


# ── activity ──

@router.get("/research/{session_id}/events")
async def list_events(session_id: str, after: int = 0, latest: bool = False):
    session = await _session_or_404(session_id)
    return {"events": await run(lambda: get_store().list_events(session["id"], max(0, after), latest=latest))}


@router.get("/research/{session_id}/events/stream")
async def stream_events(session_id: str, after: int = 0):
    session = await _session_or_404(session_id)
    sid = session["id"]
    store = get_store()

    async def generator():
        cursor = max(0, after)
        idle = 0
        while True:
            events = await run(store.list_events, sid, cursor)
            for e in events:
                cursor = e["id"]
                yield f"id: {e['id']}\nevent: activity\ndata: {json.dumps(e)}\n\n"
            if events:
                idle = 0
            else:
                idle += 1
                if idle % 15 == 0:
                    yield ": keep-alive\n\n"
            current = await run(store.get_session, sid)
            if current is None:
                break
            if not jobs.is_running(sid) and current["status"] not in ACTIVE and not events:
                # Drain events written between the last poll and the job finishing (e.g. "Completed").
                for e in await run(store.list_events, sid, cursor):
                    cursor = e["id"]
                    yield f"id: {e['id']}\nevent: activity\ndata: {json.dumps(e)}\n\n"
                yield f"event: status\ndata: {json.dumps({'status': current['status']})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
