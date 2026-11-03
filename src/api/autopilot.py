"""Autopilot control API (/api/autopilot): choose the active topic, pause/resume, see the queue."""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.library import safe_name, topic_dir
from src.llm.llm import free_requests_remaining
from src.paths import living_report_path
from src.research.autopilot import QUEUE_STATUSES, autopilot
from src.research.store import get_store

router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])
run = asyncio.to_thread


class TopicRequest(BaseModel):
    topic: str
    focus: str | None = Field(default=None, max_length=2000)


class FocusRequest(BaseModel):
    focus: str = Field(default="", max_length=2000)


def _state() -> dict:
    store = get_store()
    ap = store.get_autopilot()
    topic = ap.get("active_topic")
    out = {
        "enabled": ap["enabled"], "state": ap["state"], "message": ap["state_message"],
        "active_topic": topic, "requested_topic": ap.get("requested_topic"),
        "next_search_at": ap.get("next_search_at"), "paused_until": ap.get("paused_until"),
        "last_error": ap.get("last_error") or "", "updated_at": ap.get("updated_at"),
        "free_requests_remaining": free_requests_remaining(),
        "current_paper": None, "queue": [], "recent": [], "session_id": None, "focus": "",
        "counts": {"queued": 0, "processed": 0, "selected": 0},
        "living_report": None,
    }
    if not topic:
        return out
    row = store.get_topic(topic) or {}
    out["focus"] = row.get("description", "")
    sid = row.get("session_id")
    if sid and store.get_session(sid):
        out["session_id"] = sid
        if ap.get("current_paper_id"):
            p = store.get_paper(ap["current_paper_id"])
            if p:
                out["current_paper"] = {k: p.get(k) for k in ("id", "ref", "title", "year", "venue", "processing_status")}
        out["queue"] = store.queued_papers(sid, QUEUE_STATUSES, limit=12)
        out["recent"] = store.recent_processed(sid, limit=12)
        selected = store.list_papers(sid, selection="selected")
        out["counts"] = {"queued": sum(1 for p in selected if p["processing_status"] in QUEUE_STATUSES),
                         "processed": store.count_processed(sid), "selected": len(selected)}
    path = living_report_path(topic)
    if path.exists():
        from datetime import datetime, timezone
        out["living_report"] = {"filename": path.name,
                                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    return out


@router.get("")
async def get_autopilot():
    return await run(_state)


@router.post("/topic")
async def set_topic(body: TopicRequest):
    topic = safe_name(body.topic, "topic")
    if not topic_dir(topic).is_dir():
        raise HTTPException(404, "Topic not found")
    status = await run(request_topic, topic, body.focus)
    return {"status": status, **(await run(_state))}


def request_topic(topic: str, focus: str | None = None) -> str:
    """Put `topic` on autopilot. Returns "active", or "pending" when the worker is mid-task: it switches
    as soon as the current paper (or search round) is finished."""
    store = get_store()
    if focus is not None:
        _set_focus(topic, focus)
    ap = store.get_autopilot()
    if ap.get("active_topic") == topic:
        store.update_autopilot(requested_topic=None, enabled=True)
        status = "active"
    elif ap["state"] in ("processing", "searching", "screening", "verifying") and ap.get("active_topic"):
        store.update_autopilot(requested_topic=topic, enabled=True)
        status = "pending"
    else:
        # paused_until is kept: switching topics doesn't lift a quota or outage pause.
        store.update_autopilot(active_topic=topic, requested_topic=None, enabled=True, next_search_at=None,
                               current_paper_id=None, state="idle",
                               state_message=f"Starting on {topic.replace('_', ' ')}")
        status = "active"
    autopilot.notify()
    return status


def _set_focus(topic: str, focus: str) -> None:
    store = get_store()
    focus = " ".join(focus.split())
    row = store.ensure_topic(topic)
    if row.get("description", "") == focus:
        return
    store.update_topic(topic, description=focus)
    if row.get("session_id") and store.get_session(row["session_id"]):
        # A new focus means new searches: re-analyse the topic and search again soon.
        store.update_session(row["session_id"], description=focus, topic_analysis=None)
        if store.get_autopilot().get("active_topic") == topic:
            store.update_autopilot(next_search_at=None)


@router.put("/topics/{topic}/focus")
async def set_focus(topic: str, body: FocusRequest):
    topic = safe_name(topic, "topic")
    if not topic_dir(topic).is_dir():
        raise HTTPException(404, "Topic not found")
    await run(_set_focus, topic, body.focus)
    autopilot.notify()
    return {"topic": topic, "focus": " ".join(body.focus.split())}


@router.post("/pause")
async def pause():
    await run(get_store().update_autopilot, enabled=False)
    autopilot.notify()
    return await run(_state)


@router.post("/resume")
async def resume():
    await run(get_store().update_autopilot, enabled=True, paused_until=None)
    autopilot.notify()
    return await run(_state)


@router.post("/search-now")
async def search_now():
    await run(get_store().update_autopilot, next_search_at=None)
    autopilot.notify()
    return await run(_state)
