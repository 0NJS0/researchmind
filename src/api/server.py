"""FastAPI application: library + research APIs, health, and the built React app."""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.autopilot import router as autopilot_router
from src.api.library import router as library_router
from src.api.validator import router as validator_router
from src.config import AUTOPILOT_ENABLED, CORS_ORIGINS, public_settings, validate_config
from src.db import check_database, init_schema
from src.ingest.vectorstore import check_vectorstore
from src.llm.llm import check_llm
from src.research.api import router as research_router
from src.research.autopilot import autopilot
from src.research.enrich import check_openalex
from src.research.searxng import SearXNGClient
from src.research.store import get_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs at INFO; OpenAlex takes its API key as a URL parameter, so keep those quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HEALTH_TTL = 30  # seconds; health checks call external services, so results are cached briefly
DB_RETRY_SECONDS = 5  # first retry delay when PostgreSQL is not reachable at startup
DB_RETRY_MAX_SECONDS = 60
DB_STARTUP_WAIT_SECONDS = 20


def _prepare_database() -> None:
    init_schema()
    # Background jobs live in-process; sessions that were running when the server stopped cannot resume.
    count = get_store().mark_interrupted_sessions()
    from src.research.validator import mark_interrupted
    mark_interrupted()
    if count:
        logger.info("Marked %d interrupted research session(s)", count)
    # Whatever the autopilot was doing before the restart is no longer running; it resumes from the queue.
    store = get_store()
    ap = store.get_autopilot()
    if ap.get("state") in ("processing", "searching", "screening", "verifying") or ap.get("current_paper_id"):
        store.update_autopilot(state="idle", state_message="Resuming after a restart", current_paper_id=None)


async def _start_when_database_ready() -> None:
    """Prepare the database, then start the autopilot. If PostgreSQL is not reachable, keep retrying.

    After a homelab reboot the database often comes up after this container, so a single attempt at
    startup is not enough. The server keeps running meanwhile so /health and the UI can explain what is wrong."""
    delay = DB_RETRY_SECONDS
    while True:
        try:
            await asyncio.to_thread(_prepare_database)
            break
        except Exception as e:
            logger.error("Database not ready (retrying in %ds): %s", delay, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, DB_RETRY_MAX_SECONDS)
    if AUTOPILOT_ENABLED:
        autopilot.start()  # resumes the active topic from the database


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Report/re-index jobs and Ask pipelines hold worker threads for minutes; the default pool (cpu+4)
    # would leave ordinary requests queued behind them.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=64, thread_name_prefix="api"))
    task = asyncio.create_task(_start_when_database_ready())
    # Normally the database is ready within a moment: wait for it so the first requests see a prepared schema.
    await asyncio.wait({task}, timeout=DB_STARTUP_WAIT_SECONDS)
    yield
    task.cancel()
    if AUTOPILOT_ENABLED:
        await asyncio.to_thread(autopilot.stop, 5)


app = FastAPI(title="ResearchMind", lifespan=lifespan)
app.include_router(library_router)
app.include_router(research_router)
app.include_router(autopilot_router)
app.include_router(validator_router)

if CORS_ORIGINS:  # only needed when the UI is served from another origin (e.g. the Vite dev server)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])


_health_cache: dict = {"at": 0.0, "value": None}


def _health() -> dict:
    try:
        validate_config()
        config = {"ok": True, "message": "validated"}
    except ValueError as e:
        config = {"ok": False, "message": str(e)}
    db_ok, db_msg = check_database()
    vs_ok, vs_msg = check_vectorstore() if db_ok else (False, "database unavailable")
    llm_ok, llm_msg = check_llm()
    searx_ok, searx_msg = SearXNGClient().check()
    oa_ok, oa_msg = check_openalex()
    ap_ok, ap_msg = _autopilot_health() if db_ok else (False, "database unavailable")
    core_ok = config["ok"] and db_ok and llm_ok
    return {
        "status": "healthy" if core_ok and vs_ok and searx_ok else ("degraded" if core_ok else "unhealthy"),
        "config": config,
        "database": {"ok": db_ok, "message": db_msg},
        "vectorstore": {"ok": vs_ok, "message": vs_msg},
        "llm": {"ok": llm_ok, "message": llm_msg},
        "searxng": {"ok": searx_ok, "message": searx_msg},
        "openalex": {"ok": oa_ok, "message": oa_msg},
        "autopilot": {"ok": ap_ok, "message": ap_msg},
    }


AUTOPILOT_STALE_MINUTES = 30  # longer than any single step (a slow paper with retries takes minutes)


def _autopilot_health() -> tuple[bool, str]:
    """Is the worker loop alive? It records a heartbeat before every step."""
    if not AUTOPILOT_ENABLED:
        return True, "disabled (AUTOPILOT_ENABLED=false)"
    from datetime import datetime, timezone
    ap = get_store().get_autopilot()
    beat = ap.get("heartbeat_at")
    if not beat:
        return False, "worker has not started yet"
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds() / 60
    if age > AUTOPILOT_STALE_MINUTES:
        return False, f"no heartbeat for {age:.0f} min — the worker may be stuck; restart the container"
    state = ap.get("state_message") or ap.get("state") or ""
    return True, f"alive · {state[:80]}"


async def _cached_health(refresh: bool = False) -> dict:
    now = time.monotonic()
    if refresh or _health_cache["value"] is None or now - _health_cache["at"] > HEALTH_TTL:
        _health_cache["value"] = await asyncio.to_thread(_health)
        _health_cache["at"] = now
    return _health_cache["value"]


@app.get("/health")
async def health():
    """Liveness for Docker: the process is up. Service details are at /api/health."""
    return {"status": "ok"}


@app.get("/api/health")
async def api_health(refresh: bool = False):
    return await _cached_health(refresh)


@app.get("/api/config")
async def api_config():
    return public_settings()


# ── React app (frontend/dist), mounted last so API routes take precedence ──

_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class _SPAStaticFiles(StaticFiles):
    """Static files with an index.html fallback so client-side routes (e.g. /research/<id>) load."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            # Not for missing build assets: after a redeploy an old tab asks for old chunk names, and answering
            # with index.html (text/html) makes the lazy import fail obscurely; a 404 lets the app reload itself.
            if e.status_code == 404 and not path.startswith(("api/", "assets/")):
                return await super().get_response("index.html", scope)
            raise


if _FRONTEND_DIST.is_dir():
    app.mount("/", _SPAStaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
