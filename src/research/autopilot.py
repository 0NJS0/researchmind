"""The autopilot: continuous research on the user's active topic.

One worker thread, forever:
  switch topic if requested → (paused? wait) → take the highest-priority queued paper and process it
  completely (summary → comparison → gaps → synthesis) → publish it to the Library and the topic's
  reports → next paper. When the queue is empty it searches SearXNG for more (expanding queries from
  the papers and gaps so far), screens and ranks the results, and if nothing new turns up it waits
  AUTOPILOT_RESEARCH_INTERVAL_HOURS before searching again.

A topic switch is only applied between tasks, so the paper (or search round) in progress always
finishes first. State lives in the `autopilot` table, so the worker resumes after restarts.
"""
import logging
import threading
from datetime import datetime, timedelta, timezone

import psycopg

from src.config import (
    AUTOPILOT_IDLE_SECONDS, AUTOPILOT_MAX_PAPERS_PER_TOPIC, AUTOPILOT_QUOTA_RESERVE,
    AUTOPILOT_RESEARCH_INTERVAL_HOURS, DATABASE_URL, DB_SCHEMA, DISCOVERY_ENABLED, NARRATIVE_EVERY,
)
from src.llm.llm import (
    LLMError, LLMHTTPError, NetworkDown, QuotaExhausted, RateLimited, free_requests_remaining, internet_reachable,
    is_outage,
)
from src.llm.structured import StructuredOutputError
from src.paths import analysis_path, living_report_path, papers_root
from src.research import llm_tasks
from src.research.agent import ResearchAgent, _outage_message
from src.research.content import fetch_content, fetch_pdf
from src.research.dedup import find_duplicate
from src.research.enrich import OpenAlexClient, enrich_fields, parse_work
from src.research.literature import LiteratureSearch
from src.research.searxng import SearchUnavailable
from src.research.topic_ops import upgrade_to_full_text
from src.research.verify import next_gap_to_verify, verify_gap
from src.research.jobs import jobs
from src.research.publish import (
    attach_pdf, incoming_pdf_path, incoming_text_path, library_paper_exists, publish_to_library, read_meta,
    replace_library_text, seed_library_papers,
)
from src.research.ranking import priority
from src.research.report_render import render_living_report, render_paper_analysis
from src.research.store import ResearchStore, get_store
from src.research.synthesis import render_synthesis

logger = logging.getLogger(__name__)

QUEUE_STATUSES = ("pending", "fetching", "summarizing", "analyzing")
PROCESSED_OK = ("done", "analysis_failed")
LOCK_KEY = "researchmind_autopilot"
GAP_CHECK_EVERY = 3  # papers processed between gap duplication tests (when the queue isn't empty)
MAX_ATTEMPTS = 3     # a paper that fails this often is left as failed
RETRY_DELAY = timedelta(minutes=30)       # × attempt number
QUOTA_RECHECK = timedelta(minutes=30)     # a low-quota pause re-reads the real remaining budget this often
PUBLISH_RETRY = timedelta(minutes=30)     # how often to retry papers that were analysed but not published
AUTOPILOT_MARK = "<!-- written by the ResearchMind autopilot -->\n"  # rewritten freely; hand-made analyses are kept
FULLTEXT_RECHECK = timedelta(days=7)      # how often abstract-only papers are checked for a free full text
DRIP_MIN_SELECTED = 2                     # a round selecting fewer papers than this waits DRIP_WAIT
DRIP_WAIT = timedelta(hours=3)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_dt(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def next_quota_reset(now: datetime) -> datetime:
    """OpenRouter's free-model daily counters reset at 00:00 UTC (resume a few minutes after)."""
    today = now.replace(hour=0, minute=5, second=0, microsecond=0)
    return today if now < today else today + timedelta(days=1)


def _lock_alive(conn) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


class Autopilot:
    def __init__(self, store: ResearchStore | None = None, *, agent_factory=ResearchAgent,
                 quota_check=free_requests_remaining, enricher: OpenAlexClient | None = None, clock=utcnow):
        self._store = store
        self.agent_factory = agent_factory
        self.quota_check = quota_check
        self.enricher = enricher or OpenAlexClient()
        self.clock = clock
        self.wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._claimed_session: str | None = None
        self._publish_check_at: dict[str, datetime] = {}  # when to look for unpublished papers again, per session
        self._gap_checks_after: datetime | None = None  # gap checks postponed while search is down
        self._pruned_at: datetime | None = None

    @property
    def store(self) -> ResearchStore:
        return self._store or get_store()

    # ── lifecycle ──

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._publish_check_at.clear()
        self._thread = threading.Thread(target=self._run, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self.wake.set()
        if self._thread:
            self._thread.join(timeout)

    def notify(self) -> None:
        """Wake the worker early (topic changed, resumed, search requested)."""
        self.wake.set()

    def _sleep(self, seconds: float) -> None:
        self.wake.wait(max(0.0, seconds))
        self.wake.clear()

    def _run(self) -> None:
        lock_conn = None
        while not self._stop.is_set():
            if lock_conn is not None and not _lock_alive(lock_conn):
                # The database restarted: the advisory lock is gone with the old connection, so another
                # worker may have taken it. Re-acquire before doing anything else.
                logger.warning("Autopilot lost its database lock; re-acquiring")
                _close_quietly(lock_conn)
                lock_conn = None
            if lock_conn is None:
                lock_conn = self._acquire_lock()
                if lock_conn is None:
                    # Another process holds the autopilot (or a just-stopped server hasn't released it yet).
                    self._sleep(10)
                    continue
            try:
                self._housekeeping()
                delay = self.step()
            except Exception as e:
                logger.exception("Autopilot step failed")
                self._safe_update(state="error", state_message="Unexpected error — retrying in a minute",
                                  last_error=f"{e.__class__.__name__}: {str(e)[:300]}")
                delay = 60
            if delay:
                self._sleep(delay)
        if lock_conn is not None:
            lock_conn.close()
        if self._claimed_session:
            jobs.release(self._claimed_session)

    def _housekeeping(self) -> None:
        """A heartbeat for /api/health (a hung worker becomes visible) and daily retention."""
        self.store.heartbeat()
        now = self.clock()
        if self._pruned_at is None or now - self._pruned_at > timedelta(days=1):
            self._pruned_at = now
            removed = self.store.prune()
            if any(removed.values()):
                logger.info("Retention: removed %(events)d old events and %(llm_cache)d old cache entries", removed)

    def _acquire_lock(self):
        """Session-level advisory lock on a dedicated connection: at most one autopilot per database."""
        try:
            conn = psycopg.connect(DATABASE_URL, autocommit=True, options=f"-c search_path={DB_SCHEMA},public",
                                   connect_timeout=10, keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                   keepalives_count=3)
            got = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (f"{LOCK_KEY}:{DB_SCHEMA}",)).fetchone()[0]
            if got:
                return conn
            conn.close()
        except psycopg.Error as e:
            logger.warning("Autopilot could not reach the database: %s", e.__class__.__name__)
        return None

    def _safe_update(self, **fields) -> None:
        try:
            self.store.update_autopilot(**fields)
        except Exception:
            logger.exception("Could not update autopilot state")

    def _set(self, state: str, message: str = "", **fields) -> None:
        self.store.update_autopilot(state=state, state_message=message, **fields)

    # ── one unit of work ──

    def step(self) -> float:
        """Do at most one task. Returns how long to sleep before the next step (0 = continue)."""
        store = self.store
        ap = store.get_autopilot()

        # 1. Topic switches take effect only here, i.e. after the previous task finished.
        if ap.get("requested_topic") and ap["requested_topic"] != ap.get("active_topic"):
            store.update_autopilot(active_topic=ap["requested_topic"], requested_topic=None, next_search_at=None,
                                   current_paper_id=None, state="idle",  # a quota/outage pause stays
                                   state_message=f"Switched to {ap['requested_topic'].replace('_', ' ')}")
            ap = store.get_autopilot()
        elif ap.get("requested_topic"):
            store.update_autopilot(requested_topic=None)

        topic = ap.get("active_topic")
        if not ap.get("enabled"):
            self._release()
            if ap.get("state") != "stopped":
                self._set("stopped", "Paused by you", current_paper_id=None)
            return AUTOPILOT_IDLE_SECONDS
        if not topic:
            self._release()
            if ap.get("state") != "idle" or ap.get("state_message") != "Choose a topic to start":
                self._set("idle", "Choose a topic to start", current_paper_id=None)
            return AUTOPILOT_IDLE_SECONDS
        if not (papers_root() / topic).is_dir():
            store.update_autopilot(active_topic=None, state="idle", state_message="The active topic was deleted")
            return 0

        now = self.clock()
        paused_until = _as_dt(ap.get("paused_until"))
        if paused_until and now < paused_until:
            return min(300.0, (paused_until - now).total_seconds())
        if paused_until:
            store.update_autopilot(paused_until=None)

        remaining = self.quota_check()
        if remaining is not None and remaining <= AUTOPILOT_QUOTA_RESERVE:
            self._pause_for_quota(f"Daily free-request budget used ({remaining} left, kept for Ask)", recheck=True)
            return 60

        session_id = self.ensure_session(topic)
        if not self._claim(session_id):
            self._set("waiting", "Waiting for the manual research job on this topic to finish", current_paper_id=None)
            return AUTOPILOT_IDLE_SECONDS
        agent = self.agent_factory(store, session_id)
        try:
            return self._work(agent, ap, topic, session_id, now)
        except LLMError as e:
            if not is_outage(e):
                raise
            return self._pause_for_outage(e)

    def _work(self, agent: ResearchAgent, ap: dict, topic: str, session_id: str, now: datetime) -> float:
        store = self.store
        due = self._publish_check_at.get(session_id)
        if due is None or now >= due:
            # At start-up, then every PUBLISH_RETRY: papers whose publish was cut off (restart) or failed
            # (e.g. no internet for the embeddings) reach the Library and report once things work again.
            self._publish_check_at[session_id] = now + PUBLISH_RETRY
            try:
                self._publish_unpublished(agent, topic, session_id)
            except QuotaExhausted as e:
                self._pause_for_quota(str(e)[:200])
                return 60
        seeded = seed_library_papers(store, session_id, topic)
        if seeded:
            store.add_event(session_id, "info", f"Added {seeded} uploaded Library paper(s) to the analysis queue")
        self._enrich_uploads(session_id)
        self._rank_unscored(session_id)

        # 2. One paper at a time, best first — with a gap duplication test after every few papers.
        paper = store.next_queued_paper(session_id, QUEUE_STATUSES, now)
        gap = next_gap_to_verify(store, session_id, now) \
            if not self._gap_checks_after or now >= self._gap_checks_after else None
        if gap and (paper is None or store.processed_since_last_gap_check(session_id) >= GAP_CHECK_EVERY):
            return self._verify(agent, topic, session_id, gap)
        if paper:
            return self._process(agent, topic, session_id, paper)

        # 2b. Nothing queued: look again for free full texts of papers that only had an abstract (weekly).
        stale = store.next_fulltext_recheck(session_id, now - FULLTEXT_RECHECK)
        if stale:
            return self._recheck_full_text(agent, topic, session_id, stale)

        # 3. Queue empty: search for more when due.
        next_search = _as_dt(ap.get("next_search_at"))
        if next_search and now < next_search:
            if ap.get("state") != "waiting":
                self._set("waiting", f"All queued papers processed — next search {next_search.strftime('%Y-%m-%d %H:%M UTC')}",
                          current_paper_id=None)
            return min(AUTOPILOT_IDLE_SECONDS * 10, (next_search - now).total_seconds())
        return self._search(agent, topic, session_id)

    # ── helpers ──

    def ensure_session(self, topic: str) -> str:
        store = self.store
        row = store.ensure_topic(topic)
        sid = row.get("session_id")
        if sid and store.get_session(sid):
            if (store.get_session(sid) or {}).get("description") != row.get("description", ""):
                store.update_session(sid, description=row.get("description", ""))
            return sid
        session = store.create_session(topic.replace("_", " "), row.get("description", ""))
        store.update_topic(topic, session_id=session["id"])
        store.add_event(session["id"], "info", f"Autopilot started research on “{topic.replace('_', ' ')}”")
        return session["id"]

    def _claim(self, session_id: str) -> bool:
        if self._claimed_session and self._claimed_session != session_id:
            jobs.release(self._claimed_session)
            self._claimed_session = None
        if not jobs.claim(session_id):
            return False
        if self._claimed_session != session_id:
            # e.g. after a restart the session was marked "interrupted": the autopilot owns it again.
            session = self.store.get_session(session_id) or {}
            if session.get("status") != "autopilot":
                self.store.update_session(session_id, status="autopilot", status_message="Managed by the autopilot")
        self._claimed_session = session_id
        return True

    def _release(self) -> None:
        if self._claimed_session:
            jobs.release(self._claimed_session)
            self._claimed_session = None

    def _pause_for_quota(self, message: str, recheck: bool = False) -> None:
        """Pause until the daily reset. With `recheck` (the estimate came from OpenRouter's /key figures) the
        real budget is read again every half hour, so a late or early reset doesn't cost a whole day."""
        now = self.clock()
        reset = next_quota_reset(now)
        until = min(reset, now + QUOTA_RECHECK) if recheck else reset
        self._set("paused", f"{message}. Resumes {reset.strftime('%Y-%m-%d %H:%M UTC')}",
                  paused_until=until, current_paper_id=None)

    def _pause_for_outage(self, err: LLMError) -> float:
        """The LLM can't be used right now: keep all work queued and try again later."""
        if isinstance(err, LLMHTTPError):
            delay = timedelta(minutes=30)      # rejected key / no credit: needs the user; check twice an hour
        elif isinstance(err, RateLimited):
            delay = timedelta(minutes=3)
        else:
            delay = timedelta(minutes=10)
        until = self.clock() + delay
        message = _outage_message(err)
        logger.warning("Autopilot paused: %s (%s)", message, str(err)[:200])
        self._set("paused", f"{message} Retrying at {until.strftime('%H:%M UTC')}.", paused_until=until,
                  current_paper_id=None, last_error=str(err)[:300])
        return 60

    def _rank(self, paper: dict) -> None:
        score, detail = priority(paper)
        self.store.update_paper(paper["id"], priority_score=score, priority_detail=detail)

    def _enrich_uploads(self, session_id: str) -> None:
        """Uploaded PDFs only have a file name: look them up by title for authors, year, venue and citations."""
        store = self.store
        for p in store.list_papers(session_id, selection="selected"):
            if p.get("added_by") != "user" or p.get("enriched_at"):
                continue
            try:
                work = self.enricher.find(p)
            except Exception:
                work = None
            if work is None and self.enricher.failed:
                continue  # OpenAlex unreachable: try again on a later step instead of giving up on this paper
            fields: dict = {"enriched_at": self.clock()}
            if work:
                e = parse_work(work)
                fields.update({"publication_type": e.publication_type, "peer_reviewed": e.peer_reviewed})
                if work.get("title") and len(work["title"]) > 10:
                    fields["title"] = work["title"]
                for key, value in (("year", e.year), ("citation_count", e.citation_count), ("venue", e.venue),
                                   ("doi", e.doi), ("pdf_url", e.oa_pdf_url)):
                    if value and not p.get(key):
                        fields[key] = value
                authors = [a.get("author", {}).get("display_name") for a in work.get("authorships") or []]
                if any(authors) and not p.get("authors"):
                    fields["authors"] = [a for a in authors if a][:20]
                if fields.get("doi") and not p.get("url"):
                    fields["url"] = f"https://doi.org/{fields['doi']}"
            store.update_paper(p["id"], **fields)

    def _rank_unscored(self, session_id: str) -> None:
        for p in self.store.queued_papers(session_id, QUEUE_STATUSES, limit=500):
            if p.get("priority_score") is None:
                self._rank(p)

    def _process(self, agent: ResearchAgent, topic: str, session_id: str, paper: dict) -> float:
        store = self.store
        label = f"{paper['ref']}: {paper['title']}"
        attempts = (paper.get("attempts") or 0) + 1
        self._set("processing", f"Processing {label}", current_paper_id=paper["id"])
        store.update_paper(paper["id"], attempts=attempts, retry_at=None)
        try:
            agent.process_paper(paper["id"])
        except QuotaExhausted as e:
            store.update_paper(paper["id"], processing_status="pending", attempts=attempts - 1)
            self._pause_for_quota(str(e)[:200])
            return 60
        except LLMError as e:
            if not is_outage(e):
                return self._failed(session_id, paper, attempts, e)
            store.update_paper(paper["id"], processing_status="pending", attempts=attempts - 1)  # not its fault
            raise
        except psycopg.Error as e:
            # Database trouble (e.g. a restart mid-paper): keep the paper queued, bounded by the attempt limit.
            logger.warning("Database error while processing %s: %s", paper["id"], e.__class__.__name__)
            if attempts >= MAX_ATTEMPTS:
                return self._failed(session_id, paper, attempts, e)
            return 30
        except Exception as e:  # one bad paper must not stop the autopilot
            return self._failed(session_id, paper, attempts, e)

        paper = store.get_paper(paper["id"])
        if paper is None:  # deleted by the user while it was being processed
            self._set("processing", f"Finished {label}", current_paper_id=None)
            return 0
        if paper["processing_status"] in PROCESSED_OK and paper["content_status"] != "unavailable":
            try:
                self.publish(agent, topic, session_id, paper)
            except QuotaExhausted as e:
                self._pause_for_quota(str(e)[:200])
                return 60
            except Exception:
                logger.exception("Publishing %s failed", paper["id"])
                store.add_event(session_id, "warning", f"{paper['ref']}: could not be added to the Library/report")
        self._schedule_retry(paper, attempts)
        self._set("processing", f"Finished {label}", current_paper_id=None)
        store.update_session(session_id, status="autopilot", status_message=f"Last processed {paper['ref']}")
        return 0

    def _failed(self, session_id: str, paper: dict, attempts: int, err: BaseException) -> float:
        logger.exception("Autopilot failed on paper %s", paper["id"], exc_info=err)
        self.store.update_paper(paper["id"], processing_status="failed", error=str(err)[:500], processed_at=self.clock())
        self.store.add_event(session_id, "error", f"{paper['ref']}: processing failed ({err.__class__.__name__})")
        row = self.store.get_paper(paper["id"])
        if row:
            self._schedule_retry(row, attempts)
        return 0

    def _schedule_retry(self, paper: dict, attempts: int) -> None:
        """A failed or partly analysed paper goes back in the queue later, up to MAX_ATTEMPTS tries in total."""
        if paper["processing_status"] not in ("failed", "analysis_failed") or attempts >= MAX_ATTEMPTS:
            return
        self.store.update_paper(paper["id"], processing_status="pending", published_at=None,
                                retry_at=self.clock() + RETRY_DELAY * attempts)

    def _recheck_full_text(self, agent: ResearchAgent, topic: str, session_id: str, paper: dict) -> float:
        """Preprints often get an arXiv, repository or journal copy later: if one is free now, re-analyse the
        paper from its full text (no LLM requests unless a full text is found)."""
        self._set("processing", f"Looking for a free full text of {paper['ref']}: {paper['title'][:90]}",
                  current_paper_id=paper["id"])
        try:
            res = fetch_content(paper, transport=agent.fetch_transport, cancel_event=agent.cancel_event,
                                finder=agent.fulltext)
            if res.unreachable and not internet_reachable():
                raise NetworkDown("No internet connection while looking for a full text")
            if res.status == "full_text":
                upgrade_to_full_text(self.store, topic, paper, res.text, res.pdf_bytes,
                                     source=res.source, pdf_url=res.metadata.get("pdf_url", ""))
            else:
                self.store.update_paper(paper["id"], fulltext_checked_at=self.clock())
        finally:
            self.store.update_autopilot(current_paper_id=None)
        return 0

    def _publish_unpublished(self, agent: ResearchAgent, topic: str, session_id: str) -> None:
        """Publish papers whose analysis completed but whose publish was cut off (restart) or failed."""
        for paper in self.store.unpublished_papers(session_id, PROCESSED_OK):
            self.store.update_autopilot(current_paper_id=paper["id"])  # deleting / attaching it waits meanwhile
            try:
                self.publish(agent, topic, session_id, paper)
                logger.info("Published %s after an earlier interrupted or failed attempt", paper["id"])
            except QuotaExhausted:
                raise
            except LLMError:
                logger.exception("Publishing %s failed; will retry later", paper["id"])
                return  # an LLM/embedding outage affects the rest too: try again next round
            except Exception:
                logger.exception("Publishing %s failed; will retry later", paper["id"])
            finally:
                self.store.update_autopilot(current_paper_id=None)

    def publish(self, agent: ResearchAgent, topic: str, session_id: str, paper: dict) -> None:
        """Add the processed paper to the Library and to the topic's reports."""
        store = self.store
        if paper.get("library_paper_id") and not library_paper_exists(topic, paper["library_paper_id"]):
            paper = {**paper, "library_paper_id": ""}  # its Library folder was removed: publish it afresh
        pdf = agent.pdf_cache.pop(paper["id"], None)
        incoming = incoming_pdf_path(topic, paper["id"])
        incoming_text = incoming_text_path(topic, paper["id"])
        staged: list[tuple] = []  # (path, mtime) of staged files used here, removed once published
        if pdf is None and incoming.is_file():
            staged.append((incoming, incoming.stat().st_mtime_ns))
            pdf = incoming.read_bytes()        # attached by the user, or found by the weekly full-text check
        if pdf is None and paper.get("pdf_url") and paper.get("content_source") != "library" and not paper.get("library_paper_id"):
            # Content may have been fetched in an earlier run (e.g. before a restart): get the PDF for the Library.
            pdf = fetch_pdf(paper["pdf_url"], agent.fetch_transport)
        already_in_library = bool(paper.get("library_paper_id"))
        lib_id = publish_to_library(topic, paper, pdf)
        if lib_id != paper.get("library_paper_id"):
            store.update_paper(paper["id"], library_paper_id=lib_id)
        if already_in_library:
            # A re-publish after a full text arrived: replace the Library copy (abstract) with it.
            if pdf is not None and staged:
                attach_pdf(topic, lib_id, pdf)
            elif incoming_text.is_file():
                staged.append((incoming_text, incoming_text.stat().st_mtime_ns))
                replace_library_text(topic, lib_id, incoming_text.read_text(encoding="utf-8"))
        for path, mtime in staged:
            # Keep a file that was replaced meanwhile (e.g. a PDF attached during this publish): it is used next time.
            if path.is_file() and path.stat().st_mtime_ns == mtime:
                path.unlink(missing_ok=True)
        # Per-paper analysis (keep an existing analysis the user generated for an uploaded paper).
        path = analysis_path(topic, lib_id)
        if paper.get("added_by") != "user" or not path.exists() or read_meta(topic, lib_id).get("origin") == "agent" \
                or path.read_text(encoding="utf-8", errors="replace").startswith(AUTOPILOT_MARK):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(AUTOPILOT_MARK + render_paper_analysis(store.get_paper(paper["id"]), store.get_summary(paper["id"]),
                                                  store.relations_for_paper(paper["id"]),
                                                  store.gaps_for_paper(paper["id"])), encoding="utf-8")
        self.maybe_update_narrative(topic, session_id, agent.provider)
        self.write_living_report(topic, session_id)
        store.update_paper(paper["id"], published_at=self.clock())
        store.add_event(session_id, "success", f"{paper['ref']} added to the Library and the living report",
                        {"paper_id": paper["id"], "library_paper_id": lib_id})

    def maybe_update_narrative(self, topic: str, session_id: str, provider=None) -> None:
        store = self.store
        row = store.ensure_topic(topic)
        processed = store.count_processed(session_id)
        if processed - (row.get("narrative_paper_count") or 0) < NARRATIVE_EVERY and row.get("narrative"):
            return
        if processed < 2:
            return
        session = store.get_session(session_id) or {}
        papers = [p for p in store.list_papers(session_id, selection="selected") if p.get("has_summary")]
        try:
            narrative = llm_tasks.write_report_narrative(
                topic.replace("_", " "), render_synthesis(session.get("synthesis"), limit=6000),
                "\n".join(f"{g['ref']} ({g['confidence']}, {g['status']}): {g['title']}" for g in store.list_gaps(session_id)),
                "\n".join(f"{p['ref']}: {p['title']}, {p.get('year') or 'n.d.'}, {p.get('venue') or '—'}" for p in papers),
                provider=provider)
        except (LLMError, StructuredOutputError) as e:
            logger.warning("Narrative update failed: %s", e)
            return
        store.update_topic(topic, narrative=narrative, narrative_paper_count=processed)
        store.add_event(session_id, "info", "Report narrative rewritten")

    def write_living_report(self, topic: str, session_id: str) -> None:
        row = self.store.ensure_topic(topic)
        path = living_report_path(topic)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_living_report(self.store, topic, session_id, row.get("narrative", "")), encoding="utf-8")

    def _drop_duplicate_versions(self, session_id: str, new_ids: list[str]) -> list[str]:
        """Enrichment can reveal that two selected records are one work (e.g. arXiv preprint + ACL paper).
        Keep the better version (peer reviewed, then more cited) and reject the other."""
        store = self.store
        kept = []
        for pid in new_ids:
            paper = store.get_paper(pid)
            if paper["selection_status"] != "selected":
                continue  # already rejected as a version of an earlier paper in this loop
            others = [p for p in store.list_papers(session_id, selection="selected") if p["id"] != pid]
            dup = find_duplicate(paper, others)
            if not dup:
                kept.append(pid)
                continue
            better_new = dup["processing_status"] not in ("done", "analysis_failed") and (
                (paper.get("peer_reviewed") is True and dup.get("peer_reviewed") is not True)
                or (paper.get("citation_count") or 0) > 2 * (dup.get("citation_count") or 0))
            loser, winner = (dup, paper) if better_new else (paper, dup)
            store.update_paper(loser["id"], selection_status="rejected", processing_status="skipped",
                               relevance_reason=f"Another version of {winner['ref']}: {winner['title'][:80]}")
            store.add_event(session_id, "info", f"{loser['ref']} is another version of {winner['ref']} — kept one")
            if winner["id"] == pid:
                kept.append(pid)
                if loser["id"] in kept:
                    kept.remove(loser["id"])
        return kept

    def _verify(self, agent: ResearchAgent, topic: str, session_id: str, gap: dict) -> float:
        self._set("verifying", f"Checking {gap['ref']} against the wider literature: {gap['title'][:90]}",
                  current_paper_id=None)
        try:
            verify_gap(self.store, session_id, gap, topic.replace("_", " "), provider=agent.provider,
                       literature=LiteratureSearch(searx=agent.searx, openalex=self.enricher))
        except QuotaExhausted as e:
            self._pause_for_quota(str(e)[:200])
            return 60
        except SearchUnavailable as e:
            logger.info("Gap check for %s postponed: %s", gap["ref"], e)
            self._gap_checks_after = self.clock() + timedelta(minutes=15)
            return 0
        except (LLMError, StructuredOutputError) as e:
            if is_outage(e):
                raise
            # Mark it checked-with-error so one stubborn gap can't block the loop; retried after the re-check window.
            logger.warning("Gap check failed for %s: %s", gap["ref"], e)
            self.store.update_gap(gap["id"], verified_at=self.clock(), verification_note=f"Check failed: {str(e)[:200]}")
        self.write_living_report(topic, session_id)
        return 0

    def _search(self, agent: ResearchAgent, topic: str, session_id: str) -> float:
        store = self.store
        interval = timedelta(hours=AUTOPILOT_RESEARCH_INTERVAL_HOURS)
        if len(store.list_papers(session_id, selection="selected")) >= AUTOPILOT_MAX_PAPERS_PER_TOPIC:
            self._set("waiting", f"Topic limit of {AUTOPILOT_MAX_PAPERS_PER_TOPIC} papers reached",
                      next_search_at=self.clock() + interval, current_paper_id=None)
            return 0
        self._set("searching", "Searching for new papers", current_paper_id=None)
        requeued = store.requeue_unchecked(session_id)
        if requeued:
            store.add_event(session_id, "info", f"{requeued} earlier results that were never checked are screened again")
        searx_ok, _ = agent.searx.check()
        leftovers = [p["id"] for p in store.list_papers(session_id, selection="candidate")]
        if not searx_ok and not DISCOVERY_ENABLED and not leftovers:
            # Don't spend LLM requests on queries that can't be searched.
            self._set("waiting", "Search service unavailable — retrying in 15 minutes",
                      next_search_at=self.clock() + timedelta(minutes=15))
            return 0
        try:
            session = store.get_session(session_id) or {}
            if not session.get("topic_analysis"):  # first round, or the focus changed (analysis cleared)
                queries, source = agent.understand_topic(), "topic"
            else:
                try:
                    queries = agent.expansion_queries()
                except (LLMError, StructuredOutputError) as e:
                    if is_outage(e):
                        raise
                    queries = []
                queries, source = (queries or [session.get("topic") or topic.replace("_", " ")]), "expansion"
            ids, failed = agent.search(queries, source) if searx_ok else ([], True)
            if DISCOVERY_ENABLED:
                # Recent top-venue papers the web search misses (proceedings, arXiv newest, OpenAlex recent).
                self._set("searching", "Checking recent top-venue proceedings, arXiv and OpenAlex")
                ids += agent.discover([session.get("topic") or topic.replace("_", " "), *queries])
            leftovers = [p["id"] for p in store.list_papers(session_id, selection="candidate") if p["id"] not in ids]
            if failed and not ids and not leftovers:
                self._set("waiting", "Search service unavailable — retrying in 15 minutes",
                          next_search_at=self.clock() + timedelta(minutes=15))
                return 0
            self._set("screening", "Assessing relevance of new results")
            selected = agent.screen(ids + leftovers, continuous=True)
        except QuotaExhausted as e:
            self._pause_for_quota(str(e)[:200])
            return 60

        for pid in selected:
            paper = store.get_paper(pid)
            try:
                updates = enrich_fields(paper, self.enricher)
            except Exception:
                updates = {}
            enriched = {} if (not updates and self.enricher.failed) else {"enriched_at": self.clock()}
            store.update_paper(pid, **enriched, **updates)
        selected = self._drop_duplicate_versions(session_id, selected)
        for pid in selected:
            self._rank(store.get_paper(pid))
        store.update_session(session_id, status="autopilot",
                             status_message=f"Queued {len(selected)} new papers" if selected else "Waiting for new papers")
        if len(selected) >= DRIP_MIN_SELECTED:
            store.update_autopilot(next_search_at=None)   # a productive round: search again when the queue is empty
            store.add_event(session_id, "success", f"Queued {len(selected)} new papers, best first")
        elif selected:
            # Only a trickle of new papers: don't spend another screening round right away.
            store.update_autopilot(next_search_at=self.clock() + DRIP_WAIT)
            store.add_event(session_id, "success", f"Queued {len(selected)} new paper; next search in "
                                                   f"{int(DRIP_WAIT.total_seconds() // 3600)} h")
        else:
            nxt = self.clock() + interval
            self._set("waiting", f"No new relevant papers — next search {nxt.strftime('%Y-%m-%d %H:%M UTC')}",
                      next_search_at=nxt)
        return 0


autopilot = Autopilot()

