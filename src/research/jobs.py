"""In-process background jobs for research sessions (one active job per session).

Jobs are plain worker threads, so they can be started from the event loop or from a worker thread.
"""
import asyncio
import logging
import threading
from typing import Callable

from src.research.agent import RESUMABLE, ResearchAgent
from src.research.store import ResearchStore

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self):
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._followup: set[str] = set()
        self._claimed: set[str] = set()   # sessions the autopilot is working on
        self._lock = threading.Lock()
        self.agent_factory: Callable[..., ResearchAgent] = ResearchAgent
        self.loop = None  # kept for API compatibility; not needed with threads

    def is_running(self, session_id: str) -> bool:
        if session_id in self._claimed:
            return True
        thread = self._threads.get(session_id)
        return bool(thread and thread.is_alive())

    def claim(self, session_id: str) -> bool:
        """Mark a session as owned by the autopilot: manual jobs on it are refused (409).
        Returns False while a manual job is still running on it (two agents must never share a session)."""
        with self._lock:
            thread = self._threads.get(session_id)
            if session_id not in self._claimed and thread and thread.is_alive():
                return False
            self._claimed.add(session_id)
            return True

    def release(self, session_id: str) -> None:
        with self._lock:
            self._claimed.discard(session_id)

    def claimed(self) -> set[str]:
        return set(self._claimed)

    def start(self, store: ResearchStore, session_id: str, job: str, loop=None, **kwargs) -> bool:
        """Start a job ('research', 'process' or 'gaps'). Returns False if one is already running."""
        with self._lock:
            if self.is_running(session_id):
                return False
            cancel = threading.Event()
            agent = self.agent_factory(store, session_id, cancel_event=cancel)
            runner = {"research": agent.run_research, "process": agent.run_processing,
                      "gaps": agent.run_gap_rebuild}[job]
            thread = threading.Thread(target=self._run, args=(store, session_id, runner, kwargs, cancel),
                                      name=f"research-{session_id[:8]}-{job}", daemon=True)
            self._threads[session_id] = thread
            self._cancel[session_id] = cancel
            thread.start()
            return True

    def _run(self, store: ResearchStore, session_id: str, runner, kwargs: dict, cancel: threading.Event) -> None:
        try:
            runner(**kwargs)
        except Exception:
            logger.exception("Job for session %s crashed", session_id)
        finally:
            with self._lock:
                if self._threads.get(session_id) is threading.current_thread():
                    self._threads.pop(session_id, None)
                    self._cancel.pop(session_id, None)
                wants_followup = session_id in self._followup
                self._followup.discard(session_id)
            if wants_followup and not cancel.is_set():
                self._start_followup(store, session_id)

    def _start_followup(self, store: ResearchStore, session_id: str) -> None:
        """Process papers the user added while the previous job was already past its processing step."""
        try:
            pending = [p for p in store.list_papers(session_id, selection="selected")
                       if p["processing_status"] in RESUMABLE]
            if pending:
                store.update_session(session_id, status="processing", status_message="Processing added papers")
                self.start(store, session_id, "process", paper_ids=None)
        except Exception:
            logger.exception("Could not start follow-up processing for session %s", session_id)

    def request_followup(self, session_id: str) -> None:
        with self._lock:
            self._followup.add(session_id)

    def cancel(self, session_id: str) -> bool:
        event = self._cancel.get(session_id)
        if event and self.is_running(session_id):
            event.set()
            with self._lock:
                self._followup.discard(session_id)
            return True
        return False

    async def wait(self, session_id: str, timeout: float = 120) -> None:
        thread = self._threads.get(session_id)
        if thread:
            await asyncio.to_thread(thread.join, timeout)


jobs = JobManager()
