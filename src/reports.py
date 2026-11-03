"""Registry of background library jobs (reports, re-indexing) kept in memory so the UI can show progress."""
import threading
import uuid
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MAX_FINISHED = 100  # finished jobs kept for the UI; older ones are dropped (the server runs for months)


class ReportJobs:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, topic: str, kind: str = "report") -> dict | None:
        """Register a job; returns None if a job for this topic is already running."""
        with self._lock:
            if any(j["status"] == "running" and (j["topic"] == topic or "*" in (j["topic"], topic))
                   for j in self._jobs.values()):
                return None
            job = {"id": uuid.uuid4().hex, "kind": kind, "topic": topic, "status": "running", "stage": "starting",
                   "done": 0, "total": 0, "current": "", "error": "", "warnings": [], "file": "",
                   "started_at": _now(), "finished_at": None}
            self._jobs[job["id"]] = job
            finished = [j for j in self._jobs.values() if j["status"] != "running"]
            for old in sorted(finished, key=lambda j: j["started_at"])[:max(0, len(finished) - MAX_FINISHED)]:
                del self._jobs[old["id"]]
            return dict(job)

    def busy(self, topic: str) -> bool:
        """A report or re-index job is running on this topic (or on the whole library)."""
        with self._lock:
            return any(j["status"] == "running" and j["topic"] in (topic, "*") for j in self._jobs.values())

    def update(self, job_id: str | None, **fields) -> None:
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if "warning" in fields:
                    job["warnings"].append(fields.pop("warning"))
                job.update(fields)
                if fields.get("status") in ("completed", "failed"):
                    job["finished_at"] = _now()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return {**job, "warnings": list(job["warnings"])} if job else None

    def list(self, topic: str | None = None) -> list[dict]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values() if topic is None or j["topic"] == topic]
        return sorted(jobs, key=lambda j: j["started_at"], reverse=True)


report_jobs = ReportJobs()
