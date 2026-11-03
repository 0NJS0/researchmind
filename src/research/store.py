"""PostgreSQL persistence for research sessions, papers, summaries, gaps, evidence and events.

Every method runs in its own short transaction, except inside ``with store.atomic():`` where all
calls made by that thread share one transaction (used to rebuild gaps all-or-nothing). Activity
events always commit immediately so progress stays visible while an atomic block runs.
"""
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from src.db import connection, init_schema

JSON_COLUMNS = {"authors", "engines", "topic_analysis", "synthesis", "sections", "data", "priority_detail",
                "verification_refs", "result"}
ACTIVE_STATUSES = ("created", "searching", "processing", "summarizing", "analyzing")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _row(row: dict | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    out.pop("seq", None)
    for key, value in out.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat(timespec="seconds")
    return out


def _clean(value: Any) -> Any:
    """PostgreSQL text/JSONB cannot hold NUL bytes, which scraped pages and PDFs sometimes contain."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _encode(key: str, value: Any) -> Any:
    value = _clean(value)
    if key in JSON_COLUMNS and value is not None:
        return Jsonb(value)
    return value


class ResearchStore:
    def __init__(self):
        init_schema()
        self._local = threading.local()

    # ── plumbing ──

    @contextmanager
    def _conn(self):
        shared = getattr(self._local, "conn", None)
        if shared is not None:
            yield shared
            return
        with connection() as conn:
            yield conn

    @contextmanager
    def atomic(self):
        """Group every store call made by this thread into one transaction."""
        if getattr(self._local, "conn", None) is not None:
            yield
            return
        with connection() as conn:
            self._local.conn = conn
            try:
                yield
            finally:
                self._local.conn = None

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._conn() as conn:
            return _row(conn.execute(sql, params).fetchone())

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            return [_row(r) for r in conn.execute(sql, params).fetchall()]

    def _update(self, table: str, row_id: str, fields: dict) -> None:
        if not fields:
            return
        fields = {**fields}
        if table in ("sessions", "papers", "gaps", "paper_summaries"):
            fields["updated_at"] = datetime.now(timezone.utc)
        cols = ", ".join(f"{k} = %s" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE {table} SET {cols} WHERE id = %s",
                         (*[_encode(k, v) for k, v in fields.items()], row_id))

    def _insert(self, table: str, fields: dict) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("%s" for _ in fields)
        with self._conn() as conn:
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})",
                         tuple(_encode(k, v) for k, v in fields.items()))

    def _next_seq(self, session_id: str, column: str) -> int:
        with self._conn() as conn:
            return conn.execute(f"UPDATE sessions SET {column} = {column} + 1 WHERE id = %s RETURNING {column}",
                                (session_id,)).fetchone()[column]

    # ── sessions ──

    def create_session(self, topic: str, description: str = "") -> dict:
        sid = new_id()
        self._insert("sessions", {"id": sid, "topic": topic, "description": description})
        return self.get_session(sid)

    def get_session(self, session_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM sessions WHERE id = %s", (session_id,))

    def list_sessions(self) -> list[dict]:
        return self._fetchall(
            """SELECT s.id, s.topic, s.description, s.status, s.status_message, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM papers p WHERE p.session_id = s.id AND p.selection_status = 'selected') AS paper_count,
                      (SELECT COUNT(*) FROM gaps g WHERE g.session_id = s.id AND g.status != 'merged') AS gap_count
               FROM sessions s ORDER BY s.updated_at DESC"""
        )

    def update_session(self, session_id: str, **fields) -> None:
        self._update("sessions", session_id, fields)

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    def session_counts(self, session_id: str) -> dict:
        row = self._fetchone(
            """SELECT
                 (SELECT COUNT(*) FROM papers WHERE session_id = %(s)s AND selection_status = 'selected') AS papers,
                 (SELECT COUNT(*) FROM papers WHERE session_id = %(s)s) AS candidates,
                 (SELECT COUNT(*) FROM papers WHERE session_id = %(s)s AND selection_status = 'rejected') AS rejected,
                 (SELECT COUNT(*) FROM papers p JOIN paper_summaries s ON s.paper_id = p.id
                   WHERE p.session_id = %(s)s) AS summarized,
                 (SELECT COUNT(*) FROM gaps WHERE session_id = %(s)s AND status != 'merged') AS gaps,
                 (SELECT COUNT(*) FROM search_queries WHERE session_id = %(s)s) AS queries""",
            {"s": session_id})
        return {k: int(v) for k, v in row.items()}

    def mark_interrupted_sessions(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "UPDATE sessions SET status = 'failed', status_message = 'Interrupted by a server restart. "
                "Use \"Continue research\" to resume.', updated_at = now() WHERE status = ANY(%s)",
                (list(ACTIVE_STATUSES),),
            ).rowcount

    # ── queries ──

    def add_query(self, session_id: str, query: str, source: str, iteration: int) -> dict:
        qid = new_id()
        self._insert("search_queries", {"id": qid, "session_id": session_id, "query": query,
                                        "source": source, "iteration": iteration})
        return self._fetchone("SELECT * FROM search_queries WHERE id = %s", (qid,))

    def update_query(self, query_id: str, **fields) -> None:
        self._update("search_queries", query_id, fields)

    def list_queries(self, session_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM search_queries WHERE session_id = %s ORDER BY seq", (session_id,))

    # ── papers ──

    PAPER_FIELDS = {"title", "authors", "abstract", "url", "pdf_url", "doi", "arxiv_id", "source", "engines",
                    "venue", "publication_date", "year", "citation_count", "selection_status",
                    "relevance_score", "relevance_reason", "relevance_method", "found_by_query", "added_by",
                    "content_status", "processing_status", "raw_content", "content_source", "priority_score",
                    "priority_detail", "publication_type", "peer_reviewed", "library_paper_id", "enriched_at"}

    def add_paper(self, session_id: str, fields: dict, assign_ref: bool = False) -> dict:
        """Insert a paper. Refs (P1, P2, ...) are only given to selected papers so they stay contiguous."""
        pid = new_id()
        ref = f"P{self._next_seq(session_id, 'paper_seq')}" if assign_ref else ""
        row = {k: v for k, v in fields.items() if k in self.PAPER_FIELDS and v is not None}
        self._insert("papers", {"id": pid, "session_id": session_id, "ref": ref, **row})
        return self.get_paper(pid)

    def get_paper(self, paper_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM papers WHERE id = %s", (paper_id,))

    def list_papers(self, session_id: str, selection: str | None = "selected",
                    include_content: bool = False) -> list[dict]:
        cols = "p.*" if include_content else ", ".join(
            f"p.{c}" for c in ("id", "session_id", "ref", "title", "authors", "abstract", "url", "pdf_url", "doi",
                               "arxiv_id", "source", "engines", "venue", "publication_date", "year",
                               "citation_count", "content_status", "content_source", "selection_status",
                               "relevance_score", "relevance_reason", "relevance_method", "processing_status",
                               "error", "found_by_query", "added_by", "created_at", "updated_at", "priority_score",
                               "priority_detail", "publication_type", "peer_reviewed", "library_paper_id",
                               "processed_at", "enriched_at", "published_at", "attempts", "retry_at",
                               "fulltext_checked_at", "screen_deferrals"))
        sql = f"""SELECT {cols}, (s.id IS NOT NULL) AS has_summary, length(p.raw_content) AS content_length
                  FROM papers p LEFT JOIN paper_summaries s ON s.paper_id = p.id WHERE p.session_id = %s"""
        params: tuple = (session_id,)
        if selection:
            sql += " AND p.selection_status = %s"
            params += (selection,)
        # Selected papers in ref order (P1, P2, ...) — the order they are processed and cited in.
        sql += " ORDER BY (p.ref = ''), NULLIF(substr(p.ref, 2), '')::int, p.seq"
        return self._fetchall(sql, params)

    def update_paper(self, paper_id: str, **fields) -> None:
        self._update("papers", paper_id, fields)

    def select_paper(self, paper_id: str, **fields) -> dict:
        paper = self.get_paper(paper_id)
        if not paper["ref"]:
            fields["ref"] = f"P{self._next_seq(paper['session_id'], 'paper_seq')}"
        self.update_paper(paper_id, selection_status="selected", **fields)
        return self.get_paper(paper_id)

    def next_queued_paper(self, session_id: str, statuses: tuple[str, ...], now=None) -> dict | None:
        """The selected, unprocessed paper with the highest priority (the autopilot's next task).
        Papers waiting to be retried after a failure are skipped until their `retry_at`."""
        return self._fetchone(
            """SELECT * FROM papers WHERE session_id = %s AND selection_status = 'selected'
                 AND processing_status = ANY(%s) AND (retry_at IS NULL OR retry_at <= COALESCE(%s, now()))
               ORDER BY priority_score DESC NULLS LAST, seq LIMIT 1""", (session_id, list(statuses), now))

    def queued_papers(self, session_id: str, statuses: tuple[str, ...], limit: int = 10) -> list[dict]:
        return self._fetchall(
            """SELECT id, ref, title, year, venue, citation_count, publication_type, peer_reviewed, url,
                      relevance_score, priority_score, priority_detail, added_by, processing_status
               FROM papers WHERE session_id = %s AND selection_status = 'selected' AND processing_status = ANY(%s)
               ORDER BY priority_score DESC NULLS LAST, seq LIMIT %s""", (session_id, list(statuses), limit))

    def recent_processed(self, session_id: str, limit: int = 10) -> list[dict]:
        return self._fetchall(
            """SELECT id, ref, title, year, venue, citation_count, content_status, processing_status,
                      library_paper_id, processed_at, added_by
               FROM papers WHERE session_id = %s AND processed_at IS NOT NULL
               ORDER BY processed_at DESC LIMIT %s""", (session_id, limit))

    def processed_since_last_gap_check(self, session_id: str) -> int:
        """Papers processed after the most recent gap duplication test in this session."""
        return self._fetchone(
            """SELECT COUNT(*) AS n FROM papers p WHERE p.session_id = %(s)s AND p.processed_at IS NOT NULL
                 AND p.processed_at > COALESCE((SELECT MAX(verified_at) FROM gaps WHERE session_id = %(s)s),
                                               '-infinity'::timestamptz)""", {"s": session_id})["n"]

    def next_fulltext_recheck(self, session_id: str, checked_before) -> dict | None:
        """An analysed paper that only had its abstract (or title), not looked for a full text recently."""
        return self._fetchone(
            """SELECT * FROM papers WHERE session_id = %s AND selection_status = 'selected'
                 AND content_status IN ('abstract', 'metadata_only') AND content_source <> 'library'
                 AND processing_status IN ('done', 'analysis_failed')
                 AND (fulltext_checked_at IS NULL OR fulltext_checked_at < %s)
               ORDER BY priority_score DESC NULLS LAST, seq LIMIT 1""", (session_id, checked_before))

    def requeue_unchecked(self, session_id: str) -> int:
        """Papers an older version rejected without any relevance check (its screening cap) become candidates
        again, so the autopilot looks at them."""
        with self._conn() as conn:
            return conn.execute(
                """UPDATE papers SET selection_status = 'candidate', relevance_reason = ''
                   WHERE session_id = %s AND selection_status = 'rejected'
                     AND relevance_reason = 'Lower-ranked result; relevance-check limit reached'""",
                (session_id,)).rowcount

    def paper_with_pdf_hash(self, session_id: str, sha256: str) -> dict | None:
        return self._fetchone("SELECT id, ref, title FROM papers WHERE session_id = %s AND pdf_sha256 = %s",
                              (session_id, sha256))

    def delete_summary(self, paper_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM paper_summaries WHERE paper_id = %s", (paper_id,))

    def unpublished_papers(self, session_id: str, statuses: tuple[str, ...]) -> list[dict]:
        """Processed papers that never reached the Library/reports (e.g. the server stopped mid-publish)."""
        return self._fetchall(
            """SELECT * FROM papers WHERE session_id = %s AND selection_status = 'selected'
                 AND processing_status = ANY(%s) AND content_status <> 'unavailable'
                 AND processed_at IS NOT NULL AND published_at IS NULL
               ORDER BY processed_at""", (session_id, list(statuses)))

    def count_processed(self, session_id: str) -> int:
        return self._fetchone("SELECT COUNT(*) AS n FROM papers WHERE session_id = %s AND processed_at IS NOT NULL",
                              (session_id,))["n"]

    def paper_by_library_id(self, session_id: str, library_paper_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM papers WHERE session_id = %s AND library_paper_id = %s",
                              (session_id, library_paper_id))

    def delete_paper(self, paper_id: str, remember: bool = True) -> None:
        """Delete a paper and everything attached to it (summary, relations, evidence cascade).
        `remember` keeps a tombstone so later searches don't re-add it."""
        paper = self.get_paper(paper_id)
        if not paper:
            return
        with self._conn() as conn:
            if remember:
                conn.execute("INSERT INTO deleted_papers (session_id, title, doi, arxiv_id, url) VALUES (%s, %s, %s, %s, %s)",
                             (paper["session_id"], paper["title"], paper.get("doi") or "", paper.get("arxiv_id") or "",
                              paper.get("url") or ""))
            conn.execute("UPDATE autopilot SET current_paper_id = NULL WHERE current_paper_id = %s", (paper_id,))
            conn.execute("DELETE FROM papers WHERE id = %s", (paper_id,))

    def tombstones(self, session_id: str) -> list[dict]:
        rows = self._fetchall("SELECT title, doi, arxiv_id, url FROM deleted_papers WHERE session_id = %s", (session_id,))
        return [{**r, "tombstone": True, "id": None, "authors": []} for r in rows]

    def paper_gap_counts(self, session_id: str) -> dict[str, int]:
        rows = self._fetchall(
            """SELECT e.paper_id, COUNT(DISTINCT e.gap_id) AS n FROM gap_evidence e JOIN gaps g ON g.id = e.gap_id
               WHERE g.session_id = %s AND g.status != 'merged' GROUP BY e.paper_id""", (session_id,))
        return {r["paper_id"]: int(r["n"]) for r in rows}

    def papers_by_ref(self, session_id: str) -> dict[str, dict]:
        return {p["ref"]: p for p in self.list_papers(session_id, selection="selected") if p["ref"]}

    # ── summaries ──

    def save_summary(self, paper_id: str, analysis_level: str, sections: dict, model: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO paper_summaries (id, paper_id, analysis_level, sections, model)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (paper_id) DO UPDATE SET analysis_level = EXCLUDED.analysis_level,
                       sections = EXCLUDED.sections, model = EXCLUDED.model, updated_at = now()""",
                (new_id(), paper_id, analysis_level, Jsonb(_clean(sections)), model),
            )

    def get_summary(self, paper_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM paper_summaries WHERE paper_id = %s", (paper_id,))

    def summaries_for_session(self, session_id: str) -> dict[str, dict]:
        rows = self._fetchall(
            "SELECT s.* FROM paper_summaries s JOIN papers p ON p.id = s.paper_id WHERE p.session_id = %s",
            (session_id,))
        return {r["paper_id"]: r for r in rows}

    # ── relations ──

    def replace_relations(self, session_id: str, paper_id: str, relations: list[dict]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM paper_relations WHERE paper_id = %s", (paper_id,))
            for rel in relations:
                conn.execute(
                    """INSERT INTO paper_relations (id, session_id, paper_id, related_paper_id, relationship,
                                                    similarities, differences, contradiction)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (new_id(), session_id, paper_id, rel["related_paper_id"], rel["relationship"],
                     *_clean([rel.get("similarities", ""), rel.get("differences", ""), rel.get("contradiction", "")])),
                )

    def relations_for_paper(self, paper_id: str) -> list[dict]:
        return self._fetchall(
            """SELECT r.*, p.ref AS related_ref, p.title AS related_title,
                      CASE WHEN r.paper_id = %(p)s THEN r.related_paper_id ELSE r.paper_id END AS other_paper_id
               FROM paper_relations r
               JOIN papers p ON p.id = CASE WHEN r.paper_id = %(p)s THEN r.related_paper_id ELSE r.paper_id END
               WHERE r.paper_id = %(p)s OR r.related_paper_id = %(p)s
               ORDER BY r.seq""",
            {"p": paper_id})

    # ── gaps ──

    def create_gap(self, session_id: str, fields: dict) -> dict:
        gid = new_id()
        ref = f"G{self._next_seq(session_id, 'gap_seq')}"
        self._insert("gaps", {"id": gid, "session_id": session_id, "ref": ref,
                              "title": fields["title"], "description": fields.get("description", ""),
                              "gap_type": fields.get("gap_type", "other"),
                              "category": fields.get("category", "potential_gap")})
        return self.get_gap(gid)

    def get_gap(self, gap_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM gaps WHERE id = %s", (gap_id,))

    def update_gap(self, gap_id: str, **fields) -> None:
        self._update("gaps", gap_id, fields)

    def delete_gap(self, gap_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM gaps WHERE id = %s OR merged_into = %s", (gap_id, gap_id))

    def list_gaps(self, session_id: str, include_merged: bool = False) -> list[dict]:
        sql = """SELECT g.*,
                   (SELECT COUNT(*) FROM gap_evidence e WHERE e.gap_id = g.id) AS evidence_count,
                   (SELECT COUNT(DISTINCT e.paper_id) FROM gap_evidence e WHERE e.gap_id = g.id) AS paper_count
                 FROM gaps g WHERE g.session_id = %s"""
        if not include_merged:
            sql += " AND g.status != 'merged'"
        sql += " ORDER BY g.seq"
        gaps = self._fetchall(sql, (session_id,))
        related = self._fetchall(
            """SELECT DISTINCT e.gap_id, p.id, p.ref, p.title FROM gap_evidence e
               JOIN papers p ON p.id = e.paper_id JOIN gaps g ON g.id = e.gap_id WHERE g.session_id = %s
               ORDER BY p.ref""",
            (session_id,))
        by_gap: dict[str, list] = {}
        for r in related:
            by_gap.setdefault(r["gap_id"], []).append({"id": r["id"], "ref": r["ref"], "title": r["title"]})
        for g in gaps:
            g["related_papers"] = by_gap.get(g["id"], [])
        return gaps

    def add_evidence(self, gap_id: str, paper_id: str, evidence: str, relationship: str, basis: str,
                     source_paper_id: str | None = None) -> bool:
        """Insert evidence unless the same paper already gave identical evidence. Returns True if added.
        `source_paper_id` is the paper whose analysis produced it (the evidence may cite another paper)."""
        with self._conn() as conn:
            return conn.execute(
                """INSERT INTO gap_evidence (id, gap_id, paper_id, evidence, relationship, basis, source_paper_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (gap_id, paper_id, relationship, evidence) DO NOTHING""",
                (new_id(), gap_id, paper_id, _clean(evidence), relationship, basis, source_paper_id or paper_id)).rowcount > 0

    def evidence_for_gap(self, gap_id: str) -> list[dict]:
        return self._fetchall(
            """SELECT e.*, p.ref AS paper_ref, p.title AS paper_title FROM gap_evidence e
               JOIN papers p ON p.id = e.paper_id WHERE e.gap_id = %s ORDER BY e.seq""",
            (gap_id,))

    def move_evidence(self, from_gap: str, to_gap: str) -> None:
        with self._conn() as conn:
            # Evidence the target already has would violate the unique key: drop those duplicates first.
            conn.execute(
                """DELETE FROM gap_evidence e USING gap_evidence t
                   WHERE e.gap_id = %s AND t.gap_id = %s AND t.paper_id = e.paper_id
                     AND t.relationship = e.relationship AND t.evidence = e.evidence""",
                (from_gap, to_gap))
            conn.execute("UPDATE gap_evidence SET gap_id = %s WHERE gap_id = %s", (to_gap, from_gap))

    def remove_paper_evidence(self, session_id: str, paper_id: str) -> list[str]:
        """Delete the evidence this paper's analysis produced (whichever papers it cites) and its history
        entries; return the ids of the gaps it touched. Rows from before `source_paper_id` existed fall back
        to the cited paper."""
        with self._conn() as conn:
            rows = conn.execute(
                """DELETE FROM gap_evidence e USING gaps g
                   WHERE e.gap_id = g.id AND g.session_id = %s
                     AND (e.source_paper_id = %s OR (e.source_paper_id IS NULL AND e.paper_id = %s))
                   RETURNING e.gap_id""",
                (session_id, paper_id, paper_id)).fetchall()
            conn.execute("DELETE FROM gap_history WHERE session_id = %s AND paper_id = %s", (session_id, paper_id))
        return list(dict.fromkeys(r["gap_id"] for r in rows))

    def add_gap_history(self, gap_id: str, session_id: str, action: str, detail: str,
                        paper_id: str | None = None) -> None:
        self._insert("gap_history", {"id": new_id(), "gap_id": gap_id, "session_id": session_id,
                                     "paper_id": paper_id, "action": action, "detail": detail})

    def gap_history(self, gap_id: str) -> list[dict]:
        return self._fetchall(
            """SELECT h.*, p.ref AS paper_ref FROM gap_history h LEFT JOIN papers p ON p.id = h.paper_id
               WHERE h.gap_id = %s ORDER BY h.seq""", (gap_id,))

    def gaps_for_paper(self, paper_id: str) -> list[dict]:
        return self._fetchall(
            """SELECT DISTINCT ON (g.seq, e.relationship, e.evidence) g.id, g.ref, g.title, g.gap_type,
                      g.category, g.confidence, g.status, e.relationship, e.evidence
               FROM gap_evidence e JOIN gaps g ON g.id = e.gap_id
               WHERE e.paper_id = %s AND g.status != 'merged' ORDER BY g.seq, e.relationship, e.evidence""",
            (paper_id,))

    def reset_gaps(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM gaps WHERE session_id = %s", (session_id,))
            conn.execute("UPDATE sessions SET gap_seq = 0, synthesis = NULL, synthesis_updated_at = NULL "
                         "WHERE id = %s", (session_id,))

    # ── topics & autopilot ──

    def get_topic(self, name: str) -> dict | None:
        return self._fetchone("SELECT * FROM topics WHERE name = %s", (name,))

    def topic_for_session(self, session_id: str) -> str | None:
        row = self._fetchone("SELECT name FROM topics WHERE session_id = %s", (session_id,))
        return row["name"] if row else None

    def topics_by_session(self) -> dict[str, str]:
        return {r["session_id"]: r["name"]
                for r in self._fetchall("SELECT name, session_id FROM topics WHERE session_id IS NOT NULL")}

    def ensure_topic(self, name: str) -> dict:
        with self._conn() as conn:
            conn.execute("INSERT INTO topics (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (name,))
        return self.get_topic(name)

    def update_topic(self, name: str, **fields) -> None:
        self.ensure_topic(name)
        if not fields:
            return
        cols = ", ".join(f"{k} = %s" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE topics SET {cols} WHERE name = %s", (*[_clean(v) for v in fields.values()], name))

    def delete_topic(self, name: str) -> None:
        topic = self.get_topic(name)
        with self._conn() as conn:
            if topic and topic["session_id"]:
                conn.execute("DELETE FROM sessions WHERE id = %s", (topic["session_id"],))
            conn.execute("DELETE FROM topics WHERE name = %s", (name,))
            conn.execute("UPDATE autopilot SET active_topic = NULL WHERE active_topic = %s", (name,))
            conn.execute("UPDATE autopilot SET requested_topic = NULL WHERE requested_topic = %s", (name,))

    def get_autopilot(self) -> dict:
        row = self._fetchone("SELECT * FROM autopilot WHERE id = 1")
        if row is None:
            with self._conn() as conn:
                conn.execute("INSERT INTO autopilot (id) VALUES (1) ON CONFLICT (id) DO NOTHING")
            row = self._fetchone("SELECT * FROM autopilot WHERE id = 1")
        return row

    def update_autopilot(self, **fields) -> None:
        fields["updated_at"] = datetime.now(timezone.utc)
        cols = ", ".join(f"{k} = %s" for k in fields)
        with connection() as conn:  # always committed immediately: the UI polls this row
            conn.execute(f"UPDATE autopilot SET {cols} WHERE id = 1", tuple(_clean(v) for v in fields.values()))

    def heartbeat(self) -> None:
        with connection() as conn:
            conn.execute("UPDATE autopilot SET heartbeat_at = now() WHERE id = 1")

    def prune(self, events_days: int = 90, error_days: int = 365, cache_days: int = 60) -> dict[str, int]:
        """Retention for a server that runs for months: old activity events and old LLM cache entries."""
        with connection() as conn:
            events = conn.execute(
                """DELETE FROM events WHERE (type <> 'error' AND created_at < now() - make_interval(days => %s))
                                         OR created_at < now() - make_interval(days => %s)""",
                (events_days, error_days)).rowcount
            cache = conn.execute("DELETE FROM llm_cache WHERE created_at < now() - make_interval(days => %s)",
                                 (cache_days,)).rowcount
        return {"events": events, "llm_cache": cache}

    # ── events (always committed immediately, even inside atomic()) ──

    def add_event(self, session_id: str, type_: str, message: str, data: dict | None = None) -> None:
        with connection() as conn:
            conn.execute("INSERT INTO events (session_id, type, message, data) VALUES (%s, %s, %s, %s)",
                         (session_id, type_, _clean(message), Jsonb(_clean(data)) if data is not None else None))

    def list_events(self, session_id: str, after: int = 0, limit: int = 500, latest: bool = False) -> list[dict]:
        """Events in order. `latest` returns the most recent `limit` events (a long-running topic has thousands)."""
        with connection() as conn:
            if latest:
                rows = conn.execute("SELECT * FROM (SELECT * FROM events WHERE session_id = %s AND id > %s "
                                    "ORDER BY id DESC LIMIT %s) t ORDER BY id", (session_id, after, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM events WHERE session_id = %s AND id > %s ORDER BY id LIMIT %s",
                                    (session_id, after, limit)).fetchall()
        return [_row(r) for r in rows]


_store: ResearchStore | None = None
_store_lock = threading.Lock()


def get_store() -> ResearchStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ResearchStore()
        return _store


def set_store(store: ResearchStore | None) -> None:
    """Override the global store (used by tests)."""
    global _store
    with _store_lock:
        _store = store
