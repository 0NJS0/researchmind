"""PostgreSQL access: one connection pool and the whole schema (vectors, research sessions, LLM cache).

The schema is created on startup (idempotent). Vectors are sent to Postgres as text literals and cast
with ``::vector``, so no client-side adapter registration is needed.
"""
import logging
import re
import threading
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import DATABASE_URL, DB_POOL_MAX, DB_POOL_MIN, DB_SCHEMA, mask_dsn

logger = logging.getLogger(__name__)

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class DatabaseError(RuntimeError):
    pass


SCHEMA_SQL = """
-- ── Vector store (uploaded PDFs) ──
CREATE TABLE IF NOT EXISTS embedding_meta (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    page INTEGER NOT NULL DEFAULT 0,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_topic_paper ON chunks (topic, paper_id, page, chunk_index);

-- ── LLM output cache (validated structured outputs only) ──
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Research agent ──
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'created',
    status_message TEXT NOT NULL DEFAULT '',
    topic_analysis JSONB,
    synthesis JSONB,
    synthesis_updated_at TIMESTAMPTZ,
    iteration INTEGER NOT NULL DEFAULT 0,
    paper_seq INTEGER NOT NULL DEFAULT 0,
    gap_seq INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS search_queries (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    result_count INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_queries_session ON search_queries (session_id, seq);
CREATE TABLE IF NOT EXISTS papers (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ref TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]',
    abstract TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    pdf_url TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    arxiv_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    engines JSONB NOT NULL DEFAULT '[]',
    venue TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT '',
    year INTEGER,
    citation_count INTEGER,
    raw_content TEXT NOT NULL DEFAULT '',
    content_status TEXT NOT NULL DEFAULT 'pending',
    content_source TEXT NOT NULL DEFAULT '',
    selection_status TEXT NOT NULL DEFAULT 'candidate',
    relevance_score DOUBLE PRECISION,
    relevance_reason TEXT NOT NULL DEFAULT '',
    relevance_method TEXT NOT NULL DEFAULT '',
    processing_status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    found_by_query TEXT NOT NULL DEFAULT '',
    added_by TEXT NOT NULL DEFAULT 'agent',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_papers_session ON papers (session_id, selection_status);
CREATE TABLE IF NOT EXISTS paper_summaries (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    analysis_level TEXT NOT NULL,
    sections JSONB NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_relations (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    related_paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    similarities TEXT NOT NULL DEFAULT '',
    differences TEXT NOT NULL DEFAULT '',
    contradiction TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_relations_paper ON paper_relations (paper_id);
CREATE INDEX IF NOT EXISTS idx_relations_related ON paper_relations (related_paper_id);
CREATE TABLE IF NOT EXISTS gaps (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ref TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    gap_type TEXT NOT NULL DEFAULT 'other',
    category TEXT NOT NULL DEFAULT 'potential_gap',
    confidence TEXT NOT NULL DEFAULT 'low',
    status TEXT NOT NULL DEFAULT 'open',
    merged_into TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_gaps_session ON gaps (session_id, seq);
CREATE TABLE IF NOT EXISTS gap_evidence (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gaps(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    evidence TEXT NOT NULL,
    relationship TEXT NOT NULL,
    basis TEXT NOT NULL DEFAULT 'inferred',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (gap_id, paper_id, relationship, evidence)
);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON gap_evidence (paper_id);
CREATE TABLE IF NOT EXISTS gap_history (
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gaps(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    paper_id TEXT REFERENCES papers(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_history_gap ON gap_history (gap_id, seq);
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, id);

-- ── Topics (Library folders) and the continuous autopilot ──
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    narrative TEXT NOT NULL DEFAULT '',
    narrative_paper_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS autopilot (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    active_topic TEXT,
    requested_topic TEXT,
    state TEXT NOT NULL DEFAULT 'idle',
    state_message TEXT NOT NULL DEFAULT '',
    current_paper_id TEXT,
    next_search_at TIMESTAMPTZ,
    paused_until TIMESTAMPTZ,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO autopilot (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- Papers the user deleted: searches skip them so the autopilot doesn't re-add them.
CREATE TABLE IF NOT EXISTS deleted_papers (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    arxiv_id TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deleted_papers_session ON deleted_papers (session_id);
"""

# Columns added after the first release: applied to existing databases on startup.
MIGRATIONS_SQL = """
ALTER TABLE papers ADD COLUMN IF NOT EXISTS priority_score DOUBLE PRECISION;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS priority_detail JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS publication_type TEXT NOT NULL DEFAULT '';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS peer_reviewed BOOLEAN;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS library_paper_id TEXT NOT NULL DEFAULT '';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS retry_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS fulltext_checked_at TIMESTAMPTZ;
-- The paper whose analysis produced a piece of gap evidence (which may cite other papers): re-analysing a
-- paper retracts exactly its own contributions.
ALTER TABLE gap_evidence ADD COLUMN IF NOT EXISTS source_paper_id TEXT;
ALTER TABLE autopilot ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;  -- the worker loop is alive
ALTER TABLE papers ADD COLUMN IF NOT EXISTS pdf_sha256 TEXT NOT NULL DEFAULT '';  -- a PDF the user attached
ALTER TABLE papers ADD COLUMN IF NOT EXISTS screen_deferrals INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at);
CREATE INDEX IF NOT EXISTS idx_llm_cache_created ON llm_cache (created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON gap_evidence (source_paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_queue ON papers (session_id, selection_status, processing_status, priority_score DESC);
ALTER TABLE gaps ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'unverified';
ALTER TABLE gaps ADD COLUMN IF NOT EXISTS verification_note TEXT NOT NULL DEFAULT '';
ALTER TABLE gaps ADD COLUMN IF NOT EXISTS verification_refs JSONB;
ALTER TABLE gaps ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS validations (
    id TEXT PRIMARY KEY,
    idea TEXT NOT NULL,
    project_type TEXT NOT NULL DEFAULT 'empirical_quantitative',
    topic TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    stage TEXT NOT NULL DEFAULT '',
    result JSONB,
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_schema_ready = False


def _schema_name() -> str:
    schema = DB_SCHEMA.strip().lower() or "public"
    if not _SCHEMA_RE.match(schema):
        raise DatabaseError(f"Invalid DB_SCHEMA '{DB_SCHEMA}'")
    return schema


def get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            if not DATABASE_URL:
                raise DatabaseError("DATABASE_URL is not configured.")
            schema = _schema_name()
            _pool = ConnectionPool(
                DATABASE_URL,
                min_size=max(0, DB_POOL_MIN),
                max_size=max(1, DB_POOL_MAX),
                kwargs={"row_factory": dict_row, "options": f"-c search_path={schema},public"},
                name="researchmind",
                open=False,
                # After a database restart, pooled connections are dead: test each one before handing it out.
                check=ConnectionPool.check_connection,
            )
            try:
                _pool.open(wait=True, timeout=15)
            except Exception as e:
                _pool = None
                raise DatabaseError(f"Cannot connect to PostgreSQL at {mask_dsn(DATABASE_URL)}: "
                                    f"{e.__class__.__name__}") from e
        return _pool


@contextmanager
def connection():
    """A pooled connection inside a transaction (committed on success, rolled back on error)."""
    with get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    """Create the pgvector extension and every table. Safe to call repeatedly and concurrently."""
    global _schema_ready
    if _schema_ready:
        return
    schema = _schema_name()
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('researchmind_schema'))")
        try:
            with conn.transaction():
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        except psycopg.Error as e:
            raise DatabaseError(
                "The pgvector extension is not available and could not be created "
                f"({e.__class__.__name__}). Install pgvector on the server and run "
                "`CREATE EXTENSION vector;` once as a superuser, then restart ResearchMind."
            ) from e
        if schema != "public":
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        conn.execute(SCHEMA_SQL)
        conn.execute(MIGRATIONS_SQL)
    _schema_ready = True
    logger.info("Database schema ready (schema=%s)", schema)


def pgvector_version() -> tuple[int, ...]:
    with connection() as conn:
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not row:
        return ()
    return tuple(int(p) for p in re.findall(r"\d+", row["extversion"])[:3])


def check_database() -> tuple[bool, str]:
    if not DATABASE_URL:
        return False, "DATABASE_URL not configured"
    try:
        with connection() as conn:
            conn.execute("SELECT 1")
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if not row:
            return False, "connected, but the pgvector extension is not installed"
        return True, f"connected (pgvector {row['extversion']})"
    except DatabaseError as e:
        return False, str(e)
    except psycopg.Error as e:
        return False, f"database error ({e.__class__.__name__})"


def close_pool() -> None:
    global _pool, _schema_ready
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        _schema_ready = False
