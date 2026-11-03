"""Chunk storage and similarity search on PostgreSQL + pgvector.

The embedding dimension depends on EMBEDDING_MODEL and is only known after the first call, so the
``chunks.embedding`` column is dimensionless and the HNSW index is built on a cast expression once
the dimension is recorded in ``embedding_meta``. Changing the embedding model requires re-indexing.
"""
import logging
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL, PAPERS_DIR
from src.db import DatabaseError, connection, init_schema, pgvector_version
from src.ingest.embeddings import get_embeddings

logger = logging.getLogger(__name__)

HNSW_MAX_DIM = 2000        # pgvector limit for vector indexes
HALFVEC_MAX_DIM = 4000     # halfvec indexes (pgvector >= 0.7)


class EmbeddingMismatch(DatabaseError):
    pass


def _vec(values: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in values) + "]"


# ── embedding metadata & index ──

def get_embedding_meta() -> dict | None:
    init_schema()
    with connection() as conn:
        return conn.execute("SELECT model, dim FROM embedding_meta WHERE id = 1").fetchone()


def _create_index(conn, dim: int) -> None:
    version = pgvector_version()
    if version and version < (0, 5):
        logger.warning("pgvector %s has no HNSW support; searches will scan sequentially", version)
        return
    if dim <= HNSW_MAX_DIM:
        conn.execute(f"CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_{dim} ON chunks "
                     f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops)")
    elif dim <= HALFVEC_MAX_DIM and version >= (0, 7):
        conn.execute(f"CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_{dim} ON chunks "
                     f"USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)")
    else:
        logger.warning("Embedding dimension %d is too large to index with pgvector %s; "
                       "searches will scan sequentially", dim, ".".join(map(str, version)))


def _ensure_dim(conn, dim: int) -> None:
    # The meta row may not exist yet (FOR UPDATE then locks nothing): serialise first-time index creation.
    conn.execute("SELECT pg_advisory_xact_lock(hashtext('researchmind_embedding_meta'))")
    meta = conn.execute("SELECT model, dim FROM embedding_meta WHERE id = 1 FOR UPDATE").fetchone()
    if meta and meta["model"] == EMBEDDING_MODEL and meta["dim"] == dim:
        return
    has_chunks = conn.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()["exists"]
    if meta and has_chunks:
        raise EmbeddingMismatch(
            f"The index was built with embedding model '{meta['model']}' ({meta['dim']} dims) but "
            f"EMBEDDING_MODEL is '{EMBEDDING_MODEL}' ({dim} dims). Re-index everything "
            "(Library → Re-index all, or `python -m src.ingest --reset`).")
    _drop_indexes(conn)
    conn.execute("INSERT INTO embedding_meta (id, model, dim) VALUES (1, %s, %s) "
                 "ON CONFLICT (id) DO UPDATE SET model = EXCLUDED.model, dim = EXCLUDED.dim, created_at = now()",
                 (EMBEDDING_MODEL, dim))
    _create_index(conn, dim)


def _drop_indexes(conn) -> None:
    rows = conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' "
                        "AND schemaname = current_schema() AND indexname LIKE 'chunks_embedding_hnsw_%%'").fetchall()
    for r in rows:
        if re.fullmatch(r"chunks_embedding_hnsw_\d+", r["indexname"]):
            conn.execute(f"DROP INDEX IF EXISTS {r['indexname']}")


def reset_index() -> None:
    """Delete every chunk and the recorded embedding model (needed after changing EMBEDDING_MODEL)."""
    init_schema()
    with connection() as conn:
        conn.execute("TRUNCATE chunks")
        conn.execute("DELETE FROM embedding_meta")
        _drop_indexes(conn)


# ── writes ──

def _check_model_before_embedding() -> None:
    """Fail before spending embedding requests when the index was built with another model."""
    with connection() as conn:
        meta = conn.execute("SELECT model, dim FROM embedding_meta WHERE id = 1").fetchone()
        if not meta or meta["model"] == EMBEDDING_MODEL:
            return
        if conn.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()["exists"]:
            raise EmbeddingMismatch(
                f"The index was built with embedding model '{meta['model']}' but EMBEDDING_MODEL is "
                f"'{EMBEDDING_MODEL}'. Re-index everything (Library → Re-index all, or `python -m src.ingest --reset`).")


def add_documents(docs: list[Document], progress=None) -> int:
    """Embed and store chunks. Each (topic, paper_id) is replaced atomically, so re-indexing never
    leaves duplicates. Returns the number of chunks stored."""
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        return 0
    init_schema()
    _check_model_before_embedding()
    vectors = get_embeddings().embed_documents([d.page_content for d in docs], progress=progress)
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise EmbeddingMismatch("The embedding model returned vectors of different sizes; nothing was stored.")

    by_paper: dict[tuple[str, str], list[tuple[Document, list[float]]]] = defaultdict(list)
    for doc, vec in zip(docs, vectors):
        md = doc.metadata
        by_paper[(md.get("topic", ""), md.get("paper_id", "unknown"))].append((doc, vec))

    with connection() as conn:
        _ensure_dim(conn, dim)
        for (topic, paper_id), items in by_paper.items():
            conn.execute("DELETE FROM chunks WHERE topic = %s AND paper_id = %s", (topic, paper_id))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (topic, paper_id, source, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
                    [(topic, paper_id, str(d.metadata.get("source", "")).replace("\x00", ""),
                      int(d.metadata.get("page") or 0), i, d.page_content.replace("\x00", ""), _vec(v))
                     for i, (d, v) in enumerate(items)],
                )
    return len(docs)


def delete_paper(paper_id: str, topic: str) -> int:
    init_schema()
    with connection() as conn:
        return conn.execute("DELETE FROM chunks WHERE topic = %s AND paper_id = %s", (topic, paper_id)).rowcount


def delete_topic_papers(topic: str) -> int:
    init_schema()
    with connection() as conn:
        return conn.execute("DELETE FROM chunks WHERE topic = %s", (topic,)).rowcount


# ── reads ──

def _to_document(row: dict) -> Document:
    return Document(page_content=row["content"], metadata={
        "chunk_id": row["id"], "topic": row["topic"], "paper_id": row["paper_id"], "source": row["source"],
        "page": row["page"], "chunk_index": row["chunk_index"],
        **({"score": round(float(row["score"]), 4)} if row.get("score") is not None else {}),
    })


def similarity_search(query: str, k: int = 10, topic: str | None = None) -> list[Document]:
    meta = get_embedding_meta()
    if not meta:
        return []  # nothing indexed yet
    if meta["model"] != EMBEDDING_MODEL:
        raise EmbeddingMismatch(f"The index was built with '{meta['model']}' but EMBEDDING_MODEL is "
                                f"'{EMBEDDING_MODEL}'. Re-index to search.")
    qvec = get_embeddings().embed_query(query)
    dim = meta["dim"]
    if len(qvec) != dim:
        raise EmbeddingMismatch(f"Query embedding has {len(qvec)} dims but the index has {dim}. Re-index.")
    lit = _vec(qvec)
    cols = "id, topic, paper_id, source, page, chunk_index, content"
    with connection() as conn:
        if topic:
            # Exact search within one topic: uses the topic index, never misses rows to HNSW post-filtering.
            rows = conn.execute(
                f"SELECT {cols}, 1 - (embedding <=> %s::vector) AS score FROM chunks WHERE topic = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s", (lit, topic, lit, k)).fetchall()
        else:
            # Match the HNSW index expression so the planner can use it.
            if dim <= HNSW_MAX_DIM:
                cast = f"vector({dim})"
            elif dim <= HALFVEC_MAX_DIM and pgvector_version() >= (0, 7):
                cast = f"halfvec({dim})"
            else:
                cast = "vector"
            rows = conn.execute(
                f"SELECT {cols}, 1 - (embedding::{cast} <=> %s::{cast}) AS score FROM chunks "
                f"ORDER BY embedding::{cast} <=> %s::{cast} LIMIT %s", (lit, lit, k)).fetchall()
    return [_to_document(r) for r in rows]


class Retriever:
    def __init__(self, k: int, topic: str | None):
        self.k = k
        self.topic = topic

    def invoke(self, query: str) -> list[Document]:
        return similarity_search(query, self.k, self.topic)


def get_retriever(k: int = 10, topic: str | None = None) -> Retriever:
    return Retriever(k, topic)


def get_topic_chunks(topic: str, paper_id: str | None = None) -> list[Document]:
    """Every chunk of a topic (or one paper) in reading order."""
    init_schema()
    sql = "SELECT id, topic, paper_id, source, page, chunk_index, content FROM chunks WHERE topic = %s"
    params: tuple = (topic,)
    if paper_id:
        sql += " AND paper_id = %s"
        params += (paper_id,)
    sql += " ORDER BY paper_id, chunk_index"
    with connection() as conn:
        return [_to_document(r) for r in conn.execute(sql, params).fetchall()]


def get_chunk(chunk_id: int) -> dict | None:
    init_schema()
    with connection() as conn:
        return conn.execute("SELECT id, topic, paper_id, source, page, chunk_index, content FROM chunks "
                            "WHERE id = %s", (chunk_id,)).fetchone()


def list_papers(topic: str | None = None) -> list[dict]:
    init_schema()
    sql = ("SELECT topic, paper_id, min(source) AS source, count(*) AS chunks, max(page) AS pages, "
           "max(created_at) AS indexed_at FROM chunks")
    params: tuple = ()
    if topic:
        sql += " WHERE topic = %s"
        params = (topic,)
    sql += " GROUP BY topic, paper_id ORDER BY topic, paper_id"
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{**r, "indexed_at": r["indexed_at"].isoformat() if r["indexed_at"] else None} for r in rows]


def paper_exists(paper_id: str, topic: str) -> bool:
    init_schema()
    with connection() as conn:
        return conn.execute("SELECT EXISTS (SELECT 1 FROM chunks WHERE topic = %s AND paper_id = %s)",
                            (topic, paper_id)).fetchone()["exists"]


def list_topics() -> list[str]:
    """Topics are folders under PAPERS_DIR (created even before any paper is uploaded)."""
    folder = Path(PAPERS_DIR)
    if not folder.exists():
        return []
    return sorted(t.name for t in folder.iterdir() if t.is_dir() and not t.name.startswith((".", "_")))


def check_vectorstore() -> tuple[bool, str]:
    try:
        meta = get_embedding_meta()
        with connection() as conn:
            count = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    except DatabaseError as e:
        return False, str(e)
    except Exception as e:
        return False, f"vector store error ({e.__class__.__name__})"
    if meta and meta["model"] != EMBEDDING_MODEL and count:
        return False, f"index built with '{meta['model']}' — re-index for '{EMBEDDING_MODEL}'"
    return True, f"{count} chunks" + (f" · {meta['dim']}-dim" if meta else "")
