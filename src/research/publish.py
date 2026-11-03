"""Bridge between the research agent and the Library: discovered papers become Library papers
(stored under PAPERS_DIR and indexed in pgvector, so Ask and Reports cover them), and the user's
uploaded Library papers are seeded into the topic's research session."""
import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.ingest.chunker import chunk_documents
from src.ingest.loader import TEXT_FILE, load_pdf
from src.ingest.vectorstore import add_documents, get_topic_chunks, list_papers
from src.paths import inside_root, papers_root, to_name
from src.research.store import ResearchStore

logger = logging.getLogger(__name__)

META_FILE = "paper.json"


def read_meta(topic: str, paper_id: str) -> dict:
    path = papers_root() / topic / paper_id / META_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _folder_for(topic: str, research_id: str) -> str | None:
    """A Library folder already created for this research paper by an earlier, interrupted publish."""
    root = papers_root() / topic
    if not research_id or not root.is_dir():
        return None
    for folder in root.iterdir():
        if folder.is_dir() and not folder.name.startswith("_") and \
                read_meta(topic, folder.name).get("research_paper_id") == research_id:
            return folder.name
    return None


def _unique_id(topic: str, title: str) -> str:
    base = to_name(title, max_len=90) or "paper"
    candidate, n = base, 2
    while (papers_root() / topic / candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def publish_to_library(topic: str, paper: dict, pdf_bytes: bytes | None = None) -> str:
    """Save a processed research paper into the topic's Library folder and index it.

    Returns the Library paper id. Papers that came from the Library (user uploads) are left as they are."""
    if paper.get("library_paper_id"):
        return paper["library_paper_id"]
    # Reuse the folder of a publish that was cut off (e.g. by a restart) instead of creating a duplicate.
    paper_id = _folder_for(topic, paper.get("id", "")) or \
        _unique_id(topic, paper.get("title") or paper.get("ref") or "paper")
    folder = inside_root(topic, paper_id)
    if folder is None:
        raise ValueError("unsafe library path")
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "title": paper.get("title", ""), "authors": paper.get("authors") or [], "year": paper.get("year"),
        "venue": paper.get("venue", ""), "citation_count": paper.get("citation_count"),
        "doi": paper.get("doi", ""), "url": paper.get("url", ""), "pdf_url": paper.get("pdf_url", ""),
        "content": paper.get("content_status", ""), "origin": "agent", "research_ref": paper.get("ref", ""),
        "research_paper_id": paper.get("id", ""),
    }
    # Metadata first, so an interrupted publish leaves a folder that the retry recognises.
    (folder / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    docs: list[Document] = []
    if pdf_bytes:
        pdf_path = folder / f"{paper_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        try:
            docs = load_pdf(pdf_path, paper_id, topic, source=pdf_path.name)
        except Exception as e:  # corrupt PDF: fall back to the extracted text
            logger.warning("Could not re-read saved PDF for %s: %s", paper_id, e)
    if not docs:
        text = paper.get("raw_content") or paper.get("abstract") or ""
        if text.strip():
            # Kept on disk so a later re-index can rebuild this paper (it has no PDF to re-read).
            (folder / TEXT_FILE).write_text(text.replace("\x00", ""), encoding="utf-8")
            docs = [Document(page_content=text, metadata={"paper_id": paper_id, "topic": topic,
                                                           "source": paper.get("url") or "web", "page": 0})]
    if docs:
        add_documents(chunk_documents(docs))
    return paper_id


def attach_pdf(topic: str, library_id: str, pdf_bytes: bytes) -> bool:
    """Store a PDF for an existing Library paper and re-index it from the PDF (page-level citations)."""
    folder = inside_root(topic, library_id)
    if folder is None or not folder.is_dir():
        return False
    pdf_path = folder / f"{library_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    try:
        docs = load_pdf(pdf_path, library_id, topic, source=pdf_path.name)
    except Exception as e:
        logger.warning("Saved PDF for %s is unreadable: %s", library_id, e)
        pdf_path.unlink(missing_ok=True)
        return False
    if docs:
        add_documents(chunk_documents(docs))   # replaces the text-only chunks
    (folder / TEXT_FILE).unlink(missing_ok=True)  # the PDF is the source now
    meta_path = folder / META_FILE
    meta = read_meta(topic, library_id)
    if meta:
        meta["content"] = "full_text"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def seed_library_papers(store: ResearchStore, session_id: str, topic: str) -> int:
    """Add the topic's Library papers that the research session does not know yet (user uploads).

    Their text comes from the indexed chunks, so they are analysed without any download."""
    added = 0
    known = {p["library_paper_id"] for p in store.list_papers(session_id, selection=None) if p.get("library_paper_id")}
    for lib in list_papers(topic):
        pid = lib["paper_id"]
        if pid in known:
            continue
        meta = read_meta(topic, pid)
        if meta.get("origin") == "agent":
            continue  # published by the agent from a paper that was later removed from the session
        chunks = get_topic_chunks(topic, pid)
        text = "\n".join(c.page_content for c in chunks)
        if not text.strip():
            continue
        title = meta.get("title") or pid.replace("_", " ")
        doi = meta.get("doi") or ""
        from src.research.dedup import find_duplicate
        same = find_duplicate({"title": title, "doi": doi, "arxiv_id": meta.get("arxiv_id") or ""},
                              [p for p in store.list_papers(session_id, selection="selected") if not p.get("library_paper_id")])
        if same and meta.get("title"):  # identified by title/DOI (not just a file name): the same paper
            store.update_paper(same["id"], library_paper_id=pid, raw_content=text, content_status="full_text",
                               content_source="library")
            known.add(pid)
            continue
        store.add_paper(session_id, {
            "title": title, "raw_content": text, "content_status": "full_text", "content_source": "library",
            "selection_status": "selected", "relevance_score": 1.0, "relevance_method": "manual",
            "relevance_reason": "Uploaded to the Library by the user", "added_by": "user",
            "library_paper_id": pid, "processing_status": "pending", "priority_score": 2.0,
            "year": meta.get("year"), "venue": meta.get("venue") or "",
            # A link / DOI given with the upload identifies the paper for OpenAlex (authors, venue, citations).
            "doi": doi, "arxiv_id": meta.get("arxiv_id") or "",
            "url": meta.get("url") or (f"https://doi.org/{doi}" if doi else ""),
        }, assign_ref=True)
        added += 1
    return added


INCOMING_DIR = "_incoming"  # PDFs attached to papers that are not in the Library yet


def incoming_pdf_path(topic: str, research_paper_id: str) -> Path:
    return papers_root() / topic / INCOMING_DIR / f"{research_paper_id}.pdf"


def incoming_text_path(topic: str, research_paper_id: str) -> Path:
    """A full text found without a PDF (an HTML article), waiting to replace the abstract in the Library."""
    return papers_root() / topic / INCOMING_DIR / f"{research_paper_id}.txt"


def replace_library_text(topic: str, library_id: str, text: str) -> bool:
    """Replace a text-only Library paper's content (e.g. abstract → full article) and re-index it."""
    folder = inside_root(topic, library_id)
    if folder is None or not folder.is_dir():
        return False
    (folder / TEXT_FILE).write_text(text, encoding="utf-8")
    from src.ingest.loader import load_text
    docs = load_text(folder, library_id, topic)
    if docs:
        add_documents(chunk_documents(docs))
    return True


def library_paper_exists(topic: str, paper_id: str) -> bool:
    return Path(papers_root() / topic / paper_id).is_dir()
