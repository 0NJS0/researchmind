import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: Path, paper_id: str, topic: str, source: str | None = None) -> list[Document]:
    """One Document per page with non-empty text. Page numbers are 1-based."""
    docs = []
    with fitz.open(str(path)) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text().replace("\x00", "").strip()
            if not text:
                continue
            docs.append(Document(page_content=text, metadata={
                "paper_id": paper_id, "topic": topic, "source": source or path.name, "page": i + 1}))
    return docs


TEXT_FILE = "text.txt"  # extracted text of a paper found by the agent without a PDF


def load_text(paper_dir: Path, paper_id: str, topic: str) -> list[Document]:
    path = paper_dir / TEXT_FILE
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    if not text:
        return []
    try:
        source = json.loads((paper_dir / "paper.json").read_text(encoding="utf-8")).get("url") or "web"
    except (OSError, ValueError):
        source = "web"
    return [Document(page_content=text, metadata={"paper_id": paper_id, "topic": topic, "source": source, "page": 0})]


def load_papers(folder: str, topic: str | None = None) -> list[Document]:
    """Load every PDF under <folder>/<topic>/<paper_id>/*.pdf (or the saved text of papers without one)."""
    documents = []
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return documents

    topics = [topic] if topic else [t.name for t in folder_path.iterdir() if t.is_dir()]
    for topic_name in topics:
        topic_dir = folder_path / topic_name
        if not topic_dir.is_dir():
            continue
        for paper_entry in sorted(topic_dir.iterdir()):
            if paper_entry.name.startswith(("_", ".")):
                continue  # _reports, in-progress uploads and other internal entries
            if not paper_entry.is_dir():
                logger.warning("Skipping '%s': papers must be in a subdirectory under '%s'", paper_entry, topic_name)
                continue
            pdfs = sorted(f for f in paper_entry.iterdir() if f.is_file() and f.suffix.lower() == ".pdf")
            for pdf in pdfs:
                try:
                    documents.extend(load_pdf(pdf, paper_entry.name, topic_name))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", pdf, e)
            if not pdfs and (paper_entry / TEXT_FILE).is_file():
                try:
                    documents.extend(load_text(paper_entry, paper_entry.name, topic_name))
                except OSError as e:
                    logger.warning("Failed to load %s: %s", paper_entry / TEXT_FILE, e)

    return documents
