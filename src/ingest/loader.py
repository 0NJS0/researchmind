import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)


def load_papers(folder: str, topic: str | None = None) -> list:
    documents = []
    folder_path = Path(folder)

    topics = [topic] if topic else [t.name for t in folder_path.iterdir() if t.is_dir()]

    for topic_name in topics:
        topic_dir = folder_path / topic_name
        if not topic_dir.is_dir():
            continue
        for paper_entry in topic_dir.iterdir():
            if not paper_entry.is_dir():
                logger.warning("Skipping '%s': papers must be in a subdirectory under '%s'", paper_entry, topic_name)
                continue
            pdfs = list(paper_entry.glob("*.pdf"))
            for pdf in pdfs:
                try:
                    loader = PyMuPDFLoader(str(pdf))
                    docs = loader.load()
                    for d in docs:
                        d.metadata["paper_id"] = paper_entry.name
                        d.metadata["source"] = pdf.name
                        d.metadata["topic"] = topic_name
                    documents.extend(docs)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", pdf, e)

    return documents
