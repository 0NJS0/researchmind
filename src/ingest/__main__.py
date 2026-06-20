import argparse
import logging
import sys
from src.ingest.loader import load_papers
from src.ingest.chunker import chunk_documents
from src.ingest.vectorstore import add_documents
from src.config import PAPERS_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest papers into Qdrant")
    parser.add_argument("--topic", type=str, default=None, help="Topic folder to ingest (default: all topics)")
    args = parser.parse_args()

    if not PAPERS_DIR:
        logger.error("Set PAPERS_DIR in .env")
        sys.exit(1)

    try:
        topic_label = args.topic or "all topics"
        logger.info("Loading papers from %s [topic: %s]...", PAPERS_DIR, topic_label)
        docs = load_papers(PAPERS_DIR, topic=args.topic)
        logger.info("Loaded %d pages", len(docs))

        chunks = chunk_documents(docs)
        logger.info("Split into %d chunks", len(chunks))

        logger.info("Adding %d chunks to Qdrant collection '%s'...", len(chunks), COLLECTION_NAME)
        add_documents(chunks)
        logger.info("Done — documents added to Qdrant")
    except Exception as e:
        logger.error("Ingest failed: %s", e)
        logger.exception("Traceback")
        sys.exit(1)

if __name__ == "__main__":
    main()