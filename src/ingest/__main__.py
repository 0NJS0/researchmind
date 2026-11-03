import argparse
import logging
import sys

from src.config import EMBEDDING_MODEL, PAPERS_DIR, validate_config
from src.db import init_schema
from src.ingest.chunker import chunk_documents
from src.ingest.loader import load_papers
from src.ingest.vectorstore import add_documents, delete_topic_papers, reset_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index papers under PAPERS_DIR into PostgreSQL/pgvector")
    parser.add_argument("--topic", type=str, default=None, help="Topic folder to ingest (default: all topics)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the whole index first (required after changing EMBEDDING_MODEL)")
    args = parser.parse_args()

    try:
        validate_config()
        init_schema()
        if args.reset:
            logger.info("Resetting the vector index")
            reset_index()
        elif args.topic:
            delete_topic_papers(args.topic)

        logger.info("Loading papers from %s [topic: %s]...", PAPERS_DIR, args.topic or "all topics")
        docs = load_papers(PAPERS_DIR, topic=args.topic)
        logger.info("Loaded %d pages", len(docs))
        chunks = chunk_documents(docs)
        logger.info("Split into %d chunks; embedding with %s...", len(chunks), EMBEDDING_MODEL)
        stored = add_documents(chunks, progress=lambda done, total: logger.info("Embedded %d/%d", done, total))
        logger.info("Done — %d chunks stored", stored)
    except Exception as e:
        logger.error("Ingest failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
