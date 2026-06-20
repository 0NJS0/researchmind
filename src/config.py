import os
from dotenv import load_dotenv

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")
PAPERS_DIR = os.getenv("PAPERS_DIR", "")


def validate_config():
    missing = []
    if not LLM_URL:
        missing.append("LLM_URL")
    if not LLM_MODEL:
        missing.append("LLM_MODEL")
    if not PAPERS_DIR:
        missing.append("PAPERS_DIR")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )

        

