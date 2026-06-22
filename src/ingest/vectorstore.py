import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_URL, COLLECTION_NAME, PAPERS_DIR
from src.ingest.embeddings import get_embeddings
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.exceptions import ResponseHandlingException
from httpx import ConnectTimeout

logger = logging.getLogger(__name__)


QDRANT_TIMEOUT = 300


def _make_client(timeout: int = QDRANT_TIMEOUT):
    return QdrantClient(url=QDRANT_URL, timeout=timeout)


def _topic_filter(topic: str | None, prefix: str = "metadata.") -> models.Filter | None:
    if topic:
        return models.Filter(
            must=[models.FieldCondition(key=f"{prefix}topic", match=models.MatchValue(value=topic))]
        )
    return None


def check_qdrant():
    try:
        client = _make_client(timeout=5)
        client.get_collections()
        return True, "connected"
    except (ConnectionError, ConnectTimeout, ResponseHandlingException):
        return False, "Qdrant endpoint unreachable"
    except UnexpectedResponse:
        return False, "Qdrant endpoint unreachable"


def create_store():
    client = _make_client()
    try:
        client.get_collection(COLLECTION_NAME)
    except UnexpectedResponse:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 1024, "distance": "Cosine"},
        )
    except (ConnectTimeout, ResponseHandlingException, ConnectionError) as e:
        raise ConnectionError(
            f"Cannot connect to Qdrant at {QDRANT_URL}. Make sure Qdrant is running "
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def add_documents(docs):
    store = create_store()
    store.add_documents(docs)


def get_retriever(k: int = 10, topic: str | None = None):
    store = create_store()
    search_kwargs = {"k": k}
    if topic:
        search_kwargs["filter"] = _topic_filter(topic)
    return store.as_retriever(search_kwargs=search_kwargs)


def _get_meta(payload: dict, key: str, default: str = "") -> str:
    meta = payload.get("metadata", {})
    return meta.get(key, default)


def list_papers(topic: str | None = None):
    client = _make_client(timeout=30)
    try:
        client.get_collection(COLLECTION_NAME)
    except UnexpectedResponse:
        return []
    except (ConnectTimeout, ResponseHandlingException, ConnectionError) as e:
        raise ConnectionError(f"Cannot connect to Qdrant at {QDRANT_URL}. Error: {e}")

    papers = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=_topic_filter(topic),
        )
        for point in points:
            payload = point.payload or {}
            paper_id = _get_meta(payload, "paper_id", "unknown")
            source = _get_meta(payload, "source", "unknown")
            paper_topic = _get_meta(payload, "topic", topic or "unknown")
            if paper_id not in papers:
                papers[paper_id] = {"source": source, "topic": paper_topic, "chunks": 0}
            papers[paper_id]["chunks"] += 1
        if next_offset is None:
            break

    return [
        {"paper_id": pid, "source": info["source"], "topic": info["topic"], "chunks": info["chunks"]}
        for pid, info in papers.items()
    ]


def paper_exists(paper_id: str, topic: str) -> bool:
    try:
        client = _make_client(timeout=10)
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=False,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.paper_id", match=models.MatchValue(value=paper_id)),
                    models.FieldCondition(key="metadata.topic", match=models.MatchValue(value=topic)),
                ]
            ),
        )
        return len(points) > 0
    except (UnexpectedResponse, ConnectTimeout, ResponseHandlingException, ConnectionError):
        return False


def delete_paper(paper_id: str, topic: str) -> bool:
    try:
        client = _make_client(timeout=30)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.paper_id", match=models.MatchValue(value=paper_id)),
                        models.FieldCondition(key="metadata.topic", match=models.MatchValue(value=topic)),
                    ]
                )
            ),
        )
        return True
    except (UnexpectedResponse, ConnectTimeout, ResponseHandlingException, ConnectionError) as e:
        raise ConnectionError(f"Failed to delete paper: {e}")


def delete_topic_papers(topic: str) -> bool:
    try:
        client = _make_client(timeout=30)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=_topic_filter(topic)
            ),
        )
        return True
    except (UnexpectedResponse, ConnectTimeout, ResponseHandlingException, ConnectionError) as e:
        logger.warning("Failed to clear topic '%s': %s", topic, e)
        return False


def list_topics() -> list[str]:
    folder = Path(PAPERS_DIR)
    if not folder.exists():
        return []
    return sorted([t.name for t in folder.iterdir() if t.is_dir()])
