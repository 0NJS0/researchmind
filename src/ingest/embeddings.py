"""Embeddings via an OpenAI-compatible /embeddings endpoint (OpenRouter by default)."""
import logging
import threading

from src.config import (
    is_allowed_model, EMBEDDING_API_KEY, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, EMBEDDING_TIMEOUT, EMBEDDING_URL,
    LLM_PROVIDER, OPENROUTER_APP_NAME, OPENROUTER_APP_URL,
)
from src.llm.llm import LLMError, limiter_for, post_json

logger = logging.getLogger(__name__)


class EmbeddingError(LLMError):
    pass


class RemoteEmbeddings:
    def __init__(self, model: str = EMBEDDING_MODEL, base_url: str = EMBEDDING_URL,
                 api_key: str = EMBEDDING_API_KEY, batch_size: int = EMBEDDING_BATCH_SIZE):
        if not model:
            raise EmbeddingError("EMBEDDING_MODEL is not configured.")
        if not is_allowed_model(model):
            raise EmbeddingError(f"'{model}' is not a free model and LLM_FREE_ONLY is on.")
        self.model = model
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = api_key
        self.batch_size = max(1, batch_size)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if LLM_PROVIDER == "openrouter" or "openrouter.ai" in self.url:
            headers["X-Title"] = OPENROUTER_APP_NAME
            if OPENROUTER_APP_URL:
                headers["HTTP-Referer"] = OPENROUTER_APP_URL
        return headers

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = post_json(self.url, {"model": self.model, "input": texts, "encoding_format": "float"},
                         self._headers(), timeout=EMBEDDING_TIMEOUT, limiter=limiter_for(self.url))
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError(f"Embedding endpoint returned {len(items or [])} vectors for {len(texts)} inputs")
        items = sorted(items, key=lambda d: d.get("index", 0))
        vectors = [d.get("embedding") for d in items]
        if not all(isinstance(v, list) and v for v in vectors):
            raise EmbeddingError("Embedding endpoint returned an empty or non-float vector")
        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            raise EmbeddingError("Embedding endpoint returned vectors of different sizes")
        return vectors

    def embed_documents(self, texts: list[str], progress=None) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i + self.batch_size]))
            if progress:
                progress(min(len(texts), i + self.batch_size), len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


_embeddings: RemoteEmbeddings | None = None
_lock = threading.Lock()


def get_embeddings() -> RemoteEmbeddings:
    global _embeddings
    with _lock:
        if _embeddings is None:
            _embeddings = RemoteEmbeddings()
        return _embeddings


def set_embeddings(embeddings) -> None:
    """Override the embeddings client (used by tests)."""
    global _embeddings
    with _lock:
        _embeddings = embeddings
