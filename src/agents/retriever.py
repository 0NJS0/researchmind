import logging

from src.ingest.vectorstore import get_retriever
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)


def retrieve_agent(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    k = state.get("k", 10)
    question = state.get("question", "")[:80]
    logger.info("Retrieving documents for: %s", question)
    retriever = get_retriever(k=k, topic=topic or None)
    docs = retriever.invoke(state.get("question", ""))
    logger.info("Retrieved %d documents", len(docs))
    return {"documents": docs}

    

