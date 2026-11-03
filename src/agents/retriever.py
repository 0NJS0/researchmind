import logging

from src.agents.common import error_text
from src.graph.state import ResearchState
from src.ingest.vectorstore import get_topic_chunks, similarity_search
from src.reports import report_jobs

logger = logging.getLogger(__name__)


def retrieve_agent(state: ResearchState) -> dict:
    topic = state.get("topic") or None
    try:
        if state.get("mode") == "report":
            # A report covers every paper in the topic, so take all chunks in reading order.
            report_jobs.update(state.get("job_id"), stage="retrieving")
            docs = get_topic_chunks(topic) if topic else []
        else:
            question = state.get("question", "")
            logger.info("Retrieving documents for: %s", question[:80])
            docs = similarity_search(question, k=state.get("k", 10), topic=topic)
    except Exception as e:
        logger.exception("Retrieval failed")
        return {"documents": [], "errors": [error_text("Retrieval", e)]}
    logger.info("Retrieved %d chunks", len(docs))
    if not docs:
        where = f"topic '{topic}'" if topic else "the library"
        return {"documents": [], "errors": [f"No indexed content found in {where}. Upload or re-index papers first."]}
    return {"documents": docs}
