from src.ingest.vectorstore import get_retriever
from src.graph.state import ResearchState


def retrieve_agent(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    k = state.get("k", 10)
    retriever = get_retriever(k=k, topic=topic or None)
    docs = retriever.invoke(state.get("question", ""))
    return {"documents": docs}

    

