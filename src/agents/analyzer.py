from src.llm.llm import llm
from src.graph.state import ResearchState


def analysis_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"analysis": "No documents found to analyze."}

    context = ""

    for d in docs:
        paper_id = d.metadata.get("paper_id", "unknown")
        context += f"\nPaper: {paper_id}\n{d.page_content}"

    prompt=f"""
You are a research analyst.
Analyze these papers.
Extract:
1. Dataset used
2. Model architecture
3. Training method
4. Evaluation metrics
5. Main results
Papers:
{context}
"""
    try:
        response= llm.invoke(prompt)
        return {
            "analysis": response.content
        }

    except Exception as e:
        return {"analysis": f"Analysis failed: {e}"}