from src.llm.llm import llm
from src.graph.state import ResearchState

ERROR_PREFIXES = ("No documents found to analyze.", "Analysis failed:")


def comparison_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    if not analysis or analysis.startswith(ERROR_PREFIXES):
        return {"comparison": "No analysis available to compare."}

    prompt = f"""
You are comparing research papers.
Based on this analysis:
{analysis}
Create a comparison table:
Paper |
Dataset |
Model |
Method |
Strength |
Weakness
"""
    try:
        result = llm.invoke(prompt)
        return {"comparison": result.content}
    except Exception as e:
        return {"comparison": f"comparison failed: {e}"}