from src.llm.llm import llm
from src.graph.state import ResearchState

ERROR_PREFIXES = (
    "No analysis available to compare.",
    "comparison failed:",
)


def gap_agent(state: ResearchState) -> dict:
    comparison = state.get("comparison", "")
    if not comparison or comparison.startswith(ERROR_PREFIXES):
        return {"gaps": "No comparison data available to analyze gaps."}

    prompt = f"""
You are a senior research advisor.
Analyze these papers:
{comparison}
Find:
1. Research limitations
2. Missing experiments
3. Unsolved problems
4. Future research opportunities
Suggest realistic publishable directions.
"""
    try:
        result = llm.invoke(prompt)
        return {"gaps": result.content}
    except Exception as e:
        return {"gaps": f"Gap Detector failed: {e}"}