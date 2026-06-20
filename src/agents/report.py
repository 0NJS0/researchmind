from src.llm.llm import llm
from src.graph.state import ResearchState

ERROR_PREFIXES = (
    "No documents found to analyze.",
    "Analysis failed:",
    "No analysis available to compare.",
    "comparison failed:",
    "No comparison data available to analyze gaps.",
    "Gap Detector failed:",
)


def report_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    comparison = state.get("comparison", "")
    gaps = state.get("gaps", "")

    if (not analysis or not comparison or not gaps
            or analysis.startswith(ERROR_PREFIXES)
            or comparison.startswith(ERROR_PREFIXES)
            or gaps.startswith(ERROR_PREFIXES)):
        return {"report": "Insufficient data to generate a report."}

    history = state.get("history", "")
    history_section = f"\nConversation History:\n{history}\n" if history else ""

    prompt = f"""
Create a professional research report.

Question:
{state.get("question", "")}
{history_section}
Analysis:
{analysis}

Comparison:
{comparison}

Research Gaps:
{gaps}

Format:
# Summary

# Methodology comparison

# Research Gaps

# Future Work
"""
    try:
        result = llm.invoke(prompt)
        return {"report": result.content}
    except Exception as e:
        return {"report": f"Report Generation failed: {e}"}