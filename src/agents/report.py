import logging

from src.llm.llm import llm
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)

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
    domain = state.get("domain_classification", "")

    if (not analysis or not comparison or not gaps
            or analysis.startswith(ERROR_PREFIXES)
            or comparison.startswith(ERROR_PREFIXES)
            or gaps.startswith(ERROR_PREFIXES)):
        return {"report": "Insufficient data to generate a report."}

    logger.info("Generating final report...")

    history = state.get("history", "")
    history_section = f"\nConversation History:\n{history}\n" if history else ""
    domain_hint = f"\nField of study: {domain}\n" if domain else "\n"

    prompt = f"""You are writing a comprehensive academic research synthesis report.{domain_hint}

Synthesize the following data into a polished, detailed report. Integrate findings across papers — do not simply repeat the raw sections. Present a clear narrative.

Original Question: {state.get("question", "")}
{history_section}

--- PER-PAPER ANALYSES ---
{analysis}

--- COMPARATIVE ANALYSIS ---
{comparison}

--- RESEARCH GAPS ---
{gaps}

Produce the report with ALL of the following sections:

# Executive Summary
2-3 paragraphs covering what was studied, how, key findings, main gaps, and the most important next step.

# Research Landscape
Summarize the questions, data, methods, and approaches across all papers. What are the dominant paradigms? What are the outliers?

# Key Findings
Numbered list of the most important conclusions. For each, note the strength of evidence (e.g., "Supported by 3 papers" vs. "Single paper finding").

# Comparative Analysis
How papers differ on central dimensions. Trade-offs revealed between different approaches.

# Limitations & Research Gaps
Per-paper limitations and cross-cutting open problems. Missing investigations.

# Future Directions
Concrete next steps: what to investigate, what resources needed, estimated effort.

# References
List all paper IDs referenced in this report.

Write in a formal academic tone. Be specific with numbers, names, and evidence. Prioritize well-supported claims and explicitly flag speculative or single-paper findings as preliminary."""
    try:
        result = llm.invoke(prompt)
        return {"report": result.content}
    except Exception as e:
        return {"report": f"Report Generation failed: {e}"}