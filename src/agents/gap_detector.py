import logging

from src.llm.llm import llm
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)

ERROR_PREFIXES = (
    "No analysis available to compare.",
    "comparison failed:",
)


def gap_agent(state: ResearchState) -> dict:
    comparison = state.get("comparison", "")
    if not comparison or comparison.startswith(ERROR_PREFIXES):
        return {"gaps": "No comparison data available to analyze gaps."}

    logger.info("Detecting research gaps...")

    prompt = f"""You are identifying research gaps and opportunities for future work.

Based on the comparison below, produce a thorough gap analysis.

{comparison}

Organize into these sections:

## Per-Paper Gaps
Specific limitations or unanswered questions from each individual paper.

## Cross-Cutting Gaps
Problems, questions, or limitations that none of the papers fully address.

## Missing Investigations
Critical studies, analyses, or examinations that should have been done but were not:
- Alternative approaches or perspectives not explored
- Additional data, sources, cases, or populations needed
- Follow-up work that would validate or extend the findings
- Long-term or longitudinal work absent

## Open Research Questions
- Theoretical, empirical, or methodological questions left unanswered
- Debates or disagreements that need resolution

## Concrete Next Steps
For each significant gap, propose a specific next investigation:
- What would be studied or done
- What resources would be needed
- What success would look like
- Estimated effort (modest / substantial / multi-year)

## Most Impactful Next Step
The single most valuable investigation to pursue next."""
    try:
        result = llm.invoke(prompt)
        return {"gaps": result.content}
    except Exception as e:
        return {"gaps": f"Gap Detector failed: {e}"}