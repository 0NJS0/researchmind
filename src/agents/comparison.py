import logging

from src.llm.llm import llm
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)

ERROR_PREFIXES = ("No documents found to analyze.", "Analysis failed:")


def comparison_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    domain = state.get("domain_classification", "")
    if not analysis or analysis.startswith(ERROR_PREFIXES):
        return {"comparison": "No analysis available to compare."}

    logger.info("Comparing analyses...")

    domain_hint = f"\nDomain of these papers: {domain}\n" if domain else "\n"

    prompt = f"""You are producing a structured comparison of research papers.{domain_hint}

Based on the detailed per-paper analyses below, produce a comprehensive comparison.

{analysis}

## Comparison Table (markdown)
Build a table comparing ALL papers across meaningful dimensions for this field. Choose columns that capture what matters for this type of research. Include at minimum:
- Paper identification
- Data / sources used
- Methodology / approach
- Key results
- Limitations
Add any additional columns important for this specific domain.

## Cross-Paper Synthesis
- Common patterns and themes across papers
- Contradictions or disagreements and what might explain them
- Methodological quality differences

## Unique Contributions
What each paper contributes that others in this set do not"""
    try:
        result = llm.invoke(prompt)
        return {"comparison": result.content}
    except Exception as e:
        return {"comparison": f"comparison failed: {e}"}