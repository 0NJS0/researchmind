import logging

from src.llm.llm import llm
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)


def classifier_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"domain_classification": "Unknown domain"}

    logger.info("Classifying domain from %d documents...", len(docs))

    sample = ""
    for d in docs[:3]:
        pid = d.metadata.get("paper_id", "unknown")
        sample += f"Paper: {pid}\n{d.page_content[:2000]}\n\n"

    prompt = f"""Read these research paper excerpts and classify the academic domain.

{sample}

Return a single line in this exact format:
Domain: <domain> | Subfield: <subfield> | Type: <research type> | Methodology: <methodology>

Examples:
- "Domain: Computer Science | Subfield: Natural Language Processing | Type: Empirical | Methodology: Deep Learning"
- "Domain: Medicine | Subfield: Oncology | Type: Clinical Trial | Methodology: Randomized Controlled Trial"
- "Domain: Economics | Subfield: Labor Economics | Type: Empirical | Methodology: Econometric Analysis"
- "Domain: Sociology | Subfield: Political Sociology | Type: Survey | Methodology: Statistical Analysis"
- "Domain: Biology | Subfield: Genomics | Type: Experimental | Methodology: Sequencing Analysis"
- "Domain: History | Subfield: Medieval History | Type: Archival | Methodology: Textual Analysis"
- "Domain: Psychology | Subfield: Cognitive Psychology | Type: Experimental | Methodology: Behavioral Study"

Classification:"""
    try:
        response = llm.invoke(prompt)
        classification = response.content.strip()
        logger.info("Classified as: %s", classification)
        return {"domain_classification": classification}
    except Exception as e:
        logger.warning("Domain classification failed: %s", e)
        return {"domain_classification": f"Classification failed: {e}"}
