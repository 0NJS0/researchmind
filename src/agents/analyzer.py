import logging

from src.llm.llm import llm
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)


def analysis_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    domain = state.get("domain_classification", "")
    if not docs:
        return {"analysis": "No documents found to analyze."}

    logger.info("Analyzing %d documents...", len(docs))

    context = ""

    for d in docs:
        paper_id = d.metadata.get("paper_id", "unknown")
        context += f"\nPaper: {paper_id}\n{d.page_content}\n"

    domain_hint = f"\nDomain of these papers: {domain}\n" if domain else "\n"

    prompt = f"""You are analyzing research papers.{domain_hint}
Papers:
{context}

For each paper, produce a structured analysis under headings that are standard for this field. Use terminology appropriate to the domain — the exact section names should match what a researcher in this field would expect.

Regardless of field, every analysis MUST include ALL of the following information:

## Data / Sources / Study Material
What data, evidence, sources, or subjects were used. Be specific: size, origin, characteristics.

## Methodology / Approach / Architecture
How the research was conducted. Describe the methods, models, frameworks, or procedures in detail. Include configurations, parameters, and implementation specifics where available.

## Results / Findings / Evidence
What the research discovered. Be specific with numbers, metrics, effect sizes, or qualitative evidence. Include comparisons to baselines or alternatives when applicable.

## Key Contributions / Findings
The 3-5 most important takeaways. What the authors emphasize as their main contribution.

## Limitations
Explicit limitations stated by the authors, and any limitations implied by the methodology.

## What Makes This Work Unique
The novel insight, approach, or contribution that distinguishes this from prior work.

If specific details within any section are not present in the text, write "Not specified." Do not fabricate information.

Use sub-headings and terminology that are natural for the domain. For example:
- Computer Science sections might be called: Training Dataset, Model Architecture, Evaluation Metrics
- Clinical Medicine sections might be called: Study Population, Intervention Protocol, Outcome Measures
- Economics sections might be called: Data Sources, Identification Strategy, Effect Estimates
- History sections might be called: Primary Sources, Analytical Framework, Key Arguments
- Biology sections might be called: Experimental Model, Assay Protocol, Statistical Analysis
- Psychology sections might be called: Participant Sample, Experimental Design, Statistical Tests"""
    try:
        response = llm.invoke(prompt)
        return {
            "analysis": response.content
        }

    except Exception as e:
        return {"analysis": f"Analysis failed: {e}"}