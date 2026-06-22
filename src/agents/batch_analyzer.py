from collections import defaultdict
from pathlib import Path
import logging

from src.llm.llm import llm
from src.graph.state import ResearchState
from src.config import PAPERS_DIR

logger = logging.getLogger(__name__)


def batch_analyzer_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    topic = state.get("topic", "")
    domain = state.get("domain_classification", "")
    if not docs:
        return {"analysis": "No documents found to analyze."}

    papers: dict[str, list] = defaultdict(list)
    for d in docs:
        pid = d.metadata.get("paper_id", "unknown")
        papers[pid].append(d)

    logger.info("batch_analyzer: analyzing %d papers (%d chunks)", len(papers), len(docs))

    domain_hint = f"\nDomain of these papers: {domain}\n" if domain else "\n"

    paper_analyses = {}
    for pid, chunks in papers.items():
        context = "\n".join(c.page_content for c in chunks)
        prompt = f"""You are analyzing a single research paper.{domain_hint}
Paper ID: {pid}

Paper content:
{context}

Produce a structured analysis using headings that are standard for this field. Use terminology appropriate to the domain.

Regardless of field, the analysis MUST include ALL of the following information:

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
- Computer Science sections: Training Dataset, Model Architecture, Evaluation Metrics
- Clinical Medicine sections: Study Population, Intervention Protocol, Outcome Measures
- Economics sections: Data Sources, Identification Strategy, Effect Estimates
- History sections: Primary Sources, Analytical Framework, Key Arguments
- Biology sections: Experimental Model, Assay Protocol, Statistical Analysis
- Psychology sections: Participant Sample, Experimental Design, Statistical Tests"""
        try:
            resp = llm.invoke(prompt)
            paper_analyses[pid] = resp.content
        except Exception as e:
            paper_analyses[pid] = f"Analysis failed for {pid}: {e}"

        paper_reports_dir = Path(PAPERS_DIR) / topic / pid / "_reports"
        paper_reports_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = paper_reports_dir / f"{pid}_analysis.md"
        with open(analysis_file, "w") as f:
            f.write(paper_analyses[pid])
        logger.info("Saved analysis for '%s' to %s", pid, analysis_file)

    combined = "\n\n".join(f"## {pid}\n{analysis}" for pid, analysis in paper_analyses.items())
    return {"analysis": combined}
