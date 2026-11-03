import logging

from src.agents.common import (by_paper, complete, domain_text, error_text, history_block,
                               input_budget_chars, truncate)
from src.agents.prompts import ANALYZE_QUERY_PROMPT, NOT_REPORTED
from src.graph.state import ResearchState
from src.research.prompts import data

logger = logging.getLogger(__name__)


def format_excerpts(docs, budget: int) -> str:
    per_excerpt = max(400, budget // max(1, len(docs)))
    blocks = []
    for pid, chunks in by_paper(docs).items():
        for d in chunks:
            page = d.metadata.get("page") or 1   # text-only papers have no pages: cite them as page 1
            blocks.append(f'<excerpt paper="{data(pid)}" page="{page}">\n{data(truncate(d.page_content, per_excerpt))}\n</excerpt>')
    return "\n".join(blocks)


def analysis_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"analysis": "", "paper_count": 0}
    logger.info("Analyzing %d chunks...", len(docs))
    question = state.get("question", "")
    history = history_block(state)
    budget = input_budget_chars(ANALYZE_QUERY_PROMPT) - len(question) - len(history)
    prompt = ANALYZE_QUERY_PROMPT.format(question=data(question), history=history, domain=domain_text(state),
                                         excerpts=format_excerpts(docs, budget), not_reported=NOT_REPORTED)
    try:
        return {"analysis": complete(prompt), "paper_count": len(by_paper(docs))}
    except Exception as e:
        logger.exception("Analysis failed")
        return {"analysis": "", "paper_count": 0, "errors": [error_text("Analysis", e)]}
