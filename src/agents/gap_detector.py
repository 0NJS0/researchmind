import logging

from src.agents.common import complete, domain_text, error_text, input_budget_chars, truncate
from src.agents.comparison import focus_line
from src.agents.prompts import GAP_PROMPT
from src.graph.state import ResearchState
from src.research.prompts import data
from src.reports import report_jobs

logger = logging.getLogger(__name__)


def gap_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {"gaps": ""}
    report_jobs.update(state.get("job_id"), stage="identifying gaps")
    logger.info("Detecting research gaps...")
    focus = focus_line(state)
    budget = input_budget_chars(GAP_PROMPT) - len(focus)
    comparison = state.get("comparison") or "(single paper — no comparison)"
    comparison = truncate(comparison, int(budget * 0.35))
    prompt = GAP_PROMPT.format(focus=focus, domain=domain_text(state), comparison=data(comparison),
                               analysis=data(truncate(analysis, budget - len(comparison))))
    try:
        return {"gaps": complete(prompt)}
    except Exception as e:
        logger.exception("Gap detection failed")
        return {"gaps": "", "errors": [error_text("Gap detection", e)]}
