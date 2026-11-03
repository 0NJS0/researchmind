import logging

from src.agents.common import complete, domain_text, error_text, input_budget_chars, truncate
from src.agents.prompts import COMPARISON_PROMPT
from src.graph.state import ResearchState
from src.research.prompts import data
from src.reports import report_jobs

logger = logging.getLogger(__name__)


def focus_line(state) -> str:
    if state.get("mode") == "report":
        return f'Research topic: "{state.get("topic", "")}"\n'
    return f"<question>{state.get('question', '')}</question>\nFocus the comparison on what matters for this question.\n"


def comparison_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {"comparison": ""}
    if state.get("paper_count", 0) < 2:
        return {"comparison": ""}  # nothing to compare with a single paper
    report_jobs.update(state.get("job_id"), stage="comparing papers", current="")
    logger.info("Comparing analyses...")
    focus = focus_line(state)
    budget = input_budget_chars(COMPARISON_PROMPT) - len(focus)
    prompt = COMPARISON_PROMPT.format(focus=focus, domain=domain_text(state), analysis=data(truncate(analysis, budget)))
    try:
        return {"comparison": complete(prompt)}
    except Exception as e:
        logger.exception("Comparison failed")
        return {"comparison": "", "errors": [error_text("Comparison", e)]}
