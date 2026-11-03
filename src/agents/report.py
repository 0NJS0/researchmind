import logging

from src.agents.common import (by_paper, complete, domain_text, error_text, history_block,
                               input_budget_chars, truncate)
from src.agents.prompts import ANSWER_PROMPT, REPORT_PROMPT
from src.graph.state import ResearchState
from src.research.prompts import data
from src.reports import report_jobs

logger = logging.getLogger(__name__)


def failure_report(state: ResearchState) -> str:
    errors = state.get("errors") or ["No analysis could be produced."]
    lines = "\n".join(f"- {e}" for e in dict.fromkeys(errors))
    return f"## No answer could be generated\n\n{lines}\n"


def report_agent(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {"report": failure_report(state)}

    report_mode = state.get("mode") == "report"
    report_jobs.update(state.get("job_id"), stage="writing report")
    logger.info("Generating final %s...", "report" if report_mode else "answer")
    template = REPORT_PROMPT if report_mode else ANSWER_PROMPT
    question = state.get("question", "")
    history = "" if report_mode else history_block(state)
    budget = input_budget_chars(template) - len(question) - len(history) - len(state.get("topic", ""))
    comparison = truncate(state.get("comparison") or "(not applicable)", int(budget * 0.25))
    gaps = truncate(state.get("gaps") or "(none identified)", int(budget * 0.2))
    analysis = truncate(analysis, budget - len(comparison) - len(gaps))
    fields = dict(domain=domain_text(state), analysis=data(analysis), comparison=data(comparison), gaps=data(gaps))
    if report_mode:
        paper_ids = ", ".join(by_paper(state.get("documents", [])).keys())
        prompt = template.format(topic=state.get("topic", "").replace("_", " "), paper_ids=paper_ids, **fields)
    else:
        prompt = template.format(question=data(question), history=history, **fields)
    try:
        return {"report": complete(prompt)}
    except Exception as e:
        logger.exception("Report generation failed")
        return {"report_failed": True,
                "report": failure_report({**state, "errors": [*(state.get("errors") or []),
                                                               error_text("Report generation", e)]}),
                "errors": [error_text("Report generation", e)]}
