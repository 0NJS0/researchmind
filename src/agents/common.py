"""Helpers shared by the document-analysis agents."""
from collections import defaultdict

from src.agents.prompts import ANALYST_SYSTEM_PROMPT
from src.llm.llm import get_provider
from src.research.prompts import data
from src.research.llm_tasks import input_budget_chars, truncate  # noqa: F401  (re-exported)


def complete(prompt: str, max_tokens: int | None = None) -> str:
    return get_provider().complete(prompt, system=ANALYST_SYSTEM_PROMPT, max_tokens=max_tokens)


def domain_text(state) -> str:
    """The detected field plus its conventions, so analyses adapt to how that field reports research."""
    d = state.get("domain") or {}
    parts = [d.get("domain"), d.get("subfield"), d.get("research_type"), d.get("methodology")]
    text = " · ".join(data(p) for p in parts if p) or "unknown"
    if d.get("key_quantities"):
        text += "\nResults this field reports (capture exactly): " + "; ".join(data(x) for x in d["key_quantities"])
    if d.get("quality_signals"):
        text += "\nEvidence is strong in this field when: " + "; ".join(data(x) for x in d["quality_signals"])
    return text


def history_block(state) -> str:
    history = (state.get("history") or "").strip()
    return f"<conversation_history>\n{data(history)}\n</conversation_history>\n" if history else ""


def by_paper(docs) -> dict[str, list]:
    papers: dict[str, list] = defaultdict(list)
    for d in docs:
        papers[d.metadata.get("paper_id", "unknown")].append(d)
    return papers


def error_text(stage: str, e: Exception) -> str:
    return f"{stage} failed: {str(e)[:300] or e.__class__.__name__}"
