"""Cheap lexical relevance scoring used before (and as a fallback for) LLM relevance checks."""
import re

from src.research.models import NormalizedPaper
from src.research.normalize import normalize_title

STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or that the their this to using
use used via with within without towards toward based study studies paper approach analysis new novel
""".split())


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def terms(text: str) -> set[str]:
    return {_stem(t) for t in normalize_title(text).split() if t not in STOPWORDS and len(t) > 1}


def topic_terms(topic: str, concepts: list[str] | None = None) -> tuple[set[str], set[str]]:
    """Return (core terms from the topic, expanded terms from LLM concepts/synonyms)."""
    core = terms(topic)
    expanded = set()
    for c in concepts or []:
        expanded |= terms(c)
    return core, expanded - core


def _acronyms(text: str) -> set[str]:
    return {a.lower() for a in re.findall(r"\b[A-Z]{2,6}s?\b", text or "")}


def lexical_score(paper: NormalizedPaper | dict, core: set[str], expanded: set[str]) -> float:
    """0..1 score from term overlap between the topic and the paper's title/abstract."""
    p = paper.model_dump() if isinstance(paper, NormalizedPaper) else paper
    title_terms = terms(p.get("title", "")) | _acronyms(p.get("title", ""))
    abstract_terms = terms(p.get("abstract", "")) | _acronyms(p.get("abstract", ""))
    if not core:
        return 0.0
    title_core = len(core & title_terms) / len(core)
    abstract_core = len(core & abstract_terms) / len(core)
    expanded_hits = len(expanded & (title_terms | abstract_terms))
    expanded_part = min(1.0, expanded_hits / 4) if expanded else 0.0
    score = 0.45 * title_core + 0.3 * abstract_core + 0.25 * expanded_part
    if p.get("academic_signal"):
        score += 0.05
    else:
        score *= 0.6  # blogs, news and videos are rarely papers
    return round(min(1.0, score), 3)
