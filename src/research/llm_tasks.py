"""LLM operations used by the research agent. Each is a separate, budgeted, structured call."""
import hashlib
import logging

from src.config import LLM_CONTEXT_TOKENS, LLM_MAX_OUTPUT_TOKENS, MAX_CONTENT_CHUNKS
from src.llm.llm import LLMProvider, QuotaExhausted, get_provider
from src.llm.structured import StructuredOutputError, invoke_structured
from src.research import prompts
from src.research.models import (
    SUMMARY_SECTIONS, GapOperations, NormalizedPaper, PaperComparisonOut, PaperSummaryOut,
    RelevanceBatch, SearchQueries, SynthesisOut, TopicAnalysis,
)

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.2   # conservative for academic English
PROMPT_OVERHEAD_TOKENS = 250 + len(prompts.RESEARCH_SYSTEM_PROMPT) // 3

LEVEL_DESCRIPTIONS = {
    "full_text": "full text (may be truncated or condensed into notes)",
    "abstract": "ABSTRACT ONLY — the analysis is abstract-level; do not describe details the abstract does not give",
    "metadata_only": "TITLE/METADATA ONLY — almost nothing is known; keep every section empty unless stated",
}


def _structured(prompt: str, schema, **kwargs):
    return invoke_structured(prompt, schema, system=prompts.RESEARCH_SYSTEM_PROMPT, **kwargs)


def input_budget_chars(template: str, max_output: int = LLM_MAX_OUTPUT_TOKENS) -> int:
    """Characters available for the variable parts of a prompt built from `template`."""
    tokens = LLM_CONTEXT_TOKENS - max_output - PROMPT_OVERHEAD_TOKENS
    return max(1500, int(tokens * CHARS_PER_TOKEN) - len(template))


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rsplit(" ", 1)[0] + " …[truncated]"


# ── Topic & search ──

def field_guidance(analysis: TopicAnalysis | None) -> str:
    """The topic's field profile as prompt guidance ('' when unknown)."""
    if not analysis or not analysis.has_profile():
        return ""
    lines = [f"Field conventions for {prompts.data(analysis.field) or 'this field'} (read the paper the way its experts do):"]
    if analysis.designs:
        lines.append("- Typical designs: " + "; ".join(prompts.data(x) for x in analysis.designs[:5]))
    if analysis.key_quantities:
        lines.append("- Results to capture exactly: " + "; ".join(prompts.data(x) for x in analysis.key_quantities[:5]))
    if analysis.quality_signals:
        lines.append("- Evidence is strong when: " + "; ".join(prompts.data(x) for x in analysis.quality_signals[:5]))
    return "\n".join(lines) + "\n"


def analyze_topic(topic: str, description: str = "", provider: LLMProvider | None = None) -> TopicAnalysis:
    prompt = prompts.TOPIC_ANALYSIS_PROMPT.format(topic=prompts.data(topic), focus_block=prompts.focus_block(description))
    return _structured(prompt, TopicAnalysis, provider=provider, max_tokens=900)


def generate_search_queries(topic: str, analysis: TopicAnalysis | None, n: int,
                            previous: list[str] | None = None, context: str = "",
                            provider: LLMProvider | None = None, focus: str = "") -> list[str]:
    analysis = analysis or TopicAnalysis()
    previous = previous or []
    prompt = prompts.SEARCH_QUERY_GENERATION_PROMPT.format(
        topic=prompts.data(topic),
        focus_block=prompts.focus_block(focus),
        field=analysis.field or "unknown",
        concepts=", ".join(analysis.key_concepts + analysis.synonyms) or "—",
        subtopics=", ".join(analysis.subtopics) or "—",
        context=truncate(context, 2500),
        n=n,
        previous="\n".join(f"- {q}" for q in previous[-25:]) or "- (none)",
    )
    result = _structured(prompt, SearchQueries, provider=provider, max_tokens=400)
    seen = {q.lower() for q in previous}
    out = []
    for q in result.queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


def evaluate_paper_relevance(topic: str, scope: str, papers: list[NormalizedPaper],
                             provider: LLMProvider | None = None, focus: str = "") -> RelevanceBatch:
    items = []
    for i, p in enumerate(papers, 1):
        meta = ", ".join(filter(None, [str(p.year or ""), p.venue or p.source]))
        abstract = prompts.data(truncate(p.abstract, 450)) or "(no abstract)"
        items.append(f"[{i}] {prompts.data(p.title)} ({prompts.data(meta)})\n    {abstract}")
    prompt = prompts.PAPER_RELEVANCE_PROMPT.format(topic=prompts.data(topic), scope=prompts.data(truncate(scope, 400) or topic),
                                                   focus_block=prompts.focus_block(focus), items="\n".join(items))
    return _structured(prompt, RelevanceBatch, provider=provider or get_provider("fast"),
                             max_tokens=120 + 60 * len(papers))


# ── Paper summary ──

def _chunks(text: str, size: int) -> list[str]:
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > size and current:
            chunks.append(current)
            current = ""
        while len(para) > size:
            chunks.append(para[:size])
            para = para[size:]
        current += para + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def _select_chunks(chunks: list[str], limit: int) -> list[str]:
    """Keep the opening (problem, method) and the closing (results, conclusions) of the paper."""
    if len(chunks) <= limit:
        return chunks
    if limit <= 1:
        return chunks[:1]
    head = (limit + 1) // 2
    tail = limit - head
    return chunks[:head] + (chunks[-tail:] if tail else [])


def _cached_notes(provider: LLMProvider, prompt: str) -> str:
    """Notes for one part of a paper, from the LLM cache when this exact part was condensed before (a paper whose
    processing was interrupted, or the same paper in a report) — they cost free-quota requests."""
    from src.llm.structured import _cache
    key = hashlib.sha256(f"notes|{provider.name}|{getattr(provider, 'model', '')}|{prompts.NOTES_SYSTEM_PROMPT}|{prompt}"
                         .encode("utf-8")).hexdigest()
    cached = _cache.get(key) if _cache else None
    if cached:
        return cached
    text = provider.complete(prompt, system=prompts.NOTES_SYSTEM_PROMPT, max_tokens=600).strip()
    if _cache and len(text) > 40:  # don't keep an empty / broken reply
        _cache.set(key, text)
    return text


def prepare_content(title: str, content: str, budget: int, provider: LLMProvider | None = None,
                    guidance: str = "") -> str:
    """Fit paper content into `budget` chars, condensing long papers chunk-by-chunk into notes."""
    if len(content) <= budget:
        return content
    notes_budget = input_budget_chars(prompts.PAPER_NOTES_PROMPT, max_output=600)
    chunks = _select_chunks(_chunks(content, notes_budget), max(1, MAX_CONTENT_CHUNKS))
    if len(chunks) == 1:
        return truncate(content, budget)
    provider = provider or get_provider()
    notes = []
    for i, chunk in enumerate(chunks, 1):
        prompt = prompts.PAPER_NOTES_PROMPT.format(part=i, total=len(chunks), title=prompts.data(title),
                                                   content=prompts.data(chunk), field_guidance=guidance)
        try:
            notes.append(f"[Part {i}]\n" + _cached_notes(provider, prompt))
        except QuotaExhausted:
            raise
        except Exception as e:
            logger.warning("Notes extraction failed for part %d of '%s': %s", i, title, e)
    if not notes:
        return truncate(content, budget)
    return truncate("Condensed notes from the full text:\n" + "\n\n".join(notes), budget)


def paper_type_note(paper: dict) -> str:
    """Tell the summariser when the paper is not a standard empirical study, so sections are used sensibly."""
    kind = (paper.get("publication_type") or "").lower()
    title = (paper.get("title") or "").lower()
    if kind == "review" or any(w in title for w in ("survey", "review", "meta-analysis", "systematic literature")):
        return ("Paper type: review/survey. Use \"methodology\" for how the literature was searched, selected and "
                "synthesised, \"dataset\" for the body of studies reviewed (how many, which period), and "
                "\"key_findings\" for the review's synthesised conclusions and its research agenda.")
    if any(w in title for w in ("position paper", "perspective", "theory of", "a framework for", "towards a")):
        return ("Paper type: conceptual/position paper. \"dataset\" may be empty; use \"methodology\" for the "
                "argument or framework and \"key_findings\" for its main claims.")
    if any(w in title for w in ("benchmark", "dataset", "corpus")):
        return ("Paper type: dataset/benchmark paper. Use \"dataset\" for how the resource was built and its size, "
                "and \"key_findings\" for the baseline results it reports.")
    return ""


def summarize_paper(topic: str, paper: dict, content: str, level: str,
                    provider: LLMProvider | None = None, guidance: str = "") -> PaperSummaryOut:
    budget = input_budget_chars(prompts.PAPER_SUMMARY_PROMPT) - len(paper.get("title", ""))
    if level == "metadata_only":
        content = f"Title: {paper.get('title', '')}\nVenue: {paper.get('venue', '')}\n{content}".strip()
    content = prepare_content(paper.get("title", ""), content, budget - len(guidance), provider, guidance)
    prompt = prompts.PAPER_SUMMARY_PROMPT.format(
        topic=prompts.data(topic), ref=paper["ref"], title=prompts.data(paper.get("title", "")),
        level_description=LEVEL_DESCRIPTIONS.get(level, level), paper_type_note=paper_type_note(paper),
        field_guidance=guidance,
        content=prompts.data(content) or "(no content available)", rules=prompts.GROUNDING_RULES)
    def check(out: PaperSummaryOut) -> str | None:
        if level in ("full_text", "abstract") and not (out.research_problem or out.methodology or out.key_findings):
            return ("the summary is empty — fill research_problem, methodology and key_findings from the paper "
                    "content, using exactly the keys shown")
        return None
    try:
        return _structured(prompt, PaperSummaryOut, provider=provider, check=check)
    except StructuredOutputError:
        # One more chance with a different model before the paper is marked failed.
        fallback = get_provider("fast")
        if provider is not None or fallback.model == get_provider().model:
            raise
        logger.warning("Summary of '%s' failed with the main model; retrying with %s", paper.get("title", "")[:60],
                       fallback.model)
        return _structured(prompt, PaperSummaryOut, provider=fallback, check=check)


# ── Compact renderings used as context for later steps ──

def render_summary(ref: str, title: str, sections: dict, level: str, limit: int = 1800,
                   only: tuple[str, ...] | None = None) -> str:
    lines = [f"{ref}: {title} [analysis level: {level}]"]
    for key in only or SUMMARY_SECTIONS:
        items = sections.get(key) or []
        if items:
            text = " ".join(f"{s['text']}{' (inferred)' if s.get('inferred') else ''}" for s in items)
            lines.append(f"- {key.replace('_', ' ')}: {text}")
    return truncate("\n".join(lines), limit)


# ── Cross-paper steps ──

def compare_papers(topic: str, new_paper: str, existing: list[str],
                   provider: LLMProvider | None = None, guidance: str = "") -> PaperComparisonOut:
    budget = input_budget_chars(prompts.PAPER_COMPARISON_PROMPT, max_output=700)
    new_part = truncate(new_paper, budget // 2)
    per_existing = max(300, (budget - len(new_part)) // max(1, len(existing)))
    prompt = prompts.PAPER_COMPARISON_PROMPT.format(
        topic=prompts.data(topic), new_paper=prompts.data(new_part),
        existing="\n\n".join(prompts.data(truncate(e, per_existing)) for e in existing),
        field_guidance=guidance, rules=prompts.GROUNDING_RULES)
    return _structured(prompt, PaperComparisonOut, provider=provider, max_tokens=700)


def identify_research_gaps(topic: str, ref: str, paper: str, gaps: str, relations: str,
                           provider: LLMProvider | None = None, guidance: str = "") -> GapOperations:
    budget = input_budget_chars(prompts.RESEARCH_GAP_PROMPT)
    paper_part = truncate(paper, int(budget * 0.45))
    rel_part = truncate(relations or "(no earlier papers)", int(budget * 0.2))
    gaps_part = truncate(gaps or "(no gaps yet)", budget - len(paper_part) - len(rel_part))
    prompt = prompts.RESEARCH_GAP_PROMPT.format(
        topic=prompts.data(topic), ref=ref, paper=prompts.data(paper_part), relations=prompts.data(rel_part),
        gaps=prompts.data(gaps_part), field_guidance=guidance, rules=prompts.GROUNDING_RULES)
    return _structured(prompt, GapOperations, provider=provider)


def update_research_synthesis(topic: str, synthesis: str, paper: str, gaps: str,
                              provider: LLMProvider | None = None) -> SynthesisOut:
    budget = input_budget_chars(prompts.RESEARCH_SYNTHESIS_PROMPT)
    paper_part = truncate(paper, int(budget * 0.35))
    gaps_part = truncate(gaps or "(none yet)", int(budget * 0.15))
    synth_part = truncate(synthesis or "(empty — this is the first paper)",
                          budget - len(paper_part) - len(gaps_part))
    prompt = prompts.RESEARCH_SYNTHESIS_PROMPT.format(
        topic=prompts.data(topic), synthesis=prompts.data(synth_part), paper=prompts.data(paper_part),
        gaps=prompts.data(gaps_part), rules=prompts.GROUNDING_RULES)
    return _structured(prompt, SynthesisOut, provider=provider)


def write_report_narrative(topic: str, synthesis: str, gaps: str, papers: str,
                           provider: LLMProvider | None = None) -> str:
    budget = input_budget_chars(prompts.REPORT_NARRATIVE_PROMPT)
    synth_part = truncate(synthesis or "(empty)", int(budget * 0.45))
    gaps_part = truncate(gaps or "(none yet)", int(budget * 0.25))
    papers_part = truncate(papers, budget - len(synth_part) - len(gaps_part))
    prompt = prompts.REPORT_NARRATIVE_PROMPT.format(topic=prompts.data(topic), synthesis=prompts.data(synth_part),
                                                    gaps=prompts.data(gaps_part), papers=prompts.data(papers_part))
    from src.agents.prompts import ANALYST_SYSTEM_PROMPT
    return (provider or get_provider()).complete(prompt, system=ANALYST_SYSTEM_PROMPT).strip()
