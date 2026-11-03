"""Gap duplication test: is a research gap still open, or has the wider literature already filled it?

Gaps are harvested from the papers in the review (limitations, future work, missing comparisons), so they
can be stale — the authors' "future work" may have been done since. This test searches beyond the review
(SearXNG + OpenAlex, with vocabulary from neighbouring fields), lets a critical LLM judge the hits, labels
the gap and queues any paper that addresses it so the normal pipeline adds its evidence.
"""
import logging
from datetime import datetime, timezone

from src.llm.llm import LLMProvider
from src.llm.structured import invoke_structured
from src.research import prompts
from src.research.dedup import find_duplicate
from src.research.literature import LiteratureSearch, describe, name_papers, public_record
from src.research.models import GapJudgement, SearchQueries
from src.research.ranking import priority
from src.research.searxng import SearchUnavailable
from src.research.store import ResearchStore

logger = logging.getLogger(__name__)

VERIFY_MIN_CONFIDENCE = ("medium", "high")
REVERIFY_AFTER_DAYS = 45
MAX_CANDIDATES = 12
STATUS = {"open": "verified_open", "partially_addressed": "partially_addressed", "addressed": "addressed"}


def gap_text(gap: dict) -> str:
    return f"{gap['ref']}: {gap['title']}\n{gap.get('description') or ''}".strip()


def needs_verification(gap: dict, now: datetime) -> bool:
    if gap["status"] in ("merged", "resolved") or gap["confidence"] not in VERIFY_MIN_CONFIDENCE:
        return False
    checked = gap.get("verified_at")
    if not checked:
        return True
    if isinstance(checked, str):
        checked = datetime.fromisoformat(checked)
    return (now - checked).days >= REVERIFY_AFTER_DAYS


def next_gap_to_verify(store: ResearchStore, session_id: str, now: datetime | None = None) -> dict | None:
    now = now or datetime.now(timezone.utc)
    order = {"high": 0, "medium": 1}
    gaps = [g for g in store.list_gaps(session_id) if needs_verification(g, now)]
    gaps.sort(key=lambda g: (order.get(g["confidence"], 2), -(g.get("paper_count") or 0)))
    return gaps[0] if gaps else None


def verify_gap(store: ResearchStore, session_id: str, gap: dict, topic: str, *,
               provider: LLMProvider | None = None, literature: LiteratureSearch | None = None) -> dict:
    """Run the duplication test on one gap and store the outcome. Returns the stored verification."""
    literature = literature or LiteratureSearch()
    text = gap_text(gap)
    queries = invoke_structured(
        prompts.GAP_VERIFY_QUERIES_PROMPT.format(topic=prompts.data(topic), gap=prompts.data(text), n=3), SearchQueries,
        system=prompts.RESEARCH_SYSTEM_PROMPT, provider=provider, max_tokens=400).queries[:3] or [gap["title"]]
    records, stats = literature.search(queries, per_query=10)
    if not records and stats.get("searxng_failed"):
        # The search failed: "nothing found" would be a false "still open" verdict.
        raise SearchUnavailable("Literature search unavailable; gap check postponed")
    in_review = store.list_papers(session_id, selection=None)
    candidates = literature.closest(records, f"{gap['title']} {gap.get('description', '')}", MAX_CANDIDATES,
                                    exclude=in_review)
    now = datetime.now(timezone.utc)
    if not candidates:
        result = {"verification_status": "verified_open",
                  "verification_note": f"No candidate papers found beyond the review ({len(queries)} searches).",
                  "verification_refs": []}
    else:
        judgement = invoke_structured(
            prompts.GAP_JUDGE_PROMPT.format(topic=prompts.data(topic), gap=prompts.data(text), papers=describe(candidates)),
            GapJudgement,
            system=prompts.RESEARCH_SYSTEM_PROMPT, provider=provider, max_tokens=2000)
        refs = []
        for a in judgement.addressing:
            if 1 <= a.index <= len(candidates):
                refs.append({**public_record(candidates[a.index - 1]), "how": name_papers(a.how, candidates)})
        verdict = judgement.verdict if (refs or judgement.verdict == "open") else "open"
        result = {"verification_status": STATUS[verdict],
                  "verification_note": name_papers(judgement.note, candidates)[:600]
                  or f"Checked {len(candidates)} papers beyond the review.",
                  "verification_refs": refs}
        queue_addressing_papers(store, session_id, gap, [candidates[a.index - 1] for a in judgement.addressing
                                                         if 1 <= a.index <= len(candidates)])
    store.update_gap(gap["id"], verified_at=now, **result)
    label = {"verified_open": "still open", "partially_addressed": "partly addressed elsewhere",
             "addressed": "already addressed elsewhere"}[result["verification_status"]]
    store.add_gap_history(gap["id"], session_id, "verified",
                          f"Duplication test: {label} — {result['verification_note'][:200]}")
    store.add_event(session_id, "gap", f"{gap['ref']} checked against the wider literature: {label}",
                    {"gap_id": gap["id"], "searched": stats})
    return result


def queue_addressing_papers(store: ResearchStore, session_id: str, gap: dict, records: list[dict]) -> int:
    """Papers that fill a gap belong in the review: they will add (contradicting) evidence to it."""
    existing = store.list_papers(session_id, selection=None) + store.tombstones(session_id)
    added = 0
    for r in records:
        if find_duplicate(r, existing):
            continue
        row = store.add_paper(session_id, {
            "title": r.get("title"), "abstract": r.get("abstract") or "", "url": r.get("url") or "",
            "pdf_url": r.get("pdf_url") or "", "doi": r.get("doi") or "", "year": r.get("year"),
            "venue": r.get("venue") or "", "citation_count": r.get("citation_count"),
            "authors": r.get("authors") or [], "publication_type": r.get("publication_type") or "",
            "peer_reviewed": r.get("peer_reviewed"), "source": r.get("source") or "",
            "selection_status": "selected", "relevance_score": 0.85, "relevance_method": "gap_check",
            "relevance_reason": f"Found by the duplication test of {gap['ref']}: may address this gap",
            "processing_status": "pending", "found_by_query": f"gap check {gap['ref']}",
        }, assign_ref=True)
        score, detail = priority(row)
        store.update_paper(row["id"], priority_score=score, priority_detail=detail)
        existing.append(row)
        added += 1
    return added
