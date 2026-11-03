"""Topic Validator: stress-test a research idea before investing in it.

1. Specificity check — is this a field, a neighbourhood or a testable topic? (asks, never supplies, the missing pieces)
2. Nearest-neighbour search — the wider literature (SearXNG + OpenAlex) plus the user's Library topic
3. Duplication test — how close is each rival, is the lane clear, crowded or already taken?
4. Saturation & momentum — matching works per year, citation velocity of the closest rivals
5. Pivot questions — what the student must decide to find a distinct contribution

The AI acts as a critic: the research gap has to be the researcher's own.
"""
import logging
import threading
import uuid
from datetime import date

from src.config import MAX_RESULTS_PER_QUERY
from src.db import connection
from src.llm.llm import LLMProvider, QuotaExhausted
from src.llm.structured import StructuredOutputError, invoke_structured
from src.research import prompts
from src.research.literature import LiteratureSearch, describe, name_papers, public_record
from src.research.models import DuplicationTest, SpecificityCheck
from src.research.store import _encode, get_store

logger = logging.getLogger(__name__)

PROJECT_TYPES = {
    "empirical_quantitative": "Empirical study (quantitative)",
    "empirical_qualitative": "Empirical study (qualitative)",
    "mixed_methods": "Mixed-methods study",
    "experimental": "Experiment / intervention study",
    "systematic_review": "Systematic review",
    "meta_analysis": "Meta-analysis",
    "scoping_review": "Scoping review",
    "computational": "Computational / modelling / new method or tool",
    "theoretical": "Theoretical or conceptual paper",
}
REVIEW_TYPES = {"systematic_review", "meta_analysis", "scoping_review"}
MAX_NEIGHBOURS = 10


def _update(validation_id: str, **fields) -> None:
    cols = ", ".join(f"{k} = %s" for k in fields)
    with connection() as conn:
        conn.execute(f"UPDATE validations SET {cols}, updated_at = now() WHERE id = %s",
                     (*[_encode(k, v) for k, v in fields.items()], validation_id))


def create_validation(idea: str, project_type: str, topic: str = "") -> dict:
    vid = uuid.uuid4().hex
    with connection() as conn:
        conn.execute("INSERT INTO validations (id, idea, project_type, topic, stage) VALUES (%s, %s, %s, %s, %s)",
                     (vid, idea, project_type, topic, "Checking how specific the idea is"))
    return get_validation(vid)


def get_validation(validation_id: str) -> dict | None:
    return get_store()._fetchone("SELECT * FROM validations WHERE id = %s", (validation_id,))


def list_validations(limit: int = 30) -> list[dict]:
    return get_store()._fetchall(
        "SELECT id, idea, project_type, topic, status, stage, result->>'verdict' AS verdict, created_at "
        "FROM validations ORDER BY created_at DESC LIMIT %s", (limit,))


def delete_validation(validation_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM validations WHERE id = %s", (validation_id,))


def _library_neighbours(topic: str, idea: str) -> list[dict]:
    """Papers already in the user's topic that are close to the idea (they are rivals too)."""
    if not topic:
        return []
    store = get_store()
    row = store.get_topic(topic) or {}
    sid = row.get("session_id")
    if not sid:
        return []
    summaries = store.summaries_for_session(sid)
    out = []
    for p in store.list_papers(sid, selection="selected"):
        sections = (summaries.get(p["id"]) or {}).get("sections") or {}
        digest = " ".join(s["text"] for key in ("research_problem", "methodology", "key_findings")
                          for s in sections.get(key) or [])
        out.append({"title": p["title"], "abstract": digest or p.get("abstract") or "", "year": p.get("year"),
                    "venue": p.get("venue") or "", "doi": p.get("doi") or "", "url": p.get("url") or "",
                    "pdf_url": p.get("pdf_url") or "", "publication_type": p.get("publication_type") or "",
                    "peer_reviewed": p.get("peer_reviewed"), "citation_count": p.get("citation_count"),
                    "counts_by_year": {}, "momentum": None, "authors": p.get("authors") or [],
                    "source": "library", "ref": p["ref"]})
    return out


def run_validation(validation_id: str, *, provider: LLMProvider | None = None,
                   literature: LiteratureSearch | None = None, today: date | None = None) -> dict:
    """Run all stages and store the result (called in a worker thread)."""
    v = get_validation(validation_id)
    literature = literature or LiteratureSearch()
    idea, project_type, topic = v["idea"], v["project_type"], v["topic"]
    type_label = PROJECT_TYPES.get(project_type, project_type)
    context = f"Research area it belongs to: {topic.replace('_', ' ')}\n" if topic else ""
    try:
        spec = invoke_structured(
            prompts.VALIDATOR_SPECIFICITY_PROMPT.format(idea=prompts.data(idea), project_type=type_label,
                                                        context=prompts.data(context), n=4),
            SpecificityCheck, system=prompts.RESEARCH_SYSTEM_PROMPT, provider=provider, max_tokens=900)
        result: dict = {"specificity": spec.model_dump(), "project_type_label": type_label}

        _update(validation_id, stage="Searching the literature for the closest studies")
        queries = spec.queries[:4] or [idea[:200]]
        records, stats = literature.search(queries, per_query=min(20, MAX_RESULTS_PER_QUERY))
        records += _library_neighbours(topic, idea)
        focus = f"{spec.restated or idea} {idea}"
        if project_type in REVIEW_TYPES:
            # Existing reviews are the decisive rivals for a review project: make sure they are considered.
            reviews = [r for r in records if r.get("publication_type") == "review" or "review" in (r.get("title") or "").lower()]
            candidates = LiteratureSearch.closest(reviews, focus, MAX_NEIGHBOURS // 2)
            candidates += [r for r in LiteratureSearch.closest(records, focus, MAX_NEIGHBOURS) if r not in candidates]
            candidates = candidates[:MAX_NEIGHBOURS]
        else:
            candidates = LiteratureSearch.closest(records, focus, MAX_NEIGHBOURS)
        year_counts = literature.openalex.year_counts(queries[0])
        result["search"] = {**stats, "queries": queries, "candidates_considered": len(candidates),
                            "records_found": len(records)}
        result["saturation"] = saturation(year_counts, stats.get("openalex_total", 0), today)

        if spec.level == "field":
            # Too broad to run a meaningful duplication test: show the landscape and ask the defining questions.
            result.update(verdict="too_broad", reason="This is a research field, not yet a testable topic. "
                          "Answer the questions below to narrow it, then run the validator again.",
                          neighbours=[{**public_record(r), "closeness": "related", "overlap": "", "difference": "",
                                       "source": r.get("source")} for r in candidates[:8]],
                          pivot_questions=[m.question for m in spec.missing])
        elif not candidates:
            result.update(verdict="clear_lane", reason="No closely related studies were found. Search coverage is "
                          "limited: confirm with a database search (e.g. Scopus/Web of Science) and your supervisor.",
                          neighbours=[], pivot_questions=[m.question for m in spec.missing])
        else:
            _update(validation_id, stage="Running the duplication test on the closest studies")
            def complete_enough(t: DuplicationTest) -> str | None:
                if not t.reason.strip():
                    return "the reason is missing — give the verdict, a reason naming the decisive papers, and the questions"
                if not any(1 <= n.index <= len(candidates) for n in t.neighbours):
                    return "judge every candidate by its number in 'neighbours'"
                return None
            test = invoke_structured(
                prompts.VALIDATOR_DUPLICATION_PROMPT.format(idea=prompts.data(spec.restated or idea), project_type=type_label,
                                                            papers=describe(candidates, abstract_chars=350)),
                DuplicationTest, system=prompts.RESEARCH_SYSTEM_PROMPT, provider=provider, max_tokens=4000,
                check=complete_enough)
            judged = {n.index: n for n in test.neighbours if 1 <= n.index <= len(candidates)}
            order = {"same": 0, "very_close": 1, "related": 2, "unrelated": 3}
            neighbours = []
            for i, r in enumerate(candidates, 1):
                n = judged.get(i)
                closeness = n.closeness if n else "related"
                if closeness == "unrelated":
                    continue
                neighbours.append({**public_record(r), "closeness": closeness, "overlap": n.overlap if n else "",
                                   "difference": n.difference if n else "", "source": r.get("source"),
                                   "ref": r.get("ref", "")})
            neighbours.sort(key=lambda x: (order[x["closeness"]], -(x.get("citation_count") or 0)))
            verdict = test.verdict
            if verdict == "clear_lane" and any(n["closeness"] in ("same", "very_close") for n in neighbours):
                verdict = "crowded"   # the model's own ratings contradict an all-clear
            result.update(verdict=verdict, reason=name_papers(test.reason, candidates), neighbours=neighbours,
                          pivot_questions=[name_papers(q, candidates) for q in test.pivot_questions]
                          or [m.question for m in spec.missing])
        result["rival_momentum"] = rival_momentum(result.get("neighbours") or [])
        _update(validation_id, status="completed", stage="Done", result=result)
        return result
    except QuotaExhausted as e:
        _update(validation_id, status="failed", stage="Stopped", error=f"Daily free LLM quota used up: {e}"[:500])
    except StructuredOutputError as e:
        _update(validation_id, status="failed", stage="Stopped", error=f"The model returned an unusable answer: {e}"[:500])
    except Exception as e:
        logger.exception("Validation %s failed", validation_id)
        _update(validation_id, status="failed", stage="Stopped", error=f"{e.__class__.__name__}: {str(e)[:300]}")
    return {}


def saturation(year_counts: dict[int, int], total: int, today: date | None = None) -> dict:
    """Publication volume and trend for the core query."""
    year = (today or date.today()).year
    years = list(range(year - 10, year + 1))
    series = [{"year": y, "count": year_counts.get(y, 0)} for y in years]
    last3 = sum(year_counts.get(y, 0) for y in (year - 1, year - 2, year - 3))
    prev3 = sum(year_counts.get(y, 0) for y in (year - 4, year - 5, year - 6))
    growth = round((last3 - prev3) / prev3, 2) if prev3 else None
    level = "unknown" if not total else "very dense" if total > 20000 else "dense" if total > 3000 \
        else "moderate" if total > 300 else "sparse"
    return {"total_matching": total, "per_year": series, "growth_last3_vs_prev3": growth, "density": level}


def rival_momentum(neighbours: list[dict]) -> dict:
    close = [n for n in neighbours if n["closeness"] in ("same", "very_close")]
    rising = sum(1 for n in close if (n.get("momentum") or {}).get("trend") == "rising")
    return {"close_rivals": len(close), "rising": rising,
            "newest_close_rival": max((n.get("year") or 0 for n in close), default=None) or None}


_threads: dict[str, threading.Thread] = {}


def start_validation(validation_id: str, **kwargs) -> None:
    t = threading.Thread(target=run_validation, args=(validation_id,), kwargs=kwargs,
                         name=f"validator-{validation_id[:8]}", daemon=True)
    _threads[validation_id] = t
    t.start()


def is_running(validation_id: str) -> bool:
    t = _threads.get(validation_id)
    return bool(t and t.is_alive())


def mark_interrupted() -> None:
    with connection() as conn:
        conn.execute("UPDATE validations SET status = 'failed', stage = 'Stopped', "
                     "error = 'Interrupted by a server restart — run it again.' WHERE status = 'running'")

