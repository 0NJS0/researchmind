import logging

from src.agents.common import by_paper, complete, domain_text, error_text, input_budget_chars
from src.agents.prompts import ANALYZE_PAPER_PROMPT, NOT_REPORTED, REPORT_PROMPT
from src.graph.state import ResearchState
from src.llm.llm import QuotaExhausted
from src.reports import report_jobs
from src.research.llm_tasks import prepare_content, truncate
from src.paths import analysis_path
from src.research.prompts import data

logger = logging.getLogger(__name__)


def paper_text(chunks) -> str:
    """Chunks in reading order with page markers, so the model can cite [p.N]."""
    parts, last_page = [], None
    for c in sorted(chunks, key=lambda d: d.metadata.get("chunk_index", 0)):
        page = c.metadata.get("page")
        if page and page != last_page:
            parts.append(f"[p.{page}]")
            last_page = page
        parts.append(c.page_content)
    return "\n".join(parts)


def stored_summaries(topic: str) -> dict[str, str]:
    """Library paper id → the autopilot's structured summary of it, rendered as text.

    Papers the autopilot has already analysed are not re-analysed for a report: that would repeat work already
    stored (and spend free-quota requests). Their analysis files are the autopilot's and are left alone."""
    try:
        from src.research.llm_tasks import render_summary
        from src.research.store import get_store
        store = get_store()
        sid = (store.get_topic(topic) or {}).get("session_id")
        if not sid:
            return {}
        summaries = store.summaries_for_session(sid)
        out = {}
        for p in store.list_papers(sid, selection="selected"):
            s = summaries.get(p["id"])
            if p.get("library_paper_id") and s and s.get("sections"):
                lib = p["library_paper_id"]
                # Labelled with the Library id: that is the id the report lists and cites ("Papers in scope").
                out[lib] = f"## {lib}\n" + render_summary(lib, p["title"], s["sections"], s["analysis_level"], limit=4000)
        return out
    except Exception:  # the report still works without them, just with more LLM requests
        logger.exception("Could not load stored summaries for %s", topic)
        return {}


def batch_analyzer_agent(state: ResearchState) -> dict:
    """Map step of the report: analyse each paper on its own so every paper fits the context window."""
    docs = state.get("documents", [])
    topic = state.get("topic", "")
    job_id = state.get("job_id")
    if not docs:
        return {"analysis": "", "paper_count": 0}

    papers = by_paper(docs)
    logger.info("batch_analyzer: analyzing %d papers (%d chunks)", len(papers), len(docs))
    report_jobs.update(job_id, stage="analyzing papers", total=len(papers), done=0)
    budget = input_budget_chars(ANALYZE_PAPER_PROMPT) - len(topic) - 200

    analyses, errors = {}, []
    stored = stored_summaries(topic)
    out_of_quota = False
    for n, (pid, chunks) in enumerate(papers.items(), 1):
        report_jobs.update(job_id, current=pid)
        if pid in stored:
            analyses[pid] = stored[pid]
            report_jobs.update(job_id, done=n)
            continue
        if out_of_quota:
            report_jobs.update(job_id, done=n)
            continue
        try:
            content = prepare_content(pid, paper_text(chunks), budget)
            prompt = ANALYZE_PAPER_PROMPT.format(topic=topic.replace("_", " "), domain=domain_text(state), paper_id=pid,
                                                 content=data(content), not_reported=NOT_REPORTED)
            analyses[pid] = complete(prompt)
        except QuotaExhausted as e:
            errors.append(error_text("Paper analysis", e))
            out_of_quota = True  # later LLM calls would fail the same way; papers with stored summaries still count
            continue
        except Exception as e:
            logger.exception("Analysis failed for %s", pid)
            errors.append(error_text(f"Analysis of {pid}", e))
            report_jobs.update(job_id, warning=f"Analysis of {pid} failed")
            continue
        finally:
            report_jobs.update(job_id, done=n)
        # Only successful analyses are written to disk.
        path = analysis_path(topic, pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analyses[pid], encoding="utf-8")
        logger.info("Saved analysis for '%s' to %s", pid, path)

    combined, trimmed = fair_join(analyses, int(input_budget_chars(REPORT_PROMPT) * ANALYSIS_SHARE))
    if trimmed:
        report_jobs.update(job_id, warning=f"{len(analyses)} papers: each paper's analysis was shortened to fit the "
                                           "model's context, so every paper is still covered")
    return {"analysis": combined, "paper_count": len(analyses), "errors": errors}


ANALYSIS_SHARE = 0.5  # of the write-up step's input budget (the rest: comparison, gaps, instructions)


def fair_join(parts: dict[str, str], budget: int) -> tuple[str, bool]:
    """Join per-paper texts within `budget` characters, shortening the longest ones first, so every paper stays
    represented (cutting the joined text at the end silently drops the last papers). Returns (text, shortened)."""
    sep = "\n\n"
    total = sum(len(t) for t in parts.values()) + len(sep) * max(0, len(parts) - 1)
    if total <= budget or not parts:
        return sep.join(parts.values()), False
    remaining, left = budget - len(sep) * (len(parts) - 1), len(parts)
    limits: dict[str, int] = {}
    for key, text in sorted(parts.items(), key=lambda kv: len(kv[1])):  # short papers keep all of their text
        share = max(200, remaining // left)
        limits[key] = min(len(text), share)
        remaining -= limits[key]
        left -= 1
    return sep.join(truncate(text, limits[key]) for key, text in parts.items()), True
