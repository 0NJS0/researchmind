"""Prompt evaluation against hand-checked facts from the topic's uploaded papers.

    uv run python -m evals.run                       # everything (~40-70 free LLM requests)
    uv run python -m evals.run --only summary,relevance
    uv run python -m evals.run --model qwen/qwen3.8-27b:free    # compare another free model

Writes evals/results/<timestamp>_<model>.json and .md, and compares with the previous run.
Uses the real configured models with the LLM cache disabled, so results reflect today's models.
"""
import os

os.environ["LLM_CACHE"] = "false"   # measure the model, not the cache

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from evals.scoring import answer_scores, relevance_scores, summary_scores  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class CountingProvider:
    """Wraps a provider to count requests and time them (per task)."""

    def __init__(self, inner):
        self.inner = inner
        self.name, self.model = inner.name, inner.model
        self.calls = 0
        self.seconds = 0.0

    @property
    def last_model(self):
        return getattr(self.inner, "last_model", self.model)

    def complete(self, prompt, **kw):
        self.calls += 1
        t = time.monotonic()
        try:
            return self.inner.complete(prompt, **kw)
        finally:
            self.seconds += time.monotonic() - t


def install_counters(model_override: str | None):
    """Route every get_provider() call through counting wrappers (optionally for another model)."""
    from src.config import LLM_FAST_MODEL, LLM_MODEL
    import src.llm.llm as llm
    counters = {}
    for purpose, model in (("default", model_override or LLM_MODEL), ("fast", model_override or LLM_FAST_MODEL or LLM_MODEL)):
        if model not in counters:
            counters[model] = CountingProvider(llm._build_provider(model, with_fallbacks=model_override is None))
    main = counters[model_override or LLM_MODEL]
    fast = counters[model_override or LLM_FAST_MODEL or LLM_MODEL]
    llm.get_provider = lambda purpose="default": fast if purpose == "fast" else main
    import src.agents.common as common
    import src.llm.structured as structured
    import src.research.llm_tasks as llm_tasks
    for mod in (common, structured, llm_tasks):
        mod.get_provider = llm.get_provider
    return main, fast


def eval_summaries(gold: dict, provider, limit: int | None) -> list[dict]:
    from src.ingest.vectorstore import get_topic_chunks
    from src.research import llm_tasks
    topic = gold["topic"]
    corpus = Path(gold["corpus_dir"]) if gold.get("corpus_dir") else None
    guidance = field_guidance_for(gold)
    out = []
    for i, (pid, spec) in enumerate(gold["papers"].items()):
        if limit is not None and i >= limit:
            break
        if corpus is not None:
            source, title = corpus_text(corpus, pid)
        else:
            source = "\n".join(c.page_content for c in get_topic_chunks(topic, pid))
            title = pid.replace("_", " ")
        paper = {"ref": "P1", "title": title, "publication_type": ""}
        before, t = provider.calls, time.monotonic()
        row = {"paper": pid, "chars": len(source)}
        try:
            out_obj = llm_tasks.summarize_paper(topic.replace("_", " "), paper, source, "full_text", provider,
                                                guidance=guidance)
            row.update(summary_scores(out_obj.model_dump(), source, spec["facts"]), ok=True)
        except Exception as e:  # a failure is a result, not a crash
            row.update(ok=False, error=f"{e.__class__.__name__}: {str(e)[:200]}", empty=True, fact_recall=0.0)
        row.update(requests=provider.calls - before, seconds=round(time.monotonic() - t, 1))
        out.append(row)
        print(f"  summary {pid[:50]:<50} recall={row.get('fact_recall')} unsupported={row.get('unsupported_number_statements')} "
              f"req={row['requests']} {row['seconds']}s", flush=True)
    return out


def corpus_text(corpus: Path, name: str) -> tuple[str, str]:
    """Full text and title of a corpus paper stored as .txt (first line = title) or .pdf."""
    from src.research.content import extract_pdf_text
    txt, pdf = corpus / f"{name}.txt", corpus / f"{name}.pdf"
    if txt.exists():
        text = txt.read_text(encoding="utf-8")
        return text, text.split("\n", 1)[0].strip()
    return extract_pdf_text(pdf.read_bytes(), max_pages=60), name.replace("_", " ")


_GUIDANCE: dict[str, str] = {}


def field_guidance_for(gold: dict) -> str:
    """Field profile the autopilot would use for this topic (one topic-analysis request, as in production)."""
    from src.research import llm_tasks
    topic = gold["topic"]
    if topic not in _GUIDANCE:
        try:
            analysis = llm_tasks.analyze_topic(topic.replace("_", " "), gold.get("relevance", {}).get("focus", ""))
            _GUIDANCE[topic] = llm_tasks.field_guidance(analysis)
            print(f"  field profile: {analysis.field} | key quantities: {analysis.key_quantities[:4]}", flush=True)
        except Exception as e:
            print(f"  field profile failed: {e}", flush=True)
            _GUIDANCE[topic] = ""
    return _GUIDANCE[topic]


def eval_answers(gold: dict, provider, limit: int | None) -> list[dict]:
    from src.agents.citations import check_citations
    from src.graph.workflow import research_graph
    out = []
    for i, q in enumerate(gold["questions"]):
        if limit is not None and i >= limit:
            break
        before, t = provider.calls, time.monotonic()
        try:
            result = research_graph.invoke({"question": q["q"], "history": "", "topic": gold["topic"], "k": 10,
                                            "mode": "query", "documents": [], "errors": []})
            answer = result.get("report", "")
            row = {"question": q["q"], **answer_scores(answer, q["expect"], q["expect_papers"],
                                                       check_citations(answer, result.get("documents", []))),
                   "errors": result.get("errors") or [], "ok": True, "answer": answer}
        except Exception as e:
            row = {"question": q["q"], "ok": False, "correct": False, "error": f"{e.__class__.__name__}: {str(e)[:200]}"}
        row.update(requests=provider.calls - before, seconds=round(time.monotonic() - t, 1))
        out.append(row)
        print(f"  ask     {q['q'][:50]:<50} correct={row.get('correct')} citations={row.get('citations_verified')}/"
              f"{row.get('citations')} req={row['requests']} {row['seconds']}s", flush=True)
    return out


def eval_relevance(gold: dict, provider) -> dict:
    from src.research import llm_tasks
    from src.research.models import NormalizedPaper
    spec = gold["relevance"]
    items = spec["items"]
    preds: dict[int, bool] = {}
    before, t = provider.calls, time.monotonic()
    for start in range(0, len(items), 5):
        batch = items[start:start + 5]
        papers = [NormalizedPaper(title=x["title"], abstract=x.get("abstract", ""), year=x.get("year")) for x in batch]
        try:
            res = llm_tasks.evaluate_paper_relevance(spec["topic"], spec["topic"], papers, provider, focus=spec.get("focus", ""))
        except Exception as e:
            print(f"  relevance batch failed: {e}", flush=True)
            continue
        for a in res.assessments:
            if 1 <= a.index <= len(batch):
                preds[start + a.index - 1] = bool(a.relevant and a.score >= 0.5)
    scores = relevance_scores(items, preds)
    scores.update(requests=provider.calls - before, seconds=round(time.monotonic() - t, 1))
    print(f"  relevance accuracy={scores['accuracy']} precision={scores['precision']} recall={scores['recall']} "
          f"req={scores['requests']}", flush=True)
    return scores


def eval_live_gaps(topic: str) -> dict:
    """Gap quality in the live topic (no LLM calls)."""
    from src.research.gaps import is_generic
    from src.research.store import get_store
    store = get_store()
    row = store.get_topic(topic) or {}
    sid = row.get("session_id")
    if not sid:
        return {"gaps": 0}
    gaps = store.list_gaps(sid)
    return {
        "gaps": len(gaps), "generic": sum(1 for g in gaps if is_generic(g["title"], g.get("description", ""))),
        "multi_paper": sum(1 for g in gaps if (g.get("paper_count") or 0) >= 2),
        "checked": sum(1 for g in gaps if g.get("verification_status") not in (None, "unverified")),
        "confidence": {c: sum(1 for g in gaps if g["confidence"] == c) for c in ("high", "medium", "low")},
    }


def summarise(result: dict) -> dict:
    s = result.get("summaries") or []
    a = result.get("answers") or []
    r = result.get("relevance") or {}
    ok = [x for x in s if x.get("ok")]
    valid = [x["citation_validity"] for x in a if x.get("citation_validity") is not None]
    return {
        "summary_fact_recall": round(sum(x["fact_recall"] for x in ok) / len(ok), 3) if ok else None,
        "summary_empty_or_failed": sum(1 for x in s if x.get("empty") or not x.get("ok")),
        "summary_unsupported_number_statements": sum(x.get("unsupported_number_statements", 0) for x in ok),
        "answers_correct": f"{sum(1 for x in a if x.get('correct'))}/{len(a)}" if a else None,
        "answer_citation_validity": round(sum(valid) / len(valid), 3) if valid else None,
        "answers_uncited": sum(1 for x in a if x.get("ok") and not x.get("citations")),
        "relevance_accuracy": r.get("accuracy"),
        "requests": result["requests"],
    }


def write_report(result: dict, previous: dict | None) -> Path:
    stamp = result["started_at"].replace(":", "").replace("-", "")[:15]
    slug = (result.get("gold", "").replace(".json", "") + "_" + result["model"]).replace("/", "_").replace(":", "_")
    RESULTS.mkdir(exist_ok=True)
    base = RESULTS / f"{stamp}_{slug}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    h, prev = result["headline"], (previous or {}).get("headline", {})

    def delta(key):
        a, b = h.get(key), prev.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d = a - b
            return f" ({'+' if d >= 0 else ''}{round(d, 3)} vs previous)"
        return ""

    lines = [f"# Prompt evaluation — {result['topic']}", "",
             f"*{result['started_at']} · model `{result['model']}` (screening `{result['fast_model']}`) · "
             f"{result['requests']} LLM requests*", "", "## Headline", "",
             "| Metric | Value |", "|---|---|"]
    labels = {"summary_fact_recall": "Summary fact recall (gold facts found)",
              "summary_empty_or_failed": "Summaries empty or failed",
              "summary_unsupported_number_statements": "Summary statements with numbers not in the paper",
              "answers_correct": "Ask answers containing all expected facts",
              "answer_citation_validity": "Ask citations matching a retrieved passage",
              "answers_uncited": "Ask answers with no citations at all",
              "relevance_accuracy": "Relevance screening accuracy (incl. focus exclusions)"}
    for key, label in labels.items():
        if h.get(key) is not None:
            lines.append(f"| {label} | {h[key]}{delta(key)} |")
    if result.get("summaries"):
        lines += ["", "## Summaries", "", "| Paper | Recall | Missing facts | Unsupported numbers | Requests |", "|---|---|---|---|---|"]
        for x in result["summaries"]:
            lines.append(f"| {x['paper'][:55]} | {x.get('fact_recall')} | {', '.join(x.get('missing_facts', [])) or '—'} | "
                         f"{x.get('unsupported_number_statements', '—')} | {x['requests']} |")
        for x in result["summaries"]:
            for text, nums in (x.get("unsupported_examples") or {}).items():
                lines.append(f"- {x['paper'][:40]}: “{text[:160]}” → not in paper: {', '.join(nums)}")
    if result.get("answers"):
        lines += ["", "## Ask answers", "", "| Question | Correct | Expected paper cited | Citations verified |", "|---|---|---|---|"]
        for x in result["answers"]:
            lines.append(f"| {x['question'][:70]} | {x.get('correct')} | {x.get('cites_expected_paper')} | "
                         f"{x.get('citations_verified')}/{x.get('citations')} |")
    if result.get("relevance"):
        r = result["relevance"]
        lines += ["", "## Relevance screening", "", f"Accuracy {r['accuracy']} · precision {r['precision']} · recall {r['recall']}", ""]
        lines += [f"- {m['title']}: {m['error']}{(' (' + m['why'] + ')') if m.get('why') else ''}" for m in r["mistakes"]]
    if result.get("live_gaps"):
        g = result["live_gaps"]
        lines += ["", "## Live gap quality (no LLM calls)", "", f"{g.get('gaps')} gaps · generic {g.get('generic')} · "
                  f"backed by 2+ papers {g.get('multi_paper')} · checked against wider literature {g.get('checked')} · "
                  f"confidence {g.get('confidence')}"]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", default=None,
                    help="gold file(s); default: every file in evals/gold/")
    ap.add_argument("--only", default="summary,ask,relevance,gaps", help="comma list: summary,ask,relevance,gaps")
    ap.add_argument("--model", default=None, help="evaluate another model id (default: the configured ones)")
    ap.add_argument("--limit", type=int, default=None, help="max papers / questions (quick runs)")
    args = ap.parse_args()
    only = set(args.only.split(","))
    gold_files = [Path(g) for g in args.gold] if args.gold else sorted((HERE / "gold").glob("*.json"))

    from src.db import init_schema
    init_schema()
    main_p, fast_p = install_counters(args.model)
    for gold_file in gold_files:
        gold = json.loads(gold_file.read_text())
        start_calls = main_p.calls + (fast_p.calls if fast_p is not main_p else 0)
        result = {"topic": gold["topic"], "gold": gold_file.name, "model": main_p.model, "fast_model": fast_p.model,
                  "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "only": sorted(only)}
        print(f"\nEvaluating {gold['topic']} ({gold_file.name}) with {main_p.model} (screening: {fast_p.model})", flush=True)
        if "summary" in only:
            result["summaries"] = eval_summaries(gold, main_p, args.limit)
        if "ask" in only and gold.get("questions"):
            result["answers"] = eval_answers(gold, main_p, args.limit)
        if "relevance" in only and gold.get("relevance"):
            result["relevance"] = eval_relevance(gold, fast_p)
        if "gaps" in only and not gold.get("corpus_dir"):
            result["live_gaps"] = eval_live_gaps(gold["topic"])
        result["requests"] = main_p.calls + (fast_p.calls if fast_p is not main_p else 0) - start_calls
        result["headline"] = summarise(result)
        previous = None
        for old in sorted(RESULTS.glob("*.json"), reverse=True):
            try:
                data = json.loads(old.read_text())
            except ValueError:
                continue
            if data.get("topic") == gold["topic"]:
                previous = data
                break
        path = write_report(result, previous)
        print(f"Report: {path.with_suffix('.md')}", flush=True)
        print(json.dumps(result["headline"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
