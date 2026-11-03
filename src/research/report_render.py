"""Markdown reports rendered from the research state (no LLM calls): a per-paper analysis and the
topic's living report, refreshed every time the autopilot finishes a paper."""
from datetime import datetime, timezone

from src.research.store import ResearchStore

SECTION_TITLES = {
    "research_problem": "Research problem", "motivation": "Motivation", "research_questions": "Research questions",
    "methodology": "Methodology", "dataset": "Data", "key_findings": "Key findings", "contributions": "Contributions",
    "limitations": "Limitations", "future_work": "Future work", "relevance": "Relevance to the topic",
}
CHECK_LABEL = {"unverified": "not yet", "verified_open": "✓ still open", "partially_addressed": "partly addressed",
               "addressed": "addressed elsewhere"}
LEVEL_NOTE = {
    "abstract": "> Based on the abstract only — the full text was not accessible.",
    "metadata_only": "> Based on the title and metadata only.",
}


def _cell(text) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def _meta_line(paper: dict) -> str:
    authors = paper.get("authors") or []
    who = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
    bits = [who, paper.get("venue"), str(paper.get("year") or "")]
    if paper.get("citation_count") is not None:
        bits.append(f"{paper['citation_count']} citations")
    if paper.get("peer_reviewed") is True:
        bits.append("peer reviewed")
    elif paper.get("peer_reviewed") is False:
        bits.append("preprint")
    return " · ".join(b for b in bits if b)


def render_paper_analysis(paper: dict, summary: dict | None, relations: list[dict], gaps: list[dict]) -> str:
    lines = [f"## {paper.get('title', '')}", ""]
    meta = _meta_line(paper)
    if meta:
        lines += [f"*{meta}*", ""]
    links = [f"[{label}]({url})" for label, url in (("Source", paper.get("url")), ("PDF", paper.get("pdf_url")),
                                                     ("DOI", f"https://doi.org/{paper['doi']}" if paper.get("doi") else ""))
             if url]
    if links:
        lines += [" · ".join(links), ""]
    if not summary:
        lines += ["No structured summary is available for this paper.", ""]
        return "\n".join(lines)
    note = LEVEL_NOTE.get(summary.get("analysis_level", ""))
    if note:
        lines += [note, ""]
    for key, title in SECTION_TITLES.items():
        items = (summary.get("sections") or {}).get(key) or []
        if items:
            lines += [f"### {title}"]
            lines += [f"- {s['text']}{' *(inferred)*' if s.get('inferred') else ''}" for s in items]
            lines += [""]
    if relations:
        lines += ["### Related papers in this review"]
        for r in relations:
            text = r.get("contradiction") or r.get("differences") or r.get("similarities") or ""
            lines.append(f"- **{r['related_ref']}** ({r['relationship'].replace('_', ' ')}): {r['related_title']} — {text}")
        lines += [""]
    if gaps:
        lines += ["### Evidence for research gaps"]
        lines += [f"- **{g['ref']}** {g['title']} — {g['relationship'].replace('_', ' ')}: {g['evidence']}" for g in gaps]
        lines += [""]
    return "\n".join(lines)


def _refs(item: dict) -> str:
    refs = item.get("paper_refs") or []
    return f" [{', '.join(refs)}]" if refs else ""


def render_living_report(store: ResearchStore, topic: str, session_id: str, narrative: str = "",
                         recent_limit: int = 8) -> str:
    session = store.get_session(session_id) or {}
    papers = [p for p in store.list_papers(session_id, selection="selected") if p.get("has_summary")]
    gaps = store.list_gaps(session_id)
    synthesis = session.get("synthesis") or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = topic.replace("_", " ")

    lines = [f"# {title}: living literature report", "",
             f"*Updated {now} · {len(papers)} papers analysed · {len(gaps)} open research gaps · "
             f"maintained automatically by the ResearchMind autopilot*", ""]
    if narrative.strip():
        lines += [narrative.strip(), ""]
    if synthesis.get("overview"):
        lines += ["## Current state of knowledge", "", synthesis["overview"], ""]

    def section(heading: str, key: str, strength: bool = False):
        items = synthesis.get(key) or []
        if not items:
            return
        lines.extend([f"## {heading}", ""])
        for it in items:
            tag = f" — **{it['strength'].capitalize()}**" if strength and it.get("strength") else ""
            lines.append(f"- {it['statement']}{_refs(it)}{tag}")
        lines.append("")

    section("Key findings", "known_findings", strength=True)
    section("Contradictions", "contradictory_findings")
    section("Common methods", "common_methods")
    section("Recurring limitations", "recurring_limitations")

    if gaps:
        lines += ["## Research gaps", "",
                  f"*Gaps reflect the {len(papers)} papers analysed here. \"Checked\" gaps were also tested against the wider "
                  "literature (duplication test); unchecked gaps may already be addressed elsewhere.*", "",
                  "| Gap | Type | Confidence | Status | Checked | Evidence from |", "|---|---|---|---|---|---|"]
        for g in gaps:
            refs = ", ".join(p["ref"] for p in g.get("related_papers") or []) or "—"
            lines.append(f"| **{g['ref']}** {_cell(g['title'])} | {_cell(g['gap_type'].replace('_', ' '))} | "
                         f"{g['confidence']} | {g['status'].replace('_', ' ')} | "
                         f"{CHECK_LABEL.get(g.get('verification_status') or 'unverified', '—')} | {refs} |")
        lines.append("")
        for g in gaps:
            if g.get("description"):
                lines.append(f"- **{g['ref']}** — {g['description']}")
            for r in (g.get("verification_refs") or [])[:3]:
                link = f"[{r['title']}]({r['url']})" if r.get("url") else r.get("title", "")
                lines.append(f"  - Addressed by {link} ({r.get('year') or 'n.d.'}): {r.get('how', '')}")
        lines.append("")

    section("Under-researched areas", "under_researched")
    section("Open research questions", "research_questions")

    recent = store.recent_processed(session_id, recent_limit)
    if recent:
        lines += ["## Recently added", ""]
        lines += [f"- **{p['ref']}** {p['title']} ({p.get('year') or 'n.d.'})" for p in recent]
        lines.append("")

    if papers:
        lines += ["## Papers in this review", "", "| Ref | Title | Year | Venue | Citations | Evidence |",
                  "|---|---|---|---|---|---|"]
        for p in papers:
            level = {"full_text": "full text", "abstract": "abstract", "metadata_only": "metadata"}.get(
                p.get("content_status"), p.get("content_status") or "")
            origin = " (uploaded)" if p.get("added_by") == "user" else ""
            lines.append(f"| {p['ref']} | {_cell(p['title'])}{origin} | {_cell(p.get('year'))} | {_cell(p.get('venue'))} | "
                         f"{_cell(p.get('citation_count'))} | {level} |")
        lines.append("")
    return "\n".join(lines)
