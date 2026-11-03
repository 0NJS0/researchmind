"""Topic-level operations used by the API: delete a paper everywhere, find a paper's PDF, export a topic."""
import csv
import io
import json
import logging
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.ingest.vectorstore import delete_paper as delete_chunks
from src.paths import analysis_path, inside_root, is_safe_name, living_report_path, papers_root
from src.research.gaps import GapEngine
from src.research.report_render import render_living_report
from src.research.store import ResearchStore
from src.research.synthesis import remove_paper, render_synthesis

logger = logging.getLogger(__name__)


class PaperBusy(RuntimeError):
    """The autopilot is processing this paper right now."""


def _session_id(store: ResearchStore, topic: str) -> str | None:
    row = store.get_topic(topic) or {}
    sid = row.get("session_id")
    return sid if sid and store.get_session(sid) else None


def delete_paper_everywhere(store: ResearchStore, topic: str, paper_id: str | None = None,
                            library_id: str | None = None) -> dict:
    """Delete a paper from the database (summary, relations, gap evidence), the vector index and the
    file system (PDF, metadata, analysis), then refresh gaps, synthesis and the living report.

    `paper_id` is the research-session id, `library_id` the Library folder name; either identifies it."""
    sid = _session_id(store, topic)
    paper = store.get_paper(paper_id) if paper_id else None
    if paper and paper["session_id"] != sid:
        raise LookupError("Paper not found in this topic")
    if not paper and library_id and sid:
        paper = store.paper_by_library_id(sid, library_id)
    if paper_id and not paper:
        raise LookupError("Paper not found")
    if paper and store.get_autopilot().get("current_paper_id") == paper["id"]:
        raise PaperBusy("The autopilot is processing this paper right now — try again in a few minutes.")

    lib_id = library_id or (paper.get("library_paper_id") if paper else "") or ""
    removed = {"database": False, "chunks": 0, "files": False}
    if paper:
        GapEngine(store, sid).retract_paper(paper["id"])        # evidence + history; empty gaps deleted
        session = store.get_session(sid) or {}
        if session.get("synthesis") and paper.get("ref"):
            store.update_session(sid, synthesis=remove_paper(session["synthesis"], paper["ref"], paper["id"]))
        store.delete_paper(paper["id"], remember=True)
        removed["database"] = True
    if paper:
        from src.research.publish import incoming_pdf_path, incoming_text_path
        incoming_pdf_path(topic, paper["id"]).unlink(missing_ok=True)   # staged full text waiting for a publish
        incoming_text_path(topic, paper["id"]).unlink(missing_ok=True)
    if lib_id and is_safe_name(lib_id):
        removed["chunks"] = delete_chunks(lib_id, topic)
        folder = inside_root(topic, lib_id)
        if folder is not None and folder.is_dir() and folder != papers_root() / topic:
            shutil.rmtree(folder, ignore_errors=True)
            removed["files"] = True
    if not any(removed.values()):
        raise LookupError("Paper not found")
    if sid:
        write_living_report(store, topic, sid)
    return {"status": "deleted", "library_id": lib_id, **removed}


def write_living_report(store: ResearchStore, topic: str, sid: str) -> None:
    path = living_report_path(topic)
    if not path.parent.exists():
        return
    narrative = (store.get_topic(topic) or {}).get("narrative", "")
    path.write_text(render_living_report(store, topic, sid, narrative), encoding="utf-8")


def pdf_path(topic: str, library_id: str) -> Path | None:
    """The locally stored PDF of a Library paper, if any."""
    if not is_safe_name(library_id):
        return None
    folder = inside_root(topic, library_id)
    if folder is None or not folder.is_dir():
        return None
    pdfs = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".pdf")
    return pdfs[0] if pdfs else None


# ── export ──

PAPER_COLUMNS = ["ref", "title", "authors", "year", "venue", "citation_count", "peer_reviewed", "doi", "url", "pdf_url",
                 "status", "origin", "evidence_level", "research_gaps", "relevance_score", "priority_score",
                 "library_id", "processed_at"]


EXPORT_PREFIX = "researchmind_export_"
EXPORT_MAX_AGE = 3600  # seconds; older export ZIPs are left-overs of aborted downloads


def sweep_exports() -> None:
    """Delete export ZIPs left behind by aborted downloads (they can be as large as all the topic's PDFs)."""
    cutoff = time.time() - EXPORT_MAX_AGE
    for path in Path(tempfile.gettempdir()).glob(f"{EXPORT_PREFIX}*.zip"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def export_topic(store: ResearchStore, topic: str, papers: list[dict], include_pdfs: bool = True) -> Path:
    """Write a ZIP with everything about a topic and return its (temporary) path."""
    sweep_exports()
    tmp = tempfile.NamedTemporaryFile(prefix=f"{EXPORT_PREFIX}{topic}_", suffix=".zip", delete=False)
    tmp.close()
    try:
        _write_export(store, topic, papers, include_pdfs, tmp.name)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return Path(tmp.name)


def _write_export(store: ResearchStore, topic: str, papers: list[dict], include_pdfs: bool, zip_path: str) -> None:
    root = papers_root() / topic
    sid = _session_id(store, topic)
    session = store.get_session(sid) if sid else None
    gaps = store.list_gaps(sid, include_merged=True) if sid else []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = topic.replace("_", " ")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        def add_text(name: str, text: str):
            z.writestr(f"{topic}/{name}", text)

        # Papers
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(PAPER_COLUMNS)
        for p in papers:
            w.writerow([p.get("ref"), p.get("title"), "; ".join(p.get("authors") or []), p.get("year"), p.get("venue"),
                        p.get("citation_count"), {True: "yes", False: "no"}.get(p.get("peer_reviewed"), ""),
                        p.get("doi"), p.get("url"), p.get("pdf_url"), p.get("status"), p.get("origin"),
                        p.get("content_status"), p.get("gap_count"), p.get("relevance_score"), p.get("priority_score"),
                        p.get("library_paper_id"), p.get("processed_at")])
        add_text("papers.csv", buf.getvalue())
        add_text("papers.json", json.dumps(papers, indent=2, ensure_ascii=False, default=str))

        # Reports
        reports = root / "_reports"
        if reports.is_dir():
            for f in sorted(reports.glob("*.md")):
                z.write(f, f"{topic}/reports/{f.name}")
        for p in papers:
            lib = p.get("library_paper_id")
            if lib and is_safe_name(lib):
                a = analysis_path(topic, lib)
                if a.exists():
                    z.write(a, f"{topic}/analyses/{(p.get('ref') + '_') if p.get('ref') else ''}{lib}.md")
                if include_pdfs:
                    pdf = pdf_path(topic, lib)
                    if pdf:
                        z.write(pdf, f"{topic}/pdfs/{(p.get('ref') + '_') if p.get('ref') else ''}{lib}.pdf")

        # Gaps (with evidence and history) and synthesis
        if sid:
            full = []
            lines = [f"# Research gaps — {title}", "", f"*Exported {stamp}*", ""]
            for g in gaps:
                evidence = store.evidence_for_gap(g["id"])
                history = store.gap_history(g["id"])
                full.append({**g, "evidence": evidence, "history": history})
                lines += [f"## {g['ref']}: {g['title']}", "",
                          f"*{g['gap_type'].replace('_', ' ')} · {g['category'].replace('_', ' ')} · confidence {g['confidence']} · "
                          f"status {g['status'].replace('_', ' ')}*", ""]
                if g.get("description"):
                    lines += [g["description"], ""]
                if evidence:
                    lines += ["**Evidence**", ""]
                    lines += [f"- {e['paper_ref']} ({e['relationship'].replace('_', ' ')}, {e['basis'].replace('_', ' ')}): "
                              f"{e['evidence']}" for e in evidence]
                    lines.append("")
            add_text("gaps.md", "\n".join(lines))
            add_text("gaps.json", json.dumps(full, indent=2, ensure_ascii=False, default=str))
            synthesis = (session or {}).get("synthesis")
            if synthesis:
                add_text("synthesis.md", f"# Synthesis — {title}\n\n*Exported {stamp}*\n\n{render_synthesis(synthesis, limit=10**6)}\n")
                add_text("synthesis.json", json.dumps(synthesis, indent=2, ensure_ascii=False))

        processed = sum(1 for p in papers if p.get("status") in ("processed", "partial"))
        add_text("README.md", "\n".join([
            f"# {title} — ResearchMind export", "", f"Exported {stamp}.", "",
            f"- {len(papers)} papers ({processed} analysed) — `papers.csv`, `papers.json`",
            f"- {sum(1 for g in gaps if g['status'] != 'merged')} research gaps — `gaps.md`, `gaps.json`",
            "- Living report and full reports — `reports/`",
            "- Per-paper analyses — `analyses/`",
            "- Paper PDFs (where available) — `pdfs/`" if include_pdfs else "- PDFs not included",
            "- Synthesis of the literature — `synthesis.md`, `synthesis.json`" if sid else "",
        ]) + "\n")



# ── full text arriving later (found by the weekly re-check, or attached by the user) ──

REANALYSIS_PRIORITY_BOOST = 5.0  # a paper that just got its full text is analysed again before new papers


def upgrade_to_full_text(store: ResearchStore, topic: str | None, paper: dict, text: str, pdf_bytes: bytes | None,
                         source: str, pdf_url: str = "") -> None:
    """Replace an abstract-only paper's content with its full text and queue it for a full re-analysis.

    The PDF goes into the Library (re-indexed from its pages) or, if the paper isn't in the Library yet, waits
    in `_incoming/` until it is published. The old abstract-level summary is dropped, so the next processing
    run summarises the full text and re-derives its comparisons, gap evidence and synthesis contribution."""
    from src.research.publish import incoming_pdf_path, incoming_text_path

    if topic:
        # Staged for the next publish, which puts it into the Library (PDF or text, re-indexed). Publishing is
        # retried until it works, so an embedding outage here can't lose the file or repeat the search.
        if pdf_bytes:
            path = incoming_pdf_path(topic, paper["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pdf_bytes)
        else:
            path = incoming_text_path(topic, paper["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text.replace("\x00", ""), encoding="utf-8")
    fields = {"raw_content": text, "content_status": "full_text", "content_source": source,
              "processing_status": "pending", "attempts": 0, "retry_at": None, "error": "",
              "priority_score": (paper.get("priority_score") or 0) + REANALYSIS_PRIORITY_BOOST,
              "fulltext_checked_at": datetime.now(timezone.utc),
              "published_at": None}  # re-published (analysis file, living report) after the re-analysis
    if pdf_url:
        fields["pdf_url"] = pdf_url
    if paper.get("selection_status") != "selected":
        store.select_paper(paper["id"], relevance_reason="Full text attached by the user", relevance_method="manual")
    store.delete_summary(paper["id"])
    store.update_paper(paper["id"], **fields)
    how = "PDF attached by you" if source == "upload" else f"free full text found ({source})"
    store.add_event(paper["session_id"], "success", f"{paper['ref']}: {how}; re-analysing it from the full text",
                    {"paper_id": paper["id"]})
