"""Evolving research-gap state.

The LLM proposes operations; this module validates and applies them. Confidence and contested
status are derived from the evidence here, deterministically, rather than taken from the model.
"""
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.research.models import GapOperation, GapOperations
from src.research.normalize import normalize_title
from src.research.relevance import terms
from src.research.store import ResearchStore

logger = logging.getLogger(__name__)

GENERIC_PATTERNS = re.compile(
    r"^(more|further|additional) (research|studies|work|investigation)( is| are)? (needed|required)"
    r"|need for (more|further) (research|studies)"
    r"|^(future|further) (work|research)$"
    r"|^limitations?$|^research gaps?$|^open (questions?|problems?)$",
    re.IGNORECASE,
)
TITLE_MERGE_RATIO = 0.82
TERM_OVERLAP_MERGE = 0.75
MIN_TITLE_LEN = 12
MIN_EVIDENCE_LEN = 15

WEIGHTS = {"supports": 1.0, "partially_supports": 0.5, "provides_context": 0.0, "contradicts": 0.0}


@dataclass
class ApplyReport:
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    addressed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        for label, items in (("new", self.created), ("updated", self.updated), ("merged", self.merged),
                             ("addressed", self.addressed)):
            if items:
                parts.append(f"{len(items)} {label}")
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected")
        return ", ".join(parts) or "no changes"


def is_generic(title: str, description: str = "") -> bool:
    t = " ".join(title.split()).strip(" .")
    if len(t) < MIN_TITLE_LEN or GENERIC_PATTERNS.search(t):
        return True
    # A description that is itself only a generic call for research adds nothing.
    return bool(description) and bool(GENERIC_PATTERNS.search(description.strip(" ."))) and len(description) < 60


def similar_gap(title: str, gaps: list[dict]) -> dict | None:
    norm = normalize_title(title)
    t_terms = terms(title)
    best, best_score = None, 0.0
    for g in gaps:
        if g["status"] == "merged":
            continue
        ratio = SequenceMatcher(None, norm, normalize_title(g["title"])).ratio()
        g_terms = terms(g["title"])
        overlap = len(t_terms & g_terms) / max(1, min(len(t_terms), len(g_terms)))
        score = max(ratio, overlap if min(len(t_terms), len(g_terms)) >= 3 else 0.0)
        if score > best_score:
            best, best_score = g, score
    if best and (best_score >= TITLE_MERGE_RATIO or best_score >= TERM_OVERLAP_MERGE):
        return best
    return None


def compute_assessment(evidence: list[dict], current_status: str) -> tuple[str, str]:
    """Return (confidence, status) from evidence.

    - independent supporting papers raise confidence (rule 3);
    - support that rests only on future-work statements stays low (rule 2);
    - contradicting papers lower confidence and mark the gap contested (rule 4).
    """
    support: dict[str, float] = {}
    future_only: dict[str, bool] = {}
    contradicting = set()
    for e in evidence:
        pid, rel = e["paper_id"], e["relationship"]
        if rel == "contradicts":
            contradicting.add(pid)
            continue
        weight = WEIGHTS.get(rel, 0.0)
        if e.get("basis") == "future_work":
            weight *= 0.5
        if weight:
            support[pid] = max(support.get(pid, 0.0), weight)
            future_only[pid] = future_only.get(pid, True) and e.get("basis") == "future_work"

    score = sum(support.values())
    if score >= 3:
        confidence = "high"
    elif score >= 1.5:
        confidence = "medium"
    else:
        confidence = "low"
    if support and all(future_only.values()):
        confidence = "low"
    if contradicting:
        confidence = {"high": "medium", "medium": "low", "low": "low"}[confidence]

    status = current_status
    if status in ("resolved", "partially_addressed", "merged"):
        pass  # explicit decisions stand until new evidence re-opens them (handled by callers)
    elif contradicting and support:
        status = "contested"
    elif not contradicting:
        status = "open"
    return confidence, status


class GapEngine:
    def __init__(self, store: ResearchStore, session_id: str):
        self.store = store
        self.session_id = session_id

    # ── context for the LLM ──

    def render_for_prompt(self, focus_text: str = "", limit: int = 15) -> str:
        """Compact list of current gaps, most relevant to `focus_text` first."""
        gaps = self.store.list_gaps(self.session_id)
        if not gaps:
            return ""
        if focus_text and len(gaps) > limit:
            focus = terms(focus_text)
            gaps.sort(key=lambda g: len(focus & terms(g["title"] + " " + g["description"])), reverse=True)
            gaps = gaps[:limit]
        lines = []
        for g in gaps:
            refs = ",".join(p["ref"] for p in g["related_papers"]) or "-"
            lines.append(f"{g['ref']} | {g['gap_type']} | {g['status']} | {g['evidence_count']} items from "
                         f"{refs} | {g['title']}")
        return "\n".join(lines)

    # ── applying operations ──

    def retract_paper(self, paper_id: str) -> None:
        """Undo a paper's earlier contribution before it is analysed again, so re-analysis never
        duplicates evidence. Gaps left without any evidence are removed."""
        for gap_id in self.store.remove_paper_evidence(self.session_id, paper_id):
            if self.store.evidence_for_gap(gap_id):
                self.refresh(gap_id)
            else:
                self.store.delete_gap(gap_id)

    def apply(self, ops: GapOperations, paper: dict) -> ApplyReport:
        report = ApplyReport()
        papers_by_ref = self.store.papers_by_ref(self.session_id)
        for op in ops.operations:
            try:
                handler = getattr(self, f"_op_{op.action}")
                handler(op, paper, papers_by_ref, report)
            except _Rejected as e:
                report.rejected.append(str(e))
                logger.info("Gap operation rejected: %s", e)
        return report

    def _gaps(self) -> list[dict]:
        return self.store.list_gaps(self.session_id)

    def _gap_by_ref(self, ref: str) -> dict:
        ref = (ref or "").strip().upper()
        for g in self.store.list_gaps(self.session_id, include_merged=True):
            if g["ref"] == ref:
                if g["status"] == "merged" and g.get("merged_into"):
                    target = self.store.get_gap(g["merged_into"])
                    if target is None:
                        raise _Rejected(f"gap '{ref}' was merged into a gap that no longer exists")
                    return target
                return g
        raise _Rejected(f"unknown gap id '{ref}'")

    def _valid_evidence(self, op: GapOperation, papers_by_ref: dict[str, dict], paper: dict) -> list[dict]:
        out = []
        for ev in op.evidence:
            ref = ev.paper_id.strip().upper()
            target = papers_by_ref.get(ref)
            if target is None:
                logger.info("Dropping evidence citing unknown paper '%s'", ev.paper_id)
                continue
            if len(ev.evidence) < MIN_EVIDENCE_LEN:
                continue
            out.append({"paper_id": target["id"], "paper_ref": ref, "evidence": ev.evidence[:800],
                        "relationship": ev.relationship, "basis": ev.basis})
        return out

    def _add_evidence(self, gap: dict, evidence: list[dict], paper: dict, reopen: bool = True) -> int:
        added = 0
        for ev in evidence:
            if self.store.add_evidence(gap["id"], ev["paper_id"], ev["evidence"], ev["relationship"], ev["basis"],
                                       source_paper_id=paper["id"]):
                added += 1
                self.store.add_gap_history(
                    gap["id"], self.session_id, f"evidence_{ev['relationship']}",
                    f"{ev['paper_ref']} {ev['relationship'].replace('_', ' ')}: {ev['evidence'][:200]}",
                    ev["paper_id"])
        self.refresh(gap["id"], reopen=reopen and added > 0 and any(
            e["relationship"] in ("supports", "partially_supports") for e in evidence))
        return added

    def refresh(self, gap_id: str, reopen: bool = False) -> None:
        """Recompute confidence/status. `reopen`: new supporting evidence re-opens a resolved gap."""
        gap = self.store.get_gap(gap_id)
        if not gap:
            return
        status = gap["status"]
        if reopen and status == "resolved":
            status = "open"
        confidence, new_status = compute_assessment(self.store.evidence_for_gap(gap_id), status)
        if confidence != gap["confidence"] or new_status != gap["status"]:
            self.store.update_gap(gap_id, confidence=confidence, status=new_status)
            if new_status != gap["status"]:
                self.store.add_gap_history(gap_id, self.session_id, "status",
                                           f"Status changed from {gap['status']} to {new_status}")
            if confidence != gap["confidence"]:
                self.store.add_gap_history(gap_id, self.session_id, "confidence",
                                           f"Confidence changed from {gap['confidence']} to {confidence}")

    def _op_create(self, op, paper, papers_by_ref, report):
        title = " ".join(op.title.split())[:200]
        description = " ".join(op.description.split())[:1500]
        if is_generic(title, description):
            raise _Rejected(f"generic gap '{title}'")
        evidence = self._valid_evidence(op, papers_by_ref, paper)
        if not evidence:
            raise _Rejected(f"gap '{title}' has no valid evidence")
        existing = similar_gap(title, self._gaps())
        if existing:
            added = self._add_evidence(existing, evidence, paper)
            if added:
                self.store.add_gap_history(existing["id"], self.session_id, "merged_observation",
                                           f"Similar gap proposed (\"{title}\") was folded into this gap",
                                           paper["id"])
                report.updated.append(existing["ref"])
            return
        gap = self.store.create_gap(self.session_id, {
            "title": title, "description": description, "gap_type": op.gap_type, "category": op.category})
        self.store.add_gap_history(gap["id"], self.session_id, "created",
                                   f"Identified while analysing {paper['ref']}", paper["id"])
        self._add_evidence(gap, evidence, paper)
        report.created.append(gap["ref"])

    def _op_add_evidence(self, op, paper, papers_by_ref, report):
        gap = self._gap_by_ref(op.gap_id)
        evidence = self._valid_evidence(op, papers_by_ref, paper)
        if not evidence:
            raise _Rejected(f"no valid evidence for {op.gap_id}")
        if self._add_evidence(gap, evidence, paper):
            report.updated.append(gap["ref"])

    def _op_update(self, op, paper, papers_by_ref, report):
        gap = self._gap_by_ref(op.gap_id)
        fields = {}
        if op.title and not is_generic(op.title) and op.title != gap["title"]:
            fields["title"] = " ".join(op.title.split())[:200]
        if op.description and op.description != gap["description"]:
            fields["description"] = " ".join(op.description.split())[:1500]
        if op.gap_type != "other" and op.gap_type != gap["gap_type"]:
            fields["gap_type"] = op.gap_type
        if not fields and not op.evidence:
            return
        if fields:
            self.store.update_gap(gap["id"], **fields)
            self.store.add_gap_history(gap["id"], self.session_id, "refined",
                                       f"Refined after {paper['ref']}: " + ", ".join(fields), paper["id"])
        evidence = self._valid_evidence(op, papers_by_ref, paper)
        if evidence:
            self._add_evidence(gap, evidence, paper)
        report.updated.append(gap["ref"])

    def _op_merge(self, op, paper, papers_by_ref, report):
        refs = list(dict.fromkeys(r.strip().upper() for r in op.gap_ids if r.strip()))
        if op.gap_id and op.gap_id.strip().upper() not in refs:
            refs.insert(0, op.gap_id.strip().upper())
        gaps = []
        for r in refs:
            g = self._gap_by_ref(r)
            if g["id"] not in {x["id"] for x in gaps}:
                gaps.append(g)
        if len(gaps) < 2:
            raise _Rejected("merge needs at least two distinct gaps")
        target, others = gaps[0], gaps[1:]
        fields = {}
        if op.title and not is_generic(op.title):
            fields["title"] = " ".join(op.title.split())[:200]
        if op.description:
            fields["description"] = " ".join(op.description.split())[:1500]
        if fields:
            self.store.update_gap(target["id"], **fields)
        for g in others:
            self.store.move_evidence(g["id"], target["id"])
            self.store.update_gap(g["id"], status="merged", merged_into=target["id"])
            self.store.add_gap_history(g["id"], self.session_id, "merged", f"Merged into {target['ref']}",
                                       paper["id"])
            self.store.add_gap_history(target["id"], self.session_id, "merged",
                                       f"Absorbed {g['ref']} (\"{g['title']}\")", paper["id"])
            report.merged.append(g["ref"])
        self.refresh(target["id"])

    def _op_mark_addressed(self, op, paper, papers_by_ref, report):
        gap = self._gap_by_ref(op.gap_id)
        status = op.status if op.status in ("partially_addressed", "resolved") else "partially_addressed"
        evidence = self._valid_evidence(op, papers_by_ref, paper)
        if not evidence:
            # Claiming a gap is addressed requires saying where.
            raise _Rejected(f"{gap['ref']} marked addressed without evidence")
        for ev in evidence:
            if ev["relationship"] == "supports":
                ev["relationship"] = "contradicts"  # evidence that a gap is addressed argues against it
        self.store.update_gap(gap["id"], status=status)
        self.store.add_gap_history(gap["id"], self.session_id, "addressed",
                                   f"Marked {status.replace('_', ' ')} by {paper['ref']}", paper["id"])
        self._add_evidence(gap, evidence, paper, reopen=False)
        report.addressed.append(gap["ref"])


class _Rejected(Exception):
    pass
