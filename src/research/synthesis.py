"""Validation and rendering of the running research synthesis."""
from src.research.models import SynthesisOut

LIST_FIELDS = ("known_findings", "common_methods", "contradictory_findings",
               "recurring_limitations", "under_researched", "research_questions")
MAX_ITEMS = 8
ESTABLISHED_MIN_PAPERS = 3


def validate_synthesis(out: SynthesisOut, valid_refs: set[str], ref_to_id: dict[str, str]) -> dict:
    """Drop unknown paper IDs and derive finding strength from the number of supporting papers.

    Returned shape (stored as JSON):
      {"overview": str, "<list field>": [{"statement", "paper_refs", "paper_ids", "strength"?}], ...}
    """
    result: dict = {"overview": out.overview.strip()[:2000]}
    for key in LIST_FIELDS:
        items = []
        for item in getattr(out, key)[:MAX_ITEMS]:
            refs = [r.upper() for r in dict.fromkeys(item.paper_ids) if r.upper() in valid_refs]
            entry = {"statement": item.statement.strip()[:500], "paper_refs": refs,
                     "paper_ids": [ref_to_id[r] for r in refs]}
            if key == "known_findings":
                if not refs:
                    continue  # a "known" finding must be attributable to at least one paper
                # Rule 5: established vs emerging is decided by the evidence count, not the model.
                entry["strength"] = "established" if len(refs) >= ESTABLISHED_MIN_PAPERS else "emerging"
            if key == "contradictory_findings" and len(refs) < 2:
                entry["strength"] = "unverified"  # a contradiction should involve two or more papers
            items.append(entry)
        result[key] = items
    return result


def render_synthesis(synthesis: dict | None, limit: int = 3000) -> str:
    if not synthesis:
        return ""
    lines = []
    if synthesis.get("overview"):
        lines.append(f"Overview: {synthesis['overview']}")
    for key in LIST_FIELDS:
        items = synthesis.get(key) or []
        if items:
            lines.append(f"{key.replace('_', ' ').capitalize()}:")
            for it in items:
                refs = ", ".join(it.get("paper_refs") or []) or "no papers"
                lines.append(f"- {it['statement']} [{refs}]")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + " …"


def remove_paper(synthesis: dict | None, ref: str, paper_id: str) -> dict | None:
    """Drop a deleted paper from the synthesis; findings left without support are removed."""
    if not synthesis:
        return synthesis
    out = {**synthesis}
    for key in LIST_FIELDS:
        items = []
        for it in synthesis.get(key) or []:
            refs = [r for r in it.get("paper_refs") or [] if r != ref]
            ids = [i for i in it.get("paper_ids") or [] if i != paper_id]
            if key == "known_findings" and not refs:
                continue
            entry = {**it, "paper_refs": refs, "paper_ids": ids}
            if key == "known_findings":
                entry["strength"] = "established" if len(refs) >= ESTABLISHED_MIN_PAPERS else "emerging"
            if key == "contradictory_findings" and len(refs) < 2:
                entry["strength"] = "unverified"
            items.append(entry)
        out[key] = items
    return out


def _norm(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text or "").split())


def merge_synthesis(old: dict | None, new: dict, valid_refs: set[str], ref_to_id: dict[str, str],
                    similar: float = 0.8) -> dict:
    """Make the synthesis cumulative even when the model drops items: earlier items it left out are carried
    forward (unless they now appear in another list, e.g. moved to contradictions) and dropped paper IDs
    are merged back into matching items."""
    from difflib import SequenceMatcher

    if not old:
        return new
    out = {**new}
    everywhere = [(_norm(it["statement"]), key) for key in LIST_FIELDS for it in new.get(key) or []]
    for key in LIST_FIELDS:
        items = [dict(it) for it in new.get(key) or []]
        for prev in old.get(key) or []:
            p_norm = _norm(prev.get("statement", ""))
            match = None
            for it in items:
                if SequenceMatcher(None, p_norm, _norm(it["statement"])).ratio() >= similar:
                    match = it
                    break
            refs = [r for r in prev.get("paper_refs") or [] if r in valid_refs]
            if match is not None:
                merged = list(dict.fromkeys((match.get("paper_refs") or []) + refs))
                match["paper_refs"] = merged
                match["paper_ids"] = [ref_to_id[r] for r in merged if r in ref_to_id]
                if key == "known_findings":
                    match["strength"] = "established" if len(merged) >= ESTABLISHED_MIN_PAPERS else "emerging"
                continue
            moved = any(SequenceMatcher(None, p_norm, n).ratio() >= similar for n, k in everywhere if k != key)
            if moved or len(items) >= MAX_ITEMS or (key == "known_findings" and not refs):
                continue
            items.append({**prev, "paper_refs": refs, "paper_ids": [ref_to_id[r] for r in refs if r in ref_to_id]})
        out[key] = items
    if not (out.get("overview") or "").strip():
        out["overview"] = old.get("overview", "")
    return out
