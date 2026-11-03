"""Check that every [paper_id, p.N] citation in an answer points at a passage that was actually retrieved."""
import re

_PAGE = r"(?:p\.|pp?\.\s?|page\s)\s*(\d+)(?:\s*[-–]\s*(\d+))?"
_PART = re.compile(r"([A-Za-z0-9._-]+),\s*" + _PAGE)
_CITE = re.compile(r"\[((?:[A-Za-z0-9._-]+,\s*(?:p\.|pp?\.\s?|page\s)\s*\d+(?:\s*[-–]\s*\d+)?)"
                   r"(?:\s*;\s*[A-Za-z0-9._-]+,\s*(?:p\.|pp?\.\s?|page\s)\s*\d+(?:\s*[-–]\s*\d+)?)*)\](?!\()")


def check_citations(answer: str, docs) -> dict:
    """{"total", "verified", "unverified": [{"paper_id", "page", "reason"}]} for citations in `answer`."""
    available = {(d.metadata.get("paper_id"), int(d.metadata.get("page") or 0)) for d in docs}
    papers = {pid for pid, _ in available}
    total, verified, unverified = 0, 0, []
    for block in _CITE.findall(answer or ""):
        for pid, start, end in _PART.findall(block):
            total += 1
            pages = range(int(start), int(end or start) + 1)
            if any((pid, p) in available for p in pages) or (pid, 0) in available:  # (pid, 0): text-only paper
                verified += 1
            else:
                unverified.append({"paper_id": pid, "page": int(start),
                                   "reason": "page not among retrieved passages" if pid in papers else "paper not retrieved"})
    return {"total": total, "verified": verified, "unverified": unverified}
