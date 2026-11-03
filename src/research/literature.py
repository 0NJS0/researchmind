"""Searching the wider literature (SearXNG + OpenAlex) for a claim, a gap or a research idea.

Used by the gap duplication test and the Topic Validator. Results are plain dicts:
title, abstract, year, venue, doi, url, pdf_url, publication_type, peer_reviewed, citation_count,
counts_by_year {year: citations}, momentum {per_year, trend}, source.
"""
import logging
import re
from datetime import date

from src.research.dedup import find_duplicate, merge_metadata
from src.research.enrich import OpenAlexClient, parse_work
from src.research.relevance import terms
from src.research.searxng import SearchUnavailable, SearXNGClient

logger = logging.getLogger(__name__)


def abstract_from_index(index: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not index:
        return ""
    positions = [(pos, word) for word, poss in index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))[:3000]


def momentum(counts_by_year: dict[int, int], today: date | None = None) -> dict:
    """Citation velocity: citations per year over the last two full years, and whether that is rising."""
    year = (today or date.today()).year
    recent = sum(counts_by_year.get(y, 0) for y in (year - 1, year - 2))
    before = sum(counts_by_year.get(y, 0) for y in (year - 3, year - 4))
    if recent + before < 4:
        trend = "new" if not before else "quiet"
    elif recent > 1.3 * before:
        trend = "rising"
    elif recent < 0.7 * before:
        trend = "cooling"
    else:
        trend = "steady"
    return {"per_year": round(recent / 2, 1), "trend": trend}


def from_openalex(work: dict) -> dict:
    e = parse_work(work)
    counts = {}
    for c in work.get("counts_by_year") or []:
        try:
            counts[int(c["year"])] = int(c["cited_by_count"])
        except (KeyError, TypeError, ValueError):
            continue
    doi = e.doi
    return {
        "title": work.get("title") or "", "abstract": abstract_from_index(work.get("abstract_inverted_index")),
        "year": e.year, "venue": e.venue, "doi": doi, "url": f"https://doi.org/{doi}" if doi else (work.get("id") or ""),
        "pdf_url": e.oa_pdf_url, "publication_type": e.publication_type, "peer_reviewed": e.peer_reviewed,
        "citation_count": e.citation_count, "counts_by_year": counts, "momentum": momentum(counts),
        "authors": [a.get("author", {}).get("display_name") for a in work.get("authorships") or []
                    if a.get("author", {}).get("display_name")][:8],
        "source": "openalex",
    }


class LiteratureSearch:
    def __init__(self, searx: SearXNGClient | None = None, openalex: OpenAlexClient | None = None):
        self.searx = searx or SearXNGClient()
        self.openalex = openalex or OpenAlexClient()

    def search(self, queries: list[str], per_query: int = 15) -> tuple[list[dict], dict]:
        """Run every query on both sources. Returns (unique records, stats)."""
        records: list[dict] = []
        stats = {"queries": len(queries), "searxng_results": 0, "openalex_results": 0, "openalex_total": 0,
                 "searxng_failed": False, "openalex_failed": False}
        for q in queries:
            data = self.openalex.search_works(q, per_page=per_query)
            if data is None and self.openalex.enabled:
                stats["openalex_failed"] = True
            if data:
                stats["openalex_total"] = max(stats["openalex_total"], int((data.get("meta") or {}).get("count") or 0))
                for w in data.get("results") or []:
                    if w.get("title"):
                        records.append(from_openalex(w))
                        stats["openalex_results"] += 1
            try:
                for p in self.searx.search(q, per_query):
                    records.append({**p.model_dump(), "publication_type": "", "peer_reviewed": None,
                                    "counts_by_year": {}, "momentum": None, "source": "searxng"})
                    stats["searxng_results"] += 1
            except SearchUnavailable:
                stats["searxng_failed"] = True
        unique: list[dict] = []
        for r in records:
            dup = find_duplicate(r, unique)
            if dup is None:
                unique.append(r)
            else:
                dup.update(merge_metadata(dup, r))
                if r["source"] == "openalex" and dup.get("source") != "openalex":
                    # OpenAlex carries citations and momentum: prefer its numbers.
                    for key in ("citation_count", "counts_by_year", "momentum", "publication_type", "peer_reviewed"):
                        if r.get(key) not in (None, {}, ""):
                            dup[key] = r[key]
        return unique, stats

    @staticmethod
    def closest(records: list[dict], text: str, limit: int = 15, exclude: list[dict] | None = None) -> list[dict]:
        """Keep the records whose title/abstract share the most vocabulary with `text`."""
        focus = terms(text)
        scored = []
        for r in records:
            if exclude and find_duplicate(r, exclude):
                continue
            overlap = len(focus & terms(f"{r.get('title', '')} {r.get('title', '')} {r.get('abstract', '')}"))
            scored.append((overlap / max(1, len(focus)), r))
        scored.sort(key=lambda x: (x[0], x[1].get("citation_count") or 0), reverse=True)
        return [r for s, r in scored[:limit] if s > 0]


def describe(records: list[dict], abstract_chars: int = 400) -> str:
    """Numbered list of records for a prompt."""
    lines = []
    for i, r in enumerate(records, 1):
        meta = ", ".join(str(x) for x in (r.get("year"), r.get("venue"), r.get("publication_type")) if x)
        cites = f", {r['citation_count']} citations" if r.get("citation_count") is not None else ""
        abstract = (r.get("abstract") or "").replace("\n", " ")[:abstract_chars]
        from src.research.prompts import data
        lines.append(f"[{i}] {data(r.get('title', ''))} ({data(meta)}{cites})\n    {data(abstract) or '(no abstract)'}")
    return "\n".join(lines)


def public_record(r: dict) -> dict:
    """The fields shown to users (no raw abstracts beyond a snippet)."""
    return {k: r.get(k) for k in ("title", "year", "venue", "doi", "url", "pdf_url", "publication_type",
                                  "peer_reviewed", "citation_count", "momentum", "authors")} | {
        "abstract": (r.get("abstract") or "")[:600]}


def name_papers(text: str, candidates: list[dict]) -> str:
    """Replace the prompt's list numbers ("Paper 6", "Papers 2 and 5", "[3]") with short paper titles."""
    def short(n: int) -> str:
        if not 1 <= n <= len(candidates):
            return str(n)
        words = (candidates[n - 1].get("title") or "").split()
        title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        year = candidates[n - 1].get("year")
        return f"“{title}”" + (f" ({year})" if year else "")

    def plural(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        if not all(1 <= n <= len(candidates) for n in nums):
            return m.group(0)
        return ", ".join(short(n) for n in nums[:-1]) + (" and " if len(nums) > 1 else "") + short(nums[-1])

    text = re.sub(r"\b(Papers?|papers?|Candidates?|candidates?)\s+(\d+(?:\s*(?:,\s*and|,|and|&)\s*\d+)*)", plural, text)
    return re.sub(r"\[(\d+)\]", lambda m: short(int(m.group(1))), text)
