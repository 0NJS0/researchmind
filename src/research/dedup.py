"""Paper deduplication using DOI, arXiv ID, canonical URL, normalised title and title+author similarity."""
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from src.research.models import NormalizedPaper
from src.research.normalize import canonical_url, clean_arxiv_id, clean_doi, normalize_title

TITLE_EXACT_MIN_LEN = 20          # very short titles ("Introduction") are too ambiguous to match on
TITLE_SIMILAR = 0.93
TITLE_SIMILAR_WITH_AUTHOR = 0.85
# DOI prefixes of preprint servers: a preprint and its published version carry different DOIs.
PREPRINT_DOI_PREFIXES = ("10.48550/", "10.2139/", "10.21203/", "10.1101/", "10.20944/", "10.36227/", "10.31219/",
                         "10.22541/", "10.26434/", "10.31234/", "10.35542/")


def is_preprint_doi(doi: str) -> bool:
    return (doi or "").lower().startswith(PREPRINT_DOI_PREFIXES)


def identity_keys(paper: dict | NormalizedPaper) -> set[str]:
    p = paper.model_dump() if isinstance(paper, NormalizedPaper) else paper
    keys = set()
    doi = clean_doi(p.get("doi", "")) or clean_doi(p.get("url", ""))
    if doi:
        keys.add(f"doi:{doi}")
    arxiv = clean_arxiv_id(p.get("arxiv_id", "")) or clean_arxiv_id(p.get("url", "")) \
        or clean_arxiv_id(p.get("pdf_url", ""))
    if arxiv:
        keys.add(f"arxiv:{arxiv}")
    for field in ("url", "pdf_url"):
        url = canonical_url(p.get(field, ""))
        if url and urlsplit(url).path not in ("", "/"):  # a bare domain identifies nothing
            keys.add(f"url:{url}")
    title = normalize_title(p.get("title", ""))
    if len(title) >= TITLE_EXACT_MIN_LEN:
        keys.add(f"title:{title}")
    return keys


def _surnames(authors) -> set[str]:
    out = set()
    for a in authors or []:
        parts = normalize_title(str(a)).split()
        if parts:
            out.add(parts[-1])
    return out


def _conflicting_ids(a: dict, b: dict) -> bool:
    """Two records with different DOIs (or arXiv IDs) are different papers even if titles match."""
    for field, clean in (("doi", clean_doi), ("arxiv_id", clean_arxiv_id)):
        x, y = clean(a.get(field, "")), clean(b.get(field, ""))
        if field == "doi" and (is_preprint_doi(x) != is_preprint_doi(y)):
            continue  # preprint vs. published version of the same work: decided by title/authors
        if x and y and x != y:
            return True
    return False


def find_duplicate(candidate: dict | NormalizedPaper, existing: list[dict]) -> dict | None:
    """Return the existing paper that `candidate` duplicates, or None."""
    c = candidate.model_dump() if isinstance(candidate, NormalizedPaper) else candidate
    c_keys = identity_keys(c)
    c_title = normalize_title(c.get("title", ""))
    c_authors = _surnames(c.get("authors"))

    for paper in existing:
        if _conflicting_ids(c, paper):
            continue
        if c_keys & identity_keys(paper):
            return paper

    if len(c_title) < TITLE_EXACT_MIN_LEN:
        return None
    for paper in existing:
        if _conflicting_ids(c, paper):
            continue
        p_title = normalize_title(paper.get("title", ""))
        if not p_title:
            continue
        ratio = SequenceMatcher(None, c_title, p_title).ratio()
        if ratio >= TITLE_SIMILAR:
            return paper
        if ratio >= TITLE_SIMILAR_WITH_AUTHOR and c_authors & _surnames(paper.get("authors")):
            return paper
    return None


def merge_metadata(existing: dict, incoming: dict | NormalizedPaper) -> dict:
    """Fields to update on `existing` so it gains metadata only the duplicate carried."""
    inc = incoming.model_dump() if isinstance(incoming, NormalizedPaper) else incoming
    updates = {}
    for field in ("pdf_url", "doi", "arxiv_id", "venue", "publication_date"):
        if not existing.get(field) and inc.get(field):
            updates[field] = inc[field]
    # Prefer the published version's identity over a preprint's.
    if is_preprint_doi(existing.get("doi", "")) and inc.get("doi") and not is_preprint_doi(inc["doi"]):
        updates["doi"] = inc["doi"]
        if inc.get("venue"):
            updates["venue"] = inc["venue"]
    from src.research.scholarly import top_venue  # (scholarly imports this module)
    if inc.get("venue") and top_venue(inc["venue"]) and not top_venue(existing.get("venue") or ""):
        updates["venue"] = inc["venue"]   # "CVPR 2025" from the proceedings beats "arXiv (Cornell University)"
    if not existing.get("year") and inc.get("year"):
        updates["year"] = inc["year"]
    if not existing.get("authors") and inc.get("authors"):
        updates["authors"] = inc["authors"]
    if len(inc.get("abstract") or "") > len(existing.get("abstract") or "") + 50:
        updates["abstract"] = inc["abstract"]
    if inc.get("citation_count") is not None and existing.get("citation_count") is None:
        updates["citation_count"] = inc["citation_count"]
    engines = list(dict.fromkeys((existing.get("engines") or []) + (inc.get("engines") or [])))
    if engines != (existing.get("engines") or []):
        updates["engines"] = engines
    return updates


def dedupe_batch(candidates: list[NormalizedPaper]) -> tuple[list[NormalizedPaper], int]:
    """Collapse duplicates within one batch of search results. Returns (unique, removed_count)."""
    unique: list[NormalizedPaper] = []
    as_dicts: list[dict] = []
    removed = 0
    for cand in candidates:
        dup = find_duplicate(cand, as_dicts)
        if dup is None:
            unique.append(cand)
            as_dicts.append(cand.model_dump())
            continue
        removed += 1
        idx = as_dicts.index(dup)
        updates = merge_metadata(dup, cand)
        if updates:
            merged = unique[idx].model_copy(update=updates)
            unique[idx] = merged
            as_dicts[idx] = merged.model_dump()
    return unique, removed
