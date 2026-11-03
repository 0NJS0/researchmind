"""Normalisation of search results and paper identifiers (DOI, arXiv ID, URL, title)."""
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from src.research.models import NormalizedPaper

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([^\s?#]+?)(?:\.pdf)?(?:[?#]|$)", re.IGNORECASE)
ARXIV_NEW_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
ARXIV_OLD_ID_RE = re.compile(r"^([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20\d{2})\b")

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                   "ref", "referrer", "fbclid", "gclid", "src"}

# Hints only: a result is also treated as academic when it carries a DOI, arXiv ID or PDF link,
# so nothing here assumes which SearXNG engines are enabled.
ACADEMIC_DOMAINS = (
    "arxiv.org", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "dl.acm.org",
    "ieeexplore.ieee.org", "link.springer.com", "springer.com", "sciencedirect.com", "aclanthology.org",
    "openreview.net", "nature.com", "science.org", "wiley.com", "tandfonline.com", "mdpi.com",
    "frontiersin.org", "plos.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "researchgate.net",
    "jstor.org", "sagepub.com", "doi.org", "scholar.google", "europepmc.org", "core.ac.uk",
    "proceedings.neurips.cc", "proceedings.mlr.press", "openaccess.thecvf.com", "usenix.org",
)


def clean_doi(value: str) -> str:
    if not value:
        return ""
    value = unquote(value.strip())
    m = DOI_RE.search(value)
    if not m:
        return ""
    doi = m.group(1).rstrip(".,;)]}").lower()
    # Strip suffixes publishers append to DOI URLs.
    doi = re.sub(r"(/full|/abstract|/pdf|\.pdf)$", "", doi)
    return doi


def clean_arxiv_id(value: str) -> str:
    """Return the version-less arXiv identifier from an ID or URL, or ''."""
    if not value:
        return ""
    value = value.strip()
    m = ARXIV_URL_RE.search(value)
    candidate = m.group(1) if m else value
    candidate = re.sub(r"^arxiv:", "", candidate, flags=re.IGNORECASE)
    for pattern in (ARXIV_NEW_ID_RE, ARXIV_OLD_ID_RE):
        mm = pattern.match(candidate)
        if mm:
            return mm.group(1).lower()
    return ""


def canonical_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", parts.path) or "/"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in TRACKING_PARAMS])
    return urlunsplit(("https", host, path, query, ""))


def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # stray HTML tags from search snippets
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_text(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return " ".join(text.split())


def extract_year(value: str) -> int | None:
    if not value:
        return None
    m = YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


def _authors(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        parts = re.split(r";|\band\b|,(?=\s*[A-Z][a-z]+\s)", value)
    elif isinstance(value, list):
        parts = []
        for a in value:
            if isinstance(a, dict):
                a = a.get("name") or " ".join(filter(None, [a.get("given"), a.get("family")]))
            parts.append(str(a))
    else:
        return []
    return [clean_text(p) for p in parts if clean_text(p)][:30]


def is_academic_url(url: str) -> bool:
    host = urlsplit(url).netloc.lower() if url else ""
    return any(host == d or host.endswith("." + d) or d in host for d in ACADEMIC_DOMAINS)


def normalize_searxng_result(result: dict, query: str = "") -> NormalizedPaper | None:
    """Map a raw SearXNG result (any engine/template) to a NormalizedPaper, or None if unusable."""
    title = clean_text(result.get("title"))
    url = str(result.get("url") or "").strip()
    if not title or not url:
        return None

    pdf_url = str(result.get("pdf_url") or "").strip()
    html_url = str(result.get("html_url") or "").strip()
    doi = clean_doi(str(result.get("doi") or "")) or clean_doi(url) or clean_doi(html_url)
    arxiv_id = clean_arxiv_id(url) or clean_arxiv_id(pdf_url) or clean_arxiv_id(str(result.get("arxiv_id") or ""))
    if arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    if not pdf_url and url.lower().endswith(".pdf"):
        pdf_url = url

    published = result.get("publishedDate") or result.get("published_date") or result.get("pubdate") or ""
    published = str(published) if published else ""
    engines = result.get("engines") or ([result["engine"]] if result.get("engine") else [])
    abstract = clean_text(result.get("abstract") or result.get("content"))

    citation_count = result.get("citation_count") or result.get("citations")
    try:
        citation_count = int(citation_count) if citation_count is not None else None
    except (TypeError, ValueError):
        citation_count = None

    academic = bool(doi or arxiv_id or pdf_url or result.get("category") == "science"
                    or result.get("template") == "paper.html" or is_academic_url(url))

    return NormalizedPaper(
        title=title,
        authors=_authors(result.get("authors")),
        abstract=abstract,
        url=html_url or url,
        pdf_url=pdf_url,
        source=urlsplit(url).netloc.lower().removeprefix("www."),
        engines=[str(e) for e in engines],
        publication_date=published[:10] if published else "",
        year=extract_year(published) or extract_year(str(result.get("year") or "")),
        doi=doi,
        arxiv_id=arxiv_id,
        venue=clean_text(result.get("journal") or result.get("publisher") or ""),
        citation_count=citation_count,
        found_by_query=query,
        academic_signal=academic,
    )
