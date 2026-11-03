"""Discovery of recent papers from top venues, beyond the general web search.

SearXNG mostly returns arXiv, IEEE and PubMed results ranked by general relevance, so papers that are new (few
citations yet) and published at conferences with their own proceedings sites (CVPR, ICCV, NeurIPS, …) were rarely
found. This module adds three sources:

  * official proceedings listings — CVF (CVPR, ICCV, WACV), ECVA (ECCV), NeurIPS — matched on title against the
    topic terms (the listings are fetched at most once a day);
  * arXiv, newest first — its "comments" field usually says "Accepted to CVPR 2026", which becomes the venue;
  * OpenAlex restricted to the last two years.

It also recognises top venues by name, so ranking can treat a fresh CVPR paper as well-published even before
anyone has cited it.
"""
import html
import logging
import os
import re
import threading
import time
from datetime import date
from urllib.parse import quote_plus, urljoin

import httpx

from src.research.content import make_http_client, _download
from src.research.enrich import OpenAlexClient
from src.research.fulltext import pattern_pdf_urls
from src.research.literature import from_openalex
from src.research.models import NormalizedPaper
from src.research.normalize import clean_arxiv_id, clean_text
from src.research.relevance import terms

logger = logging.getLogger(__name__)

# Canonical short name → patterns found in venue strings (OpenAlex, Crossref, arXiv comments, proceedings).
TOP_VENUES: dict[str, str] = {
    "CVPR": r"\bCVPR\b(?!W)|(?<!Workshops on )Computer Vision and Pattern Recognition(?! Workshops?)",
    "ICCV": r"\bICCV\b(?!W)|IEEE/CVF International Conference on Computer Vision\b(?! Workshops?)|"
            r"^International Conference on Computer Vision$",
    "ECCV": r"\bECCV\b(?!W)|European Conference on Computer Vision\b(?! Workshops?)",
    "WACV": r"\bWACV\b|Winter Conference on Applications of Computer Vision",
    "NeurIPS": r"\bNeurIPS\b|\bNIPS\b|Neural Information Processing Systems",
    "ICML": r"\bICML\b|^International Conference on Machine Learning$|"
            r"Proceedings of the \d+\w* International Conference on Machine Learning$",
    "ICLR": r"\bICLR\b|International Conference on Learning Representations",
    "AAAI": r"\bAAAI\b(?! Conference on Web)|AAAI Conference on Artificial Intelligence",
    "IJCAI": r"\bIJCAI\b",
    "ACL": r"(?<!Findings of the )\bACL\b(?! Findings)|^Annual Meeting of the Association for Computational Linguistics",
    "EMNLP": r"(?<!Findings of )\bEMNLP\b|Empirical Methods in Natural Language Processing",
    "NAACL": r"\bNAACL\b",
    "KDD": r"(?<![A-Z])KDD\b|ACM SIGKDD|Knowledge Discovery and Data Mining",
    "MICCAI": r"\bMICCAI\b|Medical Image Computing and Computer.Assisted Intervention",
    "TPAMI": r"\bTPAMI\b|Transactions on Pattern Analysis and Machine Intelligence",
    "IJCV": r"\bIJCV\b|^International Journal of Computer Vision$",
    "TIP": r"IEEE Transactions on Image Processing",
    "TNNLS": r"\bTNNLS\b|Transactions on Neural Networks and Learning Systems",
    "JMLR": r"\bJMLR\b|Journal of Machine Learning Research",
    "TMLR": r"\bTMLR\b|Transactions on Machine Learning Research",
    "ACM MM": r"\bACM MM\b(?! Asia)|ACM Multimedia(?! Asia)|ACM International Conference on Multimedia(?! in Asia)",
    "Nature": r"^Nature$",
    "Science": r"^Science$",
}
# Workshops, "Findings" tracks and satellite events are not the main venue.
_NOT_MAIN = re.compile(r"\bworkshops?\b|\bfindings\b|\bsymposium\b|\bdoctoral consortium\b|\bchallenge\b", re.I)
# More venues can be added without code changes: TOP_VENUES_EXTRA="ICRA,IROS,MIDL" (matched as whole words).
for _extra in filter(None, (v.strip() for v in os.getenv("TOP_VENUES_EXTRA", "").split(","))):
    TOP_VENUES.setdefault(_extra, rf"\b{re.escape(_extra)}\b")
_TOP_RE = [(name, re.compile(p, re.I)) for name, p in TOP_VENUES.items()]
_ARXIV_SPACING = 5.0   # arXiv asks for at most one request every 3 seconds (its limiter is stricter in practice)
ARXIV_BACKOFF = 30.0   # wait after arXiv signals the rate limit (406 / 429 / 503)
_arxiv_lock = threading.Lock()
_arxiv_last = 0.0


def top_venue(text: str | None) -> str | None:
    """The top venue a venue string (or arXiv comment) names, e.g. "IEEE/CVF CVPR 2025" → "CVPR"."""
    if not text or _NOT_MAIN.search(text):
        return None
    # The venue named first wins: "Accepted by TPAMI; conference version at ECCV 2022" is a TPAMI paper.
    hits = [(m.start(), name) for name, rx in _TOP_RE if (m := rx.search(text))]
    return min(hits)[1] if hits else None


def venue_from_comment(comment: str) -> tuple[str, int | None] | None:
    """arXiv comments such as "Accepted to CVPR 2026 (Highlight)" or "ICML 2025 camera-ready"."""
    if not comment or re.search(r"\bunder review\b|\bsubmitted to\b|\bworkshop\b|\bextension\b|\bextended\b|"
                                r"\bversion of\b|\bour \w+ \d{4} paper\b|\bchallenge\b", comment, re.I):
        return None
    if not re.search(r"\baccepted\b|\bto appear\b|\bcamera.ready\b|\bpublished (?:in|at)\b|\bappears? (?:in|at)\b|"
                     r"\bin proceedings\b|^\s*\(?[A-Z][\w ]*\d{4}\)?\s*(?:\(|$|,|;|\.)", comment, re.I):
        return None   # a venue merely mentioned ("we compare with CVPR 2024 winners") is not where it was published
    name = top_venue(comment)
    if not name:
        return None
    year = re.search(r"\b(20\d{2})\b", comment)
    return name, int(year.group(1)) if year else None


# ── arXiv ──

_arxiv_blocked_until = 0.0


def _arxiv_slot(transport=None) -> None:
    """Wait for this caller's turn: at most one arXiv request per spacing interval, none during a back-off."""
    global _arxiv_last
    if transport:  # tests
        return
    with _arxiv_lock:
        delay = max(_arxiv_last + _ARXIV_SPACING, _arxiv_blocked_until) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _arxiv_last = time.monotonic()


def _arxiv_get(query: str, *, max_results: int, sort_newest: bool, transport=None) -> str | None:
    """Raw Atom feed. The query goes in unencoded-colon form: arXiv answers 406 to `abs%3A…`."""
    _arxiv_slot(transport)
    params = f"search_query={quote_plus(query, safe=':')}&max_results={max_results}"  # spaces as "+", colon literal
    if sort_newest:
        params += "&sortBy=submittedDate&sortOrder=descending"
    try:
        with httpx.Client(timeout=30, transport=transport,
                          headers={"User-Agent": "ResearchMind/0.2 (academic research assistant)"}) as client:
            resp = client.get(f"https://export.arxiv.org/api/query?{params}")
            if resp.status_code in (406, 429, 503) and not transport:
                # arXiv answers 406/429/503 while a client is over its rate limit: every caller backs off, once.
                global _arxiv_blocked_until
                with _arxiv_lock:
                    _arxiv_blocked_until = max(_arxiv_blocked_until, time.monotonic() + ARXIV_BACKOFF)
                _arxiv_slot(transport)
                resp = client.get(f"https://export.arxiv.org/api/query?{params}")
    except httpx.HTTPError as e:
        logger.info("arXiv search failed: %s", e.__class__.__name__)
        return None
    if resp.status_code != 200:
        logger.info("arXiv search returned HTTP %s", resp.status_code)
        return None
    return resp.text


def arxiv_query(text: str, field: str = "abs") -> str:
    """`abs:"few-shot anomaly detection"` — a phrase query on one field."""
    phrase = " ".join(re.sub(r'["()]', " ", text).split())[:200]
    return f'{field}:"{phrase}"'


def parse_arxiv_feed(xml: str) -> list[NormalizedPaper]:
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml or "", re.S):
        def tag(name: str) -> str:
            m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", entry, re.S)
            return html.unescape(" ".join(m.group(1).split())) if m else ""
        abs_url = tag("id")
        arxiv_id = clean_arxiv_id(abs_url)
        if not arxiv_id:
            continue
        published = tag("published")[:10]
        venue_year = venue_from_comment(tag("arxiv:comment"))
        year = int(published[:4]) if published[:4].isdigit() else None
        venue = ""
        if venue_year and not (venue_year[1] and year and venue_year[1] < year):  # not an older conference version
            venue = f"{venue_year[0]} {venue_year[1] or ''}".strip()
            year = venue_year[1] or year
        doi = tag("arxiv:doi")
        out.append(NormalizedPaper(
            title=clean_text(tag("title")), abstract=clean_text(tag("summary")),
            authors=[" ".join(a.split()) for a in re.findall(r"<name>(.*?)</name>", entry)][:20],
            url=f"https://arxiv.org/abs/{arxiv_id}", pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            source="arxiv.org", engines=["arxiv"], publication_date=published, year=year, doi=doi,
            arxiv_id=arxiv_id, venue=venue, academic_signal=True))
    return out


def arxiv_terms_query(text: str, max_terms: int = 5) -> str:
    """`abs:few AND abs:shot AND abs:anomaly AND abs:detection` — every key term, in any order (a topic like
    "Anomaly Detection On Few Shot Learning" is rarely written as that exact phrase)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w not in _ARXIV_STOP and len(w) > 1]
    words = list(dict.fromkeys(words))[:max_terms]
    return " AND ".join(f"abs:{w}" for w in words)


_ARXIV_STOP = {"a", "an", "and", "the", "of", "on", "in", "for", "to", "with", "via", "by", "from", "using",
               "learning", "based", "towards", "toward", "study", "survey", "review", "approach", "method", "methods"}


def arxiv_recent(query: str, max_results: int = 25, transport=None) -> list[NormalizedPaper]:
    q = arxiv_terms_query(query)
    if not q:
        return []
    return parse_arxiv_feed(_arxiv_get(q, max_results=max_results, sort_newest=True, transport=transport) or "")


# ── OpenAlex, last two years ──

def openalex_recent(client: OpenAlexClient, query: str, years: int = 2, per_page: int = 25) -> list[NormalizedPaper]:
    if not client.enabled:
        return []
    since = f"{date.today().year - years + 1}-01-01"
    data = client._get("/works", {"search": query[:250], "per-page": per_page,
                                  "filter": f"from_publication_date:{since}", "select": _OPENALEX_SELECT})
    out = []
    for work in (data or {}).get("results") or []:
        r = from_openalex(work)
        if not r["title"]:
            continue
        out.append(NormalizedPaper(
            title=clean_text(r["title"]), abstract=r["abstract"], authors=r["authors"], url=r["url"],
            pdf_url=r["pdf_url"] or "", source="openalex.org", engines=["openalex"], year=r["year"],
            doi=r["doi"] or "", venue=r["venue"] or "", citation_count=r["citation_count"], academic_signal=True))
    return out


_OPENALEX_SELECT = ("id,doi,title,publication_year,cited_by_count,type,primary_location,best_oa_location,"
                    "open_access,ids,authorships,abstract_inverted_index")


# ── official proceedings listings ──

class _Listing:
    def __init__(self, venue: str, year: int, url: str, pattern: re.Pattern, base: str):
        self.venue, self.year, self.url, self.pattern, self.base = venue, year, url, pattern, base


_CVF_ROW = re.compile(r'<dt class="ptitle"><br>\s*<a href="?([^">]+)"?>\s*([^<]+?)\s*</a>', re.S)
_NEURIPS_ROW = re.compile(r'<a title="paper title" href="([^"]+)">([^<]+)</a>')


def default_listings(today: date | None = None) -> list[_Listing]:
    y = (today or date.today()).year
    out = []
    for year in (y, y - 1):
        out.append(_Listing("CVPR", year, f"https://openaccess.thecvf.com/CVPR{year}?day=all", _CVF_ROW,
                            "https://openaccess.thecvf.com/"))
        out.append(_Listing("WACV", year, f"https://openaccess.thecvf.com/WACV{year}", _CVF_ROW,
                            "https://openaccess.thecvf.com/"))
        if year % 2 == 1:
            out.append(_Listing("ICCV", year, f"https://openaccess.thecvf.com/ICCV{year}?day=all", _CVF_ROW,
                                "https://openaccess.thecvf.com/"))
        out.append(_Listing("NeurIPS", year, f"https://proceedings.neurips.cc/paper_files/paper/{year}", _NEURIPS_ROW,
                            "https://proceedings.neurips.cc"))
    # ECCV (even years) — one page lists every edition; filtered by year from the paper paths.
    out.append(_Listing("ECCV", 0, "https://www.ecva.net/papers.php", _CVF_ROW, "https://www.ecva.net/"))
    return out


class ProceedingsIndex:
    """Titles of recent top-venue proceedings, fetched at most once a day, matched against the topic terms."""

    TTL = 24 * 3600
    RETRY = 30 * 60   # after a network error

    def __init__(self, listings: list[_Listing] | None = None, transport=None):
        self.listings = listings
        self.transport = transport
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def _load(self, listing: _Listing) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(listing.url)
            if hit and now - hit[0] < self.TTL:
                return hit[1]
        rows: list[dict] = []
        ttl = self.TTL
        try:
            with make_http_client(self.transport) as client:
                data, _ = _download(client, listing.url)
            page = data.decode("utf-8", errors="replace")
            this_year = date.today().year
            for href, title in listing.pattern.findall(page):
                year = listing.year
                if listing.venue == "ECCV":
                    m = re.search(r"eccv_(\d{4})", href)
                    year = int(m.group(1)) if m else 0
                    if year < this_year - 2:
                        continue
                rows.append({"title": clean_text(html.unescape(title)), "url": urljoin(listing.base, href), "year": year,
                             "venue": f"{listing.venue} {year}"})
        except Exception as e:  # not published yet (404): try tomorrow; a network error: try again soon
            missing = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404
            ttl = self.TTL if missing else self.RETRY
            logger.info("Proceedings listing %s unavailable: %s", listing.url, e.__class__.__name__)
        with self._lock:
            self._cache[listing.url] = (now - (self.TTL - ttl), rows)
        return rows

    def match(self, core: set[str], expanded: set[str], limit: int = 30) -> list[NormalizedPaper]:
        """Listed papers whose titles contain the topic's core terms (at least two, or all if fewer)."""
        need = min(2, len(core)) or 1
        scored = []
        for listing in self.listings or default_listings():
            for row in self._load(listing):
                t = terms(row["title"])
                core_hits = len(core & t)
                if core_hits < need:
                    continue
                scored.append((core_hits + 0.3 * len(expanded & t), row))
        scored.sort(key=lambda x: (x[0], x[1]["year"]), reverse=True)
        seen, out = set(), []
        for _, row in scored:
            key = row["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            pdf = next(iter(pattern_pdf_urls(row["url"])), "")
            out.append(NormalizedPaper(title=row["title"], url=row["url"], pdf_url=pdf, year=row["year"] or None,
                                       venue=row["venue"], source=row["url"].split("/")[2], engines=["proceedings"],
                                       academic_signal=True))
            if len(out) >= limit:
                break
        return out


_shared: ProceedingsIndex | None = None


def shared_index() -> ProceedingsIndex:
    """One process-wide cache, so the listings (tens of MB) are downloaded at most once a day."""
    global _shared
    if _shared is None:
        _shared = ProceedingsIndex()
    return _shared


def fetch_abstract(paper: NormalizedPaper, transport=None) -> str:
    """The abstract from a proceedings paper page (CVF / NeurIPS / ECVA pages are small and static)."""
    try:
        with make_http_client(transport) as client:
            data, ctype = _download(client, paper.url)
        page = data.decode("utf-8", errors="replace")
    except Exception:
        return ""
    for pattern in (r'<div id="abstract"[^>]*>(.*?)</div>', r"<h4>Abstract</h4>\s*(?:<p>)?(.*?)</p>",
                    r'<meta name="citation_abstract" content="([^"]+)"', r'<p[^>]*id="abstract"[^>]*>(.*?)</p>'):
        m = re.search(pattern, page, re.S | re.I)
        if m:
            return clean_text(html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))))[:3000]
    return ""

