"""Finding a free full-text PDF for a paper when its own page only shows the abstract.

Sources, cheapest and most reliable first:
  1. PDF links on the paper's landing page (meta tags, <link rel=alternate>, "Download PDF" anchors) and
     well-known URL patterns (arXiv, ACL Anthology, OpenReview, PMC, bioRxiv/medRxiv, NeurIPS, CVF, PMLR, MDPI);
  2. every open-access copy OpenAlex knows (repositories, author copies, the published version of a preprint);
  3. arXiv (exact-title search) and Semantic Scholar's open-access PDF link.

Only legitimately free copies are used: sites that block automated access (e.g. SSRN's bot check) are not
worked around; the user can attach the PDF they downloaded in their browser instead.
"""
import logging
import re
import threading
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from src.research.enrich import OpenAlexClient, titles_match
from src.research.normalize import clean_arxiv_id

logger = logging.getLogger(__name__)

API_TIMEOUT = 15
_spacing = {"arxiv": 3.0, "s2": 1.1}      # seconds between requests (arXiv asks for 3 s; S2 allows ~1/s unauthenticated)
_last: dict[str, float] = {}
_lock = threading.Lock()


def _wait(service: str, spacing: dict[str, float]) -> None:
    with _lock:
        delay = _last.get(service, 0.0) + spacing[service] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last[service] = time.monotonic()


def _dedupe(urls: list[str]) -> list[str]:
    seen, out = set(), []
    for u in urls:
        u = (u or "").strip()
        if u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ── 1. landing pages and URL patterns ──

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta" and a.get("name", "").lower() in ("citation_pdf_url", "eprints.document_url", "bepress_citation_pdf_url"):
            self.links.append(a.get("content", ""))
        elif tag == "link" and "pdf" in a.get("type", "").lower():
            self.links.append(a.get("href", ""))
        elif tag == "a" and a.get("href"):
            self._href, self._text = a["href"], []
            if a.get("type", "").lower() == "application/pdf" or "download" in a:
                self.links.append(a["href"])

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            href, text = self._href, " ".join(self._text).lower()
            path = urlsplit(href).path.lower()
            if path.endswith(".pdf") or ("pdf" in text and ("download" in text or "full" in text or text.strip() == "pdf")) \
                    or re.search(r"/(pdf|pdfdirect|epdf|download)(/|$)", path):
                self.links.append(href)
            self._href = None


def landing_pdf_links(html: str, base_url: str) -> list[str]:
    """PDF links a landing page offers (absolute URLs, most specific first)."""
    parser = _LinkParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML: keep whatever was parsed
        pass
    links = [urljoin(base_url, h) for h in parser.links if h and not h.startswith(("javascript:", "mailto:", "#"))]
    return _dedupe(links)[:6]


def pattern_pdf_urls(url: str) -> list[str]:
    """Direct PDF links for well-known scholarly sites, derived from a landing page URL."""
    if not url:
        return []
    parts = urlsplit(url)
    host, path = parts.netloc.lower().removeprefix("www."), parts.path.rstrip("/")
    out: list[str] = []
    if host.endswith("arxiv.org"):
        arxiv_id = clean_arxiv_id(url)
        if arxiv_id:
            out.append(f"https://arxiv.org/pdf/{arxiv_id}")
    elif host == "aclanthology.org" and path and not path.endswith(".pdf"):
        out.append(f"https://aclanthology.org{path}.pdf")
    elif host == "openreview.net":
        forum = parse_qs(parts.query).get("id")
        if forum:
            out.append(f"https://openreview.net/pdf?id={forum[0]}")
    elif host in ("biorxiv.org", "medrxiv.org") and "/content/" in path and not path.endswith(".pdf"):
        out.append(f"https://www.{host}{re.sub(r'\.(abstract|full)$', '', path)}.full.pdf")
    elif "ncbi.nlm.nih.gov" in host or host == "europepmc.org":
        pmc = re.search(r"(PMC\d+)", url, re.I)
        if pmc:
            out.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc.group(1).upper()}&blobtype=pdf")
    elif host == "proceedings.neurips.cc" and path.endswith(".html"):
        out.append(f"https://{host}" + re.sub(r"-Abstract(-\w+)?\.html$", r"-Paper\1.pdf", path).replace("/hash/", "/file/"))
    elif host == "openaccess.thecvf.com" and "/html/" in path and path.endswith(".html"):
        out.append(f"https://{host}" + path.replace("/html/", "/papers/").removesuffix(".html") + ".pdf")
    elif host == "proceedings.mlr.press" and path.endswith(".html"):
        stem = path.removesuffix(".html")
        out.append(f"https://{host}{stem}/{stem.rsplit('/', 1)[-1]}.pdf")
    elif host == "mdpi.com" and re.fullmatch(r"/\d{4}-\d{3,4}[\dX]?/\d+/\d+/\d+", path):
        out.append(f"https://www.mdpi.com{path}/pdf")
    return out


# ── 2 & 3. scholarly indexes ──

class FullTextFinder:
    """Collects candidate PDF URLs for a paper from OpenAlex, arXiv and Semantic Scholar."""

    def __init__(self, openalex: OpenAlexClient | None = None, transport: httpx.BaseTransport | None = None):
        self.openalex = openalex or OpenAlexClient(transport=transport)
        self.transport = transport
        # Polite spacing for the public APIs (not needed for an injected test transport).
        self.spacing = dict(_spacing) if transport is None else {k: 0.0 for k in _spacing}

    def _get(self, service: str, url: str, params: dict) -> httpx.Response | None:
        _wait(service, self.spacing)
        try:
            with httpx.Client(timeout=API_TIMEOUT, transport=self.transport,
                              headers={"User-Agent": "ResearchMind/0.2 (academic research assistant)"}) as client:
                resp = client.get(url, params=params)
        except httpx.HTTPError as e:
            logger.info("%s lookup failed: %s", service, e.__class__.__name__)
            return None
        return resp if resp.status_code == 200 else None

    def openalex_urls(self, paper: dict) -> list[str]:
        try:
            work = self.openalex.find(paper)
        except Exception:
            work = None
        if not work:
            return []
        locations = [work.get("best_oa_location") or {}] + list(work.get("locations") or [])
        pdfs, pages = [], []
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            if loc.get("pdf_url"):
                pdfs.append(loc["pdf_url"])
            landing = loc.get("landing_page_url") or ""
            if loc.get("is_oa") and landing:
                pdfs.extend(pattern_pdf_urls(landing))
                pages.append(landing)       # repository pages usually link the PDF
        ids = work.get("ids") or {}
        arxiv_id = clean_arxiv_id(str(ids.get("arxiv") or "")) or clean_arxiv_id(str((work.get("doi") or "")))
        if arxiv_id:
            pdfs.insert(0, f"https://arxiv.org/pdf/{arxiv_id}")
        return _dedupe(pdfs + pages)

    def arxiv_urls(self, paper: dict) -> list[str]:
        """Exact-title search on arXiv (shares the app-wide arXiv rate limit with the discovery search)."""
        from src.research.scholarly import _arxiv_get, arxiv_query, parse_arxiv_feed  # scholarly imports this module
        title = (paper.get("title") or "").strip()
        if len(title) < 15:
            return []
        xml = _arxiv_get(arxiv_query(re.sub(r"[^\w\s-]", " ", title), field="ti"), max_results=3,
                         sort_newest=False, transport=self.transport)
        return [p.pdf_url for p in parse_arxiv_feed(xml or "") if titles_match(title, p.title)]

    def semantic_scholar_urls(self, paper: dict) -> list[str]:
        fields = {"fields": "title,openAccessPdf,externalIds"}
        doi = paper.get("doi") or ""
        data = None
        if doi:
            resp = self._get("s2", f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}", fields)
            data = resp.json() if resp is not None else None
        if data is None and len(paper.get("title") or "") >= 15:
            resp = self._get("s2", "https://api.semanticscholar.org/graph/v1/paper/search",
                             {**fields, "query": paper["title"][:250], "limit": 3})
            hits = (resp.json().get("data") or []) if resp is not None else []
            data = next((h for h in hits if titles_match(paper["title"], h.get("title") or "")), None)
        if not isinstance(data, dict):
            return []
        out = []
        arxiv_id = ((data.get("externalIds") or {}).get("ArXiv") or "").strip()
        if arxiv_id:
            out.append(f"https://arxiv.org/pdf/{arxiv_id}")
        pdf = (data.get("openAccessPdf") or {}).get("url")
        if pdf:
            out.append(pdf)
        return out

    def candidates(self, paper: dict, exclude: set[str] = frozenset()) -> list[str]:
        """Candidate URLs (PDFs first, then repository landing pages), minus those already tried."""
        urls: list[str] = []
        for source in (self.openalex_urls, self.arxiv_urls, self.semantic_scholar_urls):
            try:
                urls += source(paper)
            except Exception:  # one index failing must not stop the others
                logger.exception("Full-text lookup failed in %s", source.__name__)
        return [u for u in _dedupe(urls) if u not in exclude]
