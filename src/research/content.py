"""Obtain paper content: abstract → HTML (scholarly meta tags / full text) → PDF → metadata."""
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from src.config import ALLOW_PRIVATE_FETCH, FETCH_TIMEOUT, FETCH_TOTAL_TIMEOUT, MAX_DOWNLOAD_MB
from src.research.normalize import clean_arxiv_id, clean_doi, clean_text, extract_year

logger = logging.getLogger(__name__)

MIN_FULL_TEXT_CHARS = 3000
MAX_FULLTEXT_ATTEMPTS = 10   # downloads per paper (landing page + PDF candidates)
FULLTEXT_BUDGET = 240        # seconds for the whole full-text search of one paper
MIN_ABSTRACT_CHARS = 150
SECTION_MARKERS = ("introduction", "method", "results", "conclusion", "references", "discussion",
                   "related work", "experiments", "evaluation")


class FetchBlocked(RuntimeError):
    pass


class FetchCancelled(RuntimeError):
    pass


@dataclass
class ContentResult:
    text: str = ""
    status: str = "unavailable"          # full_text | abstract | metadata_only | unavailable
    source: str = ""                     # which step produced the text
    metadata: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pdf_bytes: bytes | None = None       # kept when the full text came from a PDF (saved to the Library)
    unreachable: bool = False            # every download failed before reaching a server


# ── SSRF guard ──

def _is_public_ip(value: str) -> bool:
    addr = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    # is_global excludes private, loopback, link-local, shared (100.64/10), reserved and documentation ranges.
    return addr.is_global and not addr.is_multicast


class HostUnresolvable(FetchBlocked):
    """DNS lookup failed: a dead domain, or no internet connection."""


def _resolve_public(host: str) -> str:
    """Resolve `host` once and return an address to connect to; every resolved address must be public."""
    try:
        return host if _is_public_ip(host) else _raise_blocked(host)
    except ValueError:
        pass  # not an IP literal
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise HostUnresolvable(f"Cannot resolve host: {host}") from e
    addrs = [info[4][0] for info in infos]
    if not addrs or not all(_is_public_ip(a) for a in addrs):
        _raise_blocked(host)
    return addrs[0]


def _raise_blocked(host: str):
    raise FetchBlocked(f"Refusing to fetch non-public address: {host}")


def _is_public_host(host: str) -> bool:
    try:
        _resolve_public(host)
        return True
    except HostUnresolvable:
        raise  # not a private address: a dead domain or no internet (the caller tells them apart)
    except FetchBlocked:
        return False


class PinnedTransport(httpx.HTTPTransport):
    """Connects to the IP address that was validated, not to whatever a second DNS lookup returns.

    Closes the DNS-rebinding window between the SSRF check and the connection. The original host name
    is kept for the Host header and TLS (SNI + certificate verification)."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not ALLOW_PRIVATE_FETCH:
            ip = _resolve_public(host)
            if ip != host:
                request.url = request.url.copy_with(host=ip)
                request.extensions = {**request.extensions, "sni_hostname": host}
        return super().handle_request(request)


def check_url_allowed(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchBlocked(f"Unsupported URL: {url[:100]}")
    if not ALLOW_PRIVATE_FETCH and not _is_public_host(parts.hostname):
        raise FetchBlocked(f"Refusing to fetch non-public address: {parts.hostname}")


def _guard(request: httpx.Request) -> None:
    check_url_allowed(str(request.url))


def make_http_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
        max_redirects=5,
        transport=transport or PinnedTransport(),
        event_hooks={"request": [_guard]},  # runs for every redirect hop as well
        headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchMind/0.1; academic research assistant)",
                 "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                 # Only encodings whose decoded chunks stay small: a zstd/brotli "bomb" can expand a tiny body
                 # into gigabytes before the size check below ever sees it.
                 "Accept-Encoding": "gzip, deflate"},
    )


def _download(client: httpx.Client, url: str, cancel_event=None) -> tuple[bytes, str]:
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    deadline = time.monotonic() + FETCH_TOTAL_TIMEOUT
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").lower()
        encoding = resp.headers.get("content-encoding", "identity").lower().strip()
        if encoding not in ("", "identity", "gzip", "x-gzip", "deflate"):
            raise ValueError(f"unsupported content encoding '{encoding[:20]}'")
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise ValueError("document too large")
        chunks, total = [], 0
        for chunk in resp.iter_bytes():
            if cancel_event is not None and cancel_event.is_set():
                raise FetchCancelled()
            if time.monotonic() > deadline:  # a slow-drip server must not hold the worker indefinitely
                raise TimeoutError("download took too long")
            total += len(chunk)
            if total > limit:
                raise ValueError("document too large")
            chunks.append(chunk)
    return b"".join(chunks), ctype


# ── HTML parsing ──

class _PageParser(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form", "button", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            if name and a.get("content"):
                self.meta.setdefault(name, []).append(a["content"])
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "section", "tr"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.text_parts.append(data)

    def first(self, *names: str) -> str:
        for n in names:
            if self.meta.get(n):
                return self.meta[n][0].strip()
        return ""


def parse_html(html: str, base_url: str = "") -> tuple[dict, str]:
    """Return (scholarly metadata from meta tags, visible page text)."""
    parser = _PageParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML: keep whatever was parsed
        pass
    scholarly = bool(parser.first("citation_title", "dc.title"))
    # Generic description tags are only trusted as an abstract on pages that declare scholarly metadata.
    abstract_tags = ("citation_abstract", "dc.description", "description", "og:description") if scholarly \
        else ("citation_abstract",)
    meta = {
        "title": clean_text(parser.first("citation_title", "dc.title")),
        "abstract": clean_text(parser.first(*abstract_tags)),
        "authors": [clean_text(a) for a in parser.meta.get("citation_author", []) if clean_text(a)],
        "doi": clean_doi(parser.first("citation_doi", "dc.identifier", "prism.doi")),
        "pdf_url": parser.first("citation_pdf_url"),
        "publication_date": parser.first("citation_publication_date", "citation_date",
                                         "citation_online_date", "dc.date"),
        "venue": clean_text(parser.first("citation_journal_title", "citation_conference_title",
                                         "citation_publisher")),
        "arxiv_id": clean_arxiv_id(parser.first("citation_arxiv_id")),
    }
    if meta["pdf_url"] and base_url:
        meta["pdf_url"] = urljoin(base_url, meta["pdf_url"])
    text = "".join(parser.text_parts).replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return {k: v for k, v in meta.items() if v}, text


def looks_like_full_text(text: str) -> bool:
    if len(text) < MIN_FULL_TEXT_CHARS:
        return False
    lower = text.lower()
    return sum(1 for m in SECTION_MARKERS if m in lower) >= 3


# ── PDF ──

def extract_pdf_text(data: bytes, max_pages: int = 40) -> str:
    import fitz  # PyMuPDF

    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = [doc[i].get_text() for i in range(min(len(doc), max_pages))]
    text = "\n".join(pages).replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)           # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Cut the reference list: it wastes context and invites citation confusion.
    m = re.search(r"\n\s*(references|bibliography)\s*\n", text, re.IGNORECASE)
    if m and m.start() > len(text) * 0.5:
        text = text[: m.start()]
    return text.strip()


def fetch_pdf(url: str, transport: httpx.BaseTransport | None = None) -> bytes | None:
    """Download a PDF (with the same SSRF guard, size and time limits). None if it isn't a PDF."""
    if not url:
        return None
    try:
        with make_http_client(transport) as client:
            data, ctype = _download(client, url)
    except Exception as e:
        logger.info("PDF download failed for %s: %s", url, e.__class__.__name__)
        return None
    return data if data[:4] == b"%PDF" else None


# ── Orchestration ──

def _internet_reachable() -> bool:
    from src.llm.llm import internet_reachable
    return internet_reachable()


_CONNECT_ERRORS = (httpx.NetworkError, httpx.TimeoutException, HostUnresolvable)


def fetch_content(paper: dict, transport: httpx.BaseTransport | None = None, cancel_event=None,
                  finder=None) -> ContentResult:
    """Best-effort content retrieval. Never raises for network/content problems.

    Tries the landing page, then every PDF candidate: the known PDF link, links the landing page offers,
    well-known URL patterns and (with `finder`) other free copies from OpenAlex, arXiv and Semantic Scholar.
    Returns early (with whatever was found) if `cancel_event` is set during a download."""
    from src.research.fulltext import landing_pdf_links, pattern_pdf_urls

    result = ContentResult()
    abstract = (paper.get("abstract") or "").strip()
    url = paper.get("url") or ""
    pdf_url = paper.get("pdf_url") or ""
    arxiv_id = paper.get("arxiv_id") or ""
    if arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    full_text = ""
    attempts, connect_failures = 0, 0
    tried: set[str] = set()
    budget_ends = time.monotonic() + FULLTEXT_BUDGET
    queue: list[str] = []

    def failed(e: BaseException, what: str, target: str) -> None:
        nonlocal connect_failures
        connect_failures += isinstance(e, _CONNECT_ERRORS)
        result.notes.append(str(e) if isinstance(e, FetchBlocked) else f"{what} unavailable ({e.__class__.__name__})")
        logger.info("%s fetch failed for %s: %s", what, target, e)

    with make_http_client(transport) as client:
        def take_pdf(data: bytes, source_url: str) -> bool:
            nonlocal full_text
            text = extract_pdf_text(data)
            if len(text) >= MIN_FULL_TEXT_CHARS:
                full_text, result.source, result.pdf_bytes = text, "pdf", data
                result.metadata["pdf_url"] = source_url
                return True
            result.notes.append("PDF contained too little extractable text")
            return False

        # 1. Landing page: scholarly meta tags (abstract, PDF link, DOI), PDF links and sometimes full text.
        if url and not url.lower().split("?", 1)[0].endswith(".pdf"):
            attempts += 1
            tried.add(url)
            try:
                data, ctype = _download(client, url, cancel_event)
                if "pdf" in ctype or data[:4] == b"%PDF":
                    take_pdf(data, url)
                else:
                    html = data.decode("utf-8", errors="replace")
                    meta, text = parse_html(html, url)
                    result.metadata.update(meta)
                    if len(meta.get("abstract", "")) > len(abstract):
                        abstract = meta["abstract"]
                    if not pdf_url and meta.get("pdf_url"):
                        pdf_url = meta["pdf_url"]
                    if looks_like_full_text(text):
                        full_text = text
                        result.source = "html"
                    queue += landing_pdf_links(html, url)
            except FetchCancelled:
                return result
            except Exception as e:
                failed(e, "Landing page", url)

        # 2. PDF candidates: the known link, patterns of the landing page, links it offers; then other copies.
        queue = [pdf_url, *pattern_pdf_urls(url), *pattern_pdf_urls(pdf_url), *queue]
        asked_finder = False
        while not full_text and time.monotonic() < budget_ends and attempts < MAX_FULLTEXT_ATTEMPTS:
            candidate = next((u for u in queue if u and u not in tried), None)
            if candidate is None:
                # Every download failed before reaching a server: one dead site, or no internet at all?
                offline = attempts > 0 and connect_failures == attempts and not _internet_reachable()
                if finder is None or asked_finder or offline:
                    break
                asked_finder = True
                queue += finder.candidates({**paper, **{k: v for k, v in result.metadata.items() if v}}, exclude=tried)
                continue
            tried.add(candidate)
            attempts += 1
            try:
                data, ctype = _download(client, candidate, cancel_event)
                if data[:4] == b"%PDF" or "pdf" in ctype:
                    take_pdf(data, candidate)
                elif "html" in ctype or data.lstrip()[:1] == b"<":
                    # A repository or publisher page: follow the PDF links it offers (one level deep).
                    html = data.decode("utf-8", errors="replace")
                    meta, _ = parse_html(html, candidate)
                    queue += [meta.get("pdf_url", ""), *landing_pdf_links(html, candidate), *pattern_pdf_urls(candidate)]
                else:
                    result.notes.append("PDF link did not return a PDF")
            except FetchCancelled:
                return result
            except Exception as e:
                failed(e, "PDF", candidate)

    # Every download failed before reaching a server: maybe the sites are down, maybe our internet is.
    result.unreachable = attempts > 0 and connect_failures == attempts and not full_text
    if full_text and result.source == "pdf":
        pdf_url = result.metadata.get("pdf_url") or pdf_url
    if pdf_url and pdf_url != paper.get("pdf_url"):
        result.metadata["pdf_url"] = pdf_url
    if abstract and abstract != (paper.get("abstract") or ""):
        result.metadata["abstract"] = abstract
    if result.metadata.get("publication_date") and not paper.get("year"):
        year = extract_year(result.metadata["publication_date"])
        if year:
            result.metadata["year"] = year

    if full_text:
        header = f"Abstract: {abstract}\n\n" if abstract and abstract[:200] not in full_text else ""
        result.text = header + full_text
        result.status = "full_text"
    elif len(abstract) >= MIN_ABSTRACT_CHARS:
        result.text = abstract
        result.status = "abstract"
        result.source = result.source or "abstract"
    elif paper.get("title"):
        result.text = abstract
        result.status = "metadata_only"
        result.source = "metadata"
    return result
