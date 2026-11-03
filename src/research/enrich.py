"""Paper metadata enrichment from OpenAlex (free scholarly index, no API key).

Adds what SearXNG does not provide — citation counts, whether the work is peer reviewed, the venue,
and open-access PDF links. Lookups go DOI → arXiv id → title search (only accepted when the title
matches closely). Failures are never fatal: the paper keeps its search metadata.
"""
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import httpx

from src.config import OPENALEX_API_KEY, OPENALEX_EMAIL, OPENALEX_ENABLED, OPENALEX_URL
from src.research.normalize import normalize_title

logger = logging.getLogger(__name__)

TITLE_MATCH = 0.9
FIELDS = ("id,doi,title,publication_year,cited_by_count,type,primary_location,best_oa_location,locations,open_access,"
          "ids,authorships")
SEARCH_FIELDS = FIELDS + ",abstract_inverted_index,counts_by_year"
PEER_REVIEWED_SOURCES = {"journal", "conference", "book series", "ebook platform"}
PREPRINT_SOURCES = {"repository"}
MIN_INTERVAL = 0.15  # OpenAlex allows ~10 requests/s; stay well below

_lock = threading.Lock()
_last = 0.0
# Shared across clients: OpenAlex meters credits per key per day (reset 00:00 UTC).
_quota = {"limit": None, "remaining": None, "exhausted_until": None, "last_error": ""}


def quota_state() -> dict:
    return dict(_quota)


def reset_quota_state() -> None:
    _quota.update(limit=None, remaining=None, exhausted_until=None, last_error="")


def _next_reset() -> datetime:
    now = datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)


@dataclass
class Enrichment:
    year: int | None = None
    citation_count: int | None = None
    publication_type: str = ""        # OpenAlex work type: article, conference-paper, preprint, review, ...
    peer_reviewed: bool | None = None
    venue: str = ""
    oa_pdf_url: str = ""
    doi: str = ""


class OpenAlexClient:
    def __init__(self, base_url: str = OPENALEX_URL, email: str = OPENALEX_EMAIL,
                 transport: httpx.BaseTransport | None = None, enabled: bool = OPENALEX_ENABLED,
                 api_key: str = OPENALEX_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_key = api_key
        self.transport = transport
        self.enabled = enabled
        self.failed = False  # the last find() hit a network/server error (so "not found" is not trustworthy)

    def search_works(self, query: str, per_page: int = 25, select: str | None = None) -> dict | None:
        """Raw OpenAlex search: {"meta": {"count": …}, "results": [...]} (includes abstracts and yearly citations)."""
        if not self.enabled or not query.strip():
            return None
        return self._get("/works", {"search": query[:250], "per-page": per_page,
                                    "select": select or SEARCH_FIELDS})

    def year_counts(self, query: str) -> dict[int, int]:
        """How many works match `query`, per publication year (a saturation / momentum signal)."""
        if not self.enabled or not query.strip():
            return {}
        data = self._get("/works", {"search": query[:250], "group_by": "publication_year"})
        out = {}
        for g in (data or {}).get("group_by") or []:
            try:
                out[int(g["key"])] = int(g["count"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        global _last
        until = _quota["exhausted_until"]
        if until and datetime.now(timezone.utc) < until:
            return None  # daily credits used up: don't waste requests until the reset
        with _lock:
            delay = _last + MIN_INTERVAL - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _last = time.monotonic()
        params = dict(params or {})
        if "group_by" not in params:
            params.setdefault("select", FIELDS)
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            with httpx.Client(timeout=15, transport=self.transport,
                              headers={"User-Agent": "ResearchMind/0.2"}) as client:
                resp = client.get(f"{self.base_url}{path}", params=params)
        except httpx.HTTPError as e:
            logger.info("OpenAlex request failed: %s", e.__class__.__name__)
            _quota["last_error"] = f"unreachable ({e.__class__.__name__})"
            self.failed = True
            return None
        for key, header in (("limit", "x-ratelimit-limit"), ("remaining", "x-ratelimit-remaining")):
            value = resp.headers.get(header)
            if value and value.lstrip("-").isdigit():
                _quota[key] = int(value)
        if resp.status_code == 429:
            self.failed = True
            if _quota.get("remaining") == 0 or "daily" in resp.text.lower() or "credit" in resp.text.lower():
                _quota.update(exhausted_until=_next_reset(), remaining=0, last_error="daily credit limit reached")
                logger.warning("OpenAlex daily credit limit reached; skipping OpenAlex until the next reset")
            else:  # a short burst limit: back off for a minute, not a day
                _quota.update(exhausted_until=datetime.now(timezone.utc) + timedelta(minutes=1),
                              last_error="rate limited (burst)")
            return None
        if resp.status_code in (401, 403):
            _quota["last_error"] = "API key rejected"
            logger.warning("OpenAlex rejected the API key (HTTP %s)", resp.status_code)
            return None
        if resp.status_code != 200:
            self.failed = self.failed or resp.status_code >= 500
            return None
        _quota["last_error"] = ""
        try:
            return resp.json()
        except ValueError:
            return None

    def find(self, paper: dict) -> dict | None:
        self.failed = False
        if not self.enabled:
            return None
        if paper.get("doi"):
            work = self._get(f"/works/doi:{paper['doi']}")
            if work:
                return work
        if paper.get("arxiv_id"):
            work = self._get(f"/works/doi:10.48550/arXiv.{paper['arxiv_id']}")
            if work:
                return work
        title = (paper.get("title") or "").strip()
        if len(title) < 15:
            return None
        # General search first, then a title-only search (finds workshop papers the general ranking buries).
        words = " ".join(w for w in normalize_title(title).split() if w not in STOPWORDS)[:250]
        for params in ({"search": title[:250], "per-page": 5}, {"filter": f"title.search:{words}", "per-page": 5}):
            data = self._get("/works", params)
            for work in (data or {}).get("results") or []:
                if titles_match(title, work.get("title") or ""):
                    return work
        return None


STOPWORDS = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with", "by", "is", "are", "via", "from"}


def titles_match(ours: str, theirs: str) -> bool:
    """Near-identical titles, or ours is a shortened form of theirs (every meaningful word of ours appears,
    e.g. a file name like "Holistic Evaluation of LLMs Robustness Accuracy and Toxicity")."""
    a, b = normalize_title(ours), normalize_title(theirs)
    if not a or not b:
        return False
    if SequenceMatcher(None, a, b).ratio() >= TITLE_MATCH:
        return True
    words = [_stem(w) for w in a.split() if w not in STOPWORDS]
    theirs_words = {_stem(w) for w in b.split()}
    if len(words) < 6 or len(b) > 3 * len(a):
        return False
    return sum(w in theirs_words for w in words) / len(words) >= 0.95


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def parse_work(work: dict) -> Enrichment:
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    source_type = (source.get("type") or "").lower()
    work_type = (work.get("type") or "").lower()
    if work_type in ("preprint", "posted-content") or source_type in PREPRINT_SOURCES:
        peer_reviewed = False
    elif source_type in PEER_REVIEWED_SOURCES or work_type in ("conference-paper", "proceedings-article"):
        peer_reviewed = True
    elif work_type in ("article", "review", "book-chapter") and source.get("display_name"):
        peer_reviewed = True
    else:
        peer_reviewed = None
    oa = work.get("best_oa_location") or {}
    pdf = oa.get("pdf_url") or (work.get("open_access") or {}).get("oa_url") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return Enrichment(
        year=work.get("publication_year"),
        citation_count=work.get("cited_by_count"),
        publication_type=work_type,
        peer_reviewed=peer_reviewed,
        venue=source.get("display_name") or "",
        oa_pdf_url=pdf if isinstance(pdf, str) and pdf.lower().split("?")[0].endswith(".pdf") else "",
        doi=doi,
    )


def enrich_fields(paper: dict, client: OpenAlexClient | None = None) -> dict:
    """Paper-row updates from OpenAlex (only filling or improving fields). Empty dict if nothing found."""
    client = client or OpenAlexClient()
    work = client.find(paper)
    if not work:
        return {}
    e = parse_work(work)
    updates: dict = {"publication_type": e.publication_type, "peer_reviewed": e.peer_reviewed}
    if e.citation_count is not None:
        updates["citation_count"] = e.citation_count
    if e.year and not paper.get("year"):
        updates["year"] = e.year
    if e.venue and not paper.get("venue"):
        updates["venue"] = e.venue
    if e.doi and not paper.get("doi"):
        updates["doi"] = e.doi
    if e.oa_pdf_url and not paper.get("pdf_url"):
        updates["pdf_url"] = e.oa_pdf_url
    return updates


def check_openalex() -> tuple[bool, str]:
    """For /api/health: credits left today (a cheap singleton lookup, which costs no credits)."""
    if not OPENALEX_ENABLED:
        return True, "disabled"
    client = OpenAlexClient()
    client._get("/works/doi:10.1145/3691620.3695018", {"select": "id"})
    q = quota_state()
    if q["exhausted_until"] and datetime.now(timezone.utc) < q["exhausted_until"]:
        return False, f"daily credits used up — resumes {q['exhausted_until'].strftime('%H:%M UTC')}"
    if q["last_error"]:
        return False, q["last_error"]
    key = "with API key" if OPENALEX_API_KEY else "anonymous — add OPENALEX_API_KEY for 10× more"
    if q["remaining"] is not None and q["limit"]:
        return True, f"{q['remaining']:,}/{q['limit']:,} credits left today · {key}"
    return True, f"connected · {key}"
