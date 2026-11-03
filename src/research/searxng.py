"""SearXNG search client."""
import logging

import httpx

from src.config import SEARXNG_URL, SEARXNG_CATEGORIES, SEARXNG_ENGINES, SEARXNG_TIMEOUT, MAX_RESULTS_PER_QUERY
from src.research.models import NormalizedPaper
from src.research.normalize import normalize_searxng_result

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "Search service is currently unavailable."


class SearchUnavailable(RuntimeError):
    pass


class SearXNGClient:
    def __init__(self, base_url: str = SEARXNG_URL, categories: str = SEARXNG_CATEGORIES,
                 engines: str = SEARXNG_ENGINES, timeout: float = SEARXNG_TIMEOUT,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.categories = categories
        self.engines = engines
        self.timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self._transport,
                            headers={"Accept": "application/json", "User-Agent": "ResearchMind/0.1"})

    def _params(self, query: str, page: int) -> dict:
        params = {"q": query, "format": "json", "pageno": page}
        if self.categories:
            params["categories"] = self.categories
        if self.engines:
            params["engines"] = self.engines
        return params

    def search_raw(self, query: str, page: int = 1) -> list[dict]:
        if not self.base_url:
            raise SearchUnavailable("SEARXNG_URL is not configured.")
        try:
            with self._client() as client:
                resp = client.get(f"{self.base_url}/search", params=self._params(query, page))
        except httpx.HTTPError as e:
            raise SearchUnavailable(f"{UNAVAILABLE_MESSAGE} ({e.__class__.__name__})") from e
        if resp.status_code == 403:
            raise SearchUnavailable("SearXNG refused the JSON API (enable `json` under search.formats).")
        if resp.status_code >= 400:
            raise SearchUnavailable(f"{UNAVAILABLE_MESSAGE} (HTTP {resp.status_code})")
        try:
            data = resp.json()
        except ValueError as e:
            raise SearchUnavailable("SearXNG returned a non-JSON response.") from e
        if not isinstance(data, dict):
            raise SearchUnavailable("SearXNG returned an unexpected response.")
        unresponsive = data.get("unresponsive_engines") or []
        results = data.get("results") or []
        if unresponsive:
            logger.info("SearXNG unresponsive engines for '%s': %s", query, unresponsive)
            if not results and page == 1:
                # SearXNG itself answered, but its engines didn't (typically: the homelab has no internet).
                # Reporting "no results" here would make the autopilot wait 12 hours for nothing.
                raise SearchUnavailable("SearXNG could not reach its search engines (internet connection down?)")
        return results

    def search(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[NormalizedPaper]:
        """Search and normalise, fetching further pages until max_results or results run out."""
        papers: list[NormalizedPaper] = []
        page = 1
        while len(papers) < max_results and page <= 3:
            raw = self.search_raw(query, page)
            if not raw:
                break
            for r in raw:
                p = normalize_searxng_result(r, query)
                if p:
                    papers.append(p)
            page += 1
        return papers[:max_results]

    def check(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "SEARXNG_URL not configured"
        try:
            self.search_raw("test")
            return True, "connected"
        except SearchUnavailable as e:
            return False, str(e)
