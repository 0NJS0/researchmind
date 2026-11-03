"""The research agent: a LangGraph loop of search → screen → process → reflect.

Every step records concise activity events (never model reasoning) so the UI can show progress.
"""
import logging
import threading
from datetime import date, datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.config import (
    MAX_SEARCH_ITERATIONS, MAX_PAPERS_PER_SESSION, MAX_QUERIES_PER_ITERATION, MAX_RESULTS_PER_QUERY,
    MAX_LLM_RELEVANCE_CHECKS, MIN_LEXICAL_RELEVANCE, ENABLE_PAPER_COMPARISON,
    AUTOPILOT_MAX_PAPERS_PER_TOPIC, AUTOPILOT_MIN_RELEVANCE,
    DISCOVERY_ABSTRACTS, DISCOVERY_PROCEEDINGS_LIMIT, DISCOVERY_QUERIES,
)
from src.llm.llm import (
    LLMError, LLMProvider, NetworkDown, QuotaExhausted, RateLimited, get_provider, internet_reachable, is_outage,
)
from src.llm.structured import StructuredOutputError
from src.research import llm_tasks, prompts
from src.research.enrich import OpenAlexClient
from src.research.fulltext import FullTextFinder
from src.research.content import fetch_content
from src.research.dedup import dedupe_batch, find_duplicate, merge_metadata
from src.research.gaps import GapEngine
from src.research.models import NormalizedPaper, TopicAnalysis
from src.research.normalize import is_academic_url
from src.research.relevance import lexical_score, terms, topic_terms
from src.research.searxng import SearchUnavailable, SearXNGClient, UNAVAILABLE_MESSAGE
from src.research.store import ResearchStore, now
from src.research.synthesis import merge_synthesis, render_synthesis, validate_synthesis, remove_paper

logger = logging.getLogger(__name__)

LLM_FAILURES = (LLMError, StructuredOutputError)


def screening_priority(lexical: float, paper: dict, today: date | None = None) -> float:
    """Order in which candidates get the (limited) LLM relevance check."""
    from src.research.scholarly import top_venue
    year = paper.get("year")
    age = ((today or date.today()).year - int(year)) if year else 99
    recent = 0.15 if age <= 1 else 0.08 if age <= 2 else 0.0
    return lexical + recent + (0.15 if top_venue(paper.get("venue")) else 0.0)


def _outage_message(err: BaseException) -> str:
    status = getattr(err, "status", None)
    if status == 401:
        return "OpenRouter rejected the API key (expired or revoked?). Update LLM_API_KEY and restart."
    if status in (402, 403):
        return "OpenRouter refused the request for this key (credit or permission). Check the key on openrouter.ai."
    if isinstance(err, NetworkDown):
        return "No internet connection; unfinished papers stay queued and resume when it is back."
    if isinstance(err, RateLimited):
        return "The free models are rate limited right now; unfinished papers stay queued."
    return "The LLM service is unreachable right now; unfinished papers stay queued."
RELEVANCE_BATCH = 5
MAX_SCREEN_DEFERRALS = 3   # search rounds an unchecked candidate may wait before it is dropped
MAX_CANDIDATE_POOL = 200   # unchecked candidates kept for later rounds
LLM_ACCEPT_SCORE = 0.5
LEXICAL_FALLBACK_ACCEPT = 0.35
MAX_RELATED_FOR_COMPARISON = 3
# Paper states that a later run should pick up again (interrupted or partially failed work).
RESUMABLE = ("pending", "fetching", "summarizing", "analyzing", "analysis_failed")
PAPER_FIELDS = ("title", "authors", "abstract", "url", "pdf_url", "doi", "arxiv_id", "source", "engines",
                "venue", "publication_date", "year", "citation_count", "found_by_query")


class Cancelled(Exception):
    pass


class AgentState(TypedDict, total=False):
    iterations_left: int
    queries: list[str]
    query_source: str
    candidate_ids: list[str]
    selected_ids: list[str]
    search_failed: bool


class ResearchAgent:
    def __init__(self, store: ResearchStore, session_id: str, *, searx: SearXNGClient | None = None,
                 provider: LLMProvider | None = None, cancel_event: threading.Event | None = None,
                 fetch_transport=None, fulltext=None, proceedings=None):
        self.store = store
        self.session_id = session_id
        self.searx = searx or SearXNGClient()
        self.provider = provider
        self.cancel_event = cancel_event or threading.Event()
        self.fetch_transport = fetch_transport
        self.fulltext = fulltext or FullTextFinder(transport=fetch_transport)  # other free copies of a paper
        self.proceedings = proceedings        # recent top-venue proceedings index (shared cache by default)
        self.openalex = OpenAlexClient(transport=fetch_transport)
        self.gaps = GapEngine(store, session_id)
        self.pdf_cache: dict[str, bytes] = {}  # paper id → PDF bytes fetched this run (for the Library)

    # ── helpers ──

    @property
    def session(self) -> dict:
        return self.store.get_session(self.session_id)

    @property
    def topic(self) -> str:
        return self.session["topic"]

    def emit(self, type_: str, message: str, **data) -> None:
        self.store.add_event(self.session_id, type_, message, data or None)
        logger.info("[%s] %s", self.session_id[:8], message)

    def set_status(self, status: str, message: str = "") -> None:
        self.store.update_session(self.session_id, status=status, status_message=message)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def topic_analysis(self) -> TopicAnalysis:
        data = self.session.get("topic_analysis")
        return TopicAnalysis.model_validate(data) if data else TopicAnalysis()

    def field_guidance(self) -> str:
        return llm_tasks.field_guidance(self.topic_analysis())

    def ensure_field_profile(self) -> None:
        """Topics analysed before field profiles existed get one (a single request, once)."""
        analysis = self.topic_analysis()
        if analysis.has_profile() or not self.session.get("topic_analysis"):
            return
        try:
            fresh = llm_tasks.analyze_topic(self.topic, self.session["description"], self.provider)
        except LLM_FAILURES as e:
            if is_outage(e):
                raise  # LLM outage / rejected key: stop and retry later, don't degrade
            logger.info("Field profile could not be created: %s", e)
            return
        merged = analysis.model_copy(update={"designs": fresh.designs, "key_quantities": fresh.key_quantities,
                                             "quality_signals": fresh.quality_signals, "field": analysis.field or fresh.field})
        self.store.update_session(self.session_id, topic_analysis=merged.model_dump())
        if merged.has_profile():
            self.emit("info", f"Field profile added for {merged.field or self.topic}",
                      key_quantities=merged.key_quantities[:5])

    def _selected_count(self) -> int:
        return len(self.store.list_papers(self.session_id, selection="selected"))

    def _all_queries(self) -> list[str]:
        return [q["query"] for q in self.store.list_queries(self.session_id)]

    # ── step 1: understand topic ──

    def understand_topic(self) -> list[str]:
        self.set_status("searching", "Understanding research topic")
        self.emit("step", "Understanding research topic")
        session = self.session
        try:
            analysis = llm_tasks.analyze_topic(self.topic, session["description"], self.provider)
            self.store.update_session(self.session_id, topic_analysis=analysis.model_dump())
            if analysis.field:
                self.emit("success", f"Identified field: {analysis.field}",
                          concepts=analysis.key_concepts[:8], key_quantities=analysis.key_quantities[:5])
        except LLM_FAILURES as e:
            if is_outage(e):
                raise  # LLM outage / rejected key: stop and retry later, don't degrade
            analysis = TopicAnalysis()
            self.emit("warning", "Topic analysis failed; searching with the topic as written", error=str(e)[:200])

        self.check_cancel()
        queries = [self.topic]  # the user's own wording is always searched
        try:
            queries += llm_tasks.generate_search_queries(
                self.topic, analysis, max(1, MAX_QUERIES_PER_ITERATION - 1), previous=[self.topic],
                provider=self.provider, focus=session["description"])
        except LLM_FAILURES as e:
            if is_outage(e):
                raise  # LLM outage / rejected key: stop and retry later, don't degrade
            fallback = [f"{self.topic} {c}" for c in (analysis.subtopics or analysis.key_concepts)[:3]]
            queries += fallback
            self.emit("warning", "Query generation failed; using fallback queries", error=str(e)[:200])
        queries = list(dict.fromkeys(q for q in queries if q))[:MAX_QUERIES_PER_ITERATION]
        self.emit("success", f"Generated {len(queries)} search queries", queries=queries)
        return queries

    # ── step 2: search ──

    def add_candidates(self, found: list[NormalizedPaper], announce: bool = False) -> tuple[list[str], int]:
        """Store new, de-duplicated candidates (never a paper the user deleted). Returns (new ids, duplicates)."""
        unique, removed = dedupe_batch(found)
        existing = self.store.list_papers(self.session_id, selection=None)
        deleted = self.store.tombstones(self.session_id)
        new_ids = []
        for cand in unique:
            if deleted and find_duplicate(cand, deleted):
                removed += 1  # the user deleted this paper: never bring it back
                continue
            dup = find_duplicate(cand, existing)
            if dup:
                removed += 1
                updates = merge_metadata(dup, cand)
                if updates:
                    self.store.update_paper(dup["id"], **updates)
                continue
            row = self.store.add_paper(self.session_id, {
                **{k: v for k, v in cand.model_dump().items() if k in PAPER_FIELDS},
                "selection_status": "candidate",
                "relevance_score": None,
            })
            existing.append(row)
            new_ids.append(row["id"])
        if announce:
            self.emit("success", f"Found {len(found)} results · removed {removed} duplicates · "
                                 f"{len(new_ids)} new candidates",
                      found=len(found), duplicates=removed, candidates=len(new_ids))
        return new_ids, removed

    def discover(self, queries: list[str]) -> list[str]:
        """Recent papers from top venues that the web search misses: official proceedings (CVPR, ICCV, WACV,
        ECCV, NeurIPS), arXiv newest-first (venue from "Accepted to …" comments) and OpenAlex's last two years.
        No LLM requests. Returns the ids of new candidates."""
        from src.research import scholarly
        analysis = self.topic_analysis()
        core, expanded = topic_terms(self.topic, analysis.key_concepts + analysis.synonyms + analysis.subtopics)
        found: list[NormalizedPaper] = []
        sources: dict[str, int] = {}

        def collect(name: str, fn) -> None:
            try:
                papers = fn()
            except Exception:
                logger.exception("Discovery source %s failed", name)
                return
            sources[name] = len(papers)
            found.extend(papers)

        proceedings = self.proceedings or (scholarly.shared_index() if self.fetch_transport is None
                                            else scholarly.ProceedingsIndex(transport=self.fetch_transport))
        collect("proceedings", lambda: proceedings.match(core, expanded, limit=DISCOVERY_PROCEEDINGS_LIMIT))
        for q in queries[:DISCOVERY_QUERIES]:
            collect("arxiv", lambda q=q: scholarly.arxiv_recent(q, 25, transport=self.fetch_transport))
            collect("openalex", lambda q=q: scholarly.openalex_recent(self.openalex, q))
        for p in found:
            p.found_by_query = p.found_by_query or "recent top-venue discovery"
        new_ids, dups = self.add_candidates(found)
        # Proceedings listings only have titles: fetch abstracts, for new candidates only, so relevance screening
        # sees what a paper is about.
        fetched = 0
        for pid in new_ids:
            row = self.store.get_paper(pid)
            if fetched >= DISCOVERY_ABSTRACTS or row.get("abstract") or "proceedings" not in (row.get("engines") or []):
                continue
            abstract = scholarly.fetch_abstract(NormalizedPaper(title=row["title"], url=row["url"]), self.fetch_transport)
            fetched += 1
            if abstract:
                self.store.update_paper(pid, abstract=abstract)
        summary = ", ".join(f"{k} {v}" for k, v in sources.items())
        self.emit("success", f"Recent top-venue discovery: {len(found)} results ({summary}) · {len(new_ids)} new candidates",
                  candidates=len(new_ids), sources=sources)
        return new_ids

    def search(self, queries: list[str], source: str) -> tuple[list[str], bool]:
        """Run queries; store new, de-duplicated candidates. Returns (candidate_ids, all_queries_failed)."""
        session = self.session
        iteration = session["iteration"] + 1
        self.store.update_session(self.session_id, iteration=iteration)
        self.set_status("searching", f"Searching academic sources (round {iteration})")
        self.emit("step", f"Searching academic sources (round {iteration})", queries=queries)

        found: list[NormalizedPaper] = []
        failures = 0
        for q in queries:
            self.check_cancel()
            qrow = self.store.add_query(self.session_id, q, source, iteration)
            try:
                results = self.searx.search(q, MAX_RESULTS_PER_QUERY)
            except SearchUnavailable as e:
                failures += 1
                self.store.update_query(qrow["id"], status="failed", error=str(e)[:300])
                self.emit("error", f"Search failed for “{q}”: {e}")
                continue
            self.store.update_query(qrow["id"], status="done", result_count=len(results))
            found.extend(results)

        if queries and failures == len(queries):
            self.emit("error", UNAVAILABLE_MESSAGE)
            return [], True

        new_ids, _ = self.add_candidates(found, announce=True)
        return new_ids, False

    # ── step 3: relevance screening ──

    def screen(self, candidate_ids: list[str], continuous: bool = False) -> list[str]:
        """Select relevant candidates. `continuous` (autopilot): no per-run slot cap beyond the topic
        limit, and a stricter relevance threshold."""
        if not candidate_ids:
            return []
        accept_score = max(LLM_ACCEPT_SCORE, AUTOPILOT_MIN_RELEVANCE) if continuous else LLM_ACCEPT_SCORE
        limit = AUTOPILOT_MAX_PAPERS_PER_TOPIC if continuous else MAX_PAPERS_PER_SESSION
        self.set_status("processing", "Assessing paper relevance")
        self.emit("step", f"Assessing relevance of {len(candidate_ids)} candidates")
        analysis = self.topic_analysis()
        core, expanded = topic_terms(self.topic, analysis.key_concepts + analysis.synonyms + analysis.subtopics)

        scored = []
        rejected = 0
        for pid in candidate_ids:
            paper = self.store.get_paper(pid)
            paper["academic_signal"] = bool(paper["doi"] or paper["arxiv_id"] or paper["pdf_url"]
                                            or paper["venue"] or is_academic_url(paper["url"]))
            score = lexical_score(paper, core, expanded)
            if score < MIN_LEXICAL_RELEVANCE:
                self.store.update_paper(pid, selection_status="rejected", relevance_score=score,
                                        relevance_method="lexical",
                                        relevance_reason="Little overlap between the result and the topic terms")
                rejected += 1
            else:
                scored.append((score, paper))
        # Which results the LLM checks first: keyword overlap, plus a boost for recent and top-venue papers
        # (new papers have few citations and weaker keyword matches, but are often what matters most).
        scored.sort(key=lambda x: screening_priority(x[0], x[1]), reverse=True)

        to_check = scored[:MAX_LLM_RELEVANCE_CHECKS]
        for score, paper in scored[MAX_LLM_RELEVANCE_CHECKS:]:
            if continuous:
                # The autopilot checks the rest in its next rounds instead of discarding them unseen, but a result
                # that keeps losing to better ones is dropped eventually (and the pool can't grow without bound).
                deferrals = (paper.get("screen_deferrals") or 0) + 1
                overflow_rank = scored.index((score, paper)) - MAX_LLM_RELEVANCE_CHECKS
                if deferrals >= MAX_SCREEN_DEFERRALS or overflow_rank >= MAX_CANDIDATE_POOL:
                    self.store.update_paper(paper["id"], selection_status="rejected", relevance_score=score,
                                            relevance_method="lexical", screen_deferrals=deferrals,
                                            relevance_reason="Not checked: weaker match than the other results "
                                                             f"for {MAX_SCREEN_DEFERRALS} search rounds")
                    rejected += 1
                else:
                    self.store.update_paper(paper["id"], relevance_score=score, relevance_method="lexical",
                                            screen_deferrals=deferrals)
                continue
            self.store.update_paper(paper["id"], selection_status="rejected", relevance_score=score,
                                    relevance_method="lexical",
                                    relevance_reason="Lower-ranked result; relevance-check limit reached")
            rejected += 1

        accepted: list[tuple[float, dict, str, str]] = []
        scope = analysis.summary or self.topic
        for i in range(0, len(to_check), RELEVANCE_BATCH):
            self.check_cancel()
            batch = to_check[i:i + RELEVANCE_BATCH]
            papers = [NormalizedPaper(**{k: p[k] for k in ("title", "abstract", "year", "venue", "source")
                                         if p.get(k) is not None}) for _, p in batch]
            try:
                result = llm_tasks.evaluate_paper_relevance(self.topic, scope, papers, self.provider,
                                                            focus=self.session.get("description") or "")
                by_index = {a.index: a for a in result.assessments}
                for j, (lex, paper) in enumerate(batch, 1):
                    a = by_index.get(j)
                    if a and a.relevant and a.score >= accept_score:
                        accepted.append((0.7 * a.score + 0.3 * lex, paper, a.reason or "Relevant to the topic", "llm"))
                    else:
                        reason = (a.reason if a else "") or "Judged not relevant to the topic"
                        self.store.update_paper(paper["id"], selection_status="rejected",
                                                relevance_score=a.score if a else lex,
                                                relevance_method="llm", relevance_reason=reason)
                        rejected += 1
            except LLM_FAILURES as e:
                if is_outage(e):
                    raise  # LLM outage / rejected key: stop and retry later, don't degrade
                self.emit("warning", "LLM relevance check failed; falling back to keyword relevance",
                          error=str(e)[:200])
                for lex, paper in batch:
                    if lex >= LEXICAL_FALLBACK_ACCEPT:
                        accepted.append((lex, paper, "Strong keyword overlap with the topic", "lexical"))
                    else:
                        self.store.update_paper(paper["id"], selection_status="rejected", relevance_score=lex,
                                                relevance_method="lexical",
                                                relevance_reason="Weak keyword overlap (LLM check unavailable)")
                        rejected += 1

        accepted.sort(key=lambda x: x[0], reverse=True)
        slots = max(0, limit - self._selected_count())
        selected = []
        for score, paper, reason, method in accepted:
            if len(selected) >= slots:
                self.store.update_paper(paper["id"], selection_status="rejected", relevance_score=round(score, 3),
                                        relevance_method=method,
                                        relevance_reason=f"Relevant, but the limit of {limit} papers was reached")
                rejected += 1
                continue
            row = self.store.select_paper(paper["id"], relevance_score=round(score, 3), relevance_reason=reason,
                                          relevance_method=method, processing_status="pending")
            selected.append(row["id"])
            self.emit("paper", f"Selected {row['ref']}: {row['title']}", paper_id=row["id"], ref=row["ref"],
                      reason=reason)
        self.emit("success", f"Selected {len(selected)} relevant papers · rejected {rejected}",
                  selected=len(selected), rejected=rejected)
        return selected

    # ── step 4: per-paper processing ──

    def process_papers(self, paper_ids: list[str], force: bool = False) -> None:
        total = len(paper_ids)
        for n, pid in enumerate(paper_ids, 1):
            self.check_cancel()
            try:
                self.process_paper(pid, force=force, position=(n, total))
            except (Cancelled, QuotaExhausted):
                self.store.update_paper(pid, processing_status="pending")
                raise
            except Exception as e:  # one bad paper must not stop the session
                if is_outage(e):  # ...but an LLM outage or a rejected key affects every paper: stop, keep it queued
                    self.store.update_paper(pid, processing_status="pending")
                    raise
                logger.exception("Processing failed for paper %s", pid)
                self.store.update_paper(pid, processing_status="failed", error=str(e)[:500])
                self.emit("error", f"Processing failed for a paper: {e.__class__.__name__}", paper_id=pid)

    def process_paper(self, paper_id: str, force: bool = False, position: tuple[int, int] = (1, 1)) -> None:
        paper = self.store.get_paper(paper_id)
        ref = paper["ref"]
        self.set_status("processing", f"Processing paper {position[0]} of {position[1]}")
        self.emit("step", f"Processing {ref} ({position[0]}/{position[1]}): {paper['title']}",
                  paper_id=paper_id, ref=ref)

        # Content
        # Uploaded / attached PDFs are the best text there is: "force" re-summarises them, it doesn't re-download.
        refetch = force and paper.get("content_source") not in ("library", "upload")
        if refetch or not paper["raw_content"] or paper["content_status"] in ("pending", "unavailable"):
            self.store.update_paper(paper_id, processing_status="fetching")
            res = fetch_content(paper, transport=self.fetch_transport, cancel_event=self.cancel_event,
                                finder=self.fulltext)
            self.check_cancel()
            if res.unreachable and not internet_reachable():
                # Don't downgrade the paper to an abstract-only analysis because our own connection is down.
                raise NetworkDown("No internet connection: the paper's text could not be downloaded")
            if res.pdf_bytes:
                self.pdf_cache[paper_id] = res.pdf_bytes
            updates = {k: v for k, v in res.metadata.items()
                       if k in ("abstract", "pdf_url", "doi", "venue", "publication_date", "year", "authors")
                       and v and not (k != "abstract" and paper.get(k))}
            if res.status != "full_text":
                updates["fulltext_checked_at"] = datetime.now(timezone.utc)  # looked everywhere; re-checked weekly
            self.store.update_paper(paper_id, raw_content=res.text, content_status=res.status,
                                    content_source=res.source, **updates)
            paper = self.store.get_paper(paper_id)
            messages = {
                "full_text": f"{ref}: full text retrieved ({res.source})",
                "abstract": f"{ref}: full text unavailable — analysis will be abstract-level",
                "metadata_only": f"{ref}: only title/metadata available — summary will be minimal",
                "unavailable": f"{ref}: no usable content",
            }
            self.emit("info" if res.status == "full_text" else "warning", messages[res.status],
                      paper_id=paper_id, ref=ref, notes=res.notes[:3])
        if paper["content_status"] == "unavailable":
            self.store.update_paper(paper_id, processing_status="skipped", error="No usable content",
                                    processed_at=datetime.now(timezone.utc))
            return

        # Summary (never regenerated unless forced)
        self.check_cancel()
        summary = self.store.get_summary(paper_id)
        if summary is None or force:
            self.set_status("summarizing", f"Summarizing {ref}")
            self.store.update_paper(paper_id, processing_status="summarizing")
            self.ensure_field_profile()
            try:
                out = llm_tasks.summarize_paper(self.topic, paper, paper["raw_content"], paper["content_status"],
                                                self.provider, guidance=self.field_guidance())
            except LLM_FAILURES as e:
                if is_outage(e):
                    raise  # temporary / account-wide: the caller puts the paper back in the queue
                self.store.update_paper(paper_id, processing_status="failed", error=f"Summary failed: {e}"[:500],
                                        processed_at=datetime.now(timezone.utc))
                self.emit("error", f"{ref}: summary generation failed", paper_id=paper_id, ref=ref)
                return
            used = self.provider or get_provider()
            model = getattr(used, "last_model", "") or getattr(used, "model", "")
            self.store.save_summary(paper_id, paper["content_status"], out.model_dump(), model)
            summary = self.store.get_summary(paper_id)
            self.emit("success", f"{ref}: structured summary generated ({paper['content_status'].replace('_', ' ')})",
                      paper_id=paper_id, ref=ref)

        if paper["content_status"] == "metadata_only":
            # Title-only information is too thin to count as evidence about the literature.
            self.store.update_paper(paper_id, processing_status="done", processed_at=datetime.now(timezone.utc))
            self.emit("info", f"{ref}: skipped gap analysis (metadata only)", paper_id=paper_id, ref=ref)
            return

        # Re-analysis replaces this paper's earlier evidence and synthesis claims instead of piling up on top.
        self.gaps.retract_paper(paper_id)
        session = self.session or {}
        if session.get("synthesis") and paper.get("ref"):
            self.store.update_session(self.session_id, synthesis=remove_paper(session["synthesis"], paper["ref"], paper_id))
        failures = self.analyze_in_context(paper, summary)
        done_at = datetime.now(timezone.utc)
        if failures:
            self.store.update_paper(paper_id, processing_status="analysis_failed", processed_at=done_at,
                                    error=f"{', '.join(failures)} failed; will retry on the next run")
        else:
            self.store.update_paper(paper_id, processing_status="done", error="", processed_at=done_at)

    def _rendered(self, paper: dict, summary: dict, limit: int = 1800) -> str:
        return llm_tasks.render_summary(paper["ref"], paper["title"], summary["sections"],
                                        summary["analysis_level"], limit=limit)

    def _related_papers(self, paper: dict, summary: dict) -> list[tuple[dict, dict]]:
        summaries = self.store.summaries_for_session(self.session_id)
        focus = terms(self._rendered(paper, summary))
        candidates = []
        for other in self.store.list_papers(self.session_id, selection="selected"):
            if other["id"] == paper["id"] or other["id"] not in summaries or not other["ref"]:
                continue
            s = summaries[other["id"]]
            overlap = len(focus & terms(self._rendered(other, s)))
            candidates.append((overlap, other, s))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [(o, s) for _, o, s in candidates[:MAX_RELATED_FOR_COMPARISON]]

    def analyze_in_context(self, paper: dict, summary: dict, compare: bool = True) -> list[str]:
        """Compare with related papers, update the gap state, then the synthesis.

        Returns the names of the steps that failed (empty when everything succeeded)."""
        ref = paper["ref"]
        failures: list[str] = []
        paper_text = self._rendered(paper, summary)

        # Comparison
        related = self._related_papers(paper, summary)
        if compare and ENABLE_PAPER_COMPARISON and related:
            self.check_cancel()
            self.set_status("analyzing", f"Comparing {ref} with related papers")
            try:
                out = llm_tasks.compare_papers(self.topic, paper_text,
                                               [self._rendered(o, s, 900) for o, s in related], self.provider,
                                               guidance=self.field_guidance())
                by_ref = {o["ref"]: o for o, _ in related}
                relations = []
                for rel in out.relations:
                    other = by_ref.get(rel.paper_id.strip().upper())
                    if other:
                        relations.append({"related_paper_id": other["id"], **rel.model_dump(exclude={"paper_id"})})
                self.store.replace_relations(self.session_id, paper["id"], relations)
                contradictions = sum(1 for r in relations if r["relationship"] == "contradicts")
                refs = ", ".join(by_ref) or "none"
                self.emit("info", f"{ref}: compared with {refs}"
                                  + (f" · {contradictions} contradiction(s) found" if contradictions else ""),
                          paper_id=paper["id"], ref=ref)
            except LLM_FAILURES as e:
                if is_outage(e):
                    raise  # LLM outage / rejected key: stop and retry later, don't degrade
                self.emit("warning", f"{ref}: comparison failed", error=str(e)[:200])

        relations_text = "\n".join(
            f"{r['related_ref']} ({r['relationship']}): {r['similarities']} {r['differences']}"
            + (f" CONTRADICTION: {r['contradiction']}" if r["contradiction"] else "")
            for r in self.store.relations_for_paper(paper["id"])
        )

        # Gaps
        self.check_cancel()
        self.set_status("analyzing", f"Updating research gaps with {ref}")
        self.store.update_paper(paper["id"], processing_status="analyzing")
        gap_paper_text = llm_tasks.render_summary(
            ref, paper["title"], summary["sections"], summary["analysis_level"], limit=1600,
            only=("research_problem", "methodology", "dataset", "key_findings", "limitations", "future_work"))
        try:
            ops = llm_tasks.identify_research_gaps(self.topic, ref, gap_paper_text,
                                                   self.gaps.render_for_prompt(paper_text), relations_text,
                                                   self.provider, guidance=self.field_guidance())
            report = self.gaps.apply(ops, paper)
            self.emit("gap", f"Research gaps updated with {ref}: {report.summary()}", paper_id=paper["id"],
                      ref=ref, created=report.created, updated=report.updated, merged=report.merged,
                      addressed=report.addressed, rejected=len(report.rejected))
        except LLM_FAILURES as e:
            if is_outage(e):
                raise  # LLM outage / rejected key: stop and retry later, don't degrade
            failures.append("gap analysis")
            self.emit("warning", f"{ref}: gap analysis failed", error=str(e)[:200])

        # Synthesis
        self.check_cancel()
        try:
            session = self.session
            out = llm_tasks.update_research_synthesis(
                self.topic, render_synthesis(session.get("synthesis")), paper_text,
                "\n".join(f"{g['ref']}: {g['title']}" for g in self.store.list_gaps(self.session_id)),
                self.provider)
            by_ref = self.store.papers_by_ref(self.session_id)
            ref_to_id = {r: p["id"] for r, p in by_ref.items()}
            synthesis = merge_synthesis(session.get("synthesis"), validate_synthesis(out, set(by_ref), ref_to_id),
                                        set(by_ref), ref_to_id)
            self.store.update_session(self.session_id, synthesis=synthesis, synthesis_updated_at=now())
            self.emit("info", f"Research synthesis updated with {ref}", paper_id=paper["id"], ref=ref)
        except LLM_FAILURES as e:
            if is_outage(e):
                raise  # LLM outage / rejected key: stop and retry later, don't degrade
            failures.append("synthesis update")
            self.emit("warning", f"{ref}: synthesis update failed", error=str(e)[:200])
        return failures

    # ── step 5: reflect / expand search ──

    def expansion_queries(self) -> list[str]:
        papers = self.store.list_papers(self.session_id, selection="selected")
        titles = "\n".join(f"- {p['title']}" for p in papers[-15:]) or "- (none)"
        gaps = "\n".join(f"- {g['title']}" for g in self.store.list_gaps(self.session_id)[:10]) or "- (none)"
        context = prompts.SEARCH_EXPANSION_CONTEXT.format(titles=titles, gaps=gaps)
        n = max(2, MAX_QUERIES_PER_ITERATION // 2)
        return llm_tasks.generate_search_queries(self.topic, self.topic_analysis(), n,
                                                 previous=self._all_queries(), context=context,
                                                 provider=self.provider, focus=self.session.get("description") or "")

    # ── graph ──

    def build_graph(self):
        def plan(state: AgentState) -> AgentState:
            if state.get("queries"):
                return {}
            if not self.session.get("topic_analysis") and not self._all_queries():
                return {"queries": self.understand_topic(), "query_source": "topic"}
            return reflect_queries(state)

        def reflect_queries(state: AgentState) -> AgentState:
            self.emit("step", "Looking for additional literature based on the papers and gaps so far")
            try:
                queries = self.expansion_queries()
            except LLM_FAILURES as e:
                if is_outage(e):
                    raise  # LLM outage / rejected key: stop and retry later, don't degrade
                self.emit("warning", "Could not generate follow-up queries", error=str(e)[:200])
                queries = []
            if queries:
                self.emit("success", f"Generated {len(queries)} follow-up queries", queries=queries)
            return {"queries": queries, "query_source": "expansion"}

        def search_node(state: AgentState) -> AgentState:
            ids, failed = self.search(state.get("queries") or [], state.get("query_source", "user"))
            return {"candidate_ids": ids, "search_failed": failed, "queries": []}

        def screen_node(state: AgentState) -> AgentState:
            # Include candidates left unscreened by an earlier cancelled or crashed run.
            ids = list(state.get("candidate_ids") or [])
            ids += [p["id"] for p in self.store.list_papers(self.session_id, selection="candidate")
                    if p["id"] not in ids]
            return {"selected_ids": self.screen(ids)}

        def process_node(state: AgentState) -> AgentState:
            pending = [p["id"] for p in self.store.list_papers(self.session_id, selection="selected")
                       if p["processing_status"] in RESUMABLE]
            self.process_papers(pending)
            return {}

        def reflect_node(state: AgentState) -> AgentState:
            left = state.get("iterations_left", 1) - 1
            if left <= 0 or state.get("search_failed"):
                return {"iterations_left": left, "queries": []}
            if self._selected_count() >= MAX_PAPERS_PER_SESSION:
                self.emit("info", f"Paper limit reached ({MAX_PAPERS_PER_SESSION}); stopping search")
                return {"iterations_left": 0, "queries": []}
            return {"iterations_left": left, **reflect_queries(state)}

        def after_plan(state: AgentState) -> str:
            return "search" if state.get("queries") else END

        def after_reflect(state: AgentState) -> str:
            return "search" if state.get("queries") and state.get("iterations_left", 0) > 0 else END

        g = StateGraph(AgentState)
        g.add_node("plan", plan)
        g.add_node("search", search_node)
        g.add_node("screen", screen_node)
        g.add_node("process", process_node)
        g.add_node("reflect", reflect_node)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", after_plan, {"search": "search", END: END})
        g.add_edge("search", "screen")
        g.add_edge("screen", "process")
        g.add_edge("process", "reflect")
        g.add_conditional_edges("reflect", after_reflect, {"search": "search", END: END})
        return g.compile()

    # ── entry points (run inside a worker thread) ──

    def run_research(self, iterations: int = MAX_SEARCH_ITERATIONS, queries: list[str] | None = None) -> None:
        def body():
            state: AgentState = {"iterations_left": max(1, iterations)}
            if queries:
                state.update(queries=queries, query_source="user")
            final = self.build_graph().invoke(state, config={"recursion_limit": 10 + 5 * max(1, iterations)})
            return final

        self._run(body, start_message="Research started")

    def run_processing(self, paper_ids: list[str] | None = None, force: bool = False) -> None:
        def body():
            ids = paper_ids or [p["id"] for p in self.store.list_papers(self.session_id, selection="selected")
                                if p["processing_status"] in (*RESUMABLE, "failed")]
            self.process_papers(ids, force=force)
            return {}

        self._run(body, start_message="Processing papers")

    def run_gap_rebuild(self) -> None:
        """Replay gap detection over every summarised paper in order (summaries are reused)."""
        def body():
            self.emit("step", "Rebuilding research gaps and synthesis from existing summaries")
            self.set_status("analyzing", "Rebuilding research gaps")
            summaries = self.store.summaries_for_session(self.session_id)
            papers = [p for p in self.store.list_papers(self.session_id, selection="selected")
                      if p["id"] in summaries and p["content_status"] not in ("metadata_only", "unavailable")]
            # All-or-nothing: the old gaps stay visible and intact until the rebuild finishes; a cancel,
            # crash or exhausted quota rolls everything back.
            with self.store.atomic():
                self.store.reset_gaps(self.session_id)
                for n, paper in enumerate(papers, 1):
                    self.check_cancel()
                    self.emit("step", f"Re-analysing paper {n} of {len(papers)}: {paper['ref']}")
                    self.analyze_in_context(paper, summaries[paper["id"]], compare=False)
            return {}

        self._run(body, start_message="Gap analysis started")

    def _run(self, body, start_message: str) -> None:
        self.emit("info", start_message)
        try:
            final = body() or {}
            counts = self.store.session_counts(self.session_id)
            if final.get("search_failed") and counts["papers"] == 0:
                self.set_status("failed", UNAVAILABLE_MESSAGE)
                return
            message = f"{counts['papers']} papers · {counts['gaps']} research gaps"
            self.set_status("completed", message)
            self.emit("done", f"Completed — {message}")
        except Cancelled:
            self.set_status("cancelled", "Stopped by user")
            self.emit("warning", "Research stopped by user")
        except QuotaExhausted as e:
            self.set_status("paused", str(e)[:300])
            self.emit("error", f"Paused: {e}"[:400])
        except LLMError as e:
            if not is_outage(e):
                logger.exception("Research job failed for session %s", self.session_id)
            message = _outage_message(e) if is_outage(e) else f"LLM error: {str(e)[:200]}"
            self.set_status("paused" if is_outage(e) else "failed", message)
            self.emit("error", f"Stopped: {message}"[:400])
        except Exception as e:
            logger.exception("Research job failed for session %s", self.session_id)
            self.set_status("failed", f"Unexpected error: {e.__class__.__name__}")
            self.emit("error", f"Research failed: {e.__class__.__name__}")
