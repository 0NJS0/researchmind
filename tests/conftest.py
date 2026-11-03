import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Configure before any src.* import reads the environment.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _test_database_url() -> str:
    """TEST_DATABASE_URL if given, otherwise a local embedded Postgres + pgvector (pgserver)."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    import pgserver
    server = pgserver.get_server(str(ROOT / ".pgtest"), cleanup_mode="stop")
    return server.get_uri()


os.environ["DATABASE_URL"] = _test_database_url()
os.environ["DB_SCHEMA"] = f"rm_test_{os.getpid()}"
os.environ["LLM_CACHE"] = "false"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_URL"] = "http://llm.invalid/v1"
os.environ["LLM_MODEL"] = "test-model"
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_MIN_INTERVAL"] = "0"
os.environ["LLM_FALLBACK_MODELS"] = ""
os.environ["EMBEDDING_MODEL"] = "test-embed"
os.environ.setdefault("PAPERS_DIR", "/tmp/researchmind-test-papers")
os.environ["SEARXNG_URL"] = "http://searx.test"
os.environ["MAX_SEARCH_ITERATIONS"] = "2"
os.environ["MAX_PAPERS_PER_SESSION"] = "50"
os.environ["LLM_MAX_RETRIES"] = "1"
os.environ["CORS_ORIGINS"] = ""
os.environ["AUTOPILOT_ENABLED"] = "false"
os.environ["OPENALEX_ENABLED"] = "false"
os.environ["OPENALEX_API_KEY"] = ""       # never use the real key from .env in tests
os.environ["NARRATIVE_EVERY"] = "2"

import httpx  # noqa: E402
import pytest  # noqa: E402

from src import db  # noqa: E402
from src.research.store import ResearchStore, set_store  # noqa: E402

RESEARCH_TABLES = ("topics", "events", "gap_history", "gap_evidence", "gaps", "paper_relations", "paper_summaries",
                   "papers", "search_queries", "sessions")


@pytest.fixture(scope="session", autouse=True)
def database():
    db.init_schema()
    yield
    schema = os.environ["DB_SCHEMA"]
    try:
        with db.connection() as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        db.close_pool()


def truncate(*tables: str) -> None:
    with db.connection() as conn:
        conn.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")


class FakeProvider:
    """Deterministic stand-in for an LLM. Dispatches on the prompt type."""

    name = "fake"
    model = "fake-model"

    def __init__(self, fail_on: set[str] | None = None, malformed_first: set[str] | None = None):
        self.calls: Counter = Counter()
        self.fail_on = fail_on or set()
        self.malformed_first = malformed_first or set()
        self.prompts: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, system=None, max_tokens=None, json_mode=False) -> str:
        kind = self._kind(prompt)
        self.calls[kind] += 1
        self.prompts.append((kind, prompt))
        if kind in self.fail_on:
            from src.llm.llm import LLMError
            raise LLMError(f"{kind} failed")
        if kind in self.malformed_first and self.calls[kind] == 1:
            return "Sure! Here is the answer: {not json"
        return getattr(self, f"_{kind}")(prompt)

    @staticmethod
    def _kind(prompt: str) -> str:
        markers = [
            ("You help plan an academic literature search", "topic"),
            ("You write search queries", "queries"),
            ("Decide whether each search result", "relevance"),
            ("Extract the key facts from this excerpt", "notes"),
            ("You summarise a research paper", "summary"),
            ("Compare a new paper", "compare"),
            ("You maintain the list of research gaps", "gaps"),
            ("running synthesis", "synthesis"),
            ("narrative part of a living literature review", "narrative"),
            ("check whether a proposed research gap has already been addressed", "gapqueries"),
            ("checking whether a research gap is still open", "gapjudge"),
            ("stress-testing a student's research idea", "specificity"),
            ("running a duplication test on a research idea", "duplication"),
        ]
        for marker, kind in markers:
            if marker in prompt:
                return kind
        return "unknown"

    def _topic(self, prompt):
        return json.dumps({"field": "Software Engineering", "summary": "LLMs applied to software testing",
                           "key_concepts": ["large language models", "software testing"],
                           "subtopics": ["test generation"], "synonyms": ["LLM"],
                           "designs": ["empirical evaluation on open-source projects"],
                           "key_quantities": ["line and branch coverage", "mutation score"],
                           "quality_signals": ["comparison with EvoSuite baseline", "multiple projects"]})

    def _queries(self, prompt):
        if "Papers found so far" in prompt:
            return json.dumps({"queries": ["llm test oracle generation"]})
        return json.dumps({"queries": ["llm unit test generation", "large language models software testing"]})

    def _relevance(self, prompt):
        items = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)
        out = []
        for idx, title in items:
            relevant = "cooking" not in title.lower()
            out.append({"index": int(idx), "relevant": relevant, "score": 0.9 if relevant else 0.1,
                        "reason": "About LLM-based testing" if relevant else "Unrelated subject"})
        return json.dumps({"assessments": out})

    def _notes(self, prompt):
        return "- The paper studies LLM test generation.\n- Evaluated on 10 Java projects."

    def _summary(self, prompt):
        ref = re.search(r"Paper (P\d+):", prompt).group(1)
        return json.dumps({
            "research_problem": [{"text": f"{ref} studies automated unit test generation with LLMs.",
                                  "inferred": False}],
            "methodology": [{"text": "Prompting an LLM to generate unit tests.", "inferred": False}],
            "dataset": [{"text": "Open-source Java projects.", "inferred": False}],
            "key_findings": [{"text": "Generated tests reach moderate coverage.", "inferred": False}],
            "limitations": [{"text": "Only evaluated on Java benchmarks, not industrial code.", "inferred": False}],
            "future_work": ["Evaluate on other languages."],
            "relevance": [{"text": "Directly about LLMs for testing.", "inferred": True}],
        })

    def _compare(self, prompt):
        existing = prompt.split("<existing_papers>", 1)[1].split("</existing_papers>", 1)[0]
        refs = re.findall(r"^(P\d+):", existing, re.MULTILINE)
        return json.dumps({"relations": [
            {"paper_id": r, "relationship": "complements", "similarities": "Both generate tests with LLMs",
             "differences": "Different prompting", "contradiction": ""} for r in refs]})

    def _gaps(self, prompt):
        ref = re.search(r"<new_paper>\n(P\d+):", prompt).group(1)
        current = prompt.split("<current_gaps>", 1)[1].split("</current_gaps>", 1)[0]
        if "G1 |" in current:
            return json.dumps({"operations": [
                {"action": "add_evidence", "gap_id": "G1", "evidence": [
                    {"paper_id": ref, "relationship": "supports", "basis": "stated_limitation",
                     "evidence": "Also evaluates only on open-source Java benchmarks."}]},
                {"action": "create", "title": "More research is needed", "description": "Generic",
                 "evidence": [{"paper_id": ref, "evidence": "Says future work is needed."}]},
                {"action": "create", "title": "Missing evaluation of fabricated work",
                 "description": "Cites a paper that does not exist",
                 "evidence": [{"paper_id": "P99", "evidence": "This paper does not exist at all."}]},
            ]})
        return json.dumps({"operations": [
            {"action": "create", "title": "Limited evaluation on industrial codebases",
             "description": "Studies evaluate LLM test generation on open-source benchmarks only.",
             "gap_type": "evaluation", "category": "potential_gap",
             "evidence": [{"paper_id": ref, "relationship": "supports", "basis": "stated_limitation",
                           "evidence": "Only evaluated on Java benchmarks, not industrial code."}]}]})

    def _gapqueries(self, prompt):
        return json.dumps({"queries": ["industrial codebase llm test generation evaluation", "llm unit tests proprietary software"]})

    def _gapjudge(self, prompt):
        papers = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)
        hit = next((int(i) for i, t in papers if "industrial" in t.lower()), None)
        if hit:
            return json.dumps({"verdict": "partially_addressed", "note": "One industrial evaluation exists.",
                               "addressing": [{"index": hit, "how": "Evaluates LLM test generation at a company."}]})
        return json.dumps({"verdict": "open", "addressing": [], "note": "No paper evaluates industrial code."})

    def _specificity(self, prompt):
        idea = prompt.split("<idea>", 1)[1].split("</idea>", 1)[0].strip().lower()
        if len(idea.split()) <= 4:
            return json.dumps({"level": "field", "restated": idea, "queries": [idea],
                               "missing": [{"knob": "population", "question": "Which population?"},
                                           {"knob": "outcome", "question": "Which outcome?"}]})
        return json.dumps({"level": "topic", "restated": idea, "missing": [],
                           "queries": ["llm unit test generation java", "large language models software testing"]})

    def _duplication(self, prompt):
        papers = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)
        neighbours = [{"index": int(i), "closeness": "same" if "unit test generation" in t.lower() else "related",
                       "overlap": "Same question", "difference": "Different data"} for i, t in papers]
        same = any(n["closeness"] == "same" for n in neighbours)
        return json.dumps({"neighbours": neighbours, "verdict": "duplicated" if same else "clear_lane",
                           "reason": "Paper [1] already did this." if same else "Nothing close found.",
                           "pivot_questions": ["Which population has not been studied?", "Could a different design add value?"]})

    def _narrative(self, prompt):
        refs = sorted(set(re.findall(r"^(P\d+):", prompt, re.MULTILINE)))
        return (f"## Executive summary\nLLM test generation is promising [{', '.join(refs)}].\n\n"
                "## Research landscape\nPrompt-based generation dominates.\n\n## Future directions\n- Address G1.")

    def _synthesis(self, prompt):
        refs = sorted(set(re.findall(r"\b(P\d+)\b", prompt)))
        return json.dumps({
            "overview": "LLM-based test generation is promising but evaluated narrowly.",
            "known_findings": [{"statement": "LLM-generated tests reach moderate coverage.", "paper_ids": refs}],
            "common_methods": [{"statement": "Prompt-based generation", "paper_ids": refs[:1]}],
            "contradictory_findings": [],
            "recurring_limitations": [{"statement": "Java-only benchmarks", "paper_ids": refs + ["P77"]}],
            "under_researched": [{"statement": "Industrial settings", "paper_ids": []}],
            "research_questions": ["How do LLM tests perform on industrial code?"],
        })


ABSTRACT_TAIL = (" We report results on several open-source projects, compare against baselines and discuss"
                 " threats to validity and directions for practitioners.")


def searx_result(title, url, content="", **extra):
    content = content + ABSTRACT_TAIL if content else content
    return {"title": title, "url": url, "content": content, "engine": "arxiv", "category": "science", **extra}


SEARX_RESULTS = {
    "default": [
        searx_result("Large Language Models for Unit Test Generation: An Empirical Study",
                     "https://arxiv.org/abs/2305.00001v2",
                     "We study large language models for software testing, generating unit tests for Java.",
                     authors=["Alice Smith", "Bob Lee"], publishedDate="2023-05-01"),
        searx_result("Large Language Models for Unit Test Generation: An Empirical Study",
                     "https://www.semanticscholar.org/paper/abc123",
                     "We study large language models for software testing, generating unit tests for Java.",
                     doi="", authors=["Alice Smith"]),
        searx_result("ChatGPT versus Search-Based Software Testing Tools",
                     "https://doi.org/10.1145/1234567.7654321",
                     "A comparison of large language models and search-based software testing for test generation.",
                     doi="10.1145/1234567.7654321", authors=["Carol Chen"], publishedDate="2024-01-10"),
        searx_result("Cooking with Large Language Models: Recipes for Software Testing Kitchens",
                     "https://example-journal.org/cooking",
                     "Large language models software testing kitchen recipes cooking.", doi="10.9999/cook.1"),
        searx_result("Quantum annealing for protein folding", "https://example.org/quantum",
                     "Unrelated physics result."),
    ],
    "llm test oracle generation": [
        searx_result("Test Oracle Generation with Large Language Models",
                     "https://arxiv.org/abs/2401.00002", "Large language models for software testing oracles.",
                     authors=["Dan Wu"]),
    ],
}


def make_searx_transport(results=SEARX_RESULTS, fail=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("connection refused", request=request)
        q = request.url.params.get("q", "")
        page = int(request.url.params.get("pageno", "1"))
        if page > 1:
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": results.get(q, results["default"])})
    return httpx.MockTransport(handler)


def not_found_transport():
    return httpx.MockTransport(lambda request: httpx.Response(404, text="not found"))


@pytest.fixture
def store():
    truncate(*RESEARCH_TABLES)
    with db.connection() as conn:
        conn.execute("UPDATE autopilot SET enabled = TRUE, active_topic = NULL, requested_topic = NULL, state = 'idle', "
                     "state_message = '', current_paper_id = NULL, next_search_at = NULL, paused_until = NULL")
    s = ResearchStore()
    set_store(s)
    yield s
    set_store(None)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture(autouse=True)
def openalex_quota():
    from src.research.enrich import reset_quota_state
    reset_quota_state()
    yield
    reset_quota_state()


@pytest.fixture(autouse=True)
def allow_hosts(monkeypatch):
    # Tests use fake hostnames with mock transports; skip DNS-based SSRF checks except where tested.
    import src.research.content as content
    monkeypatch.setattr(content, "_is_public_host", lambda host: host not in ("localhost", "127.0.0.1"))


@pytest.fixture(autouse=True)
def online(monkeypatch):
    """Tests never probe the real internet; individual tests simulate an outage explicitly."""
    import src.llm.llm as llm_mod
    monkeypatch.setattr(llm_mod, "internet_reachable", lambda timeout=5.0: True)
