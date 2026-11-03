"""Topic Validator and gap duplication test, with a fake LLM, mocked SearXNG and a mocked OpenAlex."""
import time
from datetime import date

import httpx
import pytest

from src.research import validator
from src.research.enrich import OpenAlexClient
from src.research.literature import LiteratureSearch, abstract_from_index, momentum, name_papers
from src.research.searxng import SearXNGClient
from src.research.verify import next_gap_to_verify, verify_gap
from tests.conftest import FakeProvider, make_searx_transport, truncate

TODAY = date(2026, 9, 1)


def inverted(text):
    idx = {}
    for i, w in enumerate(text.split()):
        idx.setdefault(w, []).append(i)
    return idx


OA_WORKS = [
    {"title": "Large Language Models for Unit Test Generation: An Empirical Study", "doi": "https://doi.org/10.1/llmtests",
     "publication_year": 2024, "cited_by_count": 120, "type": "article",
     "primary_location": {"source": {"display_name": "ICSE", "type": "conference"}},
     "abstract_inverted_index": inverted("We evaluate LLM unit test generation on Java projects"),
     "counts_by_year": [{"year": 2025, "cited_by_count": 60}, {"year": 2024, "cited_by_count": 40},
                        {"year": 2023, "cited_by_count": 10}]},
    {"title": "An Industrial Evaluation of LLM Test Generation at a Software Company", "doi": "https://doi.org/10.1/industrial",
     "publication_year": 2025, "cited_by_count": 9, "type": "article",
     "primary_location": {"source": {"display_name": "ESEC/FSE", "type": "conference"}},
     "abstract_inverted_index": inverted("We deploy LLM unit test generation on industrial proprietary codebases"),
     "counts_by_year": [{"year": 2025, "cited_by_count": 9}]},
    {"title": "A Systematic Review of Large Language Models in Software Testing", "doi": "https://doi.org/10.1/review",
     "publication_year": 2024, "cited_by_count": 300, "type": "review",
     "primary_location": {"source": {"display_name": "TSE", "type": "journal"}},
     "abstract_inverted_index": inverted("A systematic review of large language models for software testing and test generation"),
     "counts_by_year": [{"year": 2025, "cited_by_count": 200}, {"year": 2024, "cited_by_count": 100}]},
]


def openalex_transport(works=OA_WORKS):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/works":
            return httpx.Response(404)
        if request.url.params.get("group_by") == "publication_year":
            return httpx.Response(200, json={"group_by": [{"key": str(y), "count": 100 + 20 * (y - 2016)}
                                                          for y in range(2016, 2027)]})
        return httpx.Response(200, json={"meta": {"count": 4200}, "results": works})
    return httpx.MockTransport(handler)


def literature(works=OA_WORKS):
    return LiteratureSearch(searx=SearXNGClient("http://searx.test", transport=make_searx_transport()),
                            openalex=OpenAlexClient(transport=openalex_transport(works), enabled=True))


@pytest.fixture
def clean(store):
    truncate("validations")
    yield store


def test_literature_helpers():
    assert abstract_from_index({"world": [1], "hello": [0]}) == "hello world"
    assert momentum({2025: 60, 2024: 40, 2023: 10}, TODAY)["trend"] == "rising"
    assert momentum({2025: 2, 2024: 3, 2023: 30, 2022: 30}, TODAY)["trend"] == "cooling"
    records, stats = literature().search(["llm unit test generation"], per_query=10)
    titles = [r["title"] for r in records]
    assert titles.count("Large Language Models for Unit Test Generation: An Empirical Study") == 1  # merged
    merged = next(r for r in records if r["title"].startswith("Large Language Models for Unit Test"))
    assert merged["citation_count"] == 120 and merged["momentum"]["trend"] in ("rising", "steady")
    assert stats["openalex_total"] == 4200 and stats["searxng_results"] > 0


def test_specific_idea_fails_duplication_test(clean):
    v = validator.create_validation("Evaluate LLM unit test generation on open-source Java projects",
                                    "empirical_quantitative")
    result = validator.run_validation(v["id"], provider=FakeProvider(), literature=literature(), today=TODAY)
    stored = validator.get_validation(v["id"])
    assert stored["status"] == "completed" and stored["result"]["verdict"] == "duplicated"
    assert result["specificity"]["level"] == "topic"
    top = result["neighbours"][0]
    assert top["closeness"] == "same" and top["url"] and top["title"].startswith("Large Language Models for Unit Test")
    assert result["pivot_questions"]
    sat = result["saturation"]
    assert sat["total_matching"] == 4200 and sat["density"] == "dense" and len(sat["per_year"]) == 11
    assert sat["growth_last3_vs_prev3"] > 0
    assert result["rival_momentum"]["close_rivals"] >= 1


def test_broad_idea_is_sent_back_with_questions(clean):
    v = validator.create_validation("LLMs and software testing", "empirical_quantitative")
    provider = FakeProvider()
    result = validator.run_validation(v["id"], provider=provider, literature=literature(), today=TODAY)
    assert result["verdict"] == "too_broad"
    assert result["pivot_questions"] == ["Which population?", "Which outcome?"]
    assert provider.calls["duplication"] == 0          # no duplication test on a field


def test_review_projects_compare_against_existing_reviews(clean):
    v = validator.create_validation("Systematic review of large language models for software test generation",
                                    "systematic_review")
    result = validator.run_validation(v["id"], provider=FakeProvider(), literature=literature(), today=TODAY)
    assert any(n["publication_type"] == "review" for n in result["neighbours"])


def test_validator_api(clean, monkeypatch):
    from fastapi.testclient import TestClient
    from src.api.server import app
    monkeypatch.setattr(validator, "start_validation",
                        lambda vid: validator.run_validation(vid, provider=FakeProvider(), literature=literature()))
    with TestClient(app) as c:
        assert c.post("/api/validator", json={"idea": "too short"}).status_code == 422
        assert c.post("/api/validator", json={"idea": "A long enough research idea here", "project_type": "x"}).status_code == 422
        r = c.post("/api/validator", json={"idea": "Evaluate LLM unit test generation on open-source Java projects",
                                           "project_type": "empirical_quantitative"})
        assert r.status_code == 202
        vid = r.json()["id"]
        body = c.get(f"/api/validator/{vid}").json()
        assert body["status"] == "completed" and body["result"]["verdict"] == "duplicated"
        assert c.get("/api/validator").json()["validations"][0]["verdict"] == "duplicated"
        assert len(c.get("/api/validator/project-types").json()["project_types"]) == len(validator.PROJECT_TYPES)
        assert c.delete(f"/api/validator/{vid}").status_code == 200
        assert c.get(f"/api/validator/{vid}").status_code == 404


# ── gap duplication test ──

def make_gap(store, confidence="medium"):
    s = store.create_session("LLMs for software testing")
    p = store.add_paper(s["id"], {"title": "LLM test generation on Java", "selection_status": "selected"}, assign_ref=True)
    g = store.create_gap(s["id"], {"title": "Limited evaluation on industrial codebases",
                                   "description": "LLM unit test generation is only evaluated on open-source projects.",
                                   "gap_type": "evaluation"})
    store.add_evidence(g["id"], p["id"], "Only open-source Java benchmarks were used.", "supports", "stated_limitation")
    store.update_gap(g["id"], confidence=confidence)
    return s["id"], store.get_gap(g["id"])


def test_gap_check_finds_paper_that_addresses_it_and_queues_it(store):
    sid, gap = make_gap(store)
    assert next_gap_to_verify(store, sid)["id"] == gap["id"]
    result = verify_gap(store, sid, store.list_gaps(sid)[0], "LLMs for software testing",
                        provider=FakeProvider(), literature=literature())
    assert result["verification_status"] == "partially_addressed"
    assert result["verification_refs"][0]["title"].startswith("An Industrial Evaluation")
    stored = store.get_gap(gap["id"])
    assert stored["verified_at"] and stored["verification_note"]
    queued = [p for p in store.list_papers(sid) if p["relevance_method"] == "gap_check"]
    assert len(queued) == 1 and queued[0]["processing_status"] == "pending" and queued[0]["citation_count"] == 9
    assert any(h["action"] == "verified" for h in store.gap_history(gap["id"]))
    assert next_gap_to_verify(store, sid) is None          # not re-checked until the re-check window passes


def test_gap_check_open_when_nothing_addresses_it(store):
    sid, gap = make_gap(store)
    works = [w for w in OA_WORKS if "Industrial" not in w["title"]]
    result = verify_gap(store, sid, store.list_gaps(sid)[0], "LLMs for software testing",
                        provider=FakeProvider(), literature=literature(works))
    assert result["verification_status"] == "verified_open" and result["verification_refs"] == []


def test_low_confidence_gaps_are_not_checked(store):
    sid, _ = make_gap(store, confidence="low")
    assert next_gap_to_verify(store, sid) is None


def test_autopilot_checks_gaps_between_papers(env_autopilot):
    ap, store, sid = env_autopilot
    events = [e["message"] for e in store.list_events(sid)]
    assert any("checked against the wider literature" in m for m in events)
    assert any(g["verification_status"] != "unverified" for g in store.list_gaps(sid, include_merged=True))


@pytest.fixture
def env_autopilot(env):
    """Run the autopilot until the queue is empty; gaps reach medium confidence along the way."""
    from tests.test_autopilot import activate, session_of, upload
    ap, store, root = env["ap"], env["store"], env["root"]
    upload(root, "Testing", "u1", "LLM unit test generation study on Java projects. " * 30)
    activate(store, "Testing")
    for _ in range(12):
        ap.step()
    return ap, store, session_of(store, "Testing")


from tests.test_autopilot import env  # noqa: E402,F401  (fixture reuse)


def test_list_numbers_are_replaced_with_paper_titles():
    cands = [{"title": "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradients", "year": 2024},
             {"title": "ToxicDetector", "year": 2024}]
    out = name_papers("Paper 2 and [1] overlap; Papers 1 and 2 differ from paper 9.", cands)
    assert "“ToxicDetector” (2024)" in out and "“GradSafe: Detecting Jailbreak Prompts for LLMs…” (2024)" in out
    assert "Paper 2" not in out and "[1]" not in out and "paper 9" in out.lower()
    assert "8" not in name_papers("Papers 1, 2, and 2 agree", cands + [{"title": "x"}] * 6)


def test_openalex_api_key_and_daily_limit_handling():
    from src.research import enrich
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if len(seen) == 1:
            return httpx.Response(200, json={"results": []}, headers={"x-ratelimit-limit": "10000", "x-ratelimit-remaining": "9990"})
        return httpx.Response(429, json={"error": "limit"}, headers={"x-ratelimit-remaining": "0"})

    client = OpenAlexClient(transport=httpx.MockTransport(handler), enabled=True, api_key="secret-key")
    client.search_works("toxic prompts")
    assert seen[0]["api_key"] == "secret-key"
    assert enrich.quota_state()["remaining"] == 9990 and enrich.quota_state()["limit"] == 10000
    assert client.search_works("again") is None                     # 429: daily budget spent
    assert enrich.quota_state()["exhausted_until"] is not None
    assert client.search_works("third") is None and len(seen) == 2  # no more requests until the reset
    records, stats = LiteratureSearch(searx=SearXNGClient("http://searx.test", transport=make_searx_transport()),
                                      openalex=client).search(["x"])
    assert stats["openalex_failed"] is True and records              # SearXNG results still come through
