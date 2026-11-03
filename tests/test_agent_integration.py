"""End-to-end agent runs with a fake LLM, a mocked SearXNG and mocked paper hosts."""
from src.research.agent import ResearchAgent
from src.research.searxng import SearXNGClient
from tests.conftest import FakeProvider, make_searx_transport, not_found_transport


def make_agent(store, session_id, provider=None, searx_fail=False):
    return ResearchAgent(store, session_id, provider=provider or FakeProvider(),
                         searx=SearXNGClient("http://searx.test", transport=make_searx_transport(fail=searx_fail)),
                         fetch_transport=not_found_transport())


def test_full_research_session(store):
    s = store.create_session("LLMs for software testing")
    provider = FakeProvider()
    make_agent(store, s["id"], provider).run_research(iterations=2)

    session = store.get_session(s["id"])
    assert session["status"] == "completed", session["status_message"]
    assert session["topic_analysis"]["field"] == "Software Engineering"

    # Topic → search: the user's own topic is always searched, plus generated and follow-up queries.
    queries = [q["query"] for q in store.list_queries(s["id"])]
    assert queries[0] == "LLMs for software testing"
    assert "llm test oracle generation" in queries        # iteration 2 (expansion)
    assert {q["iteration"] for q in store.list_queries(s["id"])} == {1, 2}

    # Search → paper ingestion: duplicates collapsed, irrelevant results rejected.
    selected = store.list_papers(s["id"], selection="selected")
    titles = [p["title"] for p in selected]
    assert titles.count("Large Language Models for Unit Test Generation: An Empirical Study") == 1
    assert "ChatGPT versus Search-Based Software Testing Tools" in titles
    assert "Test Oracle Generation with Large Language Models" in titles
    assert [p["ref"] for p in selected] == ["P1", "P2", "P3"]
    rejected = {p["title"]: p for p in store.list_papers(s["id"], selection="rejected")}
    assert rejected["Quantum annealing for protein folding"]["relevance_method"] == "lexical"
    assert rejected["Cooking with Large Language Models: Recipes for Software Testing Kitchens"][
        "relevance_method"] == "llm"
    assert all(p["relevance_reason"] for p in selected)

    # Paper → summary (abstract-level because the paper hosts return 404).
    for p in selected:
        assert p["processing_status"] == "done"
        assert p["content_status"] == "abstract"
        summary = store.get_summary(p["id"])
        assert summary["analysis_level"] == "abstract"
        assert summary["sections"]["limitations"][0]["inferred"] is False
        assert summary["sections"]["relevance"][0]["inferred"] is True

    # Multiple papers → evolving gap analysis with traceable evidence.
    gaps = store.list_gaps(s["id"])
    assert len(gaps) == 1, [g["title"] for g in gaps]      # generic + fabricated gaps rejected
    gap = gaps[0]
    assert gap["paper_count"] == 3 and gap["confidence"] == "high"
    history = [h["action"] for h in store.gap_history(gap["id"])]
    assert history[0] == "created" and "confidence" in history
    assert {e["paper_ref"] for e in store.evidence_for_gap(gap["id"])} == {"P1", "P2", "P3"}

    # Comparison relations recorded against earlier papers.
    p3 = next(p for p in selected if p["ref"] == "P3")
    assert {r["related_ref"] for r in store.relations_for_paper(p3["id"])} == {"P1", "P2"}

    # Synthesis: unknown paper IDs dropped, strength derived from evidence count.
    synthesis = session["synthesis"]
    assert synthesis["known_findings"][0]["strength"] == "established"
    assert "P77" not in synthesis["recurring_limitations"][0]["paper_refs"]

    # Every paper summarised exactly once; no re-summarisation when new papers arrive.
    assert provider.calls["summary"] == 3
    events = [e["message"] for e in store.list_events(s["id"])]
    assert any(m.startswith("Generated") and "search queries" in m for m in events)
    assert any("removed" in m and "duplicates" in m for m in events)
    assert any(m.startswith("Completed") for m in events)


def test_continue_search_does_not_resummarize(store):
    s = store.create_session("LLMs for software testing")
    first = FakeProvider()
    make_agent(store, s["id"], first).run_research(iterations=1)
    assert first.calls["summary"] == 2

    second = FakeProvider()
    make_agent(store, s["id"], second).run_research(iterations=1, queries=["llm test oracle generation"])
    assert second.calls["topic"] == 0
    assert second.calls["summary"] == 1                    # only the newly found paper
    assert len(store.list_papers(s["id"], selection="selected")) == 3


def test_search_unavailable_fails_gracefully(store):
    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"], searx_fail=True).run_research(iterations=2)
    session = store.get_session(s["id"])
    assert session["status"] == "failed"
    assert session["status_message"] == "Search service is currently unavailable."
    assert all(q["status"] == "failed" for q in store.list_queries(s["id"]))


def test_llm_failures_are_isolated(store):
    s = store.create_session("LLMs for software testing")
    provider = FakeProvider(fail_on={"relevance", "gaps"}, malformed_first={"summary"})
    make_agent(store, s["id"], provider).run_research(iterations=1)
    session = store.get_session(s["id"])
    assert session["status"] == "completed"
    selected = store.list_papers(s["id"], selection="selected")
    assert selected and all(p["relevance_method"] == "lexical" for p in selected)   # keyword fallback
    assert all(store.get_summary(p["id"]) for p in selected)                     # malformed reply retried
    assert store.list_gaps(s["id"]) == []
    events = [e["message"] for e in store.list_events(s["id"])]
    assert any("gap analysis failed" in m for m in events)


def test_summary_failure_marks_paper_failed_and_continues(store):
    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"], FakeProvider(fail_on={"summary"})).run_research(iterations=1)
    papers = store.list_papers(s["id"], selection="selected")
    assert papers and all(p["processing_status"] == "failed" for p in papers)
    assert store.get_session(s["id"])["status"] == "completed"


def test_gap_rebuild_replays_existing_summaries(store):
    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"]).run_research(iterations=1)
    before = store.list_gaps(s["id"])
    provider = FakeProvider()
    make_agent(store, s["id"], provider).run_gap_rebuild()
    after = store.list_gaps(s["id"])
    assert provider.calls["summary"] == 0
    assert [g["title"] for g in after] == [g["title"] for g in before]
    assert after[0]["ref"] == "G1"


def test_cancellation(store):
    s = store.create_session("LLMs for software testing")
    agent = make_agent(store, s["id"])
    agent.cancel_event.set()
    agent.run_research(iterations=1)
    assert store.get_session(s["id"])["status"] == "cancelled"


def test_failed_gap_step_is_retried_without_duplicating_evidence(store):
    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"], FakeProvider(fail_on={"synthesis"})).run_research(iterations=1)
    papers = store.list_papers(s["id"], selection="selected")
    assert papers and all(p["processing_status"] == "analysis_failed" for p in papers)
    gap = store.list_gaps(s["id"])[0]
    evidence_before = len(store.evidence_for_gap(gap["id"]))

    provider = FakeProvider()
    make_agent(store, s["id"], provider).run_processing()
    papers = store.list_papers(s["id"], selection="selected")
    assert all(p["processing_status"] == "done" for p in papers)
    assert provider.calls["summary"] == 0                  # summaries are reused
    gaps = store.list_gaps(s["id"])
    assert len(gaps) == 1
    assert len(store.evidence_for_gap(gaps[0]["id"])) == evidence_before  # replaced, not duplicated


def test_gap_rebuild_is_all_or_nothing(store):
    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"]).run_research(iterations=1)
    before = [(g["ref"], g["title"], g["evidence_count"]) for g in store.list_gaps(s["id"])]

    class CancelAfterFirstGapCall(FakeProvider):
        def complete(self, prompt, **kw):
            out = super().complete(prompt, **kw)
            if self._kind(prompt) == "gaps":
                agent.cancel_event.set()
            return out

    agent = make_agent(store, s["id"], CancelAfterFirstGapCall())
    agent.run_gap_rebuild()
    assert store.get_session(s["id"])["status"] == "cancelled"
    after = [(g["ref"], g["title"], g["evidence_count"]) for g in store.list_gaps(s["id"])]
    assert after == before                                 # rolled back: old gaps intact


def test_quota_exhaustion_pauses_the_session(store):
    from src.llm.llm import QuotaExhausted

    class QuotaProvider(FakeProvider):
        def complete(self, prompt, **kw):
            if self._kind(prompt) == "summary":
                raise QuotaExhausted("daily limit reached")
            return super().complete(prompt, **kw)

    s = store.create_session("LLMs for software testing")
    make_agent(store, s["id"], QuotaProvider()).run_research(iterations=2)
    session = store.get_session(s["id"])
    assert session["status"] == "paused"
    assert "daily limit" in session["status_message"]
    papers = store.list_papers(s["id"], selection="selected")
    assert papers and all(p["processing_status"] == "pending" for p in papers)  # resumable later


def test_leftover_candidates_are_screened_on_the_next_run(store):
    s = store.create_session("LLMs for software testing")
    store.add_paper(s["id"], {"title": "Large language models for software test generation at scale",
                              "abstract": "We study large language models for software testing " * 5,
                              "selection_status": "candidate"})
    make_agent(store, s["id"]).run_research(iterations=1)
    assert store.list_papers(s["id"], selection="candidate") == []


def test_nul_bytes_from_scraped_content_are_stored_safely(store):
    s = store.create_session("LLMs for software testing")
    paper = store.add_paper(s["id"], {"title": "Bad\x00title", "abstract": "x\x00y", "authors": ["A\x00"]}, assign_ref=True)
    store.update_paper(paper["id"], raw_content="full\x00text")
    store.save_summary(paper["id"], "full_text", {"key_findings": [{"text": "a\x00b", "inferred": False}]}, "m")
    store.add_event(s["id"], "info", "msg\x00", {"k": "v\x00"})
    stored = store.get_paper(paper["id"])
    assert stored["title"] == "Badtitle" and stored["raw_content"] == "fulltext" and stored["authors"] == ["A"]
    assert store.get_summary(paper["id"])["sections"]["key_findings"][0]["text"] == "ab"


def test_field_profile_adapts_later_prompts(store):
    s = store.create_session("LLMs for software testing")
    provider = FakeProvider()
    make_agent(store, s["id"], provider).run_research(iterations=1)
    analysis = store.get_session(s["id"])["topic_analysis"]
    assert analysis["key_quantities"] == ["line and branch coverage", "mutation score"]
    summary_prompt = next(p for kind, p in provider.prompts if kind == "summary")
    gap_prompt = next(p for kind, p in provider.prompts if kind == "gaps")
    for prompt in (summary_prompt, gap_prompt):
        assert "Field conventions for Software Engineering" in prompt and "mutation score" in prompt


def test_field_profile_is_backfilled_for_older_topics(store):
    s = store.create_session("LLMs for software testing")
    store.update_session(s["id"], topic_analysis={"field": "Software Engineering", "key_concepts": ["llm"]})
    provider = FakeProvider()
    agent = make_agent(store, s["id"], provider)
    agent.ensure_field_profile()
    assert store.get_session(s["id"])["topic_analysis"]["quality_signals"]
    assert store.get_session(s["id"])["topic_analysis"]["key_concepts"] == ["llm"]   # existing analysis kept
    agent.ensure_field_profile()
    assert provider.calls["topic"] == 1                                              # only once


def test_prompts_carry_examples_from_several_fields():
    from src.research import prompts
    for marker in ("macro-F1", "95% CI", "difference-in-differences"):
        assert marker in prompts.PAPER_SUMMARY_PROMPT
    assert "HR 0.76" in prompts.PAPER_NOTES_PROMPT and "β = 0.12" in prompts.PAPER_NOTES_PROMPT
