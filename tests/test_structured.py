import pytest

from src.llm import structured
from src.llm.structured import StructuredOutputError, extract_json, invoke_structured
from src.research.models import GapOperations, PaperSummaryOut, RelevanceBatch, SynthesisOut


class ScriptedProvider:
    name = "scripted"
    model = "m"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, *, system=None, max_tokens=None, json_mode=False):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Result: {"a": {"b": "x}"}} trailing prose') == {"a": {"b": "x}"}}
    assert extract_json('{"a": [1, 2,], }') == {"a": [1, 2]}                  # trailing commas
    assert extract_json('{"queries": ["a", "b"') == {"queries": ["a", "b"]}     # truncated output
    with pytest.raises(ValueError):
        extract_json("no json at all")
    with pytest.raises(ValueError):
        extract_json("")


def test_invoke_structured_retries_with_error_then_succeeds():
    p = ScriptedProvider(["I think the answer is yes", '{"assessments": [{"index": 1, "relevant": "yes", "score": 4}]}'])
    out = invoke_structured("PROMPT", RelevanceBatch, provider=p, retries=1, use_cache=False)
    assert out.assessments[0].relevant is True
    assert out.assessments[0].score == pytest.approx(0.8)  # 1-5 scale normalised
    assert "could not be used" in p.prompts[1]


def test_invoke_structured_raises_after_retries():
    p = ScriptedProvider(["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        invoke_structured("PROMPT", RelevanceBatch, provider=p, retries=1, use_cache=False)


def test_cache_stores_only_validated_output(monkeypatch):
    from tests.conftest import truncate
    truncate("llm_cache")
    monkeypatch.setattr(structured, "_cache", structured.PostgresCache())
    bad = ScriptedProvider(["garbage", "garbage"])
    with pytest.raises(StructuredOutputError):
        invoke_structured("P1", RelevanceBatch, provider=bad, retries=1)
    good = ScriptedProvider(['{"assessments": []}'])
    invoke_structured("P1", RelevanceBatch, provider=good, retries=0)
    # Second call is served from cache: no provider call needed.
    empty = ScriptedProvider([])
    assert invoke_structured("P1", RelevanceBatch, provider=empty).assessments == []


def test_summary_schema_coerces_loose_output():
    s = PaperSummaryOut.model_validate({
        "research_problem": "A plain string",
        "key_findings": ["Finding one", {"text": "Finding two", "inferred": True}, {"statement": "Three"}],
        "limitations": "Not specified",
        "dataset": None,
    })
    assert [x.text for x in s.research_problem] == ["A plain string"]
    assert [x.inferred for x in s.key_findings] == [False, True, False]
    assert s.limitations == [] and s.dataset == []
    assert not s.is_empty()


def test_gap_operations_skip_malformed_and_normalise_vocab():
    ops = GapOperations.model_validate({"operations": [
        {"action": "strengthen", "gap_id": "G1", "evidence": {"paper_id": "P1", "evidence": "x" * 20,
                                                              "relationship": "Supports", "basis": "limitation"}},
        {"action": "explode"},
        {"action": "create", "title": "T", "gap_type": "Population / Domain", "category": "open question"},
    ]})
    assert [o.action for o in ops.operations] == ["add_evidence", "create"]
    assert ops.operations[0].evidence[0].relationship == "supports"
    assert ops.operations[0].evidence[0].basis == "stated_limitation"
    assert ops.operations[1].gap_type == "population_domain"
    assert ops.operations[1].category == "open_question"


def test_synthesis_schema_accepts_strings():
    s = SynthesisOut.model_validate({"overview": ["a", "b"], "known_findings": ["x"],
                                     "research_questions": [{"text": "q?"}]})
    assert s.overview == "a b"
    assert s.known_findings[0].statement == "x"
    assert s.research_questions[0].statement == "q?"


def test_check_rejects_empty_but_valid_output_and_retries():
    p = ScriptedProvider(['{}', '{"research_problem": [{"text": "Real problem", "inferred": false}]}'])
    out = invoke_structured("PROMPT", PaperSummaryOut, provider=p, retries=1, use_cache=False,
                            check=lambda o: "summary is empty" if o.is_empty() else None)
    assert out.research_problem[0].text == "Real problem"
    assert "summary is empty" in p.prompts[1]


def test_cached_output_must_pass_check(monkeypatch):
    from tests.conftest import truncate
    truncate("llm_cache")
    monkeypatch.setattr(structured, "_cache", structured.PostgresCache())
    invoke_structured("P-empty", PaperSummaryOut, provider=ScriptedProvider(["{}"]), retries=0)  # cached, no check
    p = ScriptedProvider(['{"research_problem": ["Real"]}'])
    out = invoke_structured("P-empty", PaperSummaryOut, provider=p, retries=0,
                            check=lambda o: "empty" if o.is_empty() else None)
    assert out.research_problem[0].text == "Real" and len(p.prompts) == 1


def test_summary_schema_unwraps_nested_and_variant_keys():
    s = PaperSummaryOut.model_validate({"summary": {"Research Problem": ["P"], "Key-Findings": ["F"], "Methods": ["M"]}})
    assert [x.text for x in s.research_problem] == ["P"]
    assert [x.text for x in s.key_findings] == ["F"] and [x.text for x in s.methodology] == ["M"]
    assert PaperSummaryOut.model_validate({"P12": {"results": ["R"]}}).key_findings[0].text == "R"
