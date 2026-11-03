import pytest

from src.research.gaps import GapEngine, compute_assessment, is_generic
from src.research.models import GapOperations
from src.research.synthesis import validate_synthesis
from src.research.models import SynthesisOut


@pytest.fixture
def session(store):
    s = store.create_session("LLMs for software testing")
    papers = [store.add_paper(s["id"], {"title": f"Paper number {i} about testing", "selection_status": "selected"},
                              assign_ref=True) for i in range(1, 5)]
    return s, papers


def ops(*operations):
    return GapOperations.model_validate({"operations": list(operations)})


def ev(ref, relationship="supports", basis="stated_limitation", text="Evaluated only on open-source benchmarks."):
    return {"paper_id": ref, "relationship": relationship, "basis": basis, "evidence": text}


CREATE = {"action": "create", "title": "Limited evaluation on industrial codebases",
          "description": "Only open-source benchmarks are used.", "gap_type": "evaluation"}


def test_create_gap_with_evidence(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    report = engine.apply(ops({**CREATE, "evidence": [ev("P1")]}), papers[0])
    assert report.created == ["G1"]
    gap = store.list_gaps(s["id"])[0]
    assert gap["confidence"] == "low" and gap["status"] == "open"
    assert gap["evidence_count"] == 1 and gap["related_papers"][0]["ref"] == "P1"
    history = store.gap_history(gap["id"])
    assert history[0]["action"] == "created"


def test_rejects_generic_unknown_papers_and_missing_evidence(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    report = engine.apply(ops(
        {"action": "create", "title": "More research is needed", "evidence": [ev("P1")]},
        {**CREATE, "evidence": [ev("P42")]},                   # fabricated citation
        {**CREATE, "title": "Lack of multilingual datasets", "evidence": []},
        {"action": "add_evidence", "gap_id": "G9", "evidence": [ev("P1")]},
    ), papers[0])
    assert report.created == [] and len(report.rejected) == 4
    assert store.list_gaps(s["id"]) == []


def test_similar_gap_is_merged_instead_of_duplicated(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    engine.apply(ops({**CREATE, "evidence": [ev("P1")]}), papers[0])
    report = engine.apply(ops({**CREATE, "title": "Limited evaluation on industrial code bases",
                               "evidence": [ev("P2")]}), papers[1])
    assert report.created == [] and report.updated == ["G1"]
    gaps = store.list_gaps(s["id"])
    assert len(gaps) == 1 and gaps[0]["paper_count"] == 2


def test_independent_support_strengthens_confidence(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    engine.apply(ops({**CREATE, "evidence": [ev("P1")]}), papers[0])
    engine.apply(ops({"action": "add_evidence", "gap_id": "G1", "evidence": [ev("P2")]}), papers[1])
    assert store.list_gaps(s["id"])[0]["confidence"] == "medium"
    engine.apply(ops({"action": "add_evidence", "gap_id": "G1", "evidence": [ev("P3")]}), papers[2])
    assert store.list_gaps(s["id"])[0]["confidence"] == "high"


def test_contradiction_marks_gap_contested(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    engine.apply(ops({**CREATE, "evidence": [ev("P1"), ev("P2", text="Another open-source-only evaluation.")]}),
                 papers[1])
    engine.apply(ops({"action": "add_evidence", "gap_id": "G1",
                      "evidence": [ev("P3", "contradicts", "result", "Evaluates at an industrial partner.")]}),
                 papers[2])
    gap = store.list_gaps(s["id"])[0]
    assert gap["status"] == "contested"
    assert gap["confidence"] == "low"  # medium downgraded by the contradiction


def test_future_work_alone_stays_low_confidence():
    evidence = [{"paper_id": p, "relationship": "supports", "basis": "future_work"} for p in "abcd"]
    assert compute_assessment(evidence, "open") == ("low", "open")


def test_merge_operation_moves_evidence(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    engine.apply(ops({**CREATE, "evidence": [ev("P1")]}), papers[0])
    engine.apply(ops({"action": "create", "title": "No multilingual evaluation datasets",
                      "gap_type": "dataset", "evidence": [ev("P2", text="Uses English-only prompts and data.")]}),
                 papers[1])
    report = engine.apply(ops({"action": "merge", "gap_ids": ["G1", "G2"],
                               "title": "Narrow evaluation settings"}), papers[2])
    assert report.merged == ["G2"]
    gaps = store.list_gaps(s["id"])
    assert len(gaps) == 1 and gaps[0]["title"] == "Narrow evaluation settings"
    assert gaps[0]["evidence_count"] == 2
    merged = [g for g in store.list_gaps(s["id"], include_merged=True) if g["status"] == "merged"]
    assert merged[0]["merged_into"] == gaps[0]["id"]
    # Operations that reference a merged gap are redirected to the surviving gap.
    engine.apply(ops({"action": "add_evidence", "gap_id": "G2", "evidence": [ev("P3")]}), papers[2])
    assert store.list_gaps(s["id"])[0]["paper_count"] == 3


def test_mark_addressed_requires_evidence_and_reopens_on_new_support(store, session):
    s, papers = session
    engine = GapEngine(store, s["id"])
    engine.apply(ops({**CREATE, "evidence": [ev("P1")]}), papers[0])
    report = engine.apply(ops({"action": "mark_addressed", "gap_id": "G1", "status": "resolved"}), papers[1])
    assert report.rejected and store.list_gaps(s["id"])[0]["status"] == "open"

    engine.apply(ops({"action": "mark_addressed", "gap_id": "G1", "status": "resolved",
                      "evidence": [ev("P2", text="Runs a full industrial case study at a bank.")]}), papers[1])
    gap = store.list_gaps(s["id"])[0]
    assert gap["status"] == "resolved"
    evidence = store.evidence_for_gap(gap["id"])
    assert evidence[-1]["relationship"] == "contradicts"

    engine.apply(ops({"action": "add_evidence", "gap_id": "G1", "evidence": [ev("P3")]}), papers[2])
    assert store.list_gaps(s["id"])[0]["status"] == "contested"


def test_is_generic():
    assert is_generic("More research is needed")
    assert is_generic("Future work")
    assert is_generic("Gap")
    assert not is_generic("No evaluation of LLM test generation on C++ projects")


def test_synthesis_validation_drops_unknown_papers_and_derives_strength():
    out = SynthesisOut.model_validate({
        "overview": "o",
        "known_findings": [{"statement": "A", "paper_ids": ["P1", "P2", "P3"]},
                           {"statement": "B", "paper_ids": ["P1", "P99"]},
                           {"statement": "C", "paper_ids": ["P99"]}],
        "contradictory_findings": [{"statement": "D", "paper_ids": ["P1"]}],
    })
    refs = {"P1", "P2", "P3"}
    result = validate_synthesis(out, refs, {r: r.lower() for r in refs})
    assert [(f["statement"], f["strength"]) for f in result["known_findings"]] == [("A", "established"),
                                                                                    ("B", "emerging")]
    assert result["known_findings"][1]["paper_refs"] == ["P1"]
    assert result["contradictory_findings"][0]["strength"] == "unverified"


def test_synthesis_merge_is_cumulative():
    from src.research.synthesis import merge_synthesis
    refs = {"P1", "P2", "P3"}
    ids = {"P1": "a", "P2": "b", "P3": "c"}
    old = {"overview": "old", "known_findings": [
        {"statement": "LLM tests reach moderate coverage on Java", "paper_refs": ["P1", "P2"], "paper_ids": ["a", "b"]},
        {"statement": "Prompt context improves test quality", "paper_refs": ["P1"], "paper_ids": ["a"]},
        {"statement": "Assertions are often wrong", "paper_refs": ["P2"], "paper_ids": ["b"]}],
        "contradictory_findings": [], "common_methods": [], "recurring_limitations": [], "under_researched": [],
        "research_questions": []}
    new = {"overview": "new", "known_findings": [
        {"statement": "LLM tests reach moderate coverage on Java projects", "paper_refs": ["P3"], "paper_ids": ["c"]}],
        "contradictory_findings": [{"statement": "Assertions are often wrong", "paper_refs": ["P2", "P3"], "paper_ids": ["b", "c"]}],
        "common_methods": [], "recurring_limitations": [], "under_researched": [], "research_questions": []}
    merged = merge_synthesis(old, new, refs, ids)
    kf = {it["statement"]: it for it in merged["known_findings"]}
    assert kf["LLM tests reach moderate coverage on Java projects"]["paper_refs"] == ["P3", "P1", "P2"]   # IDs merged back
    assert kf["LLM tests reach moderate coverage on Java projects"]["strength"] == "established"
    assert "Prompt context improves test quality" in kf                                                  # dropped item carried forward
    assert "Assertions are often wrong" not in kf                                                        # moved, not duplicated
    assert merged["overview"] == "new"
