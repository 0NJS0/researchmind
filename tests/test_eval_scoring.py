from evals.scoring import answer_scores, fact_recall, numbers, relevance_scores, summary_scores, unsupported_numbers


def test_numbers_and_unsupported():
    assert numbers("accuracy of 96.39% and 2.00% FPR, 60× smaller, 13 benchmarks, 3 models, 1,750 prompts") == \
        {"96.39", "2", "60", "13", "1750"}
    assert unsupported_numbers("F1 rose from 0.61 to 0.80", "F1-score from 0.61 to 0.75") == ["0.8"]
    assert numbers("sub-dialogue M(900, 1000, 1100)") == {"900", "1000", "1100"}          # a list, not 900,100
    assert unsupported_numbers("Europe 57%, Australia 6%, mixed methods 9%", "Europe 91 57 Australia 9 6 ... 9 % mixed") == []


def test_fact_recall_and_summary():
    facts = [{"id": "a", "match": ["96.39"]}, {"id": "b", "match": ["DToT", "Decision-Tree-of-Thought"]}]
    assert fact_recall(["It reaches 96.39% accuracy"], facts) == (0.5, ["b"])
    s = summary_scores({"key_findings": [{"text": "Accuracy 96.39% vs 91.2% baseline", "inferred": False}],
                        "research_problem": [], "methodology": []}, "accuracy of 96.39%", facts)
    assert s["unsupported_number_statements"] == 1 and s["fact_recall"] == 0.5 and not s["empty"]


def test_answer_and_relevance():
    a = answer_scores("ToxicDetector reaches 96.39% [Efficient_Detection, p.1]", [["96.39"], ["2.00", "2%"]],
                      ["Efficient_Detection"], {"total": 1, "verified": 1, "unverified": []})
    assert a["facts_found"] == 1 and not a["correct"] and a["cites_expected_paper"] and a["citation_validity"] == 1.0
    items = [{"title": "a", "label": True}, {"title": "b", "label": False}, {"title": "c", "label": True}]
    r = relevance_scores(items, {0: True, 1: True, 2: False})
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["accuracy"] == round(1 / 3, 3) and len(r["mistakes"]) == 2
