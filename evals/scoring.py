"""Pure scoring functions for the prompt evaluation (no LLM calls, unit-tested)."""
import re

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?![\d])|\d+(?:\.\d+)?)\s?(%|×|x\b)?")


def _canon(value: str) -> str:
    value = value.replace(",", "")
    return value.rstrip("0").rstrip(".") if "." in value else value


def norm(text: str) -> str:
    return " ".join((text or "").lower().replace("‑", "-").replace("–", "-").split())


def numbers(text: str) -> set[str]:
    """Numbers worth checking: decimals, percentages, multipliers and integers ≥ 10 (years included)."""
    out = set()
    for raw, unit in _NUM.findall(text or ""):
        value = raw.replace(",", "")
        if "." in value or unit or (value.isdigit() and int(value) >= 10):
            out.add(value.rstrip("0").rstrip(".") if "." in value else value)
    return out


def all_numbers(text: str) -> set[str]:
    """Every number in a text, including small integers (for checking what the source contains)."""
    return {_canon(raw) for raw, _unit in _NUM.findall(text or "")}


def unsupported_numbers(statement: str, source: str) -> list[str]:
    """Numbers in a statement that never appear in the source text (candidate hallucinations)."""
    src = all_numbers(source)
    return sorted(n for n in numbers(statement) if n not in src)


def matches(text: str, alternatives: list[str]) -> bool:
    t = norm(text)
    return any(norm(a) in t for a in alternatives)


def fact_recall(texts: list[str], facts: list[dict]) -> tuple[float, list[str]]:
    """Share of gold facts found anywhere in `texts`; returns (recall, missing fact ids)."""
    joined = " ".join(texts)
    missing = [f["id"] for f in facts if not matches(joined, f["match"])]
    return (1 - len(missing) / len(facts)) if facts else 1.0, missing


def summary_scores(sections: dict, source: str, facts: list[dict]) -> dict:
    statements = [s for items in sections.values() for s in items or []]
    texts = [s["text"] for s in statements]
    recall, missing = fact_recall(texts, facts)
    unsupported = {s["text"]: unsupported_numbers(s["text"], source) for s in statements}
    unsupported = {k: v for k, v in unsupported.items() if v}
    return {
        "statements": len(statements),
        "empty": not (sections.get("research_problem") or sections.get("methodology") or sections.get("key_findings")),
        "fact_recall": round(recall, 3), "missing_facts": missing,
        "unsupported_number_statements": len(unsupported), "unsupported_examples": dict(list(unsupported.items())[:3]),
        "inferred_with_numbers": sum(1 for s in statements if s.get("inferred") and numbers(s["text"])),
    }


def answer_scores(answer: str, expect: list[list[str]], expect_papers: list[str], citation_check: dict) -> dict:
    hit = [any(matches(answer, [alt]) for alt in group) for group in expect]
    cited = {p for p in expect_papers if p in (answer or "")}
    total = citation_check.get("total", 0)
    return {
        "facts_found": sum(hit), "facts_expected": len(expect), "correct": all(hit),
        "cites_expected_paper": bool(cited) or not expect_papers,
        "citations": total, "citations_verified": citation_check.get("verified", 0),
        "citation_validity": round(citation_check.get("verified", 0) / total, 3) if total else None,
        "unverified": citation_check.get("unverified", [])[:5],
    }


def relevance_scores(items: list[dict], predictions: dict[int, bool]) -> dict:
    tp = fp = tn = fn = 0
    mistakes = []
    for i, item in enumerate(items):
        pred = predictions.get(i)
        if pred is None:
            mistakes.append({"title": item["title"], "error": "no assessment"})
            fn += item["label"]
            tn += not item["label"]
            continue
        if pred and item["label"]:
            tp += 1
        elif pred and not item["label"]:
            fp += 1
            mistakes.append({"title": item["title"], "error": "selected but should be rejected", "why": item.get("why", "")})
        elif not pred and item["label"]:
            fn += 1
            mistakes.append({"title": item["title"], "error": "rejected but should be selected"})
        else:
            tn += 1
    n = len(items) or 1
    return {"accuracy": round((tp + tn) / n, 3), "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "mistakes": mistakes}
