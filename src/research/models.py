"""Typed data structures for the research agent: normalised papers and LLM output schemas."""
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Domain vocabularies ──

SESSION_STATUSES = ("created", "searching", "processing", "summarizing", "analyzing",
                    "completed", "failed", "cancelled", "paused", "autopilot")
CONTENT_STATUSES = ("pending", "full_text", "abstract", "metadata_only", "unavailable")
GAP_TYPES = ("methodological", "dataset", "evaluation", "population_domain", "temporal",
             "contradiction", "unexplored_combination", "replication", "practical_deployment", "other")
GAP_CATEGORIES = ("potential_gap", "open_question", "contradictory_evidence", "research_opportunity")
GAP_STATUSES = ("open", "contested", "partially_addressed", "resolved", "merged")
EVIDENCE_RELATIONSHIPS = ("supports", "contradicts", "partially_supports", "provides_context")
EVIDENCE_BASES = ("stated_limitation", "future_work", "result", "inferred")
SUMMARY_SECTIONS = (
    "research_problem", "motivation", "research_questions", "methodology", "dataset",
    "key_findings", "contributions", "limitations", "future_work", "relevance",
)


def _pick(value: Any, allowed: tuple[str, ...], default: str) -> str:
    if not isinstance(value, str):
        return default
    v = re.sub(r"[\s\-/]+", "_", value.strip().lower())
    if v in allowed:
        return v
    for option in allowed:  # tolerate near-misses such as "population" or "deployment"
        if v and (v in option or option in v):
            return option
    return default


def _norm_key(key: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ── Normalised search result / paper ──

class NormalizedPaper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = ""
    engines: list[str] = Field(default_factory=list)
    publication_date: str = ""
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    citation_count: int | None = None
    found_by_query: str = ""
    academic_signal: bool = False


# ── LLM output schemas ──

class TopicAnalysis(BaseModel):
    field: str = ""
    summary: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    # Field profile: how this field does and reports research (adapts later prompts to the discipline).
    designs: list[str] = Field(default_factory=list)
    key_quantities: list[str] = Field(default_factory=list)
    quality_signals: list[str] = Field(default_factory=list)

    @field_validator("key_concepts", "subtopics", "synonyms", "designs", "key_quantities", "quality_signals",
                     mode="before")
    @classmethod
    def _strings(cls, v):
        return [str(x).strip() for x in _as_list(v) if str(x).strip()][:12]

    def has_profile(self) -> bool:
        return bool(self.key_quantities or self.quality_signals or self.designs)


class SearchQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)

    @field_validator("queries", mode="before")
    @classmethod
    def _clean(cls, v):
        out = []
        for q in _as_list(v):
            if isinstance(q, dict):
                q = q.get("query") or q.get("text") or ""
            q = " ".join(str(q).split())[:200]
            if q:
                out.append(q)
        return out


class RelevanceAssessment(BaseModel):
    index: int
    relevant: bool = False
    score: float = 0.0
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if v > 1:  # model answered on a 1-5 or 0-10 scale
            v = v / 5 if v <= 5 else v / 10
        return max(0.0, min(1.0, v))

    @field_validator("relevant", mode="before")
    @classmethod
    def _bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "relevant", "1")
        return bool(v)


class RelevanceBatch(BaseModel):
    assessments: list[RelevanceAssessment] = Field(default_factory=list)


class Statement(BaseModel):
    text: str
    inferred: bool = False


def _statements(v) -> list[dict]:
    out = []
    for item in _as_list(v):
        if isinstance(item, str):
            text, inferred = item, False
        elif isinstance(item, dict):
            text = item.get("text") or item.get("statement") or ""
            inferred = bool(item.get("inferred", False))
        else:
            continue
        text = " ".join(str(text).split())
        if text and text.lower().rstrip(".") not in ("not specified", "n/a", "none", "unknown"):
            out.append({"text": text, "inferred": inferred})
    return out


class PaperSummaryOut(BaseModel):
    research_problem: list[Statement] = Field(default_factory=list)
    motivation: list[Statement] = Field(default_factory=list)
    research_questions: list[Statement] = Field(default_factory=list)
    methodology: list[Statement] = Field(default_factory=list)
    dataset: list[Statement] = Field(default_factory=list)
    key_findings: list[Statement] = Field(default_factory=list)
    contributions: list[Statement] = Field(default_factory=list)
    limitations: list[Statement] = Field(default_factory=list)
    future_work: list[Statement] = Field(default_factory=list)
    relevance: list[Statement] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data):
        """Tolerate {"summary": {...}} wrappers and key variants like "Research Problem" / "key-findings"."""
        if not isinstance(data, dict):
            return data
        for _ in range(2):
            if any(_norm_key(k) in SUMMARY_SECTIONS for k in data):
                break
            inner = [v for v in data.values() if isinstance(v, dict)]
            if len(inner) == 1:
                data = inner[0]
        aliases = {"findings": "key_findings", "results": "key_findings", "method": "methodology",
                   "methods": "methodology", "data": "dataset", "datasets": "dataset", "problem": "research_problem",
                   "limitation": "limitations", "contribution": "contributions", "questions": "research_questions"}
        out = {}
        for k, v in data.items():
            key = _norm_key(k)
            key = aliases.get(key, key)
            if key in SUMMARY_SECTIONS and key not in out:
                out[key] = v
        return out

    @field_validator(*SUMMARY_SECTIONS, mode="before")
    @classmethod
    def _coerce(cls, v):
        return _statements(v)

    def is_empty(self) -> bool:
        return not any(getattr(self, s) for s in SUMMARY_SECTIONS)


class PaperRelationOut(BaseModel):
    paper_id: str
    relationship: str = "related"
    similarities: str = ""
    differences: str = ""
    contradiction: str = ""

    @field_validator("relationship", mode="before")
    @classmethod
    def _rel(cls, v):
        return _pick(v, ("similar_approach", "extends", "contradicts", "complements",
                         "different_setting", "related"), "related")

    @field_validator("similarities", "differences", "contradiction", mode="before")
    @classmethod
    def _text(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return str(v)


class PaperComparisonOut(BaseModel):
    relations: list[PaperRelationOut] = Field(default_factory=list)


class EvidenceIn(BaseModel):
    paper_id: str
    evidence: str
    relationship: str = "supports"
    basis: str = "inferred"

    @field_validator("relationship", mode="before")
    @classmethod
    def _rel(cls, v):
        return _pick(v, EVIDENCE_RELATIONSHIPS, "supports")

    @field_validator("basis", mode="before")
    @classmethod
    def _basis(cls, v):
        return _pick(v, EVIDENCE_BASES, "inferred")

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v):
        return " ".join(str(v or "").split())


class GapOperation(BaseModel):
    action: Literal["create", "add_evidence", "update", "merge", "mark_addressed"]
    gap_id: str = ""
    gap_ids: list[str] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    gap_type: str = "other"
    category: str = "potential_gap"
    status: str = ""
    evidence: list[EvidenceIn] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"new": "create", "add": "create", "support": "add_evidence",
                   "strengthen": "add_evidence", "contradict": "add_evidence", "weaken": "add_evidence",
                   "evidence": "add_evidence", "resolve": "mark_addressed", "address": "mark_addressed",
                   "addressed": "mark_addressed", "edit": "update", "refine": "update"}
        return aliases.get(v, v)

    @field_validator("gap_type", mode="before")
    @classmethod
    def _type(cls, v):
        return _pick(v, GAP_TYPES, "other")

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        return _pick(v, GAP_CATEGORIES, "potential_gap")

    @field_validator("evidence", "gap_ids", mode="before")
    @classmethod
    def _list(cls, v):
        return _as_list(v)


class GapOperations(BaseModel):
    operations: list[GapOperation] = Field(default_factory=list)

    @field_validator("operations", mode="before")
    @classmethod
    def _skip_invalid(cls, v):
        # Keep valid operations even if some are malformed.
        out = []
        for op in _as_list(v):
            try:
                out.append(GapOperation.model_validate(op))
            except Exception:
                continue
        return out


class SynthesisItem(BaseModel):
    statement: str
    paper_ids: list[str] = Field(default_factory=list)

    @field_validator("paper_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        return [str(x).strip() for x in _as_list(v) if str(x).strip()]


def _items(v) -> list:
    out = []
    for item in _as_list(v):
        if isinstance(item, str):
            item = {"statement": item}
        elif isinstance(item, dict) and "statement" not in item:
            item = {**item, "statement": item.get("text") or item.get("finding") or ""}
        if isinstance(item, dict) and str(item.get("statement", "")).strip():
            out.append(item)
    return out


class SynthesisOut(BaseModel):
    overview: str = ""
    known_findings: list[SynthesisItem] = Field(default_factory=list)
    common_methods: list[SynthesisItem] = Field(default_factory=list)
    contradictory_findings: list[SynthesisItem] = Field(default_factory=list)
    recurring_limitations: list[SynthesisItem] = Field(default_factory=list)
    under_researched: list[SynthesisItem] = Field(default_factory=list)
    research_questions: list[SynthesisItem] = Field(default_factory=list)

    @field_validator("known_findings", "common_methods", "contradictory_findings",
                     "recurring_limitations", "under_researched", "research_questions", mode="before")
    @classmethod
    def _coerce(cls, v):
        return _items(v)

    @field_validator("overview", mode="before")
    @classmethod
    def _overview(cls, v):
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v or "")


# ── Gap duplication test / Topic Validator outputs ──

class AddressingPaper(BaseModel):
    index: int
    how: str = ""


class GapJudgement(BaseModel):
    verdict: str = "open"
    addressing: list[AddressingPaper] = Field(default_factory=list)
    note: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        return _pick(v, ("open", "partially_addressed", "addressed"), "open")

    @field_validator("addressing", mode="before")
    @classmethod
    def _list(cls, v):
        out = []
        for item in _as_list(v):
            if isinstance(item, int):
                item = {"index": item}
            if isinstance(item, dict) and str(item.get("index", "")).strip().isdigit():
                out.append({"index": int(item["index"]), "how": str(item.get("how") or item.get("reason") or "")})
        return out


class MissingKnob(BaseModel):
    knob: str = ""
    question: str = ""


class SpecificityCheck(BaseModel):
    level: str = "neighbourhood"
    restated: str = ""
    missing: list[MissingKnob] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)

    @field_validator("level", mode="before")
    @classmethod
    def _level(cls, v):
        return _pick(v, ("field", "neighbourhood", "topic"), "neighbourhood")

    @field_validator("missing", mode="before")
    @classmethod
    def _missing(cls, v):
        out = []
        for item in _as_list(v):
            if isinstance(item, str):
                item = {"knob": "", "question": item}
            if isinstance(item, dict) and (item.get("question") or item.get("knob")):
                out.append({"knob": str(item.get("knob") or ""), "question": str(item.get("question") or "")})
        return out

    @field_validator("queries", mode="before")
    @classmethod
    def _queries(cls, v):
        return SearchQueries._clean(v)


class Neighbour(BaseModel):
    index: int
    closeness: str = "related"
    overlap: str = ""
    difference: str = ""

    @field_validator("closeness", mode="before")
    @classmethod
    def _closeness(cls, v):
        return _pick(v, ("same", "very_close", "related", "unrelated"), "related")


class DuplicationTest(BaseModel):
    neighbours: list[Neighbour] = Field(default_factory=list)
    verdict: str = "crowded"
    reason: str = ""
    pivot_questions: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        return _pick(v, ("duplicated", "crowded", "clear_lane"), "crowded")

    @field_validator("neighbours", mode="before")
    @classmethod
    def _neighbours(cls, v):
        out = []
        for item in _as_list(v):
            try:
                out.append(Neighbour.model_validate(item))
            except Exception:
                continue
        return out

    @field_validator("pivot_questions", mode="before")
    @classmethod
    def _questions(cls, v):
        return [str(q).strip() for q in _as_list(v) if str(q).strip()][:6]
