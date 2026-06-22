from typing import TypedDict, NotRequired


class ResearchState(TypedDict):
    question: str
    documents: list
    analysis: str
    comparison: str
    gaps: str
    report: str
    history: str
    topic: str
    k: int
    domain_classification: NotRequired[str]
