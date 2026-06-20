from typing import TypedDict


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

