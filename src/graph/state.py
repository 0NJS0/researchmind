import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    history: str
    topic: str
    k: int
    mode: str                                   # "query" (answer a question) | "report" (whole-topic report)
    job_id: str                                 # report job to update with progress (report mode)
    documents: list                             # retrieved chunks (langchain Documents)
    domain: dict                                # {"domain", "subfield", "research_type", "methodology"}
    analysis: str
    paper_count: int
    comparison: str
    gaps: str
    report: str
    report_failed: bool                         # the final write-up failed: `report` only lists the errors
    errors: Annotated[list[str], operator.add]  # every stage appends what went wrong, instead of magic strings
