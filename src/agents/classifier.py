import logging

from pydantic import BaseModel, field_validator

from src.agents.common import by_paper, error_text
from src.agents.prompts import CLASSIFY_PROMPT
from src.graph.state import ResearchState
from src.llm.structured import invoke_structured
from src.research.prompts import RESEARCH_SYSTEM_PROMPT, data

logger = logging.getLogger(__name__)


class DomainOut(BaseModel):
    domain: str = ""
    subfield: str = ""
    research_type: str = ""
    methodology: str = ""
    key_quantities: list[str] = []
    quality_signals: list[str] = []

    @field_validator("domain", "subfield", "research_type", "methodology", mode="before")
    @classmethod
    def _text(cls, v):
        return " ".join(str(v or "").split())[:120]

    @field_validator("key_quantities", "quality_signals", mode="before")
    @classmethod
    def _list(cls, v):
        items = v if isinstance(v, list) else ([v] if v else [])
        return [" ".join(str(x).split())[:100] for x in items if str(x).strip()][:4]


def classifier_agent(state: ResearchState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"domain": {}}
    sample = "\n\n".join(
        f'<excerpt paper="{data(pid)}">\n{data(chunks[0].page_content[:1500])}\n</excerpt>'
        for pid, chunks in list(by_paper(docs).items())[:3])
    try:
        out = invoke_structured(CLASSIFY_PROMPT.format(sample=sample), DomainOut, max_tokens=300,
                                system=RESEARCH_SYSTEM_PROMPT)
    except Exception as e:
        # The field is only a hint for later prompts; carry on without it.
        logger.warning("Domain classification failed: %s", e)
        return {"domain": {}, "errors": [error_text("Field classification", e)]}
    logger.info("Classified as: %s", out.model_dump())
    return {"domain": out.model_dump()}
