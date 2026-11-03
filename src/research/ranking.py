"""Priority of a candidate paper: relevant, recent and well-published papers are processed first."""
import math
from datetime import date

from src.config import RANK_W_IMPACT, RANK_W_RECENCY, RANK_W_RELEVANCE, RANK_W_VENUE

RECENCY_HALF_LIFE_YEARS = 3.0
IMPACT_REFERENCE_CITES_PER_YEAR = 50.0
USER_PAPER_BONUS = 1.0  # the user's own uploads are always analysed before discovered papers
REVIEW_BONUS = 0.08     # review papers map the field and its research agenda: read them early


def recency(year: int | None, today: date | None = None) -> float:
    if not year:
        return 0.3
    age = max(0, (today or date.today()).year - int(year))
    return 1.0 if age <= 1 else 0.5 ** ((age - 1) / RECENCY_HALF_LIFE_YEARS)


def impact(citations: int | None, year: int | None, today: date | None = None) -> float:
    """Citations per year since publication, on a log scale (fair to new papers)."""
    if citations is None:
        return 0.2
    years = max(1, (today or date.today()).year - int(year) + 1) if year else 3
    per_year = max(0, citations) / years
    return min(1.0, math.log1p(per_year) / math.log1p(IMPACT_REFERENCE_CITES_PER_YEAR))


def venue(peer_reviewed: bool | None, has_venue: bool, venue_name: str = "") -> float:
    """Top venues (CVPR, NeurIPS, ICML, TPAMI, …) score highest, even when an index lists the paper as an arXiv
    preprint; then other peer-reviewed venues, then preprints."""
    from src.research.scholarly import top_venue
    if top_venue(venue_name):
        return 1.0
    if peer_reviewed is True:
        return 0.75
    if peer_reviewed is False:
        return 0.4
    return 0.55 if has_venue else 0.3


FRESH_TOP_VENUE_IMPACT = 0.6  # a new paper from a top venue hasn't had time to be cited: don't rank it as uncited


def priority(paper: dict, today: date | None = None) -> tuple[float, dict]:
    """Return (score, breakdown). Relevance comes from the screening step (0-1)."""
    parts = {
        "relevance": round(float(paper.get("relevance_score") or 0.0), 3),
        "recency": round(recency(paper.get("year"), today), 3),
        "impact": round(impact(paper.get("citation_count"), paper.get("year"), today), 3),
        "venue": round(venue(paper.get("peer_reviewed"), bool(paper.get("venue")), paper.get("venue") or ""), 3),
    }
    age = ((today or date.today()).year - int(paper["year"])) if paper.get("year") else 99
    if parts["venue"] >= 1.0 and age <= 2:
        parts["impact"] = max(parts["impact"], FRESH_TOP_VENUE_IMPACT)
    score = (RANK_W_RELEVANCE * parts["relevance"] + RANK_W_RECENCY * parts["recency"]
             + RANK_W_IMPACT * parts["impact"] + RANK_W_VENUE * parts["venue"])
    if paper.get("added_by") == "user":
        score += USER_PAPER_BONUS
    if (paper.get("publication_type") or "") == "review" or "review" in (paper.get("title") or "").lower().split():
        score += REVIEW_BONUS
    return round(score, 4), parts
