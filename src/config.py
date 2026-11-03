"""All configuration comes from the environment (.env). Nothing secret is hardcoded here."""
import os
import re

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str = "") -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LLM provider ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()  # openrouter | openai (any OpenAI-compatible server) | anthropic
LLM_URL = (os.getenv("LLM_URL") or (OPENROUTER_BASE_URL if LLM_PROVIDER == "openrouter" else "")).rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_FALLBACK_MODELS = _list("LLM_FALLBACK_MODELS")          # tried in order when the main model is rate limited/down
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "")             # optional cheaper model for relevance checks
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
_HOSTED = LLM_PROVIDER in ("openrouter", "anthropic")
# Hosted models have large windows; small self-hosted servers often run 4K. Prompts are budgeted to fit.
LLM_CONTEXT_TOKENS = _int("LLM_CONTEXT_TOKENS", 32768 if _HOSTED else 4096)
LLM_MAX_OUTPUT_TOKENS = _int("LLM_MAX_OUTPUT_TOKENS", 4096 if _HOSTED else 1024)
LLM_JSON_MODE = _bool("LLM_JSON_MODE", True)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 3)
LLM_TIMEOUT = _float("LLM_TIMEOUT", 180)
# Reasoning models (most free OpenRouter models): "exclude" hides reasoning from the reply, "low" asks for
# little reasoning, "off" sends nothing. Reasoning tokens still count against max_tokens.
LLM_REASONING = os.getenv("LLM_REASONING", "exclude").strip().lower()
LLM_MIN_INTERVAL = _float("LLM_MIN_INTERVAL", 3.5 if LLM_PROVIDER == "openrouter" else 0.0)  # seconds between calls
LLM_CACHE = _bool("LLM_CACHE", True)                         # cache validated structured outputs in Postgres
# Refuse any model that is not free (ids ending in ":free"). On by default for OpenRouter.
LLM_FREE_ONLY = _bool("LLM_FREE_ONLY", LLM_PROVIDER == "openrouter")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "")     # optional attribution headers for OpenRouter
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "ResearchMind")

# ── Embeddings ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
EMBEDDING_URL = (os.getenv("EMBEDDING_URL") or LLM_URL or OPENROUTER_BASE_URL).rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or LLM_API_KEY
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 32)
EMBEDDING_TIMEOUT = _float("EMBEDDING_TIMEOUT", 120)

# ── Database ──
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
DB_POOL_MIN = _int("DB_POOL_MIN", 1)
DB_POOL_MAX = _int("DB_POOL_MAX", 20)  # autopilot + background jobs + API requests

# ── Files ──
PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")

# ── API ──
CORS_ORIGINS = _list("CORS_ORIGINS")   # empty = same-origin only (the React app is served by the API)

# ── Research agent ──
SEARXNG_URL = os.getenv("SEARXNG_URL", "")
SEARXNG_CATEGORIES = os.getenv("SEARXNG_CATEGORIES", "science")
SEARXNG_ENGINES = os.getenv("SEARXNG_ENGINES", "")           # comma-separated; empty = instance defaults
SEARXNG_TIMEOUT = _float("SEARXNG_TIMEOUT", 20)
MAX_SEARCH_ITERATIONS = _int("MAX_SEARCH_ITERATIONS", 3)
MAX_RESULTS_PER_QUERY = _int("MAX_RESULTS_PER_QUERY", 20)
MAX_PAPERS_PER_SESSION = _int("MAX_PAPERS_PER_SESSION", 30)
MAX_QUERIES_PER_ITERATION = _int("MAX_QUERIES_PER_ITERATION", 6)
MAX_LLM_RELEVANCE_CHECKS = _int("MAX_LLM_RELEVANCE_CHECKS", 30)
MIN_LEXICAL_RELEVANCE = _float("MIN_LEXICAL_RELEVANCE", 0.15)
# Recent top-venue discovery (proceedings, arXiv newest, OpenAlex last 2 years) in every autopilot search round.
DISCOVERY_ENABLED = _bool("DISCOVERY_ENABLED", True)
DISCOVERY_QUERIES = _int("DISCOVERY_QUERIES", 2)                   # queries sent to arXiv + OpenAlex per round
DISCOVERY_PROCEEDINGS_LIMIT = _int("DISCOVERY_PROCEEDINGS_LIMIT", 30)
DISCOVERY_ABSTRACTS = _int("DISCOVERY_ABSTRACTS", 20)              # proceedings abstracts fetched per round
MAX_CONTENT_CHUNKS = _int("MAX_CONTENT_CHUNKS", 3)
FETCH_TIMEOUT = _float("FETCH_TIMEOUT", 20)
FETCH_TOTAL_TIMEOUT = _float("FETCH_TOTAL_TIMEOUT", 90)       # hard cap for one download
MAX_DOWNLOAD_MB = _int("MAX_DOWNLOAD_MB", 80)
ALLOW_PRIVATE_FETCH = _bool("ALLOW_PRIVATE_FETCH", False)
ENABLE_PAPER_COMPARISON = _bool("ENABLE_PAPER_COMPARISON", True)

# ── Autopilot (continuous research on the active topic) ──
AUTOPILOT_ENABLED = _bool("AUTOPILOT_ENABLED", True)
AUTOPILOT_RESEARCH_INTERVAL_HOURS = _float("AUTOPILOT_RESEARCH_INTERVAL_HOURS", 12)
AUTOPILOT_QUOTA_RESERVE = _int("AUTOPILOT_QUOTA_RESERVE", 250)       # free requests/day kept for Ask & manual use
AUTOPILOT_MIN_RELEVANCE = _float("AUTOPILOT_MIN_RELEVANCE", 0.5)
AUTOPILOT_MAX_PAPERS_PER_TOPIC = _int("AUTOPILOT_MAX_PAPERS_PER_TOPIC", 300)
AUTOPILOT_IDLE_SECONDS = _float("AUTOPILOT_IDLE_SECONDS", 30)          # sleep when there is nothing to do
NARRATIVE_EVERY = _int("NARRATIVE_EVERY", 5)                           # LLM report narrative every N new papers

# ── Paper metadata enrichment (OpenAlex: citations, venue type, open-access PDFs) ──
OPENALEX_ENABLED = _bool("OPENALEX_ENABLED", True)
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")                   # free key: 10,000 credits/day (100 searches without)
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")                       # optional: OpenAlex "polite pool"
OPENALEX_URL = os.getenv("OPENALEX_URL", "https://api.openalex.org").rstrip("/")

# ── Ranking weights: newer + well-published + relevant papers first ──
RANK_W_RELEVANCE = _float("RANK_W_RELEVANCE", 0.45)
RANK_W_RECENCY = _float("RANK_W_RECENCY", 0.25)
RANK_W_IMPACT = _float("RANK_W_IMPACT", 0.20)
RANK_W_VENUE = _float("RANK_W_VENUE", 0.10)


def is_allowed_model(model: str) -> bool:
    return not LLM_FREE_ONLY or model.endswith(":free")


def mask_dsn(dsn: str) -> str:
    """Hide the password in a connection string before it is logged or displayed."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", dsn or "")


def validate_config() -> None:
    missing = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not LLM_MODEL:
        missing.append("LLM_MODEL")
    if not LLM_URL and LLM_PROVIDER != "anthropic":
        missing.append("LLM_URL")
    if LLM_PROVIDER in ("openrouter", "anthropic") and not LLM_API_KEY:
        missing.append("LLM_API_KEY")
    if not EMBEDDING_MODEL:
        missing.append("EMBEDDING_MODEL")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}. Check your .env file.")
    paid = [m for m in [LLM_MODEL, LLM_FAST_MODEL, *LLM_FALLBACK_MODELS, EMBEDDING_MODEL] if m and not is_allowed_model(m)]
    if paid:
        raise ValueError(f"LLM_FREE_ONLY is on but these models are not free: {', '.join(paid)}")


def public_settings() -> dict:
    """Non-secret settings for the UI's status page. Never include keys or connection strings."""
    return {
        "llm_provider": LLM_PROVIDER,
        "llm_free_only": LLM_FREE_ONLY,
        "llm_model": LLM_MODEL,
        "llm_fallback_models": LLM_FALLBACK_MODELS,
        "llm_fast_model": LLM_FAST_MODEL or LLM_MODEL,
        "llm_context_tokens": LLM_CONTEXT_TOKENS,
        "llm_max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        "embedding_model": EMBEDDING_MODEL,
        "searxng_configured": bool(SEARXNG_URL),
        "database_configured": bool(DATABASE_URL),
        "autopilot": {
            "enabled": AUTOPILOT_ENABLED,
            "research_interval_hours": AUTOPILOT_RESEARCH_INTERVAL_HOURS,
            "quota_reserve": AUTOPILOT_QUOTA_RESERVE,
            "max_papers_per_topic": AUTOPILOT_MAX_PAPERS_PER_TOPIC,
            "openalex": OPENALEX_ENABLED,
            "openalex_key": bool(OPENALEX_API_KEY),
        },
        "limits": {
            "max_search_iterations": MAX_SEARCH_ITERATIONS,
            "max_results_per_query": MAX_RESULTS_PER_QUERY,
            "max_papers_per_session": MAX_PAPERS_PER_SESSION,
            "max_queries_per_iteration": MAX_QUERIES_PER_ITERATION,
            "max_download_mb": MAX_DOWNLOAD_MB,
        },
    }
