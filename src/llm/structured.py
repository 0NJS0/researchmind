"""Structured (JSON) LLM calls with validation, recovery and a validated-output cache."""
import hashlib
import json
import logging
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import DATABASE_URL, LLM_CACHE
from src.llm.llm import LLMProvider, get_provider, strip_reasoning

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Validation retries only; transport retries (429/5xx) happen inside the provider.
STRUCTURED_RETRIES = 1

JSON_SYSTEM_PROMPT = (
    "You are a precise research assistant that answers with a single JSON object. "
    "Output only the JSON object: no prose before or after it, no markdown fences, no comments. "
    "Use double quotes for all keys and strings. If information is missing, use an empty string or "
    "an empty list instead of inventing it."
)


class StructuredOutputError(RuntimeError):
    pass


# ── JSON extraction / recovery ──

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES_OUTSIDE_STRINGS = re.compile(r"[“”](?=\s*[:,}\]])|(?<=[{\[,:]\s)[“”]|(?<=[{\[,:])[“”]")


def _balanced_json_candidates(text: str):
    """Yield substrings that start at '{' or '[' and end at the matching bracket, longest-first per start."""
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        stack = []
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c in "{[":
                stack.append("}" if c == "{" else "]")
            elif c in "}]":
                if not stack or stack.pop() != c:
                    break
                if not stack:
                    yield text[start:i + 1]
                    break
        else:
            # Unterminated (e.g. truncated output): try closing the open brackets.
            if stack and not in_string:
                yield text[start:] + "".join(reversed(stack))


def _repair(candidate: str) -> str:
    # Curly quotes used as JSON delimiters (not inside string values) become straight quotes.
    fixed = _SMART_QUOTES_OUTSIDE_STRINGS.sub('"', candidate)
    return _TRAILING_COMMA_RE.sub(r"\1", fixed)


def extract_json(text: str):
    """Best-effort parse of a JSON value from an LLM response. Raises ValueError.

    Objects are preferred over bare arrays, so prose like "see [P1]" before the answer is skipped.
    """
    text = strip_reasoning(text or "")
    if not text.strip():
        raise ValueError("empty response")
    sources = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    fallback = None
    for source in sources:
        source = source.strip()
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            pass
        for candidate in _balanced_json_candidates(source):
            for attempt in (candidate, _repair(candidate)):
                try:
                    value = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
                if fallback is None:
                    fallback = value
                break
    if fallback is not None:
        return fallback
    raise ValueError("no valid JSON object found in response")


def parse_structured(text: str, schema: type[T]) -> T:
    data = extract_json(text)
    return schema.model_validate(data)


# ── Validated-output cache (Postgres) ──

class PostgresCache:
    def get(self, key: str) -> str | None:
        from src.db import connection
        try:
            with connection() as conn:
                row = conn.execute("SELECT value FROM llm_cache WHERE key = %s", (key,)).fetchone()
                return row["value"] if row else None
        except Exception as e:
            logger.warning("LLM cache read failed: %s", e.__class__.__name__)
            return None

    def set(self, key: str, value: str) -> None:
        from src.db import connection
        try:
            with connection() as conn:
                conn.execute("INSERT INTO llm_cache (key, value) VALUES (%s, %s) "
                             "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = now()",
                             (key, value))
        except Exception as e:
            logger.warning("LLM cache write failed: %s", e.__class__.__name__)


_cache = PostgresCache() if (LLM_CACHE and DATABASE_URL) else None


def _cache_key(provider: LLMProvider, system: str, prompt: str, schema: type[BaseModel]) -> str:
    raw = f"{provider.name}|{provider.model}|{schema.__name__}|{system}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Public API ──

def invoke_structured(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    provider: LLMProvider | None = None,
    max_tokens: int | None = None,
    retries: int | None = None,
    use_cache: bool = True,
    check: Callable[[T], str | None] | None = None,
) -> T:
    """Call the LLM, parse and validate its JSON against `schema`.

    Malformed output is recovered where possible, otherwise the call is retried once with the
    validation error appended. Only validated output is cached; after the retries are used up,
    StructuredOutputError is raised and nothing is stored. Transport errors (LLMError and
    subclasses such as QuotaExhausted) propagate unchanged. `check` rejects output that parses but is
    useless (e.g. an empty summary): it returns an error message, which is fed back like a parse error.
    """
    provider = provider or get_provider()
    retries = STRUCTURED_RETRIES if retries is None else retries
    system = system or JSON_SYSTEM_PROMPT

    key = _cache_key(provider, system, prompt, schema) if (_cache and use_cache) else None
    if key:
        cached = _cache.get(key)
        if cached is not None:
            try:
                result = schema.model_validate_json(cached)
                if not (check and check(result)):
                    return result
            except ValidationError:
                pass

    current_prompt = prompt
    last_error = ""
    for attempt in range(retries + 1):
        text = provider.complete(current_prompt, system=system, max_tokens=max_tokens, json_mode=True)
        try:
            result = parse_structured(text, schema)
            problem = check(result) if check else None
            if problem:
                raise ValueError(problem)
        except (ValueError, ValidationError) as e:
            last_error = _short_error(e)
            logger.warning("Structured output invalid for %s (attempt %d): %s | reply starts: %r",
                           schema.__name__, attempt + 1, last_error, (text or "")[:200])
            current_prompt = (
                f"{prompt}\n\nIMPORTANT: your previous reply could not be used ({last_error}).\n"
                "Reply again with ONLY one valid JSON object in exactly the requested format. "
                "No prose, no markdown fences."
            )
            continue
        if key:
            _cache.set(key, result.model_dump_json())
        return result

    raise StructuredOutputError(f"{schema.__name__}: invalid LLM output after {retries + 1} attempts ({last_error})")


def _short_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for err in error.errors()[:3]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            parts.append(f"{loc}: {err.get('msg')}")
        return "; ".join(parts)
    return str(error)[:200]
