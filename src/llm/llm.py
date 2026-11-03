"""LLM providers (OpenRouter / any OpenAI-compatible server / Anthropic) over plain HTTP.

Built for OpenRouter's free tier, which is strict and quirky:
* rate limits (HTTP 429) are retried honouring Retry-After; a *daily* cap raises QuotaExhausted
  immediately instead of burning retries;
* a client-side limiter spaces requests (LLM_MIN_INTERVAL) so bursts don't trip per-minute limits;
* fallback models (LLM_FALLBACK_MODELS) are tried when a model is rate limited, overloaded or gone;
* most free models are reasoning models: reasoning is excluded from replies, stray <think> blocks are
  stripped, and a reply that spent its whole budget on reasoning is retried with a larger budget;
* JSON mode is only requested from models that advertise it, and is switched off if rejected;
* models that reject a system role get the system prompt folded into the user message.
"""
import logging
import random
import re
import threading
import time
from typing import Protocol

import httpx

from src.config import (
    is_allowed_model, LLM_API_KEY, LLM_FALLBACK_MODELS, LLM_FAST_MODEL, LLM_JSON_MODE, LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES, LLM_MIN_INTERVAL, LLM_MODEL, LLM_PROVIDER, LLM_REASONING, LLM_TEMPERATURE,
    LLM_TIMEOUT, LLM_URL, OPENROUTER_APP_NAME, OPENROUTER_APP_URL,
)

logger = logging.getLogger(__name__)

MAX_RETRY_WAIT = 60.0          # never sleep longer than this for one retry
MAX_OUTPUT_CAP = 16384
REASONING_MIN_TOKENS = 2048    # reasoning tokens count against max_tokens even when excluded
MAX_BUDGET_GROWS = 2           # bigger-budget retries per model for a cut-off or empty reply
CATALOGUE_RETRY_SECONDS = 600  # retry a failed model-catalogue fetch after this long
ANTHROPIC_URL = "https://api.anthropic.com/v1"


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhausted(RuntimeError):
    """The account's daily request quota is used up; retrying today is pointless.

    Deliberately *not* an LLMError: code that tolerates individual LLM failures (and carries on with
    the next paper) must not swallow this — the whole job has to stop.
    """


class LLMHTTPError(LLMError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class ServiceUnavailable(LLMError):
    """Every configured model failed for a temporary reason (network, 5xx, timeouts): retry later."""


class NetworkDown(ServiceUnavailable):
    """No internet connection: nothing outside the LAN can be reached."""


def internet_reachable(timeout: float = 5.0) -> bool:
    """True if the LLM endpoint (or, failing that, OpenAlex) answers at all, with any HTTP status. Used to tell
    "this site is down" from "our internet connection is down" before a paper is downgraded for lack of content.
    No credentials are sent."""
    for url in (LLM_URL, "https://api.openalex.org"):
        if not url:
            continue
        try:
            httpx.head(url, timeout=timeout)
            return True
        except httpx.HTTPError:
            continue
    return False


def is_outage(err: BaseException) -> bool:
    """The LLM service can't be used right now (not a problem with this particular input).

    Work should be put back and retried later instead of being recorded as failed: rate limits, outages,
    and a rejected/expired API key or exhausted credit (401/402/403 that aren't model-specific)."""
    if isinstance(err, (RateLimited, ServiceUnavailable)):
        return True
    return isinstance(err, LLMHTTPError) and err.status in (401, 402, 403) and not _model_gone(err)


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None,
                 json_mode: bool = False) -> str:
        ...


# ── shared HTTP plumbing (also used by the embeddings client) ──

class RateLimiter:
    """Minimum spacing between requests to one host, shared by all threads."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if delay > 0:
            time.sleep(delay)


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def limiter_for(url: str, min_interval: float = LLM_MIN_INTERVAL) -> RateLimiter:
    host = httpx.URL(url).host or url
    with _limiters_lock:
        if host not in _limiters:
            _limiters[host] = RateLimiter(min_interval)
        return _limiters[host]


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = str(err.get("message") or "")
        raw = (err.get("metadata") or {}).get("raw") if isinstance(err.get("metadata"), dict) else None
        return f"{msg} {raw}"[:500] if raw else msg[:500]
    if isinstance(err, str):
        return err[:500]
    return str(data)[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    reset = resp.headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        seconds = int(reset) / (1000 if len(reset) > 11 else 1) - time.time()
        if seconds > 0:
            return seconds
    return None


_TRANSIENT_PREFIXES = ("network error", "invalid JSON from", "unexpected response from", "request failed")
_RATE_LIMIT_RE = re.compile(r"rate[- ]?limit", re.I)


def _is_account_limit(message: str) -> bool:
    """OpenRouter's per-minute limit applies to the whole key: switching models doesn't help, waiting does."""
    m = (message or "").lower()
    return "per-min" in m or "per min" in m or "free-models-per-min" in m


def _is_daily_cap(message: str) -> bool:
    m = message.lower()
    return "per-day" in m or "per day" in m or "daily" in m or "free-models-per-day" in m


def post_json(url: str, payload: dict, headers: dict, *, timeout: float = LLM_TIMEOUT,
              retries: int = LLM_MAX_RETRIES, limiter: RateLimiter | None = None) -> dict:
    """POST JSON with retries for transient failures. Raises LLMHTTPError for client errors."""
    last: Exception | None = None
    timeouts = 0
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = LLMError(f"network error ({e.__class__.__name__})")
            if isinstance(e, httpx.TimeoutException):
                timeouts += 1
                if timeouts > 1:  # a model that times out twice is overloaded: let the caller try another one
                    break
            _backoff(attempt, retries, None, str(last))
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                last = LLMError("invalid JSON from LLM endpoint")
                _backoff(attempt, retries, None, str(last))
                continue
            # Some gateways report upstream failures inside a 200 body.
            if isinstance(data, dict) and isinstance(data.get("error"), dict) and not data.get("choices") \
                    and not data.get("data") and not data.get("content"):
                code = data["error"].get("code")
                msg = str(data["error"].get("message") or "upstream error")
                if code == 429 or _RATE_LIMIT_RE.search(msg):
                    if _is_daily_cap(msg):
                        raise QuotaExhausted(_quota_message(msg))
                    last = RateLimited(msg)
                    if not _is_account_limit(msg):
                        break  # the model's provider is busy: another model is the better retry
                else:
                    last = LLMError(msg[:300])
                _backoff(attempt, retries, None, msg)
                continue
            if not isinstance(data, dict):
                last = LLMError("unexpected response from LLM endpoint")
                _backoff(attempt, retries, None, str(last))
                continue
            return data

        message = _error_message(resp)
        if resp.status_code == 429:
            if _is_daily_cap(message):
                raise QuotaExhausted(_quota_message(message))
            wait = _retry_after(resp)
            last = RateLimited(message or "rate limited", wait)
            if not _is_account_limit(message):
                break  # the model's provider is busy: another model is the better retry
            _backoff(attempt, retries, wait, "rate limited")
            continue
        if resp.status_code in (408, 409, 425, 500, 502, 503, 504, 520, 522, 524, 529):
            last = LLMHTTPError(resp.status_code, message)
            _backoff(attempt, retries, _retry_after(resp), f"HTTP {resp.status_code}")
            continue
        raise LLMHTTPError(resp.status_code, message)
    raise last or LLMError("request failed")


def _backoff(attempt: int, retries: int, wait: float | None, reason: str) -> None:
    if attempt >= retries:
        return
    delay = wait if wait is not None else (2 ** attempt) * 2 + random.uniform(0, 1)
    delay = min(MAX_RETRY_WAIT, max(0.5, delay))
    logger.warning("LLM request failed (%s); retrying in %.1fs", reason, delay)
    time.sleep(delay)


def _quota_message(detail: str) -> str:
    return ("The OpenRouter free-tier daily request limit is used up. It resets daily; adding credits "
            f"raises the limit. ({detail[:160]})")


# ── reply cleaning ──

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> blocks some reasoning models put in their visible reply."""
    text = _THINK_RE.sub("", text or "")
    # An unterminated opening block means the reply was cut off mid-reasoning.
    m = re.search(r"<(think|thinking|reasoning)>", text, re.IGNORECASE)
    if m:
        closing = re.search(r"</(think|thinking|reasoning)>", text[m.end():], re.IGNORECASE)
        text = text[:m.start()] if not closing else text
    # Some models emit only the closing tag.
    parts = re.split(r"</(?:think|thinking|reasoning)>", text, flags=re.IGNORECASE)
    return parts[-1].strip()


# ── OpenAI-compatible chat (OpenRouter, vLLM, llama.cpp, Ollama …) ──

class _ModelCaps:
    """Per-model capabilities learnt from OpenRouter's catalogue and from rejected requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self._catalogue: dict[str, set[str]] | None = None
        self._catalogue_failed_at = 0.0
        self.no_json: set[str] = set()
        self.no_system: set[str] = set()

    def catalogue(self) -> dict[str, set[str]]:
        with self._lock:
            retry_due = self._catalogue_failed_at and time.monotonic() - self._catalogue_failed_at > CATALOGUE_RETRY_SECONDS
            if self._catalogue is None or (not self._catalogue and retry_due):
                self._catalogue = {}
                if LLM_PROVIDER == "openrouter":
                    try:
                        resp = httpx.get(f"{LLM_URL}/models", timeout=15)
                        resp.raise_for_status()
                        for m in resp.json().get("data", []):
                            self._catalogue[m["id"]] = set(m.get("supported_parameters") or [])
                        self._catalogue_failed_at = 0.0
                    except Exception as e:  # catalogue is an optimisation; fall back to trial and error
                        self._catalogue = {}
                        self._catalogue_failed_at = time.monotonic()
                        logger.info("Could not load the OpenRouter model catalogue: %s", e.__class__.__name__)
            return self._catalogue

    def supports_json(self, model: str) -> bool:
        if not LLM_JSON_MODE or model in self.no_json:
            return False
        params = self.catalogue().get(model)
        if params is None:
            return LLM_PROVIDER != "openrouter" or not self.catalogue()
        return "response_format" in params or "structured_outputs" in params

    def supports_reasoning(self, model: str) -> bool:
        params = self.catalogue().get(model)
        return params is not None and ("reasoning" in params or "include_reasoning" in params)

    def exists(self, model: str) -> bool | None:
        cat = self.catalogue()
        return (model in cat) if cat else None


caps = _ModelCaps()


def _default_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if LLM_PROVIDER == "openrouter":
        headers["X-Title"] = OPENROUTER_APP_NAME
        if OPENROUTER_APP_URL:
            headers["HTTP-Referer"] = OPENROUTER_APP_URL
    return headers


def _looks_like_unsupported_param(message: str) -> bool:
    m = message.lower()
    return ("response_format" in m or "json_object" in m or "json mode" in m
            or "requested parameters" in m or "structured output" in m)


def _looks_like_no_system_role(message: str) -> bool:
    m = message.lower()
    return ("system" in m and ("role" in m or "not supported" in m or "instruction" in m)) \
        or "developer instruction" in m


def _model_gone(err: LLMHTTPError) -> bool:
    """The model can't serve this key right now (removed, restricted, no provider): try the next one."""
    m = err.message.lower()
    if _looks_like_unsupported_param(m):
        return False
    if err.status == 404 and ("model" in m or "no endpoints" in m):
        return True
    return err.status in (402, 403) and any(s in m for s in ("only available", "not available", "not allowed",
                                                              "no endpoints", "requires", "moderation"))


class OpenAICompatProvider:
    def __init__(self, model: str, fallbacks: list[str] | None = None, *, base_url: str = LLM_URL,
                 api_key: str = LLM_API_KEY, name: str | None = None):
        self.model = model
        self.fallbacks = [m for m in (fallbacks or []) if m and m != model]
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name or ("openrouter" if LLM_PROVIDER == "openrouter" else "openai")
        self.last_model = model

    def _payload(self, model: str, system: str | None, prompt: str, max_tokens: int, json_mode: bool) -> dict:
        if system and model in caps.no_system:
            messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
        elif system:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "temperature": LLM_TEMPERATURE, "max_tokens": max_tokens}
        if json_mode and caps.supports_json(model):
            payload["response_format"] = {"type": "json_object"}
        if self.name == "openrouter" and LLM_REASONING in ("exclude", "low") and caps.supports_reasoning(model):
            payload["reasoning"] = {"exclude": True} if LLM_REASONING == "exclude" \
                else {"effort": "low", "exclude": True}
        return payload

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None,
                 json_mode: bool = False) -> str:
        budget = max_tokens or LLM_MAX_OUTPUT_TOKENS
        errors: list[str] = []
        last_error: LLMError | None = None
        transient = True  # every failure so far was temporary (network, 5xx, rate limits)
        models = [self.model, *self.fallbacks]
        for model in models:
            try:
                text = self._complete_one(model, system, prompt, budget, json_mode)
                self.last_model = model
                return text
            except QuotaExhausted:
                raise
            except RateLimited as e:
                if _is_account_limit(str(e)):
                    raise  # every model shares the key's per-minute limit
                errors.append(f"{model}: rate limited")
                if model != models[-1]:
                    logger.warning("Model %s is rate limited; trying the next fallback model", model)
                last_error = e
            except LLMHTTPError as e:
                if not (_model_gone(e) or e.status >= 500):
                    raise
                transient = transient and e.status >= 500
                errors.append(f"{model}: {e}")
                last_error = e
            except LLMError as e:
                transient = transient and str(e).startswith(_TRANSIENT_PREFIXES)
                errors.append(f"{model}: {e}")
                last_error = e
        if isinstance(last_error, RateLimited) and transient:
            raise RateLimited("All configured models are rate limited: " + "; ".join(errors)[:400])
        if transient:
            raise ServiceUnavailable("All configured models are unavailable: " + "; ".join(errors)[:400])
        raise LLMError("LLM call failed: " + "; ".join(errors)[:500])

    def _complete_one(self, model: str, system: str | None, prompt: str, budget: int, json_mode: bool) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = _default_headers(self.api_key)
        limiter = limiter_for(url)
        if caps.supports_reasoning(model):
            budget = max(budget, REASONING_MIN_TOKENS)
        truncated_text = ""
        grows = 0
        for _ in range(4):  # adaptation rounds: JSON mode off, system role folded, bigger budget
            payload = self._payload(model, system, prompt, budget, json_mode)
            try:
                data = post_json(url, payload, headers, limiter=limiter)
            except LLMHTTPError as e:
                if e.status in (400, 404, 422) and "response_format" in payload and \
                        _looks_like_unsupported_param(e.message):
                    logger.warning("Model %s rejected JSON mode; continuing without it", model)
                    caps.no_json.add(model)
                    continue
                if e.status in (400, 422) and system and model not in caps.no_system and \
                        _looks_like_no_system_role(e.message):
                    logger.warning("Model %s rejected the system role; folding it into the user message", model)
                    caps.no_system.add(model)
                    continue
                raise
            choices = data.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = _content_text(message.get("content"))
            text = strip_reasoning(content)
            truncated = choice.get("finish_reason") == "length"
            can_grow = budget < MAX_OUTPUT_CAP and grows < MAX_BUDGET_GROWS
            if text and truncated and json_mode and can_grow:
                # A cut-off JSON reply can still "parse" after bracket repair, silently losing its last fields.
                truncated_text = text
                budget, grows = min(MAX_OUTPUT_CAP, max(budget * 2, REASONING_MIN_TOKENS)), grows + 1
                logger.info("Reply from %s was cut off by max_tokens; retrying with max_tokens=%d", model, budget)
                continue
            if text:
                return text
            if truncated and can_grow:
                # A reasoning model spent the whole budget thinking: give it more room.
                budget, grows = min(MAX_OUTPUT_CAP, max(budget * 2, REASONING_MIN_TOKENS)), grows + 1
                logger.info("Empty reply from %s (token budget exhausted); retrying with max_tokens=%d",
                            model, budget)
                continue
            if truncated_text:
                return truncated_text  # the best reply we got; the caller validates it
            raise LLMError(f"{model} returned an empty reply")
        if truncated_text:
            return truncated_text
        raise LLMError(f"{model} could not produce a usable reply")


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


# ── Anthropic (Messages API) ──

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str):
        self.model = model
        self.last_model = model

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None,
                 json_mode: bool = False) -> str:
        url = f"{(LLM_URL or ANTHROPIC_URL).rstrip('/')}/messages"
        payload = {"model": self.model, "max_tokens": max_tokens or LLM_MAX_OUTPUT_TOKENS,
                   "temperature": LLM_TEMPERATURE, "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        headers = {"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = post_json(url, payload, headers, limiter=limiter_for(url))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text.strip():
            raise LLMError(f"{self.model} returned an empty reply")
        return text.strip()


# ── provider registry ──

_providers: dict[str, LLMProvider] = {}
_providers_lock = threading.Lock()


def _build_provider(model: str, with_fallbacks: bool) -> LLMProvider:
    if not model:
        raise LLMError("LLM_MODEL is not configured.")
    for m in [model, *(LLM_FALLBACK_MODELS if with_fallbacks else [])]:
        if not is_allowed_model(m):
            raise LLMError(f"'{m}' is not a free model and LLM_FREE_ONLY is on. Use a model id ending in ':free'.")
    if LLM_PROVIDER in ("openai", "openrouter", "local", "openai-compatible", "openai_compatible"):
        if not LLM_URL:
            raise LLMError("LLM_URL is not configured.")
        return OpenAICompatProvider(model, LLM_FALLBACK_MODELS if with_fallbacks else [])
    if LLM_PROVIDER == "anthropic":
        return AnthropicProvider(model)
    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'")


def get_provider(purpose: str = "default") -> LLMProvider:
    """Return the provider for a purpose. 'fast' uses LLM_FAST_MODEL when configured."""
    model = LLM_FAST_MODEL if purpose == "fast" and LLM_FAST_MODEL else LLM_MODEL
    with _providers_lock:
        if model not in _providers:
            _providers[model] = _build_provider(model, with_fallbacks=True)
        return _providers[model]


_quota_cache: dict = {"at": 0.0, "value": None}


def free_requests_remaining(max_age: float = 60.0) -> int | None:
    """OpenRouter free-model requests left today (None when unknown or not using OpenRouter)."""
    if LLM_PROVIDER != "openrouter" or not LLM_API_KEY:
        return None
    now = time.monotonic()
    if _quota_cache["value"] is not None and now - _quota_cache["at"] < max_age:
        return _quota_cache["value"]
    try:
        r = httpx.get(f"{LLM_URL}/key", headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=8)
        free = ((r.json() or {}).get("data") or {}).get("free_model_daily_requests") or {}
        value = free.get("remaining") if isinstance(free.get("remaining"), int) else None
    except (httpx.HTTPError, ValueError):
        value = None
    _quota_cache.update(at=now, value=value)
    return value


def check_llm() -> tuple[bool, str]:
    if not LLM_MODEL:
        return False, "LLM_MODEL not configured"
    if LLM_PROVIDER == "openrouter":
        if not LLM_API_KEY:
            return False, "LLM_API_KEY not configured"
        try:
            r = httpx.get(f"{LLM_URL}/key", headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=8)
        except httpx.HTTPError:
            return False, "OpenRouter unreachable"
        if r.status_code in (401, 403):
            return False, "OpenRouter rejected the API key"
        if r.status_code != 200:
            return False, f"OpenRouter returned HTTP {r.status_code}"
        exists = caps.exists(LLM_MODEL)
        if exists is False:
            return False, f"Model '{LLM_MODEL}' is not available on OpenRouter"
        info = (r.json() or {}).get("data") or {}
        free = info.get("free_model_daily_requests") or {}
        suffix = f" · {free['remaining']}/{free['limit']} free requests left today" \
            if isinstance(free.get("remaining"), int) and isinstance(free.get("limit"), int) else ""
        if isinstance(free.get("remaining"), int) and free["remaining"] <= 0:
            return False, "OpenRouter free-model daily limit reached (resets daily)"
        return True, f"connected · {LLM_MODEL}{suffix}"
    if LLM_PROVIDER == "anthropic":
        return (True, f"configured · {LLM_MODEL}") if LLM_API_KEY else (False, "LLM_API_KEY not configured")
    try:
        headers = _default_headers(LLM_API_KEY)
        r = httpx.get(f"{LLM_URL}/models", headers=headers, timeout=5)
        return (True, f"connected · {LLM_MODEL}") if r.status_code == 200 else \
            (False, f"LLM endpoint returned HTTP {r.status_code}")
    except httpx.HTTPError:
        return False, "LLM endpoint unreachable"
