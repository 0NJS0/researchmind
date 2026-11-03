# Research Agent — Architecture Notes

This document records the Phase 1 analysis of the existing ResearchMind codebase and
the design used to extend it into an agentic research assistant (topic → SearXNG →
paper discovery → structured summaries → evolving research gaps → synthesis).

## 1. Existing architecture map (before this change)

| Concern | What exists | Reused? |
|---|---|---|
| Frontend | Streamlit single page (`src/ui/app.py`) talking to the API over HTTP | Kept as-is. A new React app (`frontend/`) is added alongside it. |
| Backend | FastAPI (`src/api/server.py`), no routers, no auth, CORS `*` | Extended: new `/api/*` router mounted on the same app. Old endpoints untouched. |
| Storage | Qdrant (vectors + metadata) and markdown files under `PAPERS_DIR`. No relational DB. | Research sessions need relational, incremental state (papers, summaries, gaps, evidence, events), so they are stored in PostgreSQL tables (originally SQLite; moved to Postgres alongside pgvector). |
| LLM | `src/llm/llm.py`: one module-level LangChain `ChatOpenAI` against an OpenAI-compatible endpoint | Extended into a provider abstraction (`get_provider()`); the legacy `llm` object is preserved for the existing agents. |
| Structured output | None — every agent returns free-form markdown | Added `src/llm/structured.py`: JSON extraction, recovery, Pydantic validation, constrained retry, and a validated-output cache. |
| Search | Vector search only (Qdrant retriever). No web search. | New SearXNG client (`src/research/searxng.py`). |
| Orchestration | LangGraph linear pipelines (`src/graph/`) | Reused: the research agent is a LangGraph `StateGraph` with a conditional loop (search → screen → process → reflect → search …). |
| PDF extraction | PyMuPDF via LangChain loader | Reused (PyMuPDF directly on downloaded bytes). |
| Background jobs | One `asyncio.create_task` in `/report` (task reference not retained) | Replaced for the new workflow by a small job manager that keeps task references, supports cancellation and records progress events. |
| Streaming | SSE in `/query/stream` | Same SSE pattern reused for agent activity (`/api/research/{id}/events/stream`). |
| Auth | None | Unchanged (out of scope); credentials stay server-side. |
| Tests | None | Added `tests/` (pytest) and `frontend` tests (Vitest + Testing Library). |
| Config | `.env` via `python-dotenv` in `src/config.py` | Extended with new variables; all limits are env-driven. |

## 2. Design

```
React (frontend/) ──/api──▶ FastAPI router (src/research/api.py)
                               │
                               ├── JobManager (src/research/jobs.py) — background asyncio tasks
                               │        └── Research agent: LangGraph (src/research/agent.py)
                               │               understand_topic → search → screen → process → reflect ─┐
                               │                                     ▲                               │
                               │                                     └──── more queries? ────────────┘
                               ├── ResearchStore (PostgreSQL, src/research/store.py)
                               ├── SearXNG client + normalisation + dedup
                               ├── Content fetcher (abstract → HTML meta/full text → PDF)
                               ├── LLM tasks (one prompt per task, structured JSON)
                               └── Gap engine (deterministic state updates, evidence, confidence)
```

### Per-paper incremental loop

```
new paper → fetch content → summarize_paper → compare_papers (vs related existing papers)
          → identify_research_gaps (LLM proposes operations on the current gap state)
          → gap engine applies/validates operations → update_research_synthesis
```

Papers are summarised once. Adding a paper never re-summarises existing papers; only
the gap state and synthesis are updated.

### Gap engine

The LLM never writes the gap table directly. It proposes operations
(`create`, `add_evidence`, `update`, `merge`, `mark_addressed`) against the current
gaps, referenced by short IDs (`G3`, `P7`). The engine then:

* rejects operations that cite unknown papers/gaps (no fabricated citations);
* rejects generic gaps ("more research is needed") and gaps without evidence;
* converts a `create` whose title is near-identical to an existing gap into `add_evidence` (merge);
* records every change in `gap_history` so the UI can show how a gap evolved;
* computes **confidence deterministically** from evidence, not from the LLM:
  independent supporting papers raise it, contradicting papers lower it and mark the gap
  `contested`, and gaps supported only by "future work" statements are capped at `low`
  (rule 2: future work is evidence, not proof).

### Context budget

Prompts are budgeted from `LLM_CONTEXT_TOKENS` / `LLM_MAX_OUTPUT_TOKENS` so the pipeline
works with small local models (e.g. a 4K-context llama.cpp server). Long papers are
condensed chunk-by-chunk into notes before summarisation (`MAX_CONTENT_CHUNKS`).

### Data model (PostgreSQL)

Everything lives in the same PostgreSQL database as the paper vectors (`src/db.py` creates the schema):
`sessions`, `search_queries`, `papers`, `paper_summaries`, `paper_relations`, `gaps`,
`gap_evidence`, `gap_history`, `events`. Summaries are stored as JSON sections where each
statement carries an `inferred` flag, so the UI can distinguish what the paper states
from what the model inferred.


## 3. Reliability notes (free-tier LLMs)

* **Quota:** a daily-cap 429 raises `QuotaExhausted`, which the per-paper error handling deliberately does not
  swallow. The job stops, the session becomes `paused`, and unfinished papers go back to `pending` so
  *Continue research* resumes them.
* **Partial failures:** if the gap or synthesis step fails for a paper, the paper is marked `analysis_failed`
  (not `done`) and is retried on the next run. Before a paper is re-analysed, its earlier evidence is retracted,
  so evidence is never duplicated.
* **Gap rebuild** runs in one transaction (`store.atomic()`). The old gaps stay visible until it finishes, and a
  cancel or error rolls it back. Activity events are committed separately, so progress still streams.
* **Screening:** candidates left unscreened by a cancelled run are picked up by the next screening step.
* **Content fetching (SSRF):** every host is resolved once and the connection goes to that validated IP
  (`PinnedTransport`), which closes the DNS-rebinding window. Only globally routable addresses are allowed,
  including on redirects. Downloads have a size cap, a total deadline and a cancellation check.
* **Jobs** run in worker threads (one per session). A paper added while a job is past its processing step
  triggers a follow-up processing job when that job ends.


## 4. Autopilot (continuous mode)

`src/research/autopilot.py` runs one worker thread (single instance per database via a Postgres advisory
lock). Each `step()` is one task:

1. apply a pending topic switch (`autopilot.requested_topic`) — only here, so the task in progress always finishes;
2. honour pause / quota (`/key` → `free_model_daily_requests.remaining` vs `AUTOPILOT_QUOTA_RESERVE`);
3. ensure the topic's research session (`topics.session_id`) and seed uploaded Library papers into it;
4. process the highest-`priority_score` queued paper with `ResearchAgent.process_paper`, then
   `publish`: Library folder + `paper.json` + pgvector chunks (`src/research/publish.py`), per-paper analysis and
   living report (`src/research/report_render.py`), LLM narrative every `NARRATIVE_EVERY` papers;
5. with an empty queue: search (topic analysis first time, expansion queries afterwards), `screen(continuous=True)`,
   OpenAlex enrichment (`src/research/enrich.py`), ranking (`src/research/ranking.py`); nothing new → wait 12 h.

The session is "claimed" in `JobManager`, so manual jobs on it return 409 while the autopilot owns it. Deleting a
paper from the Library rejects it in the session and retracts its gap evidence.

## 5. Prompt design (all prompts are general; they adapt per topic)

Every prompt is one template for all topics; runtime inputs make it adapt:

* **Topic and focus** — the user's focus is a *binding* scope statement in topic analysis, query generation and
  relevance screening (exclusions override the model's judgement).
* **Field profile** — the topic analysis also returns the field's typical `designs`, the `key_quantities` its
  papers must report exactly (e.g. ORs with 95% CIs; F1 on named datasets; regression coefficients with SEs) and its
  `quality_signals`. `llm_tasks.field_guidance()` injects this into the note-taking, summary, comparison and gap
  prompts; a missing quality signal is a valid methodological gap. The Ask/Report pipeline gets the same from its
  classifier (`agents/common.domain_text`). Older topics are back-filled once.
* **Paper type** — reviews, position papers and dataset papers get section guidance (`paper_type_note`).
* **Examples from several fields** (computer science, clinical epidemiology, economics), never one field only.

House rules shared by all prompts: inputs in XML-style tags and scrubbed (`prompts.data`) so scraped text cannot
close a block; untrusted text is data, never instructions; grounding (no invented papers/numbers/IDs, "not
reported" instead of guessing); no computed numbers presented as the authors'; critic, not cheerleader; exact JSON
contracts with verdict/reason first where truncation could cut them off.

Code backs the prompts where a rule must hold: cumulative synthesis merge (`synthesis.merge_synthesis`), citation
verification for Ask (`agents/citations.py`), deterministic gap confidence, rejection of fabricated IDs, retries for
empty or truncated output.

Quality is measured with `evals/` (two fields: LLM toxicity detection, and public health/economics).
