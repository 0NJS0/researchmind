# ResearchMind

![ResearchMind living literature report](docs/screenshot/reports.png)

Research assistant for literature reviews:

- **Autopilot** (runs 24/7): choose one topic and it keeps finding new relevant papers through SearXNG. It
  processes them **one at a time, newest and best-published first** (ranked by relevance, recency, citations
  and peer-review status via OpenAlex). Each finished paper is added to the Library and to the topic's
  **living report**. Changing the topic takes effect after the current paper.
- **Library**: upload PDFs into topics. Text is split into passages, embedded, and stored in PostgreSQL with pgvector.
- **Ask**: ask questions about a topic. Answers are grounded in the retrieved passages and carry page-level citations (`[paper, p.4]`) that you can click.
- **Reports**: a map-reduce literature report. Each paper is analysed, the papers are compared, research gaps are identified, and everything is synthesised.
- **Topics & research gaps** (on the Autopilot page): *New topic* creates the Library topic and puts it on the
  autopilot (if it is busy with another topic, it switches after the current paper). The active topic's queue,
  latest papers, evolving evidence-backed research gaps, synthesis and activity are shown as tabs; *All topics*
  opens any topic's full view (papers, gaps, synthesis, searches).

It runs as a single container: a FastAPI backend that also serves the React UI. PostgreSQL, SearXNG and the LLM (OpenRouter) are external services.

![ResearchMind architecture](docs/screenshot/architecture.png)

## Screenshots

**Autopilot**: researches the active topic around the clock, shows the queue, recently added papers, gaps, synthesis and the free-tier quota.

![Autopilot](docs/screenshot/autopilot.png)

**Research gaps**: gaps harvested from the papers, each with its evidence. Confidence comes from independent supporting papers, not from the model, and each gap is checked against the wider literature to see whether it is still open or already addressed elsewhere.

![Research gaps](docs/screenshot/gaps.png)

**Papers**: every paper in a topic with its peer-review status, citations, linked gaps and full-text links, plus the papers the screening rejected.

![Papers](docs/screenshot/library.png)

**Topic validator**: stress-tests a research idea (is it specific, has it been done, how crowded is the field) before you commit to it. When an idea is too broad, it asks the questions that would narrow it down instead of choosing a topic for you.

![Topic validator](docs/screenshot/topic_validator.png)

## Configuration

Copy `.env.example` to `.env` and fill it in. **Nothing secret is hardcoded, and `.env` is git-ignored.**

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | OpenRouter API key |
| `LLM_MODEL`, `LLM_FALLBACK_MODELS` | Main chat model and models to fall back on when it is rate limited (free models end in `:free`) |
| `EMBEDDING_MODEL` | OpenRouter embedding model (default `nvidia/nemotron-3-embed-1b:free`) |
| `DATABASE_URL` | `postgresql://user:pass@host:5432/db`. The app creates the `vector` extension and its tables on startup. |
| `SEARXNG_URL` | Your SearXNG instance. It must have `json` enabled under `search.formats`. |

Optional tuning variables are documented inline in `.env.example`.

**PostgreSQL:** the pgvector extension must be installed on the server. If the app's database role is not allowed to create extensions, run `CREATE EXTENSION vector;` once as a superuser.

**Changing `EMBEDDING_MODEL`** makes the stored vectors incompatible. Use *Library → ⋯ → Re-index entire library*, or run `uv run python -m src.ingest --reset`.

### About the OpenRouter free tier

Free models are rate limited (about 20 requests per minute, plus a daily cap). The client handles this:

- Requests are spaced out (`LLM_MIN_INTERVAL`).
- A 429 is retried after the `Retry-After` delay, then the next fallback model is tried.
- When the **daily** cap is hit, research sessions stop with status **Paused**. Finished work is kept, and *Continue research* resumes after the quota resets.
- Most free models are reasoning models. Reasoning is excluded from replies, `<think>` blocks are stripped, and JSON mode is only requested from models that support it.

## Running

### Docker (homelab)

```bash
cp .env.example .env    # then edit
mkdir -p ./data/papers && sudo chown -R 1000:1000 ./data   # papers folder (or set PAPERS_HOST_DIR in .env)
docker compose up -d --build
# UI + API on http://<homelab-ip>:47813  (port set in docker-compose.yml / RESEARCHMIND_PORT)
```

### Local development

```bash
uv sync                                  # Python deps
uv run uvicorn src.api.server:app --reload --port 8000

cd frontend && pnpm install && pnpm dev  # UI on http://localhost:5173 (proxies /api to :8000)
```

To index existing PDFs from the command line, lay them out as `papers/<topic>/<paper_id>/*.pdf` and run:

```bash
uv run python -m src.ingest [--topic NAME] [--reset]
```

## Tests

```bash
uv run pytest            # backend: uses an embedded Postgres + pgvector (pgserver), or TEST_DATABASE_URL if set
cd frontend && pnpm test && pnpm build
```

The backend tests use a fake LLM, a fake embedding model and mocked SearXNG and paper hosts, so they need no network access or API keys.

## How the autopilot works

1. You pick the **active topic** (and an optional focus) on the Autopilot page. Only that topic is researched.
2. The topic's uploaded papers are analysed first. Then the agent searches SearXNG, screens the results for
   relevance, enriches them from OpenAlex (citations, peer-reviewed vs preprint, open-access PDFs) and ranks them:
   `0.45·relevance + 0.25·recency + 0.20·citations-per-year + 0.10·peer-reviewed venue`.
3. It takes the **best queued paper**, processes it completely (full text → structured summary → comparison →
   research gaps → synthesis), saves it into the Library (PDF + index, so Ask can use it), writes its analysis,
   and re-renders the living report. Then it takes the next paper.
4. When the queue is empty it searches again, using queries derived from the papers and gaps found so far. If
   nothing new turns up, it waits `AUTOPILOT_RESEARCH_INTERVAL_HOURS` (default 12 h).
5. It stops for the day before the OpenRouter free quota runs out (`AUTOPILOT_QUOTA_RESERVE` requests are kept
   for your own questions) and resumes after the daily reset. It also resumes automatically after a restart:
   if PostgreSQL is not up yet (e.g. after a homelab reboot) the app keeps retrying and starts the autopilot once
   it connects; a paper that was interrupted is processed again (finished LLM steps come from the cache), and one
   that was analysed but not yet published is added to the Library and the living report.

A topic change requested while a paper is being processed is applied as soon as that paper is finished.

### Recent papers from top venues

Every search round also checks sources the web search misses, without any LLM requests: the official proceedings
of CVPR, ICCV, WACV (CVF open access), ECCV (ECVA) and NeurIPS for the current and previous year (titles matched
to the topic, abstracts fetched), arXiv newest-first (an "Accepted to CVPR 2026" comment becomes the venue) and
OpenAlex limited to the last two years. Top venues (CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, ACL, TPAMI, …;
add more with `TOP_VENUES_EXTRA`) rank highest, and a new top-venue paper is not penalised for having no
citations yet. Results the relevance check can't cover in one round stay queued for the next rounds (recent and
top-venue papers are checked first). Settings: `DISCOVERY_ENABLED`, `DISCOVERY_QUERIES`, `DISCOVERY_PROCEEDINGS_LIMIT`.

### Full text instead of abstracts

When a paper's own page only shows the abstract, the agent looks for a free copy of the full PDF: PDF links on
the page, known PDF link formats (arXiv, ACL Anthology, OpenReview, PMC, bioRxiv/medRxiv, NeurIPS, CVF, PMLR,
MDPI), every open-access copy OpenAlex lists (repositories, author copies, the published version of a preprint),
an arXiv title search and Semantic Scholar. Papers still left with only an abstract are checked again weekly,
and re-analysed from the full text when a free copy appears.

Some sites (e.g. SSRN) block automated downloads. On the **Papers** page, the *Needs full text* filter lists these
papers: download the PDF in your browser and use **Attach PDF** on the paper.
**Upload paper** (Papers page) takes a PDF with its link or DOI and title; if the topic already has that paper with only
its abstract, the PDF becomes its full text instead of a duplicate. The same file can't be uploaded twice. It is re-analysed from the full
text ahead of new papers.

## Prompt evaluation

`evals/` measures prompt quality against hand-checked facts from the uploaded papers
(`evals/gold/llm_toxicity_detection.json`, every fact with its page and source sentence). The `public_health_economics` eval needs papers that are not in this repo; see
[`evals/corpus/README.md`](evals/corpus/README.md) for where to get them:

```bash
uv run python -m evals.run                         # all checks, ~40-70 free LLM requests
uv run python -m evals.run --only relevance        # quick check
uv run python -m evals.run --model qwen/qwen3.8-27b:free   # compare another free model
```

It reports summary fact recall, summary statements with numbers that do not appear in the paper, whether Ask
answers contain the expected facts and cite the right paper, the share of citations that match a retrieved
passage, relevance-screening accuracy (including focus exclusions) and live gap quality. Results go to
`evals/results/` as JSON and Markdown, compared with the previous run. Re-run it whenever prompts or models change.

## API overview

All endpoints are under `/api`. `/health` is a liveness probe. `/api/health` reports the database, LLM, index and SearXNG. `/api/config` returns non-secret settings.

| Area | Endpoints |
|---|---|
| Library | `GET/POST /api/library/topics`, `DELETE /api/library/topics/{topic}`, `GET /api/library/papers`, `DELETE /api/library/papers/{topic}/{paper}`, `POST /api/library/upload?topic=`, `POST /api/library/ingest` |
| Ask | `POST /api/library/query`, `POST /api/library/query/stream` (SSE) |
| Reports | `POST /api/library/reports`, `GET /api/library/jobs/{id}`, `GET /api/library/reports?topic=`, `GET/DELETE /api/library/reports/{topic}/{file}` |
| Autopilot | `GET /api/autopilot`, `POST /api/autopilot/topic` {topic, focus}, `POST /api/autopilot/pause`, `…/resume`, `…/search-now`, `PUT /api/autopilot/topics/{topic}/focus` |
| Research | `GET /api/research`, `POST /api/research` (creates the topic and puts it on the autopilot), `GET/DELETE /api/research/{id}`, `POST /api/research/{id}/search` (put on autopilot / search now), `…/cancel`, `…/papers`, `…/gaps`, `…/synthesis`, `…/events/stream` (SSE), `GET /api/papers/{id}`, `GET /api/gaps/{id}` |

See [docs/RESEARCH_AGENT.md](docs/RESEARCH_AGENT.md) for how the research agent and the gap engine work.

## Project layout

```
src/
  api/          server.py (app, health, SPA) · library.py (library/Q&A/report API)
  agents/       document pipeline agents + prompts.py (structured, grounded prompts)
  graph/        LangGraph pipelines for Ask (workflow.py) and Reports (report_graph.py)
  ingest/       PDF loading, chunking, OpenRouter embeddings, pgvector store
  llm/          llm.py (OpenRouter/OpenAI-compatible/Anthropic client) · structured.py (JSON output + cache)
  research/     research agent: SearXNG, content fetching, relevance, gap engine, Postgres store
  db.py         connection pool + schema
  config.py     all settings (from the environment)
frontend/       React + Vite + Tailwind UI
tests/          pytest suite
```

## License

MIT, see [LICENSE](LICENSE).
