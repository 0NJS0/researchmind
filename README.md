# ResearchMind

Multi-paper research analysis platform. Upload PDFs, index them into a vector database, and generate per-paper or cross-paper research reports using a self-hosted LLM.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────┐
│  Streamlit  │────▶│  FastAPI     │────▶│  Qdrant   │     │  LLM    │
│  (src/ui)   │     │  (src/api)   │     │  (vector  │     │ (self-  │
│   :8501     │◀────│   :8000      │◀────│   DB)     │     │ hosted) │
└─────────────┘     └──────┬───────┘     └───────────┘     └─────────┘
                           │
                    ┌──────┴───────┐
                    │  LangGraph   │
                    │  Pipeline    │
                    │ (src/graph)  │
                    └──────────────┘
```

## Project Structure

```
researchmind/
├── src/
│   ├── api/server.py         # FastAPI endpoints (query, upload, ingest, report, CRUD)
│   ├── ui/app.py             # Streamlit frontend
│   ├── config.py             # Environment variable loading
│   ├── graph/
│   │   ├── state.py          # ResearchState TypedDict
│   │   └── workflow.py       # LangGraph pipeline definition
│   ├── agents/
│   │   ├── retriever.py      # Qdrant document retrieval
│   │   ├── analyzer.py       # Per-paper analysis
│   │   ├── comparison.py     # Cross-paper comparison
│   │   ├── gap_detector.py   # Research gap identification
│   │   └── report.py         # Final report generation
│   ├── ingest/
│   │   ├── __main__.py       # CLI ingestion entry point
│   │   ├── loader.py         # PDF loading via PyMuPDF
│   │   ├── chunker.py        # Text splitting
│   │   ├── embeddings.py     # BGE-M3 embedding model
│   │   └── vectorstore.py    # Qdrant operations
│   └── llm/
│       └── llm.py            # LLM client (OpenAI-compatible)
├── pyproject.toml
├── uv.lock
├── .env                      # Configuration (gitignored)
├── .gitignore
├── .python-version
└── papers/                   # PDFs and generated reports (gitignored)
    └── <topic>/
        ├── _reports/              # Combined reports
        │   └── report_<ts>.md
        └── <paper_id>/
            ├── paper.pdf
            └── _reports/
                └── <paper_id>_analysis.md   # Per-paper analysis
```

## How It Works

### Ingestion Pipeline

1. **Upload** a PDF through the UI or place it in `papers/<topic>/<paper_id>/`
2. **Loader** extracts text via PyMuPDF
3. **Chunker** splits text into 1200-character chunks (200 overlap)
4. **Embeddings** (`BAAI/bge-m3`) converts chunks to 1024-dim vectors
5. **Qdrant** stores vectors with metadata (`paper_id`, `topic`, `source`)

### Query Pipeline (LangGraph)

When you ask a question, a 5-stage LangGraph pipeline runs:

```
retriever ─▶ analysis ─▶ comparison ─▶ gap ─▶ report ─▶ END
```

| Stage | Agent | Description |
|---|---|---|
| 1 | **Retriever** | Searches Qdrant for the top-k chunks relevant to the question, filtered by topic |
| 2 | **Analyzer** | For each paper, extracts dataset, model architecture, training method, metrics, results |
| 3 | **Comparison** | Produces a comparison table across papers (datasets, models, methods, strengths, weaknesses) |
| 4 | **Gap Detector** | Identifies research limitations, missing experiments, unsolved problems, future directions |
| 5 | **Report** | Composes the final structured report from analysis, comparison, and gaps |

### Batch Report Pipeline

The `/report` endpoint uses a **map-reduce** approach to stay within the LLM's context window:

1. **Map**: Retrieves all chunks (k=100) for a topic, groups by `paper_id`, analyzes each paper individually
2. **Save**: Each individual analysis is saved to `papers/<topic>/<paper_id>/_reports/<paper_id>_analysis.md`
3. **Reduce**: All individual analyses are combined and the LLM generates a cross-paper report
4. **Save**: The combined report is saved to `papers/<topic>/_reports/report_<timestamp>.md`

Reports run in a background task and persist to disk — they survive server restarts and can be viewed from the UI.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **Qdrant** — vector database running on `localhost:6333`
- **LLM** — OpenAI-compatible API endpoint (e.g., [llama.cpp](https://github.com/ggml-org/llama.cpp) server, [vLLM](https://github.com/vllm-project/vllm), [Ollama](https://ollama.com/), etc.)

### Quick Qdrant (Docker)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd researchmind

# Create virtual environment and install
uv sync

# Configure environment
cp .env.example .env   # then edit .env
```

### `.env` Configuration

```
LLM_URL=http://localhost:3000/v1        # Your LLM endpoint
LLM_MODEL=your-model-name               # Model name (e.g., llama3.2, gemma2, etc.)
LLM_API_KEY=none                        # API key (use "none" if not required)
QDRANT_URL=http://localhost:6333        # Qdrant endpoint
COLLECTION_NAME=papers                  # Qdrant collection name
PAPERS_DIR=/path/to/papers/             # Directory for PDFs and reports
```

## Running

Start both services in separate terminals:

```bash
# Terminal 1: API server
uv run uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI
uv run streamlit run src/ui/app.py --server.port 8501
```

### CLI Ingestion

Alternatively, ingest papers from the command line:

```bash
# Ingest all topics
uv run python -m src.ingest

# Ingest a specific topic
uv run python -m src.ingest --topic LLM_Toxicity_Detection
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check (config, Qdrant, LLM) |
| `POST` | `/upload` | Upload a single PDF (`file` + `topic` form params) |
| `POST` | `/ingest` | Batch re-index all papers for a topic |
| `POST` | `/query` | Run the LangGraph pipeline on a question |
| `POST` | `/query/stream` | SSE-streaming version of query |
| `POST` | `/report` | Generate a batch report (map-reduce, background) |
| `GET` | `/papers` | List indexed papers (optionally filtered by topic) |
| `GET` | `/papers/report` | Get a single paper's analysis report |
| `DELETE` | `/papers` | Delete a paper (with `paper_id`) or clear all (no params) |
| `GET` | `/topics` | List all topics |
| `POST` | `/topics` | Create a new topic |
| `DELETE` | `/topics/{topic}` | Delete a topic and all its papers |
| `GET` | `/reports` | List all saved reports for a topic |
| `GET` | `/reports/content` | Get the content of a saved report by file path |

## UI Features

- **Topic management**: Create, select, and delete research topics
- **PDF upload**: Upload papers directly into a topic
- **Sidebar paper list**: View indexed papers with chunk counts, per-paper analysis, and delete
- **Ask questions**: Run the multi-stage LangGraph pipeline
- **Batch report**: Generate comprehensive cross-paper reports (background, async)
- **Saved Reports**: Browse and view per-paper analyses and combined reports from disk

## Key Design Decisions

- **Map-reduce over single-pass**: LLM context is 32K tokens. Single-pass with k=100 chunks (~30K tokens) overflows. Map-reduce analyzes each paper individually then combines — no information loss.
- **Background report generation**: Reports can take 20-30+ minutes per LLM call. Background tasks with file writes avoid UI timeouts.
- **Persistent file storage**: Reports are saved as markdown files — survive restarts, viewable offline.
- **Qdrant scroll API**: Uses `scroll_filter=` (not `filter=`) for qdrant-client v1.18+ compatibility.
- **Nested metadata**: All Qdrant payloads use `metadata.paper_id`, `metadata.topic` keys for consistent access.

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` / `uvicorn` | API server |
| `streamlit` | Web UI |
| `qdrant-client` | Vector database client |
| `langchain-qdrant` | LangChain Qdrant integration |
| `langchain-community` | Document loaders, embeddings |
| `sentence-transformers` | BGE-M3 embedding model |
| `langgraph` | LLM pipeline orchestration |
| `pymupdf` | PDF text extraction |
| `python-dotenv` | Environment configuration |
| `langchain-openai` | OpenAI-compatible LLM client |
