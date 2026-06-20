import json
import logging
import shutil
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pathlib import Path
import asyncio
from src.graph.workflow import research_graph
from src.ingest.loader import load_papers
from langchain_community.document_loaders import PyMuPDFLoader
from src.ingest.chunker import chunk_documents
from src.ingest.vectorstore import add_documents, list_papers, check_qdrant, paper_exists, delete_paper, list_topics, delete_topic_papers
from src.llm.llm import check_llm
from src.config import PAPERS_DIR, validate_config
from starlette.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_HISTORY = 5


def _serialize(obj):
    if hasattr(obj, "page_content"):
        return {"page_content": obj.page_content, "metadata": obj.metadata}
    if hasattr(obj, "content"):
        return {"content": obj.content}
    return str(obj)

logger = logging.getLogger(__name__)

conversations: dict[str, list[dict]] = defaultdict(list)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    topic: str = ""


@app.get("/health")
async def health():
    config_ok = True
    config_msg = ""

    try:
        validate_config()
    except ValueError as e:
        config_ok = False
        config_msg = str(e)
        

    qdrant_ok, qdrant_msg = await asyncio.to_thread(check_qdrant)
    llm_ok,llm_msg = await asyncio.to_thread(check_llm)

    all_ok=qdrant_ok and llm_ok and config_ok

    message = "all services healthy" if all_ok else "one or more services degraded"

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy" if all_ok else "degraded",
            "config": {"ok": config_ok, "message": config_msg if not config_ok else "validated"},
            "qdrant": {"ok": qdrant_ok, "message": qdrant_msg},
            "llm": {"ok": llm_ok, "message": llm_msg},
            "message": message,
        },
    )


@app.post("/query")
async def query(request: QueryRequest):
    try:
        history = ""
        if request.session_id:
            prev = conversations[request.session_id]
            if prev:
                history = "Previous conversations:\n" + "\n".join(
                    f"Q: {h['question']}\nA: {h['report'][:500]}"
                    for h in prev[-MAX_HISTORY:]
                ) + "\n\n"

        result = await asyncio.to_thread(
            research_graph.invoke,
            {
                "question": request.question,
                "documents": [],
                "analysis": "",
                "comparison": "",
                "gaps": "",
                "report": "",
                "history": history,
                "topic": request.topic,
                "k": 10,
            },
        )
        report = result.get("report", "No report generated.")

        if request.session_id:
            conversations[request.session_id].append(
                {"question": request.question, "report": report}
            )

        return {"report": report}
    except Exception as e:
        return {"report": "Pipeline failed. Check server logs for details."}


def _generate_report_sync(topic: str) -> str:
    from src.llm.llm import llm as llm_model
    from src.ingest.vectorstore import get_retriever
    from collections import defaultdict

    retriever = get_retriever(k=100, topic=topic)
    docs = retriever.invoke(
        "Generate a comprehensive research report covering all papers: "
        "comparative analysis of methodologies, key findings, research gaps, and future directions"
    )
    if not docs:
        return "No documents found."

    # Group chunks by paper_id
    papers: dict[str, list] = defaultdict(list)
    for d in docs:
        pid = d.metadata.get("paper_id", "unknown")
        papers[pid].append(d)

    logger.info("Report: analyzing %d papers (%d chunks)", len(papers), len(docs))

    # Map step: analyze each paper individually
    paper_analyses = {}
    for pid, chunks in papers.items():
        context = "\n".join(c.page_content for c in chunks)
        prompt = f"""
Analyze this research paper and extract:
1. Dataset used
2. Model architecture
3. Training method
4. Evaluation metrics
5. Main results and findings

Paper ({pid}):
{context}

Provide a concise summary.
"""
        try:
            resp = llm_model.invoke(prompt)
            paper_analyses[pid] = resp.content
        except Exception as e:
            paper_analyses[pid] = f"Analysis failed for {pid}: {e}"

        # Save individual paper analysis
        paper_reports_dir = Path(PAPERS_DIR) / topic / pid / "_reports"
        paper_reports_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = paper_reports_dir / f"{pid}_analysis.md"
        with open(analysis_file, "w") as f:
            f.write(paper_analyses[pid])
        logger.info("Saved analysis for '%s' to %s", pid, analysis_file)

    combined = "\n\n".join(f"## {pid}\n{analysis}" for pid, analysis in paper_analyses.items())

    # Reduce step: compare and generate final report
    reduce_prompt = f"""
You are a senior research advisor. Based on the following individual paper analyses, create a comprehensive report.

{combined}

Format the report:
# Summary

# Methodology Comparison
| Paper | Dataset | Model Architecture | Training Method | Evaluation Metrics |

# Key Findings

# Comparative Analysis

# Research Gaps

# Future Work
"""
    try:
        result = llm_model.invoke(reduce_prompt)
        return result.content
    except Exception as e:
        return f"Report generation failed: {e}"


@app.post("/report")
async def report(topic: str = ""):
    if not topic:
        return {"status": "error", "error": "Topic is required."}

    async def run_report():
        try:
            report_text = await asyncio.to_thread(_generate_report_sync, topic)
            reports_dir = Path(PAPERS_DIR) / topic / "_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            filepath = reports_dir / filename
            with open(filepath, "w") as f:
                f.write(report_text)
            logger.info("Combined report saved to %s", filepath)
        except Exception as e:
            logger.exception("Background report generation failed")

    asyncio.create_task(run_report())
    logger.info("Report generation started in background for topic '%s'", topic)
    return {"status": "started", "message": "Report generation started. Individual paper analyses and the combined report will appear in Saved Reports as they complete."}


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def event_generator():
        history = ""
        if request.session_id:
            prev = conversations[request.session_id]
            if prev:
                history = "Previous conversations:\n" + "\n".join(
                    f"Q: {h['question']}\nA: {h['report'][:500]}"
                    for h in prev[-MAX_HISTORY:]
                ) + "\n\n"

        initial_state = {
            "question": request.question,
            "documents": [],
            "analysis": "",
            "comparison": "",
            "gaps": "",
            "report": "",
            "history": history,
            "topic": request.topic,
            "k": 10,
        }
        try:
            async for event in research_graph.astream(initial_state):
                for node, output in event.items():
                    yield f"data: {json.dumps({'node': node, 'output': output}, default=_serialize)}\n\n"
        except Exception:
            yield f"data: {json.dumps({'node': 'error', 'output': 'Pipeline failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest(topic: str | None = None):
    try:
        if topic:
            logger.info("Clearing existing points for topic '%s'", topic)
            await asyncio.to_thread(delete_topic_papers, topic)

        logger.info("Loading papers from %s [topic: %s]", PAPERS_DIR, topic or "all")
        docs = await asyncio.to_thread(load_papers, PAPERS_DIR, topic=topic)
        if not docs:
            logger.info("No papers found")
            return {"status": "no papers found", "pages": 0, "chunks": 0}
        logger.info("Loaded %d pages", len(docs))

        logger.info("Chunking documents...")
        chunks = await asyncio.to_thread(chunk_documents, docs)
        logger.info("Split into %d chunks", len(chunks))

        logger.info("Adding to Qdrant...")
        await asyncio.to_thread(add_documents, chunks)
        logger.info("Done")

        return {
            "status": "indexed",
            "pages": len(docs),
            "chunks": len(chunks),
        }
    except Exception as e:
        logger.exception("Ingest failed")
        return {"status": "error", "error": "Ingest failed. Check server logs for details."}


@app.get("/papers")
async def papers(topic: str | None = None):
    try:
        result = await asyncio.to_thread(list_papers, topic=topic)
        # Check if each paper has a saved analysis
        for p in result:
            report_path = Path(PAPERS_DIR) / topic / p["paper_id"] / "_reports" / f"{p['paper_id']}_analysis.md"
            p["has_report"] = report_path.exists()
        return {"papers": result}
    except Exception as e:
        logger.error("Papers failed: %s", e)
        return {"papers": [], "error": "Failed to list papers. Check server logs for details."}


@app.get("/papers/report")
async def paper_report(paper_id: str = "", topic: str = ""):
    if not paper_id or not topic:
        return {"status": "error", "error": "paper_id and topic are required."}
    report_path = Path(PAPERS_DIR) / topic / paper_id / "_reports" / f"{paper_id}_analysis.md"
    if not report_path.exists():
        return {"status": "error", "error": "No report found. Generate a report first."}
    try:
        content = report_path.read_text()
        return {"status": "ok", "report": content, "paper_id": paper_id}
    except Exception as e:
        logger.exception("Failed to read paper report")
        return {"status": "error", "error": "Failed to read report."}


@app.get("/reports")
async def list_reports(topic: str = ""):
    if not topic:
        return {"status": "error", "error": "Topic is required."}
    reports = []
    topic_dir = Path(PAPERS_DIR) / topic

    # Combined reports
    combined_dir = topic_dir / "_reports"
    if combined_dir.exists():
        for f in sorted(combined_dir.glob("report_*.md"), reverse=True):
            ts = f.stem.replace("report_", "").replace("_", " ")[:15]
            reports.append({
                "type": "combined",
                "label": f"Combined Report ({ts})",
                "file": str(f),
                "filename": f.name,
                "paper_id": None,
            })

    # Individual paper reports
    for paper_dir in topic_dir.iterdir():
        if not paper_dir.is_dir() or paper_dir.name.startswith("_"):
            continue
        rpt_dir = paper_dir / "_reports"
        if rpt_dir.exists():
            for f in rpt_dir.glob("*_analysis.md"):
                reports.append({
                    "type": "paper",
                    "label": f"Paper: {paper_dir.name}",
                    "file": str(f),
                    "filename": f.name,
                    "paper_id": paper_dir.name,
                })

    return {"status": "ok", "reports": reports}


@app.get("/reports/content")
async def report_content(file: str = ""):
    if not file:
        return {"status": "error", "error": "File path is required."}
    path = Path(file)
    if not path.exists() or not path.is_relative_to(PAPERS_DIR):
        return {"status": "error", "error": "Invalid file path."}
    try:
        content = path.read_text()
        return {"status": "ok", "content": content, "file": file}
    except Exception as e:
        return {"status": "error", "error": f"Failed to read report: {e}"}


@app.post("/upload")
async def upload_pdf(file: UploadFile, topic: str = ""):
    if not file.filename:
        return {"status": "error", "error": "No filename provided."}
    if not topic:
        return {"status": "error", "error": "Topic is required."}

    raw_name = file.filename
    if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
        return {"status": "error", "error": "Invalid filename"}

    safe_name = Path(raw_name).name
    paper_name = Path(safe_name).stem.replace(" ", "_")
    paper_dir = Path(PAPERS_DIR) / topic / paper_name
    paper_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = paper_dir / safe_name
    header = await file.read(4)
    if not header.startswith(b"%PDF"):
        await file.close()
        return {"status": "error", "error": "File is not a valid PDF."}

    bytes_written = len(header)
    try:
        with pdf_path.open("wb") as f:
            f.write(header)
            while chunk := await file.read(65536):
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    pdf_path.unlink(missing_ok=True)
                    return {"status": "error", "error": "File too large."}
                await asyncio.to_thread(f.write, chunk)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        pdf_path.unlink(missing_ok=True)
        return {"status": "error", "error": "Failed to save file."}

    try:
        logger.info("Indexing uploaded file %s", file.filename)

        exists = await asyncio.to_thread(paper_exists, paper_name, topic)
        if exists:
            logger.info("Paper '%s' already indexed in topic '%s', skipping", paper_name, topic)
            return {"status": "already indexed", "paper_id": paper_name}

        loader = PyMuPDFLoader(str(pdf_path))
        docs = await asyncio.to_thread(loader.load)
        for d in docs:
            d.metadata["paper_id"] = paper_name
            d.metadata["source"] = file.filename
            d.metadata["topic"] = topic
        logger.info("Loaded %d pages", len(docs))

        chunks = await asyncio.to_thread(chunk_documents, docs)
        logger.info("%d chunks, adding to Qdrant...", len(chunks))
        await asyncio.to_thread(add_documents, chunks)
        logger.info("Done")
    except Exception as e:
        logger.exception("Upload indexing failed")
        return {"status": "uploaded but indexing failed", "error": "Indexing failed. Check server logs for details."}

    return {"status": "uploaded and indexed"}


@app.delete("/papers")
async def delete_paper_endpoint(paper_id: str | None = None, topic: str = ""):
    if paper_id is None:
        # Clear all topics and papers
        try:
            topics = await asyncio.to_thread(list_topics)
            for t in topics:
                await asyncio.to_thread(delete_topic_papers, t)
                topic_dir = Path(PAPERS_DIR) / t
                if topic_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, topic_dir, ignore_errors=True)
            logger.info("Cleared all papers and topics")
            return {"status": "cleared"}
        except Exception as e:
            logger.exception("Failed to clear all")
            return {"status": "error", "error": "Failed to clear all."}

    if not topic:
        return {"status": "error", "error": "Topic is required."}
    paper_dir = Path(PAPERS_DIR) / topic / paper_id
    try:
        if paper_dir.exists():
            await asyncio.to_thread(shutil.rmtree, paper_dir, ignore_errors=True)
        await asyncio.to_thread(delete_paper, paper_id, topic)
        logger.info("Deleted paper '%s'", paper_id)
        return {"status": "deleted", "paper_id": paper_id}
    except Exception as e:
        logger.exception("Failed to delete paper '%s'", paper_id)
        return {"status": "error", "error": "Failed to delete paper."}


@app.get("/topics")
async def get_topics():
    try:
        topics = await asyncio.to_thread(list_topics)
        return {"topics": topics}
    except Exception as e:
        logger.exception("Failed to list topics")
        return {"topics": [], "error": "Failed to list topics."}


@app.delete("/topics/{topic}")
async def delete_topic(topic: str):
    if not topic:
        return {"status": "error", "error": "Topic name is required."}
    try:
        await asyncio.to_thread(delete_topic_papers, topic)
        topic_dir = Path(PAPERS_DIR) / topic
        if topic_dir.exists():
            await asyncio.to_thread(shutil.rmtree, topic_dir, ignore_errors=True)
        logger.info("Deleted topic '%s'", topic)
        return {"status": "deleted", "topic": topic}
    except Exception as e:
        logger.exception("Failed to delete topic '%s'", topic)
        return {"status": "error", "error": "Failed to delete topic."}


@app.post("/topics")
async def create_topic(topic: str):
    if not topic or not topic.strip():
        return {"status": "error", "error": "Topic name is required."}
    safe = topic.strip().replace(" ", "_")
    topic_dir = Path(PAPERS_DIR) / safe
    try:
        topic_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created topic '%s'", safe)
        return {"status": "created", "topic": safe}
    except Exception as e:
        logger.exception("Failed to create topic")
        return {"status": "error", "error": "Failed to create topic."}
