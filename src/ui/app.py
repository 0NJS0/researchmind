import os
import streamlit as st
import requests

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_healthy():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return False, {}


def fetch_topics():
    try:
        r = requests.get(f"{API_URL}/topics", timeout=10)
        r.raise_for_status()
        return r.json().get("topics", [])
    except requests.exceptions.RequestException:
        return []


def fetch_papers(topic):
    try:
        params = {"topic": topic} if topic else {}
        r = requests.get(f"{API_URL}/papers", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("papers", [])
    except requests.exceptions.RequestException:
        return []


def trigger_ingest(topic):
    try:
        params = {"topic": topic} if topic else {}
        r = requests.post(f"{API_URL}/ingest", params=params, timeout=600)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


st.title("ResearchMind")

# ── Initialize session state ──
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "paper_report" not in st.session_state:
    st.session_state.paper_report = None

# ── Sidebar ──
with st.sidebar:
    st.header("Status")
    healthy, health_data = api_healthy()
    if healthy:
        qdrant_ok = health_data.get("qdrant", {}).get("ok", False)
        if qdrant_ok:
            st.markdown("🟢 API · 🟢 Qdrant")
        else:
            qdrant_msg = health_data.get("qdrant", {}).get("message", "")
            st.markdown("🟢 API · 🔴 Qdrant",
                        help=qdrant_msg)
            st.error(qdrant_msg)
    else:
        st.markdown("🔴 Disconnected")
        st.warning("Start the API server:\n\n"
                    "`uv run uvicorn src.api.server:app --reload`")

    st.divider()

    # Topic selector
    st.header("Topic")
    topics = fetch_topics() if healthy else []

    selected = st.selectbox(
        "Select topic",
        options=[""] + topics,
        format_func=lambda t: t if t else "— Select a topic —",
        key="topic",
    )

    new_topic = st.text_input("Create new topic", placeholder="topic-name")
    if st.button("Create Topic", use_container_width=True) and new_topic:
        try:
            r = requests.post(f"{API_URL}/topics", params={"topic": new_topic.strip()}, timeout=10)
            r.raise_for_status()
            st.success(f"Topic '{new_topic.strip()}' created")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to create topic: {e}")

    if selected:
        with st.popover("🗑️ Delete Topic", use_container_width=True):
            st.warning(f"Delete topic '{selected}' and all its papers?")
            if st.button("Yes, delete topic", type="primary", use_container_width=True):
                try:
                    r = requests.delete(f"{API_URL}/topics/{selected}", timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    if data.get("status") == "deleted":
                        st.success(f"Deleted topic '{selected}'")
                        st.session_state.topic = ""
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {data.get('error', 'unknown')}")
                except Exception as e:
                    st.error(f"Failed to delete topic: {e}")

    st.divider()

    # Paper list (topic-scoped)
    st.header("Indexed Papers")
    papers = fetch_papers(selected) if healthy and selected else []
    if papers:
        for p in papers:
            has_rpt = p.get("has_report", False)
            cols = st.columns([0.65, 0.15, 0.2])
            with cols[0]:
                st.markdown(f"📄 **{p['paper_id']}**  \n_{p['source']}_ · {p['chunks']} chunks")
            with cols[1]:
                if has_rpt:
                    if st.button("📄", key=f"rpt_{p['paper_id']}", help="View paper analysis"):
                        try:
                            r = requests.get(
                                f"{API_URL}/papers/report",
                                params={"paper_id": p["paper_id"], "topic": selected},
                                timeout=10,
                            )
                            r.raise_for_status()
                            data = r.json()
                            if data.get("status") == "ok":
                                st.session_state.paper_report = {
                                    "paper_id": p["paper_id"],
                                    "content": data["report"],
                                }
                                st.rerun()
                            else:
                                st.error(data.get("error", "Failed to load report"))
                        except Exception as e:
                            st.error(f"Failed to load report: {e}")
            with cols[2]:
                if st.button("✕", key=f"del_{p['paper_id']}", help="Delete this paper"):
                    try:
                        r = requests.delete(
                            f"{API_URL}/papers",
                            params={"paper_id": p["paper_id"], "topic": selected},
                            timeout=10,
                        )
                        r.raise_for_status()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete paper: {e}")

        if st.button("🔄 Re-index papers", use_container_width=True):
            with st.spinner(f"Re-indexing {selected} ..."):
                result = trigger_ingest(selected)
                if result.get("status") == "indexed":
                    st.success(f"{result['chunks']} chunks from {result['pages']} pages")
                elif result.get("status") == "no papers found":
                    st.info(f"No papers found in topic '{selected}'")
                else:
                    msg = result.get("error", "unknown")
                    st.error(f"Ingest failed: {msg}")
            st.rerun()

        with st.popover("🧹 Clear All", use_container_width=True):
            st.warning("Delete all topics and papers?")
            if st.button("Yes, clear everything", type="primary", use_container_width=True):
                try:
                    r = requests.delete(f"{API_URL}/papers", timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    if data.get("status") == "cleared":
                        st.success("All papers and topics cleared")
                        st.session_state.topic = ""
                        st.rerun()
                    else:
                        st.error(f"Clear failed: {data.get('error', 'unknown')}")
                except Exception as e:
                    st.error(f"Failed to clear all: {e}")
    elif healthy and selected:
        st.info("No papers indexed yet for this topic.")
    elif healthy and not selected:
        st.info("Select a topic above.")
    else:
        st.info("API not connected.")

# ── Main area ──

# Upload (topic-scoped)
uploaded = st.file_uploader("Upload Paper", type=["pdf"], disabled=not selected, key="pdf_upload")

if uploaded and selected and st.session_state.get("last_file") != uploaded.name:
    try:
        files = {"file": (uploaded.name, uploaded, "application/pdf")}
        resp = requests.post(f"{API_URL}/upload", params={"topic": selected}, files=files, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.last_file = uploaded.name
        if data.get("status") == "error" or "error" in data:
            st.error(f"Upload issue: {data.get('error', data.get('status', 'unknown'))}")
        elif data.get("status") == "already indexed":
            st.info(f"Already indexed: '{data.get('paper_id', uploaded.name)}'")
        else:
            st.success(f"Uploaded and indexed under '{selected}'")
            del st.session_state["pdf_upload"]
            st.rerun()
    except requests.exceptions.RequestException:
        st.error("Could not connect to the API server. Is it running?")
    except Exception as e:
        st.error(f"Upload failed: {e}")
elif uploaded and not selected:
    st.warning("Select a topic before uploading.")

# Query (topic-scoped)
question = st.text_input("Ask a research question", disabled=not selected)

col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze", disabled=not selected, use_container_width=True) and selected:
        if not question:
            st.warning("Please enter a question first.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "topic": selected},
                    timeout=300,
                )
                response.raise_for_status()
                st.markdown(response.json().get("report", "No report returned."))
            except requests.exceptions.RequestException:
                st.error("Could not connect to the API server. Is it running?")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

with col2:
    if st.button("📄 Report", disabled=not selected, use_container_width=True) and selected:
        try:
            response = requests.post(
                f"{API_URL}/report",
                params={"topic": selected},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "started":
                st.info("Report generation started — check the Saved Reports section below for individual paper analyses and the combined report as they complete.")
            else:
                st.error(data.get("message", "Report generation failed."))
        except Exception as e:
            st.error(f"Failed to start report: {e}")

# ── Saved Reports (topic-scoped) ──
if selected:
    st.divider()
    st.subheader("Saved Reports")
    try:
        r = requests.get(f"{API_URL}/reports", params={"topic": selected}, timeout=10)
        r.raise_for_status()
        reports_data = r.json()
        if reports_data.get("status") == "ok" and reports_data["reports"]:
            report_options = {rp["label"]: rp for rp in reports_data["reports"]}
            selected_label = st.selectbox(
                "Select a report to view",
                options=list(report_options.keys()),
                key="selected_report",
            )
            if selected_label:
                rp = report_options[selected_label]
                if st.button("Load Report", use_container_width=True):
                    try:
                        resp = requests.get(
                            f"{API_URL}/reports/content",
                            params={"file": rp["file"]},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        if data.get("status") == "ok":
                            st.session_state.saved_report = data["content"]
                            st.rerun()
                        else:
                            st.error(data.get("error", "Failed to load report"))
                    except Exception as e:
                        st.error(f"Failed to load report: {e}")
        else:
            st.info("No reports saved yet. Use the Report button above.")
    except Exception as e:
        st.error(f"Failed to fetch reports: {e}")

if st.session_state.get("saved_report"):
    st.divider()
    st.markdown(st.session_state.saved_report)
    if st.button("Close saved report"):
        st.session_state.saved_report = None
        st.rerun()

# Display individual paper report when selected
if st.session_state.get("paper_report"):
    rpt = st.session_state.paper_report
    st.divider()
    st.subheader(f"Analysis: {rpt['paper_id']}")
    st.markdown(rpt["content"])
    if st.button("Close report"):
        st.session_state.paper_report = None
        st.rerun()
