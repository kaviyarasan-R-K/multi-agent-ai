import os
import re
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from src.multi_agent_ai.pipelines.pipelines import run_search_pipeline

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Research Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Subtle Custom Styling ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    .hero-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.5rem;
    }
    .report-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# --- Sidebar: Configuration & Past Runs ---
with st.sidebar:
    st.header("⚙️ Agent Settings")

    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"],
        index=0,
    )
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.0, step=0.1)
    enable_revision = st.toggle("Auto-Revise on Critic Feedback", value=True)

    st.divider()
    st.subheader("🔑 API Key Status")
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.caption(f"{'✅' if gemini_ok else '❌'} `GEMINI_API_KEY`")
    st.caption(f"{'✅' if tavily_ok else '❌'} `TAVILY_API_KEY`")

    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Recent Reports")
        for idx, item in enumerate(reversed(st.session_state.history)):
            label = f"{item['topic'][:34]}{'...' if len(item['topic']) > 34 else ''}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.active_result = item

# --- Header ---
st.title("🔬 Multi-Agent Research Studio")
st.markdown(
    '<p class="hero-subtitle">Autonomous Search → Deep Web Scraping → Structured Drafting → Editorial Fact-Checking</p>',
    unsafe_allow_html=True,
)

# --- Quick Sample Topic Chips ---
sample_topics = [
    "Electric Vehicle Manufacturing Growth in Tamil Nadu",
    "Agentic AI Frameworks in Production 2026",
    "India Semiconductor Mission Progress & Investments",
]

cols = st.columns(len(sample_topics))
for col, sample in zip(cols, sample_topics):
    if col.button(f"✨ {sample[:32]}...", use_container_width=True):
        st.session_state.topic_input = sample

# --- Search Form ---
with st.form("research_form", clear_on_submit=False):
    topic = st.text_input(
        "Research Topic",
        value=st.session_state.topic_input,
        placeholder="Enter a topic, market trend, or technical question to research...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🚀 Run Research Pipeline", type="primary", use_container_width=True)

# --- Execute Pipeline ---
if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    elif not (gemini_ok and tavily_ok):
        st.error("Missing `GEMINI_API_KEY` or `TAVILY_API_KEY` in your `.env` file.")
    else:
        st.session_state.topic_input = topic
        with st.status("Running multi-agent research workflow...", expanded=True) as status:
            try:
                result = run_search_pipeline(topic.strip())
                result["timestamp"] = datetime.now().strftime("%I:%M %p")
                result["model"] = model_name
                st.session_state.active_result = result
                st.session_state.history.append(result)
                status.update(label="✅ Research & editorial review complete!", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Pipeline encountered an error", state="error", expanded=True)
                st.exception(exc)

# --- Render Results ---
result = st.session_state.active_result
if result:
    st.divider()

    # Top Summary Metrics
    approved = "APPROVED" in result["critique"].upper() and not result.get("revised")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", result.get("model", model_name))
    m2.metric("Critic Verdict", "Approved" if approved else "Revised")
    m3.metric("Report Length", f"{len(result['final_report'].split()):,} words")
    m4.metric("Completed", result.get("timestamp", "Just now"))

    # Output Tabs
    tab_report, tab_critic, tab_sources, tab_draft = st.tabs(
        ["📄 Final Report", "🧐 Critic Review", "🔎 Search & Scraped Data", "📝 Initial Draft"]
    )

    with tab_report:
        slug = re.sub(r"[^a-z0-9]+", "-", result["topic"].lower()).strip("-")[:40]
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=result["final_report"],
            file_name=f"{slug or 'research-report'}.md",
            mime="text/markdown",
        )
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(result["final_report"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_critic:
        if "NEEDS REVISION" in result["critique"].upper():
            st.info("The Critic flagged areas for improvement. Auto-revision applied." if result.get("revised") else "The Critic flagged areas for improvement.")
        else:
            st.success("The Critic approved the report on the first pass.")
        st.markdown(result["critique"])

    with tab_sources:
        with st.expander("🔍 Search Agent Notes & URLs", expanded=True):
            st.markdown(result["search_notes"])
        with st.expander("🌐 Reader Agent Full-Page Extractions", expanded=True):
            st.markdown(result["reader_notes"])

    with tab_draft:
        st.markdown(result["draft"])