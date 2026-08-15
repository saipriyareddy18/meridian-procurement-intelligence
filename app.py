"""
Meridian Supply Chain RAG — submission UI
HCLTech Campus Ambassador × The Economic Times
"""

from __future__ import annotations

import os
from pathlib import Path

# Fix Windows SSL issues BEFORE importing Gemini clients
os.environ.setdefault("ALLOW_INSECURE_SSL", "1")
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

import streamlit as st
from dotenv import load_dotenv, set_key

load_dotenv(override=True)

import config  # noqa: E402  — applies SSL patch
from ingest import DATA_DIR, chunk_count_fast, ingest_data_folder, ingest_pdfs, reset_caches
from rag import ask, clear_answer_cache

st.set_page_config(
    page_title="Meridian Procurement Intelligence | ET × HCL",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENV_PATH = Path(__file__).resolve().parent / ".env"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700;800&family=Newsreader:opsz,wght@6..72,500;6..72,600;6..72,700&display=swap');
:root { --et-red:#d4010f; --ink:#111827; --muted:#5b6472; --line:#e5e7eb; --navy:#0b1324; }
html, body, [class*="css"] { font-family:"Libre Franklin",sans-serif; color:var(--ink); }
.stApp { background: radial-gradient(900px 280px at 100% 0%, rgba(212,1,15,.10), transparent 55%), linear-gradient(180deg,#0b1324 0%,#0b1324 150px,#f3f4f6 150px,#f3f4f6 100%); }
div[data-testid="stToolbar"] { visibility:hidden; height:0; }
.block-container { padding-top:1.1rem!important; padding-bottom:2.5rem!important; max-width:1180px; }
section[data-testid="stSidebar"] { background:#0f172a!important; border-right:1px solid rgba(255,255,255,.06); }
section[data-testid="stSidebar"] * { color:#e5e7eb!important; }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] small { color:#94a3b8!important; }
section[data-testid="stSidebar"] div[data-baseweb="input"] input { background:#1e293b!important; color:#f8fafc!important; border:1px solid #334155!important; }
.et-hero{color:#fff;margin-bottom:1rem}
.et-kicker{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:rgba(255,255,255,.72);margin-bottom:.5rem}
.et-title{font-family:"Newsreader",Georgia,serif;font-size:2.3rem;line-height:1.12;margin:0 0 .4rem;font-weight:700}
.et-title em{font-style:italic;color:#fecaca}
.et-sub{margin:0;max-width:42rem;color:rgba(255,255,255,.78);font-size:1rem;line-height:1.5}
.et-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.2rem;box-shadow:0 10px 30px rgba(15,23,42,.05);margin-bottom:1rem}
.et-card h3{font-family:"Newsreader",Georgia,serif;font-size:1.2rem;margin:0 0 .25rem;color:var(--navy)}
.et-card p{margin:0;color:var(--muted);font-size:.92rem;line-height:1.45}
.answer-box{border-left:3px solid var(--et-red);background:#fff;padding:.95rem 1rem;margin:.4rem 0 .7rem;line-height:1.55}
.source-row{padding:.4rem 0;border-bottom:1px solid var(--line);font-size:.92rem;color:var(--muted)}
.source-row b{color:var(--ink)}
.ok-pill{display:inline-block;margin-top:.5rem;background:#ecfdf5;color:#047857;font-size:.72rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:.28rem .55rem;border-radius:999px}
.stButton>button[kind="primary"]{background:linear-gradient(180deg,#e11d2e,#b9101d)!important;border:1px solid #9f0d18!important;color:#fff!important;font-weight:700!important;border-radius:10px!important}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="et-hero">
  <div class="et-kicker">HCLTech Campus Ambassador · The Economic Times</div>
  <h1 class="et-title">Meridian <em>Procurement Intelligence</em></h1>
  <p class="et-sub">Ask about suppliers, penalties, approvals and Q1 performance. Answers are grounded in Meridian PDFs with page citations.</p>
</div>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Workspace")
    st.caption("1) Save API key · 2) Index · 3) Ask")

    st.markdown("**Gemini API key** (campus alternative to OpenAI)")
    st.caption("[Create free key](https://aistudio.google.com/apikey)")
    gemini_key = st.text_input("GEMINI_API_KEY", type="password", placeholder="AIza...", label_visibility="collapsed")
    if st.button("Save key", use_container_width=True):
        key = (gemini_key or "").strip()
        if not key:
            st.error("Paste a key first.")
        else:
            os.environ["LLM_PROVIDER"] = "gemini"
            os.environ["GEMINI_API_KEY"] = key
            st.session_state["GEMINI_API_KEY"] = key
            try:
                if not ENV_PATH.exists():
                    ENV_PATH.write_text("LLM_PROVIDER=gemini\n", encoding="utf-8")
                set_key(str(ENV_PATH), "LLM_PROVIDER", "gemini")
                set_key(str(ENV_PATH), "GEMINI_API_KEY", key)
                st.success("Key saved")
            except Exception as exc:
                st.warning(f"Saved for this session only ({exc})")

    if st.session_state.get("GEMINI_API_KEY"):
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = st.session_state["GEMINI_API_KEY"]

    st.divider()
    st.markdown("**Documents**")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    chunks = chunk_count_fast()
    st.metric("Indexed chunks", chunks)
    force_rebuild = st.checkbox("Force rebuild index", value=False)

    if st.button("Index Documents", type="primary", use_container_width=True):
        try:
            with st.spinner("Indexing…"):
                reset_caches()
                clear_answer_cache()
                # Always index the two Meridian assignment PDFs from data/
                meridian = sorted(
                    p
                    for p in DATA_DIR.glob("*.pdf")
                    if "Meridian" in p.name or "meridian" in p.name.lower()
                )
                paths = list(meridian)
                if uploaded_files:
                    for f in uploaded_files:
                        # Don't let random uploads replace Meridian assignment docs
                        if "Meridian" in f.name or f.name.lower().endswith(".pdf"):
                            dest = DATA_DIR / f.name
                            # Keep Meridian originals; save extras alongside
                            if "Meridian" not in f.name:
                                dest = DATA_DIR / f.name
                                dest.write_bytes(f.getbuffer())
                                paths.append(dest)
                if not paths:
                    st.warning("Meridian PDFs missing from data/. Re-download the assignment zip.")
                    st.stop()
                result = ingest_pdfs(
                    paths,
                    clear_first=True,
                    skip_if_unchanged=False,
                )
            if result.get("skipped"):
                st.info(f"Already indexed · {result['chunks']} chunks")
            else:
                st.success(f"{result['files']} files · {result['chunks']} chunks")
                st.rerun()
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.caption("If you see SSL certificate errors, restart the app after saving .env (ALLOW_INSECURE_SSL=1).")

    st.caption("Bundled: Review Q1 + Policy Handbook")

left, right = st.columns([1.55, 1], gap="large")

EXAMPLES = [
    "Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
    "How many line stoppages happened in Q1, what was the total downtime, and what caused them?",
    "What is the approval authority for a purchase order worth ₹1.4 crore?",
    "What are the four supplier classification categories, and what qualifies a supplier as Critical?",
    "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?",
    "The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?",
    "Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?",
    "Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?",
    "Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?",
    "What is the annual salary of the Head of Procurement?",
]

with right:
    st.markdown('<div class="et-card"><h3>Assignment test questions</h3><p>Click to load into the question box.</p></div>', unsafe_allow_html=True)
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"Q{i+1}. {ex[:70]}{'…' if len(ex)>70 else ''}", key=f"ex_{i}", use_container_width=True):
            st.session_state["question_input"] = ex
            st.rerun()

with left:
    st.markdown('<div class="et-card"><h3>Ask the knowledge base</h3><p>Answers cite document name and page. Cross-document questions pull from both PDFs.</p></div>', unsafe_allow_html=True)
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    st.text_area("Question", height=120, key="question_input", label_visibility="collapsed")
    top_k = st.slider("Retrieved chunks (top_k)", 4, 8, 6)
    ask_clicked = st.button("Generate answer", type="primary", use_container_width=True)

if ask_clicked:
    q = (st.session_state.get("question_input") or "").strip()
    if not q:
        st.warning("Enter a question.")
    elif chunk_count_fast() == 0:
        st.warning("Index documents first.")
    else:
        with st.spinner("Retrieving from review + policy…"):
            try:
                result = ask(q, top_k=top_k)
                st.session_state.history.insert(0, {"q": q, **result})
            except Exception as exc:
                st.error(f"Error: {exc}")

if st.session_state.history:
    st.markdown("### Answers")
    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Q:** {item['q']}")
            st.markdown("**A:**")
            st.write(item["answer"])
            grouped = item.get("grouped_sources") or {}
            if grouped:
                st.markdown("**Sources**")
                for file_name, entries in grouped.items():
                    pages = ", ".join(str(e["page"]) for e in entries)
                    st.markdown(f'<div class="source-row"><b>{file_name}</b> · pages {pages}</div>', unsafe_allow_html=True)
                if len(grouped) >= 2:
                    st.markdown('<span class="ok-pill">Both documents cited</span>', unsafe_allow_html=True)
            bits = []
            if item.get("cached"):
                bits.append("cached")
            if item.get("latency_ms") is not None:
                bits.append(f"{item['latency_ms']} ms")
            if bits:
                st.caption(" · ".join(bits))

st.markdown(
    """
<div class="et-card">
  <h3>About</h3>
  <p>Built for the <b>HCLTech Campus Ambassador</b> programme with <b>The Economic Times</b>. Uses RAG over Meridian’s performance review and procurement handbook with grounded refusals for missing facts.</p>
</div>
""",
    unsafe_allow_html=True,
)
