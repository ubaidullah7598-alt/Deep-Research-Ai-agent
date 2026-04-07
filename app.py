"""
Deep Research AI Agent — Streamlit Frontend
Run: streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import HumanMessage

# Allow importing sibling 'agent' package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph import get_graph
from state import AgentState

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b2e;
    border-right: 1px solid #2d3561;
}

/* ── Cards ── */
.card {
    background: #1a1f35;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-planning    { background:#1e3a5f; color:#60b4ff; }
.badge-researching { background:#1e4d3a; color:#4ade80; }
.badge-analyzing   { background:#3d2e5a; color:#c084fc; }
.badge-synthesizing{ background:#4d3820; color:#fb923c; }
.badge-done        { background:#14532d; color:#86efac; }
.badge-error       { background:#4c1d1d; color:#fca5a5; }
.badge-idle        { background:#1f2937; color:#9ca3af; }

/* ── Action log item ── */
.action-item {
    padding: 0.35rem 0;
    border-bottom: 1px solid #22283d;
    font-size: 0.88rem;
    color: #c8cfe0;
}

/* ── Research card title ── */
.sub-task-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #818cf8;
    margin-bottom: 0.3rem;
}

/* ── Final report ── */
.report-container {
    background: #12172a;
    border: 1px solid #3730a3;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    line-height: 1.75;
}

/* ── Progress label ── */
.progress-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 0.2rem;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60b4ff 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #6b7280;
    font-size: 1.05rem;
}

/* ── Metric box ── */
.metric-box {
    background: #1a1f35;
    border: 1px solid #2d3561;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60b4ff;
}
.metric-box .lbl {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "running":       False,
        "final_state":   None,
        "stream_steps":  [],      # list of (node_name, partial_state) tuples
        "start_time":    None,
        "elapsed":       0.0,
        "hf_token":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    hf_token = st.text_input(
        "HuggingFace API Token",
        value=st.session_state["hf_token"],
        type="password",
        help="Required for Mistral-7B planner & BART summariser. "
             "Get yours at https://huggingface.co/settings/tokens",
    )
    if hf_token:
        st.session_state["hf_token"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.markdown("## 🤖 Models Used")

    st.markdown("""
| Role | Model |
|------|-------|
| 🧠 Planner / Analyst | `mistralai/Mistral-7B-Instruct-v0.3` |
| 📝 Summariser | `facebook/bart-large-cnn` |
| 🔍 Search | DuckDuckGo (no key) |
    """)

    st.markdown("---")
    st.markdown("## 🔄 Agent Workflow")
    st.markdown("""
```
USER QUERY
    │
 ① PLANNER       ← LLM creates sub-tasks
    │
 ② RESEARCHER ←──┐
    │  (loop)     │
    └─────────────┘
    │
 ③ ANALYZER       ← LLM cross-analyses
    │
 ④ SYNTHESIZER    ← LLM writes report
    │
  REPORT
```
    """)

    st.markdown("---")
    st.caption("Built with LangGraph · HuggingFace · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: status badge HTML
# ─────────────────────────────────────────────────────────────────────────────

STATUS_EMOJI = {
    "planning":    "🗺️",
    "researching": "🔍",
    "analyzing":   "🧠",
    "synthesizing":"✍️",
    "done":        "✅",
    "error":       "❌",
    "idle":        "💤",
}

def status_badge(status: str) -> str:
    emoji = STATUS_EMOJI.get(status, "❔")
    return (
        f'<span class="badge badge-{status}">'
        f'{emoji} {status.upper()}'
        f'</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <h1>🔬 Deep Research Agent</h1>
    <p>Autonomous AI that plans, searches, analyses, and writes research reports</p>
</div>
""", unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Research Query",
        placeholder="e.g. What are the latest breakthroughs in fusion energy?",
        label_visibility="collapsed",
        disabled=st.session_state["running"],
    )
with col_btn:
    run_btn = st.button(
        "🚀 Run",
        use_container_width=True,
        disabled=st.session_state["running"] or not query.strip(),
        type="primary",
    )

# ── Layout: left = live dashboard, right = report ────────────────────────────
col_dash, col_report = st.columns([1, 2], gap="large")


# ─────────────────────────────────────────────────────────────────────────────
# Run the agent (streaming node-by-node)
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and query.strip():
    # Reset state for new run
    st.session_state["running"]      = True
    st.session_state["stream_steps"] = []
    st.session_state["final_state"]  = None
    st.session_state["start_time"]   = time.time()
    st.rerun()

if st.session_state["running"]:
    initial_state: AgentState = {
        "query":              query or "",
        "plan":               [],
        "current_task_index": 0,
        "research_results":   [],
        "analysis":           "",
        "final_report":       "",
        "status":             "planning",
        "iterations":         0,
        "actions_taken":      [],
        "error_message":      None,
        "messages":           [HumanMessage(content=query or "")],
    }

    # ── Dashboard placeholders (render while streaming) ───────────────────
    with col_dash:
        st.markdown("### 📊 Live Dashboard")
        status_placeholder   = st.empty()
        metrics_placeholder  = st.empty()
        plan_placeholder     = st.empty()
        actions_placeholder  = st.empty()

    with col_report:
        st.markdown("### 📄 Report Preview")
        report_placeholder   = st.empty()
        sources_placeholder  = st.empty()

    # ── Stream the graph ──────────────────────────────────────────────────
    graph = get_graph()
    last_state: Dict[str, Any] = dict(initial_state)

    try:
        for step in graph.stream(initial_state, stream_mode="updates"):
            # step = { node_name: partial_state_update }
            for node_name, update in step.items():
                last_state.update(update)
                st.session_state["stream_steps"].append((node_name, dict(last_state)))

                # ── Status badge ─────────────────────────────────────────
                current_status = last_state.get("status", "idle")
                status_placeholder.markdown(
                    f'<div class="card" style="text-align:center">'
                    f'<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.4rem">Current Node: <b style="color:#818cf8">{node_name}</b></div>'
                    f'{status_badge(current_status)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Metrics ───────────────────────────────────────────────
                plan_len  = len(last_state.get("plan") or [])
                done_tasks = last_state.get("current_task_index", 0)
                iters      = last_state.get("iterations", 0)
                elapsed    = time.time() - st.session_state["start_time"]

                metrics_placeholder.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.6rem;margin-bottom:1rem">'
                    f'<div class="metric-box"><div class="val">{plan_len}</div><div class="lbl">Sub-tasks</div></div>'
                    f'<div class="metric-box"><div class="val">{done_tasks}</div><div class="lbl">Completed</div></div>'
                    f'<div class="metric-box"><div class="val">{iters}</div><div class="lbl">Searches</div></div>'
                    f'<div class="metric-box"><div class="val">{elapsed:.1f}s</div><div class="lbl">Elapsed</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Research plan ─────────────────────────────────────────
                plan = last_state.get("plan") or []
                if plan:
                    plan_html = '<div class="card"><div style="font-weight:700;margin-bottom:0.6rem;color:#c0c7e0">🗺️ Research Plan</div>'
                    idx_done  = last_state.get("current_task_index", 0)
                    for i, task in enumerate(plan):
                        if i < idx_done:
                            icon, color = "✅", "#4ade80"
                        elif i == idx_done:
                            icon, color = "⏳", "#fbbf24"
                        else:
                            icon, color = "○", "#4b5563"
                        plan_html += (
                            f'<div style="padding:0.3rem 0;color:{color};font-size:0.85rem">'
                            f'{icon} {i+1}. {task}</div>'
                        )
                    plan_html += "</div>"
                    plan_placeholder.markdown(plan_html, unsafe_allow_html=True)

                # ── Action log ────────────────────────────────────────────
                actions = last_state.get("actions_taken") or []
                if actions:
                    actions_html = '<div class="card"><div style="font-weight:700;margin-bottom:0.6rem;color:#c0c7e0">⚡ Action Log</div>'
                    for a in reversed(actions[-8:]):
                        actions_html += f'<div class="action-item">{a}</div>'
                    actions_html += "</div>"
                    actions_placeholder.markdown(actions_html, unsafe_allow_html=True)

                # ── Partial report preview ─────────────────────────────────
                report = last_state.get("final_report", "")
                if report:
                    with report_placeholder.container():
                        st.markdown(
                            f'<div class="report-container">{report}</div>',
                            unsafe_allow_html=True,
                        )

                    # Sources
                    sources_list = []
                    for r in (last_state.get("research_results") or []):
                        for s in r.get("sources", [])[:2]:
                            if s:
                                sources_list.append(s)
                    if sources_list:
                        with sources_placeholder.expander("📚 Sources consulted"):
                            for src in sources_list[:15]:
                                st.caption(src)
                else:
                    analysis = last_state.get("analysis", "")
                    if analysis:
                        report_placeholder.markdown(
                            f'<div class="card">'
                            f'<div style="font-weight:700;margin-bottom:0.6rem;color:#c084fc">🧠 Analysis in progress…</div>'
                            f'<div style="font-size:0.88rem;color:#c8cfe0">{analysis[:600]}{"…" if len(analysis)>600 else ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ── Run complete ───────────────────────────────────────────────────
        st.session_state["final_state"] = last_state
        st.session_state["running"]     = False
        st.session_state["elapsed"]     = time.time() - st.session_state["start_time"]
        st.rerun()

    except Exception as exc:
        st.session_state["running"]   = False
        st.session_state["final_state"] = {**last_state, "status": "error", "error_message": str(exc)}
        st.error(f"Agent error: {exc}")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Post-run: display final state
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state["final_state"] is not None:
    fs: Dict[str, Any] = st.session_state["final_state"]

    with col_dash:
        st.markdown("### 📊 Run Summary")

        status = fs.get("status", "done")
        st.markdown(
            f'<div class="card" style="text-align:center">{status_badge(status)}</div>',
            unsafe_allow_html=True,
        )

        plan_len  = len(fs.get("plan") or [])
        done_tasks = fs.get("current_task_index", 0)
        iters      = fs.get("iterations", 0)
        elapsed    = st.session_state.get("elapsed", 0.0)

        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.6rem;margin-bottom:1rem">'
            f'<div class="metric-box"><div class="val">{plan_len}</div><div class="lbl">Sub-tasks</div></div>'
            f'<div class="metric-box"><div class="val">{done_tasks}</div><div class="lbl">Completed</div></div>'
            f'<div class="metric-box"><div class="val">{iters}</div><div class="lbl">Searches</div></div>'
            f'<div class="metric-box"><div class="val">{elapsed:.1f}s</div><div class="lbl">Time</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Plan
        plan = fs.get("plan") or []
        if plan:
            plan_html = '<div class="card"><div style="font-weight:700;margin-bottom:0.6rem;color:#c0c7e0">🗺️ Research Plan</div>'
            for i, task in enumerate(plan):
                plan_html += f'<div style="padding:0.3rem 0;color:#4ade80;font-size:0.85rem">✅ {i+1}. {task}</div>'
            plan_html += "</div>"
            st.markdown(plan_html, unsafe_allow_html=True)

        # Action log
        actions = fs.get("actions_taken") or []
        if actions:
            with st.expander("⚡ Full Action Log", expanded=True):
                for a in actions:
                    st.markdown(f'<div class="action-item">{a}</div>', unsafe_allow_html=True)

        # Error
        if fs.get("error_message"):
            st.error(fs["error_message"])

        # New search button
        if st.button("🔄 New Research", use_container_width=True):
            st.session_state["final_state"] = None
            st.rerun()

    with col_report:
        st.markdown("### 📄 Research Report")

        report = fs.get("final_report", "")
        if report:
            st.markdown(
                f'<div class="report-container">{report}</div>',
                unsafe_allow_html=True,
            )

            # Download buttons
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=report,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_col2:
                html_report = f"<html><body style='font-family:sans-serif;max-width:860px;margin:auto;padding:2rem'>{report}</body></html>"
                st.download_button(
                    "⬇️ Download HTML",
                    data=html_report,
                    file_name="research_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

            # Research details
            results = fs.get("research_results") or []
            if results:
                with st.expander("🔬 Research Details (per sub-task)", expanded=False):
                    for i, r in enumerate(results):
                        st.markdown(f'<div class="sub-task-title">#{i+1} {r["task"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'**Summary:** {r["summary"]}')
                        if r.get("sources"):
                            st.caption("Sources: " + " · ".join(r["sources"][:3]))
                        st.divider()

        # Analysis
        analysis = fs.get("analysis", "")
        if analysis and not report:
            st.markdown("#### 🧠 Analysis")
            st.markdown(analysis)

else:
    # ── Idle / welcome state ─────────────────────────────────────────────────
    with col_dash:
        st.markdown("""
<div class="card" style="text-align:center;padding:2rem">
    <div style="font-size:3rem">🤖</div>
    <div style="font-size:1.1rem;font-weight:600;margin:0.5rem 0;color:#c0c7e0">Agent Ready</div>
    <div style="color:#6b7280;font-size:0.9rem">Enter a query and press Run to begin</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="card">
    <div style="font-weight:700;margin-bottom:0.8rem;color:#c0c7e0">💡 Example Queries</div>
    <div style="color:#818cf8;font-size:0.88rem;line-height:2">
    • What are the latest breakthroughs in fusion energy?<br>
    • How is AI transforming drug discovery in 2024?<br>
    • What is the current state of quantum computing?<br>
    • Explain the geopolitical impact of semiconductor supply chains<br>
    • What are the best practices in LLM fine-tuning?
    </div>
</div>
        """, unsafe_allow_html=True)

    with col_report:
        st.markdown("""
<div class="card" style="padding:2rem">
    <div style="font-size:1.1rem;font-weight:700;color:#c0c7e0;margin-bottom:1rem">How It Works</div>
    <div style="color:#9ca3af;line-height:1.9;font-size:0.9rem">
    <b style="color:#60b4ff">① Planner</b> — Mistral-7B breaks your query into 3–5 focused sub-tasks<br><br>
    <b style="color:#4ade80">② Researcher</b> — DuckDuckGo searches run for each sub-task; BART summarises the snippets<br><br>
    <b style="color:#c084fc">③ Analyzer</b> — Mistral-7B cross-analyses all summaries, identifying patterns and gaps<br><br>
    <b style="color:#fb923c">④ Synthesizer</b> — A polished Markdown report is generated with sections, insights, and sources
    </div>
</div>
        """, unsafe_allow_html=True)
