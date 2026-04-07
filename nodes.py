"""
LangGraph nodes – each function is one node in the research graph.

Flow:
  planner_node
      ↓
  researcher_node  ←──┐
      ↓  (loop)       │
  [all tasks done?] ──┘
      ↓
  analyzer_node
      ↓
  synthesizer_node
      ↓
  END
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from config import MAX_SEARCH_RESULTS
from state import AgentState, ResearchResult
from tools import llm_call, summarize_text, web_search

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_action(state: AgentState, msg: str) -> list[str]:
    actions = list(state.get("actions_taken") or [])
    actions.append(msg)
    return actions


def _extract_json_array(text: str) -> list[str]:
    """Pull the first JSON array from *text*; fall back to splitting on newlines."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item).strip() for item in result if str(item).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: numbered-list lines
    lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip()
             for line in text.splitlines() if line.strip()]
    return [l for l in lines if l]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Planner
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are an expert research planner.
Given a user's research query, produce a list of 3-5 focused, specific web-search
sub-tasks that together will fully answer the query.

Rules:
- Return ONLY a valid JSON array of strings, nothing else.
- Each string must be a self-contained search query (≤ 12 words).
- Do NOT add any commentary, markdown fences, or numbering outside the array.

Example output:
["What is quantum entanglement?", "Real-world applications of quantum entanglement 2024", "Quantum entanglement vs classical correlation", "Recent breakthroughs in quantum communication"]
"""


def planner_node(state: AgentState) -> Dict[str, Any]:
    """Convert the user query into an ordered research plan."""
    query = state["query"]
    logger.info("[PLANNER] query=%r", query)

    raw = llm_call(
        user_prompt=f"Create a research plan for: {query}",
        system_prompt=PLANNER_SYSTEM,
    )

    plan = _extract_json_array(raw)
    if not plan:
        plan = [query]  # last-resort fallback

    actions = _append_action(state, f"📋 Plan created — {len(plan)} sub-tasks")

    return {
        "plan": plan,
        "current_task_index": 0,
        "research_results": [],
        "status": "researching",
        "iterations": 0,
        "actions_taken": actions,
        "error_message": None,
        "messages": [AIMessage(content=f"Research plan:\n" + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(plan)))],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Researcher  (loops until all sub-tasks done)
# ─────────────────────────────────────────────────────────────────────────────

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Execute one research sub-task: search → collect → summarise."""
    plan  = state["plan"]
    idx   = state.get("current_task_index", 0)
    iters = state.get("iterations", 0)

    if idx >= len(plan):
        # All tasks done – signal transition to analyzer
        return {"status": "analyzing"}

    task = plan[idx]
    logger.info("[RESEARCHER] task[%d]=%r", idx, task)

    # ── Web search ─────────────────────────────────────────────────────────
    raw_results = web_search(task, max_results=MAX_SEARCH_RESULTS)

    snippets: list[str] = []
    sources:  list[str] = []
    for r in raw_results:
        body  = r.get("body",  "") or ""
        title = r.get("title", "") or ""
        href  = r.get("href",  "") or ""
        if body:
            snippets.append(body)
        if title or href:
            sources.append(f"{title} — {href}" if title else href)

    # ── Summarise with HF BART ─────────────────────────────────────────────
    combined = " ".join(snippets)
    summary  = summarize_text(combined)

    result: ResearchResult = {
        "task":         task,
        "raw_snippets": snippets,
        "sources":      sources,
        "summary":      summary,
    }

    research_results = list(state.get("research_results") or [])
    research_results.append(result)

    actions = _append_action(state, f"🔍 Researched ({idx+1}/{len(plan)}): {task}")

    next_idx = idx + 1
    new_status = "researching" if next_idx < len(plan) else "analyzing"

    return {
        "current_task_index": next_idx,
        "research_results":   research_results,
        "status":             new_status,
        "iterations":         iters + 1,
        "actions_taken":      actions,
        "messages": [AIMessage(content=f"✅ Completed sub-task {idx+1}/{len(plan)}: '{task}'\nSummary: {summary[:200]}…")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – Analyzer
# ─────────────────────────────────────────────────────────────────────────────

ANALYZER_SYSTEM = """You are a senior research analyst.
You receive summaries from multiple web-search sub-tasks and must produce a
structured analysis. Use the following sections:

## Key Insights
(bullet points of the most important facts discovered)

## Patterns & Trends
(recurring themes, emerging trends)

## Conflicting Information
(note any contradictions or gaps in the evidence)

## Recommended Focus Areas
(what aspects deserve deeper attention in the final report)

Be specific, cite sub-task topics when relevant, and stay concise.
"""


def analyzer_node(state: AgentState) -> Dict[str, Any]:
    """Cross-analyse all research summaries with the LLM."""
    query   = state["query"]
    results = state.get("research_results") or []
    logger.info("[ANALYZER] analysing %d sub-tasks", len(results))

    compiled = "\n\n".join(
        f"Sub-task: {r['task']}\nSummary: {r['summary']}"
        for r in results
    )

    analysis = llm_call(
        system_prompt=ANALYZER_SYSTEM,
        user_prompt=(
            f"Original query: {query}\n\n"
            f"Research findings:\n{compiled}\n\n"
            "Produce your structured analysis now."
        ),
    )

    actions = _append_action(state, "🧠 Cross-analysis completed")

    return {
        "analysis":      analysis,
        "status":        "synthesizing",
        "actions_taken": actions,
        "messages":      [AIMessage(content="Analysis complete — proceeding to report synthesis.")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM = """You are an expert report writer.
Produce a comprehensive, well-structured research report in Markdown.

Required sections (use ## headings):
1. Executive Summary
2. Background
3. Key Findings  (use sub-sections or bullet points)
4. Detailed Analysis
5. Conclusions
6. Recommendations

Guidelines:
- Professional, neutral tone.
- Back every claim with evidence from the research findings.
- Keep the report self-contained and readable without the raw data.
- Do NOT invent facts; stick to what the research provided.
"""


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final polished research report."""
    query    = state["query"]
    analysis = state.get("analysis", "")
    results  = state.get("research_results") or []
    logger.info("[SYNTHESIZER] generating final report")

    # Build a source list
    all_sources: list[str] = []
    for r in results:
        for src in r.get("sources", [])[:2]:
            if src and src not in all_sources:
                all_sources.append(src)

    summaries_block = "\n\n".join(
        f"**{r['task']}**\n{r['summary']}" for r in results
    )

    report_body = llm_call(
        system_prompt=SYNTHESIZER_SYSTEM,
        user_prompt=(
            f"Research query: \"{query}\"\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Research summaries:\n{summaries_block}\n\n"
            "Write the full research report now."
        ),
        max_new_tokens=1500,
    )

    # Append source list at the bottom
    if all_sources:
        sources_md = "\n".join(f"- {s}" for s in all_sources[:12])
        report_body += f"\n\n---\n## Sources\n{sources_md}"

    actions = _append_action(state, "📄 Final report ready")

    return {
        "final_report":  report_body,
        "status":        "done",
        "actions_taken": actions,
        "messages":      [AIMessage(content="✅ Research complete — final report generated.")],
    }
