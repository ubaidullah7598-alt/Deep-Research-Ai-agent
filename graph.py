"""
LangGraph workflow for the Deep Research Agent.

Graph topology:

    [START]
       │
   planner_node          ← creates the research plan
       │
   researcher_node ◄─────┐
       │                  │ loop until all sub-tasks done
       ▼                  │
  (router) ───"more"──────┘
       │
    "done"
       │
   analyzer_node          ← cross-analyses all summaries
       │
   synthesizer_node       ← produces the final markdown report
       │
    [END]
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from nodes import (
    analyzer_node,
    planner_node,
    researcher_node,
    synthesizer_node,
)
from state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────────────────────────────────────

def _research_router(state: AgentState) -> Literal["more_research", "analyze"]:
    """
    After researcher_node runs, decide:
      • continue looping through sub-tasks, or
      • advance to the analyzer.
    """
    status    = state.get("status", "researching")
    idx       = state.get("current_task_index", 0)
    plan_len  = len(state.get("plan") or [])
    iters     = state.get("iterations", 0)

    from .config import MAX_RESEARCH_LOOPS
    if status == "analyzing" or idx >= plan_len or iters >= MAX_RESEARCH_LOOPS:
        logger.debug("_research_router → analyze (idx=%d, plan_len=%d)", idx, plan_len)
        return "analyze"

    logger.debug("_research_router → more_research (idx=%d/%d)", idx, plan_len)
    return "more_research"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the research agent graph."""
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",     planner_node)
    graph.add_node("researcher",  researcher_node)
    graph.add_node("analyzer",    analyzer_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge("planner", "researcher")

    graph.add_conditional_edges(
        "researcher",
        _research_router,
        {
            "more_research": "researcher",   # loop back
            "analyze":       "analyzer",     # move forward
        },
    )

    graph.add_edge("analyzer",    "synthesizer")
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    logger.info("Research agent graph compiled successfully.")
    return compiled


# ── Singleton compiled graph ──────────────────────────────────────────────────
_graph = None


def get_graph():
    """Return a singleton compiled graph (built once per process)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
