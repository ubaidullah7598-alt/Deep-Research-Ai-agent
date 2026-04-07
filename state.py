"""
LangGraph state for the Deep Research Agent.

Every node receives the full state dict and returns a (partial) update.
LangGraph merges updates automatically.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ResearchResult(TypedDict):
    """One completed research sub-task."""
    task: str                  # the sub-task / search query
    raw_snippets: List[str]    # raw text snippets from web search
    sources: List[str]         # URLs / titles
    summary: str               # HF BART summarised version


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str                          # original user question

    # ── Planning ───────────────────────────────────────────────────────────────
    plan: List[str]                     # ordered list of search sub-tasks

    # ── Research loop ──────────────────────────────────────────────────────────
    current_task_index: int             # which plan step we are on
    research_results: List[ResearchResult]

    # ── Analysis & Synthesis ───────────────────────────────────────────────────
    analysis: str                       # structured analysis from LLM
    final_report: str                   # full markdown report

    # ── Agent bookkeeping ──────────────────────────────────────────────────────
    status: str                         # planning | researching | analyzing | synthesizing | done | error
    iterations: int                     # total researcher-node invocations
    actions_taken: List[str]            # human-readable log of every action
    error_message: Optional[str]        # last error if any

    # ── Message history (LangGraph managed) ───────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
