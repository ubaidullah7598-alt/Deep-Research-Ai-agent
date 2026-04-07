"""
Tools available to the agent:
  • web_search  – DuckDuckGo (no API key required)
  • summarize   – HuggingFace Inference API (facebook/bart-large-cnn)
  • llm_call    – HuggingFace Inference API (Mistral-7B-Instruct)
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any
from ddgs import DDGS
from huggingface_hub import InferenceClient

# Import from config
from config import (
    HF_TOKEN,
    SUMMARIZER_MODEL,
    PLANNER_MODEL,
    MAX_SEARCH_RESULTS,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# ── Shared HF client ──────────────────────────────────────────────────────────
_hf_client: InferenceClient | None = None

def _get_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)
    return _hf_client


# ─────────────────────────────────────────────────────────────────────────────
# 1. Web Search
# ─────────────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return a list of result dicts."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        logger.debug("web_search(%r) → %d results", query, len(results))
        return results
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return [{"title": "Search error", "body": str(exc), "href": ""}]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Summarisation (Fixed - Minimal parameters only)
# ─────────────────────────────────────────────────────────────────────────────
def summarize_text(text: str, max_length: int = 180, min_length: int = 40) -> str:
    """
    Summarise text using facebook/bart-large-cnn.
    Uses ONLY supported parameters for the summarization task on HF Inference API.
    """
    if not text or len(text.strip()) < 80:
        return text.strip()

    clipped = text[:3500]   # BART safe limit

    try:
        client = _get_client()

        # Minimal call - ONLY model + truncation (no max_length, min_length, parameters, etc.)
        result = client.summarization(
            clipped,
            model=SUMMARIZER_MODEL,   # Should be "facebook/bart-large-cnn"
            truncation=True
        )

        # Safe extraction
        if isinstance(result, dict):
            summary = result.get("summary_text", str(result))
        else:
            summary = getattr(result, "summary_text", None) or str(result)

        summary = summary.strip()
        logger.debug("summarize_text → %d chars", len(summary))
        return summary

    except Exception as exc:
        logger.warning("summarize_text fell back to truncation: %s", exc)
        cutoff = max_length + 100
        return clipped[:cutoff].strip() + " …"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM Call (Mistral via chat_completion)
# ─────────────────────────────────────────────────────────────────────────────
def llm_call(
    user_prompt: str,
    system_prompt: str = "",
    model: str = PLANNER_MODEL,
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Send a chat-completion request using Mistral Instruct model."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = _get_client()
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["\n\n"]   # Use 'stop' (not stop_sequences)
        )
        reply: str = response.choices[0].message.content or ""
        logger.debug("llm_call → %d chars", len(reply))
        return reply.strip()

    except Exception as exc:
        logger.error("llm_call failed: %s", exc)
        return f"[LLM error: {exc}]"