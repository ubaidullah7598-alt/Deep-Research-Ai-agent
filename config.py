"""
Configuration for the Deep Research Agent.
Set HF_TOKEN in your .env file or as an environment variable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace ────────────────────────────────────────────────────────────────
hf_TOKEN = os.getenv("HF_TOKEN", "Enter your HuggingFace token here")

# LLM used for planning, analysis, and synthesis (via HF Inference API)
PLANNER_MODEL      = "meta-llama/Llama-3.3-70B-Instruct"

# HuggingFace model used for summarising raw search snippets
SUMMARIZER_MODEL   = "facebook/bart-large-cnn"

# ── Agent behaviour ────────────────────────────────────────────────────────────
MAX_PLAN_STEPS       = 5    # max sub-tasks the planner can create
MAX_SEARCH_RESULTS   = 5    # results per DuckDuckGo query
MAX_RESEARCH_LOOPS   = 10   # safety ceiling for the research loop
LLM_MAX_NEW_TOKENS   = 1024
LLM_TEMPERATURE      = 0.3
