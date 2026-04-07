HEAD
# Deep-Research-Ai-agent
A powerful AI agent for deep research that integrates Hugging Face models for NLP inference, leverages LangGraph for structured AI workflows, and features an interactive Streamlit interface for real-time streaming outputs.

# 🔬 Deep Research Agent

An autonomous AI research agent built with **LangGraph**, **HuggingFace**, and **Streamlit**  
that plans, searches, analyses, and writes research reports — all on its own.

---

## Architecture

```
deep_research_agent/
├── agent/
│   ├── config.py      ← model names, token limits, env vars
│   ├── state.py       ← AgentState TypedDict (shared across all nodes)
│   ├── tools.py       ← web_search (DuckDuckGo) + summarize + llm_call
│   ├── nodes.py       ← 4 LangGraph nodes: planner, researcher, analyzer, synthesizer
│   ├── graph.py       ← LangGraph StateGraph + routing logic
│   └── __init__.py
├── frontend/
│   └── app.py         ← Streamlit UI with live streaming dashboard
├── .env.example
├── requirements.txt
└── README.md
```

## LangGraph Workflow

```
[START]
   │
planner_node          Mistral-7B creates 3–5 focused sub-tasks
   │
researcher_node ◄─────┐   DuckDuckGo search → BART summary
   │  (loop)           │
   └───────────────────┘   repeats for every sub-task
   │
analyzer_node         Mistral-7B cross-analyses all summaries
   │
synthesizer_node      Mistral-7B generates final Markdown report
   │
[END]
```

## Models Used

| Role | Model | Provider |
|------|-------|----------|
| Planning, Analysis, Synthesis | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace Inference API |
| Summarisation | `facebook/bart-large-cnn` | HuggingFace Inference API |
| Web search | DuckDuckGo | No API key needed |

## Quick Start

### 1. Clone / copy the project

```bash
git clone <repo>
cd deep_research_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your HuggingFace token

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

You need a **free** HuggingFace account:  
→ https://huggingface.co/settings/tokens (create a "Read" token)

Also accept the Mistral-7B terms of use:  
→ https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

### 4. Run the Streamlit app

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### 5. (Optional) Use the agent from Python directly

```python
from agent import get_graph
from langchain_core.messages import HumanMessage

graph = get_graph()

result = graph.invoke({
    "query": "What are the latest breakthroughs in fusion energy?",
    "plan": [],
    "current_task_index": 0,
    "research_results": [],
    "analysis": "",
    "final_report": "",
    "status": "planning",
    "iterations": 0,
    "actions_taken": [],
    "error_message": None,
    "messages": [HumanMessage(content="What are the latest breakthroughs in fusion energy?")],
})

print(result["final_report"])
```

## How each node works

### ① `planner_node`
Sends the user's query to Mistral-7B with a strict system prompt asking it to
return a **JSON array** of 3–5 search sub-tasks. Falls back gracefully if the
model produces non-JSON output.

### ② `researcher_node`  (loops)
Picks the next sub-task from the plan, runs a **DuckDuckGo** search, collects
text snippets, then sends them to **facebook/bart-large-cnn** via the HF
Inference API for summarisation. Loops back until all sub-tasks are complete.

### ③ `analyzer_node`
Concatenates all sub-task summaries and asks Mistral-7B to identify:
key insights, patterns/trends, conflicting evidence, and recommended focus areas.

### ④ `synthesizer_node`
Uses the analysis + all summaries to generate a full Markdown report with
sections: Executive Summary, Background, Key Findings, Detailed Analysis,
Conclusions, Recommendations, and Sources.

## Environment variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace API token | Yes |

## Customisation

- **Change the LLM**: edit `PLANNER_MODEL` in `agent/config.py`
- **Change the summariser**: edit `SUMMARIZER_MODEL` in `agent/config.py`
- **Adjust plan depth**: edit `MAX_PLAN_STEPS` in `agent/config.py`
- **Add new tools**: extend `agent/tools.py` and wire them into the relevant node
- **Add new nodes**: define a function in `agent/nodes.py` and add it to the graph in `agent/graph.py`
 20e5976 (First updation)
