# Multi-Agent Collaborative Reasoning with Chain-of-Thought

A framework that decomposes complex tasks across specialized LLM agents. A planner breaks the problem down, domain specialists (coder, researcher, analyst) work on their parts, and a synthesizer merges everything into a coherent answer.

## how it works

```
task → planner agent → subtask decomposition
                            ↓
              coder / researcher / analyst (parallel)
                            ↓
                    synthesizer agent → final answer
```

all agents share a single TinyLlama-1.1B model but use different system prompts to specialize their behavior. the orchestrator manages the three-phase pipeline: plan, execute, synthesize.

**agents:**
- **planner** — breaks the task into 2-4 focused subtasks
- **coder** — writes clean, commented code solutions
- **researcher** — provides structured analysis with evidence
- **analyst** — gives quantitative insights and recommendations
- **synthesizer** — merges all outputs into a unified response

## setup

```bash
pip install -r requirements.txt
python main.py
```

runs at `localhost:8005`. swagger docs at `/docs`.

## api

| endpoint | method | what it does |
|----------|--------|-------------|
| `/health` | GET | model status + available agents |
| `/agents` | GET | list agents with descriptions |
| `/orchestrate` | POST | run full multi-agent pipeline |
| `/agent/single` | POST | run a single agent directly |

## architecture

```
core/
├── config.py        → model settings, role prompts
├── agents.py        → Role enum, AgentPool, shared model
└── orchestrator.py  → 3-phase pipeline coordination

api/
├── schemas.py       → pydantic request/response models
└── routes.py        → endpoint handlers

main.py              → FastAPI entry point, lifespan
app.py               → streamlit frontend
```

## streamlit demo

```bash
streamlit run app.py
```

enter a task, pick which agents to use, watch the pipeline run phase by phase.

## requirements

- python 3.10+
- ~3GB for model weights
- GPU recommended but not required
