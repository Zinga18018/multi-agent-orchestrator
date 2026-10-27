# Multi-Agent Orchestrator

One TinyLlama model is reused with planner, coder, researcher, analyst and synthesizer prompts. The orchestrator generates a plan, calls the selected specialist roles sequentially, then generates a synthesis.

```text
task → planner → selected specialists, one at a time → synthesizer
```

These are prompted roles, not independently equipped agents: there is no web research, tool use, code execution or external evidence verification. The plan is generated text, not a parsed task-routing specification.

## Run the API locally

Use Python 3.10+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8005
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` for activation. Open `http://127.0.0.1:8005/docs`.

Startup loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, downloading weights if absent from the cache. CUDA is selected when available, otherwise CPU. Dependencies are unpinned; no supported-environment test matrix is committed.

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Model status and available roles |
| GET | `/agents` | Role descriptions |
| POST | `/orchestrate` | Planner, sequential specialists, synthesis |
| POST | `/agent/single` | One prompted role |

The [explicit loopback host](https://uvicorn.dev/settings/) is for local use. `python main.py` instead binds all interfaces with reload enabled. There is no API authentication and CORS permits all origins.

## Streamlit status

`streamlit run app.py` loads a separate model-backed interface, but its Run action currently has a code defect: [app.py](app.py) passes `agents=` to `Orchestrator.run`, whose parameter is `agent_names`. This raises a keyword-argument error. The API passes the arguments differently and does not have that particular mismatch. This documentation change does not fix the UI.

The interface also waits for the full pipeline result before displaying phases; it is not live phase-by-phase streaming.

## Implementation and limits

[core/agents.py](core/agents.py) handles model loading and role prompts; [core/orchestrator.py](core/orchestrator.py) performs the sequential loop. Inputs are truncated to 2,048 tokens, so long plans and specialist outputs can lose context. Outputs are sampled and may contradict one another or contain unsupported claims. No reasoning-quality evaluation, speed benchmark or test suite is committed.

Setup, model execution and browser flows were not rerun for this documentation update.

## Related implementations

Related versions live in [ML-Portfolio-Projects](https://github.com/Zinga18018/ML-Portfolio-Projects): the [API variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/project5-multi-agent) and a separate [HF Space variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/hf-spaces/multi-agent). They are different implementations; their interfaces or deployment claims do not establish this repository's behavior.

