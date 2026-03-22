from fastapi import HTTPException

from core.agents import Role
from core.config import ROLE_PROMPTS
from .schemas import TaskRequest, SingleAgentRequest


def register_routes(app, pool, orchestrator):
    """wire up multi-agent endpoints."""

    @app.get("/health")
    async def health():
        return pool.health()

    @app.get("/agents")
    async def list_agents():
        return {
            "agents": [
                {"role": role, "description": prompt.split(".")[0].strip()}
                for role, prompt in ROLE_PROMPTS.items()
            ]
        }

    @app.post("/orchestrate")
    async def orchestrate(req: TaskRequest):
        if not pool.is_loaded:
            raise HTTPException(503, "model not loaded")
        return orchestrator.run(req.task, req.agents, req.max_tokens)

    @app.post("/agent/single")
    async def run_single(req: SingleAgentRequest):
        if not pool.is_loaded:
            raise HTTPException(503, "model not loaded")

        try:
            role = Role(req.agent)
        except ValueError:
            raise HTTPException(400, f"unknown agent: {req.agent}")

        result = pool.run(role, req.task, req.max_tokens)
        return {
            "agent": result.role,
            "task": req.task[:200],
            "output": result.output,
            "inference_ms": result.inference_ms,
            "model": pool.config.model_id,
        }
