"""
Multi-Agent Task Orchestrator
==============================
A framework where multiple specialized AI agents collaborate on
complex tasks. A planner decomposes the work, specialists execute
their parts, and a synthesizer merges everything into a final answer.

Usage:
    python main.py
    # Then open http://localhost:8005/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import AgentPool, Orchestrator, OrchestratorConfig
from api import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

config = OrchestratorConfig()
pool = AgentPool(config)
orchestrator = Orchestrator(pool)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pool.load()
    yield


app = FastAPI(
    title="Multi-Agent Task Orchestrator",
    description="TinyLlama-powered multi-agent task decomposition and synthesis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, pool, orchestrator)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)
