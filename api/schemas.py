from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    task: str
    max_tokens: int = Field(default=300, ge=50, le=500)
    agents: list[str] = Field(
        default=["planner", "coder", "researcher", "synthesizer"],
        description="which agents to include in the pipeline",
    )


class SingleAgentRequest(BaseModel):
    agent: str = "researcher"
    task: str
    max_tokens: int = Field(default=300, ge=50, le=500)
