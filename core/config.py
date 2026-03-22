from dataclasses import dataclass


@dataclass
class OrchestratorConfig:
    """settings for the multi-agent system."""

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    default_max_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_input_tokens: int = 2048
    use_fp16: bool = True
    port: int = 8005


# ---- agent role prompts ----

ROLE_PROMPTS = {
    "planner": (
        "You are a Planning Agent. Your job is to:\n"
        "1. Break down the user's task into 2-4 clear subtasks\n"
        "2. Assign each subtask to the appropriate specialist (coder, researcher, or analyst)\n"
        "3. Define the order of execution\n"
        "Output a numbered plan with agent assignments."
    ),
    "coder": (
        "You are a Code Agent. You write clean, efficient code.\n"
        "Given a coding subtask, produce the solution with comments.\n"
        "Focus on correctness and readability."
    ),
    "researcher": (
        "You are a Research Agent. You analyze topics in depth.\n"
        "Given a research question, provide a structured analysis with key findings,\n"
        "evidence, and conclusions."
    ),
    "analyst": (
        "You are an Analysis Agent. You examine data and situations.\n"
        "Given an analysis task, provide quantitative insights, comparisons,\n"
        "and actionable recommendations."
    ),
    "synthesizer": (
        "You are a Synthesis Agent. You combine outputs from multiple agents\n"
        "into a coherent final response. Merge findings, resolve conflicts,\n"
        "and produce a unified, well-structured answer."
    ),
}
