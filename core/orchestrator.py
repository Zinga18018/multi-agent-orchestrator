import time
import logging

from .agents import AgentPool, Role, AgentOutput

logger = logging.getLogger(__name__)


class Orchestrator:
    """runs the full multi-agent pipeline: plan -> specialist execution -> synthesis.

    the orchestrator coordinates three phases:
      1. planner breaks down the task into subtasks
      2. specialist agents (coder, researcher, analyst) each handle their part
      3. synthesizer merges all outputs into a coherent final answer
    """

    def __init__(self, pool: AgentPool):
        self.pool = pool

    def run(self, task: str, agent_names: list[str],
            max_tokens: int = 300) -> dict:
        """execute the full pipeline and return structured results."""
        total_start = time.perf_counter()
        outputs: list[AgentOutput] = []

        # phase 1 -- planning
        logger.info("[planner] decomposing: %s", task[:80])
        plan = self.pool.run(Role.PLANNER, task, max_tokens)
        outputs.append(plan)

        # phase 2 -- specialist execution
        specialist_texts = []
        for name in agent_names:
            if name in ("planner", "synthesizer"):
                continue
            try:
                role = Role(name)
            except ValueError:
                continue

            context = (
                f"Original task: {task}\n\n"
                f"Plan from planner:\n{plan.output}\n\n"
                f"Your specific assignment: Complete the {name} portion of this plan."
            )
            logger.info("[%s] executing...", name)
            result = self.pool.run(role, context, max_tokens)
            outputs.append(result)
            specialist_texts.append(f"[{name.upper()} AGENT]:\n{result.output}")

        # phase 3 -- synthesis
        synth_context = (
            f"Original task: {task}\n\n"
            f"Plan:\n{plan.output}\n\n"
            f"Agent Results:\n{''.join(specialist_texts)}\n\n"
            f"Synthesize all the above into a final, coherent response."
        )
        logger.info("[synthesizer] merging outputs...")
        synthesis = self.pool.run(Role.SYNTHESIZER, synth_context, max_tokens)
        outputs.append(synthesis)

        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "task": task,
            "plan": plan.output,
            "agent_outputs": [
                {"agent": o.role, "output": o.output, "inference_ms": o.inference_ms}
                for o in outputs
            ],
            "final_synthesis": synthesis.output,
            "total_time_ms": round(total_ms, 1),
            "agents_used": len(outputs),
            "model": self.pool.config.model_id,
        }
