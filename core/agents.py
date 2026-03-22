import time
import logging
from enum import Enum
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import OrchestratorConfig, ROLE_PROMPTS

logger = logging.getLogger(__name__)


class Role(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentOutput:
    role: str
    output: str
    inference_ms: float


class AgentPool:
    """manages a shared TinyLlama model and exposes per-role inference.

    all agents share the same underlying LLM -- what differentiates
    them is the system prompt that steers the generation. this keeps
    memory usage low while still supporting multiple specialist roles.
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """pull model weights and move to the best available device."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading %s on %s", self.config.model_id, self.device)

        dtype = torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("agent pool ready on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def run(self, role: Role, context: str, max_tokens: int = 300) -> AgentOutput:
        """run a single agent role on the provided context."""
        system_prompt = ROLE_PROMPTS.get(role.value, ROLE_PROMPTS["researcher"])
        prompt = (
            f"<|system|>\n{system_prompt}</s>\n"
            f"<|user|>\n{context}</s>\n"
            f"<|assistant|>\n"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        elapsed = (time.perf_counter() - start) * 1000

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return AgentOutput(
            role=role.value,
            output=text,
            inference_ms=round(elapsed, 1),
        )

    def health(self) -> dict:
        return {
            "status": "healthy" if self.is_loaded else "loading",
            "model": self.config.model_id,
            "device": str(self.device),
            "available_agents": [r.value for r in Role],
        }
