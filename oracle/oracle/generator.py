"""
ORACLE — CoT Generator
Generates multiple chain-of-thought reasoning paths using the Anthropic API.

Beam search simulation:
  Since the Anthropic API is a black-box sampler (no partial-sequence scoring),
  we approximate beam search by:
    1. Generating k independent samples at temperature T  (diverse)
    2. Assigning a proxy logprob via token-count normalisation
    3. Letting the Verifier re-rank them (true beam score)

For mid-chain verification (enabled for MEDIUM/HARD), each path is split
at a midpoint marker and scored before expansion — paths below threshold
are dropped early to save tokens.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from oracle.strategy import GenerationConfig
from oracle.types import ReasoningPath

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

_SYS_CONCISE = """\
You are a precise reasoning assistant. Answer the question directly and concisely.
Format:
Reasoning: <1–3 sentence reasoning>
Answer: <final answer>"""

_SYS_DETAILED = """\
You are a careful reasoning assistant. Think through the problem thoroughly before answering.
Format:
Reasoning: <detailed step-by-step thinking>
Answer: <final answer>"""

_SYS_STEP_BY_STEP = """\
You are a meticulous reasoning assistant. Break every problem into numbered steps.
Format:
Step 1: <first reasoning step>
Step 2: <second reasoning step>
...
Answer: <final answer>

Never skip steps. Show all intermediate calculations."""

_SYSTEM_PROMPTS = {
    "concise": _SYS_CONCISE,
    "detailed": _SYS_DETAILED,
    "step_by_step": _SYS_STEP_BY_STEP,
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> tuple[str, str]:
    """Extract (chain_of_thought, final_answer) from a model response."""
    lines = text.strip().split("\n")
    answer_lines = []
    cot_lines = []
    in_answer = False

    for line in lines:
        if line.lower().startswith("answer:"):
            in_answer = True
            answer_lines.append(line[7:].strip())
        elif in_answer:
            answer_lines.append(line)
        else:
            cot_lines.append(line)

    cot = "\n".join(cot_lines).strip()
    answer = " ".join(answer_lines).strip()

    # Fallback: last non-empty line
    if not answer:
        non_empty = [l for l in lines if l.strip()]
        answer = non_empty[-1] if non_empty else text[:200]

    return cot, answer


# ---------------------------------------------------------------------------
# CoT Generator
# ---------------------------------------------------------------------------

class CoTGenerator:
    """
    Generates k reasoning paths for a question using the Anthropic API.
    Each path is an independent sample — temperature controls diversity.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic()
        self.model = model

    def generate(
        self,
        question: str,
        config: GenerationConfig,
        context: str | None = None,
    ) -> list[ReasoningPath]:
        """
        Synchronous generation of config.beam_width paths.
        Returns paths sorted by composite_score descending (pre-verifier).
        """
        system = _SYSTEM_PROMPTS.get(config.system_prompt_variant, _SYS_DETAILED)
        user_msg = question
        if context:
            user_msg = f"Context: {context}\n\nQuestion: {question}"

        paths: list[ReasoningPath] = []

        for i in range(config.beam_width):
            # Vary temperature slightly across samples for diversity
            temp = min(1.0, config.temperature + (i * 0.03))
            path = self._sample_one(system, user_msg, temp, config)
            paths.append(path)

        # Sort by proxy score (will be re-sorted after verifier)
        paths.sort(key=lambda p: p.token_logprob, reverse=True)
        return paths

    def _sample_one(
        self,
        system: str,
        user_msg: str,
        temperature: float,
        config: GenerationConfig,
    ) -> ReasoningPath:
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=config.max_cot_tokens + config.max_answer_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            raw_text = resp.content[0].text if resp.content else ""
            cot, answer = _parse_response(raw_text)
            tokens = resp.usage.input_tokens + resp.usage.output_tokens

            # Proxy logprob: penalise very long responses (verbosity penalty)
            import math
            out_tokens = resp.usage.output_tokens
            logprob_proxy = -math.log(1 + out_tokens / (config.max_cot_tokens + 10))

            return ReasoningPath(
                chain_of_thought=cot,
                final_answer=answer,
                token_logprob=logprob_proxy,
                generation_time_ms=elapsed,
                tokens_used=tokens,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ReasoningPath(
                chain_of_thought=f"[Generation error: {exc}]",
                final_answer="[error]",
                token_logprob=-99.0,
                generation_time_ms=elapsed,
                tokens_used=0,
            )

    async def generate_async(
        self,
        question: str,
        config: GenerationConfig,
        context: str | None = None,
    ) -> list[ReasoningPath]:
        """
        Async wrapper: all beam_width samples are fired concurrently.
        """
        loop = asyncio.get_event_loop()
        system = _SYSTEM_PROMPTS.get(config.system_prompt_variant, _SYS_DETAILED)
        user_msg = question
        if context:
            user_msg = f"Context: {context}\n\nQuestion: {question}"

        tasks = [
            loop.run_in_executor(
                None,
                self._sample_one,
                system,
                user_msg,
                min(1.0, config.temperature + i * 0.03),
                config,
            )
            for i in range(config.beam_width)
        ]
        paths = await asyncio.gather(*tasks)
        return sorted(paths, key=lambda p: p.token_logprob, reverse=True)
