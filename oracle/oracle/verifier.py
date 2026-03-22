"""
ORACLE — Verifier Model
Scores each (question, chain_of_thought, answer) triple on a 0–1 scale.

The verifier is itself an LLM call — it acts as a process reward model (PRM).
It evaluates:
  1. Logical coherence of the reasoning chain
  2. Faithfulness of the answer to the reasoning
  3. Factual plausibility (where checkable)
  4. Completeness — does the answer address the question?

The score is extracted from a structured JSON response, making it robust
to formatting variance.
"""
from __future__ import annotations

import json
import re
import time

from oracle.types import ReasoningPath

# ---------------------------------------------------------------------------
# Verifier prompt
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM = """\
You are an expert reasoning verifier. Given a question, a chain of thought, and a proposed answer,
you must evaluate the quality of the reasoning and answer.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
  "logical_coherence": <0.0–1.0>,
  "answer_faithfulness": <0.0–1.0>,
  "factual_plausibility": <0.0–1.0>,
  "completeness": <0.0–1.0>,
  "overall_score": <0.0–1.0>,
  "critical_flaw": "<one-sentence description of the biggest problem, or 'none'>"
}

Scoring guide:
- logical_coherence: Does each reasoning step follow from the previous?
- answer_faithfulness: Does the final answer follow from the reasoning chain?
- factual_plausibility: Are the facts stated likely to be correct?
- completeness: Does the answer fully address what was asked?
- overall_score: Holistic score; NOT a simple average — weight logical coherence most.
"""

_VERIFIER_TEMPLATE = """\
QUESTION: {question}

CHAIN OF THOUGHT:
{chain_of_thought}

PROPOSED ANSWER: {answer}

Evaluate the above reasoning and answer."""


class Verifier:
    """
    Scores ReasoningPath objects using a separate LLM verification call.
    Supports batched scoring to amortise latency.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic()
        self.model = model

    def score(self, question: str, path: ReasoningPath) -> ReasoningPath:
        """Score a single path in-place and return it."""
        score, _ = self._call_verifier(question, path)
        path.verifier_score = score
        return path

    def score_batch(
        self, question: str, paths: list[ReasoningPath]
    ) -> list[ReasoningPath]:
        """Score all paths. Returns them sorted by verifier_score desc."""
        for path in paths:
            self.score(question, path)
        return sorted(paths, key=lambda p: p.verifier_score, reverse=True)

    def _call_verifier(
        self, question: str, path: ReasoningPath
    ) -> tuple[float, dict]:
        user_msg = _VERIFIER_TEMPLATE.format(
            question=question,
            chain_of_thought=path.chain_of_thought or "(no chain of thought provided)",
            answer=path.final_answer or "(no answer)",
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.0,   # deterministic scoring
                system=_VERIFIER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip() if resp.content else "{}"

            # Strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

            data = json.loads(raw)
            overall = float(data.get("overall_score", 0.5))
            overall = max(0.0, min(1.0, overall))
            return overall, data

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract a bare float from the response
            try:
                match = re.search(r'"overall_score"\s*:\s*([0-9.]+)', raw)
                if match:
                    return float(match.group(1)), {}
            except Exception:
                pass
            return 0.5, {}
        except Exception:
            return 0.5, {}

    def quick_score(self, question: str, answer: str) -> float:
        """
        Lightweight scoring without a full CoT — used for greedy path validation.
        """
        dummy = ReasoningPath(chain_of_thought="Direct answer.", final_answer=answer)
        score, _ = self._call_verifier(question, dummy)
        return score
