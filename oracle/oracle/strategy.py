"""
ORACLE — Strategy Selector
Maps a DifficultyEstimate to generation hyperparameters:
beam width, temperature, CoT depth, and whether to run mid-chain verification.
Also implements the recalibration loop that adjusts thresholds based on
accumulated benchmark data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from oracle.types import Difficulty, QuestionType, Strategy


@dataclass
class GenerationConfig:
    """Complete set of hyperparameters for one generation run."""
    strategy: Strategy
    beam_width: int
    temperature: float
    max_cot_tokens: int
    max_answer_tokens: int
    mid_chain_verify: bool        # run verifier at each beam step
    retry_on_low_score: bool      # second pass if all scores < threshold
    retry_temperature_boost: float
    system_prompt_variant: str    # "concise" | "detailed" | "step_by_step"


# ---------------------------------------------------------------------------
# Default strategy table
# ---------------------------------------------------------------------------

_STRATEGY_TABLE: dict[tuple[Difficulty, QuestionType], GenerationConfig] = {
    # Easy questions → greedy, fast, no verification overhead
    (Difficulty.EASY, QuestionType.FACTUAL): GenerationConfig(
        strategy=Strategy.GREEDY, beam_width=1, temperature=0.2,
        max_cot_tokens=200, max_answer_tokens=100,
        mid_chain_verify=False, retry_on_low_score=False,
        retry_temperature_boost=0.0, system_prompt_variant="concise",
    ),
    (Difficulty.EASY, QuestionType.ARITHMETIC): GenerationConfig(
        strategy=Strategy.GREEDY, beam_width=1, temperature=0.0,
        max_cot_tokens=300, max_answer_tokens=80,
        mid_chain_verify=False, retry_on_low_score=False,
        retry_temperature_boost=0.0, system_prompt_variant="step_by_step",
    ),
    # Medium questions → beam search with verifier scoring at the end
    (Difficulty.MEDIUM, QuestionType.FACTUAL): GenerationConfig(
        strategy=Strategy.BEAM, beam_width=4, temperature=0.6,
        max_cot_tokens=400, max_answer_tokens=150,
        mid_chain_verify=False, retry_on_low_score=True,
        retry_temperature_boost=0.2, system_prompt_variant="detailed",
    ),
    (Difficulty.MEDIUM, QuestionType.ARITHMETIC): GenerationConfig(
        strategy=Strategy.BEAM, beam_width=4, temperature=0.3,
        max_cot_tokens=500, max_answer_tokens=100,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.15, system_prompt_variant="step_by_step",
    ),
    (Difficulty.MEDIUM, QuestionType.MULTI_HOP): GenerationConfig(
        strategy=Strategy.BEAM, beam_width=6, temperature=0.65,
        max_cot_tokens=600, max_answer_tokens=200,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.2, system_prompt_variant="step_by_step",
    ),
    (Difficulty.MEDIUM, QuestionType.LOGICAL): GenerationConfig(
        strategy=Strategy.BEAM, beam_width=5, temperature=0.5,
        max_cot_tokens=500, max_answer_tokens=150,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.2, system_prompt_variant="step_by_step",
    ),
    # Hard questions → diverse sampling, full pipeline
    (Difficulty.HARD, QuestionType.FACTUAL): GenerationConfig(
        strategy=Strategy.DIVERSE, beam_width=10, temperature=0.85,
        max_cot_tokens=800, max_answer_tokens=250,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.15, system_prompt_variant="detailed",
    ),
    (Difficulty.HARD, QuestionType.ARITHMETIC): GenerationConfig(
        strategy=Strategy.DIVERSE, beam_width=8, temperature=0.5,
        max_cot_tokens=800, max_answer_tokens=150,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.1, system_prompt_variant="step_by_step",
    ),
    (Difficulty.HARD, QuestionType.MULTI_HOP): GenerationConfig(
        strategy=Strategy.DIVERSE, beam_width=12, temperature=0.9,
        max_cot_tokens=1000, max_answer_tokens=300,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.1, system_prompt_variant="step_by_step",
    ),
    (Difficulty.HARD, QuestionType.LOGICAL): GenerationConfig(
        strategy=Strategy.DIVERSE, beam_width=10, temperature=0.8,
        max_cot_tokens=900, max_answer_tokens=250,
        mid_chain_verify=True, retry_on_low_score=True,
        retry_temperature_boost=0.15, system_prompt_variant="step_by_step",
    ),
    (Difficulty.HARD, QuestionType.OPEN_ENDED): GenerationConfig(
        strategy=Strategy.DIVERSE, beam_width=6, temperature=0.95,
        max_cot_tokens=1200, max_answer_tokens=500,
        mid_chain_verify=False, retry_on_low_score=False,
        retry_temperature_boost=0.0, system_prompt_variant="detailed",
    ),
}

# Fallback defaults per difficulty
_FALLBACK: dict[Difficulty, GenerationConfig] = {
    Difficulty.EASY: _STRATEGY_TABLE[(Difficulty.EASY, QuestionType.FACTUAL)],
    Difficulty.MEDIUM: _STRATEGY_TABLE[(Difficulty.MEDIUM, QuestionType.FACTUAL)],
    Difficulty.HARD: _STRATEGY_TABLE[(Difficulty.HARD, QuestionType.FACTUAL)],
}


class StrategySelector:
    """
    Selects GenerationConfig for a question.
    Supports optional recalibration from benchmark logs.
    """

    def __init__(self, calibration_path: str | None = None):
        self._overrides: dict[str, dict] = {}
        if calibration_path and os.path.exists(calibration_path):
            self._load_calibration(calibration_path)

    def select(
        self,
        difficulty: Difficulty,
        question_type: QuestionType,
        budget_tokens: int | None = None,
    ) -> GenerationConfig:
        key = (difficulty, question_type)
        cfg = _STRATEGY_TABLE.get(key, _FALLBACK[difficulty])

        # Apply any live recalibration overrides
        override_key = f"{difficulty.value}_{question_type.value}"
        if override_key in self._overrides:
            ov = self._overrides[override_key]
            cfg = GenerationConfig(
                strategy=cfg.strategy,
                beam_width=ov.get("beam_width", cfg.beam_width),
                temperature=ov.get("temperature", cfg.temperature),
                max_cot_tokens=cfg.max_cot_tokens,
                max_answer_tokens=cfg.max_answer_tokens,
                mid_chain_verify=cfg.mid_chain_verify,
                retry_on_low_score=cfg.retry_on_low_score,
                retry_temperature_boost=cfg.retry_temperature_boost,
                system_prompt_variant=cfg.system_prompt_variant,
            )

        # Token budget clamp
        if budget_tokens is not None:
            bw = max(1, min(cfg.beam_width, budget_tokens // (cfg.max_cot_tokens + cfg.max_answer_tokens)))
            cfg = GenerationConfig(**{**cfg.__dict__, "beam_width": bw})

        return cfg

    def recalibrate(self, benchmark_summary: dict) -> dict[str, dict]:
        """
        Adjust beam_width and temperature based on accuracy-vs-cost data.

        benchmark_summary schema (per tier):
        {
          "easy_factual":   {"accuracy": 0.92, "avg_tokens": 210, "avg_latency_ms": 400},
          "medium_multi_hop": {...},
          ...
        }

        Rules:
        - accuracy < 0.70 and strategy != DIVERSE → increase beam_width by 2
        - accuracy > 0.92 and beam_width > 1      → decrease beam_width by 1
        - avg_tokens > 1500                        → reduce temperature by 0.1
        """
        adjustments: dict[str, dict] = {}
        for tier, stats in benchmark_summary.items():
            acc = stats.get("accuracy", 1.0)
            avg_tok = stats.get("avg_tokens", 0)
            parts = tier.split("_", 1)
            if len(parts) != 2:
                continue
            diff_str, qtype_str = parts
            try:
                diff = Difficulty(diff_str)
                qtype = QuestionType(qtype_str)
            except ValueError:
                continue

            cfg = _STRATEGY_TABLE.get((diff, qtype), _FALLBACK[diff])
            new_bw = cfg.beam_width
            new_temp = cfg.temperature

            if acc < 0.70 and cfg.strategy != Strategy.DIVERSE:
                new_bw = min(cfg.beam_width + 2, 16)
            elif acc > 0.92 and cfg.beam_width > 1:
                new_bw = max(cfg.beam_width - 1, 1)

            if avg_tok > 1500:
                new_temp = max(cfg.temperature - 0.1, 0.0)

            if new_bw != cfg.beam_width or new_temp != cfg.temperature:
                adjustments[tier] = {"beam_width": new_bw, "temperature": round(new_temp, 2)}

        self._overrides.update(adjustments)
        return adjustments

    def _load_calibration(self, path: str) -> None:
        with open(path) as f:
            self._overrides = json.load(f)

    def save_calibration(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._overrides, f, indent=2)
