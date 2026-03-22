"""
ORACLE — Main Engine
Orchestrates the full pipeline:
  Question → Estimator → Strategy Selector → CoT Generator
  → Verifier → Aggregator → Benchmark Logger → Answer
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from oracle.aggregator import AggregationResult, aggregate
from oracle.benchmark import BenchmarkLogger
from oracle.estimator import DifficultyEstimate, estimate_difficulty
from oracle.generator import CoTGenerator
from oracle.strategy import GenerationConfig, StrategySelector
from oracle.types import Difficulty, QueryRecord, QuestionType, ReasoningPath, Strategy
from oracle.verifier import Verifier


LOW_SCORE_RETRY_THRESHOLD = 0.35   # if all verifier scores below this, retry


class Oracle:
    """
    The main ORACLE reasoning engine.

    Usage:
        engine = Oracle()
        result = engine.ask("What is the capital of France?")
        print(result.final_answer)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        log_path: str = "oracle_benchmark.jsonl",
        calibration_path: Optional[str] = "oracle_calibration.json",
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.generator = CoTGenerator(model=model)
        self.verifier = Verifier(model=model)
        self.selector = StrategySelector(calibration_path=calibration_path)
        self.logger = BenchmarkLogger(log_path=log_path)
        self._calibration_path = calibration_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        force_difficulty: Optional[Difficulty] = None,
        force_strategy: Optional[Strategy] = None,
    ) -> QueryRecord:
        """
        Full pipeline: classify → select strategy → generate → verify → aggregate → log.
        Returns a populated QueryRecord with the final answer and all paths.
        """
        t_start = time.perf_counter()

        # 1. Difficulty estimation
        estimate = estimate_difficulty(question)
        difficulty = force_difficulty or estimate.difficulty
        q_type = estimate.question_type
        self._log(f"[Estimator] {difficulty.value} | {q_type.value} | hops≈{estimate.reasoning_hops}")

        # 2. Strategy selection
        config = self.selector.select(difficulty, q_type, budget_tokens)
        if force_strategy:
            from oracle.strategy import GenerationConfig
            config = GenerationConfig(**{**config.__dict__, "strategy": force_strategy})
        self._log(f"[Strategy] {config.strategy.value} | beam={config.beam_width} | temp={config.temperature}")

        # 3. Generate paths
        self._log(f"[Generator] Generating {config.beam_width} path(s)...")
        paths = self.generator.generate(question, config, context=context)

        # 4. Verify paths
        self._log(f"[Verifier] Scoring {len(paths)} path(s)...")
        paths = self.verifier.score_batch(question, paths)

        # 5. Optional retry if scores are all low
        if config.retry_on_low_score and paths:
            top_score = max(p.verifier_score for p in paths)
            if top_score < LOW_SCORE_RETRY_THRESHOLD:
                self._log(f"[Retry] Top score {top_score:.2f} < threshold; retrying with higher temp...")
                boosted_temp = min(1.0, config.temperature + config.retry_temperature_boost)
                retry_config = config.__class__(**{
                    **config.__dict__,
                    "temperature": boosted_temp,
                    "beam_width": max(2, config.beam_width // 2),
                })
                retry_paths = self.generator.generate(question, retry_config, context=context)
                retry_paths = self.verifier.score_batch(question, retry_paths)
                paths = sorted(paths + retry_paths, key=lambda p: p.verifier_score, reverse=True)

        # 6. Aggregate
        result: AggregationResult = aggregate(paths, q_type)
        self._log(f"[Aggregator] confidence={result.confidence:.2f} | consensus={result.consensus_ratio:.2f}")

        # 7. Build record
        t_end = time.perf_counter()
        total_tokens = sum(p.tokens_used for p in paths)
        record = QueryRecord(
            question=question,
            question_type=q_type,
            difficulty=difficulty,
            strategy_used=config.strategy,
            beam_width=config.beam_width,
            temperature=config.temperature,
            paths=paths,
            winning_path=result.winning_path,
            final_answer=result.winning_answer,
            ground_truth=ground_truth,
            total_tokens=total_tokens,
            total_latency_ms=(t_end - t_start) * 1000,
            model_used=self.model,
        )

        # 8. Evaluate if ground truth provided
        if ground_truth is not None:
            from oracle.aggregator import _answers_match
            record.is_correct = _answers_match(
                result.winning_answer, ground_truth, q_type
            )
            self._log(f"[Eval] Correct: {record.is_correct}")

        # 9. Log
        self.logger.log(record)

        self._log(
            f"[Done] '{result.winning_answer[:80]}' | "
            f"{total_tokens} tokens | {record.total_latency_ms:.0f}ms"
        )
        return record

    # ------------------------------------------------------------------
    # Recalibration
    # ------------------------------------------------------------------

    def recalibrate(self) -> dict[str, dict]:
        """
        Run the recalibration loop based on accumulated benchmark data.
        Returns a dict of applied adjustments.
        """
        summary = self.logger.benchmark_summary_for_calibration()
        adjustments = self.selector.recalibrate(summary)
        if self._calibration_path:
            self.selector.save_calibration(self._calibration_path)
        self._log(f"[Recalibrate] {len(adjustments)} tier(s) adjusted: {list(adjustments.keys())}")
        return adjustments

    # ------------------------------------------------------------------
    # Stats passthrough
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return self.logger.overall_stats()

    def tier_stats(self):
        return self.logger.compute_tier_stats()

    def recent(self, n: int = 20) -> list[dict]:
        return self.logger.recent_records(n)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
