"""
ORACLE — Benchmark Logger & Evaluator
Persists QueryRecords to a JSONL file and computes accuracy/cost statistics
per difficulty×question_type tier.  Drives the recalibration feedback loop.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from oracle.types import Difficulty, QueryRecord, QuestionType


@dataclass
class TierStats:
    tier: str
    total: int
    correct: int
    accuracy: float
    avg_tokens: float
    avg_latency_ms: float
    avg_verifier_score: float
    cost_efficiency: float       # accuracy / avg_tokens (higher = better)


class BenchmarkLogger:
    """
    Append-only JSONL log of every QueryRecord.
    Provides in-memory aggregation for the dashboard and recalibration.
    """

    def __init__(self, log_path: str = "oracle_benchmark.jsonl"):
        self.log_path = log_path
        self._records: list[QueryRecord] = []
        self._load_existing()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, record: QueryRecord) -> None:
        self._records.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Read / aggregate
    # ------------------------------------------------------------------

    def all_records(self) -> list[QueryRecord]:
        return list(self._records)

    def compute_tier_stats(self) -> dict[str, TierStats]:
        """
        Group records by difficulty_questiontype and compute stats.
        Returns a dict keyed by e.g. "medium_multi_hop".
        """
        buckets: dict[str, list[QueryRecord]] = defaultdict(list)
        for rec in self._records:
            key = f"{rec.difficulty.value}_{rec.question_type.value}"
            buckets[key].append(rec)

        stats: dict[str, TierStats] = {}
        for tier, records in buckets.items():
            evaluated = [r for r in records if r.is_correct is not None]
            correct = sum(1 for r in evaluated if r.is_correct)
            accuracy = correct / len(evaluated) if evaluated else 0.0
            avg_tok = sum(r.total_tokens for r in records) / len(records)
            avg_lat = sum(r.total_latency_ms for r in records) / len(records)
            avg_vs = sum(r.avg_verifier_score for r in records) / len(records)
            stats[tier] = TierStats(
                tier=tier,
                total=len(records),
                correct=correct,
                accuracy=round(accuracy, 4),
                avg_tokens=round(avg_tok, 1),
                avg_latency_ms=round(avg_lat, 1),
                avg_verifier_score=round(avg_vs, 4),
                cost_efficiency=round(accuracy / (avg_tok / 100 + 1e-6), 4),
            )
        return stats

    def overall_stats(self) -> dict:
        evaluated = [r for r in self._records if r.is_correct is not None]
        correct = sum(1 for r in evaluated if r.is_correct)
        total = len(self._records)
        return {
            "total_queries": total,
            "evaluated": len(evaluated),
            "overall_accuracy": round(correct / len(evaluated), 4) if evaluated else 0.0,
            "total_tokens_used": sum(r.total_tokens for r in self._records),
            "avg_latency_ms": round(
                sum(r.total_latency_ms for r in self._records) / total, 1
            ) if total else 0.0,
            "strategy_breakdown": self._strategy_breakdown(),
        }

    def recent_records(self, n: int = 20) -> list[dict]:
        """Return the n most recent records as dicts (for the dashboard)."""
        return [r.to_dict() for r in sorted(
            self._records, key=lambda r: r.timestamp, reverse=True
        )[:n]]

    def benchmark_summary_for_calibration(self) -> dict[str, dict]:
        """Returns the format expected by StrategySelector.recalibrate()."""
        tier_stats = self.compute_tier_stats()
        return {
            tier: {
                "accuracy": s.accuracy,
                "avg_tokens": s.avg_tokens,
                "avg_latency_ms": s.avg_latency_ms,
            }
            for tier, s in tier_stats.items()
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strategy_breakdown(self) -> dict:
        from collections import Counter
        c = Counter(r.strategy_used.value for r in self._records)
        return dict(c)

    def _load_existing(self) -> None:
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = QueryRecord(
                        query_id=d["query_id"],
                        question=d["question"],
                        question_type=QuestionType(d["question_type"]),
                        difficulty=Difficulty(d["difficulty"]),
                        final_answer=d["final_answer"],
                        ground_truth=d.get("ground_truth"),
                        is_correct=d.get("is_correct"),
                        total_tokens=d["total_tokens"],
                        total_latency_ms=d["total_latency_ms"],
                        timestamp=d["timestamp"],
                        model_used=d.get("model_used", "unknown"),
                    )
                    from oracle.types import Strategy
                    rec.strategy_used = Strategy(d["strategy_used"])
                    self._records.append(rec)
                except Exception:
                    pass  # skip malformed lines
