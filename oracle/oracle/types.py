"""
ORACLE — Outcome-Ranked Adaptive Cognitive Logic Engine
Core data types and enumerations.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Strategy(str, Enum):
    GREEDY = "greedy"
    BEAM = "beam"
    DIVERSE = "diverse"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    ARITHMETIC = "arithmetic"
    MULTI_HOP = "multi_hop"
    LOGICAL = "logical"
    OPEN_ENDED = "open_ended"


@dataclass
class ReasoningPath:
    """A single chain-of-thought candidate with its verifier score."""
    path_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chain_of_thought: str = ""
    final_answer: str = ""
    token_logprob: float = 0.0        # avg log-prob from the generator
    verifier_score: float = 0.0       # 0–1 from the verifier model
    generation_time_ms: float = 0.0
    tokens_used: int = 0

    @property
    def composite_score(self) -> float:
        """Blend generator confidence with verifier score."""
        import math
        lp_norm = 1.0 / (1.0 + math.exp(-self.token_logprob))  # sigmoid normalise
        return 0.4 * lp_norm + 0.6 * self.verifier_score


@dataclass
class QueryRecord:
    """Full record of one ORACLE query, stored in the benchmark log."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    question_type: QuestionType = QuestionType.FACTUAL
    difficulty: Difficulty = Difficulty.MEDIUM
    strategy_used: Strategy = Strategy.BEAM
    beam_width: int = 4
    temperature: float = 0.7
    paths: list[ReasoningPath] = field(default_factory=list)
    winning_path: Optional[ReasoningPath] = None
    final_answer: str = ""
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    model_used: str = "claude-sonnet-4-20250514"

    @property
    def avg_verifier_score(self) -> float:
        if not self.paths:
            return 0.0
        return sum(p.verifier_score for p in self.paths) / len(self.paths)

    @property
    def score_variance(self) -> float:
        if len(self.paths) < 2:
            return 0.0
        avg = self.avg_verifier_score
        return sum((p.verifier_score - avg) ** 2 for p in self.paths) / len(self.paths)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "strategy_used": self.strategy_used.value,
            "beam_width": self.beam_width,
            "temperature": self.temperature,
            "final_answer": self.final_answer,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "avg_verifier_score": self.avg_verifier_score,
            "score_variance": self.score_variance,
            "num_paths": len(self.paths),
            "paths": [
                {
                    "path_id": p.path_id,
                    "chain_of_thought": p.chain_of_thought,
                    "final_answer": p.final_answer,
                    "verifier_score": p.verifier_score,
                    "composite_score": p.composite_score,
                    "tokens_used": p.tokens_used,
                    "generation_time_ms": p.generation_time_ms,
                }
                for p in self.paths
            ],
        }
