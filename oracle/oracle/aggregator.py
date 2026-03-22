"""
ORACLE — Self-Consistency Aggregator
Selects the best final answer from a set of scored ReasoningPaths.

Algorithm
─────────
1. Cluster answers by semantic similarity (normalised string matching)
2. Each cluster gets a weight = sum of composite_scores of its members
3. The cluster with the highest total weight wins
4. Within the winning cluster, return the path with the highest composite_score

For numeric answers, we use a tolerance-based equality check.
For text answers, we normalise whitespace/case/punctuation before clustering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from oracle.types import QuestionType, ReasoningPath


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Strip punctuation, lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalise_numeric(text: str) -> Optional[float]:
    """Try to extract a numeric value from an answer string."""
    text = text.replace(",", "").strip()
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _answers_match(a: str, b: str, question_type: QuestionType) -> bool:
    """Return True if two answer strings should be considered the same."""
    if question_type == QuestionType.ARITHMETIC:
        va, vb = _normalise_numeric(a), _normalise_numeric(b)
        if va is not None and vb is not None:
            return abs(va - vb) < 1e-6
    # Generic: normalised string equality
    return _normalise_text(a) == _normalise_text(b)


# ---------------------------------------------------------------------------
# Cluster + weight
# ---------------------------------------------------------------------------

@dataclass
class AnswerCluster:
    representative: str
    members: list[ReasoningPath]

    @property
    def total_weight(self) -> float:
        return sum(p.composite_score for p in self.members)

    @property
    def best_path(self) -> ReasoningPath:
        return max(self.members, key=lambda p: p.composite_score)


@dataclass
class AggregationResult:
    winning_answer: str
    winning_path: ReasoningPath
    clusters: list[AnswerCluster]
    confidence: float          # winning_weight / total_weight
    consensus_ratio: float     # fraction of paths in winning cluster


def aggregate(
    paths: list[ReasoningPath],
    question_type: QuestionType = QuestionType.FACTUAL,
) -> AggregationResult:
    """
    Aggregate a list of scored ReasoningPaths into a single best answer.
    Paths should already be sorted by verifier_score descending.
    """
    if not paths:
        raise ValueError("No paths to aggregate")

    if len(paths) == 1:
        p = paths[0]
        cluster = AnswerCluster(representative=p.final_answer, members=[p])
        return AggregationResult(
            winning_answer=p.final_answer,
            winning_path=p,
            clusters=[cluster],
            confidence=p.composite_score,
            consensus_ratio=1.0,
        )

    # Build clusters
    clusters: list[AnswerCluster] = []
    for path in paths:
        placed = False
        for cluster in clusters:
            if _answers_match(path.final_answer, cluster.representative, question_type):
                cluster.members.append(path)
                placed = True
                break
        if not placed:
            clusters.append(AnswerCluster(
                representative=path.final_answer,
                members=[path],
            ))

    # Sort clusters by total weight
    clusters.sort(key=lambda c: c.total_weight, reverse=True)
    winner = clusters[0]
    total_weight = sum(c.total_weight for c in clusters) or 1e-9

    return AggregationResult(
        winning_answer=winner.representative,
        winning_path=winner.best_path,
        clusters=clusters,
        confidence=winner.total_weight / total_weight,
        consensus_ratio=len(winner.members) / len(paths),
    )
