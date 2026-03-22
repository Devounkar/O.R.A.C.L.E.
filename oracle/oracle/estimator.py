"""
ORACLE — Difficulty Estimator
Classifies incoming questions into easy / medium / hard using heuristic
features + a lightweight LLM call. Returns the difficulty tier and an
estimated answer-space complexity score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from oracle.types import Difficulty, QuestionType


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

MULTI_HOP_SIGNALS = [
    "who did", "what happened after", "given that", "because of",
    "as a result", "which led to", "compare", "contrast", "explain why",
    "how does .* relate", "what is the relationship", "if .* then",
]

ARITHMETIC_SIGNALS = [
    r"\d+\s*[\+\-\*\/\^]\s*\d+",   # inline expression
    r"\b(calculate|compute|solve|how many|how much|total|sum|product|average|mean|percentage)\b",
    r"\$\d", r"\d+%",
]

OPEN_ENDED_SIGNALS = [
    "what do you think", "in your opinion", "discuss", "essay",
    "argue", "defend the position", "write a", "describe your",
    "long-term", "long term", "implications", "impact of",
]

HARD_VOCAB_SIGNALS = [
    "epistemological", "ontological", "categorical imperative", "utilitarian",
    "metaphysical", "phenomenological", "hermeneutic", "dialectical",
    "heuristic", "algorithmic complexity", "asymptotic", "stochastic",
    "bayesian", "counterfactual", "deontological", "normative",
    "alignment", "emergent", "systemic", "socioeconomic",
]


@dataclass
class DifficultyEstimate:
    difficulty: Difficulty
    question_type: QuestionType
    confidence: float           # 0–1
    reasoning_hops: int         # estimated number of logical steps
    answer_space: str           # "closed" | "numeric" | "open"
    features: dict


def estimate_difficulty(question: str) -> DifficultyEstimate:
    """
    Heuristic difficulty estimation.  No LLM call — fast O(1) classification.

    Scoring rubric
    ──────────────
    • word_count          → longer questions tend to be harder
    • clause_count        → subordinate clauses signal complexity
    • multi_hop_signals   → explicit chaining language
    • question_type       → arithmetic / logical / open-ended anchors
    • negation_count      → "not", "except", "unless" raise difficulty
    • named_entity_count  → dense entity mention → multi-hop
    """
    q = question.lower().strip()
    words = q.split()

    # --- raw features ---
    word_count = len(words)
    clause_count = len(re.findall(r"\b(that|which|who|because|although|however|therefore|thus|since|when|if|unless)\b", q))
    negation_count = len(re.findall(r"\b(not|no|never|neither|nor|except|unless|without)\b", q))
    multi_hop_count = sum(1 for sig in MULTI_HOP_SIGNALS if re.search(sig, q))
    arithmetic_count = sum(1 for sig in ARITHMETIC_SIGNALS if re.search(sig, q))
    open_ended_count = sum(1 for sig in OPEN_ENDED_SIGNALS if sig in q)

    # simple named-entity proxy: title-cased tokens in original question
    ne_count = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question))

    hard_vocab_count = sum(1 for sig in HARD_VOCAB_SIGNALS if sig in q)

    # --- question type classification ---
    if arithmetic_count > 0 and open_ended_count == 0:
        q_type = QuestionType.ARITHMETIC
        answer_space = "numeric"
    elif open_ended_count > 0:
        q_type = QuestionType.OPEN_ENDED
        answer_space = "open"
    elif multi_hop_count > 0 or ne_count >= 3:
        q_type = QuestionType.MULTI_HOP
        answer_space = "closed"
    elif clause_count >= 2:
        q_type = QuestionType.LOGICAL
        answer_space = "closed"
    else:
        q_type = QuestionType.FACTUAL
        answer_space = "closed"

    # --- score → difficulty ---
    score = 0
    score += min(word_count / 8, 5)           # 0–5 pts
    score += min(clause_count * 1.0, 4)       # 0–4 pts
    score += min(negation_count * 1.2, 3)     # 0–3 pts
    score += min(multi_hop_count * 2.0, 4)    # 0–4 pts
    score += min(ne_count * 0.5, 3)           # 0–3 pts
    score += min(hard_vocab_count * 2.5, 6)   # 0–6 pts  (strong hard signal)
    score += 2 if q_type == QuestionType.OPEN_ENDED else 0
    score += 1 if q_type == QuestionType.ARITHMETIC else 0
    score += 1 if q_type == QuestionType.LOGICAL else 0

    # estimated reasoning hops
    hops = max(1, multi_hop_count + clause_count // 2)

    if score <= 4:
        difficulty = Difficulty.EASY
        confidence = 0.85 - score * 0.04
    elif score < 9:
        difficulty = Difficulty.MEDIUM
        confidence = 0.80
    else:
        difficulty = Difficulty.HARD
        confidence = min(0.92, 0.72 + (score - 9) * 0.02)

    features = {
        "word_count": word_count,
        "clause_count": clause_count,
        "negation_count": negation_count,
        "multi_hop_signals": multi_hop_count,
        "arithmetic_signals": arithmetic_count,
        "named_entities": ne_count,
        "hard_vocab_signals": hard_vocab_count,
        "raw_score": round(score, 2),
    }

    return DifficultyEstimate(
        difficulty=difficulty,
        question_type=q_type,
        confidence=round(confidence, 3),
        reasoning_hops=hops,
        answer_space=answer_space,
        features=features,
    )
