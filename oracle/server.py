"""
ORACLE — FastAPI Backend
Exposes the ORACLE engine as a REST API consumed by the dashboard.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from oracle.engine import Oracle
from oracle.types import Difficulty, Strategy

# ---------------------------------------------------------------------------
# App & engine initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ORACLE API",
    description="Outcome-Ranked Adaptive Cognitive Logic Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton engine (shared across requests)
_engine: Optional[Oracle] = None


def get_engine() -> Oracle:
    global _engine
    if _engine is None:
        _engine = Oracle(
            log_path=os.getenv("ORACLE_LOG", "oracle_benchmark.jsonl"),
            calibration_path=os.getenv("ORACLE_CAL", "oracle_calibration.json"),
            verbose=True,
        )
    return _engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    ground_truth: Optional[str] = None
    context: Optional[str] = None
    budget_tokens: Optional[int] = None
    force_difficulty: Optional[str] = None   # "easy" | "medium" | "hard"
    force_strategy: Optional[str] = None     # "greedy" | "beam" | "diverse"


class PathSummary(BaseModel):
    path_id: str
    chain_of_thought: str
    final_answer: str
    verifier_score: float
    composite_score: float
    tokens_used: int
    generation_time_ms: float


class AskResponse(BaseModel):
    query_id: str
    question: str
    difficulty: str
    question_type: str
    strategy_used: str
    beam_width: int
    temperature: float
    final_answer: str
    ground_truth: Optional[str]
    is_correct: Optional[bool]
    total_tokens: int
    total_latency_ms: float
    avg_verifier_score: float
    score_variance: float
    num_paths: int
    paths: list[PathSummary]
    timestamp: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    engine = get_engine()

    force_diff = None
    if req.force_difficulty:
        try:
            force_diff = Difficulty(req.force_difficulty)
        except ValueError:
            raise HTTPException(400, f"Invalid difficulty: {req.force_difficulty}")

    force_strat = None
    if req.force_strategy:
        try:
            force_strat = Strategy(req.force_strategy)
        except ValueError:
            raise HTTPException(400, f"Invalid strategy: {req.force_strategy}")

    try:
        record = engine.ask(
            question=req.question,
            ground_truth=req.ground_truth,
            context=req.context,
            budget_tokens=req.budget_tokens,
            force_difficulty=force_diff,
            force_strategy=force_strat,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return AskResponse(
        query_id=record.query_id,
        question=record.question,
        difficulty=record.difficulty.value,
        question_type=record.question_type.value,
        strategy_used=record.strategy_used.value,
        beam_width=record.beam_width,
        temperature=record.temperature,
        final_answer=record.final_answer,
        ground_truth=record.ground_truth,
        is_correct=record.is_correct,
        total_tokens=record.total_tokens,
        total_latency_ms=record.total_latency_ms,
        avg_verifier_score=record.avg_verifier_score,
        score_variance=record.score_variance,
        num_paths=len(record.paths),
        paths=[
            PathSummary(
                path_id=p.path_id,
                chain_of_thought=p.chain_of_thought,
                final_answer=p.final_answer,
                verifier_score=p.verifier_score,
                composite_score=p.composite_score,
                tokens_used=p.tokens_used,
                generation_time_ms=p.generation_time_ms,
            )
            for p in record.paths
        ],
        timestamp=record.timestamp,
    )


@app.get("/stats")
def stats():
    engine = get_engine()
    return engine.stats()


@app.get("/tier-stats")
def tier_stats():
    engine = get_engine()
    ts = engine.tier_stats()
    return {
        k: {
            "tier": v.tier,
            "total": v.total,
            "correct": v.correct,
            "accuracy": v.accuracy,
            "avg_tokens": v.avg_tokens,
            "avg_latency_ms": v.avg_latency_ms,
            "avg_verifier_score": v.avg_verifier_score,
            "cost_efficiency": v.cost_efficiency,
        }
        for k, v in ts.items()
    }


@app.get("/recent")
def recent(n: int = 20):
    engine = get_engine()
    return engine.recent(n)


@app.post("/recalibrate")
def recalibrate():
    engine = get_engine()
    adjustments = engine.recalibrate()
    return {"adjustments": adjustments, "message": f"{len(adjustments)} tier(s) updated"}


@app.get("/")
def root():
    return {"message": "ORACLE API running. See /docs for Swagger UI."}
