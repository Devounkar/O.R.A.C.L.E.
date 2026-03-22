"""
ORACLE — Test Suite
Tests for the estimator, strategy selector, aggregator, and benchmark logger.
Does not require API keys (no live LLM calls).
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oracle.estimator import estimate_difficulty
from oracle.types import Difficulty, QuestionType, ReasoningPath, QueryRecord, Strategy
from oracle.strategy import StrategySelector
from oracle.aggregator import aggregate, _normalise_text, _normalise_numeric, _answers_match
from oracle.benchmark import BenchmarkLogger


# ─────────────────────────────────────────────────────────────────────────────
# Estimator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimator:
    def test_simple_factual_is_easy(self):
        est = estimate_difficulty("What is the capital of France?")
        assert est.difficulty == Difficulty.EASY
        assert est.question_type == QuestionType.FACTUAL

    def test_arithmetic_detected(self):
        est = estimate_difficulty("Calculate the total if you have 15 items at $4.99 each.")
        assert est.question_type == QuestionType.ARITHMETIC

    def test_multi_hop_detected(self):
        est = estimate_difficulty(
            "Who was the president of the United States when the Berlin Wall fell, "
            "and what economic policy did he pursue as a result?"
        )
        assert est.difficulty in (Difficulty.MEDIUM, Difficulty.HARD)
        assert est.question_type in (QuestionType.MULTI_HOP, QuestionType.FACTUAL)
        assert est.reasoning_hops >= 1

    def test_hard_question(self):
        est = estimate_difficulty(
            "Compare and contrast the epistemological implications of Kantian categorical "
            "imperatives versus utilitarian calculus in the context of AI alignment, "
            "given that neither framework was designed for non-human agents and both "
            "rely on assumptions about rational agency that may not hold."
        )
        assert est.difficulty == Difficulty.HARD

    def test_open_ended_detected(self):
        est = estimate_difficulty("Discuss the long-term economic effects of automation on labour markets.")
        assert est.question_type == QuestionType.OPEN_ENDED

    def test_features_present(self):
        est = estimate_difficulty("What is 2 + 2?")
        assert "word_count" in est.features
        assert "raw_score" in est.features
        assert 0 < est.confidence <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategySelector:
    def setup_method(self):
        self.sel = StrategySelector()

    def test_easy_factual_greedy(self):
        cfg = self.sel.select(Difficulty.EASY, QuestionType.FACTUAL)
        assert cfg.beam_width == 1
        assert cfg.temperature <= 0.3

    def test_hard_multi_hop_diverse(self):
        from oracle.types import Strategy
        cfg = self.sel.select(Difficulty.HARD, QuestionType.MULTI_HOP)
        assert cfg.strategy == Strategy.DIVERSE
        assert cfg.beam_width >= 8
        assert cfg.mid_chain_verify is True

    def test_budget_clamp(self):
        cfg = self.sel.select(Difficulty.HARD, QuestionType.MULTI_HOP, budget_tokens=500)
        assert cfg.beam_width >= 1

    def test_recalibrate_increases_beam(self):
        summary = {"easy_factual": {"accuracy": 0.50, "avg_tokens": 300, "avg_latency_ms": 200}}
        adjustments = self.sel.recalibrate(summary)
        assert "easy_factual" in adjustments
        assert adjustments["easy_factual"]["beam_width"] > 1

    def test_recalibrate_decreases_beam(self):
        # Artificially high-accuracy medium tier
        summary = {"medium_factual": {"accuracy": 0.95, "avg_tokens": 400, "avg_latency_ms": 600}}
        adjustments = self.sel.recalibrate(summary)
        if "medium_factual" in adjustments:
            from oracle.strategy import _STRATEGY_TABLE
            original_bw = _STRATEGY_TABLE[(Difficulty.MEDIUM, QuestionType.FACTUAL)].beam_width
            assert adjustments["medium_factual"]["beam_width"] <= original_bw


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregator:
    def _make_path(self, answer: str, vs: float) -> ReasoningPath:
        p = ReasoningPath(chain_of_thought="CoT", final_answer=answer)
        p.verifier_score = vs
        return p

    def test_single_path(self):
        paths = [self._make_path("Paris", 0.9)]
        result = aggregate(paths)
        assert result.winning_answer == "Paris"
        assert result.consensus_ratio == 1.0

    def test_majority_wins(self):
        paths = [
            self._make_path("Paris", 0.8),
            self._make_path("Paris", 0.7),
            self._make_path("London", 0.9),
        ]
        result = aggregate(paths)
        # Paris has 2 paths totalling 1.5; London has 1 path at 0.9
        assert result.winning_answer.lower() == "paris"

    def test_high_score_wins_tiebreak(self):
        paths = [
            self._make_path("42", 0.95),
            self._make_path("42", 0.85),
            self._make_path("41", 0.80),
        ]
        result = aggregate(paths, QuestionType.ARITHMETIC)
        assert "42" in result.winning_answer

    def test_normalise_case_insensitive(self):
        assert _normalise_text("Paris") == _normalise_text("paris")
        assert _normalise_text("THE ANSWER IS 42.") == _normalise_text("the answer is 42")

    def test_numeric_tolerance(self):
        assert _normalise_numeric("42.0") == pytest.approx(42.0)
        assert _normalise_numeric("$1,234.56") == pytest.approx(1234.56)
        assert _normalise_numeric("no number here") is None

    def test_answers_match_numeric(self):
        assert _answers_match("42", "42.0", QuestionType.ARITHMETIC)
        assert not _answers_match("42", "43", QuestionType.ARITHMETIC)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate([])


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark logger tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkLogger:
    def setup_method(self, method):
        import tempfile, os
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        self.logger = BenchmarkLogger(log_path=self.tmp.name)

    def teardown_method(self, method):
        os.unlink(self.tmp.name)

    def _make_record(self, diff: Difficulty, correct: bool) -> QueryRecord:
        rec = QueryRecord(
            question="test?",
            question_type=QuestionType.FACTUAL,
            difficulty=diff,
            strategy_used=Strategy.BEAM,
            final_answer="ans",
            ground_truth="ans",
            is_correct=correct,
            total_tokens=300,
            total_latency_ms=500.0,
        )
        return rec

    def test_log_and_reload(self):
        rec = self._make_record(Difficulty.EASY, True)
        self.logger.log(rec)
        logger2 = BenchmarkLogger(log_path=self.tmp.name)
        assert len(logger2.all_records()) == 1

    def test_tier_stats(self):
        self.logger.log(self._make_record(Difficulty.EASY, True))
        self.logger.log(self._make_record(Difficulty.EASY, True))
        self.logger.log(self._make_record(Difficulty.EASY, False))
        stats = self.logger.compute_tier_stats()
        assert "easy_factual" in stats
        s = stats["easy_factual"]
        assert s.total == 3
        assert s.correct == 2
        assert abs(s.accuracy - 2/3) < 0.01

    def test_overall_stats(self):
        self.logger.log(self._make_record(Difficulty.HARD, True))
        overall = self.logger.overall_stats()
        assert overall["total_queries"] == 1
        assert overall["overall_accuracy"] == 1.0

    def test_benchmark_summary_keys(self):
        self.logger.log(self._make_record(Difficulty.MEDIUM, True))
        summary = self.logger.benchmark_summary_for_calibration()
        assert "medium_factual" in summary
        assert "accuracy" in summary["medium_factual"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
