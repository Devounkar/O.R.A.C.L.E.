# O.R.A.C.L.E.[README.md](https://github.com/user-attachments/files/26163852/README.md)
# ORACLE
### Outcome-Ranked Adaptive Cognitive Logic Engine

A production-grade reasoning system that classifies question difficulty, selects an optimal generation strategy, produces multiple chain-of-thought candidates, scores them with a verifier model, and aggregates to the best answer — with a live recalibration loop.

---

## Architecture

```
Question
   │
   ▼
┌─────────────────────┐
│  Difficulty         │  Heuristic feature extraction (O(1), no API call)
│  Estimator          │  → Easy / Medium / Hard  +  Question Type
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Strategy           │  Lookup table: (difficulty × question_type)
│  Selector           │  → beam_width, temperature, CoT depth, verify flags
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CoT Generator      │  k independent samples with varying temperature
│  (Beam Search)      │  → list[ReasoningPath]  (chain_of_thought + answer)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Verifier Model     │  PRM: scores each (Q, CoT, A) triple → 0–1
│                     │  Evaluates: coherence, faithfulness, plausibility
└────────┬────────────┘
         │    ↙ retry if all scores < 0.35
         ▼
┌─────────────────────┐
│  Self-Consistency   │  Cluster by answer equality, weight by composite_score
│  Aggregator         │  → winning_answer + confidence
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Benchmark Logger   │  JSONL append, tier stats, recalibration feed
└─────────────────────┘
         │
         ▼  (periodic)
┌─────────────────────┐
│  Recalibration      │  Adjusts beam_width + temperature per tier
│  Loop               │  based on accuracy-vs-cost data
└─────────────────────┘
```

---

## Modules

| File | Responsibility |
|------|---------------|
| `oracle/types.py` | Core dataclasses: `ReasoningPath`, `QueryRecord`, enums |
| `oracle/estimator.py` | Heuristic difficulty & question-type classification |
| `oracle/strategy.py` | Strategy table, `StrategySelector`, recalibration logic |
| `oracle/generator.py` | CoT generation via Anthropic API, beam simulation |
| `oracle/verifier.py` | PRM scoring via separate LLM call |
| `oracle/aggregator.py` | Answer clustering + weighted majority vote |
| `oracle/benchmark.py` | JSONL logger, tier statistics, calibration summary |
| `oracle/engine.py` | Main `Oracle` class — orchestrates the full pipeline |
| `server.py` | FastAPI REST API |
| `main.py` | CLI entrypoint |
| `frontend/index.html` | Dashboard UI |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Ask a question from CLI
python main.py ask "Which planet has the most moons as of 2024?"

# 4. Run the demo batch
python main.py demo

# 5. Start the API server
python main.py serve
# or: uvicorn server:app --reload

# 6. Open the dashboard
open frontend/index.html
```

---

## CLI Reference

```bash
python main.py ask "Your question" [--ground-truth "expected"] [--context "..."]
python main.py demo          # 5 diverse demo questions
python main.py stats         # print benchmark statistics
python main.py recalibrate   # run the recalibration loop
python main.py serve         # start FastAPI on port 8000
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Submit a question, get full reasoning record |
| `/stats` | GET | Overall accuracy, tokens, latency |
| `/tier-stats` | GET | Per-tier breakdown |
| `/recent?n=N` | GET | Last N query records |
| `/recalibrate` | POST | Run recalibration loop |
| `/health` | GET | Health check |

### POST /ask body

```json
{
  "question": "What is the boiling point of water at sea level?",
  "ground_truth": "100°C",
  "context": null,
  "budget_tokens": null,
  "force_difficulty": null,
  "force_strategy": null
}
```

---

## Strategy Table

| Difficulty | Type | Strategy | Beam | Temp | Mid-verify |
|------------|------|----------|------|------|------------|
| Easy | Factual | Greedy | 1 | 0.2 | No |
| Easy | Arithmetic | Greedy | 1 | 0.0 | No |
| Medium | Factual | Beam | 4 | 0.6 | No |
| Medium | Multi-hop | Beam | 6 | 0.65 | Yes |
| Hard | Factual | Diverse | 10 | 0.85 | Yes |
| Hard | Multi-hop | Diverse | 12 | 0.9 | Yes |
| Hard | Open-ended | Diverse | 6 | 0.95 | No |

---

## Recalibration Rules

- `accuracy < 0.70` → `beam_width + 2`
- `accuracy > 0.92` → `beam_width − 1`
- `avg_tokens > 1500` → `temperature − 0.1`

Calibration overrides are saved to `oracle_calibration.json` and loaded on startup.

---

## Tests

```bash
pytest tests/ -v
```

All tests run without API keys (no live LLM calls).

---

## Scoring Formulas

**Composite score** (used for ranking and clustering weight):
```
composite = 0.4 × sigmoid(token_logprob) + 0.6 × verifier_score
```

**Difficulty raw score**:
```
score = min(word_count/10, 4)
      + min(clause_count × 0.8, 3)
      + min(negation_count × 1.0, 2)
      + min(multi_hop_signals × 1.5, 3)
      + min(named_entities × 0.4, 2)
      + 2 if open_ended
      + 1 if arithmetic
```
- score ≤ 4 → Easy
- score ≤ 9 → Medium
- score > 9 → Hard
