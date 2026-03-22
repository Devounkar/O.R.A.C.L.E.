#!/usr/bin/env python3
"""
ORACLE — CLI Entrypoint
Run the engine from the command line or kick off the demo batch.

Usage:
    python main.py ask "What is the capital of France?"
    python main.py ask "What is 123 * 456?" --ground-truth "56088"
    python main.py demo
    python main.py stats
    python main.py recalibrate
    python main.py serve            # starts FastAPI server
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def cmd_ask(args):
    from oracle.engine import Oracle
    engine = Oracle(verbose=True)
    record = engine.ask(
        question=args.question,
        ground_truth=args.ground_truth,
        context=args.context,
    )
    print("\n" + "═" * 60)
    print(f"  ANSWER: {record.final_answer}")
    print(f"  Difficulty: {record.difficulty.value}  |  Strategy: {record.strategy_used.value}")
    print(f"  Paths: {len(record.paths)}  |  Tokens: {record.total_tokens}  |  Latency: {record.total_latency_ms:.0f}ms")
    if record.is_correct is not None:
        print(f"  Correct: {record.is_correct}")
    print("═" * 60)


def cmd_demo(args):
    """Run a diverse batch of demo questions."""
    from oracle.engine import Oracle
    engine = Oracle(verbose=True)

    demo_questions = [
        {
            "question": "What is 2 + 2?",
            "ground_truth": "4",
        },
        {
            "question": "What is the capital of Japan?",
            "ground_truth": "Tokyo",
        },
        {
            "question": (
                "A store sells apples for $1.20 each and oranges for $0.80 each. "
                "If Maria buys 5 apples and 7 oranges, how much does she spend in total?"
            ),
            "ground_truth": "11.60",
        },
        {
            "question": (
                "Which element has the atomic number 79, is known for its malleability, "
                "and has been used as a currency standard throughout history?"
            ),
            "ground_truth": "Gold",
        },
        {
            "question": (
                "If all mammals are warm-blooded, and dolphins are mammals, "
                "but not all warm-blooded animals can breathe underwater — "
                "can dolphins breathe underwater?"
            ),
            "ground_truth": "No",
        },
    ]

    results = []
    for item in demo_questions:
        print(f"\n{'─' * 50}")
        print(f"Q: {item['question'][:80]}...")
        record = engine.ask(**item)
        results.append({
            "question": item["question"][:60],
            "answer": record.final_answer[:60],
            "correct": record.is_correct,
            "tokens": record.total_tokens,
            "latency_ms": round(record.total_latency_ms),
        })

    print("\n\n" + "═" * 60)
    print("  DEMO RESULTS SUMMARY")
    print("═" * 60)
    for r in results:
        mark = "✓" if r["correct"] else ("✗" if r["correct"] is False else "?")
        print(f"  {mark}  {r['question'][:50]:<50}  {r['tokens']:>5} tok  {r['latency_ms']:>5}ms")
    correct = sum(1 for r in results if r["correct"])
    print(f"\n  Accuracy: {correct}/{len(results)}")
    print("═" * 60)


def cmd_stats(args):
    from oracle.engine import Oracle
    engine = Oracle(verbose=False)
    stats = engine.stats()
    tiers = engine.tier_stats()

    print("\n ORACLE BENCHMARK STATISTICS")
    print("═" * 50)
    print(json.dumps(stats, indent=2))
    print("\n TIER BREAKDOWN")
    print("─" * 50)
    for tier, s in sorted(tiers.items()):
        bar = "█" * int(s.accuracy * 20)
        print(f"  {tier:<25} {bar:<20} {s.accuracy:.0%}  ({s.total} queries)")


def cmd_recalibrate(args):
    from oracle.engine import Oracle
    engine = Oracle(verbose=True)
    adjustments = engine.recalibrate()
    if adjustments:
        print("\nApplied adjustments:")
        print(json.dumps(adjustments, indent=2))
    else:
        print("No adjustments needed — all tiers within targets.")


def cmd_serve(args):
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="ORACLE — Adaptive Reasoning Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--ground-truth", type=str, default=None)
    p_ask.add_argument("--context", type=str, default=None)

    sub.add_parser("demo", help="Run demo batch of questions")
    sub.add_parser("stats", help="Print benchmark statistics")
    sub.add_parser("recalibrate", help="Run the recalibration loop")
    sub.add_parser("serve", help="Start the FastAPI server")

    args = parser.parse_args()

    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "recalibrate":
        cmd_recalibrate(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
