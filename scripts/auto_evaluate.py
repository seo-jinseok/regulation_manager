#!/usr/bin/env python
"""
Auto Evaluation & Improvement Script via RAG.

This script orchestrates:
1. Running evaluation on test cases (evaluation_dataset.json).
2. Analyzing failures to mimic negative feedback.
3. Using AutoLearnUseCase to generate specific improvement suggestions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.application.auto_learn import AutoLearnUseCase
    from src.rag.application.evaluate import EvaluationUseCase
    from src.rag.infrastructure.feedback import FeedbackCollector

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def setup_components():
    """Initialize all necessary components."""
    from src.rag.application.auto_learn import AutoLearnUseCase
    from src.rag.application.evaluate import EvaluationUseCase
    from src.rag.application.search_usecase import SearchUseCase
    from src.rag.infrastructure.chroma_store import ChromaVectorStore
    from src.rag.infrastructure.feedback import FeedbackCollector
    from src.rag.infrastructure.llm_adapter import LLMClientAdapter

    # 1. Store
    db_path = "data/chroma_db"
    store = ChromaVectorStore(persist_directory=db_path)

    # 2. LLM
    llm = None
    try:
        llm = LLMClientAdapter(provider="ollama")  # Default, can be overridden by env
    except Exception as e:
        print(f"Warning: LLM init failed ({e}). LLM features will be disabled.")

    # 3. Search
    search = SearchUseCase(store, llm_client=llm, use_reranker=True)

    # 4. Evaluation
    eval_usecase = EvaluationUseCase(search_usecase=search)

    # 5. Feedback & AutoLearn
    # We use a temporary file for feedback to avoid polluting the main log during auto-eval
    temp_feedback_path = PROJECT_ROOT / "data" / "temp_auto_feedback.jsonl"
    feedback_collector = FeedbackCollector(feedback_path=str(temp_feedback_path))
    # Clear previous temp feedback
    feedback_collector.clear_feedback()

    auto_learn = AutoLearnUseCase(feedback_collector=feedback_collector, llm_client=llm)

    return eval_usecase, auto_learn, feedback_collector


def run_evaluation_loop(
    eval_usecase: EvaluationUseCase, feedback_collector: FeedbackCollector
):
    """Run evaluation and feed failures to feedback collector."""
    print("🚀 Starting Automated Evaluation...")
    summary = eval_usecase.run_evaluation(top_k=5)

    print(eval_usecase.format_summary(summary))

    if summary.failed_cases == 0:
        print("✅ All test cases passed! No improvements needed.")
        return False

    print(f"\nCAPTURING {summary.failed_cases} FAILURES FOR ANALYSIS...")

    # Convert failures to feedback
    for result in summary.results:
        if not result.passed:
            # We treat failed cases as "negative" feedback for the system to learn from
            # For this to work, we need the result (even if wrong) to attach feedback to
            # If no results found, we can't really attach feedback to a specific chunk,
            # but AutoLearn might handle general query failures.

            # Use the top result if available, or a dummy if empty
            if result.found_rule_codes:
                # Assuming index 0 is the top result
                # In a real scenario, we'd need the Chunk ID.
                # But FeedbackEntry uses result_id.
                # Let's see if we can get it from the search result if we had access.
                # EvaluationResult stores found_rule_codes but not IDs.
                # We might need to re-search or adjust logic.
                # For now, let's just log it with a placeholder if we can't get ID easily.
                # Wait, record_feedback needs result_id and rule_code.
                pass

            # Since EvaluationResult abstracts away the full search result objects list
            # (it only keeps strings), we might want to re-run search or
            # improve EvaluationResult to keep reference.
            # OR, simpler: just record it with "simulation" placeholders.

            feedback_collector.record_feedback(
                query=result.test_case.query,
                result_id="simulated_failure",
                rule_code=result.found_rule_codes[0]
                if result.found_rule_codes
                else "NONE",
                rating="negative",
                comment=f"Auto-eval failure. Expected: {result.test_case.expected_intents} / {result.test_case.expected_keywords}",
                matched_intents=result.matched_intents,
            )
            print(f"  - Recorded failure for: {result.test_case.query}")

    return True


def analyze_and_suggest(auto_learn: AutoLearnUseCase):
    """Analyze the collected feedback and print suggestions."""
    print("\n🧠 Analyzing Failures & Generating Suggestions...")
    result = auto_learn.analyze_feedback()
    print(auto_learn.format_suggestions(result))

    # Save suggestions to file for other tools to pick up
    output_path = PROJECT_ROOT / "data" / "output" / "improvement_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suggestions_data = [
        {
            "type": s.type,
            "priority": s.priority,
            "description": s.description,
            "suggested_value": s.suggested_value,
        }
        for s in result.suggestions
    ]

    output_path.write_text(
        json.dumps(
            {"timestamp": "now", "suggestions": suggestions_data},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n💾 Suggestions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto Evaluation & Improvement Script")
    parser.add_argument("--run", action="store_true", help="Run full evaluation loop")
    args = parser.parse_args()

    if args.run:
        eval_uc, auto_learn, fb_collector = setup_components()
        has_failures = run_evaluation_loop(eval_uc, fb_collector)
        if has_failures:
            analyze_and_suggest(auto_learn)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
