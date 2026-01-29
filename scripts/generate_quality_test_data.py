#!/usr/bin/env python
"""
Generate test quality metrics data for dashboard demonstration.

This script creates sample metrics and history files for testing
the RAG Quality Dashboard.

Usage:
    uv run python scripts/generate_quality_test_data.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


def generate_mock_metrics() -> dict:
    """Generate mock quality metrics."""
    import random

    random.seed(42)  # For reproducibility

    return {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": {
            "score": random.uniform(0.75, 0.95),
            "trend": random.choice(["improve", "maintain", "decrease"]),
            "change": random.uniform(-0.05, 0.08),
        },
        "answer_relevancy": {
            "score": random.uniform(0.70, 0.90),
            "trend": random.choice(["improve", "maintain", "decrease"]),
            "change": random.uniform(-0.04, 0.06),
        },
        "precision": {
            "score": random.uniform(0.72, 0.92),
            "trend": random.choice(["improve", "maintain", "decrease"]),
            "change": random.uniform(-0.03, 0.07),
        },
        "recall": {
            "score": random.uniform(0.68, 0.88),
            "trend": random.choice(["improve", "maintain", "decrease"]),
            "change": random.uniform(-0.06, 0.05),
        },
        "personas": {
            "faculty": {
                "score": random.uniform(0.80, 0.95),
                "count": random.randint(30, 60),
            },
            "student": {
                "score": random.uniform(0.70, 0.85),
                "count": random.randint(80, 150),
            },
            "staff": {
                "score": random.uniform(0.75, 0.90),
                "count": random.randint(20, 50),
            },
        },
    }


def generate_mock_history(days: int = 30) -> list:
    """Generate mock quality metrics history."""
    import random

    random.seed(42)  # For reproducibility

    history = []
    base_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        date = base_date + timedelta(days=i)

        # Add some realistic trends and variance
        trend_factor = i / days  # Slight improvement over time
        daily_variance = random.uniform(-0.03, 0.03)

        history.append(
            {
                "timestamp": date.isoformat(),
                "faithfulness": min(0.95, 0.75 + trend_factor * 0.15 + daily_variance),
                "answer_relevancy": min(
                    0.90, 0.70 + trend_factor * 0.12 + daily_variance
                ),
                "precision": min(0.92, 0.72 + trend_factor * 0.14 + daily_variance),
                "recall": min(0.88, 0.68 + trend_factor * 0.13 + daily_variance),
            }
        )

    return history


def main():
    """Generate test data files."""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate metrics file
    metrics_path = data_dir / "quality_metrics.json"
    metrics = generate_mock_metrics()

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated metrics file: {metrics_path}")
    print(f"   - Faithfulness: {metrics['faithfulness']['score']:.1%}")
    print(f"   - Answer Relevancy: {metrics['answer_relevancy']['score']:.1%}")
    print(f"   - Precision: {metrics['precision']['score']:.1%}")
    print(f"   - Recall: {metrics['recall']['score']:.1%}")

    # Generate history file
    history_path = data_dir / "quality_history.jsonl"
    history = generate_mock_history(days=30)

    with open(history_path, "w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Generated history file: {history_path}")
    print(f"   - Records: {len(history)} days")
    print(
        f"   - Date range: {history[0]['timestamp'][:10]} to {history[-1]['timestamp'][:10]}"
    )

    print("\n🚀 You can now run the quality dashboard:")
    print("   uv run python -m src.rag.interface.web.quality_dashboard")


if __name__ == "__main__":
    main()
