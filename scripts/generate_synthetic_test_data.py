#!/usr/bin/env python3
"""
Flip-the-RAG Synthetic Test Data Generation Demo

Generates high-quality test cases from regulation documents for RAG evaluation.

Usage:
    python scripts/generate_synthetic_test_data.py \
        --regulation data/output/규정집_rag.json \
        --output data/synthetic_test_dataset.json \
        --target-size 500

This script demonstrates:
1. Loading regulation documents from JSON
2. Classifying sections (procedural/conditional/factual)
3. Generating diverse questions using Flip-the-RAG approach
4. Extracting ground truth answers
5. Validating test case quality (semantic similarity)
6. Saving the dataset to JSON
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.rag.domain.evaluation.synthetic_data import SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("synthetic_data_generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def print_statistics(stats: dict) -> None:
    """Print generation statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("DATASET GENERATION STATISTICS")
    print("=" * 60)

    print(f"\nTotal Sections Processed: {stats.get('total_sections', 0)}")
    print(f"Valid Test Cases Generated: {stats.get('valid_test_cases', 0)}")
    print(f"Validation Failures: {stats.get('validation_failures', 0)}")

    print("\nSection Type Distribution:")
    distribution = stats.get("section_type_distribution", {})
    for section_type, count in distribution.items():
        percentage = (
            (count / stats.get("valid_test_cases", 1)) * 100
            if stats.get("valid_test_cases", 0) > 0
            else 0
        )
        print(f"  - {section_type}: {count} ({percentage:.1f}%)")

    # Calculate success rate
    if stats.get("total_sections", 0) > 0:
        success_rate = (
            stats.get("valid_test_cases", 0) / stats.get("total_sections", 1)
        ) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")

    print("=" * 60 + "\n")


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Check if regulation file exists
    regulation_path = Path(args.regulation)
    if not regulation_path.exists():
        raise FileNotFoundError(f"Regulation file not found: {args.regulation}")

    # Check if output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate target size
    if args.target_size < 1:
        raise ValueError("Target size must be at least 1")

    # Validate semantic threshold
    if not 0 <= args.semantic_threshold <= 1:
        raise ValueError("Semantic threshold must be between 0 and 1")


async def main() -> None:
    """Main function to generate synthetic test dataset."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for RAG evaluation using Flip-the-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 500 test cases from a regulation document
  python scripts/generate_synthetic_test_data.py \\
      --regulation data/output/규정집_rag.json \\
      --output data/synthetic_test_dataset.json \\
      --target-size 500

  # Generate with custom quality thresholds
  python scripts/generate_synthetic_test_data.py \\
      --regulation data/output/규정집_rag.json \\
      --output data/synthetic_test_dataset.json \\
      --min-question 10 \\
      --max-question 200 \\
      --min-answer 50 \\
      --semantic-threshold 0.5
        """,
    )

    parser.add_argument(
        "--regulation",
        "-r",
        required=True,
        help="Path to regulation JSON file",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSON file for generated dataset",
    )

    parser.add_argument(
        "--target-size",
        "-t",
        type=int,
        default=500,
        help="Target number of test cases to generate (default: 500)",
    )

    parser.add_argument(
        "--min-question",
        type=int,
        default=10,
        help="Minimum question length in characters (default: 10)",
    )

    parser.add_argument(
        "--max-question",
        type=int,
        default=200,
        help="Maximum question length in characters (default: 200)",
    )

    parser.add_argument(
        "--min-answer",
        type=int,
        default=50,
        help="Minimum ground truth length in characters (default: 50)",
    )

    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.5,
        help="Minimum semantic similarity threshold (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_arguments(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Argument validation failed: {e}")
        sys.exit(1)

    # Initialize generator
    logger.info("Initializing Flip-the-RAG Synthetic Data Generator")
    generator = SyntheticDataGenerator(
        min_question_length=args.min_question,
        max_question_length=args.max_question,
        min_answer_length=args.min_answer,
        semantic_threshold=args.semantic_threshold,
    )

    # Generate dataset
    logger.info(f"Loading regulation from: {args.regulation}")
    logger.info(f"Target dataset size: {args.target_size} test cases")

    try:
        test_cases, stats = await generator.generate_dataset(
            regulation_paths=[args.regulation],
            target_size=args.target_size,
            output_path=args.output,
        )

        # Print statistics
        print_statistics(stats)

        # Verify output file
        output_path = Path(args.output)
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024  # KB
            logger.info(
                f"Dataset saved successfully: {args.output} ({file_size:.1f} KB)"
            )

            # Load and verify dataset
            with open(output_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            total_cases = dataset["metadata"]["total_test_cases"]
            logger.info(f"Verified dataset contains {total_cases} test cases")

            # Show sample test case
            if dataset["test_cases"]:
                sample = dataset["test_cases"][0]
                print("\n" + "=" * 60)
                print("SAMPLE TEST CASE")
                print("=" * 60)
                print(f"Question Type: {sample.get('question_type', 'N/A')}")
                print(f"Question: {sample.get('question', 'N/A')}")
                print(f"Ground Truth: {sample.get('ground_truth', 'N/A')[:100]}...")
                print("=" * 60 + "\n")

            logger.info("✓ Dataset generation completed successfully")
            sys.exit(0)
        else:
            logger.error(f"Output file not created: {args.output}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
