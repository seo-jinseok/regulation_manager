#!/usr/bin/env python3
"""
Split test runner to prevent memory explosion.

Divides tests into smaller groups and runs them sequentially,
allowing memory to be freed between groups.

Usage:
    python scripts/run_tests_split.py              # Run all tests in groups
    python scripts/run_tests_split.py --group 0    # Run specific group
    python scripts/run_tests_split.py --workers 4  # Use xdist with 4 workers per group
"""

import argparse
import gc
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


# Test groups by directory/module
TEST_GROUPS = {
    "0_core": [
        "tests/test_converter.py",
        "tests/test_formatter_extended.py",
        "tests/test_main.py",
    ],
    "1_parsing": [
        "tests/test_converter_extended.py",
        "tests/test_converter_coverage.py",
        "tests/test_parsing_coverage.py",
        "tests/rag/unit/parsing/",
    ],
    "2_rag_core": [
        "tests/rag/unit/domain/",
        "tests/rag/unit/infrastructure/test_chroma_store.py",
        "tests/rag/unit/infrastructure/test_hybrid_search.py",
        "tests/rag/unit/infrastructure/test_reranker.py",
    ],
    "3_rag_search": [
        "tests/rag/unit/application/",
        "tests/rag/unit/infrastructure/test_query_analyzer.py",
        "tests/rag/unit/infrastructure/test_query_expander.py",
    ],
    "4_rag_interface": [
        "tests/rag/unit/interface/",
        "tests/rag/test_web_integration.py",
        "tests/rag/integration/",
    ],
    "5_automation": [
        "tests/rag/unit/automation/",
    ],
    "6_coverage_files": [
        "tests/test_focused_coverage_improvement.py",
        "tests/test_priority_modules_coverage.py",
        "tests/test_coverage_boost.py",
        "tests/test_component_analyzer_coverage.py",
        "tests/test_refine_json_coverage.py",
    ],
    "7_extended_tests": [
        "tests/test_enhance_for_rag.py",
        "tests/test_enhance_for_rag_coverage.py",
        "tests/test_search_usecase_extended2.py",
        "tests/test_formatter_extended2.py",
        "tests/test_formatter_uncovered.py",
    ],
    "8_gradio": [
        "tests/test_gradio_app_extended.py",
        "tests/test_gradio_app_coverage.py",
        "tests/rag/unit/interface/test_gradio_app_comprehensive.py",
    ],
    "9_misc": [
        "tests/test_utils.py",
        "tests/test_analysis.py",
        "tests/test_cache_manager_comprehensive.py",
        "tests/test_cache_manager_edge.py",
    ],
}

# Groups indexed by number for --group argument
GROUP_LIST = list(TEST_GROUPS.keys())


def run_pytest(
    test_paths: list[str],
    workers: int = 1,
    verbose: bool = False,
    extra_args: list[str] = None,
) -> int:
    """Run pytest on given paths with optional xdist workers."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v" if verbose else "-q",
    ]

    if workers > 1:
        cmd.extend(["-n", str(workers)])

    if extra_args:
        cmd.extend(extra_args)

    # Expand paths
    expanded_paths = []
    for path in test_paths:
        full_path = ROOT / path
        if full_path.is_dir():
            expanded_paths.append(str(full_path))
        elif full_path.is_file():
            expanded_paths.append(str(full_path))
        else:
            print(f"Warning: Path not found: {path}")

    cmd.extend(expanded_paths)

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_all_groups(
    workers: int = 1,
    verbose: bool = False,
    stop_on_failure: bool = False,
    extra_args: list[str] = None,
) -> dict[str, int]:
    """Run all test groups sequentially."""
    results = {}

    for i, group_name in enumerate(GROUP_LIST):
        print(f"\n{'#' * 60}")
        print(f"Group {i}/{len(GROUP_LIST) - 1}: {group_name}")
        print(f"{'#' * 60}")

        test_paths = TEST_GROUPS[group_name]
        returncode = run_pytest(test_paths, workers, verbose, extra_args)
        results[group_name] = returncode

        # Force garbage collection between groups
        gc.collect()

        if stop_on_failure and returncode != 0:
            print(f"\n!!! Group {group_name} failed (exit code {returncode})")
            print("Stopping due to --stop-on-failure")
            break

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Split test runner to prevent memory explosion"
    )
    parser.add_argument(
        "--group",
        type=int,
        default=None,
        help="Run specific group by index (0-%d)" % (len(GROUP_LIST) - 1),
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List all test groups",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of xdist workers per group (default: 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on first group failure",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments to pass to pytest",
    )

    args = parser.parse_args()

    if args.list_groups:
        print("Test Groups:")
        for i, name in enumerate(GROUP_LIST):
            paths = TEST_GROUPS[name]
            print(f"  {i}: {name}")
            for path in paths:
                print(f"      - {path}")
        return 0

    if args.group is not None:
        if args.group < 0 or args.group >= len(GROUP_LIST):
            print(f"Error: Group must be between 0 and {len(GROUP_LIST) - 1}")
            return 1

        group_name = GROUP_LIST[args.group]
        test_paths = TEST_GROUPS[group_name]
        returncode = run_pytest(test_paths, args.workers, args.verbose, args.extra_args)
        return returncode

    # Run all groups
    results = run_all_groups(
        args.workers, args.verbose, args.stop_on_failure, args.extra_args
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")

    failed_groups = [name for name, code in results.items() if code != 0]
    passed_groups = [name for name, code in results.items() if code == 0]

    print(f"Passed: {len(passed_groups)}/{len(results)}")
    print(f"Failed: {len(failed_groups)}/{len(results)}")

    if failed_groups:
        print("\nFailed groups:")
        for name in failed_groups:
            print(f"  - {name} (exit code {results[name]})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
