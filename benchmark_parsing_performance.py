#!/usr/bin/env python3
"""
Performance Benchmark Script for HWPX Parser.

This script benchmarks the parsing performance and generates a detailed report.

Usage:
    python benchmark_parsing_performance.py [--hwpx-file PATH] [--output PATH]

Reference: SPEC-HWXP-002, TASK-008
"""
import argparse
import json
import time
import tracemalloc
import zipfile
from pathlib import Path
from typing import Dict, Any

from src.parsing.multi_format_parser import HWPXMultiFormatParser
from src.parsing.optimized_multi_format_parser import OptimizedHWPXMultiFormatParser


def create_benchmark_file(output_path: Path, num_regulations: int = 514) -> Path:
    """Create a benchmark HWPX file with specified number of regulations."""
    hwpx_path = output_path / f"benchmark_{num_regulations}.hwpx"

    # Generate TOC
    toc_content = []
    for i in range(1, num_regulations + 1):
        toc_content.append(f"제{i}번규정 ...................... {i}")

    # Generate main content with mixed format types
    format_patterns = {
        "article": "제1조(목적) 이 규정은 {title}에 관한 사항을 규정함을 목적으로 한다.\n제2조(정의) 이 규정에서 용어를 정의한다.\n제3조(시행) 이 규정은 2024년 1월 1일부터 시행한다.",
        "list": "1. {title}에 관한 사항을 규정한다.\n2. 세부 시행 방법은 총장이 정한다.\n3. 이 규정은 2024년 1월 1일부터 시행한다.\n① 세부 사항은 따로 정한다.\② 필요한 경우 개정할 수 있다.",
        "guideline": "이 지침은 {title}에 관한 기본 사항을 정함을 목적으로 한다. 따라서 모든 부서가 준수해야 한다. 또한 정기적인 검토가 필요하다. 각 부서장은 이 지침을 숙지해야 한다.",
    }

    main_content = []
    for i in range(1, num_regulations + 1):
        title = f"제{i}번규정"
        format_type = list(format_patterns.keys())[i % 3]
        content = format_patterns[format_type].format(title=title)
        main_content.append(f"{title}\n\n{content}\n")

    with zipfile.ZipFile(hwpx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Contents/section1.xml", '\n'.join(toc_content))
        zf.writestr("Contents/section0.xml", '\n'.join(main_content))
        zf.writestr("Contents/_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')

    return hwpx_path


def benchmark_parser(parser, parser_name: str, hwpx_file: Path) -> Dict[str, Any]:
    """Benchmark a single parser."""
    print(f"\nBenchmarking {parser_name}...")

    tracemalloc.start()
    start_time = time.time()

    result = parser.parse_file(hwpx_file)

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    docs = result.get('docs', result.get('regulations', []))

    return {
        "parser": parser_name,
        "time_seconds": elapsed_time,
        "memory_current_mb": current / (1024 * 1024),
        "memory_peak_mb": peak / (1024 * 1024),
        "regulations_parsed": len(docs),
        "throughput_regs_per_sec": len(docs) / elapsed_time if elapsed_time > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark HWPX parser performance")
    parser.add_argument("--hwpx-file", type=Path, help="Path to existing HWPX file")
    parser.add_argument("--num-regulations", type=int, default=514, help="Number of regulations for benchmark")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"), help="Output file for results")
    parser.add_argument("--sequential-only", action="store_true", help="Only benchmark sequential parser")

    args = parser.parse_args()

    # Create or use existing HWPX file
    if args.hwpx_file and args.hwpx_file.exists():
        hwpx_file = args.hwpx_file
    else:
        print(f"Creating benchmark file with {args.num_regulations} regulations...")
        output_dir = args.output.parent if args.output else Path(".")
        hwpx_file = create_benchmark_file(output_dir, args.num_regulations)

    print(f"Benchmarking with: {hwpx_file}")
    print(f"Target: <60 seconds for {args.num_regulations} regulations")

    # Mock LLM client
    from unittest.mock import Mock
    mock_llm = Mock()
    mock_llm.generate.return_value = json.dumps({
        "structure_type": "unstructured",
        "confidence": 0.8,
        "provisions": [{"number": "1", "content": "Provision 1"}]
    })

    results = {
        "benchmark_file": str(hwpx_file),
        "target_time_seconds": 60.0,
        "target_memory_mb": 2048,
        "num_regulations": args.num_regulations
    }

    # Benchmark sequential parser
    print("\n" + "="*60)
    print("SEQUENTIAL PARSER (Baseline)")
    print("="*60)
    sequential_parser = HWPXMultiFormatParser(llm_client=mock_llm)
    sequential_result = benchmark_parser(sequential_parser, "Sequential", hwpx_file)
    results["sequential"] = sequential_result

    # Benchmark parallel parser
    if not args.sequential_only:
        print("\n" + "="*60)
        print("PARALLEL PARSER (Optimized)")
        print("="*60)
        parallel_parser = OptimizedHWPXMultiFormatParser(llm_client=mock_llm)
        parallel_result = benchmark_parser(parallel_parser, "Parallel", hwpx_file)
        results["parallel"] = parallel_result

        # Calculate speedup
        speedup = sequential_result["time_seconds"] / parallel_result["time_seconds"]
        results["speedup"] = speedup

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {sequential_result['time_seconds'] - parallel_result['time_seconds']:.2f}s")
        print(f"Memory overhead: {parallel_result['memory_peak_mb'] / sequential_result['memory_peak_mb']:.2f}x")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential: {sequential_result['time_seconds']:.2f}s ({sequential_result['throughput_regs_per_sec']:.1f} regs/sec)")
    if not args.sequential_only:
        print(f"Parallel: {parallel_result['time_seconds']:.2f}s ({parallel_result['throughput_regs_per_sec']:.1f} regs/sec)")
    print(f"\nTarget Met: {'YES' if sequential_result['time_seconds'] < 60 else 'NO'}")
    print(f"Memory Target Met: {'YES' if sequential_result['memory_peak_mb'] < 2048 else 'NO'}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # Return exit code based on target meeting
    return 0 if sequential_result['time_seconds'] < 60 else 1


if __name__ == "__main__":
    exit(main())
