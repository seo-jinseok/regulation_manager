#!/usr/bin/env python3
"""
Simple test script for HWPX Direct Parser

Tests the new structure analyzer with actual regulation files.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hwpx_direct_parser import HWPXDirectParser


def test_parser(file_path: Path):
    """Test parser on a single file."""
    print(f"\n{'='*60}")
    print(f"파일: {file_path.name}")
    print(f"{'='*60}")

    try:
        parser = HWPXDirectParser(
            status_callback=lambda msg: print(f"  {msg}")
        )

        result = parser.parse_file(file_path)

        # Print results
        metadata = result.get("metadata", {})
        structure = metadata.get("structure", {})

        print(f"\n[cyan]구조 정보:[/cyan]")
        if structure:
            print(f"  권한: {structure.get('authority', 'N/A')}")
            print(f"  구조: {structure.get('structure_type', 'N/A')}")
            print(f"  파트: {structure.get('has_parts', False)}")
            print(f"  챕터: {structure.get('has_chapters', False)}")
            if structure.get('part_format'):
                print(f"  파트 형식: {structure['part_format']}")
            if structure.get('chapter_format'):
                print(f"  챕터 형식: {structure['chapter_format']}")

        # Print statistics
        stats = metadata.get("parsing_statistics", {})
        if stats:
            print(f"\n[cyan]파싱 통계:[/cyan]")
            print(f"  전체 규정: {stats.get('total_regulations', 0)}")
            print(f"  파싱 성공: {stats.get('successfully_parsed', 0)}")
            print(f"  파싱 실패: {stats.get('failed_regulations', 0)}")
            success_rate = stats.get('success_rate', 0)
            print(f"  성공률: {success_rate:.1f}%")

        # Print TOC entries (first 5)
        toc = result.get("toc", [])
        print(f"\n[cyan]목차 (처음 5개):[/cyan]")
        for i, entry in enumerate(toc[:5]):
            print(f"  {i+1}. {entry.get('title', 'N/A')}")

        # Print document list (first 3)
        docs = result.get("docs", [])
        print(f"\n[cyan]문서 목록 (처음 3개):[/cyan]")
        for i, doc in enumerate(docs[:3]):
            title = doc.get("title", "N/A")
            articles = doc.get("articles", [])
            print(f"  {i+1}. {title}")
            print(f"      조 항목: {len(articles)}개")

        # Check for structure info in metadata
        completeness = metadata.get("completeness", {})
        if completeness:
            print(f"\n[cyan]완전성 검사:[/cyan]")
            print(f"  발견: {completeness.get('found', 0)}개")
            print(f"  누락: {completeness.get('missing', 0)}개")
            completeness_rate = completeness.get("coverage_rate", 0)
            print(f"  적용률: {completeness_rate:.1f}%")

        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"[red]에러: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test files - use actual file names
    test_files = [
        Path("data/input/규정집9-349(20251202).hwpx"),
        Path("data/input/학칙9-350(20251209).hwpx"),
    ]

    success_count = 0

    for file_path in test_files:
        if not file_path.exists():
            print(f"[yellow]파일을 찾을 수 없습니다: {file_path}[/yellow]")
            continue

        if test_parser(file_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"[green]테스트 완료: {success_count}/{len(test_files)} 파일 성공[/green]")
    print(f"{'='*60}")
