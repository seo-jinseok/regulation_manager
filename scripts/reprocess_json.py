#!/usr/bin/env python3
"""
기존 규정집 JSON을 재처리하여 embedding_text를 업데이트합니다.

Usage:
    uv run python scripts/reprocess_json.py data/output/규정집.json
    uv run python scripts/reprocess_json.py data/output/규정집.json --backup
    uv run python scripts/reprocess_json.py data/output/규정집.json -o data/output/규정집_v2.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enhance_for_rag import enhance_json


def main():
    parser = argparse.ArgumentParser(
        description="규정집 JSON 재처리 - embedding_text 업데이트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 백업 생성 후 원본 덮어쓰기
    uv run python scripts/reprocess_json.py data/output/규정집.json --backup

    # 새 파일로 저장
    uv run python scripts/reprocess_json.py data/output/규정집.json -o data/output/규정집_v2.json
        """,
    )
    parser.add_argument("input_file", type=Path, help="입력 JSON 파일")
    parser.add_argument(
        "--output", "-o", type=Path, help="출력 파일 (기본: 입력 파일 덮어쓰기)"
    )
    parser.add_argument(
        "--backup", action="store_true", help="원본 백업 생성 (.json.backup)"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="샘플 N개 노드의 embedding_text 출력"
    )
    args = parser.parse_args()

    # 입력 파일 존재 확인
    if not args.input_file.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {args.input_file}")
        return 1

    # 백업 생성
    if args.backup:
        backup_path = args.input_file.with_suffix(".json.backup")
        backup_path.write_text(
            args.input_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"[INFO] 백업 생성: {backup_path}")

    # JSON 로드
    print(f"[INFO] 로드 중: {args.input_file}")
    try:
        data = json.loads(args.input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")
        return 1

    # 기존 통계 수집
    docs_count = len(data.get("docs", []))
    print(f"[INFO] 문서 수: {docs_count}")

    # 재처리
    print("[INFO] 재처리 중...")
    enhanced_data = enhance_json(data)

    # 샘플 출력
    if args.sample > 0:
        print(f"\n[SAMPLE] embedding_text 예시 ({args.sample}개):")
        print("-" * 60)
        sample_count = 0
        for doc in enhanced_data.get("docs", []):
            if sample_count >= args.sample:
                break
            for node in _iterate_nodes(doc):
                if sample_count >= args.sample:
                    break
                if node.get("embedding_text"):
                    print(f"- {node['embedding_text'][:100]}...")
                    sample_count += 1
        print("-" * 60)

    # 저장
    output_path = args.output or args.input_file
    print(f"[INFO] 저장 중: {output_path}")
    output_path.write_text(
        json.dumps(enhanced_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[SUCCESS] 완료! {docs_count}개 문서 재처리됨")
    return 0


def _iterate_nodes(node, depth=0):
    """노드와 모든 하위 노드를 순회합니다."""
    yield node
    for child in node.get("children", []):
        yield from _iterate_nodes(child, depth + 1)
    for addendum in node.get("addenda", []):
        yield from _iterate_nodes(addendum, depth + 1)


if __name__ == "__main__":
    sys.exit(main())
