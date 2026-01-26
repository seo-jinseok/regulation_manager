#!/usr/bin/env python3
"""
팩트체크 자동화 도구

LLM 답변에서 규정/조항 인용을 추출하고, 실제 데이터베이스에서 검증합니다.

사용법:
    # 답변 텍스트를 파이프로 전달
    echo "「휴학규정」 제7조에 따르면..." | uv run python scripts/factcheck.py
    
    # 파일에서 읽기
    uv run python scripts/factcheck.py --file answer.txt
    
    # 직접 텍스트 입력
    uv run python scripts/factcheck.py --text "「교원인사규정」 제36조에 따르면 휴직 기간은..."
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.infrastructure.chroma_store import ChromaVectorStore


def extract_citations(text: str) -> list[dict]:
    """
    답변 텍스트에서 규정/조항 인용을 추출합니다.
    
    추출 패턴:
    - 「규정명」 제N조
    - 「규정명」 제N조 제M항
    - 「규정명」 제N조의N
    """
    patterns = [
        # 「규정명」 제N조 제M항 N호
        r"「([^」]+)」\s*제?(\d+)조(?:의(\d+))?\s*(?:제?(\d+)항)?(?:\s*(\d+)호)?",
        # 규정명 제N조 (따옴표 없이)
        r"([가-힣]+(?:규정|규칙|세칙|지침|학칙))\s*제?(\d+)조(?:의(\d+))?\s*(?:제?(\d+)항)?",
    ]

    citations = []
    seen = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            reg_name = groups[0]
            article = groups[1]
            article_sub = groups[2] if len(groups) > 2 else None
            paragraph = groups[3] if len(groups) > 3 else None

            # 중복 제거용 키
            key = (reg_name, article, article_sub)
            if key in seen:
                continue
            seen.add(key)

            citation = {
                "regulation": reg_name,
                "article": article,
                "article_sub": article_sub,
                "paragraph": paragraph,
                "original": match.group(0),
            }
            citations.append(citation)

    return citations


def verify_citation(store: ChromaVectorStore, citation: dict) -> dict:
    """
    인용이 실제 데이터베이스에 존재하는지 검증합니다.
    """
    reg_name = citation["regulation"]
    article = citation["article"]
    article_sub = citation.get("article_sub")

    # 검색 쿼리 생성
    if article_sub:
        query = f"{reg_name} 제{article}조의{article_sub}"
    else:
        query = f"{reg_name} 제{article}조"

    # 벡터 검색 수행
    from rag.domain.value_objects import Query
    results = store.search(Query(text=query), top_k=10)

    # 결과 분석
    found = False
    matched_chunks = []

    # 규정명 정규화 (띄어쓰기 등 제거)
    reg_name_normalized = reg_name.replace(" ", "").replace("·", "")
    article_pattern = f"제{article}조"

    for result in results:
        chunk = result.chunk
        # 규정명 매칭 확인 (parent_path, title, text에서 확인)
        chunk_text = f"{' '.join(chunk.parent_path)} {chunk.title} {chunk.text}"
        chunk_text_normalized = chunk_text.replace(" ", "").replace("·", "")

        # 규정명이 포함되어 있는지 확인
        reg_match = reg_name_normalized in chunk_text_normalized or reg_name in chunk_text

        # 조항 번호 매칭 확인
        article_match = article_pattern in chunk_text

        if reg_match and article_match:
            found = True
            chunk_reg = chunk.parent_path[0] if chunk.parent_path else "N/A"
            matched_chunks.append({
                "regulation": chunk_reg,
                "article": chunk.title,
                "content_preview": chunk.text[:200] if chunk.text else "",
                "score": result.score,
            })

    return {
        **citation,
        "verified": found,
        "matched_chunks": matched_chunks[:3],  # 최대 3개
    }


def format_results(results: list[dict], verbose: bool = False) -> str:
    """결과를 보기 좋게 포맷팅합니다."""
    output = []
    output.append("=" * 60)
    output.append("📋 팩트체크 결과")
    output.append("=" * 60)

    verified_count = sum(1 for r in results if r["verified"])
    total_count = len(results)

    output.append(f"\n✅ 검증 완료: {verified_count}/{total_count} ({verified_count/total_count*100:.0f}%)\n")

    for i, result in enumerate(results, 1):
        status = "✅" if result["verified"] else "❌"
        output.append(f"{i}. {status} {result['original']}")

        if result["verified"]:
            if verbose and result["matched_chunks"]:
                chunk = result["matched_chunks"][0]
                output.append(f"   → 확인됨: {chunk['regulation']} (점수: {chunk['score']:.2f})")
                if chunk["content_preview"]:
                    preview = chunk["content_preview"][:100].replace("\n", " ")
                    output.append(f"   → 내용: {preview}...")
        else:
            output.append("   → ⚠️ 해당 규정/조항을 찾을 수 없음")

        output.append("")

    output.append("=" * 60)

    if verified_count < total_count:
        output.append("⚠️ 일부 인용이 검증되지 않았습니다. 할루시네이션 가능성이 있습니다.")
    else:
        output.append("✅ 모든 인용이 검증되었습니다.")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="LLM 답변의 규정 인용을 팩트체크합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", "-t", help="검증할 텍스트")
    parser.add_argument("--file", "-f", help="검증할 텍스트가 담긴 파일")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    parser.add_argument("--json", "-j", action="store_true", help="JSON 형식으로 출력")
    parser.add_argument("--db-path", default="data/chroma_db", help="ChromaDB 경로")

    args = parser.parse_args()

    # 텍스트 입력 받기
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        print("\n❌ 오류: 검증할 텍스트를 입력해주세요.")
        sys.exit(1)

    # 인용 추출
    citations = extract_citations(text)

    if not citations:
        print("ℹ️ 텍스트에서 규정 인용을 찾을 수 없습니다.")
        sys.exit(0)

    print(f"ℹ️ {len(citations)}개의 인용을 발견했습니다. 검증 중...")

    # 벡터 저장소 로드
    store = ChromaVectorStore(persist_directory=args.db_path)

    # 각 인용 검증
    results = []
    for citation in citations:
        result = verify_citation(store, citation)
        results.append(result)

    # 결과 출력
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_results(results, verbose=args.verbose))

    # 검증 실패가 있으면 exit code 1
    if not all(r["verified"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
