#!/usr/bin/env python
"""실패 쿼리 분석 스크립트."""

import json
from pathlib import Path

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore


def main():
    store = ChromaVectorStore()
    # 평가와 동일한 조건: use_reranker=True
    usecase = SearchUseCase(store, use_reranker=True)

    # 평가 데이터셋 로드
    dataset_path = Path("data/config/evaluation_dataset.json")
    dataset = json.loads(dataset_path.read_text())
    
    # 실패 케이스 ID들 (최신 평가 결과 기준)
    failed_ids = [
        "scholarship_01",
        "parental_leave_01",
        "sick_leave_01",
        "disciplinary_01",
        "facility_use_01",
        "startup_support_01",
        "student_complaint_faculty_01",
    ]
    
    failed_cases = {tc["id"]: tc for tc in dataset["test_cases"] if tc["id"] in failed_ids}

    print("=" * 70)
    print("실패 케이스 상세 분석 (평가 조건: use_reranker=True, top_k=5)")
    print("=" * 70)

    for case_id in failed_ids:
        tc = failed_cases.get(case_id)
        if not tc:
            print(f"\n[WARN] {case_id} not found in dataset")
            continue
            
        q = tc["query"]
        min_score = tc["min_relevance_score"]
        expected_codes = tc.get("expected_rule_codes", [])
        expected_keywords = tc.get("expected_keywords", [])
        
        print(f"\n{'='*70}")
        print(f"[{case_id}] {q}")
        print(f"기대 규정: {expected_codes}, min_score: {min_score}")
        print(f"기대 키워드: {expected_keywords}")
        
        # SearchUseCase로 검색 (평가와 동일한 방식)
        results = usecase.search(q, top_k=5)
        
        # 쿼리 재작성 정보
        rewrite_info = usecase.get_last_query_rewrite()
        if rewrite_info:
            print(f"재작성 쿼리: {rewrite_info.rewritten}")
            print(f"매칭 인텐트: {rewrite_info.matched_intents}")
        
        top_score = results[0].score if results else 0.0
        found_codes = [r.chunk.rule_code for r in results[:5]]
        
        # 키워드 매칭 체크
        rewritten = rewrite_info.rewritten if rewrite_info else q
        found_keywords = [kw for kw in expected_keywords if kw.lower() in rewritten.lower()]
        keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
        
        print(f"\n[결과]")
        print(f"  top_score: {top_score:.3f} (기대: >= {min_score}) -> {'✓' if top_score >= min_score else '✗ FAIL'}")
        print(f"  키워드 커버리지: {keyword_coverage:.0%} ({found_keywords}) -> {'✓' if keyword_coverage >= 0.5 else '✗ FAIL'}")
        
        code_ok = not expected_codes or bool(set(expected_codes) & set(found_codes))
        print(f"  규정 코드: {found_codes} -> {'✓' if code_ok else '✗ FAIL'}")
        
        print(f"\n[검색 결과]")
        for i, r in enumerate(results[:5]):
            chunk = r.chunk
            reg = getattr(chunk, 'title', '?') or '?'
            rule = getattr(chunk, 'rule_code', '?') or '?'
            article = getattr(chunk, 'article_number', '?') or '?'
            print(f"  [{i+1}] {rule} {reg} 제{article}조 (score={r.score:.3f})")


if __name__ == "__main__":
    main()
