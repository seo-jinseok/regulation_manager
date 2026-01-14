#!/usr/bin/env python
"""쿼리 확장 테스트 스크립트.

동의어 사전과 의도 인식 규칙이 올바르게 작동하는지 확인합니다.
"""

from src.rag.infrastructure.query_analyzer import QueryAnalyzer


def main():
    """테스트 쿼리들에 대해 확장 및 재작성 결과를 출력합니다."""
    analyzer = QueryAnalyzer(
        synonyms_path="data/config/synonyms.json",
        intents_path="data/config/intents.json",
    )

    test_queries = [
        # 간접 표현 테스트
        "공부하기 싫어",
        "그만두고 싶어",
        "아파서 병가 쓰고 싶어",
        # 복합 의도 테스트
        "교수님이 화내고 정치 발언",
        "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래",
        # 시설/창업 테스트
        "강의실 예약하고 싶어",
        "학생 창업 지원받을 수 있어?",
        # 병가/휴직 테스트
        "육아휴직 신청하려면?",
        "징계 절차가 어떻게 돼?",
        # 장학금 테스트
        "장학금 받고 싶어",
        # 강의 면제 테스트
        "강의 면제 받으려면?",
        # 교수 과제 불만
        "교수가 과제 기한 너무 짧게 줬어",
    ]

    print("=" * 80)
    print("쿼리 확장 테스트 결과")
    print("=" * 80)

    for query in test_queries:
        expanded = analyzer.expand_query(query)
        rewrite_info = analyzer.rewrite_query_with_info(query)

        print(f"\n📝 원본: {query}")
        print(f"🔄 확장: {expanded}")
        print(f"✏️  재작성: {rewrite_info.rewritten}")

        if rewrite_info.matched_intents:
            # matched_intents can be a list of strings or IntentMatch objects
            if isinstance(rewrite_info.matched_intents[0], str):
                intents_str = ", ".join(rewrite_info.matched_intents)
            else:
                intents_str = ", ".join(
                    [f"{m.intent}({m.confidence:.2f})" for m in rewrite_info.matched_intents]
                )
            print(f"🎯 매칭 인텐트: {intents_str}")
        else:
            print("🎯 매칭 인텐트: 없음")

        print("-" * 40)

    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    main()
