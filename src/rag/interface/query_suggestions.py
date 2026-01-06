"""
쿼리 예시 및 후속 쿼리 제안 모듈.

다양한 기능을 보여주는 예시:
- AI LLM 응답 (Ask 모드)
- 단순 검색 (Search 모드)
- 규정 전문 보기 (Full View)
- 별표/서식 조회
- 의도 기반 쿼리 리라이팅
"""

from typing import List, Optional

# =============================================================================
# 시작 시 보여줄 쿼리 예시 - 다양한 기능 소개
# =============================================================================

INITIAL_QUERY_EXAMPLES = [
    # 1. AI LLM 응답 (Ask 모드) - 자연어 질문
    "휴학 신청 절차가 어떻게 되나요?",
    # 2. 단순 검색 (Search 모드) - 키워드 검색
    "교원 연구년",
    # 3. 규정 전문 보기 (Full View)
    "교원인사규정 전문",
    # 4. 별표/서식 조회
    "학칙 별표 1",
    # 5. 의도 기반 쿼리 (쿼리 리라이팅 시연)
    "학교 그만두고 싶어요",
]

# =============================================================================
# 문맥별 후속 쿼리 패턴
# =============================================================================

FOLLOWUP_PATTERNS = {
    # 학사 관련
    "휴학": ["복학 절차는?", "휴학 기간 연장은 가능한가요?", "휴학 중 등록금은?"],
    "복학": ["복학 신청 기간은?", "복학 후 수강신청은?", "휴학 관련 규정 전문"],
    "자퇴": ["자퇴 후 재입학은?", "등록금 환불은?", "자퇴 신청 서류는?"],
    "졸업": ["졸업 요건은?", "조기졸업 조건은?", "졸업유예 신청은?"],
    "수강": ["수강신청 기간은?", "수강 정정 절차는?", "수강 취소 방법은?"],
    "성적": ["성적 이의신청 기간은?", "F학점 재수강은?", "학점 포기 절차는?"],
    # 등록금 관련
    "등록금": ["분할 납부가 가능한가요?", "장학금 종류는?", "등록금 감면 기준은?"],
    "장학금": ["장학금 신청 방법은?", "성적 장학금 기준은?", "근로장학생 신청은?"],
    "환불": ["환불 신청 기간은?", "환불 비율은?", "등록금 관련 규정 전문"],
    # 교원 관련
    "연구년": ["연구년 기간은?", "해외 연수 지원은?", "연구년 중 급여는?"],
    "임용": ["임용 절차는?", "임용 자격 요건은?", "교원인사규정 전문"],
    "승진": ["승진 요건은?", "승진 심사 절차는?", "업적평가 기준은?"],
    "휴직": ["휴직 종류는?", "휴직 기간은?", "복직 절차는?"],
    # 징계 관련
    "징계": ["징계 종류는?", "출석정지 기간은?", "징계 취소 절차는?"],
    "퇴학": ["퇴학 사유는?", "재입학 가능한가요?", "징계 관련 규정 전문"],
}

# 기본 후속 쿼리 템플릿 (규정명 기반)
DEFAULT_FOLLOWUPS = [
    "{regulation} 전문 보기",
    "{regulation} 별표 보기",
    "{regulation} 관련 다른 규정은?",
    "{regulation} 요약해줘",
    "{regulation} 개정 이력은?",
]


# =============================================================================
# 공개 함수
# =============================================================================


def get_initial_examples() -> List[str]:
    """시작 시 보여줄 쿼리 예시 반환."""
    return INITIAL_QUERY_EXAMPLES.copy()


def get_followup_suggestions(
    query: str,
    regulation_title: Optional[str] = None,
    answer_text: Optional[str] = None,
) -> List[str]:
    """
    문맥 기반 후속 쿼리 3개 제안.

    1. 키워드 매칭으로 관련 후속 쿼리 찾기
    2. 규정 제목 기반 제안
    3. 기본 제안 (전문 보기, 별표 보기 등)

    Args:
        query: 사용자 쿼리
        regulation_title: 마지막으로 조회한 규정 제목
        answer_text: AI 답변 텍스트 (키워드 추출용)

    Returns:
        최대 3개의 후속 쿼리 제안
    """
    suggestions: List[str] = []
    
    # 현재 쿼리 정규화 (공백 정리, 소문자)
    normalized_query = query.lower().strip()

    def is_similar_to_query(suggestion: str) -> bool:
        """현재 쿼리와 동일하거나 유사한지 확인."""
        normalized_suggestion = suggestion.lower().strip()
        # 완전 일치
        if normalized_suggestion == normalized_query:
            return True
        # If suggestion is a substring of query, it's likely redundant (narrowing)
        if normalized_suggestion in normalized_query:
            return True
        return False

    # 1. 키워드 매칭
    search_text = f"{query} {answer_text or ''}"
    for keyword, followups in FOLLOWUP_PATTERNS.items():
        if keyword in search_text:
            # 해당 키워드에서 최대 2개 추가
            for followup in followups[:2]:
                if followup not in suggestions and not is_similar_to_query(followup):
                    suggestions.append(followup)
            if len(suggestions) >= 2:
                break

    # 2. 규정 제목 기반 기본 제안
    if regulation_title and len(suggestions) < 3:
        for template in DEFAULT_FOLLOWUPS:
            if len(suggestions) >= 3:
                break
            suggestion = template.format(regulation=regulation_title)
            if suggestion not in suggestions and not is_similar_to_query(suggestion):
                suggestions.append(suggestion)

    return suggestions[:3]


def format_examples_for_cli(examples: List[str]) -> str:
    """CLI용 예시 쿼리 포맷팅."""
    lines = []
    for i, example in enumerate(examples, 1):
        lines.append(f"  [{i}] {example}")
    return "\n".join(lines)


def format_suggestions_for_cli(suggestions: List[str]) -> str:
    """CLI용 후속 쿼리 제안 포맷팅."""
    if not suggestions:
        return ""
    lines = ["\n💡 연관 질문:"]
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"  [{i}] {suggestion}")
    return "\n".join(lines)
