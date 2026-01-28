"""
Comprehensive RAG Quality Evaluation Scenarios

This module defines diverse test scenarios covering:
- Multiple user personas (freshman, graduate, professor, staff, parent)
- Various query types (simple, complex, ambiguous, multi-part)
- Different expertise levels (beginner to advanced)
- Multi-turn conversations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PersonaType(Enum):
    """User persona types for testing."""

    FRESHMAN_STUDENT = "freshman_student"
    GRADUATE_STUDENT = "graduate_student"
    PROFESSOR = "professor"
    STAFF_MEMBER = "staff_member"
    PARENT = "parent"
    INTERNATIONAL_STUDENT = "international_student"


class QueryStyle(Enum):
    """Query style variations."""

    PRECISE = "precise"  # Well-formed, specific questions
    AMBIGUOUS = "ambiguous"  # Vague, incomplete questions
    COLLOQUIAL = "colloquial"  # Casual language
    INCORRECT_TERMINOLOGY = "incorrect_terminology"  # Wrong terms
    MULTI_PART = "multi_part"  # Multiple sub-questions
    CONTEXT_DEPENDENT = "context_dependent"  # Relies on context
    TYPO_GRAMMAR_ERROR = "typo_grammar_error"  # Errors in query


class ExpertiseLevel(Enum):
    """User expertise levels."""

    NOVICE = "novice"  # Layman's terms
    INTERMEDIATE = "intermediate"  # Some knowledge
    EXPERT = "expert"  # Technical language


class UrgencyLevel(Enum):
    """Urgency levels for queries."""

    CASUAL = "casual"  # General inquiry
    NORMAL = "normal"  # Standard question
    URGENT = "urgent"  # Time-sensitive request


@dataclass
class TestQuery:
    """A single test query."""

    query: str
    persona: PersonaType
    query_style: QueryStyle
    expertise: ExpertiseLevel
    urgency: UrgencyLevel
    expected_intent: str
    expected_entities: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    context_required: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    query: str
    expected_follow_up_type: str  # "clarification", "deepen", "shift", "refine"
    should_preserve_context: bool


@dataclass
class MultiTurnScenarioDefinition:
    """A multi-turn conversation scenario definition."""

    scenario_id: str
    persona: PersonaType
    turns: List[ConversationTurn]
    expected_resolution: bool
    description: str


# ========== PHASE 1: DIVERSE USER PERSONA TESTS ==========

FRESHMAN_QUERIES = [
    TestQuery(
        query="수강 신청 언제까지야?",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.COLLOQUIAL,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.URGENT,
        expected_intent="registration_deadline",
        expected_entities=["수강신청", "기한"],
    ),
    TestQuery(
        query="졸업하려면 학점 몇 점 필요해?",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="graduation_requirements",
        expected_entities=["졸업", "학점"],
    ),
    TestQuery(
        query="장학금 신청하는 법",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="scholarship_application",
        expected_entities=["장학금"],
        clarification_needed=True,
    ),
    TestQuery(
        query="아 그게 뭐냐면 학생회비 납부하는 거 어디서 해?",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.COLLOQUIAL,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="student_council_fee",
        expected_entities=["학생회비", "납부"],
    ),
    TestQuery(
        query="휴학하고 싶은데 어떻게 해?",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="leave_of_absence",
        expected_entities=["휴학"],
        clarification_needed=True,
    ),
]

GRADUATE_QUERIES = [
    TestQuery(
        query="박사과정 연구장려금 지급 기준과 신청 서류가 궁금합니다.",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="research_grant",
        expected_entities=["연구장려금", "지급기준", "신청서류"],
    ),
    TestQuery(
        query="논문 심사 위원 위촉 절차와 기간을 알고 싶습니다.",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="thesis_committee",
        expected_entities=["논문심사", "위원", "위촉"],
    ),
    TestQuery(
        query="졸업 요건 중 외국어 성적 제출에 관한 규정",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.URGENT,
        expected_intent="graduation_requirements_language",
        expected_entities=["졸업요건", "외국어성적"],
    ),
    TestQuery(
        query="연차 학회 참가비 지원 가능한가요?",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="conference_funding",
        expected_entities=["학회", "참가비", "지원"],
    ),
]

PROFESSOR_QUERIES = [
    TestQuery(
        query="교원 인사 평가 정책 중 연구 성과 평가 기준",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="faculty_evaluation",
        expected_entities=["인사평가", "연구성과", "평가기준"],
    ),
    TestQuery(
        query="학부생 연구원 채용 시 행정 절차",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="undergraduate_researcher",
        expected_entities=["학부생연구원", "채용", "행정절차"],
    ),
    TestQuery(
        query="연구비 집행 시 유의해야 할 규정 사항",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.URGENT,
        expected_intent="research_expenditure",
        expected_entities=["연구비", "집행", "규정"],
    ),
    TestQuery(
        query="강의 계획서 제출 마감일이 언제인가요?",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="syllabus_submission",
        expected_entities=["강의계획서", "제출", "마감일"],
    ),
]

STAFF_QUERIES = [
    TestQuery(
        query="직원 복무 규정 중 연차 사용에 관한 규정",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="annual_leave",
        expected_entities=["복무규정", "연차", "사용"],
    ),
    TestQuery(
        query="구매 입찰 진행 절차와 필요 서류",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="procurement_procedure",
        expected_entities=["구매입찰", "절차", "서류"],
    ),
    TestQuery(
        query="재직 증명서 발급 방법",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="employment_certificate",
        expected_entities=["재직증명서", "발급"],
    ),
    TestQuery(
        query="시설 예약 승인 권한자와 절차",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="facility_approval",
        expected_entities=["시설예약", "승인", "권한자"],
    ),
]

PARENT_QUERIES = [
    TestQuery(
        query="학생 복지 카드 사용 가능한 곳과 할인 혜택",
        persona=PersonaType.PARENT,
        query_style=QueryStyle.COLLOQUIAL,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="student_welfare",
        expected_entities=["복지카드", "할인"],
    ),
    TestQuery(
        query="기숙사 비용과 납부 방법",
        persona=PersonaType.PARENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="dormitory_fee",
        expected_entities=["기숙사", "비용", "납부"],
    ),
    TestQuery(
        query="학생이 휴학하면 등록금 환불되나요?",
        persona=PersonaType.PARENT,
        query_style=QueryStyle.COLLOQUIAL,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="tuition_refund",
        expected_entities=["휴학", "등록금", "환불"],
    ),
    TestQuery(
        query="성적 증명서 발급 방법과 비용",
        persona=PersonaType.PARENT,
        query_style=QueryStyle.PRECISE,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="grade_certificate",
        expected_entities=["성적증명서", "발급", "비용"],
    ),
]

# ========== PHASE 2: AMBIGUOUS AND POORLY-PHRASED QUERIES ==========

AMBIGUOUS_QUERIES = [
    TestQuery(
        query="졸업",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="graduation_requirements",
        clarification_needed=True,
    ),
    TestQuery(
        query="등록",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="registration",
        clarification_needed=True,
    ),
    TestQuery(
        query="장학",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="scholarship",
        clarification_needed=True,
    ),
    TestQuery(
        query="휴가",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.AMBIGUOUS,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="leave",
        clarification_needed=True,
    ),
]

MULTI_PART_QUERIES = [
    TestQuery(
        query="수강 신청 기간과 정정 기간, 그리고 취소 기간을 알려주세요.",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.MULTI_PART,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.URGENT,
        expected_intent="registration_periods",
        expected_entities=["수강신청", "정정", "취소", "기간"],
    ),
    TestQuery(
        query="연구장려금 신청 자격과 절차, 그리고 제출 서류가 무엇인가요?",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.MULTI_PART,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="research_grant_details",
        expected_entities=["연구장려금", "자격", "절차", "서류"],
    ),
    TestQuery(
        query="교수 승진 심사 기준, 서류 제출 마감, 그리고 심사 일정이 궁금합니다.",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.MULTI_PART,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="promotion_evaluation",
        expected_entities=["승진", "심사", "마감", "일정"],
    ),
]

INCORRECT_TERMINOLOGY_QUERIES = [
    TestQuery(
        query="학기 말 시험 일정 알려줘",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.INCORRECT_TERMINOLOGY,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.URGENT,
        expected_intent="final_exam_schedule",
        expected_entities=["시험", "일정"],
    ),
    TestQuery(
        query="교수님들 급여 체계가 어떻게 되나요?",
        persona=PersonaType.STAFF_MEMBER,
        query_style=QueryStyle.INCORRECT_TERMINOLOGY,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="faculty_salary",
        expected_entities=["교수", "급여"],
    ),
    TestQuery(
        query="학교 도서관 대출 연장 방법",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.INCORRECT_TERMINOLOGY,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="library_renewal",
        expected_entities=["도서관", "대출", "연장"],
    ),
]

TYPO_GRAMMAR_QUERIES = [
    TestQuery(
        query="성적 이의 신청하는법 알려줘",
        persona=PersonaType.FRESHMAN_STUDENT,
        query_style=QueryStyle.TYPO_GRAMMAR_ERROR,
        expertise=ExpertiseLevel.NOVICE,
        urgency=UrgencyLevel.URGENT,
        expected_intent="grade_appeal",
        expected_entities=["성적", "이의신청"],
    ),
    TestQuery(
        query="졸업 논문 제출 마감이 언제인가요??",
        persona=PersonaType.GRADUATE_STUDENT,
        query_style=QueryStyle.TYPO_GRAMMAR_ERROR,
        expertise=ExpertiseLevel.INTERMEDIATE,
        urgency=UrgencyLevel.URGENT,
        expected_intent="thesis_deadline",
        expected_entities=["졸업논문", "제출", "마감"],
    ),
    TestQuery(
        query="연구비 집행시 유의사항과 영수증 제출방법",
        persona=PersonaType.PROFESSOR,
        query_style=QueryStyle.TYPO_GRAMMAR_ERROR,
        expertise=ExpertiseLevel.EXPERT,
        urgency=UrgencyLevel.NORMAL,
        expected_intent="research_expenditure_receipt",
        expected_entities=["연구비", "집행", "영수증"],
    ),
]

# ========== PHASE 3: MULTI-TURN CONVERSATION SCENARIOS ==========

MULTI_TURN_SCENARIOS = [
    MultiTurnScenarioDefinition(
        scenario_id="freshman_registration_confusion",
        persona=PersonaType.FRESHMAN_STUDENT,
        description="Freshman confused about registration process",
        turns=[
            ConversationTurn(
                query="수강 신청 언제 해?",
                expected_follow_up_type="clarification",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="아, 그리고 정정 기간은?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="신청한 거 취소도 가능해?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="연장제 수강 신청도 같은 기간인가요?",
                expected_follow_up_type="refine",
                should_preserve_context=True,
            ),
        ],
        expected_resolution=True,
    ),
    MultiTurnScenarioDefinition(
        scenario_id="graduate_research_funding",
        persona=PersonaType.GRADUATE_STUDENT,
        description="Graduate student exploring research funding options",
        turns=[
            ConversationTurn(
                query="연구장려금 신청할 수 있어?",
                expected_follow_up_type="clarification",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="신청 자격이 어떻게 돼?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="어디에 제출하면 돼?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="학회 참가비 지원도 받을 수 있어?",
                expected_follow_up_type="shift",
                should_preserve_context=True,
            ),
        ],
        expected_resolution=True,
    ),
    MultiTurnScenarioDefinition(
        scenario_id="professor_evaluation_process",
        persona=PersonaType.PROFESSOR,
        description="Professor asking about evaluation process",
        turns=[
            ConversationTurn(
                query="연구 성과 평가 기준이 어떻게 돼?",
                expected_follow_up_type="clarification",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="논문 실적은 어떻게 계산해?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="평가 결과 이의 신청은 어떻게 해?",
                expected_follow_up_type="refine",
                should_preserve_context=True,
            ),
        ],
        expected_resolution=True,
    ),
    MultiTurnScenarioDefinition(
        scenario_id="staff_leave_inquiry",
        persona=PersonaType.STAFF_MEMBER,
        description="Staff member inquiring about leave policies",
        turns=[
            ConversationTurn(
                query="연차 사용하고 싶은데",
                expected_follow_up_type="clarification",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="신청은 어떻게 해?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="경조사 휴가도 있어?",
                expected_follow_up_type="shift",
                should_preserve_context=True,
            ),
        ],
        expected_resolution=True,
    ),
    MultiTurnScenarioDefinition(
        scenario_id="parent_tuition_refund",
        persona=PersonaType.PARENT,
        description="Parent asking about tuition refund policy",
        turns=[
            ConversationTurn(
                query="자녀가 휴학하면 등록금 환불되나요?",
                expected_follow_up_type="clarification",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="언제까지 신청하면 돼요?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
            ConversationTurn(
                query="환불금액은 어떻게 계산돼요?",
                expected_follow_up_type="deepen",
                should_preserve_context=True,
            ),
        ],
        expected_resolution=True,
    ),
]

# ========== COMPREHENSIVE TEST COLLECTION ==========

ALL_SINGLE_TURN_QUERIES = (
    FRESHMAN_QUERIES
    + GRADUATE_QUERIES
    + PROFESSOR_QUERIES
    + STAFF_QUERIES
    + PARENT_QUERIES
    + AMBIGUOUS_QUERIES
    + MULTI_PART_QUERIES
    + INCORRECT_TERMINOLOGY_QUERIES
    + TYPO_GRAMMAR_QUERIES
)


def get_all_test_queries() -> List[TestQuery]:
    """Get all single-turn test queries."""
    return ALL_SINGLE_TURN_QUERIES


def get_all_multi_turn_scenarios() -> List[MultiTurnScenarioDefinition]:
    """Get all multi-turn conversation scenarios."""
    return MULTI_TURN_SCENARIOS


def get_queries_by_persona(persona: PersonaType) -> List[TestQuery]:
    """Get all queries for a specific persona."""
    return [q for q in ALL_SINGLE_TURN_QUERIES if q.persona == persona]


def get_queries_by_style(style: QueryStyle) -> List[TestQuery]:
    """Get all queries with a specific style."""
    return [q for q in ALL_SINGLE_TURN_QUERIES if q.query_style == style]


if __name__ == "__main__":
    # Print test collection summary
    print("=" * 60)
    print("RAG Quality Evaluation Test Collection Summary")
    print("=" * 60)
    print(f"Total Single-Turn Queries: {len(ALL_SINGLE_TURN_QUERIES)}")
    print(f"Total Multi-Turn Scenarios: {len(MULTI_TURN_SCENARIOS)}")
    print(
        f"Total Conversation Turns: {sum(len(s.turns) for s in MULTI_TURN_SCENARIOS)}"
    )
    print()

    # By persona
    print("By Persona:")
    for persona in PersonaType:
        count = len(get_queries_by_persona(persona))
        print(f"  {persona.value}: {count} queries")
    print()

    # By query style
    print("By Query Style:")
    for style in QueryStyle:
        count = len(get_queries_by_style(style))
        if count > 0:
            print(f"  {style.value}: {count} queries")
    print()

    print("=" * 60)
