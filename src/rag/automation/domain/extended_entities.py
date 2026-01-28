"""
Extended Domain Entities for RAG Testing Automation System.

This module extends the base entities with additional personas, test scenarios,
and edge case definitions for comprehensive RAG system testing.

Clean Architecture: Domain layer uses only standard library (dataclasses).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .entities import (
    DifficultyLevel,
    Persona,
    PersonaType,
)


class ExtendedPersonaType(Enum):
    """Extended persona types for comprehensive RAG testing."""

    INTERNATIONAL_STUDENT = "international_student"  # 유학생: 언어/문화 장벽
    ADJUNCT_PROFESSOR = "adjunct_professor"  # 시간강사: 비정규직
    RESEARCHER = "researcher"  # 연구원: 연구 중심
    APPLICANT = "applicant"  # 입학 지원자: 잠재 학생
    COMMUNITY_MEMBER = "community_member"  # 지역사회 구성원: 외부 관심사
    TRANSFER_STUDENT = "transfer_student"  # 편입생: 다른 대학에서 옴
    RETURNING_STUDENT = "returning_student"  # 복학생: 학업 중단 후 재개
    RETIREE_STUDENT = "retiree_student"  # 재학생 노약자: 평생교육
    ONLINE_STUDENT = "online_student"  # 사이버 강의생: 원격 수업
    DISABLED_STUDENT = "disabled_student"  # 장애학생: 특수 지원 필요


class EdgeCaseCategory(Enum):
    """Categories of edge cases for RAG testing."""

    EMOTIONAL = "emotional"  # 감정적 상태: 좌절, 긴급, 혼란
    COMPLEX_SYNTHESIS = "complex_synthesis"  # 복잡한 종합: 여러 규정 통합 필요
    CROSS_REFERENCED = "cross_referenced"  # 상호 참조: 여러 문서 간 참조
    DEADLINE_CRITICAL = "deadline_critical"  # 기한 임박: 시간 압박
    CONTRADICTORY = "contradictory"  # 모순적: 규정 간 충돌
    LANGUAGE_BARRIER = "language_barrier"  # 언어 장벽: 이해 어려움
    EXCEPTIONAL = "exceptional"  # 예외적: 특수 상황
    TECHNICAL = "technical"  # 기술적: 복잡한 절차


class AmbiguityType(Enum):
    """Types of ambiguous queries."""

    MISSING_CONTEXT = "missing_context"  # 맥락 부족: 이해를 위한 정보 부족
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"  # 다중 해석: 여러 가능성
    UNCLEAR_INTENT = "unclear_intent"  # 불분명 의도: 사용자 목표 불명확
    VAGUE_TERMINOLOGY = "vague_terminology"  # 모호 용어: 애매한 단어 사용
    INCOMPLETE_THOUGHT = "incomplete_thought"  # 불완전 생각: 문장 중단


@dataclass
class ExtendedPersona(Persona):
    """
    Extended persona with additional attributes for comprehensive testing.

    Includes language proficiency, cultural context, technical expertise,
    and urgency levels.
    """

    language_proficiency: str = "native"  # native, fluent, intermediate, basic
    cultural_context: List[str] = field(default_factory=list)
    technical_expertise: str = "basic"  # basic, intermediate, advanced
    urgency_level: str = "normal"  # low, normal, high, urgent
    accessibility_needs: List[str] = field(default_factory=list)


@dataclass
class AmbiguousQuery:
    """
    An ambiguous query for testing RAG system's intent recognition.

    Contains the query, ambiguity type, expected clarifications,
    and evaluation criteria.
    """

    query_id: str  # e.g., "amb-001"
    query: str  # The ambiguous query text
    ambiguity_type: AmbiguityType  # Type of ambiguity
    difficulty: DifficultyLevel  # Difficulty level
    persona_type: Optional[PersonaType] = None  # Associated persona
    context_hints: List[str] = field(default_factory=list)  # Hints for resolution

    # Expected resolution
    expected_interpretations: List[str] = field(default_factory=list)
    expected_clarifications: List[str] = field(default_factory=list)

    # Evaluation criteria
    should_detect_ambiguity: bool = True
    should_request_clarification: bool = True
    acceptable_answers: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTurnConversationScenario:
    """
    A multi-turn conversation scenario for testing context management.

    Tests the RAG system's ability to maintain context across multiple
    interactions and handle various types of follow-up questions.
    """

    scenario_id: str  # e.g., "mt-001"
    name: str  # Scenario name
    description: str  # Scenario description

    persona_type: PersonaType  # User persona
    difficulty: DifficultyLevel  # Overall difficulty
    initial_query: str
    initial_expected_intent: str

    # Context window (must come after non-default fields)
    context_window_size: int = 3  # Context window for testing

    # Follow-up turns
    turns: List[Dict[str, Any]] = field(default_factory=list)

    # Expected outcomes
    expected_context_preservation_rate: float = 0.8
    expected_topic_transitions: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeCaseScenario:
    """
    An edge case scenario for testing RAG system robustness.

    Tests unusual, extreme, or boundary conditions that may not be
    covered by normal test scenarios.
    """

    scenario_id: str  # e.g., "edge-001"
    name: str  # Scenario name
    category: EdgeCaseCategory  # Type of edge case
    difficulty: DifficultyLevel  # Difficulty level

    persona_type: PersonaType  # User persona
    query: str  # The edge case query

    # Edge case characteristics
    is_emotional: bool = False
    is_urgent: bool = False
    is_confused: bool = False
    is_frustrated: bool = False

    # Expected system behavior
    expected_empathy_level: str = "neutral"  # neutral, empathetic, highly_empathetic
    expected_response_speed: str = "normal"  # immediate, fast, normal
    should_escalate: bool = False  # Should escalate to human

    # Expected content
    expected_regulations: List[str] = field(default_factory=list)
    expected_actions: List[str] = field(default_factory=list)

    # Evaluation criteria
    should_show_empathy: bool = False
    should_provide_contact: bool = False
    should_offer_alternatives: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalState:
    """
    Emotional state of a user for testing emotional queries.

    Captures the emotional context that should influence the
    RAG system's response style and content.
    """

    primary_emotion: str  # frustrated, confused, anxious, angry, desperate
    intensity: float  # 0.0 to 1.0
    expected_system_response: str  # How system should respond

    # Triggers and validation (must come after non-default fields)
    triggers: List[str] = field(default_factory=list)  # What triggered this state
    should_detect_emotion: bool = True
    should_adjust_tone: bool = True
    should_provide_support: bool = False


@dataclass
class DeadlineScenario:
    """
    Time-sensitive scenario for testing deadline-critical queries.

    Tests the RAG system's ability to handle urgent, time-bound
    requests with appropriate priority and clarity.
    """

    scenario_id: str  # e.g., "deadline-001"
    urgency_level: str  # low, medium, high, critical

    time_remaining: str  # e.g., "2 days", "4 hours", "1 week"
    deadline_type: str  # application, submission, payment, registration

    persona_type: PersonaType  # User persona
    query: str  # The urgent query

    # Expected system behavior
    should_highlight_deadline: bool = True
    should_expedite_info: bool = True
    should_provide_contact: bool = True

    # Required information
    required_deadline_date: bool = True
    required_submission_method: bool = True
    required_late_penalty_info: bool = True

    # Evaluation criteria
    clarity_score_threshold: float = 0.8
    completeness_score_threshold: float = 0.9


@dataclass
class CrossReferenceScenario:
    """
    Scenario testing multiple regulation references.

    Tests the RAG system's ability to synthesize information from
    multiple related regulations and provide coherent answers.
    """

    scenario_id: str  # e.g., "xref-001"
    persona_type: PersonaType  # User persona
    query: str  # The complex query

    # Required regulations
    primary_regulation: str  # Main regulation
    related_regulations: List[str] = field(default_factory=list)  # Related regulations
    conflicting_regulations: List[str] = field(default_factory=list)  # Any conflicts

    # Expected synthesis
    expected_synthesis_points: List[str] = field(default_factory=list)
    expected_hierarchies: List[str] = field(default_factory=list)  # Hierarchy info

    # Evaluation criteria
    should_cite_all_sources: bool = True
    should_resolve_conflicts: bool = True
    should_explain_relationships: bool = True


# Extended persona definitions
EXTENDED_PERSONA_DEFINITIONS = {
    ExtendedPersonaType.INTERNATIONAL_STUDENT: ExtendedPersona(
        persona_type=PersonaType.FRESHMAN,  # Base type
        name="유학생",
        description="언어 및 문화 장벽이 있음, 공식 용어 어려움",
        characteristics=[
            "언어 장벽 있음",
            "문화적 차이",
            "공식 용어 어려움",
            "비자 문제 관심",
        ],
        query_styles=[
            "번역투 표현",
            "문법적 오류 포함",
            "단순화된 질문",
            "반복 질문",
        ],
        context_hints=["비자", "language barrier", "cultural adaptation"],
        language_proficiency="intermediate",
        cultural_context=["international", "student_visa"],
        technical_expertise="basic",
        urgency_level="normal",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.ADJUNCT_PROFESSOR: ExtendedPersona(
        persona_type=PersonaType.NEW_PROFESSOR,
        name="시간강사",
        description="비정규직, 고용 불안, 혜택 제한",
        characteristics=[
            "비정규직 교원",
            "고용 불안",
            "혜택 제한",
            "시간 계약",
        ],
        query_styles=[
            "고용 조건 확인",
            "혜택 차이 문의",
            "계약 관련 질문",
            "승진 기준 문의",
        ],
        context_hints=["part-time", "contract_basics", "benefits_limits"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="advanced",
        urgency_level="high",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.RESEARCHER: ExtendedPersona(
        persona_type=PersonaType.GRADUATE,
        name="연구원",
        description="연구 중심, 연구비, 윤리 규정",
        characteristics=[
            "연구 중심",
            "연구비 관심",
            "연구 윤리",
            "발표 관심",
        ],
        query_styles=[
            "연구비 지원 문의",
            "윤리 규정 확인",
            "발표 규정 문의",
            "저작권 문의",
        ],
        context_hints=["research_funding", "ethics", "publication"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="advanced",
        urgency_level="normal",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.APPLICANT: ExtendedPersona(
        persona_type=PersonaType.FRESHMAN,
        name="입학 지원자",
        description="잠재 학생, 입학 절차, 모집 요강",
        characteristics=[
            "입학 절차 관심",
            "모집 요강 필요",
            "지원 자격 확인",
            "전형 방법 문의",
        ],
        query_styles=[
            "입학 자격 문의",
            "전형 방법 확인",
            "서류 제출 문의",
            "면접 관련 질문",
        ],
        context_hints=["admission", "application_process", "requirements"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="basic",
        urgency_level="high",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.COMMUNITY_MEMBER: ExtendedPersona(
        persona_type=PersonaType.PARENT,
        name="지역사회 구성원",
        description="외부 관심사, 공공 시설, 개방 프로그램",
        characteristics=[
            "외부 구성원",
            "공공 시설 이용",
            "개방 프로그램 관심",
            "지역 사회 참여",
        ],
        query_styles=[
            "시설 이용 문의",
            "프로그램 참여 질문",
            "개방 여부 확인",
            "이용 조건 문의",
        ],
        context_hints=["community_access", "public_facilities", "outreach"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="basic",
        urgency_level="low",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.TRANSFER_STUDENT: ExtendedPersona(
        persona_type=PersonaType.JUNIOR,
        name="편입생",
        description="다른 대학에서 옴, 학점 인정, 적응 문제",
        characteristics=[
            "편입 학생",
            "학점 인정 관심",
            "적응 문제",
            "전공 심화 필요",
        ],
        query_styles=[
            "학점 인정 문의",
            "편입 요건 확인",
            "전공 인정 질문",
            "적응 프로그램 문의",
        ],
        context_hints=["transfer", "credit_transfer", "adaptation"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="intermediate",
        urgency_level="normal",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.RETURNING_STUDENT: ExtendedPersona(
        persona_type=PersonaType.JUNIOR,
        name="복학생",
        description="휴학 후 복학, 학업 중단 후 재개",
        characteristics=[
            "복학생",
            "학업 중단 경험",
            "재적격 관심",
            "복학 절차 필요",
        ],
        query_styles=[
            "복학 절차 문의",
            "재적격 확인",
            "학점 유효 기간",
            "복학 후 적응",
        ],
        context_hints=["returning", "readmission", "academic_restart"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="intermediate",
        urgency_level="normal",
        accessibility_needs=[],
    ),
    ExtendedPersonaType.RETIREE_STUDENT: ExtendedPersona(
        persona_type=PersonaType.JUNIOR,
        name="재학생 노약자",
        description="평생교육, 연령 차이, 기술적 어려움",
        characteristics=[
            "평생교육 학습자",
            "연령 차이",
            "기술적 어려움",
            "건강 관심",
        ],
        query_styles=[
            "평생교육 문의",
            "수강료 할인",
            "건강 관련 혜택",
            "기술 지원 요청",
        ],
        context_hints=["lifelong_learning", "senior_education", "health"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="basic",
        urgency_level="low",
        accessibility_needs=["large_text", "audio_support"],
    ),
    ExtendedPersonaType.ONLINE_STUDENT: ExtendedPersona(
        persona_type=PersonaType.JUNIOR,
        name="사이버 강의생",
        description="원격 수업, 온라인 접근, 시공간 제약",
        characteristics=[
            "원격 수업",
            "온라인 접근",
            "시공간 제약",
            "디지털 리터러시",
        ],
        query_styles=[
            "온라인 수업 문의",
            "원격 시험 확인",
            "디지털 자료 접근",
            "녹화 강의 요청",
        ],
        context_hints=["online_learning", "remote_access", "digital_platforms"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="intermediate",
        urgency_level="normal",
        accessibility_needs=["screen_reader", "caption"],
    ),
    ExtendedPersonaType.DISABLED_STUDENT: ExtendedPersona(
        persona_type=PersonaType.FRESHMAN,
        name="장애학생",
        description="특수 지원 필요, 접근성, 편의 제공",
        characteristics=[
            "장애 학생",
            "접근성 필요",
            "편의 서비스",
            "특수 지원",
        ],
        query_styles=[
            "접근성 문의",
            "편의 서비스 요청",
            "특수 지원 확인",
            "시험 연장 문의",
        ],
        context_hints=["accessibility", "disability_support", "special_needs"],
        language_proficiency="native",
        cultural_context=[],
        technical_expertise="basic",
        urgency_level="high",
        accessibility_needs=["wheelchair_access", "sign_language", "braille"],
    ),
}


# Ambiguous query templates for testing
AMBIGUOUS_QUERY_TEMPLATES = {
    AmbiguityType.MISSING_CONTEXT: [
        ("그거 언제까지야?", "deadline_inquiry", DifficultyLevel.MEDIUM),
        ("신청하는데 뭐 필요해?", "application_requirements", DifficultyLevel.EASY),
        ("어디서 해야 하나요?", "location_inquiry", DifficultyLevel.MEDIUM),
        ("거기 누가 있어?", "contact_inquiry", DifficultyLevel.EASY),
        ("방금 말한 거 다시 알려줘", "context_reference", DifficultyLevel.HARD),
    ],
    AmbiguityType.MULTIPLE_INTERPRETATIONS: [
        ("바뀌었나요?", "change_inquiry", DifficultyLevel.HARD),
        ("이거 되나요?", "eligibility_check", DifficultyLevel.HARD),
        ("신청 가능해?", "application_check", DifficultyLevel.MEDIUM),
        ("어떻게 돼?", "status_inquiry", DifficultyLevel.HARD),
        ("그거 받아?", "benefit_eligibility", DifficultyLevel.MEDIUM),
    ],
    AmbiguityType.UNCLEAR_INTENT: [
        ("학교 너무 힘들어", "emotional_expression", DifficultyLevel.HARD),
        ("도와주세요", "help_request", DifficultyLevel.HARD),
        ("뭐부터 해야 되는데", "guidance_request", DifficultyLevel.MEDIUM),
        ("이게 맞나요?", "confirmation_seek", DifficultyLevel.MEDIUM),
        ("어쩌면 좋죠?", "advice_request", DifficultyLevel.HARD),
    ],
    AmbiguityType.VAGUE_TERMINOLOGY: [
        ("그 규정이 뭐야?", "regulation_inquiry", DifficultyLevel.MEDIUM),
        ("이거 관련 거 있어?", "related_inquiry", DifficultyLevel.MEDIUM),
        ("비슷한 거 알려줘", "similar_item_request", DifficultyLevel.MEDIUM),
        ("요청하는 거야?", "procedural_check", DifficultyLevel.EASY),
        ("여기서 하는 거 맞아?", "process_confirmation", DifficultyLevel.EASY),
    ],
    AmbiguityType.INCOMPLETE_THOUGHT: [
        ("휴학하고 싶은데...", "leave_incomplete", DifficultyLevel.MEDIUM),
        ("장학금 받고 싶은데...", "scholarship_incomplete", DifficultyLevel.MEDIUM),
        ("성적이 안 좋아서...", "grade_concern_incomplete", DifficultyLevel.HARD),
        ("교수님이...", "professor_reference_incomplete", DifficultyLevel.HARD),
        ("등록금이...", "tuition_incomplete", DifficultyLevel.MEDIUM),
    ],
}


def get_all_extended_personas() -> List[ExtendedPersona]:
    """Get all extended persona definitions."""
    return list(EXTENDED_PERSONA_DEFINITIONS.values())


def get_extended_persona(persona_type: ExtendedPersonaType) -> ExtendedPersona:
    """Get a specific extended persona by type."""
    return EXTENDED_PERSONA_DEFINITIONS[persona_type]
