"""
Test Scenario Templates for RAG Testing.

Infrastructure layer providing templates and generators for comprehensive
test scenarios including ambiguous queries, multi-turn conversations,
and edge cases.

Clean Architecture: Infrastructure implements domain patterns and
uses external libraries for template generation.
"""

from typing import Any, Dict, List

from ..domain.entities import (
    DifficultyLevel,
    FollowUpType,
    PersonaType,
)
from ..domain.extended_entities import (
    AmbiguityType,
    EdgeCaseCategory,
)


class AmbiguousQueryTemplates:
    """
    Template collection for ambiguous query generation.

    Provides 30+ ambiguous query templates across 5 ambiguity types
    for testing RAG system's intent recognition and clarification.
    """

    # Missing context queries (6 templates)
    MISSING_CONTEXT = [
        {
            "query": "그거 마감 언제야?",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "휴학 신청 마감",
                "장학금 신청 마감",
                "등록금 납부 마감",
                "수강 신청 마감",
            ],
            "expected_clarifications": ["어떤 절차의 마감일인지 명시 필요"],
            "context_hints": ["deadline", "application_period"],
        },
        {
            "query": "신청하는데 뭐 필요해?",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.EASY,
            "expected_interpretations": [
                "휴학 신청 서류",
                "장학금 신청 서류",
                "입학 지원 서류",
            ],
            "expected_clarifications": ["신청 종류 명시 필요"],
            "context_hints": ["application_requirements", "documents"],
        },
        {
            "query": "어디서 해야 하나요?",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": ["신청 장소", "서류 제출처", "상담 장소"],
            "expected_clarifications": ["절차/행위 명시 필요"],
            "context_hints": ["location", "office"],
        },
        {
            "query": "거기 누구 있어?",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.EASY,
            "expected_interpretations": ["상담 교수", "담당 직원", "연락처"],
            "expected_clarifications": ["부서/사무실 명시 필요"],
            "context_hints": ["contact", "staff"],
        },
        {
            "query": "방금 말한 거 다시 알려줘",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": ["이전 맥락 참조", "대화 기록"],
            "expected_clarifications": ["맥락 없이 이해 불가", "이전 대화 필요"],
            "context_hints": ["context_tracking", "conversation_history"],
        },
        {
            "query": "그 시험 언제 있지?",
            "ambiguity_type": AmbiguityType.MISSING_CONTEXT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": ["중간고사", "기말고사", "필기시험", "면접"],
            "expected_clarifications": ["과목/시험 종류 명시 필요"],
            "context_hints": ["exam_schedule", "test_date"],
        },
    ]

    # Multiple interpretations queries (6 templates)
    MULTIPLE_INTERPRETATIONS = [
        {
            "query": "바뀌었나요?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "규정 변경",
                "담당자 변경",
                "신청 기간 변경",
                "시스템 변경",
            ],
            "expected_clarifications": ["변경 대상 명시 필요"],
            "context_hints": ["changes", "updates"],
        },
        {
            "query": "이거 되나요?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "자격 요건",
                "허용 여부",
                "가능성",
                "승인 가능",
            ],
            "expected_clarifications": ["행위/자격 명시 필요"],
            "context_hints": ["eligibility", "permission"],
        },
        {
            "query": "신청 가능해?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "신청 자격",
                "기간 내 신청",
                "재신청 가능",
                "추가 신청",
            ],
            "expected_clarifications": ["신청 종류/시기 명시 필요"],
            "context_hints": ["application", "eligibility"],
        },
        {
            "query": "어떻게 돼?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "처리 결과",
                "진행 상황",
                "규정 내용",
                "상태 확인",
            ],
            "expected_clarifications": ["확인 대상 명시 필요"],
            "context_hints": ["status", "outcome"],
        },
        {
            "query": "그거 받아?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "장학금 수혜",
                "서류 접수",
                "승인 허가",
                "신청 접수",
            ],
            "expected_clarifications": ["수혜/접수 대상 명시 필요"],
            "context_hints": ["benefit", "acceptance"],
        },
        {
            "query": "여기서도 돼?",
            "ambiguity_type": AmbiguityType.MULTIPLE_INTERPRETATIONS,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "장소 허용",
                "규정 적용",
                "신청 가능",
                "서비스 이용",
            ],
            "expected_clarifications": ["장소/행위 명시 필요"],
            "context_hints": ["location", "applicability"],
        },
    ]

    # Unclear intent queries (6 templates)
    UNCLEAR_INTENT = [
        {
            "query": "학교 너무 힘들어",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "학업 어려움",
                "경제적 어려움",
                "심리적 어려움",
                "휴학 고민",
            ],
            "expected_clarifications": ["어려움 종류 파악 필요"],
            "context_hints": ["distress", "counseling"],
        },
        {
            "query": "도와주세요",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "정보 요청",
                "긴급 상황",
                "절차 안내",
                "지원 요청",
            ],
            "expected_clarifications": ["도움 종류 파악 필요"],
            "context_hints": ["help_seeking", "assistance"],
        },
        {
            "query": "뭐부터 해야 되는데",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "신청 절차",
                "준비 사항",
                "우선순위",
                "시작 순서",
            ],
            "expected_clarifications": ["목표/절차 명시 필요"],
            "context_hints": ["procedure", "guidance"],
        },
        {
            "query": "이게 맞나요?",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "이해 확인",
                "절차 확인",
                "자격 확인",
                "진행 상황 확인",
            ],
            "expected_clarifications": ["확인 대상 명시 필요"],
            "context_hints": ["confirmation", "verification"],
        },
        {
            "query": "어쩌면 좋죠?",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "해결책 요청",
                "대안 요청",
                "조언 필요",
                "다음 단계",
            ],
            "expected_clarifications": ["문제 상황 파악 필요"],
            "context_hints": ["advice", "solution"],
        },
        {
            "query": "말이 안 통해요",
            "ambiguity_type": AmbiguityType.UNCLEAR_INTENT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "이해 어려움",
                "답변 불만족",
                "소통 어려움",
                "설명 부족",
            ],
            "expected_clarifications": ["어려움 원인 파악 필요"],
            "context_hints": ["communication", "confusion"],
        },
    ]

    # Vague terminology queries (6 templates)
    VAGUE_TERMINOLOGY = [
        {
            "query": "그 규정이 뭐야?",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "특정 규정 문의",
                "용어 설명",
                "내용 확인",
            ],
            "expected_clarifications": ["규정 종류/용어 명시 필요"],
            "context_hints": ["regulation", "terminology"],
        },
        {
            "query": "이거 관련 거 있어?",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "관련 규정",
                "유사 사례",
                "관련 절차",
                "참고 자료",
            ],
            "expected_clarifications": ["주제/관련성 명시 필요"],
            "context_hints": ["related_info", "reference"],
        },
        {
            "query": "비슷한 거 알려줘",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "유사 제도",
                "대안 안내",
                "비교 대상",
                "관련 혜택",
            ],
            "expected_clarifications": ["비교 기준 명시 필요"],
            "context_hints": ["similar", "alternative"],
        },
        {
            "query": "요청하는 거야?",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.EASY,
            "expected_interpretations": [
                "신청 절차",
                "제출 방법",
                "요청 처리",
            ],
            "expected_clarifications": ["요청 종류 명시 필요"],
            "context_hints": ["request", "submission"],
        },
        {
            "query": "여기서 하는 거 맞아?",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.EASY,
            "expected_interpretations": [
                "장소 확인",
                "절차 확인",
                "담당 확인",
            ],
            "expected_clarifications": ["행위/장소 명시 필요"],
            "context_hints": ["location", "procedure"],
        },
        {
            "query": "거기 그거 어떻게 되?",
            "ambiguity_type": AmbiguityType.VAGUE_TERMINOLOGY,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "진행 상황",
                "처리 결과",
                "규정 내용",
            ],
            "expected_clarifications": ["대상/상황 명시 필요"],
            "context_hints": ["status", "outcome"],
        },
    ]

    # Incomplete thought queries (6 templates)
    INCOMPLETE_THOUGHT = [
        {
            "query": "휴학하고 싶은데...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "휴학 절차",
                "휴학 기간",
                "휴학 효과",
                "휴학 자격",
            ],
            "expected_clarifications": ["구체적 문제 파악 필요"],
            "context_hints": ["leave_of_absence", "concerns"],
        },
        {
            "query": "장학금 받고 싶은데...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "장학금 종류",
                "신청 자격",
                "신청 방법",
                "성적 기준",
            ],
            "expected_clarifications": ["관심사 파악 필요"],
            "context_hints": ["scholarship", "interest"],
        },
        {
            "query": "성적이 안 좋아서...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "재수강",
                "성적 정정",
                "학사 경고",
                "장학금 탈락",
            ],
            "expected_clarifications": ["걱정 사항 파악 필요"],
            "context_hints": ["grade_concern", "academic_difficulty"],
        },
        {
            "query": "교수님이...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.HARD,
            "expected_interpretations": [
                "성적 이의",
                "상담 요청",
                "수업 관련",
                "추천서",
            ],
            "expected_clarifications": ["문제 상황 파악 필요"],
            "context_hints": ["professor", "issue"],
        },
        {
            "query": "등록금이...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "등록금 납부",
                "등록금 반환",
                "등록금 감면",
                "분납 가능",
            ],
            "expected_clarifications": ["관심사 파악 필요"],
            "context_hints": ["tuition", "concern"],
        },
        {
            "query": "졸업하려면...",
            "ambiguity_type": AmbiguityType.INCOMPLETE_THOUGHT,
            "difficulty": DifficultyLevel.MEDIUM,
            "expected_interpretations": [
                "졸업 요건",
                "졸업 학점",
                "졸업 시기",
                "졸업 절차",
            ],
            "expected_clarifications": ["구체적 관심사 파악 필요"],
            "context_hints": ["graduation", "requirements"],
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all ambiguous query templates."""
        all_templates = []
        all_templates.extend(cls.MISSING_CONTEXT)
        all_templates.extend(cls.MULTIPLE_INTERPRETATIONS)
        all_templates.extend(cls.UNCLEAR_INTENT)
        all_templates.extend(cls.VAGUE_TERMINOLOGY)
        all_templates.extend(cls.INCOMPLETE_THOUGHT)
        return all_templates

    @classmethod
    def get_templates_by_type(
        cls, ambiguity_type: AmbiguityType
    ) -> List[Dict[str, Any]]:
        """Get templates by ambiguity type."""
        type_mapping = {
            AmbiguityType.MISSING_CONTEXT: cls.MISSING_CONTEXT,
            AmbiguityType.MULTIPLE_INTERPRETATIONS: cls.MULTIPLE_INTERPRETATIONS,
            AmbiguityType.UNCLEAR_INTENT: cls.UNCLEAR_INTENT,
            AmbiguityType.VAGUE_TERMINOLOGY: cls.VAGUE_TERMINOLOGY,
            AmbiguityType.INCOMPLETE_THOUGHT: cls.INCOMPLETE_THOUGHT,
        }
        return type_mapping.get(ambiguity_type, [])


class MultiTurnScenarioTemplates:
    """
    Template collection for multi-turn conversation scenarios.

    Provides 15 multi-turn conversation templates across different
    personas and follow-up patterns.
    """

    # Scenario 1: Freshman dormitory inquiry (4 turns)
    FRESHMAN_DORMITORY = {
        "scenario_id": "mt-freshman-dorm",
        "name": "신입생 기숙사 문의",
        "description": "신입생이 기숙사 신청에 대해 구체적인 정보를 확인",
        "persona_type": PersonaType.FRESHMAN,
        "difficulty": DifficultyLevel.EASY,
        "context_window_size": 3,
        "initial_query": "기숙사 어떻게 신청해요?",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "신청 기간은 언제인가요?",
                "expected_intent": "기간 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "구체적으로 어떤 서류가 필요한가요?",
                "expected_intent": "서류 상세",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "신청 기간이 지났으면 다른 방법은 없나요?",
                "expected_intent": "예외 확인",
            },
            {
                "turn_number": 5,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "그러니까 신청 기간 내에 서류 준비해서 제출하면 되는 거 맞죠?",
                "expected_intent": "이해 확인",
            },
        ],
        "expected_context_preservation_rate": 0.9,
        "expected_topic_transitions": ["신청 절차", "기간", "서류", "예외", "확인"],
    }

    # Scenario 2: Junior graduation requirements (5 turns)
    JUNIOR_GRADUATION = {
        "scenario_id": "mt-junior-grad",
        "name": "재학생 졸업 요건 확인",
        "description": "3학년이 졸업 요건을 다각도로 확인",
        "persona_type": PersonaType.JUNIOR,
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "졸업 요건이 어떻게 되나요?",
        "initial_expected_intent": "자격 확인",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "전공 학점은 몇 점이 필요한가요?",
                "expected_intent": "학점 상세",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "교양 학점도 따로 정해져 있나요?",
                "expected_intent": "관련 확장",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "졸업 작품 대신 다른 걸로 할 수 있나요?",
                "expected_intent": "예외 확인",
            },
            {
                "turn_number": 5,
                "follow_up_type": FollowUpType.CONDITION_CHANGE,
                "query": "만약 한 학기를 더 다니면 바뀌나요?",
                "expected_intent": "조건 변경",
            },
            {
                "turn_number": 6,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "정리하면 전공 120학점, 교양 45학점이면 졸업 가능한 거 맞죠?",
                "expected_intent": "이해 확인",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "졸업 요건",
            "전공 학점",
            "교양 학점",
            "졸업 작품",
            "연기",
            "확인",
        ],
    }

    # Scenario 3: Graduate thesis review (4 turns)
    GRADUATE_THESIS = {
        "scenario_id": "mt-grad-thesis",
        "name": "대학원생 논문 심사",
        "description": "대학원생이 논문 심사 절차를 확인",
        "persona_type": PersonaType.GRADUATE,
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "논문 심사 기준이 어떻게 됩니까?",
        "initial_expected_intent": "기준 확인",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "심사 위원은 몇 명이어야 하나요?",
                "expected_intent": "절차 상세",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "심사에 불합격하면 재심사는 가능한가요?",
                "expected_intent": "예외 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "그러니까 논문 제출 후 3개월 안에 심사가 끝나는 거 맞죠?",
                "expected_intent": "이해 확인",
            },
            {
                "turn_number": 5,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "심사 통과 후 학위 수여까지 얼마나 걸리나요?",
                "expected_intent": "관련 문의",
            },
        ],
        "expected_context_preservation_rate": 0.8,
        "expected_topic_transitions": [
            "심사 기준",
            "위원 구성",
            "불합격 처리",
            "기간 확인",
            "학위 수여",
        ],
    }

    # Scenario 4: Distressed student leave of absence (5 turns)
    DISTRESSED_LEAVE = {
        "scenario_id": "mt-distressed-leave",
        "name": "어려운 상황 학생 휴학 상담",
        "description": "어려운 상황의 학생이 휴학에 대해 문의",
        "persona_type": PersonaType.DISTRESSED_STUDENT,
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "학교 다니기 너무 힘들어서 휴학하고 싶어요",
        "initial_expected_intent": "긴급 도움 요청",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "휴학 신청하는 방법 좀 자세히 알려주실 수 있나요?",
                "expected_intent": "절차 문의",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "신청하려면 어떤 서류를 제출해야 하나요?",
                "expected_intent": "서류 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "휴학 기간은 어디까지 가능한가요?",
                "expected_intent": "관련 문의",
            },
            {
                "turn_number": 5,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "그러니까 복학할 때 따로 신청이 필요 없는 거 맞죠?",
                "expected_intent": "이해 확인",
            },
            {
                "turn_number": 6,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "휴학 중에도 학교 시설 이용 가능한가요?",
                "expected_intent": "관련 문의",
            },
        ],
        "expected_context_preservation_rate": 0.75,
        "expected_topic_transitions": [
            "휴학 고민",
            "신청 절차",
            "필요 서류",
            "휴학 기간",
            "복학 절차",
            "시설 이용",
        ],
    }

    # Scenario 5: Parent tuition inquiry (3 turns)
    PARENT_TUITION = {
        "scenario_id": "mt-parent-tuition",
        "name": "학부모 등록금 문의",
        "description": "학부모가 자녀 등록금 납부에 대해 확인",
        "persona_type": PersonaType.PARENT,
        "difficulty": DifficultyLevel.EASY,
        "context_window_size": 3,
        "initial_query": "등록금 납부 기간이 언제인가요?",
        "initial_expected_intent": "기간 확인",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "납부 방법은 어떤 것이 있나요?",
                "expected_intent": "방법 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "분납도 가능한가요?",
                "expected_intent": "예외 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "그러니까 매달 일정 금액씩 나눠서 납부하면 되는 거 맞죠?",
                "expected_intent": "이해 확인",
            },
        ],
        "expected_context_preservation_rate": 0.95,
        "expected_topic_transitions": ["납부 기간", "납부 방법", "분납 여부", "확인"],
    }

    # Scenario 6: Professor sabbatical leave (4 turns)
    PROFESSOR_SABBATICAL = {
        "scenario_id": "mt-prof-sabbatical",
        "name": "정교수 연구년 휴직",
        "description": "정교수가 연구년 휴직에 대해 확인",
        "persona_type": PersonaType.PROFESSOR,
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "연구년 휴직 신청하는 방법 알려주세요",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "신청 자격은 어떻게 되나요?",
                "expected_intent": "자격 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "연구년 기간 동안 급여는 어떻게 되나요?",
                "expected_intent": "혜택 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "연구년 중에 다른 연구를 병행해도 되나요?",
                "expected_intent": "예외 확인",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "신청 절차",
            "신청 자격",
            "급여 혜택",
            "연구 병행",
        ],
    }

    # Additional scenarios (7-15)
    # Scenario 7: International student visa (4 turns)
    INTERNATIONAL_VISA = {
        "scenario_id": "mt-international-visa",
        "name": "유학생 비자 문의",
        "description": "유학생이 학업과 관련된 비자 문제 확인",
        "persona_type": PersonaType.FRESHMAN,  # Base type, will be extended
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "학생 비자 연장 어떻게 하나요?",
        "initial_expected_intent": "비자 연장",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "어떤 서류가 필요한가요?",
                "expected_intent": "서류 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "비자 만료 전까지 얼마나 일찍 신청해야 하나요?",
                "expected_intent": "기간 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "휴학하면 비자에 문제 없나요?",
                "expected_intent": "관련 문의",
            },
        ],
        "expected_context_preservation_rate": 0.8,
        "expected_topic_transitions": [
            "비자 연장",
            "필요 서류",
            "신청 기간",
            "휴학 영향",
        ],
    }

    # Scenario 8: Staff annual leave (3 turns)
    STAFF_LEAVE = {
        "scenario_id": "mt-staff-leave",
        "name": "직원 연차 사용",
        "description": "신입 직원이 연차 사용에 대해 확인",
        "persona_type": PersonaType.NEW_STAFF,
        "difficulty": DifficultyLevel.EASY,
        "context_window_size": 3,
        "initial_query": "연차 휴가 사용하는 방법 알려주세요",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "연차는 며칠까지 사용해야 하나요?",
                "expected_intent": "사용 기간",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "연차를 미래로 미룰 수 있나요?",
                "expected_intent": "예외 확인",
            },
        ],
        "expected_context_preservation_rate": 0.9,
        "expected_topic_transitions": ["연차 사용", "사용 기간", "연차 이월"],
    }

    # Scenario 9: Dissatisfied member complaint (5 turns)
    DISSATISFIED_COMPLAINT = {
        "scenario_id": "mt-dissatisfied-complaint",
        "name": "불만 구성원 성적 항의",
        "description": "불만 있는 구성원이 성적 처리에 대해 항의",
        "persona_type": PersonaType.DISSATISFIED_MEMBER,
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "성적이 부당하게 매겨졌습니다",
        "initial_expected_intent": "불만 제기",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "이의신청은 어떻게 하나요?",
                "expected_intent": "절차 문의",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "신청 기한은 언제까지인가요?",
                "expected_intent": "기간 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "이의신청 기간이 지났으면 다른 방법은 없나요?",
                "expected_intent": "대안 확인",
            },
            {
                "turn_number": 5,
                "follow_up_type": FollowUpType.CONDITION_CHANGE,
                "query": "만약 이의신청이 기각되면 상위 단계는 없나요?",
                "expected_intent": "항소 절차",
            },
            {
                "turn_number": 6,
                "follow_up_type": FollowUpType.CONFIRMATION,
                "query": "정리하면 이의신청 → 재심사 → 항소 순서로 진행되는 거 맞죠?",
                "expected_intent": "이해 확인",
            },
        ],
        "expected_context_preservation_rate": 0.7,
        "expected_topic_transitions": [
            "불만 제기",
            "이의신청",
            "기간 확인",
            "대안",
            "항소",
        ],
    }

    # Scenario 10: Transfer student credit (4 turns)
    TRANSFER_CREDIT = {
        "scenario_id": "mt-transfer-credit",
        "name": "편입생 학점 인정",
        "description": "편입생이 기존 학점 인정에 대해 확인",
        "persona_type": PersonaType.JUNIOR,  # Base type
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "기존 학점 인정 어떻게 되나요?",
        "initial_expected_intent": "학점 인정",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "어떤 과목은 인정이 안 되나요?",
                "expected_intent": "제한 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "학점 인정 신청은 언제 하나요?",
                "expected_intent": "신청 절차",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "인정받지 못한 학점은 대체 과정으로 채울 수 있나요?",
                "expected_intent": "예외 확인",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "학점 인정",
            "인정 제한",
            "신청 절차",
            "대체 과정",
        ],
    }

    # Scenario 11: Scholarship eligibility (4 turns)
    SCHOLARSHIP_ELIGIBILITY = {
        "scenario_id": "mt-scholarship-elig",
        "name": "장학금 자격 확인",
        "description": "학생이 다양한 장학금 자격을 확인",
        "persona_type": PersonaType.FRESHMAN,
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "장학금 뭐 있나요?",
        "initial_expected_intent": "장학금 종류",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "성적 장학금은 어떤 조건이 필요한가요?",
                "expected_intent": "성적 장학금",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "근로 장학금도 신청할 수 있나요?",
                "expected_intent": "관련 장학금",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.COMPARISON,
                "query": "성적 장학금이랑 근로 장학금이랑 동시에 받을 수 있나요?",
                "expected_intent": "중복 수급",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "장학금 종류",
            "성적 장학금",
            "근로 장학금",
            "중복 수급",
        ],
    }

    # Scenario 12: Course registration (4 turns)
    COURSE_REGISTRATION = {
        "scenario_id": "mt-course-reg",
        "name": "수강 신청 절차",
        "description": "신입생이 수강 신청 절차를 확인",
        "persona_type": PersonaType.FRESHMAN,
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "수강 신청 어떻게 하나요?",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "신청 기간은 언제인가요?",
                "expected_intent": "기간 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "수강 정원이 찼으면 대기 신청도 가능한가요?",
                "expected_intent": "예외 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.CONDITION_CHANGE,
                "query": "수강 변경은 언제까지 가능한가요?",
                "expected_intent": "변경 기간",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "수강 신청",
            "신청 기간",
            "대기 신청",
            "변경 기간",
        ],
    }

    # Scenario 13: Research funding (3 turns)
    RESEARCH_FUNDING = {
        "scenario_id": "mt-research-fund",
        "name": "연구비 지원",
        "description": "연구원이 연구비 지원을 확인",
        "persona_type": PersonaType.GRADUATE,
        "difficulty": DifficultyLevel.HARD,
        "context_window_size": 3,
        "initial_query": "연구비 지원 받을 수 있나요?",
        "initial_expected_intent": "지원 가능",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "어떤 종류의 연구비가 있나요?",
                "expected_intent": "지원 종류",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "신청하는 방법은 어떻게 되나요?",
                "expected_intent": "신청 절차",
            },
        ],
        "expected_context_preservation_rate": 0.9,
        "expected_topic_transitions": ["연구비 지원", "지원 종류", "신청 절차"],
    }

    # Scenario 14: Withdrawal process (4 turns)
    WITHDRAWAL_PROCESS = {
        "scenario_id": "mt-withdrawal",
        "name": "자퇴 절차",
        "description": "학생이 자퇴 절차를 확인",
        "persona_type": PersonaType.JUNIOR,
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "자퇴하고 싶은데 어떻게 하나요?",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "어디서 신청해야 하나요?",
                "expected_intent": "신청 장소",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.RELATED_EXPANSION,
                "query": "등록금 환급은 어떻게 되나요?",
                "expected_intent": "환급 문의",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "자퇴 후 재입학은 가능한가요?",
                "expected_intent": "재입학 가능",
            },
        ],
        "expected_context_preservation_rate": 0.8,
        "expected_topic_transitions": [
            "자퇴 절차",
            "신청 장소",
            "등록금 환급",
            "재입학",
        ],
    }

    # Scenario 15: Grade appeal (4 turns)
    GRADE_APPEAL = {
        "scenario_id": "mt-grade-appeal",
        "name": "성적 이의 신청",
        "description": "학생이 성적 이의 신청 절차를 확인",
        "persona_type": PersonaType.JUNIOR,
        "difficulty": DifficultyLevel.MEDIUM,
        "context_window_size": 3,
        "initial_query": "성적 이의 어떻게 하나요?",
        "initial_expected_intent": "절차 문의",
        "turns": [
            {
                "turn_number": 2,
                "follow_up_type": FollowUpType.CLARIFICATION,
                "query": "이의신청 기간은 언제까지인가요?",
                "expected_intent": "기간 확인",
            },
            {
                "turn_number": 3,
                "follow_up_type": FollowUpType.PROCEDURAL_DEEPENING,
                "query": "어떤 서류를 제출해야 하나요?",
                "expected_intent": "서류 확인",
            },
            {
                "turn_number": 4,
                "follow_up_type": FollowUpType.EXCEPTION_CHECK,
                "query": "이의신청이 반려되면 항소할 수 있나요?",
                "expected_intent": "항소 절차",
            },
        ],
        "expected_context_preservation_rate": 0.85,
        "expected_topic_transitions": [
            "이의신청",
            "기간 확인",
            "서류 제출",
            "항소 절차",
        ],
    }

    @classmethod
    def get_all_scenarios(cls) -> List[Dict[str, Any]]:
        """Get all multi-turn scenario templates."""
        return [
            cls.FRESHMAN_DORMITORY,
            cls.JUNIOR_GRADUATION,
            cls.GRADUATE_THESIS,
            cls.DISTRESSED_LEAVE,
            cls.PARENT_TUITION,
            cls.PROFESSOR_SABBATICAL,
            cls.INTERNATIONAL_VISA,
            cls.STAFF_LEAVE,
            cls.DISSATISFIED_COMPLAINT,
            cls.TRANSFER_CREDIT,
            cls.SCHOLARSHIP_ELIGIBILITY,
            cls.COURSE_REGISTRATION,
            cls.RESEARCH_FUNDING,
            cls.WITHDRAWAL_PROCESS,
            cls.GRADE_APPEAL,
        ]


class EdgeCaseTemplates:
    """
    Template collection for edge case scenarios.

    Provides 15 edge case templates across different categories
    for testing RAG system's robustness.
    """

    # Emotional edge cases (4 templates)
    EMOTIONAL = [
        {
            "scenario_id": "edge-emotional-001",
            "name": "좌절한 학생",
            "category": EdgeCaseCategory.EMOTIONAL,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.DISTRESSED_STUDENT,
            "query": "학교생활 적응 못 하겠어요... 너무 힘들어요",
            "is_emotional": True,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": True,
            "expected_empathy_level": "highly_empathetic",
            "expected_response_speed": "normal",
            "should_escalate": True,
            "expected_regulations": ["상담 센터", "심리 지원"],
            "expected_actions": ["상담 안내", "지원 프로그램 소개"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
        {
            "scenario_id": "edge-emotional-002",
            "name": "긴급한 상황",
            "category": EdgeCaseCategory.EMOTIONAL,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.DISTRESSED_STUDENT,
            "query": "지금 당장 도와주세요! 정말 급해요!",
            "is_emotional": True,
            "is_urgent": True,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "empathetic",
            "expected_response_speed": "immediate",
            "should_escalate": True,
            "expected_regulations": ["긴급 연락처", "응급 지원"],
            "expected_actions": ["즉시 연락처 제공", "긴급 절차 안내"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
        {
            "scenario_id": "edge-emotional-003",
            "name": "분노한 교수",
            "category": EdgeCaseCategory.EMOTIONAL,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.PROFESSOR,
            "query": "이게 어떻게 된 거입니까! 시스템이 왜 이렇게 복잡합니까!",
            "is_emotional": True,
            "is_urgent": False,
            "is_confused": False,
            "is_frustrated": True,
            "expected_empathy_level": "empathetic",
            "expected_response_speed": "fast",
            "should_escalate": False,
            "expected_regulations": ["교원 권리", "불만 처리"],
            "expected_actions": ["차분한 설명", "개선 안내"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
        {
            "scenario_id": "edge-emotional-004",
            "name": "불안한 학부모",
            "category": EdgeCaseCategory.EMOTIONAL,
            "difficulty": DifficultyLevel.MEDIUM,
            "persona_type": PersonaType.PARENT,
            "query": "자녀가 학교에서 잘 지내는지 걱정돼서요...",
            "is_emotional": True,
            "is_urgent": False,
            "is_confused": False,
            "is_frustrated": False,
            "expected_empathy_level": "empathetic",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["학생 확인", "상담 절차"],
            "expected_actions": ["확인 방법 안내", "상담 예약 안내"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
    ]

    # Deadline-critical edge cases (3 templates)
    DEADLINE_CRITICAL = [
        {
            "scenario_id": "edge-deadline-001",
            "name": "마감 1시간 전",
            "category": EdgeCaseCategory.DEADLINE_CRITICAL,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.JUNIOR,
            "query": "휴학 신청 마감이 1시간 남았는데 아직 안 했어요! 어떡하죠?",
            "is_emotional": False,
            "is_urgent": True,
            "is_confused": False,
            "is_frustrated": True,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "immediate",
            "should_escalate": False,
            "expected_regulations": ["휴학 신청", "지연 제출"],
            "expected_actions": ["즉시 절차 안내", "연락처 제공"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
        {
            "scenario_id": "edge-deadline-002",
            "name": "마감 이미 지남",
            "category": EdgeCaseCategory.DEADLINE_CRITICAL,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.FRESHMAN,
            "query": "장학금 신청 마감이 지났는데... 늦게라도 신청할 수 있나요?",
            "is_emotional": True,
            "is_urgent": True,
            "is_confused": False,
            "is_frustrated": True,
            "expected_empathy_level": "empathetic",
            "expected_response_speed": "fast",
            "should_escalate": False,
            "expected_regulations": ["신청 기간", "지연 신청"],
            "expected_actions": ["예외 사유 안내", "대안 제시"],
            "should_show_empathy": True,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
        {
            "scenario_id": "edge-deadline-003",
            "name": "기한 임박 확인",
            "category": EdgeCaseCategory.DEADLINE_CRITICAL,
            "difficulty": DifficultyLevel.MEDIUM,
            "persona_type": PersonaType.GRADUATE,
            "query": "논문 제출 기간이 얼마나 남았나요?",
            "is_emotional": False,
            "is_urgent": True,
            "is_confused": False,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "fast",
            "should_escalate": False,
            "expected_regulations": ["논문 제출", "기한"],
            "expected_actions": ["정확한 날짜 제공", "연장 가능 여부"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
    ]

    # Complex synthesis edge cases (3 templates)
    COMPLEX_SYNTHESIS = [
        {
            "scenario_id": "edge-synthesis-001",
            "name": "다중 규정 통합",
            "category": EdgeCaseCategory.COMPLEX_SYNTHESIS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.JUNIOR,
            "query": "교환학생 가면서 휴학도 하고 장학금도 받고 싶은데 이게 다 가능한가요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["교환학생", "휴학", "장학금"],
            "expected_actions": ["규정 간 관계 설명", "제한 사항 안내"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
        {
            "scenario_id": "edge-synthesis-002",
            "name": "복합 자격 확인",
            "category": EdgeCaseCategory.COMPLEX_SYNTHESIS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.GRADUATE,
            "query": "외국인 유학생인데 조교 일도 하고 장학금도 받고 싶은데 가능할까요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["유학생", "조교", "장학금"],
            "expected_actions": ["자격 요건 통합", "제한 사항 설명"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": True,
        },
        {
            "scenario_id": "edge-synthesis-003",
            "name": "다중 혜택 중복",
            "category": EdgeCaseCategory.COMPLEX_SYNTHESIS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.FRESHMAN,
            "query": "기숙사 살면서 장학금도 받고 근로장학생도 하면 다 가능한가요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["기숙사", "장학금", "근로장학생"],
            "expected_actions": ["중복 수급 규정", "가능 조합 안내"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
    ]

    # Cross-referenced edge cases (3 templates)
    CROSS_REFERENCED = [
        {
            "scenario_id": "edge-xref-001",
            "name": "상호 참조 규정",
            "category": EdgeCaseCategory.CROSS_REFERENCED,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.PROFESSOR,
            "query": "교원 휴직과 연구년 휴직이 어떻게 다른가요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": False,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["교원 휴직", "연구년"],
            "expected_actions": ["두 규정 비교", "차이점 명시"],
            "should_show_empathy": False,
            "should_provide_contact": False,
            "should_offer_alternatives": False,
        },
        {
            "scenario_id": "edge-xref-002",
            "name": "관련 규정 확인",
            "category": EdgeCaseCategory.CROSS_REFERENCED,
            "difficulty": DifficultyLevel.MEDIUM,
            "persona_type": PersonaType.STAFF_MANAGER,
            "query": "휴가와 연차 그리고 병가가 각각 어떻게 다른가요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": False,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["휴가", "연차", "병가"],
            "expected_actions": ["세 가지 비교", "적용 상황 설명"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
        {
            "scenario_id": "edge-xref-003",
            "name": "계층적 규정",
            "category": EdgeCaseCategory.CROSS_REFERENCED,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.JUNIOR,
            "query": "학칙하고 학사 규정하고 수강 신청 규정이 우선순위가 어떻게 되나요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["학칙", "학사 규정", "수강 신청"],
            "expected_actions": ["우선순위 설명", "충돌 시 해결"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
    ]

    # Contradictory edge cases (2 templates)
    CONTRADICTORY = [
        {
            "scenario_id": "edge-contradict-001",
            "name": "규정 충돌",
            "category": EdgeCaseCategory.CONTRADICTORY,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.PROFESSOR,
            "query": "교원 규정에는 이렇게 돼 있는데 학부 규정에는 저렇게 돼 있어요. 어거 따르나요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["교원 규정", "학부 규정"],
            "expected_actions": ["우선순위 명시", "충돌 해결"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
        {
            "scenario_id": "edge-contradict-002",
            "name": "예외 규정",
            "category": EdgeCaseCategory.CONTRADICTORY,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.GRADUATE,
            "query": "일반적인 규정이랑 대학원 규정이랑 다른데 대학원생은 어떤 거 따르나요?",
            "is_emotional": False,
            "is_urgent": False,
            "is_confused": True,
            "is_frustrated": False,
            "expected_empathy_level": "neutral",
            "expected_response_speed": "normal",
            "should_escalate": False,
            "expected_regulations": ["일반 규정", "대학원 규정"],
            "expected_actions": ["적용 대상 명시", "섭외 원칙"],
            "should_show_empathy": False,
            "should_provide_contact": True,
            "should_offer_alternatives": False,
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all edge case templates."""
        all_templates = []
        all_templates.extend(cls.EMOTIONAL)
        all_templates.extend(cls.DEADLINE_CRITICAL)
        all_templates.extend(cls.COMPLEX_SYNTHESIS)
        all_templates.extend(cls.CROSS_REFERENCED)
        all_templates.extend(cls.CONTRADICTORY)
        return all_templates

    @classmethod
    def get_by_category(cls, category: EdgeCaseCategory) -> List[Dict[str, Any]]:
        """Get templates by category."""
        category_mapping = {
            EdgeCaseCategory.EMOTIONAL: cls.EMOTIONAL,
            EdgeCaseCategory.DEADLINE_CRITICAL: cls.DEADLINE_CRITICAL,
            EdgeCaseCategory.COMPLEX_SYNTHESIS: cls.COMPLEX_SYNTHESIS,
            EdgeCaseCategory.CROSS_REFERENCED: cls.CROSS_REFERENCED,
            EdgeCaseCategory.CONTRADICTORY: cls.CONTRADICTORY,
        }
        return category_mapping.get(category, [])


class TypoEdgeCaseTemplates:
    """
    Template collection for typo edge cases.

    REQ-P4-001: Tests RAG system's robustness to common Korean typos.
    Provides 3 typo scenarios targeting common user input errors.
    """

    # Typo edge cases (3 templates)
    TYPO_SCENARIOS = [
        {
            "scenario_id": "edge-typo-001",
            "name": "신정 → 신청",
            "category": EdgeCaseCategory.TYPO,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.FRESHMAN,
            "query": "휴학 신정 어떻게 하나요?",
            "typo_type": "consonant_confusion",
            "original_word": "신청",
            "typo_word": "신정",
            "expected_interpretation": "휴학 신청",
            "expected_correction": "신정 → 신청으로 자동 수정 안내",
            "should_provide_answer": True,
            "expected_regulations": ["휴학 규정"],
            "should_ask_clarification": False,
        },
        {
            "scenario_id": "edge-typo-002",
            "name": "규칙 → 규정",
            "category": EdgeCaseCategory.TYPO,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.JUNIOR,
            "query": "졸업 규칙이 뭐예요?",
            "typo_type": "word_confusion",
            "original_word": "규정",
            "typo_word": "규칙",
            "expected_interpretation": "졸업 규정",
            "expected_correction": "규칙 → 규정으로 안내",
            "should_provide_answer": True,
            "expected_regulations": ["졸업 요건 규정"],
            "should_ask_clarification": False,
        },
        {
            "scenario_id": "edge-typo-003",
            "name": "장학급 → 장학금",
            "category": EdgeCaseCategory.TYPO,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.FRESHMAN,
            "query": "장학급 받으려면 뭐 해야 돼요?",
            "typo_type": "sound_confusion",
            "original_word": "장학금",
            "typo_word": "장학급",
            "expected_interpretation": "장학금 신청",
            "expected_correction": "장학급 → 장학금으로 안내",
            "should_provide_answer": True,
            "expected_regulations": ["장학금 규정"],
            "should_ask_clarification": False,
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all typo edge case templates."""
        return cls.TYPO_SCENARIOS


class AmbiguousQueryEdgeCaseTemplates:
    """
    Template collection for ambiguous query edge cases.

    REQ-P4-002: Tests RAG system's handling of highly ambiguous queries.
    Provides 3 ambiguous query scenarios requiring clarification.
    """

    # Ambiguous edge cases (3 templates)
    AMBIGUOUS_SCENARIOS = [
        {
            "scenario_id": "edge-ambiguous-001",
            "name": "그거 마감 언제야",
            "category": EdgeCaseCategory.AMBIGUOUS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.FRESHMAN,
            "query": "그거 마감 언제야?",
            "ambiguity_type": "missing_context",
            "possible_interpretations": [
                "휴학 신청 마감",
                "장학금 신청 마감",
                "등록금 납부 마감",
                "수강 신청 마감",
            ],
            "expected_behavior": "ask_clarification",
            "should_provide_answer": False,
            "expected_clarification_options": [
                "휴학 신청",
                "장학금 신청",
                "등록금 납부",
                "수강 신청",
            ],
            "expected_regulations": [],
            "should_ask_clarification": True,
        },
        {
            "scenario_id": "edge-ambiguous-002",
            "name": "이거 되나요",
            "category": EdgeCaseCategory.AMBIGUOUS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.JUNIOR,
            "query": "이거 되나요?",
            "ambiguity_type": "missing_context",
            "possible_interpretations": [
                "신청 가능 여부",
                "자격 요건 충족",
                "규정 허용 여부",
                "기간 내 가능",
            ],
            "expected_behavior": "ask_clarification",
            "should_provide_answer": False,
            "expected_clarification_options": [
                "신청 가능 여부",
                "자격 요건",
                "규정 확인",
            ],
            "expected_regulations": [],
            "should_ask_clarification": True,
        },
        {
            "scenario_id": "edge-ambiguous-003",
            "name": "어떻게 해야 돼",
            "category": EdgeCaseCategory.AMBIGUOUS,
            "difficulty": DifficultyLevel.HARD,
            "persona_type": PersonaType.FRESHMAN,
            "query": "어떻게 해야 돼?",
            "ambiguity_type": "missing_context",
            "possible_interpretations": [
                "신청 절차",
                "준비 서류",
                "자격 요건",
                "기간 확인",
            ],
            "expected_behavior": "ask_clarification",
            "should_provide_answer": False,
            "expected_clarification_options": [
                "신청 방법",
                "필요 서류",
                "자격 확인",
                "기간 안내",
            ],
            "expected_regulations": [],
            "should_ask_clarification": True,
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all ambiguous query edge case templates."""
        return cls.AMBIGUOUS_SCENARIOS


class NonExistentRegulationTemplates:
    """
    Template collection for non-existent regulation queries.

    REQ-P4-003: Tests RAG system's handling of non-existent regulation queries.
    Provides 2 scenarios for regulations that don't exist in the database.
    """

    # Non-existent regulation edge cases (2 templates)
    NON_EXISTENT_SCENARIOS = [
        {
            "scenario_id": "edge-nonexist-001",
            "name": "로봇 연구 규정",
            "category": EdgeCaseCategory.NON_EXISTENT,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.GRADUATE,
            "query": "로봇 연구 규정이 어떻게 되나요?",
            "regulation_type": "non_existent",
            "expected_behavior": "inform_not_found",
            "should_provide_answer": False,
            "expected_message": "해당 규정을 찾을 수 없습니다",
            "should_suggest_alternatives": True,
            "expected_alternatives": ["연구 규정", "연구실 안전 규정", "실험실 규정"],
            "should_ask_clarification": False,
        },
        {
            "scenario_id": "edge-nonexist-002",
            "name": "드론 비행 규칙",
            "category": EdgeCaseCategory.NON_EXISTENT,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.JUNIOR,
            "query": "학교 드론 비행 규칙 알려주세요",
            "regulation_type": "non_existent",
            "expected_behavior": "inform_not_found",
            "should_provide_answer": False,
            "expected_message": "해당 규정을 찾을 수 없습니다",
            "should_suggest_alternatives": True,
            "expected_alternatives": ["시설 안전 규정", "교내 시설 이용 규정"],
            "should_ask_clarification": False,
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all non-existent regulation edge case templates."""
        return cls.NON_EXISTENT_SCENARIOS


class OutOfScopeQueryTemplates:
    """
    Template collection for out-of-scope queries.

    REQ-P4-004: Tests RAG system's handling of queries outside regulation scope.
    Provides 2 scenarios for non-regulation related queries.
    """

    # Out-of-scope edge cases (2 templates)
    OUT_OF_SCOPE_SCENARIOS = [
        {
            "scenario_id": "edge-outofscope-001",
            "name": "오늘 점심 메뉴",
            "category": EdgeCaseCategory.OUT_OF_SCOPE,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.FRESHMAN,
            "query": "오늘 학식 메뉴 뭐야?",
            "scope_type": "out_of_scope",
            "expected_behavior": "inform_out_of_scope",
            "should_provide_answer": False,
            "expected_message": "규정과 관련 없는 질문입니다",
            "should_suggest_contact": True,
            "expected_contact_suggestion": "학식 관련 부서 문의",
            "should_ask_clarification": False,
        },
        {
            "scenario_id": "edge-outofscope-002",
            "name": "교수님 연락처",
            "category": EdgeCaseCategory.OUT_OF_SCOPE,
            "difficulty": DifficultyLevel.EASY,
            "persona_type": PersonaType.FRESHMAN,
            "query": "김철수 교수님 연락처 알려주세요",
            "scope_type": "out_of_scope",
            "expected_behavior": "inform_out_of_scope",
            "should_provide_answer": False,
            "expected_message": "개인 연락처 정보는 제공할 수 없습니다",
            "should_suggest_contact": True,
            "expected_contact_suggestion": "학과 사무실 문의",
            "should_ask_clarification": False,
        },
    ]

    @classmethod
    def get_all_templates(cls) -> List[Dict[str, Any]]:
        """Get all out-of-scope query edge case templates."""
        return cls.OUT_OF_SCOPE_SCENARIOS


class ExtendedEdgeCaseTemplates:
    """
    Combined edge case templates including all new categories.

    REQ-P4-001 to REQ-P4-004: Provides 10+ edge case scenarios
    for comprehensive RAG system testing.
    """

    @classmethod
    def get_all_extended_templates(cls) -> List[Dict[str, Any]]:
        """Get all extended edge case templates (10+ scenarios)."""
        all_templates = []
        all_templates.extend(TypoEdgeCaseTemplates.get_all_templates())
        all_templates.extend(AmbiguousQueryEdgeCaseTemplates.get_all_templates())
        all_templates.extend(NonExistentRegulationTemplates.get_all_templates())
        all_templates.extend(OutOfScopeQueryTemplates.get_all_templates())
        return all_templates

    @classmethod
    def get_templates_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """Get templates by extended category name."""
        category_mapping = {
            "typo": TypoEdgeCaseTemplates.get_all_templates(),
            "ambiguous": AmbiguousQueryEdgeCaseTemplates.get_all_templates(),
            "non_existent": NonExistentRegulationTemplates.get_all_templates(),
            "out_of_scope": OutOfScopeQueryTemplates.get_all_templates(),
        }
        return category_mapping.get(category, [])
