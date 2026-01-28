"""
Domain Entities for RAG Testing Automation System.

Entities are the core business objects that encapsulate the most general
and high-level rules. They are the least likely to change when something
external changes.

Clean Architecture: Domain layer uses only standard library (dataclasses).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .value_objects import IntentAnalysis


class PersonaType(Enum):
    """10 types of user personas for testing RAG system."""

    FRESHMAN = "freshman"  # 신입생: 학교 시스템에 익숙하지 않음
    JUNIOR = "junior"  # 재학생 (3학년): 졸업 준비
    GRADUATE = "graduate"  # 대학원생: 연구/논문 중심
    NEW_PROFESSOR = "new_professor"  # 신임 교수: 제도 파악 필요
    PROFESSOR = "professor"  # 정교수: 세부 규정 확인
    NEW_STAFF = "new_staff"  # 신입 직원: 복무규정 파악
    STAFF_MANAGER = "staff_manager"  # 과장급 직원: 부서 운영
    PARENT = "parent"  # 학부모: 자녀 관련 정보
    DISTRESSED_STUDENT = "distressed_student"  # 어려운 상황의 학생: 감정적
    DISSATISFIED_MEMBER = "dissatisfied_member"  # 불만있는 구성원: 권리 주장


class QueryType(Enum):
    """Types of queries for testing."""

    FACT_CHECK = "fact_check"  # 사실 확인
    PROCEDURAL = "procedural"  # 절차 질문
    ELIGIBILITY = "eligibility"  # 자격 확인
    COMPARISON = "comparison"  # 비교 질문
    AMBIGUOUS = "ambiguous"  # 모호한 질문
    EMOTIONAL = "emotional"  # 감정 표현
    COMPLEX = "complex"  # 복합 질문
    SLANG = "slang"  # 은어/축약어


class DifficultyLevel(Enum):
    """Difficulty levels for test queries."""

    EASY = "easy"  # 단일 규정, 명확한 키워드
    MEDIUM = "medium"  # 여러 규정 연계
    HARD = "hard"  # 모호한 표현, 감정적


@dataclass
class Persona:
    """
    A user persona for RAG testing.

    Represents different types of users with their characteristics
    and query patterns.
    """

    persona_type: PersonaType
    name: str  # e.g., "신입생"
    description: str  # e.g., "학교 시스템에 익숙하지 않음, 비공식적 표현"
    characteristics: List[str] = field(default_factory=list)
    query_styles: List[str] = field(default_factory=list)
    context_hints: List[str] = field(default_factory=list)


@dataclass
class EvaluationCase:
    """
    A single evaluation case for RAG testing.

    Contains a query with its metadata including persona, difficulty,
    and intent analysis.
    """

    query: str
    persona_type: PersonaType
    difficulty: DifficultyLevel
    query_type: QueryType
    intent_analysis: Optional["IntentAnalysis"] = None
    expected_topics: List[str] = field(default_factory=list)
    expected_regulations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSession:
    """
    A test session containing multiple test cases.

    Represents a complete testing run with metadata and results.
    """

    session_id: str
    started_at: datetime
    total_test_cases: int
    test_cases: List["EvaluationCase"]
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        """Check if the session is completed."""
        return self.completed_at is not None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class QualityTestResult:
    """
    Result of executing a single evaluation case.

    Contains the query, answer, and evaluation metrics.
    """

    test_case_id: str  # Reference to EvaluationCase
    query: str
    answer: str  # Generated answer
    sources: List[str]  # Source regulation references
    confidence: float  # Confidence score from RAG system
    execution_time_ms: int  # Execution time in milliseconds
    rag_pipeline_log: Dict[str, Any]  # RAG pipeline execution details
    fact_checks: List[Any] = field(default_factory=list)  # FactCheck results
    quality_score: Optional[Any] = None  # QualityScore result
    passed: bool = False  # Overall pass/fail
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None  # Error if execution failed


class FollowUpType(Enum):
    """8 types of follow-up questions for multi-turn conversation testing."""

    CLARIFICATION = "clarification"  # 구체화: 이전 답변의 특정 부분 명확히
    RELATED_EXPANSION = "related_expansion"  # 관련 확장: 관련 주제로 확장
    EXCEPTION_CHECK = "exception_check"  # 예외 확인: 예외 사항 확인
    PROCEDURAL_DEEPENING = "procedural_deepening"  # 절차 심화: 절차 상세 질문
    CONDITION_CHANGE = "condition_change"  # 조건 변경: 조건 변화 후 재질문
    CONFIRMATION = "confirmation"  # 확인 질문: 이해 확인
    GO_BACK = "go_back"  # 되돌아가기: 이전 주제로 복귀
    COMPARISON = "comparison"  # 비교 요청: 여러 옵션 비교


@dataclass
class Turn:
    """
    A single turn in a multi-turn conversation.

    Contains the query, response, and metadata for that turn.
    """

    turn_number: int  # Turn number (1-indexed)
    query: str  # User query for this turn
    answer: str  # System response
    sources: List[str]  # Source references
    confidence: float  # Confidence score
    follow_up_type: Optional[FollowUpType] = None  # Type of follow-up if not first turn
    intent_evolution: Optional[str] = None  # How intent evolved from previous turn
    implicit_info_extracted: List[str] = field(
        default_factory=list
    )  # Implicit information extracted
    context_preserved: bool = (
        True  # Whether system maintained context from previous turns
    )


@dataclass
class MultiTurnScenario:
    """
    A multi-turn conversation scenario for testing RAG context management.

    Contains a sequence of turns that test the system's ability to maintain
    context across multiple interactions.
    """

    scenario_id: str  # Unique identifier
    persona_type: PersonaType  # User persona
    initial_query: str  # First query in the conversation
    turns: List[Turn]  # All turns in the conversation
    difficulty: DifficultyLevel  # Overall difficulty
    context_window_size: int = 3  # Number of previous turns to consider as context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        """Get total number of turns."""
        return len(self.turns)

    @property
    def context_preservation_rate(self) -> float:
        """Calculate context preservation rate across all turns."""
        if not self.turns:
            return 1.0
        preserved_count = sum(1 for turn in self.turns if turn.context_preserved)
        return preserved_count / len(self.turns)

    @property
    def follow_up_distribution(self) -> Dict[FollowUpType, int]:
        """Get distribution of follow-up question types."""
        distribution: Dict[FollowUpType, int] = {}
        for turn in self.turns:
            if turn.follow_up_type:
                distribution[turn.follow_up_type] = (
                    distribution.get(turn.follow_up_type, 0) + 1
                )
        return distribution


@dataclass
class ContextHistory:
    """
    Context tracking for multi-turn conversations.

    Maintains history of queries, answers, and implicit information
    to support context-aware follow-up generation.
    """

    scenario_id: str  # Reference to MultiTurnScenario
    conversation_history: List[Turn]  # All turns in conversation
    implicit_entities: Dict[str, Any] = field(
        default_factory=dict
    )  # Implicitly extracted entities
    topic_transitions: List[str] = field(
        default_factory=list
    )  # Topic transition history
    intent_history: List[str] = field(default_factory=list)  # Intent evolution history

    def get_recent_context(self, window_size: int = 3) -> List[Turn]:
        """Get recent turns within context window."""
        return self.conversation_history[-window_size:]

    def add_turn(self, turn: Turn) -> None:
        """Add a new turn to context history."""
        self.conversation_history.append(turn)
        if turn.intent_evolution:
            self.intent_history.append(turn.intent_evolution)


# Type aliases for backward compatibility with tests
TestResult = QualityTestResult
TestCase = EvaluationCase
