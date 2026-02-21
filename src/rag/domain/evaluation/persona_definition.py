"""
Persona Definition for RAG Quality Evaluation.

Defines user personas with language preferences, citation styles, and key requirements
for persona-specific response evaluation.

SPEC-RAG-QUALITY-010 Milestone 6: Persona Evaluation System.

Clean Architecture: Domain layer contains persona definitions and business logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class PersonaType(Enum):
    """
    Types of user personas for evaluation.

    Each persona represents a different user group with specific
    requirements for response quality and style.
    """

    FRESHMAN = "freshman"  # 신입생 - First-year student
    STUDENT = "student"  # 재학생 - Enrolled student
    PROFESSOR = "professor"  # 교수 - Faculty member
    STAFF = "staff"  # 직원 - Administrative staff
    PARENT = "parent"  # 학부모 - Parent
    INTERNATIONAL = "international"  # 외국인 유학생 - International student


@dataclass
class PersonaDefinition:
    """
    Definition of a user persona for evaluation.

    Defines characteristics and requirements for persona-specific
    response quality evaluation.

    Attributes:
        persona_id: Unique identifier for the persona (e.g., "freshman")
        name: Korean display name (e.g., "신입생")
        description: Description of the persona's characteristics
        language_level: Expected language complexity
            - "simple": Simple, easy-to-understand language
            - "normal": Standard academic language
            - "formal": Formal administrative language
            - "technical": Technical/expert-level language
        citation_preference: Expected citation style
            - "minimal": Minimal citations, just key references
            - "normal": Standard citation level
            - "detailed": Detailed citations with article references
        key_requirements: List of key requirements for this persona
    """

    persona_id: str
    name: str
    description: str
    language_level: str  # simple, normal, formal, technical
    citation_preference: str  # minimal, normal, detailed
    key_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "language_level": self.language_level,
            "citation_preference": self.citation_preference,
            "key_requirements": self.key_requirements,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "PersonaDefinition":
        """Create from dictionary."""
        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            description=data["description"],
            language_level=data["language_level"],
            citation_preference=data["citation_preference"],
            key_requirements=data.get("key_requirements", []),
        )


# Default persona definitions as specified in SPEC-RAG-QUALITY-010
DEFAULT_PERSONAS: Dict[str, PersonaDefinition] = {
    "freshman": PersonaDefinition(
        persona_id="freshman",
        name="신입생",
        description="대학 규정을 처음 접하는 1학년 학생",
        language_level="simple",
        citation_preference="minimal",
        key_requirements=[
            "간단명료한 답변",
            "최소 인용",
            "친절한 설명",
        ],
    ),
    "student": PersonaDefinition(
        persona_id="student",
        name="재학생",
        description="일반 학부생",
        language_level="normal",
        citation_preference="normal",
        key_requirements=[
            "절차 중심",
            "구체적 안내",
            "실용적 정보",
        ],
    ),
    "professor": PersonaDefinition(
        persona_id="professor",
        name="교수",
        description="교원 대상 규정",
        language_level="technical",
        citation_preference="detailed",
        key_requirements=[
            "정책/규정 중심",
            "전문 용어",
            "조항 인용",
        ],
    ),
    "staff": PersonaDefinition(
        persona_id="staff",
        name="직원",
        description="행정 담당자",
        language_level="formal",
        citation_preference="normal",
        key_requirements=[
            "행정 절차",
            "담당 부서 정보",
            "처리 기한",
        ],
    ),
    "parent": PersonaDefinition(
        persona_id="parent",
        name="학부모",
        description="학생 부모님",
        language_level="simple",
        citation_preference="minimal",
        key_requirements=[
            "친절한 설명",
            "연락처 포함",
            "이해하기 쉬운 용어",
        ],
    ),
    "international": PersonaDefinition(
        persona_id="international",
        name="외국인 유학생",
        description="한국어 비원어민",
        language_level="simple",
        citation_preference="normal",
        key_requirements=[
            "간단한 한국어",
            "복잡한 용어 설명",
            "시각적 안내",
        ],
    ),
}
