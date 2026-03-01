"""
Persona-Aware Response Generator for RAG System.

Enhances response generation by tailoring prompts to specific user personas.
This improves response quality by matching answer style, detail level, and
language to the user's expertise and preferences.

Integration:
- Used by ParallelPersonaEvaluator for persona-specific evaluation
- Enhances SearchUseCase prompt generation with persona context
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PersonaAwareGenerator:
    """
    Generates persona-aware prompts for RAG response generation.

    Tailors responses to 6 user personas:
    - freshman: Simple, clear explanations for beginners
    - graduate: Comprehensive, academic responses with detailed citations
    - professor: Formal, comprehensive with specific article references
    - staff: Administrative focus with procedures and workflows
    - parent: Parent-friendly language, simple terms, practical guidance
    - international: Mixed Korean/English with English term support
    """

    # Persona-specific prompt enhancements
    PERSONA_PROMPTS: Dict[str, str] = {
        "freshman": """

## 👤 사용자: 신입생 (초보자)
- **언어 수준**: 쉽고 명확하게 전문 용어 최소화
- **상세 수준**: 핵심 내용 위주의 간단한 설명
- **인용 스타일**: 최소한의 규정 인용 (필수时만)
- **답변 톤**: 친절하고 격려적, "처음이라 어려울 수 있어요"와 같은 공감 표현

**답변 지침**:
- 1단계: 핵심 답변을 1-2문장으로 요약
- 2단계: 간단한 절차를 3단계 이내로 번호 매겨 설명
- 3단계: 관련 규정을 간단히 인용
- 4단계: "더 궁금한 점이 있으시면 학사지원팀에 문의하세요" 안내
""",
        "graduate": """

## 👤 사용자: 대학원생
- **언어 수준**: 학술적, 전문적인 용어 사용
- **상세 수준**: 포괄적이고 심층적인 설명
- **인용 스타일**: 상세한 규정 인용 (규정명, 조항, 항까지)
- **답변 톤**: 전문적이고 정중함

**답변 지침**:
- 1단계: 핵심 답변과 관련 조항 명시
- 2단계: 자격 요건, 절차, 필요 서류를 상세히 설명
- 3단계: 예외 사항이나 중요 고려사항 언급
- 4단계: 관련 규정을 구체적으로 인용 (규정명 제X조제Y항)
- 5단계: 추가 문의처 안내 (대학원행정실 등)
""",
        "professor": """

## 👤 사용자: 교수님
- **언어 수준**: 공식적이고 학술적인 표현
- **상세 수준**: 포괄적이고 정확한 법적 해석
- **인용 스타일**: 상세한 인용 with 편/장/조 구체적 근거
- **답변 톤**: 존중하고 정중한 공식어조

**답변 지침**:
- 1단계: 관련 규정의 적용 범위와 대상 명확히
- 2단계: 구체적인 조항 번호와 내용을 인용
- 3단계: 예외 사항, 적용 기준, 해석 포인트 상세히
- 4단계: 조문 형식으로 정확한 인용 (예: 「교원인사규정」제X조제Y항)
- 5단계: 관련 규정 간의 관계나 상충 여부 언급
""",
        "staff": """

## 👤 사용자: 교직원
- **언어 수준**: 행정적 전문 용어 사용
- **상세 수준**: 절차 중심의 실무적 안내
- **인용 스타일**: 표준적 인용 (규정명, 조항)
- **답변 톤**: 업무 지향적이고 명확함

**답변 지침**:
- 1단계: 업무 처리 절차와 담당 부서 명시
- 2단계: 필요 서식/양식과 제출처 안내
- 3단계: 승인 권자와 처리 기한 정보
- 4단계: 관련 규정 인용 (업무 지침 중심)
- 5단계: 주의사항이나 자주 묻는 질문 언급

**REQ-014 규정 준수 강조**:
- 관련 규정 간의 상호 참조를 반드시 포함 (예: 「학칙」 + 「학사관리규정」)
- 법적 근거와 절차적 요구사항을 명확히 구분
- 위반 시 제재 사항이 규정에 있으면 반드시 언급
- 서류 보관 기간, 결재 단계 등 행정 실무 정보 포함
""",
        "parent": """

## 👤 사용자: 학부모
- **언어 수준**: 쉬운 용어로 설명, 전문 용어 풀이
- **상세 수준**: 간단하고 명확하게 실용적 정보
- **인용 스타일**: 최소한의 인용, 이해하기 쉽게
- **답변 톤**: 부모님께 존중하고 친절하게

**답변 지침**:
- 1단계: "학부모님께서 알아두시면 좋은 내용을 안내해 드리겠습니다"와 같은 도입
- 2단계: 학부모 관점에서 중요한 정보 먼저 (비용, 절차, 기한)
- 3단계: 자녀가 해야 할 것 vs 학부모님께서 확인하실 것 구분
- 4단계: 전문 용어은 쉽게 풀이해서 설명
- 5단계: 연락처나 문의처 안내 (학사지원팀 등)
""",
        "international": """

## 👤 사용자: 외국인유학생
- **언어 수준**: 혼합 한국어/English, 중요 용어는 English로 병기
- **상세 수준**: 적절 수준의 상세, 문화적 맥락 설명
- **인용 스타일**: 표준적 인용 (규정명, 조항)
- **답변 톤**: 환영하고 도움됨

**Answer Guidelines**:
- 1단계: 한국어로 답변하되, 중요 용어는 English로 병기
- 2단계: 외국인유학생에게 특히 중요한 정보 (비자, 등록, 언어)
- 3단계: 복잡한 용어은 풀이하여 설명
- 4단계: 필요시 English로 번역 제공
- 5단계: International Student Team 문의 안내

**REQ-015 이중 언어 용어 지원**:
- 핵심 행정 용어를 한국어/영어로 병기: 예) 수강신청(Course Registration), 휴학(Leave of Absence), 졸업(Graduation)
- 비자 관련 용어: 체류자격(Visa Status), 출입국관리사무소(Immigration Office), D-2비자(D-2 Student Visa)
- 학사 용어: 학점(Credits), 성적(GPA/Grades), 등록금(Tuition), 장학금(Scholarship)
- 절차 용어: 신청(Application), 승인(Approval), 제출(Submission), 기한(Deadline)

**For English support**: If the query is in English, provide answer in English while referencing Korean regulations.
""",
    }

    # Persona ID mapping from ParallelPersonaEvaluator to internal persona names
    PERSONA_ID_MAP = {
        "student-undergraduate": "freshman",
        "student-graduate": "graduate",
        "professor": "professor",
        "staff-admin": "staff",
        "parent": "parent",
        "student-international": "international",
    }

    def __init__(self):
        """Initialize the persona-aware generator."""
        self.logger = logging.getLogger(__name__)

    def get_persona_name(self, persona_id: str) -> str:
        """
        Convert ParallelPersonaEvaluator persona ID to internal persona name.

        Args:
            persona_id: Persona ID from evaluator (e.g., "student-undergraduate")

        Returns:
            Internal persona name (e.g., "freshman")
        """
        return self.PERSONA_ID_MAP.get(persona_id, persona_id)

    def enhance_prompt(
        self,
        base_prompt: str,
        persona: str,
        query: Optional[str] = None
    ) -> str:
        """
        Enhance base prompt with persona-specific instructions.

        Args:
            base_prompt: Original system prompt (e.g., REGULATION_QA_PROMPT)
            persona: Persona ID (e.g., "student-undergraduate", "professor")
            query: Optional query for additional context

        Returns:
            Enhanced prompt with persona-specific instructions
        """
        persona_name = self.get_persona_name(persona)
        persona_enhancement = self.PERSONA_PROMPTS.get(persona_name, "")

        if not persona_enhancement:
            self.logger.warning(f"No persona enhancement found for: {persona_name}")
            return base_prompt

        # Append persona-specific instructions to base prompt
        enhanced_prompt = base_prompt + persona_enhancement

        self.logger.debug(
            f"Enhanced prompt for persona '{persona_name}' "
            f"(added {len(persona_enhancement)} characters)"
        )

        return enhanced_prompt

    def generate_custom_prompt(
        self,
        persona: str,
        query: str,
        context: str,
        base_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a completely custom prompt for a persona.

        Args:
            persona: Persona ID
            query: User's query
            context: Retrieved document context
            base_prompt: Optional base prompt to build upon

        Returns:
            Custom prompt string
        """
        if base_prompt is None:
            # Use default base prompt if not provided
            from src.rag.application.search_usecase import REGULATION_QA_PROMPT
            base_prompt = REGULATION_QA_PROMPT

        return self.enhance_prompt(base_prompt, persona, query)

    def supports_persona(self, persona: str) -> bool:
        """
        Check if a persona is supported for enhancement.

        Args:
            persona: Persona ID to check

        Returns:
            True if persona has enhancement defined
        """
        persona_name = self.get_persona_name(persona)
        return persona_name in self.PERSONA_PROMPTS

    def get_supported_personas(self) -> list:
        """
        Get list of supported persona IDs.

        Returns:
            List of persona IDs that have enhancements
        """
        return list(self.PERSONA_ID_MAP.keys())

    def get_persona_preferences(self, persona: str) -> Dict[str, any]:
        """
        Get answer preferences for a persona.

        Args:
            persona: Persona ID

        Returns:
            Dictionary of persona preferences
        """
        from src.rag.domain.evaluation.personas import PERSONAS

        persona_name = self.get_persona_name(persona)
        persona_profile = PERSONAS.get(persona_name)

        if persona_profile:
            return persona_profile.answer_preferences

        self.logger.warning(f"No preferences found for persona: {persona_name}")
        return {}


class PersonaPromptBuilder:
    """
    Helper class for building persona-specific prompts.

    Provides fluent interface for constructing complex prompts
    with multiple persona adjustments.
    """

    def __init__(self, base_prompt: str):
        """
        Initialize prompt builder.

        Args:
            base_prompt: Base system prompt to build upon
        """
        self.base_prompt = base_prompt
        self.enhancements = []
        self.persona_context = {}

    def for_persona(self, persona: str, query: Optional[str] = None) -> "PersonaPromptBuilder":
        """
        Add persona-specific enhancement.

        Args:
            persona: Persona ID
            query: Optional query for context

        Returns:
            Self for fluent chaining
        """
        generator = PersonaAwareGenerator()
        persona_name = generator.get_persona_name(persona)
        preferences = generator.get_persona_preferences(persona)

        self.persona_context = {
            "persona_id": persona,
            "persona_name": persona_name,
            "preferences": preferences,
        }

        enhancement = generator.PERSONA_PROMPTS.get(persona_name, "")
        if enhancement:
            self.enhancements.append(enhancement)

        return self

    def with_completeness_instructions(self) -> "PersonaPromptBuilder":
        """
        Add completeness instructions to prompt.

        Returns:
            Self for fluent chaining
        """
        completeness_instruction = """

## ⚠️ Completeness Requirements (매우 중요)
- **절대 중요한 정보 누락 금지**: 규정에 명시된 기간, 기한, 자격 요건, 절차 등
  핵심 정보는 반드시 포함해야 합니다.
- **불완전한 답변 예시**:
  - ❌ "휴학은 가능합니다." (기간, 절차 누락)
  - ❌ "신청하세요." (방법, 서류 누락)
- **완전한 답변 예시**:
  - ✅ "휴학은 매학기 개시 1개월 전까지 신청해야 합니다 (학칙 제40조제1항).
    신청서는 교내시스템에서 작성하여 지도교수 승인 후 제출합니다."
"""
        self.enhancements.append(completeness_instruction)
        return self

    def with_citation_quality_instructions(self) -> "PersonaPromptBuilder":
        """
        Add citation quality instructions to prompt.

        Returns:
            Self for fluent chaining
        """
        citation_instruction = """

## 📋 Citation Quality Requirements
- **정확한 인용 필수**: 모든 정보는 반드시 규정명과 조항을 함께 인용해야 합니다.
- **인용 형식**: 「규정명」제X조 또는 「규정명」제X조제Y항
- **인용 위치**: 관련 내용을 언급한 직후에 괄호로 인용
- **인용 없는 답변 금지**: 규정 인용 없이 사실만 주장하지 마세요
"""
        self.enhancements.append(citation_instruction)
        return self

    def build(self) -> str:
        """
        Build the final enhanced prompt.

        Returns:
            Enhanced prompt string
        """
        if not self.enhancements:
            return self.base_prompt

        # Combine base prompt with all enhancements
        enhanced = self.base_prompt + "\n".join(self.enhancements)

        return enhanced


# Convenience function for quick usage
def create_persona_prompt(
    base_prompt: str,
    persona: str,
    include_completeness: bool = True,
    include_citation_quality: bool = True,
) -> str:
    """
    Create a persona-enhanced prompt with optional quality instructions.

    Args:
        base_prompt: Base system prompt
        persona: Persona ID
        include_completeness: Whether to add completeness instructions
        include_citation_quality: Whether to add citation quality instructions

    Returns:
        Enhanced prompt string
    """
    builder = PersonaPromptBuilder(base_prompt)
    builder.for_persona(persona)

    if include_completeness:
        builder.with_completeness_instructions()

    if include_citation_quality:
        builder.with_citation_quality_instructions()

    return builder.build()
