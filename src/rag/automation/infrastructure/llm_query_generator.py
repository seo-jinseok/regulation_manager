"""
Query Generator using LLM.

Infrastructure layer implementation that generates diverse queries
for RAG testing using LLM API with fallback to templates.

Clean Architecture: Infrastructure implements domain interfaces and uses
external libraries (anthropic).
"""

import hashlib
import json
import random
from typing import Dict, List, Optional

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    EvaluationCase,
    Persona,
    PersonaType,
    QueryType,
)
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.domain.repositories import ILLMClient


class LLMQueryGenerator:
    """
    Generates test queries with 3-level intent analysis using LLM or templates.

    Each query is tagged with persona, difficulty, query type, and
    includes intent analysis for RAG system validation.

    Attributes:
        llm: LLM client for generating diverse queries
        _use_llm: Flag to enable/disable LLM-based generation
        _seed: Random seed for reproducibility
    """

    # Sample query templates for each persona type (fallback)
    _QUERY_TEMPLATES = {
        PersonaType.FRESHMAN: [
            ("휴학 신청은 어떻게 하나요?", QueryType.PROCEDURAL, DifficultyLevel.EASY),
            (
                "학생회비 내는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
            ("장학금 신청 자격이 뭐야?", QueryType.ELIGIBILITY, DifficultyLevel.MEDIUM),
            ("기숙사 입주 조건 알려줘", QueryType.ELIGIBILITY, DifficultyLevel.MEDIUM),
        ],
        PersonaType.JUNIOR: [
            (
                "졸업 요건이 어떻게 되나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "복수 전공 신청하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교환 학생 갈 수 있는 조건이 뭐야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "대학원 진학하면 어떤 혜택 있어?",
                QueryType.COMPARISON,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.GRADUATE: [
            ("논문 심사 규정 알려주세요", QueryType.PROCEDURAL, DifficultyLevel.HARD),
            (
                "연구비 지원 신청하는 방법이 뭐야?",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            (
                "박사 과정 수료 기준이 어떻게 돼?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "조교 급여 지급 일정 알려줘",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.NEW_PROFESSOR: [
            (
                "교원 평가 기준이 어떻게 되나요?",
                QueryType.FACT_CHECK,
                DifficultyLevel.HARD,
            ),
            (
                "연구년 휴직 신청하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            (
                "지도 학생 수 제한이 있나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교권 보호 관련 규정 알려주세요",
                QueryType.FACT_CHECK,
                DifficultyLevel.HARD,
            ),
        ],
        PersonaType.PROFESSOR: [
            (
                "정년 보장 심사 기준이 뭐야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "학위 과정 수료 기준이 어떻게 돼?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "조교 급여 지급 일정 알려줘",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.NEW_STAFF: [
            (
                "연차 사용하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
            (
                "퇴직금 계산하는 방법이 뭐야?",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
            (
                "야간 근무 수당 지급 기준 알려줘",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            ("복지 혜택 종류 알려주세요", QueryType.FACT_CHECK, DifficultyLevel.EASY),
        ],
        PersonaType.STAFF_MANAGER: [
            (
                "부서 예산 집행 절차가 어떻게 되나요?",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            ("인사 발령 기준 알려주세요", QueryType.FACT_CHECK, DifficultyLevel.HARD),
            ("검토 기한이 며칠이야?", QueryType.FACT_CHECK, DifficultyLevel.MEDIUM),
            (
                "재정 집행 승인 권한이 어디까지야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
        ],
        PersonaType.PARENT: [
            (
                "자녀 휴학하면 등록금 환급받을 수 있나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "성적 장학금 받는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "자녀가 졸업하려면 어떤 조건이 필요한가요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "학부모 상담 예약하는 방법이 뭐야?",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
        ],
        PersonaType.DISTRESSED_STUDENT: [
            (
                "학자금 대출 못 받았어요 어떡하죠?",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "휴학하고 싶은데 마감 지났어요 도와주세요",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "성적이 너무 안 좋아서 장학금 받을 수 있나요?",
                QueryType.COMPLEX,
                DifficultyLevel.HARD,
            ),
            ("간호학과 전공 바꾸고 싶어요", QueryType.COMPLEX, DifficultyLevel.MEDIUM),
        ],
        PersonaType.DISSATISFIED_MEMBER: [
            (
                "수강 신청 시스템 왜 이렇게 복잡해요?",
                QueryType.EMOTIONAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교수님이 성적을 공정하게 매기지 않았어요",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "학교가 학생들 혜택을 줄였잖아요 이거 게 아니죠",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "규정이 자꾸 바뀌는 이유가 뭐야요?",
                QueryType.AMBIGUOUS,
                DifficultyLevel.MEDIUM,
            ),
        ],
    }

    # Regulation topics for context
    _REGULATION_TOPICS = """
    ## 대학 규정 주제 범주

    ### 학사 (Academic Affairs)
    - 휴학, 복학, 자퇴, 재입학
    - 전과, 복수전공, 부전공, 연계전공
    - 교과과정, 수업, 성적 평가
    - 졸업 요건, 학위 수여
    - 출석, 결석, 유고 결석

    ### 장학 (Scholarship)
    - 장학금 종류 (성적, 근로, 복지, 기여)
    - 신청 자격 및 제출 서류
    - 성적 기준, 소득 기준
    - 이중 수급 제한
    - 지급 일정 및 방법

    ### 복지 (Student Welfare)
    - 기숙사 입주 조건, 신청, 비용
    - 식당 이용, 식권, 결제
    - 학생회비 납부 및 용도
    - 보건센터, 심리상담
    - 학생 단체, 동아리

    ### 교무 (Academic Administration)
    - 수강 신청, 변경, 취득 실패
    - 계절 학기, 여름/겨울 학기
    - 교환 학생, 파견 학생
    - 교직 이수, 교원 자격증

    ### 연구 (Research)
    - 연구비 지원, 과제 신청
    - 논문 심사, 학위 청구
    - 지도 교수, 연구 지도
    - 학술지 게재, 발표

    ### 인사 (Human Resources - Staff)
    - 채용, 임용, 승진
    - 복무, 평가, 상벌
    - 급여, 수당, 연차
    - 정년 보장, 퇴직금

    ### 교원 (Faculty)
    - 교원 임용, 승급, 정년
    - 교원 평가, 업적 평가
    - 연구년, 안식년
    - 조교 지급, 연구비
    """

    def __init__(self, llm_client: Optional[ILLMClient] = None):
        """
        Initialize the LLM query generator.

        Args:
            llm_client: LLM client for generating diverse queries.
                      If None, operates in template-only mode.
        """
        self.llm = llm_client
        self._use_llm = llm_client is not None
        self._seed: Optional[int] = None

    def generate_for_persona(
        self,
        persona: Persona,
        count_per_difficulty: Dict[str, int],
        vary_queries: bool = True,
        seed: Optional[int] = None,
    ) -> List[EvaluationCase]:
        """
        Generate test queries for a specific persona.

        Args:
            persona: The Persona to generate queries for.
            count_per_difficulty: Dict with 'easy', 'medium', 'hard' counts.
            vary_queries: If True, use LLM for diverse queries; if False, use templates.
            seed: Random seed for reproducibility (None for random).

        Returns:
            List of EvaluationCase entities with queries and intent analysis.
        """
        self._set_seed(seed)

        # Use LLM only if available and vary_queries is True
        if vary_queries and self._use_llm and self.llm is not None:
            return self._generate_with_llm(persona, count_per_difficulty, seed)
        else:
            return self._generate_from_templates(persona, count_per_difficulty)

    def _generate_with_llm(
        self,
        persona: Persona,
        count_per_difficulty: Dict[str, int],
        seed: Optional[int] = None,
    ) -> List[EvaluationCase]:
        """
        Generate diverse queries using LLM.

        Args:
            persona: The Persona to generate queries for.
            count_per_difficulty: Dict with 'easy', 'medium', 'hard' counts.

        Returns:
            List of EvaluationCase entities generated by LLM.
        """
        if self.llm is None:
            # Fallback to templates if no LLM client
            return self._generate_from_templates(persona, count_per_difficulty)

        system_prompt = self._build_system_prompt(persona, count_per_difficulty)

        # Temperature based on seed for reproducibility
        temperature = self._get_temperature(seed)

        response = self.llm.generate(system_prompt, "", temperature)

        return self._parse_llm_response(response, persona, count_per_difficulty)

    def _build_system_prompt(
        self, persona: Persona, count_per_difficulty: Dict[str, int]
    ) -> str:
        """
        Build system prompt for LLM query generation.

        Args:
            persona: The Persona context.
            count_per_difficulty: Requested counts per difficulty level.

        Returns:
            System prompt string.
        """
        prompt = f"""당신은 대학 규정 검색 시스템을 테스트하기 위한 질문 생성 전문가입니다.

## 페르소나 정보
- **이름**: {persona.name}
- **설명**: {persona.description}
- **성격 특성**: {", ".join(persona.characteristics) if persona.characteristics else "없음"}
- **질문 스타일**: {", ".join(persona.query_styles) if persona.query_styles else "일반적"}

{self._REGULATION_TOPICS}

## 생성 요청
다음 난이도별로 질문을 생성해주세요:

- **Easy** ({count_per_difficulty.get("easy", 0)}개): 단순 정보 조회, 명확한 키워드, 단일 규정 참조
- **Medium** ({count_per_difficulty.get("medium", 0)}개): 절차/자격 확인, 여러 규정 연계 필요
- **Hard** ({count_per_difficulty.get("hard", 0)}개): 추론 필요, 여러 문서 참조, 모호하거나 감정적 표현

## 질문 유형 가이드
- **사실 확인 (fact_check)**: 구체적 사실 확인
- **절차 질문 (procedural)**: 신청 방법, 절차 문의
- **자격 확인 (eligibility)**: 자격 요건, 조건 문의
- **비교 질문 (comparison)**: 여러 옵션 비교
- **모호한 질문 (ambiguous)**: 불명확한 표현
- **감정적 질문 (emotional)**: 감정 표현 포함
- **복합 질문 (complex)**: 여러 요소 결합

## 참고용 예시 (이것과 다른 새로운 질문을 만드세요)
{self._get_template_examples(persona.persona_type)}

## 출력 형식
반드시 아래 JSON 배열 형식으로만 출력해주세요. 다른 텍스트는 포함하지 마세요:

```json
[
  {{
    "query": "질문 텍스트",
    "type": "fact_check|procedural|eligibility|comparison|ambiguous|emotional|complex",
    "difficulty": "easy|medium|hard"
  }}
]
```

## 주의사항
1. 질문은 실제 대학 구성원이 할 법한 자연스러운 표현이어야 합니다
2. 페르소나의 특성에 맞는 질문 스타일을 사용하세요
3. 각 질문은 서로 다른 주제를 다루도록 다양성을 확보하세요
4. 참고용 예시와는 다른 새로운 질문을 생성하세요
5. 반드시 유효한 JSON 배열 형식으로만 출력하세요
"""
        return prompt

    def _get_template_examples(self, persona_type: PersonaType) -> str:
        """Get template examples as reference for LLM."""
        templates = self._QUERY_TEMPLATES.get(persona_type, [])
        examples = []
        for query, query_type, difficulty in templates[:3]:  # Show first 3
            examples.append(f'- "{query}" ({query_type.value}, {difficulty.value})')
        return "\n".join(examples) if examples else "예시 없음"

    def _parse_llm_response(
        self, response: str, persona: Persona, count_per_difficulty: Dict[str, int]
    ) -> List[EvaluationCase]:
        """
        Parse LLM response into EvaluationCase entities.

        Args:
            response: LLM response string (JSON array).
            persona: The Persona context.
            count_per_difficulty: Expected counts per difficulty.

        Returns:
            List of EvaluationCase entities.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            queries_data = json.loads(response)

            test_cases = []
            difficulty_map = {
                "easy": DifficultyLevel.EASY,
                "medium": DifficultyLevel.MEDIUM,
                "hard": DifficultyLevel.HARD,
            }
            type_map = {
                "fact_check": QueryType.FACT_CHECK,
                "procedural": QueryType.PROCEDURAL,
                "eligibility": QueryType.ELIGIBILITY,
                "comparison": QueryType.COMPARISON,
                "ambiguous": QueryType.AMBIGUOUS,
                "emotional": QueryType.EMOTIONAL,
                "complex": QueryType.COMPLEX,
            }

            for q_data in queries_data:
                query = q_data.get("query", "")
                type_str = q_data.get("type", "fact_check")
                diff_str = q_data.get("difficulty", "medium")

                try:
                    query_type = type_map.get(type_str, QueryType.FACT_CHECK)
                    difficulty = difficulty_map.get(diff_str, DifficultyLevel.MEDIUM)

                    intent = self._generate_intent_analysis(query, persona)

                    test_case = EvaluationCase(
                        query=query,
                        persona_type=persona.persona_type,
                        difficulty=difficulty,
                        query_type=query_type,
                        intent_analysis=intent,
                    )
                    test_cases.append(test_case)
                except (KeyError, ValueError):
                    # Skip invalid entries
                    continue

            return (
                test_cases
                if test_cases
                else self._generate_from_templates(persona, count_per_difficulty)
            )

        except json.JSONDecodeError:
            # Fallback to templates if JSON parsing fails
            return self._generate_from_templates(persona, count_per_difficulty)

    def _generate_from_templates(
        self, persona: Persona, count_per_difficulty: Dict[str, int]
    ) -> List[EvaluationCase]:
        """
        Generate queries from predefined templates (fallback).

        Args:
            persona: The Persona to generate queries for.
            count_per_difficulty: Dict with 'easy', 'medium', 'hard' counts.

        Returns:
            List of EvaluationCase entities from templates.
        """
        templates = self._QUERY_TEMPLATES.get(persona.persona_type, [])
        test_cases = []

        for difficulty_str, count in count_per_difficulty.items():
            difficulty = DifficultyLevel(difficulty_str)
            difficulty_templates = [t for t in templates if t[2] == difficulty]

            if not difficulty_templates:
                difficulty_templates = templates

            for _ in range(count):
                if not difficulty_templates:
                    break

                query, query_type, _ = random.choice(difficulty_templates)
                intent = self._generate_intent_analysis(query, persona)

                test_case = EvaluationCase(
                    query=query,
                    persona_type=persona.persona_type,
                    difficulty=difficulty,
                    query_type=query_type,
                    intent_analysis=intent,
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_intent_analysis(self, query: str, persona: Persona) -> IntentAnalysis:
        """
        Generate 3-level intent analysis for a query.

        Args:
            query: The user query.
            persona: The persona context.

        Returns:
            IntentAnalysis with surface, hidden, and behavioral intent.
        """
        surface_intent = self._extract_surface_intent(query)
        hidden_intent = self._infer_hidden_intent(query, persona)
        behavioral_intent = self._infer_behavioral_intent(query, persona)

        return IntentAnalysis(
            surface_intent=surface_intent,
            hidden_intent=hidden_intent,
            behavioral_intent=behavioral_intent,
        )

    def _extract_surface_intent(self, query: str) -> str:
        """Extract surface intent from query."""
        if "신청" in query or "방법" in query:
            return "절차/신청 문의"
        elif "자격" in query or "조건" in query or "기준" in query:
            return "자격/요건 확인"
        elif "알려줘" in query or "어떻게" in query:
            return "정보 요청"
        elif "왜" in query or "이유" in query:
            return "원인/근거 문의"
        elif "안 돼요" in query or "없어요" in query or "어떡하죠" in query:
            return "불만/도움 요청"
        else:
            return "일반 문의"

    def _infer_hidden_intent(self, query: str, persona: Persona) -> str:
        """Infer hidden intent from query and persona context."""
        persona_intents = {
            PersonaType.FRESHMAN: "학교 시스템 적응 필요",
            PersonaType.JUNIOR: "졸업/진로 준비",
            PersonaType.GRADUATE: "연구/학위 진행",
            PersonaType.NEW_PROFESSOR: "교수 업무 수행",
            PersonaType.PROFESSOR: "교수 업무 수행",
            PersonaType.NEW_STAFF: "직무 수행",
            PersonaType.STAFF_MANAGER: "직무 수행",
            PersonaType.PARENT: "자녀 교육 지원",
            PersonaType.DISTRESSED_STUDENT: "긴급 상황 해결 필요",
            PersonaType.DISSATISFIED_MEMBER: "권리 주장 및 불만 해소",
        }

        intent_suffix = persona_intents.get(persona.persona_type, "정보 필요")
        return f"{query} - {intent_suffix}"

    def _infer_behavioral_intent(self, query: str, persona: Persona) -> str:
        """Infer ultimate behavioral intent."""
        if "신청" in query:
            return "신청서 제출"
        elif "방법" in query:
            return "절차 수행"
        elif "자격" in query or "조건" in query:
            return "자격 확인 후 행동"
        elif "알려줘" in query:
            return "정보 습득"
        elif "불만" in query or "별로" in query or "없어요" in query:
            return "문제 해결 또는 대안 요청"
        else:
            return "정보 확인"

    def _get_temperature(self, seed: Optional[int]) -> float:
        """
        Get temperature value based on seed for diversity control.

        - No seed (None): High randomness (0.9) for maximum diversity
        - With seed: Deterministic temperature in range [0.5, 0.89]

        Args:
            seed: Random seed (None for random, int for deterministic).

        Returns:
            Temperature value for LLM sampling.
        """
        if seed is None:
            return 0.9  # Maximum diversity for random generation

        # Generate deterministic temperature from seed
        # Using hash to distribute temperatures evenly across [0.5, 0.89]
        hash_val = int(hashlib.md5(str(seed).encode()).hexdigest(), 16)
        return 0.5 + (hash_val % 40) / 100.0  # 0.50 to 0.89

    def _set_seed(self, seed: Optional[int]) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value (None for no seeding).
        """
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def _make_cache_key(
        self, persona: Persona, counts: Dict[str, int], seed: Optional[int]
    ) -> str:
        """
        Generate cache key for storing generated queries.

        Args:
            persona: The Persona context.
            counts: Query counts per difficulty.
            seed: Random seed used.

        Returns:
            MD5 hash cache key.
        """
        key_data = f"{persona.persona_type.value}_{counts}_{seed}"
        return hashlib.md5(key_data.encode()).hexdigest()

    @property
    def use_llm(self) -> bool:
        """Get LLM usage flag."""
        return self._use_llm

    @use_llm.setter
    def use_llm(self, value: bool) -> None:
        """Set LLM usage flag."""
        self._use_llm = value


# Backward compatibility: keep class alias
QueryGenerator = LLMQueryGenerator
