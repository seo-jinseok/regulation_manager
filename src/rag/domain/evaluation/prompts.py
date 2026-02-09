"""
Enhanced LLM Judge Prompts for RAG Quality Assessment.

This module contains improved prompts for evaluating RAG system responses
with focus on factual consistency and hallucination detection.
"""

from typing import List, Dict, Any


class EvaluationPrompts:
    """
    Enhanced evaluation prompts for RAG quality assessment.

    Improvements over baseline:
    - Strict context adherence instructions
    - Hallucination detection patterns
    - Factual consistency checks
    - Negative examples for clarity
    """

    # System prompt for accuracy evaluation
    ACCURACY_SYSTEM_PROMPT = """
당신은 대학 규정 RAG 시스템의 응답 품질을 평가하는 전문가 판사입니다.

**평가 기준:**
1. 정확성 (Accuracy): 제공된 문맥(검색된 규정)에 기반한 사실적 정확성
2. 완전성 (Completeness): 질문에 대한 필수 정보 포함 여부
3. 인용 (Citations): 규정 참조의 정확성과 완결성
4. 문맥 관련성 (Context Relevance): 검색된 문서의 관련성

**중요 원칙:**
- 오직 제공된 검색 결과(context)에서 확인할 수 있는 정보만 평가
- 문맥에 없는 내용을 "환각(hallucination)"으로 표시
- 불충분한 정보는 "정보 부족"으로 평가하되, "모른다"라고 명시한 경우는 용인
- 규정 인용은 "규정명 + 제X조" 형식이어야 만점

**자동 실패 패턴:**
- 대학명 오류 (예: 서울대, 연세대, 고려대 등 언급)
- 가짜 연락처 (02-XXXX-XXXX 형태의 의심 번호)
- 문맥에 없는 구체적인 정보(날짜, 금액 등)를 마음대로 생성
"""

    # User prompt template for accuracy evaluation
    ACCURACY_USER_PROMPT_TEMPLATE = """
다음 RAG 시스템 응답을 평가해주세요.

**사용자 질문:**
{query}

**RAG 시스템 응답:**
{answer}

**검색된 문맥 (출처):**
{context}

**평가 작업:**
각 지표에 대해 0.0~1.0 점수를 부여하고, 이유를 설명해주세요.

1. **정확성 (Accuracy)**:
   - 1.0: 완벽하게 정확함, 환각 없음
   - 0.8-0.9: 사소한 부정확함
   - 0.5-0.7: 일부 부정확한 정보
   - 0.0-0.4: 환각 또는 심각한 오류

2. **완전성 (Completeness)**:
   - 1.0: 모든 필수 정보 포함
   - 0.8-0.9: 대부분 포함, 사소한 누락
   - 0.6-0.7: 핵심 정보 포함, 일부 누락
   - 0.0-0.5: 중요한 정보 누락

3. **인용 (Citations)**:
   - 1.0: 완벽한 "규정명 + 제X조" 형식
   - 0.8-0.9: 정확한 인용
   - 0.6-0.7: 부분적 인용
   - 0.0-0.5: 인용 없음 또는 부정확

4. **문맥 관련성 (Context Relevance)**:
   - 1.0: 모든 출처가 매우 관련성 높음
   - 0.8-0.9: 높은 관련성
   - 0.6-0.7: 적절한 관련성
   - 0.0-0.5: 낮은 관련성

**JSON 형식으로 응답:**
```json
{{
  "accuracy": <점수>,
  "accuracy_reasoning": "<이유>",
  "completeness": <점수>,
  "completeness_reasoning": "<이유>",
  "citations": <점수>,
  "citations_reasoning": "<이유>",
  "context_relevance": <점수>,
  "context_relevance_reasoning": "<이유>",
  "issues": ["<발견된 문제 목록>"],
  "strengths": ["<장점 목록>"]
}}
```
"""

    # Hallucination detection prompt
    HALLUCINATION_DETECTION_PROMPT = """
다음 텍스트에서 환각(hallucination)을 감지해주세요.

**환각이란:**
- 제공된 문맥에 없는 정보를 생성하는 것
- 구체적인 사실(날짜, 금액, 연락처 등)을 근거 없이 만들어내는 것
- 잘못된 대학명, 부서명, 규정명을 사용하는 것

**텍스트:**
{text}

**문맥:**
{context}

**감지 결과:**
환각이 발견되면 구체적으로 어디서 어떤 환각이 발생했는지 설명하세요.
"""

    # Factual consistency check prompt
    FACTUAL_CONSISTENCY_PROMPT = """
다음 응답의 사실적 일관성을 검증해주세요.

**검증 원칙:**
- 모든 주장(claim)이 문맥에서 지원되어야 함
- 문맥에 없는 세부 정보는 "지원되지 않음"으로 표시
- 모호한 일반화는 "일반화"로 표시

**응답:**
{answer}

**문맥:**
{context}

**검증 결과:**
각 주장별로 문맥에서의 지원 여부를 평가하세요.

JSON 형식:
```json
{{
  "overall_consistency": <0.0-1.0>,
  "claims": [
    {{
      "claim": "<주장>",
      "supported": <true/false>,
      "evidence": "<문맥에서의 근거 또는 '지원되지 않음'>"
    }}
  ],
  "unsupported_claims": ["<지원되지 않는 주장 목록>"]
}}
```
"""

    # Negative examples for training
    NEGATIVE_EXAMPLES = {
        "hallucination": {
            "query": "휴학 신청 방법 알려줘",
            "answer": "휴학 신청은 02-1234-5678로 전화하거나, 서울대 행정처에 방문하여 신청서를 제출해야 합니다.",
            "issues": [
                "가짜 연락처 (02-1234-5678)",
                "잘못된 대학명 (서울대)",
                "문맥에 없는 구체적 정보"
            ]
        },
        "avoidance": {
            "query": "휴학 기간이 어떻게 되나요?",
            "answer": "대학마다 다르니 확인해보세요.",
            "issues": [
                "일반적인 회피 답변",
                "구체적인 정보 제공 부족"
            ]
        },
        "insufficient_citation": {
            "query": "휴학 신청 방법",
            "answer": "학기 시작 전에 신청해야 합니다. 관련 규정을 참고하세요.",
            "issues": [
                "규정명 없음",
                "조항 번호 없음",
                "구체적인 인용 부족"
            ]
        }
    }

    @classmethod
    def format_accuracy_prompt(
        cls,
        query: str,
        answer: str,
        context: List[Dict[str, Any]],
        expected_info: List[str] = None
    ) -> tuple[str, str]:
        """
        Format accuracy evaluation prompt with context.

        Args:
            query: User's question
            answer: RAG system's response
            context: Retrieved documents with metadata
            expected_info: Optional list of expected information points

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format context as readable text
        context_text = cls._format_context(context)

        # Add expected info if provided
        expected_section = ""
        if expected_info:
            expected_section = f"\n**기대되는 정보:**\n" + "\n".join(f"- {info}" for info in expected_info)

        user_prompt = cls.ACCURACY_USER_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            context=context_text + expected_section
        )

        return cls.ACCURACY_SYSTEM_PROMPT, user_prompt

    @classmethod
    def format_hallucination_prompt(
        cls,
        text: str,
        context: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Format hallucination detection prompt.

        Args:
            text: Text to check for hallucinations
            context: Retrieved documents

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context_text = cls._format_context(context)

        user_prompt = cls.HALLUCINATION_DETECTION_PROMPT.format(
            text=text,
            context=context_text
        )

        return "당신은 텍스트에서 환각을 감지하는 전문가입니다.", user_prompt

    @classmethod
    def format_factual_consistency_prompt(
        cls,
        answer: str,
        context: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Format factual consistency check prompt.

        Args:
            answer: Response to verify
            context: Retrieved documents

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context_text = cls._format_context(context)

        user_prompt = cls.FACTUAL_CONSISTENCY_PROMPT.format(
            answer=answer,
            context=context_text
        )

        return "당신은 사실적 일관성을 검증하는 전문가입니다.", user_prompt

    @classmethod
    def _format_context(cls, context: List[Dict[str, Any]]) -> str:
        """
        Format context documents as readable text.

        Args:
            context: List of context documents

        Returns:
            Formatted context string
        """
        if not context:
            return "제공된 문맥 없음"

        lines = []
        for i, doc in enumerate(context, 1):
            title = doc.get("title", "")
            text = doc.get("text", doc.get("content", ""))[:500]  # Limit length
            score = doc.get("score", 0.0)

            lines.append(
                f"[문서 {i}] (관련성: {score:.2f})\n"
                f"제목: {title}\n"
                f"내용: {text}...\n"
            )

        return "\n".join(lines)

    @classmethod
    def get_negative_example(cls, example_type: str) -> Dict[str, Any]:
        """
        Get a negative example for training/reference.

        Args:
            example_type: Type of example ("hallucination", "avoidance", etc.)

        Returns:
            Dictionary with example data
        """
        return cls.NEGATIVE_EXAMPLES.get(example_type, {})

    @classmethod
    def list_negative_examples(cls) -> List[str]:
        """
        List available negative example types.

        Returns:
            List of example type names
        """
        return list(cls.NEGATIVE_EXAMPLES.keys())
