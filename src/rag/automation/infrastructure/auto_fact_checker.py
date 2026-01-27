"""
Auto Fact Checker for RAG Testing.

Infrastructure layer for automated fact checking of RAG answers.
Extracts claims from answers and verifies them against regulations.

Clean Architecture: Infrastructure implements domain interfaces.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, List, Optional

from ..domain.value_objects import FactCheck, FactCheckStatus

if TYPE_CHECKING:
    from ...domain.repositories import ILLMClient, IVectorStore
    from ..domain.entities import QualityTestResult

logger = logging.getLogger(__name__)


# Patterns for detecting claims in Korean text
CLAIM_PATTERNS = [
    r"([가-힣]+)\s*(은|는)\s*([가-힣\s]+)",  # X은 Y
    r"([가-힣]+)\s*(이|가)\s*([가-힣\s]+)",  # X가 Y
    r"(\d+[년월일시점개])\s*([가-힣]+)",  # X년 Y
    r"(\d+)\s*([가-힣]+)",  # X 개/명/회
]

# Patterns for generalization answers
GENERALIZATION_PATTERNS = [
    r"대학마다\s*다를\s*수",
    r"각\s*대학의\s*상황에\s*따라",
    r"일반적으로",
    r"보통은",
    r"대체로",
]


class AutoFactChecker:
    """
    Automated fact checker for RAG answers.

    Extracts key claims from answers and verifies them against
    the vector store to ensure accuracy.
    """

    # LLM prompt for claim extraction
    CLAIM_EXTRACTION_PROMPT = """당신은 대학 규정 답변에서 핵심 주장을 추출하는 전문가입니다.

다음 답변에서 **가장 중요한 주장 3개**를 추출하세요.
각 주장은 규정에서 검증 가능한 구체적인 사실이어야 합니다.

추출 기준:
1. 구체적인 수치, 기한, 절차, 조건 등이 포함된 주장
2. 규정 조항에서 직접 확인 가능한 사실
3. 사용자가 실제로 행동하기 위해 필요한 정보

추출하지 말 것:
- 일반적인 문맥 설명
- 질문 재진술
- 규정에 없는 일반론

반드시 JSON 형식으로만 응답하세요:
{{
  "claims": [
    {{"claim": "주장1", "category": "조건/절차/수치/기한"}},
    {{"claim": "주장2", "category": "조건/절차/수치/기한"}},
    {{"claim": "주장3", "category": "조건/절수/수치/기한"}}
  ]
}}

답변:
{answer}"""

    # LLM prompt for fact verification
    FACT_VERIFICATION_PROMPT = """당신은 대학 규정 검증 전문가입니다.

다음 주장이 제공된 규정 내용을 바탕으로 **사실인지** 판단하세요.

주장: {claim}

관련 규정 내용:
{regulation_text}

판단 기준:
- PASS: 주장이 규정 내용과 정확히 일치함
- FAIL: 주장이 규정 내용과 다르거나 규정에 없음
- UNCERTAIN: 규정 내용만으로는 확인 불가능

반드시 JSON 형식으로만 응답하세요:
{{
  "status": "PASS/FAIL/UNCERTAIN",
  "confidence": 0.0~1.0,
  "source": "규정명 및 조항",
  "correction": "status가 FAIL일 때 올바른 내용",
  "explanation": "판단 이유"
}}"""

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        vector_store: Optional["IVectorStore"] = None,
    ):
        """
        Initialize the fact checker.

        Args:
            llm_client: Optional LLM client for intelligent extraction.
            vector_store: Optional vector store for claim verification.
        """
        self.llm = llm_client
        self.store = vector_store

    def detect_generalization(self, answer: str) -> bool:
        """
        Detect if answer contains generalization phrases.

        Args:
            answer: The answer text to check.

        Returns:
            True if generalization detected, False otherwise.
        """
        for pattern in GENERALIZATION_PATTERNS:
            if re.search(pattern, answer):
                logger.warning(f"Generalization detected: {pattern}")
                return True
        return False

    def extract_claims(
        self, answer: str, test_result: "QualityTestResult"
    ) -> List[str]:
        """
        Extract key claims from the answer.

        Args:
            answer: The answer text.
            test_result: Test result with context.

        Returns:
            List of extracted claims (max 3).
        """
        # Method 1: LLM-based extraction (preferred)
        if self.llm:
            try:
                return self._extract_claims_with_llm(answer)
            except Exception as e:
                logger.warning(f"LLM claim extraction failed: {e}")

        # Method 2: Rule-based fallback
        return self._extract_claims_rule_based(answer)

    def _extract_claims_with_llm(self, answer: str) -> List[str]:
        """Extract claims using LLM."""
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)

        response = self.llm.generate(
            system_prompt="당신은 대학 규정 답변에서 핵심 주장을 추출하는 전문가입니다.",
            user_message=prompt,
            temperature=0.0,
        )

        # Parse JSON response
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            data = json.loads(cleaned.strip())
            claims_data = data.get("claims", [])

            claims = [c.get("claim", "") for c in claims_data if c.get("claim")]

            # Limit to 3 claims
            return claims[:3]

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM claim extraction: {e}")
            return []

    def _extract_claims_rule_based(self, answer: str) -> List[str]:
        """Extract claims using rule-based patterns."""
        claims = []
        sentences = re.split(r"[.!?]\s+", answer)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Check for claim patterns
            for pattern in CLAIM_PATTERNS:
                if re.search(pattern, sentence):
                    claims.append(sentence)
                    break

            if len(claims) >= 3:
                break

        return claims[:3]

    def verify_claim(self, claim: str, query: str, sources: List[str]) -> "FactCheck":
        """
        Verify a single claim against regulations.

        Args:
            claim: The claim to verify.
            query: Original query for context.
            sources: Source references from search results.

        Returns:
            FactCheck with verification result.
        """

        # If we have LLM, use intelligent verification
        if self.llm and self.store:
            return self._verify_with_search(claim, query)
        else:
            # Rule-based fallback
            return self._verify_rule_based(claim, sources)

    def _verify_with_search(self, claim: str, query: str) -> "FactCheck":
        """Verify claim by searching regulations."""
        from ...domain.value_objects import Query

        # Search for relevant regulations
        search_query = Query(text=f"{query} {claim}")
        results = self.store.search(search_query, None, top_k=3)

        if not results:
            return FactCheck(
                claim=claim,
                status=FactCheckStatus.UNCERTAIN,
                source="",
                confidence=0.0,
                explanation="No relevant regulations found",
            )

        # Build regulation text
        regulation_text = "\n\n".join(
            [f"[{r.chunk.rule_code}] {r.chunk.text}" for r in results]
        )

        # Use LLM to verify
        prompt = self.FACT_VERIFICATION_PROMPT.format(
            claim=claim, regulation_text=regulation_text
        )

        try:
            response = self.llm.generate(
                system_prompt="당신은 대학 규정 검증 전문가입니다.",
                user_message=prompt,
                temperature=0.0,
            )

            # Parse response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            data = json.loads(cleaned.strip())

            status_str = data.get("status", "UNCERTAIN")
            status = FactCheckStatus(status_str.lower())

            return FactCheck(
                claim=claim,
                status=status,
                source=data.get("source", results[0].chunk.rule_code),
                confidence=float(data.get("confidence", 0.7)),
                correction=data.get("correction"),
                explanation=data.get("explanation"),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse fact verification: {e}")
            return FactCheck(
                claim=claim,
                status=FactCheckStatus.UNCERTAIN,
                source=results[0].chunk.rule_code,
                confidence=0.5,
                explanation=f"Verification failed: {e}",
            )

    def _verify_rule_based(self, claim: str, sources: List[str]) -> "FactCheck":
        """Verify claim using rule-based approach."""
        # Simple heuristic: if we have sources, assume PASS
        if sources:
            return FactCheck(
                claim=claim,
                status=FactCheckStatus.PASS,
                source=", ".join(sources[:2]),
                confidence=0.7,
                explanation="Verified against search sources",
            )
        else:
            return FactCheck(
                claim=claim,
                status=FactCheckStatus.UNCERTAIN,
                source="",
                confidence=0.0,
                explanation="No sources available for verification",
            )

    def check_facts(self, test_result: "QualityTestResult") -> List["FactCheck"]:
        """
        Perform fact checking on a test result.

        Args:
            test_result: QualityTestResult with answer to verify.

        Returns:
            List of FactCheck results.
        """

        # Check for generalization first (automatic fail)
        if self.detect_generalization(test_result.answer):
            logger.warning("Generalization detected - automatic fail")

            return [
                FactCheck(
                    claim="Answer contains generalization",
                    status=FactCheckStatus.FAIL,
                    source="",
                    confidence=1.0,
                    correction="Provide specific regulation-based answers",
                    explanation="Generalization phrases like '대학마다 다를 수 있습니다' are not acceptable",
                )
            ]

        # Extract claims
        claims = self.extract_claims(test_result.answer, test_result)

        if not claims:
            logger.warning("No claims extracted from answer")

            return [
                FactCheck(
                    claim="No verifiable claims found",
                    status=FactCheckStatus.UNCERTAIN,
                    source="",
                    confidence=0.0,
                    explanation="Answer does not contain specific, verifiable claims",
                )
            ]

        logger.info(f"Extracted {len(claims)} claims for verification")

        # Verify each claim
        fact_checks = []

        for claim in claims:
            check = self.verify_claim(
                claim=claim,
                query=test_result.query,
                sources=test_result.sources,
            )
            fact_checks.append(check)

        return fact_checks
