"""
Mock LLM Client for RAG Testing Automation.

Provides mock responses for testing without actual API calls.
"""

from typing import List

from src.rag.domain.repositories import ILLMClient


class MockLLMClientForQueryGen(ILLMClient):
    """
    Mock LLM client specifically for query generation testing.

    Returns JSON-formatted query lists as expected by LLMQueryGenerator.
    """

    def __init__(self, use_template_responses: bool = True):
        """
        Initialize the mock client.

        Args:
            use_template_responses: If True, returns realistic template-like responses.
                                  If False, returns simple mock responses.
        """
        self.use_template_responses = use_template_responses
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Get the number of times generate() was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate mock response for query generation.

        Returns JSON array of queries matching the expected format.

        Args:
            system_prompt: System instructions (ignored in mock).
            user_message: User message (ignored in mock).
            temperature: Sampling temperature (ignored in mock).

        Returns:
            JSON array string with mock queries.
        """
        self._call_count += 1

        if self.use_template_responses:
            return self._get_realistic_response(system_prompt)
        else:
            return self._get_simple_response()

    def _get_realistic_response(self, system_prompt: str) -> str:
        """Generate realistic mock queries based on system prompt context."""
        # Extract difficulty hints from prompt
        easy_count = self._extract_count(system_prompt, "Easy")
        medium_count = self._extract_count(system_prompt, "Medium")
        hard_count = self._extract_count(system_prompt, "Hard")

        # Extract persona hints
        is_professor = "교수" in system_prompt or "Professor" in system_prompt
        is_student = "학생" in system_prompt or "Student" in system_prompt
        is_staff = "직원" in system_prompt or "Staff" in system_prompt

        queries = []

        # Generate easy queries
        for i in range(easy_count):
            if is_professor:
                queries.append({
                    "query": f"교권 관련 규정 {i+1}번 알려주세요",
                    "type": "fact_check",
                    "difficulty": "easy"
                })
            elif is_student:
                queries.append({
                    "query": f"수강 신청 방법 {i+1}번 알려줘",
                    "type": "procedural",
                    "difficulty": "easy"
                })
            else:  # staff
                queries.append({
                    "query": f"연차 사용 절차 {i+1}번 알려주세요",
                    "type": "procedural",
                    "difficulty": "easy"
                })

        # Generate medium queries
        for i in range(medium_count):
            if is_professor:
                queries.append({
                    "query": f"연구비 신청 자격 {i+1}번 뭐야?",
                    "type": "eligibility",
                    "difficulty": "medium"
                })
            elif is_student:
                queries.append({
                    "query": f"장학금 이중 수급 제한 {i+1}번 있어?",
                    "type": "eligibility",
                    "difficulty": "medium"
                })
            else:  # staff
                queries.append({
                    "query": f"야간 근무 수당 {i+1}번 어떻게 돼?",
                    "type": "eligibility",
                    "difficulty": "medium"
                })

        # Generate hard queries
        for i in range(hard_count):
            if is_professor:
                queries.append({
                    "query": f"정년 보장 심사 기준 {i+1}번 왜 이래요?",
                    "type": "emotional",
                    "difficulty": "hard"
                })
            elif is_student:
                queries.append({
                    "query": f"복수전공과 연계전공 차이 {i+1}번 뭐야?",
                    "type": "complex",
                    "difficulty": "hard"
                })
            else:  # staff
                queries.append({
                    "query": f"예산 집행 승인 권한 {i+1}번 어디까지야?",
                    "type": "eligibility",
                    "difficulty": "hard"
                })

        import json
        return json.dumps(queries, ensure_ascii=False)

    def _get_simple_response(self) -> str:
        """Generate simple mock response."""
        import json
        return json.dumps([
            {
                "query": "휴학 신청은 어떻게 하나요?",
                "type": "procedural",
                "difficulty": "easy"
            },
            {
                "query": "장학금 자격 기준이 뭐야?",
                "type": "eligibility",
                "difficulty": "medium"
            }
        ], ensure_ascii=False)

    def _extract_count(self, prompt: str, difficulty: str) -> int:
        """Extract count from prompt for given difficulty level."""
        import re
        pattern = f"{difficulty}\\s*\\((\\d+)개\\)"
        match = re.search(pattern, prompt)
        if match:
            return int(match.group(1))
        # Default counts
        return 1 if difficulty == "Easy" else 2 if difficulty == "Medium" else 1

    def get_embedding(self, text: str) -> List[float]:
        """
        Return mock embedding vector.

        Args:
            text: Text to embed.

        Returns:
            Mock embedding vector (384-dim for text-embedding-3-small).
        """
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 384-dim vector
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        return embedding
