"""Tests for SPEC-RAG-003 Phase 2: Answer quality improvement.

Task 2.1: CoT suppression prompt (tested indirectly via output)
Task 2.2: _strip_cot_from_answer post-processing
Task 2.3: Answer completeness prompt (tested indirectly via output)
"""

import pytest


class TestStripCotFromAnswer:
    """Tests for _strip_cot_from_answer function."""

    def _strip(self, text: str) -> str:
        from src.rag.application.search_usecase import _strip_cot_from_answer

        return _strip_cot_from_answer(text)

    def test_removes_numbered_analysis_steps(self):
        answer = (
            "1. **Analyze the User's Request**: The user asks about leave of absence.\n"
            "2. **Check constraints**: Verify regulation compliance.\n"
            "\n"
            "휴학 신청은 학기 개시 전까지 가능합니다."
        )
        result = self._strip(answer)
        assert "Analyze the User's Request" not in result
        assert "Check constraints" not in result
        assert "휴학 신청은 학기 개시 전까지 가능합니다." in result

    def test_removes_user_persona_analysis(self):
        answer = (
            "**User Persona:** Undergraduate student asking about tuition.\n"
            "\n"
            "등록금 납부 기간은 매학기 초에 공지됩니다."
        )
        result = self._strip(answer)
        assert "User Persona" not in result
        assert "등록금 납부 기간은 매학기 초에 공지됩니다." in result

    def test_removes_constraint_checklist(self):
        answer = (
            "**Constraint 1: Only use context** - Verified.\n"
            "**Constraint 2: No fabrication** - Verified.\n"
            "\n"
            "장학금 신청은 학기 시작 2주 전까지입니다."
        )
        result = self._strip(answer)
        assert "Constraint" not in result
        assert "장학금 신청은 학기 시작 2주 전까지입니다." in result

    def test_removes_step_markers(self):
        answer = "Step 1: Identify regulation.\nStep 2: Extract answer.\n\n졸업 요건은 다음과 같습니다."
        result = self._strip(answer)
        assert "Step 1:" not in result
        assert "Step 2:" not in result
        assert "졸업 요건은 다음과 같습니다." in result

    def test_removes_reasoning_headers(self):
        answer = "## Analysis\nSome reasoning...\n\n## Internal Thought Process\nMore reasoning.\n\n답변입니다."
        result = self._strip(answer)
        assert "## Analysis" not in result
        assert "Internal Thought Process" not in result
        assert "답변입니다." in result

    def test_preserves_clean_answer(self):
        answer = (
            "### 1. 핵심 답변\n"
            "휴학은 「학칙」 제40조에 따라 학기 개시 전까지 신청 가능합니다.\n\n"
            "### 2. 관련 규정\n"
            "- **규정명**: 학칙\n"
            "- **조항**: 제40조"
        )
        result = self._strip(answer)
        assert result == answer

    def test_preserves_numbered_list_in_answer(self):
        """Numbered lists in answers should NOT be stripped (not CoT)."""
        answer = (
            "졸업 요건은 다음과 같습니다:\n"
            "1. 총 130학점 이상 이수\n"
            "2. 평점평균 2.0 이상\n"
            "3. 졸업논문 합격"
        )
        result = self._strip(answer)
        assert "1. 총 130학점 이상 이수" in result
        assert "2. 평점평균 2.0 이상" in result
        assert "3. 졸업논문 합격" in result

    def test_empty_string(self):
        assert self._strip("") == ""

    def test_none_returns_none(self):
        assert self._strip(None) is None

    def test_collapses_extra_newlines(self):
        answer = "Some cot line\n\n\n\n\n유효한 답변"
        result = self._strip(answer)
        assert "\n\n\n" not in result

    def test_mixed_cot_and_valid_content(self):
        answer = (
            "1. **Analyze the User's Request**: 분석 중...\n"
            "**User Persona:** 학생\n"
            "\n"
            "### 1. 핵심 답변\n"
            "휴학 절차는 다음과 같습니다:\n"
            "1. 휴학원 작성\n"
            "2. 지도교수 승인\n"
            "3. 학과사무실 제출\n"
            "\n"
            "### 2. 관련 규정\n"
            "- 「학칙」 제40조"
        )
        result = self._strip(answer)
        assert "Analyze the User's Request" not in result
        assert "User Persona" not in result
        assert "### 1. 핵심 답변" in result
        assert "1. 휴학원 작성" in result
        assert "「학칙」 제40조" in result


class TestCotSuppressionPrompt:
    """Verify CoT suppression directives exist in the prompt."""

    def test_fallback_prompt_contains_cot_suppression(self):
        from src.rag.application.search_usecase import _get_fallback_regulation_qa_prompt

        prompt = _get_fallback_regulation_qa_prompt()
        assert "출력 형식" in prompt
        assert "SPEC-RAG-003" in prompt
        assert "분석 과정" in prompt
        assert "User Persona" in prompt

    def test_fallback_prompt_contains_completeness_requirements(self):
        from src.rag.application.search_usecase import _get_fallback_regulation_qa_prompt

        prompt = _get_fallback_regulation_qa_prompt()
        assert "답변 완전성" in prompt
        assert "절차" in prompt
        assert "기한" in prompt
        assert "자격 요건" in prompt


class TestCotPatternsCompile:
    """Verify CoT patterns compile and match expected inputs."""

    def test_patterns_are_compiled(self):
        from src.rag.application.search_usecase import _COT_PATTERNS

        assert len(_COT_PATTERNS) >= 4
        for pat in _COT_PATTERNS:
            assert hasattr(pat, "pattern")

    def test_analyze_pattern_matches(self):
        from src.rag.application.search_usecase import _COT_PATTERNS

        test_line = '1. **Analyze the User\'s Request**: The user wants...'
        matched = any(p.search(test_line) for p in _COT_PATTERNS)
        assert matched

    def test_regular_numbered_list_not_matched(self):
        from src.rag.application.search_usecase import _COT_PATTERNS

        test_line = "1. 총 130학점 이상 이수"
        matched = any(p.search(test_line) for p in _COT_PATTERNS)
        assert not matched
