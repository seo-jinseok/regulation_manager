"""
Tests for HallucinationDetector.

Tests the hallucination detection and prevention functionality.
"""

import pytest

from src.rag.automation.infrastructure.evaluation_helpers import HallucinationDetector


class TestHallucinationDetector:
    """Test HallucinationDetector functionality."""

    def test_detect_phone_numbers_with_dash(self):
        """Test phone number detection with dashes."""
        answer = "학교에 02-1234-5678로 문의하세요."

        detected = HallucinationDetector.detect_phone_numbers(answer)

        assert len(detected) == 1
        assert "02-1234-5678" in detected

    def test_detect_phone_numbers_without_dash(self):
        """Test phone number detection without dashes."""
        answer = "01012345678로 연락하세요."

        detected = HallucinationDetector.detect_phone_numbers(answer)

        assert len(detected) == 1
        assert "01012345678" in detected

    def test_detect_phone_numbers_with_parentheses(self):
        """Test phone number detection with parentheses."""
        answer = "(02) 1234-5678로 문의하세요."

        detected = HallucinationDetector.detect_phone_numbers(answer)

        assert len(detected) >= 1

    def test_detect_phone_numbers_multiple(self):
        """Test detection of multiple phone numbers."""
        answer = "학사팀 02-1234-5678, 교학과 02-8765-4321"

        detected = HallucinationDetector.detect_phone_numbers(answer)

        assert len(detected) == 2

    def test_detect_phone_numbers_none(self):
        """Test when no phone numbers present."""
        answer = "학칙 제15조에 따르면 휴학할 수 있습니다."

        detected = HallucinationDetector.detect_phone_numbers(answer)

        assert len(detected) == 0

    def test_detect_other_universities_seoul(self):
        """Test detection of Seoul National University references."""
        answer = "서울대학교에서는 이렇게 합니다."

        detected = HallucinationDetector.detect_other_universities(answer)

        assert len(detected) >= 1
        assert "서울대학교" in detected

    def test_detect_other_universities_yonsei(self):
        """Test detection of Yonsei University references."""
        answer = "연세대의 규정은 다릅니다."

        detected = HallucinationDetector.detect_other_universities(answer)

        assert len(detected) == 1
        assert "연세대" in detected

    def test_detect_other_universities_multiple(self):
        """Test detection of multiple university references."""
        answer = "서울대와 연세대, 고려대의 규정을 참고하세요."

        detected = HallucinationDetector.detect_other_universities(answer)

        assert len(detected) >= 3

    def test_detect_other_universities_english_acronyms(self):
        """Test detection of English university acronyms."""
        answer = "KAIST나 POSTECH의 경우 다릅니다."

        detected = HallucinationDetector.detect_other_universities(answer)

        assert len(detected) >= 2

    def test_detect_other_universities_none(self):
        """Test when no other university references."""
        answer = "동의대학교 규정에 따릅니다."

        detected = HallucinationDetector.detect_other_universities(answer)

        assert len(detected) == 0

    def test_detect_evasive_responses_typical(self):
        """Test detection of typical evasive phrases."""
        answer = "대학마다 다를 수 있습니다."

        detected = HallucinationDetector.detect_evasive_responses(answer)

        assert len(detected) == 1

    def test_detect_evasive_responses_generally(self):
        """Test detection of 'generally' type responses."""
        answer = "일반적으로 그렇습니다."

        detected = HallucinationDetector.detect_evasive_responses(answer)

        assert len(detected) == 1

    def test_detect_evasive_responses_multiple(self):
        """Test detection of multiple evasive phrases."""
        answer = "대학마다 다를 수 있고, 일반적으로도 그렇습니다."

        detected = HallucinationDetector.detect_evasive_responses(answer)

        assert len(detected) >= 2

    def test_detect_evasive_responses_none(self):
        """Test when no evasive responses."""
        answer = "학칙 제15조에 따라 휴학할 수 있습니다."

        detected = HallucinationDetector.detect_evasive_responses(answer)

        assert len(detected) == 0

    def test_has_hallucination_phone(self):
        """Test hallucination detection with phone numbers."""
        answer = "02-1234-5678로 문의하세요."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 1
        assert "전화번호" in issues[0]

    def test_has_hallucination_university(self):
        """Test hallucination detection with other universities."""
        answer = "서울대학교 규정을 참고하세요."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 1
        assert "대학교" in issues[0]

    def test_has_hallucination_evasive(self):
        """Test hallucination detection with evasive responses."""
        answer = "대학마다 다를 수 있습니다."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 1
        assert "회피성" in issues[0]

    def test_has_hallucination_none(self):
        """Test when no hallucination detected."""
        answer = "학칙 제15조에 따라 휴학할 수 있습니다."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is False
        assert len(issues) == 0

    def test_has_hallucination_multiple_issues(self):
        """Test detection of multiple hallucination types."""
        answer = "서울대학교에 02-1234-5678로 문의하세요. 대학마다 다를 수 있습니다."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 3

    def test_sanitize_answer_phone_removal(self):
        """Test phone number sanitization."""
        answer = "학사팀 02-1234-5678로 문의하세요."

        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert "[연락처는 학교 홈페이지를 확인하세요]" in sanitized
        assert "02-1234-5678" not in sanitized
        assert len(changes) > 0

    def test_sanitize_answer_university_replacement(self):
        """Test university reference replacement."""
        answer = "서울대학교 규정을 참고하세요."

        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert "[다른 대학교]" in sanitized
        assert "서울대학교" not in sanitized
        assert len(changes) > 0

    def test_sanitize_answer_evasive_replacement(self):
        """Test evasive response replacement."""
        answer = "대학마다 다를 수 있습니다."

        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert "[정확한 정보는 동의대학교 규정을 확인하세요]" in sanitized
        assert len(changes) > 0

    def test_sanitize_answer_combined(self):
        """Test sanitization with multiple issues."""
        answer = "서울대학교에 02-1234-5678로 문의하세요. 대학마다 다를 수 있습니다."

        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert len(changes) >= 3
        assert "[연락처는 학교 홈페이지를 확인하세요]" in sanitized
        assert "[다른 대학교]" in sanitized

    def test_block_if_hallucination_phone(self):
        """Test blocking with phone numbers."""
        answer = "02-1234-5678로 문의하세요."

        should_block, reason, issues = HallucinationDetector.block_if_hallucination(
            answer
        )

        assert should_block is True
        assert len(reason) > 0
        assert len(issues) == 1

    def test_block_if_hallucination_clean(self):
        """Test not blocking clean answer."""
        answer = "학칙 제15조에 따라 휴학할 수 있습니다."

        should_block, reason, issues = HallucinationDetector.block_if_hallucination(
            answer
        )

        assert should_block is False
        assert len(reason) == 0
        assert len(issues) == 0

    def test_block_if_hallucination_comprehensive(self):
        """Test blocking with comprehensive hallucination."""
        answer = """
        서울대학교에서는 휴학이 가능합니다.
        학사팀 02-1234-5678로 문의하세요.
        대학마다 다를 수 있습니다.
        """

        should_block, reason, issues = HallucinationDetector.block_if_hallucination(
            answer
        )

        assert should_block is True
        assert len(issues) == 3
        assert "전화번호" in reason
        assert "대학교" in reason
        assert "회피성" in reason


@pytest.mark.integration
class TestHallucinationDetectorIntegration:
    """Integration tests for HallucinationDetector."""

    def test_full_detection_sanitization_workflow(self):
        """Test complete detection and sanitization workflow."""
        # Step 1: Detect hallucination
        answer = "서울대학교에 02-1234-5678로 문의하세요."

        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 2

        # Step 2: Check if blocking is needed
        should_block, reason, _ = HallucinationDetector.block_if_hallucination(answer)

        assert should_block is True

        # Step 3: Sanitize answer
        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert len(changes) == 2
        assert "[연락처는 학교 홈페이지를 확인하세요]" in sanitized
        assert "[다른 대학교]" in sanitized

        # Step 4: Verify sanitized answer is clean
        is_clean, _ = HallucinationDetector.has_hallucination(sanitized)

        assert is_clean is False

    def test_multi_type_hallucination_handling(self):
        """Test handling answers with multiple hallucination types."""
        answer = """
        동의대학교 휴학 제도는 다음과 같습니다.
        서울대학교에서는 비슷합니다.
        학사팀 02-1234-5678로 연락하세요.
        대학마다 다를 수 있으니 확인하세요.
        """

        # Detect all issues
        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        assert has_hallucination is True
        assert len(issues) == 3

        # Sanitize
        sanitized, changes = HallucinationDetector.sanitize_answer(answer)

        assert len(changes) == 3

        # Verify all problematic content removed
        assert "서울대학교" not in sanitized
        assert "02-1234-5678" not in sanitized
        assert "대학마다 다를 수" not in sanitized

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty answer
        has_hallucination, _ = HallucinationDetector.has_hallucination("")
        assert has_hallucination is False

        # Very long answer with multiple phone formats
        long_answer = "연락처 " + ", ".join(
            [f"02-{i:04d}-{j:04d}" for i in range(10) for j in range(10)]
        )
        detected = HallucinationDetector.detect_phone_numbers(long_answer)
        assert len(detected) == 100

        # University names within valid words
        edge_answer = "서울대학교역 근처에 있습니다."
        detected = HallucinationDetector.detect_other_universities(edge_answer)
        # Should still detect even within context
        assert len(detected) >= 1


@pytest.mark.unit
class TestHallucinationDetectorPatterns:
    """Test pattern definitions and constants."""

    def test_phone_patterns_coverage(self):
        """Verify phone number patterns cover common formats."""
        test_cases = [
            "02-1234-5678",
            "010-123-4567",
            "031-123-4567",
            "0212345678",
            "01012345678",
            "(02) 1234-5678",
        ]

        for phone in test_cases:
            detected = HallucinationDetector.detect_phone_numbers(f"전화: {phone}")
            assert len(detected) >= 1, f"Failed to detect: {phone}"

    def test_university_list_completeness(self):
        """Verify university list contains major Korean universities."""
        major_universities = [
            "서울대학교",
            "연세대학교",
            "고려대학교",
            "카이스트",
            "포항공대",
        ]

        for uni in major_universities:
            assert uni in HallucinationDetector.OTHER_UNIVERSITIES or any(
                uni in u for u in HallucinationDetector.OTHER_UNIVERSITIES
            ), f"Missing university: {uni}"

    def test_evasive_patterns_coverage(self):
        """Verify evasive patterns cover common phrases."""
        test_cases = [
            "대학마다 다를 수 있습니다",
            "각 대학의 상황에 따라",
            "일반적으로 그렇습니다",
            "보통은 가능합니다",
        ]

        for phrase in test_cases:
            detected = HallucinationDetector.detect_evasive_responses(phrase)
            assert len(detected) >= 1, f"Failed to detect: {phrase}"
