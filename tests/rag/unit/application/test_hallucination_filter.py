"""
Unit tests for HallucinationFilter service.

Tests for SPEC-RAG-Q-002: Hallucination Prevention.
Validates that LLM responses are filtered against actual context.

TDD RED Phase: These tests define expected behavior.
"""

import pytest

from src.rag.application.hallucination_filter import (
    FilterMode,
    FilterResult,
    HallucinationFilter,
)


class TestFilterMode:
    """Test FilterMode enum values."""

    def test_filter_mode_values(self):
        """THEN FilterMode should have expected values."""
        assert FilterMode.WARN.value == "warn"
        assert FilterMode.SANITIZE.value == "sanitize"
        assert FilterMode.BLOCK.value == "block"
        assert FilterMode.PASSTHROUGH.value == "passthrough"


class TestFilterResult:
    """Test FilterResult dataclass."""

    def test_filter_result_creation(self):
        """THEN FilterResult should store all fields."""
        result = FilterResult(
            original_response="original",
            sanitized_response="sanitized",
            is_modified=True,
            blocked=False,
            block_reason=None,
            issues=["issue1"],
            warnings=["warning1"],
        )
        assert result.original_response == "original"
        assert result.sanitized_response == "sanitized"
        assert result.is_modified is True
        assert result.blocked is False
        assert result.block_reason is None
        assert result.issues == ["issue1"]
        assert result.warnings == ["warning1"]


class TestHallucinationFilterInit:
    """Test HallucinationFilter initialization."""

    def test_default_mode_is_sanitize(self):
        """THEN default mode should be SANITIZE."""
        filter_service = HallucinationFilter()
        assert filter_service.mode == FilterMode.SANITIZE

    def test_custom_mode(self):
        """THEN should accept custom mode."""
        filter_service = HallucinationFilter(mode=FilterMode.WARN)
        assert filter_service.mode == FilterMode.WARN


class TestValidatePhoneNumbers:
    """Test phone number validation against context (REQ-001)."""

    def test_validate_phone_numbers_in_context(self):
        """
        WHEN phone number exists in context
        THEN should keep it in response
        """
        filter_service = HallucinationFilter()
        response = "문의는 02-1234-5678로 해주세요."
        context = ["연락처: 02-1234-5678"]

        result = filter_service.validate_contacts(response, context)

        assert result[0] == response  # No change
        assert result[1] == []  # No issues

    def test_validate_phone_numbers_not_in_context(self):
        """
        WHEN phone number not in context
        THEN should replace with fallback message
        """
        filter_service = HallucinationFilter()
        response = "문의는 02-9999-9999로 해주세요."
        context = ["다른 연락처 정보가 있습니다."]

        result = filter_service.validate_contacts(response, context)

        assert "02-9999-9999" not in result[0]
        assert "직접 문의" in result[0] or "담당 부서" in result[0]
        assert len(result[1]) > 0  # Has issues

    def test_validate_phone_numbers_partial_match(self):
        """
        WHEN phone number partially matches context
        THEN should keep it (context has the number)
        """
        filter_service = HallucinationFilter()
        response = "전화: 02-1234-5678"
        context = ["학적팀 연락처는 02-1234-5678입니다."]

        result = filter_service.validate_contacts(response, context)

        assert "02-1234-5678" in result[0]

    def test_validate_korean_phone_variations(self):
        """
        WHEN Korean phone formats are used
        THEN should handle various formats (02-XXX-XXXX, 010-XXXX-XXXX)
        """
        filter_service = HallucinationFilter()
        response = "연락처: 010-1234-5678 또는 02-345-6789"
        context = ["핸드폰: 010-1234-5678", "전화: 02-345-6789"]

        result = filter_service.validate_contacts(response, context)

        assert "010-1234-5678" in result[0]
        assert "02-345-6789" in result[0]
        assert result[1] == []

    def test_validate_phone_without_hyphens(self):
        """
        WHEN phone number without hyphens
        THEN should still detect and validate
        """
        filter_service = HallucinationFilter()
        response = "전화번호는 0212345678입니다."
        context = ["연락처: 02-1234-5678"]

        result = filter_service.validate_contacts(response, context)

        # Should recognize that 0212345678 matches 02-1234-5678
        assert len(result[1]) == 0  # No issues since it matches


class TestValidateEmailAddresses:
    """Test email validation against context (REQ-001)."""

    def test_validate_email_in_context(self):
        """
        WHEN email exists in context
        THEN should keep it in response
        """
        filter_service = HallucinationFilter()
        response = "이메일: contact@dongeui.ac.kr로 문의하세요."
        context = ["공식 이메일: contact@dongeui.ac.kr"]

        result = filter_service.validate_contacts(response, context)

        assert "contact@dongeui.ac.kr" in result[0]
        assert result[1] == []

    def test_validate_email_not_in_context(self):
        """
        WHEN email not in context
        THEN should replace with fallback message
        """
        filter_service = HallucinationFilter()
        response = "이메일: fake@unknown.com로 문의하세요."
        context = ["다른 이메일 정보가 있습니다."]

        result = filter_service.validate_contacts(response, context)

        assert "fake@unknown.com" not in result[0]
        assert len(result[1]) > 0


class TestValidateDepartmentNames:
    """Test department name validation against context (REQ-002)."""

    def test_validate_department_names_in_context(self):
        """
        WHEN department name exists in context
        THEN should keep it in response
        """
        filter_service = HallucinationFilter()
        response = "학적팀에 문의하세요."
        context = ["담당 부서: 학적팀"]

        result = filter_service.validate_departments(response, context)

        assert "학적팀" in result[0]
        assert result[1] == []

    def test_validate_department_names_not_in_context(self):
        """
        WHEN department name not in context
        THEN should generalize to "담당 부서"
        """
        filter_service = HallucinationFilter()
        response = "학술연구지원팀에 문의하세요."
        context = ["다른 부서 정보가 있습니다."]

        result = filter_service.validate_departments(response, context)

        assert "학술연구지원팀" not in result[0]
        assert "담당 부서" in result[0]
        assert len(result[1]) > 0

    def test_validate_multiple_departments(self):
        """
        WHEN multiple departments mentioned
        THEN should validate each independently
        """
        filter_service = HallucinationFilter()
        response = "학적팀과 장학팀에 문의하세요."
        context = ["학적 관련: 학적팀"]  # Only 학적팀 is in context

        result = filter_service.validate_departments(response, context)

        assert "학적팀" in result[0]
        assert "장학팀" not in result[0]
        assert "담당 부서" in result[0]

    def test_validate_department_variations(self):
        """
        WHEN department name has variations
        THEN should recognize common patterns
        """
        filter_service = HallucinationFilter()
        response = "교무처에 문의하세요."
        context = ["담당: 교무처"]

        result = filter_service.validate_departments(response, context)

        assert "교무처" in result[0]
        assert result[1] == []


class TestValidateCitations:
    """Test citation validation against context (REQ-003)."""

    def test_validate_citations_in_context(self):
        """
        WHEN citation exists in context
        THEN should keep it in response
        """
        filter_service = HallucinationFilter()
        response = "학칙 제15조에 따라 처리됩니다."
        context = ["학칙 제15조: 휴학에 관한 규정"]

        result = filter_service.validate_citations(response, context)

        assert "학칙 제15조" in result[0]
        assert result[1] == []

    def test_validate_citations_not_in_context(self):
        """
        WHEN citation not in context
        THEN should generalize to "관련 규정"
        """
        filter_service = HallucinationFilter()
        response = "규정 제99조에 따라 처리됩니다."
        context = ["다른 규정 정보가 있습니다."]

        result = filter_service.validate_citations(response, context)

        assert "제99조" not in result[0]
        assert "관련 규정" in result[0] or result[0] == ""  # Either generalize or remove
        assert len(result[1]) > 0

    def test_validate_multiple_citations(self):
        """
        WHEN multiple citations mentioned
        THEN should validate each independently
        """
        filter_service = HallucinationFilter()
        response = "학칙 제10조와 제20조를 참고하세요."
        context = ["학칙 제10조: 등록에 관한 규정"]

        result = filter_service.validate_citations(response, context)

        assert "제10조" in result[0]
        assert "제20조" not in result[0]


class TestFilterModeWarn:
    """Test FilterMode.WARN behavior."""

    def test_filter_mode_warn_logs_only(self):
        """
        WHEN mode is WARN
        THEN should not modify response, only record warnings
        """
        filter_service = HallucinationFilter(mode=FilterMode.WARN)
        response = "문의: 02-9999-9999 (학술연구지원팀)"
        context = ["다른 정보만 있습니다."]

        result = filter_service.filter_response(response, context)

        assert result.sanitized_response == response  # Unchanged
        assert result.is_modified is False
        assert len(result.warnings) > 0  # Has warnings
        assert result.blocked is False


class TestFilterModeSanitize:
    """Test FilterMode.SANITIZE behavior."""

    def test_filter_mode_sanitize_auto_fixes(self):
        """
        WHEN mode is SANITIZE
        THEN should auto-fix issues
        """
        filter_service = HallucinationFilter(mode=FilterMode.SANITIZE)
        response = "문의: 02-9999-9999 (학술연구지원팀)"
        context = ["다른 정보만 있습니다."]

        result = filter_service.filter_response(response, context)

        assert result.sanitized_response != response  # Modified
        assert result.is_modified is True
        assert "02-9999-9999" not in result.sanitized_response
        assert "학술연구지원팀" not in result.sanitized_response
        assert result.blocked is False


class TestFilterModeBlock:
    """Test FilterMode.BLOCK behavior."""

    def test_filter_mode_block_with_issues(self):
        """
        WHEN mode is BLOCK and issues detected
        THEN should block the response
        """
        filter_service = HallucinationFilter(mode=FilterMode.BLOCK)
        response = "문의: 02-9999-9999 (학술연구지원팀)"
        context = ["다른 정보만 있습니다."]

        result = filter_service.filter_response(response, context)

        assert result.blocked is True
        assert result.block_reason is not None
        assert len(result.issues) > 0

    def test_filter_mode_block_without_issues(self):
        """
        WHEN mode is BLOCK but no issues
        THEN should allow the response
        """
        filter_service = HallucinationFilter(mode=FilterMode.BLOCK)
        response = "학칙 제10조에 따라 처리됩니다."
        context = ["학칙 제10조: 등록에 관한 규정"]

        result = filter_service.filter_response(response, context)

        assert result.blocked is False
        assert result.block_reason is None


class TestFilterModePassthrough:
    """Test FilterMode.PASSTHROUGH behavior."""

    def test_filter_mode_passthrough(self):
        """
        WHEN mode is PASSTHROUGH
        THEN should not filter anything
        """
        filter_service = HallucinationFilter(mode=FilterMode.PASSTHROUGH)
        response = "문의: 02-9999-9999 (학술연구지원팀)"
        context = ["다른 정보만 있습니다."]

        result = filter_service.filter_response(response, context)

        assert result.sanitized_response == response
        assert result.is_modified is False
        assert result.blocked is False


class TestEmptyContext:
    """Test handling of empty context."""

    def test_empty_context_sanitizes_everything(self):
        """
        WHEN context is empty
        THEN should replace all unverified info
        """
        filter_service = HallucinationFilter()
        response = "문의: 02-9999-9999"
        context: list[str] = []

        result = filter_service.filter_response(response, context)

        assert "02-9999-9999" not in result.sanitized_response
        assert result.is_modified is True

    def test_empty_response(self):
        """
        WHEN response is empty
        THEN should handle gracefully
        """
        filter_service = HallucinationFilter()
        response = ""
        context = ["학칙 제10조"]

        result = filter_service.filter_response(response, context)

        assert result.sanitized_response == ""
        assert result.is_modified is False


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_mixed_valid_and_invalid_info(self):
        """
        WHEN response has both valid and invalid info
        THEN should only sanitize invalid parts
        """
        filter_service = HallucinationFilter()
        response = "학칙 제10조에 따라 처리되며, 문의는 02-9999-9999로 하세요."
        context = ["학칙 제10조: 등록에 관한 규정"]

        result = filter_service.filter_response(response, context)

        assert "학칙 제10조" in result.sanitized_response
        assert "02-9999-9999" not in result.sanitized_response

    def test_response_with_all_valid_info(self):
        """
        WHEN all info in response is valid
        THEN should not modify response
        """
        filter_service = HallucinationFilter()
        response = "학적팀(02-1234-5678)에 문의하면 학칙 제10조에 따라 처리됩니다."
        context = [
            "담당 부서: 학적팀",
            "전화: 02-1234-5678",
            "학칙 제10조: 등록 규정",
        ]

        result = filter_service.filter_response(response, context)

        assert result.sanitized_response == response
        assert result.is_modified is False
        assert result.blocked is False

    def test_korean_text_variations(self):
        """
        WHEN Korean text has variations
        THEN should handle spacing and format differences
        """
        filter_service = HallucinationFilter()
        response = "학 적 팀 에 문의"
        context = ["담당부서: 학적팀"]

        result = filter_service.validate_departments(response, context)

        # Should recognize that "학 적 팀" with spaces is "학적팀"
        # This is a more lenient match
        assert "학적팀" in result[0] or "학 적 팀" in result[0]
