"""Tests for QueryHandler input validation."""

import pytest

from src.rag.interface.query_handler import QueryHandler


class TestQueryValidation:
    """Test input validation in QueryHandler."""

    def test_validate_query_normal(self):
        """정상 쿼리는 통과해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원 연구년 신청 방법")
        assert is_valid is True
        assert msg == ""

    def test_validate_query_too_long(self):
        """500자 초과 쿼리는 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        long_query = "가" * 501
        is_valid, msg = handler.validate_query(long_query)
        assert is_valid is False
        assert "너무 깁니다" in msg

    def test_validate_query_xss_script(self):
        """XSS 공격 패턴은 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        xss_query = "<script>alert('xss')</script>"
        is_valid, msg = handler.validate_query(xss_query)
        assert is_valid is False
        assert "허용되지 않는 문자" in msg

    def test_validate_query_javascript_url(self):
        """JavaScript URL은 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        js_query = "javascript:alert('xss')"
        is_valid, msg = handler.validate_query(js_query)
        assert is_valid is False

    def test_validate_query_sql_injection(self):
        """SQL Injection 패턴은 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        sql_query = "DROP TABLE users"
        is_valid, msg = handler.validate_query(sql_query)
        assert is_valid is False

    def test_validate_query_empty(self):
        """빈 쿼리는 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("")
        assert is_valid is False
        assert "검색어를 입력" in msg

    def test_validate_query_whitespace_only(self):
        """공백만 있는 쿼리는 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("   ")
        assert is_valid is False

    def test_validate_query_control_chars(self):
        """제어 문자는 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        query_with_null = "교원\x00연구년"
        is_valid, msg = handler.validate_query(query_with_null)
        assert is_valid is False
        assert "제어 문자" in msg

    def test_validate_query_korean_with_special_chars(self):
        """한글 + 특수문자 조합은 허용해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원 연구년 (신청 방법)?")
        assert is_valid is True

    def test_validate_query_iframe(self):
        """iframe 태그는 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<iframe src='evil.com'>")
        assert is_valid is False

    def test_validate_query_template_injection(self):
        """Template injection 패턴은 거부해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("${7*7}")
        assert is_valid is False

    def test_validate_query_max_length_boundary(self):
        """500자 경계값 테스트."""
        handler = QueryHandler(store=None, llm_client=None)

        # Exactly 500 chars should pass
        is_valid, _ = handler.validate_query("가" * 500)
        assert is_valid is True

        # 501 chars should fail
        is_valid, _ = handler.validate_query("가" * 501)
        assert is_valid is False

    def test_validate_query_newline_allowed(self):
        """줄바꿈 문자는 허용해야 함."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원 연구년\n신청 방법")
        assert is_valid is True
