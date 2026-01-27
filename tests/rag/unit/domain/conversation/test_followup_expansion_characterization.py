"""
Characterization tests for existing follow-up query expansion behavior.

These tests document the ACTUAL behavior of expand_followup_query and
build_history_context functions from chat_logic.py. They serve as a safety
net during refactoring to ensure behavior preservation.
"""

from src.rag.interface.chat_logic import (
    build_history_context,
    expand_followup_query,
    has_explicit_target,
    is_followup_message,
)


class TestExpandFollowupQueryCharacterization:
    """Characterization tests for expand_followup_query function."""

    def test_empty_message_returns_empty(self):
        """ACTUAL: Empty message returns empty string."""
        result = expand_followup_query("", "some context")
        assert result == ""

    def test_no_context_returns_message_unchanged(self):
        """ACTUAL: When no context provided, message is returned unchanged."""
        result = expand_followup_query("학칙에 대해 알려주세요", None)
        assert result == "학칙에 대해 알려주세요"

    def test_empty_context_returns_message_unchanged(self):
        """ACTUAL: When context is empty string, message is returned unchanged."""
        result = expand_followup_query("학칙에 대해 알려주세요", "")
        assert result == "학칙에 대해 알려주세요"

    def test_explicit_regulation_title_does_not_use_context(self):
        """ACTUAL: Regulation name in message prevents context prepending."""
        message = "교원인사규정 제7조가 뭐예요?"
        context = "학칙"
        result = expand_followup_query(message, context)
        assert result == message
        assert "학칙" not in result

    def test_article_reference_prepends_context(self):
        """ACTUAL: Article reference without regulation name prepends context."""
        message = "제7조가 뭐예요?"
        context = "교원인사규정"
        result = expand_followup_query(message, context)
        assert result == "교원인사규정 제7조가 뭐예요?"

    def test_article_reference_with_optional제(self):
        """ACTUAL: Article reference with '제' prefix prepends context."""
        message = "제 7 조 내용 알려줘"
        context = "교원인사규정"
        result = expand_followup_query(message, context)
        # Pattern should match and prepend context
        assert "교원인사규정" in result

    def test_explicit_target_keyword_does_not_use_context(self):
        """ACTUAL: Messages with explicit target keywords don't use context."""
        message = "그 규정의 제5조를 알려줘"
        context = "학칙"
        result = expand_followup_query(message, context)
        # "규정" is explicit target
        assert result == message

    def test_followup_token_prepends_context(self):
        """ACTUAL: Follow-up tokens trigger context prepending."""
        message = "그거 자세히 알려줘"
        context = "휴학 규정"
        result = expand_followup_query(message, context)
        assert result == "휴학 규정 그거 자세히 알려줘"

    def test_그럼_token_prepends_context(self):
        """ACTUAL: '그럼' token triggers context prepending."""
        message = "그럼 절차는 어떻게 되나요?"
        context = "등록금 납부"
        result = expand_followup_query(message, context)
        assert result == "등록금 납부 그럼 절차는 어떻게 되나요?"

    def test_추가로_token_prepends_context(self):
        """ACTUAL: '추가로' token triggers context prepending."""
        message = "추가로 필요한 서류가 있나요?"
        context = "휴학 신청"
        result = expand_followup_query(message, context)
        assert result == "휴학 신청 추가로 필요한 서류가 있나요?"

    def test_non_followup_without_explicit_target_no_context(self):
        """ACTUAL: Non-followup without explicit target does NOT prepend context."""
        message = "알려주세요"
        context = "학칙"
        result = expand_followup_query(message, context)
        # Without followup tokens or explicit target, context is NOT prepended
        assert result == message


class TestIsFollowupMessageCharacterization:
    """Characterization tests for is_followup_message function."""

    def test_empty_message_returns_false(self):
        """ACTUAL: Empty message returns False."""
        assert not is_followup_message("")

    def test_그거_token_is_followup(self):
        """ACTUAL: '그거' is detected as follow-up."""
        assert is_followup_message("그거 알려줘")

    def test_그것_token_is_followup(self):
        """ACTUAL: '그것' is detected as follow-up."""
        assert is_followup_message("그것 설명해줘")

    def test_이거_token_is_followup(self):
        """ACTUAL: '이거' is detected as follow-up."""
        assert is_followup_message("이거 어떻게 하나요?")

    def test_그럼_token_is_followup(self):
        """ACTUAL: '그럼' is detected as follow-up."""
        assert is_followup_message("그럼 다음은?")

    def test_추가로_token_is_followup(self):
        """ACTUAL: '추가로' is detected as follow-up."""
        assert is_followup_message("추가로 더 알려줘")

    def test_자세히_token_is_followup(self):
        """ACTUAL: '자세히' is detected as follow-up."""
        assert is_followup_message("자세히 설명해주세요")

    def test_계속_token_is_followup(self):
        """ACTUAL: '계속' is detected as follow-up."""
        assert is_followup_message("계속 이어서")

    def test_전문_token_is_followup(self):
        """ACTUAL: '전문' is detected as follow-up."""
        assert is_followup_message("전문 보여줘")

    def test_fulltext_token_is_followup(self):
        """ACTUAL: 'fulltext' is detected as follow-up."""
        assert is_followup_message("fulltext please")

    def test_regulation_name_without_followup_tokens(self):
        """ACTUAL: Regulation name without tokens is not follow-up."""
        assert not is_followup_message("교원인사규정이 뭐예요?")

    def test_article_only_without_tokens(self):
        """ACTUAL: Article reference without tokens is not follow-up."""
        assert not is_followup_message("제7조 내용이 뭐예요?")


class TestHasExplicitTargetCharacterization:
    """Characterization tests for has_explicit_target function."""

    def test_empty_returns_false(self):
        """ACTUAL: Empty string returns False."""
        assert not has_explicit_target("")

    def test_rule_code_pattern_detected(self):
        """ACTUAL: Rule code pattern like 1-2-3 is detected."""
        assert has_explicit_target("1-2-5 규정")

    def test_article_pattern_with제_detected(self):
        """ACTUAL: Pattern with '제 N 조' is detected."""
        assert has_explicit_target("제7조의 내용은")

    def test_article_pattern_without제_not_detected(self):
        """ACTUAL: Pattern without '제' prefix is NOT detected (requires '제')."""
        assert not has_explicit_target("7조가 뭐예요")

    def test_규정_suffix_detected(self):
        """ACTUAL: '규정' suffix is detected."""
        assert has_explicit_target("교원인사규정")

    def test_학칙_detected(self):
        """ACTUAL: '학칙' is detected."""
        assert has_explicit_target("학칙이 뭐예요")

    def test_정관_detected(self):
        """ACTUAL: '정관' is detected."""
        assert has_explicit_target("대학정관을 알려줘")

    def test_세칙_suffix_not_in_target_list(self):
        """ACTUAL: '세칙' is NOT in target keywords list."""
        # Only '규칙' is in the list, not '세칙'
        assert not has_explicit_target("시행세칙 설명해줘")

    def test_지침_detected(self):
        """ACTUAL: '지침' is detected."""
        assert has_explicit_target("처리지침이 어떻게 되나요")

    def test_그규정_returns_false(self):
        """ACTUAL: '그 규정' returns False (context reference)."""
        assert not has_explicit_target("그 규정에 대해")

    def test_이규정_matches_규정(self):
        """ACTUAL: '이 규정' matches '규정' keyword."""
        # "이 규정" contains "규정" as a separate word
        assert has_explicit_target("이 규정의 제3조는")

    def test_해당규정_returns_false(self):
        """ACTUAL: '해당규정' returns False."""
        assert not has_explicit_target("해당규정을 찾아줘")


class TestBuildHistoryContextCharacterization:
    """Characterization tests for build_history_context function."""

    def test_empty_history_returns_empty_string(self):
        """ACTUAL: Empty history returns empty string."""
        result = build_history_context([])
        assert result == ""

    def test_none_history_returns_empty_string(self):
        """ACTUAL: None history returns empty string."""
        result = build_history_context(None)
        assert result == ""

    def test_single_user_message(self):
        """ACTUAL: Single user message formatted correctly."""
        history = [{"role": "user", "content": "휴학은 어떻게 하나요?"}]
        result = build_history_context(history)
        assert "사용자: 휴학은 어떻게 하나요?" in result

    def test_single_assistant_message(self):
        """ACTUAL: Single assistant message formatted correctly."""
        history = [{"role": "assistant", "content": "휴학 절차는 다음과 같습니다."}]
        result = build_history_context(history)
        assert "어시스턴트: 휴학 절차는 다음과 같습니다." in result

    def test_user_assistant_pair(self):
        """ACTUAL: User and assistant messages both included."""
        history = [
            {"role": "user", "content": "휴학 방법"},
            {"role": "assistant", "content": "휴학 신청서 제출"},
        ]
        result = build_history_context(history)
        assert "사용자: 휴학 방법" in result
        assert "어시스턴트: 휴학 신청서 제출" in result

    def test_tuple_format_supported(self):
        """ACTUAL: List of tuples format is supported."""
        history = [
            ("휴학 방법", "휴학 신청서 제출"),
            ("절차는?", "서류 제출"),
        ]
        result = build_history_context(history)
        assert "사용자: 휴학 방법" in result
        assert "어시스턴트: 휴학 신청서 제출" in result
        assert "사용자: 절차는?" in result
        assert "어시스턴트: 서류 제출" in result

    def test_max_turns_limits_messages(self):
        """ACTUAL: max_turns parameter limits number of turns."""
        history = []
        for i in range(1, 6):
            question = "Question " + str(i)
            answer = "Answer " + str(i)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        result = build_history_context(history, max_turns=2)
        # Should only have last 2 turns (4 messages)
        lines = result.strip().split("\n")
        assert len(lines) <= 4

    def test_long_content_truncated(self):
        """ACTUAL: Long content is truncated with ellipsis."""
        long_text = "a" * 400
        history = [{"role": "user", "content": long_text}]
        result = build_history_context(history)
        assert "..." in result
        assert len(result) < 400  # Truncated

    def test_max_chars_enforced(self):
        """ACTUAL: max_chars parameter limits total length."""
        history = [
            {"role": "user", "content": "첫 번째 메시지입니다"},
            {"role": "assistant", "content": "첫 번째 응답입니다"},
            {"role": "user", "content": "두 번째 메시지입니다"},
        ]
        result = build_history_context(history, max_chars=50)
        assert len(result) <= 100  # Some margin for formatting

    def test_whitespace_normalized(self):
        """ACTUAL: Multiple whitespace characters are normalized."""
        history = [{"role": "user", "content": "휴학은    어떻게   하나요?"}]
        result = build_history_context(history)
        assert "휴학은 어떻게 하나요?" in result
        assert "    " not in result
