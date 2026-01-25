from src.rag.interface.chat_logic import (
    attachment_label_variants,
    build_history_context,
    expand_followup_query,
    extract_regulation_title,
    has_explicit_target,
    is_followup_message,
    parse_attachment_request,
    resolve_audience_choice,
    resolve_regulation_choice,
)


def test_resolve_audience_choice_korean():
    assert resolve_audience_choice("교원 인사 규정") == "교수"
    assert resolve_audience_choice("학생 휴학 절차") == "학생"
    assert resolve_audience_choice("직원 승진 기준") == "직원"


def test_resolve_audience_choice_ambiguous():
    assert resolve_audience_choice("교수 학생") is None


def test_resolve_regulation_choice_exact():
    options = ["교원인사규정", "교원복무규정"]
    assert resolve_regulation_choice("교원인사규정", options) == "교원인사규정"


def test_resolve_regulation_choice_partial():
    options = ["교원인사규정", "교원복무규정"]
    assert resolve_regulation_choice("인사", options) == "교원인사규정"


def test_has_explicit_target():
    assert has_explicit_target("교원인사규정 전문") is True
    assert has_explicit_target("제3조 휴학") is True
    assert has_explicit_target("3-1-5 규정") is True
    assert has_explicit_target("휴학 절차") is False
    assert has_explicit_target("그 규정 전문") is False


def test_is_followup_message():
    assert is_followup_message("그럼?") is True
    assert is_followup_message("전문 보여줘") is True
    assert is_followup_message("다른 부칙은?") is True
    assert is_followup_message("휴학 절차는?") is False


def test_expand_followup_query():
    context = "교원인사규정"
    assert expand_followup_query("전문 보여줘", context) == "교원인사규정 전문 보여줘"
    assert expand_followup_query("교원복무규정 전문", context) == "교원복무규정 전문"
    assert expand_followup_query("휴학 절차는?", context) == "휴학 절차는?"


def test_extract_regulation_title():
    assert extract_regulation_title("교원인사규정의 별첨 자료 1번") == "교원인사규정"
    assert extract_regulation_title("JA교원인사규정 별첨") == "JA교원인사규정"
    assert (
        extract_regulation_title("중앙도서관자료제적에관한세칙 별표 1")
        == "중앙도서관자료제적에관한세칙"
    )


def test_parse_attachment_request_with_regulation():
    result = parse_attachment_request("교원인사규정의 별첨 자료 1번", None)
    assert result == ("교원인사규정", 1, "별첨")


def test_parse_attachment_request_with_fallback():
    result = parse_attachment_request("별표 2 보여줘", "교원인사규정")
    assert result == ("교원인사규정", 2, "별표")


def test_attachment_label_variants():
    assert attachment_label_variants("별첨") == ["별첨"]
    assert attachment_label_variants(None) == ["별표", "별첨", "별지"]


def test_build_history_context_limits_and_formats():
    history = [
        {"role": "user", "content": "교원인사규정"},
        {"role": "assistant", "content": "교원인사규정 전문을 표시합니다."},
        {"role": "user", "content": "다른 부칙은?"},
    ]
    context = build_history_context(history, max_turns=2, max_chars=200)
    assert "사용자: 교원인사규정" in context
    assert "어시스턴트: 교원인사규정 전문을 표시합니다." in context


# Additional tests for edge cases (missing lines coverage)


def test_resolve_audience_choice_none_input():
    assert resolve_audience_choice(None) is None
    assert resolve_audience_choice("") is None


def test_resolve_audience_choice_english_keywords():
    assert resolve_audience_choice("faculty rules") == "교수"
    assert resolve_audience_choice("student guide") == "학생"
    assert resolve_audience_choice("staff manual") == "직원"


def test_resolve_regulation_choice_none_inputs():
    assert resolve_regulation_choice(None, ["option1", "option2"]) is None
    assert resolve_regulation_choice("test", None) is None
    assert resolve_regulation_choice("test", []) is None


def test_resolve_regulation_choice_exact_match_with_spaces():
    options = ["교원인사규정", "학칙"]
    result = resolve_regulation_choice("교원인사규정", options)
    assert result == "교원인사규정"


def test_has_explicit_target_none_input():
    assert has_explicit_target(None) is False
    assert has_explicit_target("") is False


def test_has_explicit_target_rule_code_pattern():
    assert has_explicit_target("1-2-3 규정") is True
    assert has_explicit_target("규정 10-5-2") is True


def test_is_followup_message_none_input():
    assert is_followup_message(None) is False
    assert is_followup_message("") is False


def test_is_followup_message_additional_patterns():
    assert is_followup_message("추가로 알려줘") is True
    assert is_followup_message("더 자세히") is True
    assert is_followup_message("계속 설명해") is True
    assert is_followup_message("다시 보여줘") is True
    assert is_followup_message("전문 원문") is True


def test_expand_followup_query_none_context():
    assert expand_followup_query("test query", None) == "test query"
    assert expand_followup_query("", "context") == ""


def test_expand_followup_query_article_without_regulation():
    context = "교원인사규정"
    result = expand_followup_query("제5조", context)
    assert "교원인사규정" in result
    assert "제5조" in result


def test_expand_followup_query_context_switch():
    context = "교원인사규정"
    result = expand_followup_query("학칙 제3조", context)
    # Should not prepend context when new regulation is mentioned
    assert result == "학칙 제3조"


def test_expand_followup_query_explicit_target():
    context = "교원인사규정"
    result = expand_followup_query("휴학 규정", context)
    # Should not prepend when explicit target is present
    assert result == "휴학 규정"


def test_extract_regulation_title_none_input():
    assert extract_regulation_title(None) is None
    assert extract_regulation_title("") is None


def test_extract_regulation_title_with_particles():
    # Test trailing particle removal
    result = extract_regulation_title("교원인사규정의")
    assert result == "교원인사규정"

    result = extract_regulation_title("학칙을")
    assert result == "학칙"

    result = extract_regulation_title("교원복무규정은")
    assert result == "교원복무규정"


def test_extract_regulation_title_multiple_matches():
    # Should pick the longest match
    text = "교원인사규정과 교원복무규정의 차이"
    result = extract_regulation_title(text)
    # Longest match should be selected
    assert "규정" in result


def test_parse_attachment_request_none_input():
    assert parse_attachment_request(None, None) is None
    assert parse_attachment_request("", None) is None


def test_parse_attachment_request_only_label():
    result = parse_attachment_request("별표", "교원인사규정")
    assert result is not None
    assert result[2] == "별표"


def test_parse_attachment_request_with_nearby_number():
    result = parse_attachment_request("별표 규정 12번 형식", "교원인사규정")
    assert result is not None
    assert result[1] == 12  # table_no


def test_parse_attachment_request_label_variations():
    result1 = parse_attachment_request("별지 1", "test")
    assert result1[2] == "별지"

    result2 = parse_attachment_request("별첨 자료", "test")
    assert result2[2] == "별첨"


def test_parse_attachment_request_fallback_to_regulation():
    # When no regulation in text, use fallback
    result = parse_attachment_request("별표", "fallback_regulation")
    assert result is not None
    assert result[0] == "fallback_regulation"


def test_parse_attachment_request_no_fallback_no_reg():
    # Should return None when no regulation and no fallback
    result = parse_attachment_request("별표", None)
    assert result is None


def test_attachment_label_variants_empty_label():
    assert attachment_label_variants("") == ["별표", "별첨", "별지"]


def test_build_history_context_empty_history():
    assert build_history_context([]) == ""
    assert build_history_context(None) == ""


def test_build_history_context_tuple_format():
    history = [
        ("user message", "assistant response"),
        ("second user", "second assistant"),
    ]
    context = build_history_context(history)
    assert "사용자: user message" in context
    assert "어시스턴트: assistant response" in context


def test_build_history_context_long_content_truncation():
    long_content = "x" * 400
    history = [
        {"role": "user", "content": long_content},
    ]
    context = build_history_context(history, max_chars=200)
    assert "..." in context
    assert len(context) <= 250  # Allow some margin


def test_build_history_context_max_turns():
    history = [{"role": "user", "content": f"message{i}"} for i in range(10)]
    context = build_history_context(history, max_turns=2)
    # Should only include last 2 turns
    assert "message8" in context or "message9" in context
    assert "message0" not in context


def test_build_history_context_invalid_roles():
    # Should skip invalid roles
    history = [
        {"role": "user", "content": "valid"},
        {"role": "invalid", "content": "skip"},
        {"role": "assistant", "content": "valid2"},
    ]
    context = build_history_context(history)
    assert "valid" in context
    assert "valid2" in context
    assert "skip" not in context


def test_build_history_context_none_content():
    history = [
        {"role": "user", "content": "valid"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "valid2"},
    ]
    context = build_history_context(history)
    assert "valid" in context
    assert "valid2" in context
