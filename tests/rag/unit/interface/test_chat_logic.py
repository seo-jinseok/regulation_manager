from src.rag.interface.chat_logic import (
    attachment_label_variants,
    build_history_context,
    extract_regulation_title,
    parse_attachment_request,
    expand_followup_query,
    has_explicit_target,
    is_followup_message,
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
    assert extract_regulation_title("중앙도서관자료제적에관한세칙 별표 1") == "중앙도서관자료제적에관한세칙"


def test_parse_attachment_request_with_regulation():
    result = parse_attachment_request("교원인사규정의 별첨 자료 1번", None)
    assert result == ("교원인사규정", 1, "별첨")


def test_parse_attachment_request_with_fallback():
    result = parse_attachment_request("별표 2 보여줘", "교원인사규정")
    assert result == ("교원인사규정", 2, "별표")


def test_attachment_label_variants():
    assert attachment_label_variants("별첨") == ["별첨", "별표"]
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
