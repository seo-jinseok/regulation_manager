from src.rag.interface.chat_logic import (
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
    assert is_followup_message("휴학 절차는?") is False


def test_expand_followup_query():
    context = "교원인사규정"
    assert expand_followup_query("전문 보여줘", context) == "교원인사규정 전문 보여줘"
    assert expand_followup_query("교원복무규정 전문", context) == "교원복무규정 전문"
    assert expand_followup_query("휴학 절차는?", context) == "휴학 절차는?"
