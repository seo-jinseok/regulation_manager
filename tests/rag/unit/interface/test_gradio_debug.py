from src.rag.application.search_usecase import QueryRewriteInfo
from src.rag.interface.gradio_app import _format_query_rewrite_debug


def test_format_query_rewrite_debug_llm_cache():
    info = QueryRewriteInfo(
        original="원문",
        rewritten="재작성",
        used=True,
        method="llm",
        from_cache=True,
        fallback=False,
        used_synonyms=True,
        used_intent=False,
        matched_intents=["근무 회피"],
    )

    output = _format_query_rewrite_debug(info)

    assert "LLM" in output
    assert "캐시" in output
    assert "원문" in output
    assert "재작성" in output
    assert "동의어 사전" in output
    # matched_intents는 있지만 used_intent=False이므로 의도 인식이 미매칭으로 표시됨


def test_format_query_rewrite_debug_rules_fallback():
    info = QueryRewriteInfo(
        original="원문",
        rewritten="규칙기반",
        used=True,
        method="rules",
        from_cache=False,
        fallback=True,
        used_synonyms=False,
        used_intent=True,
        matched_intents=["근무 회피"],
    )

    output = _format_query_rewrite_debug(info)

    assert "규칙" in output
    assert "폴백" in output  # "LLM 실패→폴백" 형태로 표시
    assert "의도 인식" in output
    assert "근무 회피" in output


def test_format_query_rewrite_debug_no_change():
    info = QueryRewriteInfo(
        original="동일",
        rewritten="동일",
        used=True,
        method="llm",
        from_cache=False,
        fallback=False,
        used_synonyms=False,
        used_intent=False,
        matched_intents=[],
    )

    output = _format_query_rewrite_debug(info)

    assert "변경 없음" in output


def test_format_query_rewrite_debug_unused():
    info = QueryRewriteInfo(
        original="원문",
        rewritten="원문",
        used=False,
        method=None,
        from_cache=False,
        fallback=False,
        used_synonyms=None,
        used_intent=None,
        matched_intents=None,
    )

    output = _format_query_rewrite_debug(info)

    assert "미적용" in output  # "쿼리 리라이팅 미적용" 형태로 표시
