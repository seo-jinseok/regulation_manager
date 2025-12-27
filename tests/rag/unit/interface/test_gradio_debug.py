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
    )

    output = _format_query_rewrite_debug(info)

    assert "LLM" in output
    assert "캐시" in output
    assert "원문" in output
    assert "재작성" in output


def test_format_query_rewrite_debug_rules_fallback():
    info = QueryRewriteInfo(
        original="원문",
        rewritten="규칙기반",
        used=True,
        method="rules",
        from_cache=False,
        fallback=True,
    )

    output = _format_query_rewrite_debug(info)

    assert "규칙" in output
    assert "LLM 실패 폴백" in output


def test_format_query_rewrite_debug_no_change():
    info = QueryRewriteInfo(
        original="동일",
        rewritten="동일",
        used=True,
        method="llm",
        from_cache=False,
        fallback=False,
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
    )

    output = _format_query_rewrite_debug(info)

    assert "적용 안됨" in output
