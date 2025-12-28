from src.rag.interface.common import decide_search_mode


def test_decide_search_mode_full_view():
    assert decide_search_mode("교원인사규정 전문") == "full_view"


def test_decide_search_mode_force_overrides_full_view():
    assert decide_search_mode("교원인사규정 전문", force_mode="search") == "search"
