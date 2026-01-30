import json
from unittest.mock import patch

from src.rag.application.full_view_usecase import FullViewUseCase
from src.rag.config import get_config, reset_config
from src.rag.infrastructure.json_loader import JSONDocumentLoader


def _write_sample_json(path):
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "목적",
                        "text": "내용",
                        "children": [],
                    }
                ],
                "addenda": [
                    {
                        "type": "addendum",
                        "display_no": "",
                        "title": "부칙",
                        "text": "부칙 내용",
                        "children": [],
                    }
                ],
            },
            {
                "doc_type": "regulation",
                "title": "교원복무규정",
                "metadata": {"rule_code": "3-1-6"},
                "content": [],
                "addenda": [],
            },
            {"doc_type": "toc", "title": "차례", "content": [], "addenda": []},
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_find_matches_exact_title(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    matches = usecase.find_matches("교원인사규정 전문")

    assert len(matches) == 1
    assert matches[0].title == "교원인사규정"


def test_find_matches_ambiguous(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    matches = usecase.find_matches("교원 규정 전문")

    assert len(matches) == 2


def test_find_matches_prefers_exact_over_prefix(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [],
                "addenda": [],
            },
            {
                "doc_type": "regulation",
                "title": "JA교원인사규정",
                "metadata": {"rule_code": "3-1-133"},
                "content": [],
                "addenda": [],
            },
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    matches = usecase.find_matches("교원인사규정 별첨")

    assert len(matches) == 1
    assert matches[0].title == "교원인사규정"


def test_get_full_view_by_rule_code(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    view = usecase.get_full_view("3-1-5")

    assert view is not None
    assert view.title == "교원인사규정"
    assert "제1조 목적" in view.toc


def test_find_matches_uses_sync_state(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    json_path = output_dir / "reg.json"
    _write_sample_json(json_path)

    sync_state_path = tmp_path / "sync_state.json"
    sync_state_path.write_text(
        json.dumps({"json_file": json_path.name}, ensure_ascii=False),
        encoding="utf-8",
    )

    reset_config()
    config = get_config()
    config.json_path = str(tmp_path / "missing.json")
    config.sync_state_path = str(sync_state_path)

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader)
    matches = usecase.find_matches("교원인사규정 전문")

    reset_config()

    assert matches
    assert matches[0].title == "교원인사규정"


def test_toc_skips_paragraphs_and_addendum_items(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "목적",
                        "text": "내용",
                        "children": [
                            {
                                "type": "paragraph",
                                "display_no": "①",
                                "title": "",
                                "text": "항 내용",
                                "children": [],
                            },
                            {
                                "type": "item",
                                "display_no": "1.",
                                "title": "",
                                "text": "호 내용",
                                "children": [],
                            },
                        ],
                    }
                ],
                "addenda": [
                    {
                        "type": "addendum",
                        "display_no": "",
                        "title": "부칙",
                        "text": "",
                        "children": [
                            {
                                "type": "addendum_item",
                                "display_no": "1.",
                                "title": "경과조치",
                                "text": "부칙 내용",
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    view = usecase.get_full_view("3-1-5")

    assert view is not None
    assert "제1조 목적" in view.toc
    assert "부칙" in view.toc
    assert all("①" not in label for label in view.toc)
    assert all("1." not in label for label in view.toc)
    assert all("경과조치" not in label for label in view.toc)


def test_find_tables_prefers_labeled_matches(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "①",
                        "title": "",
                        "text": "별표 1 기준은 다음과 같다.\n[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| A | B |\\n| --- | --- |\\n| 1 | 2 |",
                                }
                            ]
                        },
                        "children": [],
                    },
                    {
                        "type": "paragraph",
                        "display_no": "②",
                        "title": "",
                        "text": "기타 기준.\n[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| C | D |\\n| --- | --- |\\n| 3 | 4 |",
                                }
                            ]
                        },
                        "children": [],
                    },
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("3-1-5", table_no=1)

    assert len(tables) == 1
    assert "| A | B |" in tables[0].markdown


def test_find_tables_fallback_to_placeholder(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "①",
                        "title": "",
                        "text": "기준은 다음과 같다.\n[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| A | B |\\n| --- | --- |\\n| 1 | 2 |",
                                }
                            ]
                        },
                        "children": [],
                    },
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("교원인사규정", table_no=1)

    assert len(tables) == 1
    assert "| A | B |" in tables[0].markdown


def test_find_tables_matches_label_in_path(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "",
                        "title": "연구실적 인정기준 및 인정률",
                        "text": "[TABLE:1]",
                        "parent_path": ["교원인사규정", "부칙", "별표 1"],
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| A | B |\\n| --- | --- |\\n| 1 | 2 |",
                                }
                            ]
                        },
                        "children": [],
                    }
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("교원인사규정", table_no=1)

    assert len(tables) == 1
    assert "연구실적 인정기준 및 인정률" in tables[0].title


def test_find_tables_prefers_attached_files_for_label(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "",
                        "title": "",
                        "text": "별표 1 기준은 다음과 같다.\n[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| BODY | TABLE |\\n| --- | --- |\\n| 1 | 2 |",
                                }
                            ]
                        },
                        "children": [],
                    }
                ],
                "addenda": [],
                "attached_files": [
                    {
                        "title": "[별표 1]",
                        "text": "연구실적 인정기준 및 인정률\\n| ATTACHED | TABLE |\\n| --- | --- |\\n| 3 | 4 |",
                    }
                ],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("3-1-5", table_no=1, label_variants=["별표"])

    assert len(tables) == 1
    assert "ATTACHED" in tables[0].markdown


def test_find_tables_matches_addendum_label(tmp_path):
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "①",
                        "title": "",
                        "text": "별첨 1 기준은 다음과 같다.\n[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| A | B |\\n| --- | --- |\\n| 1 | 2 |",
                                }
                            ]
                        },
                        "children": [],
                    }
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("3-1-5", table_no=1, label_variants=["별첨"])

    assert len(tables) == 1
    assert "| A | B |" in tables[0].markdown


def test_find_matches_strips_view_keyword(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # "교원인사규정 전문 보기" -> "교원인사규정"
    matches = usecase.find_matches("교원인사규정 전문 보기")

    assert len(matches) == 1
    assert matches[0].title == "교원인사규정"


def test_find_matches_strips_article_reference(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # "교원인사규정 제1조" -> "교원인사규정"
    matches = usecase.find_matches("교원인사규정 제1조")

    assert len(matches) == 1
    assert matches[0].title == "교원인사규정"


def test_get_article_view_by_rule_code(tmp_path):
    """Test getting a specific article by rule code and article number."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "목적",
                        "text": "목적 내용",
                        "children": [],
                    },
                    {
                        "type": "article",
                        "display_no": "제 2 조",
                        "title": "정의",
                        "text": "정의 내용",
                        "children": [],
                    },
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # Test finding article 1
    article = usecase.get_article_view("3-1-5", 1)
    assert article is not None
    assert article["display_no"] == "제1조"
    assert article["title"] == "목적"

    # Test finding article 2 with space formatting
    article = usecase.get_article_view("3-1-5", 2)
    assert article is not None
    assert article["title"] == "정의"


def test_get_article_view_in_addenda(tmp_path):
    """Test finding article in addenda section."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [],
                "addenda": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "부칙 목적",
                        "text": "부칙 내용",
                        "children": [],
                    }
                ],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    article = usecase.get_article_view("3-1-5", 1)
    assert article is not None
    assert article["title"] == "부칙 목적"


def test_get_article_view_not_found(tmp_path):
    """Test getting non-existent article returns None."""
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    article = usecase.get_article_view("3-1-5", 99)
    assert article is None


@patch("src.rag.application.full_view_usecase.FullViewUseCase._resolve_json_path")
def test_get_article_view_no_json_path(mock_resolve_json_path, tmp_path):
    """Test get_article_view with no valid json path."""
    mock_resolve_json_path.return_value = None

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(tmp_path / "nonexistent.json"))

    article = usecase.get_article_view("3-1-5", 1)
    assert article is None


def test_get_chapter_node_by_number(tmp_path):
    """Test getting a chapter node by chapter number."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "chapter",
                        "display_no": "제1장",
                        "title": "총칙",
                        "children": [],
                    },
                    {
                        "type": "chapter",
                        "display_no": "제 2 장",
                        "title": "인사",
                        "children": [],
                    },
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # Get existing doc for chapter test
    doc = data["docs"][0]

    chapter = usecase.get_chapter_node(doc, 1)
    assert chapter is not None
    assert chapter["display_no"] == "제1장"
    assert chapter["title"] == "총칙"

    # Test with space formatting
    chapter = usecase.get_chapter_node(doc, 2)
    assert chapter is not None
    assert chapter["title"] == "인사"


def test_get_chapter_node_not_found(tmp_path):
    """Test getting non-existent chapter returns None."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "chapter",
                        "display_no": "제1장",
                        "title": "총칙",
                        "children": [],
                    }
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    doc = data["docs"][0]
    chapter = usecase.get_chapter_node(doc, 99)
    assert chapter is None


def test_get_chapter_node_none_doc(tmp_path):
    """Test get_chapter_node with None doc."""
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader)

    chapter = usecase.get_chapter_node(None, 1)
    assert chapter is None


def test_find_tables_no_label_variants(tmp_path):
    """Test find_tables with no label variants."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "type": "paragraph",
                        "display_no": "",
                        "title": "",
                        "text": "[TABLE:1]",
                        "metadata": {
                            "tables": [
                                {
                                    "format": "markdown",
                                    "markdown": "| A | B |\\n| --- | --- |",
                                }
                            ]
                        },
                        "children": [],
                    }
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("3-1-5")

    assert len(tables) == 1
    assert "| A | B |" in tables[0].markdown


def test_find_tables_no_results(tmp_path):
    """Test find_tables returns empty list when no tables found."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {"type": "paragraph", "display_no": "", "title": "", "text": "내용"}
                ],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    tables = usecase.find_tables("3-1-5", table_no=1)
    assert tables == []


def test_clean_attachment_title():
    """Test _clean_attachment_title static method."""
    result = FullViewUseCase._clean_attachment_title("[별표 1]")
    assert result == "별표 1"

    result = FullViewUseCase._clean_attachment_title("<별지2>")
    assert result == "별지2"

    result = FullViewUseCase._clean_attachment_title("")
    assert result == ""


def test_extract_attachment_label():
    """Test _extract_attachment_label static method."""
    label, number = FullViewUseCase._extract_attachment_label("별표 제1호")
    assert label == "별표"
    assert number == 1

    label, number = FullViewUseCase._extract_attachment_label("별지2")
    assert label == "별지"
    assert number == 2

    label, number = FullViewUseCase._extract_attachment_label("별표")
    assert label == "별표"
    assert number is None

    label, number = FullViewUseCase._extract_attachment_label("")
    assert label == ""
    assert number is None


def test_normalize_static_method():
    """Test _normalize static method."""
    result = FullViewUseCase._normalize("교 원 인 사")
    assert result == "교원인사"

    result = FullViewUseCase._normalize("  test  ")
    assert result == "test"


def test_tokenize_static_method():
    """Test _tokenize static method."""
    # The regex matches consecutive Korean characters
    result = FullViewUseCase._tokenize("교원인사규정")
    assert result == ["교원인사규정"]

    # Text with spaces is split into separate Korean sequences
    result = FullViewUseCase._tokenize("교원 인사 규정")
    assert "교원" in result
    assert "인사" in result
    assert "규정" in result

    # English text is ignored
    result = FullViewUseCase._tokenize("English words")
    assert result == []


def test_infer_rule_code_from_metadata(tmp_path):
    """Test _infer_rule_code from document metadata."""
    json_path = tmp_path / "reg.json"
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    doc = {"metadata": {"rule_code": "TEST-001"}, "content": []}
    result = usecase._infer_rule_code(doc)
    assert result == "TEST-001"


def test_infer_rule_code_from_content_nodes(tmp_path):
    """Test _infer_rule_code from content nodes."""
    json_path = tmp_path / "reg.json"
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    doc = {
        "metadata": {},
        "content": [{"metadata": {"rule_code": "CONTENT-001"}}],
    }
    result = usecase._infer_rule_code(doc)
    assert result == "CONTENT-001"


def test_infer_rule_code_not_found(tmp_path):
    """Test _infer_rule_code when not found."""
    json_path = tmp_path / "reg.json"
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    doc = {"metadata": {}, "content": []}
    result = usecase._infer_rule_code(doc)
    assert result == ""


def test_is_valid_json_file(tmp_path):
    """Test _is_valid_json_file filters out plan and metadata files."""
    json_path = tmp_path / "reg.json"
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # Valid regulation file
    assert usecase._is_valid_json_file("regulation.json") is True

    # Plan files
    assert usecase._is_valid_json_file("regulation_plan.json") is False

    # Improvement files
    assert usecase._is_valid_json_file("improvement_001.json") is False

    # Generated queries
    assert usecase._is_valid_json_file("generated_queries_v1.json") is False

    # Metadata files
    assert usecase._is_valid_json_file("regulation_metadata.json") is False


def test_get_full_view_returns_none_on_error(tmp_path):
    """Test get_full_view returns None when loader raises exception."""
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)

    class FailingLoader:
        def get_regulation_doc(self, json_path, identifier):
            raise RuntimeError("Load failed")

        def get_all_regulations(self, json_path):
            return []

    loader = FailingLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    view = usecase.get_full_view("3-1-5")
    assert view is None


@patch("src.rag.application.full_view_usecase.FullViewUseCase._resolve_json_path")
def test_get_full_view_no_json_path(mock_resolve_json_path, tmp_path):
    """Test get_full_view with no valid json path."""
    mock_resolve_json_path.return_value = None

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(tmp_path / "nonexistent.json"))

    view = usecase.get_full_view("3-1-5")
    assert view is None


def test_find_matches_empty_regulations(tmp_path):
    """Test find_matches returns empty list when no regulations."""
    json_path = tmp_path / "reg.json"
    data = {"docs": []}
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    matches = usecase.find_matches("test")
    assert matches == []


def test_find_matches_loader_exception(tmp_path):
    """Test find_matches handles loader exceptions."""

    class FailingLoader:
        def get_all_regulations(self, json_path):
            raise RuntimeError("Load failed")

    loader = FailingLoader()
    usecase = FullViewUseCase(loader, str(tmp_path / "test.json"))

    matches = usecase.find_matches("test")
    assert matches == []


def test_find_matches_token_based_scoring(tmp_path):
    """Test find_matches uses token-based scoring."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    # All tokens match (score 2)
    matches = usecase.find_matches("교원인사")
    assert len(matches) == 1
    assert matches[0].score >= 2

    # Partial token match (score 1)
    matches = usecase.find_matches("교원")
    assert len(matches) == 1
    assert matches[0].score >= 1


def test_find_matches_no_match(tmp_path):
    """Test find_matches returns empty when no match found."""
    json_path = tmp_path / "reg.json"
    data = {
        "docs": [
            {
                "doc_type": "regulation",
                "title": "교원인사규정",
                "metadata": {"rule_code": "3-1-5"},
                "content": [],
                "addenda": [],
            }
        ]
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    matches = usecase.find_matches("완전다른단어")
    assert matches == []
