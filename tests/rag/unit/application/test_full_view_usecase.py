import json

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
