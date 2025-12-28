import json

from src.rag.application.full_view_usecase import FullViewUseCase
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
                    {"type": "addendum", "display_no": "", "title": "부칙", "text": "부칙 내용", "children": []}
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


def test_get_full_view_by_rule_code(tmp_path):
    json_path = tmp_path / "reg.json"
    _write_sample_json(json_path)
    loader = JSONDocumentLoader()
    usecase = FullViewUseCase(loader, str(json_path))

    view = usecase.get_full_view("3-1-5")

    assert view is not None
    assert view.title == "교원인사규정"
    assert "제1조 목적" in view.toc
