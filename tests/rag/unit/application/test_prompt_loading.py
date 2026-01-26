"""Tests for prompt loading functionality."""

import json
from pathlib import Path


class TestPromptLoading:
    """Test prompt loading from prompts.json."""

    def test_prompts_json_exists(self):
        """prompts.json 파일이 존재해야 함."""
        prompts_path = Path("data/config/prompts.json")
        assert prompts_path.exists(), f"prompts.json not found at {prompts_path}"

    def test_prompts_json_valid_format(self):
        """prompts.json이 유효한 JSON 형식이어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_prompts_json_has_regulation_qa(self):
        """prompts.json에 regulation_qa 프롬프트가 있어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "regulation_qa" in data
        # Nested structure: {"regulation_qa": {"prompt": "...", "description": "..."}}
        assert isinstance(data["regulation_qa"], dict)
        assert "prompt" in data["regulation_qa"]
        assert len(data["regulation_qa"]["prompt"]) > 100  # 충분한 내용이 있어야 함

    def test_prompts_json_has_function_gemma_system(self):
        """prompts.json에 function_gemma_system 프롬프트가 있어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "function_gemma_system" in data

    def test_load_prompt_function(self):
        """_load_prompt 함수가 정상 동작해야 함."""
        from src.rag.application.search_usecase import _load_prompt

        prompt = _load_prompt("regulation_qa")
        assert prompt is not None
        assert "규정" in prompt or "대학" in prompt

    def test_load_prompt_fallback(self):
        """존재하지 않는 키는 빈 문자열을 반환해야 함."""
        from src.rag.application.search_usecase import _load_prompt

        prompt = _load_prompt("nonexistent_key")
        assert prompt == ""

    def test_regulation_qa_prompt_loaded(self):
        """REGULATION_QA_PROMPT가 로드되어야 함."""
        from src.rag.application.search_usecase import REGULATION_QA_PROMPT

        assert REGULATION_QA_PROMPT is not None
        assert len(REGULATION_QA_PROMPT) > 0
        # 핵심 키워드 확인
        assert "규정" in REGULATION_QA_PROMPT or "대학" in REGULATION_QA_PROMPT

    def test_prompt_content_no_hallucination_instruction(self):
        """프롬프트에 hallucination 방지 지침이 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        # 주요 지침 키워드 확인
        assert any(
            keyword in regulation_qa for keyword in ["추측", "명시", "알 수 없", "금지"]
        )
