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


class TestEvasiveResponsePreventionGuidelines:
    """Test evasive response prevention guidelines in prompts (SPEC-RAG-Q-011)."""

    def test_evasive_prevention_section_exists(self):
        """프롬프트에 회피성 답변 방지 섹션이 있어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "회피성 답변 방지" in regulation_qa, "Missing evasive response prevention section"

    def test_evasive_prevention_has_specific_citation_guideline(self):
        """구체적 인용 가이드라인이 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "구체적 인용" in regulation_qa, "Missing specific citation guideline"

    def test_evasive_prevention_has_vague_expression_prohibition(self):
        """모호한 표현 금지 가이드라인이 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "모호한 표현" in regulation_qa, "Missing vague expression prohibition"
        assert "일반적으로" in regulation_qa, "Missing '일반적으로' in vague expressions"

    def test_evasive_prevention_has_info_absence_guideline(self):
        """정보 부재 시 명확한 안내 가이드라인이 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "찾을 수 없습니다" in regulation_qa, "Missing info absence guideline"

    def test_evasive_prevention_has_partial_info_guideline(self):
        """부분 정보 제공 가이드라인이 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "부분 정보" in regulation_qa or "부분적" in regulation_qa, "Missing partial info guideline"

    def test_evasive_patterns_detection_warning(self):
        """회피성 패턴 감지 경고가 포함되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        # Check for evasive pattern detection warning
        assert "회피성 패턴" in regulation_qa or "회피성 답변" in regulation_qa, "Missing evasive pattern detection warning"

    def test_homepage_deflection_prohibited(self):
        """홈페이지 참고 유도가 금지되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "홈페이지" in regulation_qa, "Missing homepage deflection prohibition"

    def test_department_deflection_prohibited(self):
        """부서 문의 유도가 금지되어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        regulation_qa = data["regulation_qa"]["prompt"]
        assert "부서" in regulation_qa, "Missing department deflection prohibition"

    def test_prompt_version_updated(self):
        """프롬프트 버전이 2.5 이상이어야 함."""
        prompts_path = Path("data/config/prompts.json")
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)

        version = data["regulation_qa"]["version"]
        major, minor = map(int, version.split("."))
        assert (major, minor) >= (2, 5), f"Prompt version should be >= 2.5, got {version}"
