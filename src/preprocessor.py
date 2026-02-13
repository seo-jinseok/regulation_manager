import json
import os
import re
from pathlib import Path
from typing import Optional

from .cache_manager import CacheManager
from .llm_client import LLMClient
from .repair import RegulationRepair


def clean_page_header_pattern(text: str) -> str:
    """
    페이지 헤더 패턴을 제거하여 깨진 제목 복구.

    HWP → HTML → Markdown 변환 시 페이지 헤더가 제목에 섞여 들어가는 문제 해결.
    예: "제3편 행정 3—1—120～연구율리센터규정" → "연구율리센터규정"

    Args:
        text: 원본 텍스트 (변환된 Markdown 라인)

    Returns:
        페이지 헤더가 제거된 텍스트
    """
    if not text:
        return text

    # 규정 제목에 붙은 페이지 헤더만 제거 (TOC/INDEX는 건드리지 않음)
    # 규정 제목 패턴: 규정, 세칙, 지침 등으로 끝나는 단어
    # 뒤에 페이지 헤더가 붙어 있으면 제거

    # 패턴: "[규정유형] 페이지헤더[공백]규정유형"
    # 예: "연구율리센터규정 3—1—120～연구율리센터규정" → "연구율리센터규정"
    regulation_types = [
        '규정', '세칙', '지침', '요령', '강령', '내규', '학칙', '헌장',
        '기준', '수칙', '준칙', '요강', '운영', '정관', '시행세칙', '운영세칙',
        '운영지침', '시행지침', '안전수칙', '소프트웨어'
    ]

    # 규정 유형으로 끝나는지 확인
    for reg_type in regulation_types:
        # 패턴: "{reg_type} 페이지헤더 {reg_type}" (중복 제거)
        # 예: "연구율리센터규정 3—1—120～연구율리센터규정" → "연구율리센터규정"
        pattern = rf'({reg_type})\s+\d+[—－]\d+[—－]\d+～\s*\1'
        if re.search(pattern, text):
            text = re.sub(pattern, r'\1', text)
            break

    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class Preprocessor:
    """
    Hybrid Preprocessor for Regulation Markdown.
    1. Deterministic cleaning (Regex) for obvious line breaks and artifacts.
    2. LLM-based cleaning for ambiguous paragraph merging.
    """

    def __init__(
        self, llm_client: Optional[LLMClient] = None, cache_manager: Optional[CacheManager] = None
    ):
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.cache_namespace = None
        self._load_rules()
        if self.llm_client and hasattr(self.llm_client, "cache_namespace"):
            self.cache_namespace = self.llm_client.cache_namespace()
        self.repair_agent = None
        if self.llm_client:
            self.repair_agent = RegulationRepair(
                client=self.llm_client,
                cache_manager=self.cache_manager,
                cache_namespace=self.cache_namespace,
            )

    def _default_rules(self) -> dict:
        return {
            "inline_remove_patterns": [
                r"xml version=[^\n]+\n",
            ],
            "separator_line_pattern": r"^[_\-=~\s]{5,}$",
            "line_remove_patterns": [
                r"^동의대학교\s*규정집.*$",
                r"^\s*-\s*\d+\s*-\s*$",
                r"^\s*\|?\s*제\s*\d+\s*[편장절관].*?\b\d+[-—–]\d+[-—–]\d+\s*[~～]\s*\d+\s*\|?\s*$",
                r"^\s*\|?\s*\d+[-—–]\d+[-—–]\d+\s*\|?\s*$",
            ],
        }

    def _load_rules(self) -> None:
        rules = self._default_rules()
        rules_path = os.getenv("PREPROCESSOR_RULES_PATH")
        if rules_path:
            path = Path(rules_path)
        else:
            path = Path("data/config/preprocessor_rules.json")
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    rules["inline_remove_patterns"] = data.get(
                        "inline_remove_patterns", rules["inline_remove_patterns"]
                    )
                    rules["separator_line_pattern"] = data.get(
                        "separator_line_pattern", rules["separator_line_pattern"]
                    )
                    rules["line_remove_patterns"] = data.get(
                        "line_remove_patterns", rules["line_remove_patterns"]
                    )
            except (json.JSONDecodeError, OSError, KeyError) as e:
                # Rules file is optional; use defaults if loading fails
                import logging

                logging.getLogger(__name__).debug(
                    f"Failed to load preprocessor rules: {e}"
                )
        self.inline_remove_patterns = rules["inline_remove_patterns"]
        self.separator_line_pattern = rules["separator_line_pattern"]
        self.line_remove_patterns = rules["line_remove_patterns"]

    def clean(self, text: str, verbose_callback=None) -> str:
        """
        Main cleaning pipeline.
        """
        if verbose_callback:
            verbose_callback(
                "[dim]• HWP 불필요 요소 제거 중 (헤더, 푸터, PUA)...[/dim]"
            )

        text = self._remove_artifacts(text, verbose_callback)

        if verbose_callback:
            verbose_callback("[dim]• 끊어진 줄 연결 중 (Regex)...[/dim]")

        text = self._join_broken_lines_regex(text)

        if self.llm_client and self.repair_agent:
            if verbose_callback:
                verbose_callback("[dim]• LLM으로 문단 처리 중...[/dim]")
            text = self.repair_agent.repair_broken_lines(text)

        return text

    def _remove_artifacts(self, text: str, verbose_callback=None) -> str:
        """Remove headers, footers, page numbers, and hwp artifacts using Regex."""

        # 5. Remove inline artifacts
        for pattern in self.inline_remove_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 6. Remove long separators (underscores, dashes, special chars)
        text = re.sub(self.separator_line_pattern, "", text, flags=re.MULTILINE)

        # 7. Remove header/footer artifacts by line patterns
        for pattern in self.line_remove_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        # 9. Handle Private Use Area (PUA) characters
        text, removed_count = self.clean_pua(text)
        if verbose_callback and removed_count > 0:
            verbose_callback(f"[dim]  - PUA/숨겨진 문자 {removed_count}개 제거됨[/dim]")

        # 10. Collapse multiple empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def clean_pua(self, text: str) -> tuple[str, int]:
        """Replace or remove Private Use Area characters."""
        # Replace known PUA characters with standard Unicode equivalents
        text = text.replace("\uf85e", "·")  #  -> Middle Dot
        text = text.replace("\uf09e", "·")  #  -> Middle Dot
        text = text.replace("\uf0fc", "✓")  #  -> Check Mark

        # Remove remaining BMP Private Use Area characters (E000-F8FF)
        text, n1 = re.subn(r"[\ue000-\uf8ff]+", "", text)

        # Remove Supplementary Private Use Area (Plane 15/16) if present
        text, n2 = re.subn(r"[\U000F0000-\U000FFFFD]+", "", text)
        text, n3 = re.subn(r"[\U00100000-\U0010FFFD]+", "", text)

        return text, n1 + n2 + n3

    def _join_broken_lines_regex(self, text: str) -> str:
        """
        Join lines that are obviously broken but part of the same sentence.
        E.g. ending with a non-sentence-ending character.
        """
        # Dictionary of sentence endings (Korean common endings)
        # If a line does NOT end with these, and next line starts with text, join them.
        # This is simple/naive; careful not to merge headers.

        lines = text.split("\n")
        new_lines = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    new_lines.append(buffer)
                    buffer = ""
                new_lines.append("")  # Keep empty lines for structure
                continue

            # Check if likely header (e.g. "제1조(목적)", "제1장", "제1절", "부칙")
            # Added Section/Chapter/Part patterns to prevent merging headers into previous text
            if (
                re.match(r"^제\s*\d+\s*조", line)
                or re.match(r"^부\s*칙", line)
                or re.match(r"^제\s*\d+\s*[장절편]", line)
            ):
                if buffer:
                    new_lines.append(buffer)
                    buffer = ""
                new_lines.append(line)
                continue

            # Check for list items (1., 가., (1), etc.)
            if re.match(r"^(\d+\.|[가-하]\.|\(\d+\)|\d+\))", line):
                if buffer:
                    new_lines.append(buffer)
                    buffer = ""
                new_lines.append(line)
                continue

            if buffer:
                # Previous line in buffer. Join with space.
                buffer += " " + line
            else:
                buffer = line

            # Decide if we should flush buffer
            # If ends with ., ?, !, then likely end of sentence.
            if buffer.endswith((".", "?", "!")):
                new_lines.append(buffer)
                buffer = ""

            # Heuristic: If it looks like a table row (markdown), flush
            if buffer.startswith("|"):
                new_lines.append(buffer)
                buffer = ""

        if buffer:
            new_lines.append(buffer)

        return "\n".join(new_lines)
