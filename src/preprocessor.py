import re
from typing import List, Optional
from .llm_client import LLMClient
from .cache_manager import CacheManager
from .repair import RegulationRepair

class Preprocessor:
    """
    Hybrid Preprocessor for Regulation Markdown.
    1. Deterministic cleaning (Regex) for obvious line breaks and artifacts.
    2. LLM-based cleaning for ambiguous paragraph merging.
    """

    def __init__(self, llm_client: LLMClient = None, cache_manager: Optional[CacheManager] = None):
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.repair_agent = None
        if self.llm_client:
            self.repair_agent = RegulationRepair(client=self.llm_client, cache_manager=self.cache_manager)

    def clean(self, text: str, verbose_callback=None) -> str:
        """
        Main cleaning pipeline.
        """
        if verbose_callback:
            verbose_callback("[dim]• HWP 불필요 요소 제거 중 (헤더, 푸터, PUA)...[/dim]")
            
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
        
        # 5. Remove XML declaration (xml version=...)
        text = re.sub(r'xml version=[^\n]+\n', '', text, flags=re.IGNORECASE)
        
        # 6. Remove long separators (underscores, dashes, special chars)
        text = re.sub(r'^[_\]W\s]{5,}$', '', text, flags=re.MULTILINE)

        # 7. Remove "동의대학교 규정집" repetitive header
        text = re.sub(r'^동의대학교\s*규정집.*$', '', text, flags=re.MULTILINE)
        
        # 8. Remove page numbers/locations and TOC lines
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'.*\d+[-—]\d+[-—]\d+.*$', '', text, flags=re.MULTILINE)
        
        # 9. Handle Private Use Area (PUA) characters
        text, removed_count = self.clean_pua(text)
        if verbose_callback and removed_count > 0:
            verbose_callback(f"[dim]  - PUA/숨겨진 문자 {removed_count}개 제거됨[/dim]")

        # 10. Collapse multiple empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def clean_pua(self, text: str) -> tuple[str, int]:
        """Replace or remove Private Use Area characters."""
        # Replace known PUA characters with standard Unicode equivalents
        text = text.replace('\uf85e', '·')   #  -> Middle Dot
        text = text.replace('\uf09e', '·')   #  -> Middle Dot
        text = text.replace('\uf0fc', '✓')   #  -> Check Mark
        
        # Remove remaining BMP Private Use Area characters (E000-F8FF)
        text, n1 = re.subn(r'[\ue000-\uf8ff]+', '', text)
        
        # Remove Supplementary Private Use Area (Plane 15/16) if present
        text, n2 = re.subn(r'[\U000F0000-\U000FFFFD]+', '', text)
        text, n3 = re.subn(r'[\U00100000-\U0010FFFD]+', '', text)
        
        return text, n1 + n2 + n3

    def _join_broken_lines_regex(self, text: str) -> str:
        """
        Join lines that are obviously broken but part of the same sentence.
        E.g. ending with a non-sentence-ending character.
        """
        # Dictionary of sentence endings (Korean common endings)
        # If a line does NOT end with these, and next line starts with text, join them.
        # This is simple/naive; careful not to merge headers.
        
        lines = text.split('\n')
        new_lines = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    new_lines.append(buffer)
                    buffer = ""
                new_lines.append("") # Keep empty lines for structure
                continue

            # Check if likely header (e.g. "제1조(목적)", "제1장", "제1절", "부칙")
            # Added Section/Chapter/Part patterns to prevent merging headers into previous text
            if (re.match(r'^제\s*\d+\s*조', line) or 
                re.match(r'^부\s*칙', line) or
                re.match(r'^제\s*\d+\s*[장절편]', line)):
                
                if buffer:
                    new_lines.append(buffer)
                    buffer = ""
                new_lines.append(line)
                continue
            
            # Check for list items (1., 가., (1), etc.)
            if re.match(r'^(\d+\.|[가-하]\.|\(\d+\)|\d+\))', line):
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
            if buffer.endswith(('.', '?', '!')):
                new_lines.append(buffer)
                buffer = ""
            
            # Heuristic: If it looks like a table row (markdown), flush
            if buffer.startswith('|'):
                new_lines.append(buffer)
                buffer = ""

        if buffer:
            new_lines.append(buffer)

        return '\n'.join(new_lines)

    def _join_paragraphs_llm(self, text: str) -> str:
        """
        Use LLM to fix semantic issues in checking if paragraphs are split.
        Now uses logical units (Articles) to maximize cache reuse.
        """
        # 1. Split into logical units (Articles, Preamble, Appendices)
        units = self._split_into_logical_units(text)
        
        processed_units = []
        print(f"    [LLM] 처리할 논리 단위: {len(units)}개 분할됨...")
        
        for i, unit in enumerate(units):
            # Skip empty units
            if not unit.strip():
                processed_units.append(unit)
                continue

            # Skip very short units (unlikely to need advanced merge logic, save tokens)
            # e.g., just a title "제1조(목적)"
            if len(unit.strip().splitlines()) <= 1:
                processed_units.append(unit)
                continue

            # --- Cache Check ---
            # We use the hash of the raw unit text as the key.
            if self.cache_manager:
                cached_resp = self.cache_manager.get_cached_llm_response(unit)
                if cached_resp is not None:
                     # cache hit
                     processed_units.append(cached_resp)
                     continue

            prompt = f"""
다음은 대학 규정 문서의 일부(조항 등)입니다. HWP에서 변환되어 줄바꿈이 불완전하거나 문맥이 끊겨 있을 수 있습니다.
다음 규칙에 따라 텍스트를 정리해주세요:

1. 문맥상 끊어진 문장은 한 줄로 이으세요.
2. '제1조(목적)'과 같은 조항 제목은 반드시 줄바꿈으로 구분하세요.
3. '①', '1.', '가.' 등의 항목 번호는 새로운 줄에서 시작하게 하세요.
4. 원문의 내용을 왜곡하거나 삭제하지 마세요. 오직 줄바꿈과 띄어쓰기만 수정하세요.
5. 결과는 오직 수정된 텍스트만 출력하세요 (부연 설명 금지).

[텍스트 시작]
{unit}
[텍스트 끝]
"""
            try:
                print(f"    [LLM] 단위 처리 중 {i+1}/{len(units)}...", end="\r", flush=True)
                response = self.llm_client.complete(prompt)
                
                # Simple cleanup
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned_response = "\n".join(lines)
                
                # --- Cache Save ---
                if self.cache_manager:
                    self.cache_manager.cache_llm_response(unit, cleaned_response)
                
                processed_units.append(cleaned_response)
                
            except Exception as e:
                print(f"\n    [LLM] 경고: 단위 {i+1} 처리 실패 ({e}). 원본 사용.")
                processed_units.append(unit)
        
        print(f"\n    [LLM] 처리 완료.")
        return "\n".join(processed_units)

    def _split_into_logical_units(self, text: str) -> List[str]:
        """
        Split text into Preamble, Articles, and Appendices based on headers.
        This ensures that edits in one article don't shift the chunks for others, preserving cache.
        """
        lines = text.splitlines()
        units = []
        current_unit = []

        for line in lines:
            # Check for Article Header (e.g., "제1조", "제 2 조")
            # We want to start a new unit *before* the header, unless it's the very first line.
            is_header = re.match(r'^제\s*\d+\s*조', line.strip()) or re.match(r'^부\s*칙', line.strip())
            
            if is_header and current_unit:
                units.append("\n".join(current_unit))
                current_unit = []
            
            current_unit.append(line)
        
        if current_unit:
            units.append("\n".join(current_unit))
            
        return units