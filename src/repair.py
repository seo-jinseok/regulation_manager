import re
from src.llm_client import LLMClient

class RegulationRepair:
    def __init__(self, provider="openai", client=None, cache_manager=None):
        self.cache_manager = cache_manager
        if client:
            self.client = client
        else:
            try:
                self.client = LLMClient(provider=provider)
            except Exception as e:
                print(f"Warning: Failed to initialize LLMClient: {e}")
                self.client = None

    def repair_broken_lines(self, text: str) -> str:
        """
        Use LLM to fix semantic issues (broken lines, merging paragraphs).
        Uses logical units (Articles) to maximize cache reuse.
        """
        if not self.client:
            return text

        units = self._split_into_logical_units(text)
        processed_units = []
        
        for i, unit in enumerate(units):
            if not unit.strip():
                processed_units.append(unit)
                continue
            
            # Skip very short units
            if len(unit.strip().splitlines()) <= 1:
                processed_units.append(unit)
                continue

            if self.cache_manager:
                cached = self.cache_manager.get_cached_llm_response(unit)
                if cached: 
                    processed_units.append(cached)
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
                response = self.client.complete(prompt)
                cleaned = response.strip()
                # Remove markdown code blocks if any
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines and lines[-1].startswith("```"): lines = lines[:-1]
                    cleaned = "\n".join(lines)
                
                if self.cache_manager:
                    self.cache_manager.cache_llm_response(unit, cleaned)
                processed_units.append(cleaned)
            except Exception as e:
                print(f"Warning: Repair failed for unit {i}: {e}")
                processed_units.append(unit)
                
        return "\n".join(processed_units)

    def _split_into_logical_units(self, text: str) -> list[str]:
        lines = text.splitlines()
        units = []
        current_unit = []
        for line in lines:
            is_header = re.match(r'^제\s*\d+\s*조', line.strip()) or re.match(r'^부\s*칙', line.strip())
            if is_header and current_unit:
                units.append("\n".join(current_unit))
                current_unit = []
            current_unit.append(line)
        if current_unit:
            units.append("\n".join(current_unit))
        return units
