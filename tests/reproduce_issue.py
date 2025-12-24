
import re
from typing import Dict, List, Any

# Mocking the MetadataExtractor class for reproduction
class MetadataExtractor:
    def _normalize_title(self, title: str) -> str:
        return re.sub(r'\s+', '', title)

    def _normalize_code(self, code: str) -> str:
        code = re.sub(r'[—–]', '-', code)
        code = re.sub(r'[~～].*$', '', code)
        return code.strip()

    def _extract_index_dept(self, text: str, toc_map: Dict[str, str] = None) -> Dict[str, List[Dict[str, str]]]:
        pattern = r'찾아보기\s*\n\s*<소관부서별>(.*?)(?:$)' 
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return {}
            
        content = match.group(1)
        lines = content.splitlines()
        
        dept_index: Dict[str, List[Dict[str, str]]] = {}
        current_dept = "Unknown"
        dept_index[current_dept] = []
        
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+)')
        skip_patterns = [
            re.compile(r'^\s*\|\s*'),
            re.compile(r'^[\|\-\s]+$'),
        ]
        end_patterns = [
            re.compile(r'^제\s*\d+\s*편'),
            re.compile(r'^제\s*\d+\s*장'),
            re.compile(r'.*규정집.*'),
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check skip patterns first (Matches original logic order?)
            # Wait, let's check the source file order.
            # In source:
            # if any(pat.match(line) for pat in skip_patterns):
            #     continue
            # if any(pat.match(line) for pat in end_patterns):
            #    break
            
            if any(pat.match(line) for pat in skip_patterns):
                print(f"DEBUG: Skipping line: '{line}'")
                continue

            if any(pat.match(line) for pat in end_patterns):
                print(f"DEBUG: Breaking on end pattern at line: '{line}'")
                break

            m = entry_pattern.match(line)
            if m:
                title = m.group("title").strip()
                raw_code = m.group("code")
                
                code = self._normalize_code(raw_code)
                
                dept_index[current_dept].append({
                    "title": title,
                    "rule_code": code
                })
            else:
                if "학교법인" in line and not re.search(r'\d+\s*[-—–]\s*\d+\s*[-—–]\s*\d+', line):
                    print(f"DEBUG: Breaking on '학교법인' keyword at line: '{line}'")
                    break
                # Department name or header
                current_dept = line
                if current_dept not in dept_index:
                    dept_index[current_dept] = []
        
        # Drop empty buckets
        dept_index = {k: v for k, v in dept_index.items() if v}
            
        return dept_index

# Reproduction data with table artifacts
text_data_table = """
규정집

찾아보기
<소관부서별>

학생군사교육단
학생군사교육단운영규정 5-1-24

| 제1편 |

| 학 교 법 인 |

학교법인동의학원정관 1-0-1
"""

print("--- Testing Table format ---")
extractor = MetadataExtractor()
result = extractor._extract_index_dept(text_data_table)

import json
print(json.dumps(result, indent=2, ensure_ascii=False))
