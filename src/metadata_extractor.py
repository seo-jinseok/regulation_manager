import re
from typing import Dict, List, Any

class MetadataExtractor:
    """
    Extracts Table of Contents (TOC) and Index (Alpha/Dept) from raw markdown.
    """

    def extract(self, text: str) -> Dict[str, Any]:
        toc_list = self._extract_toc(text)
        
        # Build Source of Truth map from TOC
        # Normalize titles by removing spaces for better matching
        # Key: Normalized Title, Value: Clean Rule Code
        toc_map = {self._normalize_title(item["title"]): item["rule_code"] for item in toc_list}
        
        return {
            "toc": toc_list,
            "index_by_alpha": self._extract_index_alpha(text, toc_map),
            "index_by_dept": self._extract_index_dept(text, toc_map)
        }

    def _normalize_title(self, title: str) -> str:
        """Strip spaces and special chars for fuzzy matching."""
        return re.sub(r'\s+', '', title)

    def _normalize_code(self, code: str) -> str:
        """Standardize dashes and remove page ranges."""
        # Replace Em Dash, En Dash, etc with Hyphen
        code = re.sub(r'[—–]', '-', code)
        # Remove trailing page range starting with ~ or ～
        code = re.sub(r'[~～].*$', '', code)
        return code.strip()

    def _extract_toc(self, text: str) -> List[Dict[str, str]]:
        # Find explicit TOC section if possible
        # Pattern: Starts with "차 례" or "목 차"
        match = re.search(r'^\s*(차\s*례|목\s*차)\s*$', text, re.MULTILINE)
        if not match:
            return []
        
        start_idx = match.end()
        lines = text[start_idx:].splitlines()
        toc = []
        
        # Regex: (Title) .... (Code)
        # Capture code digits and dashes, ignore trailing page range like ~3
        # Strict pattern for TOC to ensure high quality
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+)(?:[~～]\d+)?\s*$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Stop if we hit explicit Index
            if "찾아보기" in line:
                break
                
            m = entry_pattern.match(line)
            if m:
                raw_code = m.group("code")
                toc.append({
                    "title": m.group("title").strip(),
                    "rule_code": self._normalize_code(raw_code)
                })
            
        return toc

    def _extract_index_alpha(self, text: str, toc_map: Dict[str, str] = None) -> List[Dict[str, str]]:
        # Look for "찾아보기" -> "<가나다순>"
        pattern = r'찾아보기\s*\n\s*<가나다순>(.*?)(?:찾아보기|This is the end|$)'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return []
            
        content = match.group(1)
        return self._parse_entries(content, toc_map)

    def _extract_index_dept(self, text: str, toc_map: Dict[str, str] = None) -> Dict[str, List[Dict[str, str]]]:
        # Look for "찾아보기" -> "<소관부서별>"
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
            re.compile(r'^\s*\|\s*'),  # table rows like | --- |
            re.compile(r'^[\|\-\s]+$'),  # divider lines
        ]
        end_patterns = [
            re.compile(r'^[\|\s]*제\s*\d+\s*편'),  # part header with optional table chars
            re.compile(r'^[\|\s]*제\s*\d+\s*장'),  # chapter header
            re.compile(r'.*규정집.*'),
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for end patterns FIRST, before skipping table artifacts
            if any(pat.match(line) for pat in end_patterns):
                break

            if any(pat.match(line) for pat in skip_patterns):
                continue

            m = entry_pattern.match(line)
            if m:
                title = m.group("title").strip()
                raw_code = m.group("code")
                
                lookup_key = self._normalize_title(title)
                code = toc_map.get(lookup_key, self._normalize_code(raw_code)) if toc_map else self._normalize_code(raw_code)
                
                dept_index[current_dept].append({
                    "title": title,
                    "rule_code": code
                })
            else:
                if "학교법인" in line and not re.search(r'\d+\s*[-—–]\s*\d+\s*[-—–]\s*\d+', line):
                    break
                # Department name or header
                current_dept = line
                if current_dept not in dept_index:
                    dept_index[current_dept] = []
        
        # Drop empty buckets
        dept_index = {k: v for k, v in dept_index.items() if v}
            
        return dept_index

    def _parse_entries(self, text: str, toc_map: Dict[str, str] = None) -> List[Dict[str, str]]:
        entries = []
        lines = text.splitlines()
        # Regex to capture just the code part, ignoring trailing ~stuff
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+)')
        
        for line in lines:
            if not line.strip(): 
                continue
            m = entry_pattern.match(line)
            if m:
                title = m.group("title").strip()
                raw_code = m.group("code")
                
                # Auto-correction using normalized TOC map
                lookup_key = self._normalize_title(title)
                code = toc_map.get(lookup_key, self._normalize_code(raw_code)) if toc_map else self._normalize_code(raw_code)

                entries.append({
                    "title": title,
                    "rule_code": code
                })
        return entries
