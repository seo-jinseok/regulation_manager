import re
from typing import Dict, List, Any

class MetadataExtractor:
    """
    Extracts Table of Contents (TOC) and Index (Alpha/Dept) from raw markdown.
    """

    def extract(self, text: str) -> Dict[str, Any]:
        toc_list = self._extract_toc(text)
        
        # Build Source of Truth map from TOC
        # Normalize titles by removing spaces for better matching?
        # For now, exact match or simple strip.
        toc_map = {item["title"]: item["rule_code"] for item in toc_list}
        
        return {
            "toc": toc_list,
            "index_by_alpha": self._extract_index_alpha(text, toc_map),
            "index_by_dept": self._extract_index_dept(text, toc_map)
        }

    def _extract_toc(self, text: str) -> List[Dict[str, str]]:
        # Find explicit TOC section if possible
        # Pattern: Starts with "차 례" or "목 차"
        match = re.search(r'^\s*(차\s*례|목\s*차)\s*$', text, re.MULTILINE)
        if not match:
            return []
        
        start_idx = match.end()
        # End TOC when we see "찾아보기" or the first actual regulation body like "제1편" used as a header *with content*?
        # Actually in this file, TOC corresponds to the structure.
        # Let's limit the scope. Usually TOC is at the start.
        # We can scan lines after "차례" until we see something that doesn't look like a TOC entry.
        
        lines = text[start_idx:].splitlines()
        toc = []
        
        # Heuristic: TOC entries usually end with rule code "1-0-1" or "3-2-55"
        # Regex: (Title) (Code)
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+(?:～\d+)?)\s*$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Stop if we hit explicit Index
            if "찾아보기" in line:
                break
                
            # Stop if we hit what looks like a preamble or main body text that isn't a TOC entry
            # (This is tricky because TOC entries look like headers)
            # But usually TOC is contiguous. 
            
            m = entry_pattern.match(line)
            if m:
                toc.append({
                    "title": m.group("title"),
                    "rule_code": m.group("code")
                })
            
            # If we see "제1편" etc alone, it's a section header in TOC
            # We could capture that too, but user asked for "indices" mostly.
            # Let's stick to capturing regulation links for now.
            
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
        # Note: Dept index is usually at the end.
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return {}
            
        content = match.group(1)
        lines = content.splitlines()
        
        dept_index = {}
        current_dept = "Unknown"
        dept_index[current_dept] = []
        
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+.*)\s*$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            m = entry_pattern.match(line)
            if m:
                title = m.group("title")
                raw_code = m.group("code")
                code = toc_map.get(title, raw_code) if toc_map else raw_code
                
                dept_index[current_dept].append({
                    "title": title,
                    "rule_code": code
                })
            else:
                # Likely a department name
                # Avoid capturing headers like "제1편" if they appear (unlikely here)
                current_dept = line
                if current_dept not in dept_index:
                    dept_index[current_dept] = []
        
        # Cleanup
        if not dept_index["Unknown"]:
            del dept_index["Unknown"]
            
        return dept_index

    def _parse_entries(self, text: str, toc_map: Dict[str, str] = None) -> List[Dict[str, str]]:
        entries = []
        lines = text.splitlines()
        # Relaxed pattern for Rule Code which might contain special dashes or range
        entry_pattern = re.compile(r'^\s*(?P<title>.*?)\s+(?P<code>\d+[-—]\d+[-—]\d+.*)\s*$')
        
        for line in lines:
            if not line.strip(): 
                continue
            m = entry_pattern.match(line)
            if m:
                title = m.group("title")
                raw_code = m.group("code")
                # Auto-correction
                code = toc_map.get(title, raw_code) if toc_map else raw_code

                entries.append({
                    "title": title,
                    "rule_code": code
                })
        return entries
