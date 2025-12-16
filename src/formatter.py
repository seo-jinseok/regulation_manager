import re
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

class RegulationFormatter:
    """
    Parses regulation text into structured JSON utilizing a nested Node structure:
    Regulation -> Chapter -> Article -> Paragraph -> Item
    """

    def parse(self, text: str, html_content: Optional[str] = None, verbose_callback=None) -> List[Dict[str, Any]]:
        # 1. First Pass: Flat Parsing (Existing Logic)
        if verbose_callback:
            verbose_callback("[dim]• Parsing document structure (Claims, Articles)...[/dim]")
        flat_doc_data = self._parse_flat(text)
        
        final_docs = []
        for doc_data in flat_doc_data:
            # 2. Refinement & Hierarchy Building
            
            # Title Extraction
            title, preamble = self._extract_clean_title(doc_data)
            
            # Appendices Parsing
            addenda, attached_files = self._parse_appendices(doc_data.get("appendices", []), html_content=html_content)
            
            # Build Metadata
            metadata = {
                "scan_date": "unknown", # To be populated by main/file stats if needed
                "file_name": "",       # To be populated
            }

            # Extract header metadata if HTML is available
            if html_content:
                header_entries = self._extract_header_metadata(html_content)
                if verbose_callback and header_entries:
                    verbose_callback(f"[dim]  - Found {len(header_entries)} header metadata entries[/dim]")
                # print(f"DEBUG: Found {len(header_entries)} headers. First: {header_entries[0] if header_entries else 'None'}")
                
                # Filter headers matching this doc's title
                relevant = [h for h in header_entries if self._titles_match(title, h['prefix'])]
                
                # print(f"DEBUG: Matching '{title}' against headers -> Found {len(relevant)} matches.")

                if relevant:
                    # Assume rule code is consistent across matched headers
                    rule_code = relevant[0]["rule_code"]
                    metadata["rule_code"] = rule_code
                    
                    # Calculate page range using ALL headers with this rule code
                    # (including section headers like "제3편 ...")
                    same_code_headers = [h for h in header_entries if h["rule_code"] == rule_code]
                    pages = [int(h["page"]) for h in same_code_headers if h["page"]]
                    if pages:
                        metadata["page_range"] = f"{min(pages)}~{max(pages)}"
            
            # Build Content Hierarchy (Chapter -> Article -> ...)
            content_nodes = self._build_hierarchy(doc_data["articles"])
            
            # Add Appendices to content as specific nodes or separate fields?
            # Schema "Special Cases" says: "Separate specific array"
            # However, looking at the schema:
            # { "content": [...], "metadata": { "amendments": ... } }
            # But the user schema says "appendix array" in special cases.
            # Let's keep them as root fields for now, or put them in 'appendices' lists.
            
            final_doc = {
                "part": doc_data.get("part"),
                "title": title,
                "metadata": metadata, # Placeholder
                "preamble": preamble, # Optional, maybe part of metadata or text?
                "content": content_nodes,
                "addenda": addenda,
                "attached_files": attached_files
            }
            final_docs.append(final_doc)
            
            if verbose_callback:
                art_count = len(doc_data.get("articles", []))
                add_count = len(addenda)
                att_count = len(attached_files)
                verbose_callback(f"[dim]  - Parsed: {art_count} Articles, {add_count} Addenda, {att_count} Attached Files[/dim]")
            
        # 3. Second Pass: Backfill Rule Codes from TOC
        # Scan all documents for TOC-like entries to build a global map
        # This handles cases where TOC is split into multiple docs due to "Part" headers
        toc_map = {}
        for doc in final_docs:
             # Only scan preamble (TOC entries are usually here)
             toc_map.update(self._parse_toc_rule_codes(doc["preamble"]))
             
        # print(f"DEBUG: TOC Map extracted: {len(toc_map)} entries.")
        if toc_map:
            for doc in final_docs:
                if not doc["metadata"].get("rule_code"):
                    match_code = None
                    doc_title = doc["title"]
                    if not doc_title: continue
                    
                    # 1. Direct match
                    if doc_title in toc_map:
                        match_code = toc_map[doc_title]
                    else:
                        # 2. Fuzzy/Normalized match
                        for toc_title, code in toc_map.items():
                            if self._titles_match(doc_title, toc_title):
                                match_code = code
                                break
                    
                    if match_code:
                        # print(f"DEBUG: Backfilling rule_code for '{doc_title}' -> {match_code}")
                        doc["metadata"]["rule_code"] = match_code

        return final_docs

    def _parse_toc_rule_codes(self, preamble: str) -> Dict[str, str]:
        """
        Parses the preamble of the Table of Contents to extract Title -> RuleCode mapping.
        Example line: "직제규정 3-1-1"
        """
        mapping = {}
        if not preamble: return mapping
        
        lines = preamble.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Regex: Title followed by Rule Code (d-d-d)
            # e.g. "교원인사규정 3-1-5"
            # Some titles have spaces. Rule code is at the end.
            match = re.match(r'^(.*)\s+(\d+-\d+-\d+)$', line)
            if match:
                title = match.group(1).strip()
                code = match.group(2).strip()
                mapping[title] = code
        
        return mapping

    def _create_node(self, level: str, number: str, title: Optional[str], text: Optional[str], children: List[Dict] = None) -> Dict[str, Any]:
        return {
            "level": level,
            "number": number,
            "title": title,
            "text": text,
            "children": children if children is not None else []
        }

    def _build_hierarchy(self, articles: List[Dict]) -> List[Dict]:
        roots = []
        # State tracking for hierarchy
        current_nodes = {
            "chapter": {"name": None, "node": None},
            "section": {"name": None, "node": None},
            "subsection": {"name": None, "node": None}
        }
        
        def get_parent_list(level):
            # Determine where to append the new node
            if level == "chapter": return roots
            if level == "section":
                if current_nodes["chapter"]["node"]: return current_nodes["chapter"]["node"]["children"]
                return roots # Fallback
            if level == "subsection":
                if current_nodes["section"]["node"]: return current_nodes["section"]["node"]["children"]
                if current_nodes["chapter"]["node"]: return current_nodes["chapter"]["node"]["children"]
                return roots
            if level == "article":
                if current_nodes["subsection"]["node"]: return current_nodes["subsection"]["node"]["children"]
                if current_nodes["section"]["node"]: return current_nodes["section"]["node"]["children"]
                if current_nodes["chapter"]["node"]: return current_nodes["chapter"]["node"]["children"]
                return roots
            return roots

        for art in articles:
            # Hierarchy Levels to check in order
            levels = [("chapter", r'^(제\s*(\d+)\s*[장편])\s*(.*)'), 
                      ("section", r'^(제\s*(\d+)\s*절)\s*(.*)'), 
                      ("subsection", r'^(제\s*(\d+)\s*관)\s*(.*)')]
            
            # 1. Update Hierarchy Nodes
            for lvl, regex in levels:
                raw_val = art.get(lvl)
                if raw_val != current_nodes[lvl]["name"]:
                    current_nodes[lvl]["name"] = raw_val
                    # Reset lower levels
                    if lvl == "chapter":
                        current_nodes["section"] = {"name": None, "node": None}
                        current_nodes["subsection"] = {"name": None, "node": None}
                    elif lvl == "section":
                        current_nodes["subsection"] = {"name": None, "node": None}

                    if raw_val:
                        match = re.match(regex, raw_val.strip())
                        if match:
                            num = match.group(2)
                            title = match.group(3)
                            node = self._create_node(lvl, num, title, None)
                        else:
                            node = self._create_node(lvl, "", raw_val, None)
                        
                        current_nodes[lvl]["node"] = node
                        get_parent_list(lvl).append(node)
                    else:
                        current_nodes[lvl]["node"] = None
            
            # 2. Create Article Node
            art_text = "\n".join(art.get('content', []))
            art_node = self._create_node("article", art.get('article_no'), art.get('title'), art_text)
            
            # Paragraphs & Items
            for para in art.get('paragraphs', []):
                para_num = para.get('paragraph_no')
                para_text = para.get('content')
                para_node = self._create_node("paragraph", para_num, None, para_text)
                
                for item in para.get('items', []):
                    item_num = item.get('item_no')
                    item_content = item.get('content')
                    item_node = self._create_node("item", item_num, None, item_content)
                    
                    for sub in item.get('subitems', []):
                        sub_node = self._create_node("subitem", sub.get('subitem_no'), None, sub.get('content'))
                        item_node["children"].append(sub_node)

                    para_node["children"].append(item_node)
                
                art_node["children"].append(para_node)
            
            # Append Article to lowest active parent
            get_parent_list("article").append(art_node)
                
        return roots

    def _parse_appendices(self, appendix_lines_or_str, html_content: Optional[str] = None):
        if isinstance(appendix_lines_or_str, list):
            text = "\n".join(appendix_lines_or_str).strip()
        else:
            text = str(appendix_lines_or_str).strip()
            
        if not text:
            return [], []
            
        addenda = []
        attached_files = []
        
        # Updated regex to match:
        # 1. "부 칙" (Addenda)
        # 2. "[ 별 표 ... ]" or "[ 별 지 ... ]" (Square brackets)
        # 3. "< 별 표 ... >" or "< 별 지 ... >" (Angle brackets)
        # Matches spaces flexibly.
        pattern = r'(?:^|\n)(?:\|\s*)?((?:부\s*칙)|(?:\[\s*별\s*표.*?\])|(?:\[\s*별\s*지.*?\])|(?:<\s*별\s*표.*?>)|(?:<\s*별\s*지.*?>))'
        segments = re.split(pattern, text, flags=re.IGNORECASE)
        
        # If the split separates the header from the content, segments[0] is usually empty or preamble before the first header.
        # segments[1] is header, segments[2] is content, segments[3] header, segments[4] is content...
        
        idx = 1
        while idx < len(segments):
            header = segments[idx].strip()
            content = segments[idx+1].strip() if idx+1 < len(segments) else ""
            
            # Normalize header for checking: remove spaces and brackets
            header_norm = header.replace(" ", "").replace("[", "").replace("]", "").replace("<", "").replace(">", "")
            
            if "부칙" in header_norm:
                 # Parse structure provided in 'content'
                 children_nodes = self._parse_addenda_text(content)
                 # If we successfully parsed children, we don't need the huge chunk of text
                 # duplicating the content.
                 final_text = content if not children_nodes else None
                 
                 addenda.append({
                     "title": header,
                     "text": final_text,
                     "children": children_nodes 
                 })
    
            elif "별표" in header_norm or "별지" in header_norm:
                attached_files.append({
                    "title": header,
                    "text": content
                })
                
            idx += 2
        
        # If HTML content is provided, try to extract corresponding HTML segments
        if html_content and attached_files:
            self._extract_html_segments(attached_files, html_content)
            
        return addenda, attached_files

    def _extract_html_segments(self, attached_files: List[Dict], html_content: str):
        """
        Extracts HTML segments for each attached file from the full HTML content.
        Adds an 'html' field to each attached_file dictionary.
        """
        # 1. Extract CSS Style block
        style_match = re.search(r'(<style.*?>.*?</style>)', html_content, re.DOTALL | re.IGNORECASE)
        style_block = style_match.group(1) if style_match else ""
        
        # 2. Find positions of each attached file header in HTML
        # HTML output of hwp5html typically escapes characters: < -> &lt;, > -> &gt;
        # And adds spans. e.g. <span ...>&lt;별지 1호 서식&gt;</span>
        
        lower_html = html_content.lower()
        
        # Helper to normalize title for search in HTML (simple approximation)
        def make_html_search_key(title):
            # Escape expected chars
            escaped = title.replace("<", "&lt;").replace(">", "&gt;")
            # Remove spaces for robust search (since HTML might have tags in between) -> strict match is hard.
            # Let's try to match the visible text. 
            # Ideally we find the sequence of characters.
            return escaped.lower()

        # Locate starts
        # We need to preserve order.
        # We search for the *text* of the headers.
        
        positions = []
        for i, af in enumerate(attached_files):
            title = af["title"] # e.g. <별지 1호 서식>
            search_key = make_html_search_key(title)
            
            # Simple substring search.
            pos = lower_html.find(search_key)
            
            if pos == -1:
                # Fallback: if not found, try without escaping (rare) or generic fallback
                pass
            
            if pos != -1:
                positions.append((pos, i))
        
        # Sort by position to handle out-of-order detection if any (should be in order)
        positions.sort()
        
        # Slice
        for idx, (start_pos, af_index) in enumerate(positions):
            # End pos is the start of the next attached file OR end of document
            if idx + 1 < len(positions):
                end_pos = positions[idx+1][0]
            else:
                end_pos = len(html_content)
                
            # Refine start_pos: include the container? 
            # hwp5html usually puts headers in <div class="HeaderArea"> or similar?
            # Or just Paragraphs <p>.
            # Let's backtrace to a logical block start if possible, or just exact match.
            # For "Table" preservation, we usually want the content *after* the header.
            
            # Actually, standard attached files in specific formats (Forms) usually follow the header immediately.
            # Let's grab the content starting from the header position.
            
            segment = html_content[start_pos:end_pos]
            
            # Combine with style
            # Wrap in a div to isolate?
            full_html = f"<html><head>{style_block}</head><body>{segment}</body></html>"
            
            attached_files[af_index]["html"] = full_html

    def _extract_header_metadata(self, html_content: str) -> List[Dict]:
        """
        Extracts header metadata (Rule Code, Page Number, Title Prefix) from HTML.
        Returns a list of dictionaries.
        """
        metadata_list = []
        if not html_content:
            return metadata_list

        soup = BeautifulSoup(html_content, 'html.parser')
        header_areas = soup.find_all('div', class_='HeaderArea')

        for div in header_areas:
            original_text = div.get_text(strip=True)
            # Normalize dashes and tildes
            normalized_text = re.sub(r'[\u2010-\u2015\u2212\uFF0D]', '-', original_text)
            normalized_text = re.sub(r'[~\uFF5E\u301C]', '~', normalized_text)

            # Regex for Rule Code (e.g. 3-1-97) and optional Page Number (~1)
            match = re.search(r'(\d+-\d+-\d+)(~\d+)?', normalized_text)
            if match:
                rule_code = match.group(1)
                page_part = match.group(2)
                page_number = page_part.replace('~', '') if page_part else None
                
                # Extract prefix
                prefix = normalized_text.split(rule_code)[0].strip()
                prefix = re.sub(r'[~\u301c\u2053]+$', '', prefix).strip()
                
                # Filter out section headers (e.g. "제3편 행정") which are not titles
                # Heuristic: if it contains "제" and "편", likely a section header
                # We still include them because they contain page numbers
                
                metadata_list.append({
                    "rule_code": rule_code,
                    "page": page_number,
                    "prefix": prefix
                })
        
        return metadata_list

    def _titles_match(self, doc_title: str, header_prefix: str) -> bool:
        """
        Checks if the document title matches the header prefix.
        Simple normalization and substring match.
        """
        if not doc_title or not header_prefix:
            return False
        
        norm_doc = re.sub(r'\s+', '', doc_title).lower()
        norm_header = re.sub(r'\s+', '', header_prefix).lower()
        
        return norm_doc == norm_header or norm_doc in norm_header or norm_header in norm_doc

    def _parse_addenda_text(self, text: str) -> List[Dict]:
        nodes = []
        lines = text.split('\n')
        current_node = None
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 1. Article Style (제1조) - In addenda, sometimes used, but usually it's items.
            # However, to distinguish from main articles, currently mapped to addendum_item?
            # User requested: "일반 규정의 조(article)과 다른 항목(노드?)로 처리"
            art_match = re.match(r'^(제\s*\d+\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if art_match:
                # Explicit Articles in Addenda might be treated as 'article' or 'addendum_item'?
                # Let's use 'addendum_item' for consistency within Addenda as requested.
                nodes.append(self._create_node("addendum_item", art_match.group(1), art_match.group(2), art_match.group(3)))
                current_node = nodes[-1]
                continue
                
            # 2. Numbered Item Style acting as Article (1. (시행일)...)
            num_match = re.match(r'^(\d+\.)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if num_match:
                # Treat as Item-level article -> addendum_item
                nodes.append(self._create_node("addendum_item", num_match.group(1), num_match.group(2), num_match.group(3)))
                current_node = nodes[-1]
                continue
            
            # 3. Paragraph Style (①)
            para_match = re.match(r'^([①-⑮])\s*(.*)', line)
            if para_match:
                # If inside an article/item, add as child
                if current_node:
                    current_node["children"].append(self._create_node("paragraph", para_match.group(1), None, para_match.group(2)))
                else:
                    # Orphan paragraph -> treat as Addenda Item
                    nodes.append(self._create_node("addendum_item", para_match.group(1), None, para_match.group(2)))
                    current_node = nodes[-1]
                continue
            
            # 4. Text Content (append to current node)
            if current_node:
                if current_node["text"]:
                    current_node["text"] += " " + line
                else:
                    current_node["text"] = line
            else:
                # Top level text (Prologue of Addenda?)
                # Create a generic node or ignore?
                # Usually dates: "1988. 3. 1. 제정"
                # Let's treat as a 'text' node or preamble Article?
                # Create dummy node
                nodes.append(self._create_node("text", "", None, line))
                current_node = nodes[-1]
                
        return nodes

    def _extract_clean_title(self, doc_data):
        # Heuristic combined from refine_json.py and previous formatter
        
        # 1. Try explicit tracking from parse
        regulation_title = doc_data.get('title') # from previous logic, might be None
        
        preamble_lines = doc_data.get('preamble', [])
        if isinstance(preamble_lines, str):
            preamble_lines = preamble_lines.split('\n')
            
        preamble_text = "\n".join(preamble_lines).strip()

        if regulation_title:
            return regulation_title, preamble_text
            
        # 2. Extract from Preamble (Refine Logic)
        candidates = [line.strip() for line in preamble_lines if line.strip()]
        
        best_title = ""
        for line in reversed(candidates):
            # Skip if meta info
            if (line.startswith('<') and line.endswith('>')) or (line.startswith('(') and line.endswith(')')):
                continue
            
            # Clean
            cleaned = re.sub(r'\s*[<\[\(].*?[>\]\)]', '', line).strip()
            cleaned = re.sub(r'\s*제\s*\d+\s*장.*', '', cleaned).strip()
             
            if not cleaned: continue
            
            # Verify if it looks like a title
            # Verify if it looks like a title
            suffixes = ("규정", "규칙", "세칙", "지침", "요령", "강령", "내규", "학칙", "헌장", "기준", "수칙", "준칙", "요강", "운영", "정관")
            if cleaned.endswith(suffixes) or "규정" in cleaned:
                best_title = cleaned
                break
        
        # Fallback to last line if nothing found (Previous formatter heuristic)
        if not best_title and candidates:
             best_title = candidates[-1]

        return best_title, preamble_text

    def _parse_flat(self, text: str) -> List[Dict[str, Any]]:
        # Encapsulated existing logic
        lines = text.split('\n')
        regulations = []
        current_data = {"preamble": [], "articles": [], "appendices": []}
        current_article = None
        current_paragraph = None
        current_item = None
        current_chapter = None
        current_section = None
        current_subsection = None
        regulation_title = None
        current_book_part = None # Track "Part" (Category)
        mode = 'PREAMBLE' 

        def flush_regulation(next_preamble_lines=None):
            nonlocal current_article, current_paragraph, current_item, mode, current_data, current_chapter, current_section, current_subsection, regulation_title
            if current_article:
                current_data["articles"].append(current_article)
                current_article = None
                current_paragraph = None
                current_item = None
            if current_data["articles"] or current_data["preamble"] or current_data["appendices"]:
                 reg = {
                     "part": current_book_part, # Add Part info
                     "title": regulation_title,
                     "preamble": current_data["preamble"],
                     "articles": current_data["articles"],
                     "appendices": current_data["appendices"]
                 }
                 regulations.append(reg)
            
            current_data = {
                "preamble": next_preamble_lines if next_preamble_lines else [],
                "articles": [],
                "appendices": []
            }
            mode = 'PREAMBLE'
            current_chapter = None
            current_section = None
            current_subsection = None
            regulation_title = None

        for line in lines:
            line = line.strip()
            if not line: continue

            # Part (Groups Regulations)
            part_match = re.match(r'^\|?\s*(제\s*\d+\s*편)\s*(.*)\|?$', line)
            # The markdown often has tables | 제1편 | or just text.
            # Clean md view: "제2편 학칙"
            if not part_match:
                 # Try cleaning pipe characters if they act as borders
                 clean_line = line.replace('|', '').strip()
                 part_match = re.match(r'^(제\s*\d+\s*편)\s*(.*)', clean_line)
            
            if part_match:
                # Part found (e.g. 제1편 학교법인)
                # It separates the book. Flush current regulation.
                flush_regulation()
                
                # Extract clean name from groups
                p_num = part_match.group(1).strip()
                p_name = part_match.group(2).replace('|', '').strip()
                current_book_part = f"{p_num} {p_name}".strip()
                continue

            # Chapter
            chapter_match = re.match(r'^(제\s*\d+\s*[장편])\s*(.*)', line)
            if chapter_match:
                current_chapter = line 
                current_section = None
                current_subsection = None
                continue
            
            # Section (절)
            section_match = re.match(r'^(제\s*\d+\s*절)\s*(.*)', line)
            if section_match:
                current_section = line
                current_subsection = None
                continue

            # Subsection (관)
            subsection_match = re.match(r'^(제\s*\d+\s*관)\s*(.*)', line)
            if subsection_match:
                current_subsection = line
                continue

            # Special Titles: Table of Contents (차례) and Index (찾아보기)
            
            # 1. TOC
            # Pattern: "차 례" or "차  례" alone on a line, or "목 차"
            if re.match(r'^차\s*례\s*$', line) or re.match(r'^목\s*차\s*$', line):
                 # Flush previous
                 flush_regulation()
                 regulation_title = "차례"
                 # Add the title line to preamble to ensure doc is not empty
                 current_data["preamble"].append(line)
                 continue

            # 2. Index
            # Pattern: "찾아보기" alone or with brackets
            if re.match(r'^찾아보기.*', line) and len(line) < 20: 
                 # Flush previous
                 flush_regulation()
                 regulation_title = "찾아보기"
                 current_data["preamble"].append(line)
                 continue

            # Article 1 Split
            article_1_match = re.match(r'^(제\s*1\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_1_match:
                if current_data["articles"] or current_article:
                     split_idx = -1
                     start_next_lines = []
                     if mode == 'APPENDICES':
                         for i in range(len(current_data["appendices"]) - 1, -1, -1):
                             txt = current_data["appendices"][i].strip()
                             title_candidates = ["규정", "세칙", "지침", "요령", "강령", "내규", "학칙", "헌장", "기준", "수칙", "준칙", "요강", "운영", "정관"]
                             if (any(txt.endswith(c) for c in title_candidates) or "규정" in txt) and len(txt) < 100:
                                 split_idx = i
                                 break
                         if split_idx != -1:
                             start_next_lines = current_data["appendices"][split_idx:]
                             current_data["appendices"] = current_data["appendices"][:split_idx]
                     flush_regulation(next_preamble_lines=start_next_lines)
            
            # Article
            article_match = re.match(r'^(제\s*(\d+)\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_match:
                mode = 'ARTICLES'
                if current_article: current_data["articles"].append(current_article)
                article_no = article_match.group(2)
                article_title = article_match.group(3) or ""
                content = article_match.group(4)
                current_article = {
                    "article_no": article_no,
                    "title": article_title,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "content": [], 
                    "paragraphs": []
                }
                current_paragraph = None
                current_item = None
                if content:
                    para_match = re.match(r'^([①-⑮])\s*(.*)', content)
                    if para_match:
                        current_paragraph = {
                            "paragraph_no": para_match.group(1),
                            "content": para_match.group(2),
                            "items": []
                        }
                        current_article["paragraphs"].append(current_paragraph)
                    else:
                        current_article["content"].append(content)
                continue

            # Appendices
            if re.match(r'^부\s*칙', line):
                mode = 'APPENDICES'
                current_chapter = None
                current_section = None
                current_subsection = None
                if current_article:
                    current_data["articles"].append(current_article)
                    current_article = None
                    current_paragraph = None
                    current_item = None
                current_data["appendices"].append(line)
                continue
                
            # Content
            if mode == 'ARTICLES':
                para_match = re.match(r'^([①-⑮])\s*(.*)', line)
                if para_match and current_article:
                    current_paragraph = {
                        "paragraph_no": para_match.group(1),
                        "content": para_match.group(2),
                        "items": []
                    }
                    current_article["paragraphs"].append(current_paragraph)
                    current_item = None
                    continue

                # Item (1., 2., 3.)
                item_match = re.match(r'^(\d+\.)\s*(.*)', line)
                if item_match and current_article:
                    new_item = {
                        "item_no": item_match.group(1), 
                        "content": item_match.group(2),
                        "subitems": []
                    }
                    if current_paragraph:
                        current_paragraph["items"].append(new_item)
                        current_item = new_item
                    else:
                         # Fallback: Items directly under Article (no paragraph)
                         current_article["content"].append(line)
                    continue

                # Subitem (가., 나., 다.)
                subitem_match = re.match(r'^([가-하]\.)\s*(.*)', line)
                if subitem_match and current_article:
                    if current_item:
                        current_item["subitems"].append({
                            "subitem_no": subitem_match.group(1),
                            "content": subitem_match.group(2)
                        })
                    else:
                        if current_paragraph:
                             current_paragraph["content"] += " " + line
                        else:
                             current_article["content"].append(line)
                    continue
                
                if current_paragraph:
                    current_paragraph["content"] += " " + line
                elif current_article:
                    current_article["content"].append(line)
                else:
                    current_data["preamble"].append(line)

            elif mode == 'APPENDICES':
                current_data["appendices"].append(line)
            elif mode == 'PREAMBLE':
                current_data["preamble"].append(line)

        flush_regulation()
        return regulations
