import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import uuid

STABLE_ID_NAMESPACE = uuid.UUID("f24a86f2-2c2d-4a08-9cc4-6b51b0b4043a")

class RegulationFormatter:
    """
    Parses regulation text into structured JSON utilizing a nested Node structure:
    Regulation -> Chapter -> Article -> Paragraph -> Item
    """

    def parse(
        self,
        text: str,
        html_content: Optional[str] = None,
        verbose_callback=None,
        extracted_metadata: Optional[Dict[str, Any]] = None,
        source_file_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # 1. First Pass: Flat Parsing (Existing Logic)
        if verbose_callback:
            verbose_callback("[dim]• 문서 구조 분석 중 (조항, 본문)...[/dim]")
        flat_doc_data = self._parse_flat(text)
        
        # Extract header metadata globally if HTML is available
        header_entries = []
        if html_content:
            header_entries = self._extract_header_metadata(html_content)
            if verbose_callback and header_entries:
                verbose_callback(f"[dim]  - 헤더 메타데이터 {len(header_entries)}개 발견[/dim]")

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

            # Map header metadata
            if header_entries:
                # Filter headers matching this doc's title
                relevant = [h for h in header_entries if self._titles_match(title, h['prefix'])]
                
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
                # Use the local 'title' variable which we extracted earlier
                title_display = title if title else "제목 없음"
                verbose_callback(f"[dim]  - 분석 완료: '{title_display}' ({art_count}개 조항, {add_count}개 부칙, {att_count}개 별표/서식)[/dim]")
            
        # 3. Second Pass: Backfill Rule Codes from TOC
        # Scan all documents for TOC-like entries to build a global map
        # This handles cases where TOC is split into multiple docs due to "Part" headers
        toc_map = {}
        for doc in final_docs:
             # Only scan preamble (TOC entries are usually here)
             toc_map.update(self._parse_toc_rule_codes(doc["preamble"]))
             
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
                        doc["metadata"]["rule_code"] = match_code

        if extracted_metadata is None:
            try:
                from .metadata_extractor import MetadataExtractor
                extracted_metadata = MetadataExtractor().extract(text)
            except Exception:
                extracted_metadata = None

        if extracted_metadata:
            self._populate_index_docs(final_docs, extracted_metadata)
                        
        cleaned_docs = self._reorder_and_trim_docs(final_docs)
        merged_docs = self._merge_adjacent_docs(cleaned_docs)
        self._assign_doc_types(merged_docs)
        self._extract_tables(merged_docs)
        self._assign_stable_node_ids(merged_docs, source_file_name=source_file_name)
        self._resolve_references(merged_docs)
        return merged_docs

    def _doc_has_content(self, doc: Dict[str, Any]) -> bool:
        return bool(doc.get("content")) or bool(doc.get("addenda")) or bool(doc.get("attached_files"))

    def _index_kind(self, doc: Dict[str, Any]) -> Optional[str]:
        title = (doc.get("title") or "").strip()
        if not title:
            return None
        if title in ("차례", "목차"):
            return "toc"
        if title == "찾아보기":
            preamble = doc.get("preamble") or ""
            if "<가나다순>" in preamble or "가나다순" in preamble:
                return "index_alpha"
            if "<소관부서별>" in preamble or "소관부서별" in preamble:
                return "index_dept"
            return "index"
        return None

    def _reorder_and_trim_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        order_rank = {"toc": 0, "index_alpha": 1, "index_dept": 2, "index": 3}

        index_docs = []
        content_docs = []

        for i, doc in enumerate(docs):
            if self._doc_has_content(doc):
                content_docs.append(doc)
                continue

            kind = self._index_kind(doc)
            if kind:
                doc["part"] = None
                index_docs.append((order_rank.get(kind, 99), i, doc))
                continue

            # Drop non-index, empty docs (e.g., 관리 현황표 등)
            continue

        index_docs.sort(key=lambda item: (item[0], item[1]))
        ordered_index_docs = [doc for _, _, doc in index_docs]

        return ordered_index_docs + content_docs

    def _merge_adjacent_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        merged = [docs[0]]

        for doc in docs[1:]:
            current = merged[-1]

            # Drop explicit noise docs
            if doc.get("title") and "규정집 관리 현황표" in doc.get("title"):
                continue

            # Merge doc with missing title into previous if same part
            if not doc.get("title") and doc.get("content") and current.get("part") == doc.get("part"):
                current["content"].extend(doc.get("content") or [])
                current["addenda"].extend(doc.get("addenda") or [])
                current["attached_files"].extend(doc.get("attached_files") or [])
                if doc.get("metadata"):
                    current["metadata"].update({k: v for k, v in doc["metadata"].items() if v is not None})
                continue

            merged.append(doc)

        return merged

    def _assign_doc_types(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            if doc.get("doc_type"):
                continue
            kind = self._index_kind(doc)
            if kind:
                doc["doc_type"] = kind
                continue

            rule_code = (doc.get("metadata") or {}).get("rule_code")
            if rule_code:
                doc["doc_type"] = "regulation"
                continue

            if self._doc_has_content(doc):
                doc["doc_type"] = "note"
            else:
                doc["doc_type"] = "unknown"

    def _assign_stable_node_ids(self, docs: List[Dict[str, Any]], source_file_name: Optional[str] = None) -> None:
        for doc in docs:
            doc_key = self._stable_doc_key(doc, source_file_name=source_file_name)
            for section in ("content", "addenda"):
                nodes = doc.get(section) or []
                self._assign_stable_node_ids_recursive(nodes, parent_path=section, doc_key=doc_key)

    def _stable_doc_key(self, doc: Dict[str, Any], source_file_name: Optional[str] = None) -> str:
        parts = []
        if source_file_name:
            parts.append(f"file:{source_file_name}")
        doc_type = (doc.get("doc_type") or "").strip()
        if doc_type:
            parts.append(f"type:{doc_type}")

        metadata = doc.get("metadata") or {}
        rule_code = (metadata.get("rule_code") or "").strip()
        if rule_code:
            parts.append(f"rule:{self._normalize_token(rule_code)}")
            return "|".join(parts)

        title = (doc.get("title") or "").strip()
        part = (doc.get("part") or "").strip()
        if part:
            parts.append(f"part:{self._normalize_token(part)}")
        if title:
            parts.append(f"title:{self._normalize_token(title)}")
        return "|".join(parts) or "doc:unknown"

    def _assign_stable_node_ids_recursive(self, nodes: List[Dict[str, Any]], parent_path: str, doc_key: str) -> None:
        if not nodes:
            return

        base_keys: List[str] = []
        refined_keys: List[str] = []

        for node in nodes:
            base = self._stable_node_base_key(node)
            base_keys.append(base)

        # First pass: add a content hash only when necessary
        base_counts = {}
        for base in base_keys:
            base_counts[base] = base_counts.get(base, 0) + 1

        for node, base in zip(nodes, base_keys):
            if base_counts.get(base, 0) <= 1:
                refined_keys.append(base)
                continue
            refined_keys.append(f"{base}|h:{self._stable_node_content_hash(node)}")

        # Second pass: disambiguate exact duplicates after hashing
        refined_counts = {}
        for key in refined_keys:
            refined_counts[key] = refined_counts.get(key, 0) + 1

        refined_seen = {}
        for node, key in zip(nodes, refined_keys):
            if refined_counts.get(key, 0) > 1:
                refined_seen[key] = refined_seen.get(key, 0) + 1
                key = f"{key}|i:{refined_seen[key]}"

            path = f"{parent_path}/{key}" if parent_path else key
            node["id"] = str(uuid.uuid5(STABLE_ID_NAMESPACE, f"{doc_key}::{path}"))
            self._assign_stable_node_ids_recursive(node.get("children") or [], parent_path=path, doc_key=doc_key)

    def _stable_node_base_key(self, node: Dict[str, Any]) -> str:
        node_type = (node.get("type") or "").strip()
        display_no = self._normalize_token(node.get("display_no") or "")
        sort_no = node.get("sort_no") or {}
        main = sort_no.get("main", 0)
        sub = sort_no.get("sub", 0)
        return f"{node_type}|{display_no}|{main}|{sub}"

    def _stable_node_content_hash(self, node: Dict[str, Any]) -> str:
        title = self._normalize_ws(node.get("title") or "")
        text = self._normalize_ws(node.get("text") or "")
        digest = hashlib.sha1(f"{title}|{text}".encode("utf-8")).hexdigest()
        return digest[:12]

    def _normalize_ws(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value)).strip()

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"\s+", "", str(value)).strip()

    def _resolve_references(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            rule_code = (doc.get("metadata") or {}).get("rule_code")
            if not rule_code:
                continue

            doc_index = self._build_reference_index(doc)
            for section in ("content", "addenda"):
                nodes = doc.get(section) or []
                self._resolve_references_in_nodes(
                    nodes,
                    doc_rule_code=rule_code,
                    doc_index=doc_index,
                    current_article=None,
                    current_paragraph=None,
                    current_item=None,
                )

    def _build_reference_index(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        articles: Dict[str, Dict[str, Any]] = {}

        def walk(nodes: List[Dict[str, Any]]) -> None:
            for node in nodes or []:
                if node.get("type") == "article":
                    key = self._normalize_token(node.get("display_no") or "")
                    if key:
                        articles[key] = self._build_article_reference_index(node)
                walk(node.get("children") or [])

        walk(doc.get("content") or [])
        return {"articles": articles}

    def _build_article_reference_index(self, article_node: Dict[str, Any]) -> Dict[str, Any]:
        paragraphs: Dict[int, str] = {}
        items_by_paragraph: Dict[int, Dict[int, str]] = {}
        items_direct: Dict[int, str] = {}

        for child in article_node.get("children") or []:
            if child.get("type") == "paragraph":
                para_no = int((child.get("sort_no") or {}).get("main", 0) or 0)
                if para_no > 0:
                    paragraphs[para_no] = child.get("id")
                items_by_paragraph[para_no] = {}
                for item in child.get("children") or []:
                    if item.get("type") != "item":
                        continue
                    item_no = int((item.get("sort_no") or {}).get("main", 0) or 0)
                    if item_no > 0:
                        items_by_paragraph[para_no][item_no] = item.get("id")
            elif child.get("type") == "item":
                item_no = int((child.get("sort_no") or {}).get("main", 0) or 0)
                if item_no > 0:
                    items_direct[item_no] = child.get("id")

        return {
            "id": article_node.get("id"),
            "paragraphs": paragraphs,
            "items_by_paragraph": items_by_paragraph,
            "items_direct": items_direct,
        }

    def _resolve_references_in_nodes(
        self,
        nodes: List[Dict[str, Any]],
        *,
        doc_rule_code: str,
        doc_index: Dict[str, Any],
        current_article: Optional[Dict[str, Any]],
        current_paragraph: Optional[Dict[str, Any]],
        current_item: Optional[Dict[str, Any]],
    ) -> None:
        for node in nodes or []:
            node_type = node.get("type")
            if node_type == "article":
                current_article = node
                current_paragraph = None
                current_item = None
            elif node_type == "paragraph":
                current_paragraph = node
                current_item = None
            elif node_type == "item":
                current_item = node

            for ref in node.get("references") or []:
                target_text = (ref.get("target") or ref.get("text") or "").strip()
                if not target_text:
                    continue
                target_id = self._resolve_reference_target(
                    target_text,
                    doc_index=doc_index,
                    current_article=current_article,
                    current_paragraph=current_paragraph,
                    current_item=current_item,
                )
                if target_id:
                    ref["target_node_id"] = target_id
                    ref["target_doc_rule_code"] = doc_rule_code

            self._resolve_references_in_nodes(
                node.get("children") or [],
                doc_rule_code=doc_rule_code,
                doc_index=doc_index,
                current_article=current_article,
                current_paragraph=current_paragraph,
                current_item=current_item,
            )

    def _resolve_reference_target(
        self,
        target_text: str,
        *,
        doc_index: Dict[str, Any],
        current_article: Optional[Dict[str, Any]],
        current_paragraph: Optional[Dict[str, Any]],
        current_item: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        parsed = self._parse_reference_token(target_text)

        article_key = None
        if parsed.get("article_main") is not None:
            article_key = self._format_article_key(parsed["article_main"], parsed.get("article_sub") or 0)
        elif current_article:
            article_key = self._normalize_token(current_article.get("display_no") or "")

        if not article_key:
            return None

        article_entry = (doc_index.get("articles") or {}).get(article_key)
        if not article_entry:
            return None

        para_no = parsed.get("paragraph_no")
        item_no = parsed.get("item_no")

        if para_no is None and item_no is None:
            return article_entry.get("id")

        resolved_para_no = None
        if para_no is not None:
            resolved_para_no = int(para_no)
        elif item_no is not None and current_paragraph and current_article:
            current_article_key = self._normalize_token(current_article.get("display_no") or "")
            if current_article_key == article_key:
                resolved_para_no = int((current_paragraph.get("sort_no") or {}).get("main", 0) or 0) or None

        if item_no is None:
            if resolved_para_no is None or resolved_para_no <= 0:
                return None
            return article_entry.get("paragraphs", {}).get(resolved_para_no)

        item_no_int = int(item_no)
        if item_no_int <= 0:
            return None

        if resolved_para_no is not None and resolved_para_no >= 0:
            return article_entry.get("items_by_paragraph", {}).get(resolved_para_no, {}).get(item_no_int)

        return article_entry.get("items_direct", {}).get(item_no_int)

    def _parse_reference_token(self, token: str) -> Dict[str, Optional[int]]:
        """
        Parses a reference token like:
        - 제6조
        - 제6조의2
        - 제6조제2항
        - 제6조제2항제3호
        - 제2항 (relative)
        - 제3호 (relative)
        """
        value = self._normalize_token(token)
        result: Dict[str, Optional[int]] = {
            "article_main": None,
            "article_sub": None,
            "paragraph_no": None,
            "item_no": None,
        }

        m = re.match(r"^제(\d+)조(?:의(\d+))?", value)
        if m:
            result["article_main"] = int(m.group(1))
            result["article_sub"] = int(m.group(2)) if m.group(2) else 0
            value = value[m.end() :]

        while value:
            m = re.match(r"^제(\d+)([항호])", value)
            if not m:
                break
            num = int(m.group(1))
            kind = m.group(2)
            if kind == "항" and result["paragraph_no"] is None:
                result["paragraph_no"] = num
            elif kind == "호" and result["item_no"] is None:
                result["item_no"] = num
            value = value[m.end() :]

        return result

    def _format_article_key(self, main_no: int, sub_no: int) -> str:
        if sub_no and int(sub_no) > 0:
            return f"제{int(main_no)}조의{int(sub_no)}"
        return f"제{int(main_no)}조"

    def _extract_tables(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            for section in ("content", "addenda"):
                self._extract_tables_in_nodes(doc.get(section) or [])

    def _extract_tables_in_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        for node in nodes or []:
            text = node.get("text") or ""
            updated_text, tables = self._split_markdown_tables(text)
            if tables:
                metadata = node.setdefault("metadata", {})
                metadata["tables"] = [{"format": "markdown", "markdown": t} for t in tables]
                node["text"] = updated_text
            self._extract_tables_in_nodes(node.get("children") or [])

    def _split_markdown_tables(self, text: str) -> tuple[str, List[str]]:
        if not text or "|" not in text:
            return text, []

        lines = text.splitlines()
        out_lines: List[str] = []
        tables: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if not self._is_table_row(line):
                out_lines.append(line)
                i += 1
                continue

            block: List[str] = []
            while i < len(lines) and self._is_table_row(lines[i]):
                block.append(lines[i])
                i += 1

            if len(block) >= 2 and any(self._is_table_separator_row(row) for row in block):
                tables.append("\n".join(block).strip())
                out_lines.append(f"[TABLE:{len(tables)}]")
                continue

            out_lines.extend(block)

        return "\n".join(out_lines).strip(), tables

    def _is_table_row(self, line: str) -> bool:
        stripped = (line or "").strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    def _is_table_separator_row(self, line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped.startswith("|") or "---" not in stripped:
            return False
        # Loose check: a markdown separator row is typically pipes with hyphens/colons.
        core = stripped.strip("|").strip()
        if not core:
            return False
        cells = [c.strip() for c in core.split("|") if c.strip()]
        if not cells:
            return False
        return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    def _parse_toc_rule_codes(self, preamble: str) -> Dict[str, str]:
        """
        Parses the preamble of the Table of Contents to extract Title -> RuleCode mapping.
        Example line: "직제규정 3-1-1"
        """
        mapping = {}
        if not preamble: return mapping
        
        lines = preamble.split('\n')
        for line in lines:
            line = line.strip().strip('|').strip()
            if not line: continue
            
            # Regex: Title followed by Rule Code (d-d-d)
            # e.g. "교원인사규정 3-1-5"
            # Some titles have spaces. Rule code is at the end.
            match = re.match(r'^(.*)\s+(\d+[-—–]\d+[-—–]\d+)(?:\s*[~～]\s*\d+)?$', line)
            if match:
                title = match.group(1).strip()
                code = match.group(2).strip()
                code = re.sub(r'[—–]', '-', code)
                mapping[title] = code
        
        return mapping

    def _build_index_nodes(self, entries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        nodes = []
        for idx, entry in enumerate(entries, 1):
            title = entry.get("title", "")
            rule_code = entry.get("rule_code")
            metadata = {"rule_code": rule_code} if rule_code else {}
            nodes.append(
                self._create_node(
                    "text",
                    "",
                    title,
                    rule_code or "",
                    sort_no={"main": idx, "sub": 0},
                    metadata=metadata,
                )
            )
        return nodes

    def _populate_index_docs(self, docs: List[Dict[str, Any]], extracted_metadata: Dict[str, Any]) -> None:
        toc_entries = extracted_metadata.get("toc") or []
        index_alpha_entries = extracted_metadata.get("index_by_alpha") or []
        index_dept_entries = extracted_metadata.get("index_by_dept") or {}

        for doc in docs:
            if doc.get("content"):
                continue
            kind = self._index_kind(doc)
            if kind == "toc" and toc_entries:
                doc["part"] = None
                doc["content"] = self._build_index_nodes(toc_entries)
                doc["preamble"] = ""
                doc["doc_type"] = "toc"
            elif kind == "index_alpha" and index_alpha_entries:
                doc["part"] = None
                doc["content"] = self._build_index_nodes(index_alpha_entries)
                doc["preamble"] = ""
                doc["doc_type"] = "index_alpha"
            elif kind in ("index_dept", "index") and index_dept_entries:
                doc["part"] = None
                dept_nodes = []
                for dept_idx, (dept, entries) in enumerate(sorted(index_dept_entries.items()), 1):
                    dept_node = self._create_node(
                        "text",
                        "",
                        dept,
                        "",
                        sort_no={"main": dept_idx, "sub": 0},
                    )
                    dept_node["children"] = self._build_index_nodes(entries)
                    dept_nodes.append(dept_node)
                doc["content"] = dept_nodes
                doc["preamble"] = ""
                doc["doc_type"] = "index_dept"

    def _create_node(self, node_type: str, display_no: str, title: Optional[str], text: Optional[str], sort_no: Dict[str, int] = None, children: List[Dict] = None, confidence_score: float = 1.0, references: List[Dict] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if sort_no is None:
            sort_no = {"main": 0, "sub": 0}
            
        return {
            "id": str(uuid.uuid4()),
            "type": node_type,
            "display_no": display_no,
            "sort_no": sort_no,
            "title": title or "",
            "text": text or "",
            "confidence_score": confidence_score,
            "references": references if references is not None else [],
            "metadata": metadata if metadata is not None else {},
            "children": children if children is not None else []
        }

    def _extract_references(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts cross-references (e.g., "제5조", "제10조제1항") from text.
        """
        if not text:
            return []
            
        # Pattern to match "제N조", "제N조의M", followed by optional "제K항", "제L호", etc.
        # OR just standalone "제K항", etc.
        pattern = r'제\s*\d+\s*조(?:의\s*\d+)?(?:제\s*\d+\s*[항호목])*|제\s*\d+\s*[항호목]'
        
        matches = re.finditer(pattern, text)
        refs = []
        for m in matches:
            t = m.group(0).strip()
            refs.append({
                "text": t,
                "target": t # Placeholder for resolution
            })
        return refs

    def _resolve_sort_no(self, display_no: str, node_type: str) -> Dict[str, int]:
        """
        Resolves a display string into a sorting key {main, sub}.
        e.g. "제29조의2" -> {main: 29, sub: 2}
             "①" -> {main: 1, sub: 0}
             "가." -> {main: 1, sub: 0}
        """
        main = 0
        sub = 0
        
        if node_type == "article":
            # Match "제29조" or "제29조의2"
            match = re.search(r'제\s*(\d+)\s*조(?:의\s*(\d+))?', display_no)
            if match:
                main = int(match.group(1))
                if match.group(2):
                    sub = int(match.group(2))
                    
        elif node_type in ["chapter", "section", "subsection", "part"]:
             # Match "제1장", "제1편" -> extract number
             match = re.search(r'(\d+)', display_no)
             if match:
                 main = int(match.group(1))

        elif node_type == "paragraph":
            # Match circled numbers ①..⑮
            # Unicode range for ① is \u2460 (1) to \u246e (15)
            # Simplified map for common ones
            clean = display_no.strip()
            if clean and len(clean) == 1:
                code = ord(clean)
                if 0x2460 <= code <= 0x2473: # ① ~ ⑳
                    main = code - 0x2460 + 1
                    
        elif node_type == "item":
            # Match "1."
            match = re.match(r'^(\d+)', display_no)
            if match:
                main = int(match.group(1))
                
        elif node_type == "subitem":
            # Match "가." -> Map hangul jamo to int
            # 가=1, 나=2, ...
            match = re.match(r'^([가-하])', display_no)
            if match:
                char = match.group(1)
                # '가' is 0xAC00. But subitem sequence is usually standard Jamo logic?
                # Actually regulations use: 가, 나, 다, 라, 마, 바, 사, 아, 자, 차, 카, 타, 파, 하
                # These are consecutive in Hangul Syllables but with a step.
                # Let's use a simpler lookup string for safety.
                order = "가나다라마바사아자차카타파하"
                if char in order:
                    main = order.index(char) + 1
                    
        return {"main": main, "sub": sub}

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
                        # Debugging TypeError
                        if not isinstance(raw_val, str):
                            print(f"[DEBUG] TypeError Suspicion: lvl={lvl}, type={type(raw_val)}, val={raw_val}")
                            raw_val = str(raw_val) # Force string conversion check
                            
                        match = re.match(regex, raw_val.strip())
                        if match:
                            display_no = match.group(1) # "제1장"
                            # num = match.group(2) # "1" (Unused now, we resolve sort_no)
                            title = match.group(3)
                            sort_no = self._resolve_sort_no(display_no, lvl)
                            node = self._create_node(lvl, display_no, title, None, sort_no, confidence_score=1.0)
                        else:
                            # Fallback if regex fails but value exists
                            node = self._create_node(lvl, raw_val, raw_val, None, confidence_score=0.5)
                        
                        current_nodes[lvl]["node"] = node
                        get_parent_list(lvl).append(node)
                    else:
                        current_nodes[lvl]["node"] = None
            
            # 2. Create Article Node
            art_text = "\n".join(art.get('content', []))
            art_display_no = art.get('article_no', '')
            art_sort_no = self._resolve_sort_no(art_display_no, "article")
            art_refs = self._extract_references(art_text)
            art_node = self._create_node("article", art_display_no, art.get('title'), art_text, art_sort_no, references=art_refs)
            
            # Paragraphs & Items
            for para in art.get('paragraphs', []):
                para_num = (para.get('paragraph_no', '') or '').strip()
                para_text = (para.get('content') or '').strip()
                items = para.get('items', []) or []

                # Items can appear directly under an article without an explicit paragraph marker.
                # Avoid emitting empty paragraph container nodes; attach items directly under the article.
                if not para_num and not para_text:
                    for item in items:
                        item_num = item.get('item_no', '')
                        item_content = item.get('content')
                        item_sort = self._resolve_sort_no(item_num, "item")
                        item_refs = self._extract_references(item_content)
                        item_node = self._create_node("item", item_num, None, item_content, item_sort, references=item_refs)

                        for sub in item.get('subitems', []):
                            sub_num = sub.get('subitem_no', '')
                            sub_sort = self._resolve_sort_no(sub_num, "subitem")
                            sub_content = sub.get('content', '')
                            sub_refs = self._extract_references(sub_content)
                            sub_node = self._create_node("subitem", sub_num, None, sub_content, sub_sort, references=sub_refs)
                            item_node["children"].append(sub_node)

                        art_node["children"].append(item_node)
                    continue

                para_sort = self._resolve_sort_no(para_num, "paragraph")
                para_refs = self._extract_references(para_text)
                para_node = self._create_node("paragraph", para_num, None, para_text, para_sort, references=para_refs)
                
                for item in items:
                    item_num = item.get('item_no', '')
                    item_content = item.get('content')
                    item_sort = self._resolve_sort_no(item_num, "item")
                    item_refs = self._extract_references(item_content)
                    item_node = self._create_node("item", item_num, None, item_content, item_sort, references=item_refs)
                    
                    for sub in item.get('subitems', []):
                        sub_num = sub.get('subitem_no', '')
                        sub_sort = self._resolve_sort_no(sub_num, "subitem")
                        sub_content = sub.get('content', '')
                        sub_refs = self._extract_references(sub_content)
                        sub_node = self._create_node("subitem", sub_num, None, sub_content, sub_sort, references=sub_refs)
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
                 
                 # Extract references for the addenda header text if any
                 add_refs = self._extract_references(content) if not children_nodes else []
                 
                 addenda.append(self._create_node(
                     "addendum", 
                     "", 
                     header, 
                     final_text, 
                     children=children_nodes,
                     references=add_refs,
                     metadata={"has_text": bool(final_text)}
                 ))
    
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
            
            # 1. Article Style (제1조)
            art_match = re.match(r'^(제\s*\d+\s*조(?:의\s*\d+)?)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if art_match:
                # Use 'article' type even in Addenda for better structure
                display_no = art_match.group(1)
                title = art_match.group(2)
                content = art_match.group(3)
                sort_no = self._resolve_sort_no(display_no, "article")
                refs = self._extract_references(content)
                nodes.append(self._create_node("addendum_item", display_no, title, content, sort_no, references=refs))
                current_node = nodes[-1]
                continue
                
            # 2. Numbered Item Style acting as Article (1. (시행일)...)
            num_match = re.match(r'^(\d+\.)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if num_match:
                display_no = num_match.group(1)
                title = num_match.group(2)
                content = num_match.group(3)
                sort_no = self._resolve_sort_no(display_no, "item") # mimic item sort
                refs = self._extract_references(content)
                nodes.append(self._create_node("addendum_item", display_no, title, content, sort_no, references=refs))
                current_node = nodes[-1]
                continue
            
            # 3. Paragraph Style (①)
            para_match = re.match(r'^([①-⑮])\s*(.*)', line)
            if para_match:
                display_no = para_match.group(1)
                content = para_match.group(2)
                sort_no = self._resolve_sort_no(display_no, "paragraph")
                refs = self._extract_references(content)
                
                # If inside an article/item, add as child
                if current_node:
                    current_node["children"].append(self._create_node("paragraph", display_no, None, content, sort_no, references=refs))
                else:
                    # Orphan paragraph -> treat as Addenda Item
                    nodes.append(self._create_node("addendum_item", display_no, None, content, sort_no, references=refs))
                    current_node = nodes[-1]
                continue
            
            # 4. Text Content (append to current node)
            if current_node:
                if current_node["text"]:
                    if line.startswith("|"):
                        current_node["text"] += "\n" + line
                    else:
                        current_node["text"] += " " + line
                else:
                    current_node["text"] = line
            else:
                nodes.append(self._create_node("text", "", None, line, confidence_score=0.5))
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
            
            is_valid_title = False
            if cleaned.endswith(suffixes):
                is_valid_title = True
            elif "규정" in cleaned:
                 # Filter out sentences
                 if "시행한다" not in cleaned and not re.match(r'^\d+\.', cleaned):
                     is_valid_title = True
            
            if is_valid_title:
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
            # NOTE: Avoid splitting on "제1조의2" etc (it must be the exact first article).
            article_1_match = re.match(r'^(제\s*1\s*조)(?!\s*의)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_1_match:
                if current_data["articles"] or current_article:
                     split_idx = -1
                     start_next_lines = []
                     if mode == 'APPENDICES':
                         for i in range(len(current_data["appendices"]) - 1, -1, -1):
                             txt = current_data["appendices"][i].strip()
                             title_candidates = ["규정", "세칙", "지침", "요령", "강령", "내규", "학칙", "헌장", "기준", "수칙", "준칙", "요강", "운영", "정관"]
                             
                             is_candidate = False
                             # 1. Standard Suffixes (Strong signal)
                             if any(txt.endswith(c) for c in title_candidates):
                                 is_candidate = True
                             # 2. Contains "규정" but is not a sentence/clause (Weak signal)
                             elif "규정" in txt:
                                 # Exclude common false positives:
                                 # - Starts with digit+dot (e.g. "1. (시행일)")
                                 # - Contains "시행한다" (Effective date clause)
                                 # - Starts with "부" (e.g. "부칙")
                                 if "시행한다" not in txt and not re.match(r'^\d+\.', txt) and not txt.startswith("부"):
                                     is_candidate = True
                                     
                             if is_candidate and len(txt) < 100:
                                 split_idx = i
                                 break
                         if split_idx != -1:
                             start_next_lines = current_data["appendices"][split_idx:]
                             current_data["appendices"] = current_data["appendices"][:split_idx]
                     flush_regulation(next_preamble_lines=start_next_lines)
            
            # Article
            # Update Regex to capture "의2" optional group for display purposes, but main parse logic here separates components
            # Group 1: Full Article No (제29조의2)
            # Group 2: Main No (29)
            # Group 3: Sub No (2) - Optional
            # Group 4: Title - Optional
            # Group 5: Content
            article_match = re.match(r'^(제\s*(\d+)\s*조(?:의\s*(\d+))?)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_match:
                mode = 'ARTICLES'
                if current_article: current_data["articles"].append(current_article)
                
                # We store the full display string as 'article_no' for the intermediate dict
                # The components (main/sub) will be resolved in _build_hierarchy via _resolve_sort_no
                article_no = article_match.group(1) 
                article_title = article_match.group(4) or ""
                content = (article_match.group(5) or "").lstrip()
                
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
            # Check for "부칙" OR "Attached Files" headers ([별표...], <별표...>)
            if re.match(r'^부\s*칙', line) or re.match(r'^(?:\[\s*별\s*[표지].*?\]|<\s*별\s*[표지].*?>)', line):
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
                    if not current_paragraph:
                        # Create implicit paragraph for items directly under article
                        current_paragraph = {
                            "paragraph_no": "", 
                            "content": "",
                            "items": []
                        }
                        current_article["paragraphs"].append(current_paragraph)
                        
                    current_paragraph["items"].append(new_item)
                    current_item = new_item
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
                    if line.startswith("|"):
                        current_paragraph["content"] += "\n" + line
                    else:
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
