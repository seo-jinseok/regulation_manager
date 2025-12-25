"""
Core regulation text parser.

Provides utilities for parsing regulation text into structured
intermediate format for further processing.
"""

import re
from typing import Dict, List, Any, Optional
import uuid


class RegulationParser:
    """
    Parses regulation text into structured format.

    Handles:
    - Hierarchical structure (Part > Chapter > Section > Article)
    - Paragraphs, items, subitems
    - Preamble and appendices
    """

    def parse_flat(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse regulation text into flat intermediate structure.

        Args:
            text: The raw regulation text.

        Returns:
            List of regulation dictionaries with articles, preamble, appendices.
        """
        lines = text.split("\n")
        regulations: List[Dict[str, Any]] = []
        current_data: Dict[str, Any] = {
            "preamble": [],
            "articles": [],
            "appendices": [],
        }
        current_article: Optional[Dict[str, Any]] = None
        current_paragraph: Optional[Dict[str, Any]] = None
        current_item: Optional[Dict[str, Any]] = None
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        regulation_title: Optional[str] = None
        current_book_part: Optional[str] = None
        mode = "PREAMBLE"

        def flush_regulation(next_preamble_lines: Optional[List[str]] = None) -> None:
            nonlocal current_article, current_paragraph, current_item, mode
            nonlocal current_data, current_chapter, current_section
            nonlocal current_subsection, regulation_title

            if current_article:
                current_data["articles"].append(current_article)
                current_article = None
                current_paragraph = None
                current_item = None

            if (
                current_data["articles"]
                or current_data["preamble"]
                or current_data["appendices"]
            ):
                reg = {
                    "part": current_book_part,
                    "title": regulation_title,
                    "preamble": current_data["preamble"],
                    "articles": current_data["articles"],
                    "appendices": current_data["appendices"],
                }
                regulations.append(reg)

            current_data = {
                "preamble": next_preamble_lines if next_preamble_lines else [],
                "articles": [],
                "appendices": [],
            }
            mode = "PREAMBLE"
            current_chapter = None
            current_section = None
            current_subsection = None
            regulation_title = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Part (Groups Regulations)
            part_match = re.match(r"^\|?\s*(제\s*\d+\s*편)\s*(.*)\|?$", line)
            if not part_match:
                clean_line = line.replace("|", "").strip()
                part_match = re.match(r"^(제\s*\d+\s*편)\s*(.*)", clean_line)

            if part_match:
                flush_regulation()
                p_num = part_match.group(1).strip()
                p_name = part_match.group(2).replace("|", "").strip()
                current_book_part = f"{p_num} {p_name}".strip()
                continue

            # Chapter
            chapter_match = re.match(r"^(제\s*\d+\s*[장편])\s*(.*)", line)
            if chapter_match:
                current_chapter = line
                current_section = None
                current_subsection = None
                continue

            # Section (절)
            section_match = re.match(r"^(제\s*\d+\s*절)\s*(.*)", line)
            if section_match:
                current_section = line
                current_subsection = None
                continue

            # Subsection (관)
            subsection_match = re.match(r"^(제\s*\d+\s*관)\s*(.*)", line)
            if subsection_match:
                current_subsection = line
                continue

            # TOC
            if re.match(r"^차\s*례\s*$", line) or re.match(r"^목\s*차\s*$", line):
                flush_regulation()
                regulation_title = "차례"
                current_data["preamble"].append(line)
                continue

            # Index
            if re.match(r"^찾아보기.*", line) and len(line) < 20:
                flush_regulation()
                regulation_title = "찾아보기"
                current_data["preamble"].append(line)
                continue

            # Article 1 Split
            article_1_match = re.match(
                r"^(제\s*1\s*조)(?!\s*의)\s*(?:\(([^)]+)\))?\s*(.*)", line
            )
            if article_1_match:
                if current_data["articles"] or current_article:
                    split_idx = -1
                    start_next_lines: List[str] = []
                    if mode == "APPENDICES":
                        for i in range(len(current_data["appendices"]) - 1, -1, -1):
                            txt = current_data["appendices"][i].strip()
                            title_candidates = [
                                "규정", "세칙", "지침", "요령", "강령",
                                "내규", "학칙", "헌장", "기준", "수칙",
                                "준칙", "요강", "운영", "정관",
                            ]
                            is_candidate = False
                            if any(txt.endswith(c) for c in title_candidates):
                                is_candidate = True
                            elif "규정" in txt:
                                if (
                                    "시행한다" not in txt
                                    and not re.match(r"^\d+\.", txt)
                                    and not txt.startswith("부")
                                ):
                                    is_candidate = True
                            if is_candidate and len(txt) < 100:
                                split_idx = i
                                break
                        if split_idx != -1:
                            start_next_lines = current_data["appendices"][split_idx:]
                            current_data["appendices"] = current_data["appendices"][
                                :split_idx
                            ]
                    flush_regulation(next_preamble_lines=start_next_lines)

            # Article
            article_match = re.match(
                r"^(제\s*(\d+)\s*조(?:의\s*(\d+))?)\s*(?:\(([^)]+)\))?\s*(.*)", line
            )
            if article_match:
                mode = "ARTICLES"
                if current_article:
                    current_data["articles"].append(current_article)

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
                    "paragraphs": [],
                }
                current_paragraph = None
                current_item = None

                if content:
                    para_match = re.match(r"^([①-⑮])\s*(.*)", content)
                    if para_match:
                        current_paragraph = {
                            "paragraph_no": para_match.group(1),
                            "content": para_match.group(2),
                            "items": [],
                        }
                        current_article["paragraphs"].append(current_paragraph)
                    else:
                        current_article["content"].append(content)
                continue

            # Appendices
            if re.match(r"^부\s*칙", line) or re.match(
                r"^(?:\[\s*별\s*[표지].*?\]|<\s*별\s*[표지].*?>)", line
            ):
                mode = "APPENDICES"
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
            if mode == "ARTICLES":
                para_match = re.match(r"^([①-⑮])\s*(.*)", line)
                if para_match and current_article:
                    current_paragraph = {
                        "paragraph_no": para_match.group(1),
                        "content": para_match.group(2),
                        "items": [],
                    }
                    current_article["paragraphs"].append(current_paragraph)
                    current_item = None
                    continue

                # Item (1., 2., 3.)
                item_match = re.match(r"^(\d+\.)\s*(.*)", line)
                if item_match and current_article:
                    new_item = {
                        "item_no": item_match.group(1),
                        "content": item_match.group(2),
                        "subitems": [],
                    }
                    if not current_paragraph:
                        current_paragraph = {
                            "paragraph_no": "",
                            "content": "",
                            "items": [],
                        }
                        current_article["paragraphs"].append(current_paragraph)

                    current_paragraph["items"].append(new_item)
                    current_item = new_item
                    continue

                # Subitem (가., 나., 다.)
                subitem_match = re.match(r"^([가-하]\.)\s*(.*)", line)
                if subitem_match and current_article:
                    if current_item:
                        current_item["subitems"].append(
                            {
                                "subitem_no": subitem_match.group(1),
                                "content": subitem_match.group(2),
                            }
                        )
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

            elif mode == "APPENDICES":
                current_data["appendices"].append(line)
            elif mode == "PREAMBLE":
                current_data["preamble"].append(line)

        flush_regulation()
        return regulations

    def resolve_sort_no(self, display_no: str, node_type: str) -> Dict[str, int]:
        """
        Resolve display string into sorting key.

        Args:
            display_no: The display number (e.g., "제29조의2", "①").
            node_type: The type of node.

        Returns:
            Dict with 'main' and 'sub' keys.
        """
        main = 0
        sub = 0

        if node_type == "article":
            match = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", display_no)
            if match:
                main = int(match.group(1))
                if match.group(2):
                    sub = int(match.group(2))

        elif node_type in ["chapter", "section", "subsection", "part"]:
            match = re.search(r"(\d+)", display_no)
            if match:
                main = int(match.group(1))

        elif node_type == "paragraph":
            clean = display_no.strip()
            if clean and len(clean) == 1:
                code = ord(clean)
                if 0x2460 <= code <= 0x2473:  # ① ~ ⑳
                    main = code - 0x2460 + 1

        elif node_type == "item":
            match = re.match(r"^(\d+)", display_no)
            if match:
                main = int(match.group(1))

        elif node_type == "subitem":
            match = re.match(r"^([가-하])", display_no)
            if match:
                char = match.group(1)
                order = "가나다라마바사아자차카타파하"
                if char in order:
                    main = order.index(char) + 1

        return {"main": main, "sub": sub}

    def create_node(
        self,
        node_type: str,
        display_no: str,
        title: Optional[str],
        text: Optional[str],
        sort_no: Optional[Dict[str, int]] = None,
        children: Optional[List[Dict[str, Any]]] = None,
        confidence_score: float = 1.0,
        references: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a structured node.

        Args:
            node_type: Type of the node (article, paragraph, etc.).
            display_no: Display number string.
            title: Optional title.
            text: Optional text content.
            sort_no: Sorting key.
            children: Child nodes.
            confidence_score: Parsing confidence.
            references: Cross-references.
            metadata: Additional metadata.

        Returns:
            Node dictionary.
        """
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
            "children": children if children is not None else [],
        }
