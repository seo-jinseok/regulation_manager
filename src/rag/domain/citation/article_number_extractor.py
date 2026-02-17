"""
Article Number Extraction Utility.

Extracts structured article numbers from regulation chunk titles.
Supports various Korean regulation citation formats.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ArticleType(Enum):
    """Type of article reference."""

    ARTICLE = "article"  # 제N조
    SUB_ARTICLE = "sub_article"  # 제N조의M
    CHAPTER = "chapter"  # 제N장
    TABLE = "table"  # 별표N
    FORM = "form"  # 서식N
    PARAGRAPH = "paragraph"  # 제N항 (standalone)
    ITEM = "item"  # 제N호 (standalone)
    NONE = "none"  # No article number


@dataclass
class ArticleNumber:
    """
    Structured article number extracted from title.

    Attributes:
        type: Type of article (article, sub_article, table, form, etc.)
        number: Primary number (e.g., 26 from "제26조")
        sub_number: Sub-number for 제N조의M format (e.g., 2 from "제10조의2")
        paragraph_number: Paragraph number for 제N항 format (e.g., 1 from "제1항")
        item_number: Item number for 제N호 format (e.g., 2 from "제2호")
        prefix: Korean prefix (제, 별표, 서식)
        suffix: Korean suffix (조, 장, 항, 호)
        full_text: Full matched text (e.g., "제26조", "별표1", "제26조제1항")
    """

    type: ArticleType
    number: int
    sub_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    item_number: Optional[int] = None
    prefix: str = ""
    suffix: str = ""
    full_text: str = ""

    def __str__(self) -> str:
        """Return formatted article number string."""
        if self.type == ArticleType.SUB_ARTICLE and self.sub_number:
            base = f"{self.prefix}{self.number}조의{self.sub_number}"
        else:
            base = f"{self.prefix}{self.number}{self.suffix}"

        # Append paragraph if present
        if self.paragraph_number is not None:
            base = f"{base}제{self.paragraph_number}항"

        # Append item if present
        if self.item_number is not None:
            base = f"{base}제{self.item_number}호"

        return base

    def to_citation_format(self) -> str:
        """
        Return article number in standard citation format.

        Examples:
            - "제26조"
            - "제10조의2"
            - "별표1"
            - "서식1"
            - "제26조제1항"
            - "제26조제1항제2호"
        """
        return self.full_text if self.full_text else str(self)


class ArticleNumberExtractor:
    """
    Extracts article numbers from Korean regulation titles.

    Supports:
    - 제N조 (article)
    - 제N조의M (sub-article)
    - 제N장 (chapter)
    - 별표N (table)
    - 서식N (form)
    - 제N항 (paragraph) - standalone
    - 제N호 (item) - standalone
    - Combined: 제N조제M항, 제N조제M항제P호, 제N조의M제P항
    """

    # Patterns for article number extraction (ordered by specificity)
    PATTERNS = [
        # Combined patterns (most specific first)
        # 제N조의M제P항제Q호 (sub-article with paragraph and item)
        (ArticleType.SUB_ARTICLE, r"제(\d+)조의(\d+)제(\d+)항제(\d+)호"),
        # 제N조의M제P항 (sub-article with paragraph)
        (ArticleType.SUB_ARTICLE, r"제(\d+)조의(\d+)제(\d+)항"),
        # 제N조제M항제P호 (article with paragraph and item)
        (ArticleType.ARTICLE, r"제(\d+)조제(\d+)항제(\d+)호"),
        # 제N조제M항 (article with paragraph)
        (ArticleType.ARTICLE, r"제(\d+)조제(\d+)항"),
        # 제N조제M호 (article with item)
        (ArticleType.ARTICLE, r"제(\d+)조제(\d+)호"),
        # 제N조의M (sub-article) - must come before 제N조
        (ArticleType.SUB_ARTICLE, r"제(\d+)조의(\d+)"),
        # 제N장 (chapter)
        (ArticleType.CHAPTER, r"제(\d+)장"),
        # 별표N (table/appendix)
        (ArticleType.TABLE, r"별표(\d+)"),
        # 서식N (form)
        (ArticleType.FORM, r"서식(\d+)"),
        # 제N항 (paragraph) - standalone
        (ArticleType.PARAGRAPH, r"제(\d+)항"),
        # 제N호 (item) - standalone
        (ArticleType.ITEM, r"제(\d+)호"),
        # 제N조 (article) - most common, check last
        (ArticleType.ARTICLE, r"제(\d+)조"),
    ]

    def __init__(self):
        """Initialize article number extractor."""
        # Compile patterns for performance
        self._compiled_patterns = [
            (art_type, re.compile(pattern)) for art_type, pattern in self.PATTERNS
        ]

    def extract(self, title: str) -> Optional[ArticleNumber]:
        """
        Extract article number from title.

        Args:
            title: Chunk title text (e.g., "제26조 (직원의 구분)")

        Returns:
            ArticleNumber if found, None otherwise
        """
        if not title:
            return None

        # Try each pattern in order
        for art_type, pattern in self._compiled_patterns:
            match = pattern.search(title)
            if match:
                return self._parse_match(match, art_type, title)

        # No article number found
        return None

    def _parse_match(
        self, match: re.Match, art_type: ArticleType, title: str
    ) -> ArticleNumber:
        """Parse regex match into ArticleNumber."""
        groups = match.groups()
        num_groups = len(groups)

        if art_type == ArticleType.SUB_ARTICLE:
            number = int(groups[0])
            sub_number = int(groups[1])
            prefix = "제"
            suffix = "조"

            # Build full text based on captured groups
            if num_groups == 4:
                # 제N조의M제P항제Q호
                paragraph_number = int(groups[2])
                item_number = int(groups[3])
                full_text = f"제{number}조의{sub_number}제{paragraph_number}항제{item_number}호"
                return ArticleNumber(
                    type=art_type,
                    number=number,
                    sub_number=sub_number,
                    paragraph_number=paragraph_number,
                    item_number=item_number,
                    prefix=prefix,
                    suffix=suffix,
                    full_text=full_text,
                )
            elif num_groups == 3:
                # 제N조의M제P항
                paragraph_number = int(groups[2])
                full_text = f"제{number}조의{sub_number}제{paragraph_number}항"
                return ArticleNumber(
                    type=art_type,
                    number=number,
                    sub_number=sub_number,
                    paragraph_number=paragraph_number,
                    prefix=prefix,
                    suffix=suffix,
                    full_text=full_text,
                )
            else:
                # 제N조의M
                full_text = f"제{number}조의{sub_number}"
                return ArticleNumber(
                    type=art_type,
                    number=number,
                    sub_number=sub_number,
                    prefix=prefix,
                    suffix=suffix,
                    full_text=full_text,
                )

        elif art_type == ArticleType.ARTICLE:
            number = int(groups[0])
            prefix = "제"
            suffix = "조"

            # Build full text based on captured groups
            if num_groups == 3:
                # 제N조제M항제P호 or 제N조제M호
                # Check which pattern matched based on the actual matched text
                matched_text = match.group(0)
                if "항" in matched_text and "호" in matched_text:
                    # 제N조제M항제P호
                    paragraph_number = int(groups[1])
                    item_number = int(groups[2])
                    full_text = f"제{number}조제{paragraph_number}항제{item_number}호"
                    return ArticleNumber(
                        type=art_type,
                        number=number,
                        paragraph_number=paragraph_number,
                        item_number=item_number,
                        prefix=prefix,
                        suffix=suffix,
                        full_text=full_text,
                    )
                else:
                    # 제N조제M호
                    item_number = int(groups[1])
                    full_text = f"제{number}조제{item_number}호"
                    return ArticleNumber(
                        type=art_type,
                        number=number,
                        item_number=item_number,
                        prefix=prefix,
                        suffix=suffix,
                        full_text=full_text,
                    )
            elif num_groups == 2:
                # Could be 제N조제M항 or 제N조제M호
                # Check the actual matched text to determine type
                matched_text = match.group(0)
                if "호" in matched_text:
                    # 제N조제M호
                    item_number = int(groups[1])
                    full_text = f"제{number}조제{item_number}호"
                    return ArticleNumber(
                        type=art_type,
                        number=number,
                        item_number=item_number,
                        prefix=prefix,
                        suffix=suffix,
                        full_text=full_text,
                    )
                else:
                    # 제N조제M항
                    paragraph_number = int(groups[1])
                    full_text = f"제{number}조제{paragraph_number}항"
                    return ArticleNumber(
                        type=art_type,
                        number=number,
                        paragraph_number=paragraph_number,
                        prefix=prefix,
                        suffix=suffix,
                        full_text=full_text,
                    )
            else:
                # 제N조
                full_text = f"제{number}조"
                return ArticleNumber(
                    type=art_type,
                    number=number,
                    prefix=prefix,
                    suffix=suffix,
                    full_text=full_text,
                )

        elif art_type == ArticleType.CHAPTER:
            # 제N장 format
            number = int(groups[0])
            prefix = "제"
            suffix = "장"
            full_text = f"제{number}장"

            return ArticleNumber(
                type=art_type,
                number=number,
                prefix=prefix,
                suffix=suffix,
                full_text=full_text,
            )

        elif art_type == ArticleType.TABLE:
            # 별표N format
            number = int(groups[0])
            prefix = "별표"
            suffix = ""
            full_text = f"별표{number}"

            return ArticleNumber(
                type=art_type,
                number=number,
                prefix=prefix,
                suffix=suffix,
                full_text=full_text,
            )

        elif art_type == ArticleType.FORM:
            # 서식N format
            number = int(groups[0])
            prefix = "서식"
            suffix = ""
            full_text = f"서식{number}"

            return ArticleNumber(
                type=art_type,
                number=number,
                prefix=prefix,
                suffix=suffix,
                full_text=full_text,
            )

        elif art_type == ArticleType.PARAGRAPH:
            # 제N항 format (standalone paragraph)
            number = int(groups[0])
            prefix = "제"
            suffix = "항"
            full_text = f"제{number}항"

            return ArticleNumber(
                type=art_type,
                number=number,
                prefix=prefix,
                suffix=suffix,
                full_text=full_text,
            )

        elif art_type == ArticleType.ITEM:
            # 제N호 format (standalone item)
            number = int(groups[0])
            prefix = "제"
            suffix = "호"
            full_text = f"제{number}호"

            return ArticleNumber(
                type=art_type,
                number=number,
                prefix=prefix,
                suffix=suffix,
                full_text=full_text,
            )

        # Fallback (shouldn't reach here)
        return ArticleNumber(type=ArticleType.NONE, number=0)

    def extract_all(self, text: str) -> list[ArticleNumber]:
        """
        Extract all article numbers from text.

        Args:
            text: Text to search (e.g., regulation content)

        Returns:
            List of ArticleNumber objects in order of appearance
        """
        if not text:
            return []

        results = []
        # Find all matches across all patterns
        for art_type, pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                article_num = self._parse_match(match, art_type, text)
                results.append(article_num)

        # Sort by position in text
        results.sort(key=lambda a: text.index(a.full_text) if a.full_text else 0)

        return results

    def is_article_level(self, title: str) -> bool:
        """
        Check if title represents an article-level chunk.

        Args:
            title: Chunk title

        Returns:
            True if title contains 제N조 or 제N조의M pattern
        """
        article = self.extract(title)
        return article is not None and article.type in (
            ArticleType.ARTICLE,
            ArticleType.SUB_ARTICLE,
        )
