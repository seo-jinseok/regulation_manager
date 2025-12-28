"""Legacy refinement helpers for offline analysis; not used in main pipeline."""

import json
import re

INPUT_PATH = "data/output/규정집9-349(20251202).json"
OUTPUT_PATH = "data/output/규정집_refined.json"


def clean_preamble_and_get_title(doc, index):
    preamble = doc.get("preamble", "").strip()

    # Special handling for the first regulation (University Statute/Preamble)
    # The first doc usually contains the book's TOC and other meta info in preamble.
    if index == 0:
        # Simplistic approach: The title is usually "학교법인동의학원정관" or similar in the last line of valid preamble
        # But looking at previous inspection, it ends with "제1장 총 칙"
        # Let's hardcode for safety or use regex
        return "학교법인동의학원정관", ""

    # General extraction
    lines = preamble.split("\n")
    title = ""
    # Usually the first line is the title, possibly followed by revision info
    if lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip if line looks like revision info only e.g. <개정 2022...> or (개정 ...)
            # But sometimes the title includes it? e.g. "학칙 (개정 2024)"
            # Strategy: Take the first non-empty line. Strip <...> parenthesis at the end.

            # Remove <...> or (...) patterns at the END of the string
            # Check if line starts with < or ( and ends with > or ) -> likely just meta info line
            if (line.startswith("<") and line.endswith(">")) or (
                line.startswith("(") and line.endswith(")")
            ):
                continue

            cleaned_line = re.sub(r"\s*[<\[\(].*?[>\]\)]", "", line).strip()
            # If cleaned line is empty (e.g. line was just "<2022.1.1. 개정>"), skip
            if not cleaned_line:
                continue

            # Remove "제1장 ..." if it leaked into the title line
            cleaned_line = re.sub(r"\s*제\s*\d+\s*장.*", "", cleaned_line).strip()

            title = cleaned_line
            break

    if not title:
        print(
            f"[DEBUG] Title extraction failed for index {index}. Preamble:\n---\n{preamble}\n---"
        )

    return title, preamble


def process_articles(articles):
    refined_articles = []
    current_chapter = ""
    current_part = ""  # For "제N편"

    for art in articles:
        new_content = []

        # Pre-scan content for headers (Chapter, Part)
        # Because headers might be just lines in content.

        for line in art.get("content", []):
            line = line.strip()
            # Check for Part (제N편)
            match_part = re.match(r"^(제\s*\d+\s*편\s.*)", line)
            if match_part:
                current_part = match_part.group(1).strip()
                continue  # consume

            # Check for Chapter (제N장)
            match_chapter = re.match(r"^(제\s*\d+\s*장\s.*)", line)
            if match_chapter:
                current_chapter = match_chapter.group(1).strip()
                continue  # consume

            new_content.append(line)

        art["content"] = new_content

        if current_part:
            art["part"] = current_part
        if current_chapter:
            art["chapter"] = current_chapter

        refined_articles.append(art)

    return refined_articles


def parse_articles_from_text(text):
    """
    Mini-parser to extract Article/Paragraph structure from text block (e.g. Addenda).
    Uses similar regex to RegulationFormatter but simplified.
    """
    articles = []
    current_article = None

    lines = text.split("\n")

    # Heuristic: If text doesn't look like articles, return as one "content" chunk.
    # Check if "제1조" or "1." exists?

    # Generic content fallback
    generic_content = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect Article
        # e.g. "제1조(시행일)"
        art_match = re.match(r"^(제\s*(\d+)\s*조)\s*(?:\(([^)]+)\))?\s*(.*)", line)
        if art_match:
            if current_article:
                # Flush previous article
                if generic_content:
                    current_article["content"].extend(generic_content)
                    generic_content = []
                articles.append(current_article)

            current_article = {
                "article_no": art_match.group(2),
                "title": art_match.group(3) or "",
                "content": [],
                "paragraphs": [],
            }
            context = art_match.group(4)
            if context:
                current_article["content"].append(context)
            continue

        # If we have an article, capture content
        if current_article:
            current_article["content"].append(line)
        else:
            generic_content.append(line)

    # Flush last
    if current_article:
        if generic_content:
            current_article["content"].extend(generic_content)
        articles.append(current_article)
    else:
        # No article headers found, treat as one blob (but return list of string? No, caller expects objects?)
        # Caller expects 'content' field in item.
        pass

    # If no structured articles found, return None distinctively so caller uses raw string
    if not articles:
        return None

    return articles


def parse_appendices(appendix_text):
    if not appendix_text:
        return [], []

    # Default return structure
    addenda = []
    attached_files = []

    # 1. Normalize
    text = appendix_text.strip()

    # 2. Split into blocks based on headers
    pattern = r"(?:^|\n)(?:\|\s*)?((?:부\s*칙)|(?:\[별표.*?\])|(?:\[별지.*?\]))"
    segments = re.split(pattern, text)

    idx = 1
    while idx < len(segments):
        header = segments[idx].strip()
        content = segments[idx + 1].strip() if idx + 1 < len(segments) else ""

        if "부" in header and "칙" in header:
            # It's an Addendum. Try to structure it.
            structured_articles = parse_articles_from_text(content)

            item_data = {
                "title": header,
            }
            if structured_articles:
                item_data["articles"] = structured_articles
            else:
                item_data["content"] = content

            addenda.append(item_data)

        elif "별표" in header or "별지" in header:
            # It's an attachment
            attached_files.append({"title": header, "content": content})

        idx += 2

    return addenda, attached_files


def refine_doc(doc, index):
    new_doc = doc.copy()

    # 1. Title
    title, cleaned_preamble = clean_preamble_and_get_title(doc, index)
    new_doc["title"] = title
    new_doc["preamble"] = cleaned_preamble

    # 2. Articles & Chapters
    new_doc["articles"] = process_articles(doc.get("articles", []))

    # 3. Appendices
    addenda, attachments = parse_appendices(doc.get("appendices", ""))
    new_doc["addenda"] = addenda
    new_doc["attached_files"] = attachments

    # Remove raw appendices field to stay clean
    new_doc.pop("appendices", None)

    return new_doc


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    refined_docs = [refine_doc(d, i) for i, d in enumerate(data["docs"])]

    data["docs"] = refined_docs

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved refined JSON to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
