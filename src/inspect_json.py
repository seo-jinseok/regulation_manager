import argparse
import glob
import json
import re


def analyze_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        docs = data
        file_name = None
    else:
        docs = data.get("docs", [])
        file_name = data.get("file_name")

    if file_name:
        print(f"File Name: {file_name}")
    print(f"\nTotal Regulations (docs): {len(docs)}")

    def collect_articles(nodes, collected):
        for node in nodes:
            if node.get("type") == "article":
                collected.append(node)
            collect_articles(node.get("children", []), collected)

    for i, doc in enumerate(docs):
        print(f"\n--- Regulation {i + 1} ---")
        preamble = doc.get("preamble", "")
        print(f"Preamble length: {len(preamble)} chars")
        print(f"Preamble start: {preamble[:100].replace(chr(10), ' ')}...")

        articles = []
        collect_articles(doc.get("content", []), articles)
        print(f"Total Articles: {len(articles)}")

        if articles:
            print(
                f"First Article: {articles[0].get('display_no')} - {articles[0].get('title')}"
            )
            print(
                f"Last Article: {articles[-1].get('display_no')} - {articles[-1].get('title')}"
            )

        chapter_in_content = 0
        for art in articles:
            content_str = art.get("text") or ""
            if re.search(r"제\s*\d+\s*장", content_str):
                chapter_in_content += 1

        print(f"Articles with 'Chapter' in content: {chapter_in_content}")

        if articles:
            last = articles[-1]
            last_content = last.get("text") or ""
            if "부칙" in last_content or "시행일" in last_content:
                print("  [Info] Last article seems to contain Addenda info.")


def _resolve_input_path(input_path: str) -> str:
    if input_path:
        return input_path
    candidates = [
        p for p in glob.glob("data/output/*.json") if not p.endswith("_metadata.json")
    ]
    if not candidates:
        raise FileNotFoundError("No JSON files found in data/output.")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Inspect a regulation JSON file.")
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a JSON file (default: first in data/output)",
    )
    args = parser.parse_args()
    path = _resolve_input_path(args.input_path)
    analyze_json(path)


if __name__ == "__main__":
    main()
