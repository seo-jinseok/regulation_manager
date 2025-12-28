#!/usr/bin/env python3
"""
Generate a synonym dictionary from regulation JSON using a local LLM.

Example:
  uv run python scripts/generate_synonyms.py \\
    --json-path data/output/regulations.json \\
    --output data/synonyms.json \\
    --provider lmstudio --model eeve-korean-instruct-7b-v2.0-preview-mlx \\
    --base-url http://localhost:1234
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.rag.infrastructure.hybrid_search import QueryAnalyzer
from src.rag.infrastructure.json_loader import JSONDocumentLoader
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

SYSTEM_PROMPT = """You are a lexicon generator for Korean university regulations.
Given Korean regulatory terms, generate colloquial, misspelled, or simplified variants.

Rules:
- Keep each list short (3-5 items).
- Only include Korean variants (no explanations).
- Do not repeat the original term.
- Output JSON only (no markdown)."""


def _normalize_term(term: str) -> str:
    term = term.replace("\u3000", " ").strip()
    term = re.sub(r"\s+", " ", term)
    term = term.strip("()[]{}<>:;,.\"'")
    return term


def _extract_terms(
    json_path: Path,
    min_length: int,
    max_terms: Optional[int],
) -> List[str]:
    loader = JSONDocumentLoader()
    chunks = loader.load_all_chunks(str(json_path))

    stopwords = set(QueryAnalyzer.STOPWORDS)
    counter: Counter[str] = Counter()

    for chunk in chunks:
        candidates: List[str] = []
        if chunk.title:
            candidates.append(chunk.title)
        if chunk.parent_path:
            candidates.extend(chunk.parent_path)
        if chunk.keywords:
            candidates.extend(kw.term for kw in chunk.keywords if kw.term)

        for raw in candidates:
            if not isinstance(raw, str):
                continue
            term = _normalize_term(raw)
            if not term:
                continue
            if len(term) < min_length:
                continue
            if term in stopwords:
                continue
            if QueryAnalyzer.ARTICLE_PATTERN.search(term):
                continue
            if term.isdigit():
                continue
            counter[term] += 1

    if not counter:
        return []

    ranked = [term for term, _ in counter.most_common()]
    if max_terms:
        ranked = ranked[:max_terms]
    return ranked


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _extract_json(text: str) -> Optional[Dict[str, List[str]]]:
    try:
        data = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return None

    if not isinstance(data, dict):
        return None
    return data


def _merge_synonyms(
    base: Dict[str, List[str]],
    updates: Dict[str, List[str]],
    max_synonyms: int,
) -> None:
    for term, synonyms in updates.items():
        if not isinstance(term, str):
            continue
        term = term.strip()
        if not term:
            continue
        if isinstance(synonyms, str):
            synonyms_list = [synonyms]
        elif isinstance(synonyms, list):
            synonyms_list = synonyms
        else:
            continue

        cleaned = []
        for synonym in synonyms_list:
            if not isinstance(synonym, str):
                continue
            synonym = synonym.strip()
            if not synonym or synonym == term:
                continue
            cleaned.append(synonym)

        if not cleaned:
            continue

        existing = base.setdefault(term, [])
        for synonym in cleaned:
            if synonym not in existing:
                existing.append(synonym)
            if len(existing) >= max_synonyms:
                break


def _generate_with_llm(
    terms: List[str],
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    batch_size: int,
    max_synonyms: int,
) -> Dict[str, List[str]]:
    llm = LLMClientAdapter(
        provider=provider,
        model=model or None,
        base_url=base_url or None,
    )

    synonyms: Dict[str, List[str]] = {}
    for batch in _chunked(terms, batch_size):
        prompt_lines = ["Terms:"]
        prompt_lines.extend(f"- {term}" for term in batch)
        prompt_lines.append(
            f"\nReturn JSON mapping each term to up to {max_synonyms} variants."
        )

        response = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message="\n".join(prompt_lines),
            temperature=0.0,
        )
        parsed = _extract_json(response)
        if not parsed:
            continue
        _merge_synonyms(synonyms, parsed, max_synonyms=max_synonyms)

    return synonyms


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synonym dictionary.")
    parser.add_argument("--json-path", required=True, help="Regulation JSON path")
    parser.add_argument("--output", default="data/synonyms.json", help="Output JSON")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="LLM base URL")
    parser.add_argument("--batch-size", type=int, default=20, help="LLM batch size")
    parser.add_argument(
        "--max-terms", type=int, default=300, help="Max terms to expand"
    )
    parser.add_argument("--min-length", type=int, default=2, help="Minimum term length")
    parser.add_argument(
        "--max-synonyms",
        type=int,
        default=4,
        help="Maximum synonyms per term",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (output empty terms list).",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    terms = _extract_terms(
        json_path=json_path,
        min_length=args.min_length,
        max_terms=args.max_terms,
    )

    synonyms: Dict[str, List[str]] = {}
    if not args.no_llm and terms:
        synonyms = _generate_with_llm(
            terms=terms,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            max_synonyms=args.max_synonyms,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_json": str(json_path),
        "selection": {
            "min_length": args.min_length,
            "max_terms": args.max_terms,
            "term_count": len(terms),
        },
        "llm": {
            "provider": None if args.no_llm else args.provider,
            "model": None if args.no_llm else args.model,
            "base_url": None if args.no_llm else args.base_url,
            "batch_size": None if args.no_llm else args.batch_size,
        },
        "terms": synonyms,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Wrote {output_path} ({len(synonyms)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
