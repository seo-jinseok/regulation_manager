#!/usr/bin/env python3
"""
Learn from Failures - Automatic Intent Generation from Failed Queries.

This script analyzes failed queries from evaluation results and automatically
generates new intents/triggers/synonyms to improve future performance.

Usage:
    uv run python scripts/learn_from_failures.py --analyze
    uv run python scripts/learn_from_failures.py --generate
    uv run python scripts/learn_from_failures.py --apply
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.infrastructure.llm_adapter import LLMClientAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FailedQuery:
    """Represents a failed query from evaluation."""

    query: str
    expected_keywords: list[str]
    expected_regulation: Optional[str]
    actual_results: list[dict]
    failure_reason: str
    score: float


@dataclass
class GeneratedIntent:
    """Represents an auto-generated intent."""

    id: str
    label: str
    triggers: list[str]
    patterns: list[str]
    keywords: list[str]
    audience: str = "all"
    weight: float = 1.5
    source_query: str = ""
    confidence: float = 0.0


@dataclass
class GeneratedSynonym:
    """Represents an auto-generated synonym group."""

    canonical: str
    synonyms: list[str]
    source_query: str = ""


# System prompt for intent generation
INTENT_GENERATION_PROMPT = """당신은 대학 규정 검색 시스템의 전문가입니다.
실패한 검색 쿼리를 분석하여 새로운 인텐트를 생성합니다.

## 작업
주어진 실패 쿼리와 기대 결과를 분석하여 intents.json에 추가할 새 인텐트를 생성하세요.

## 출력 형식 (반드시 JSON)
{
    "id": "intent_id_snake_case",
    "label": "의도 설명 (한글)",
    "triggers": [
        "원본 쿼리",
        "유사 표현 1",
        "유사 표현 2"
    ],
    "patterns": [
        "정규식 패턴 (선택사항)"
    ],
    "keywords": [
        "검색 키워드 1",
        "검색 키워드 2"
    ],
    "audience": "all|student|faculty|staff",
    "confidence": 0.9
}

## 규칙
1. id는 영문 snake_case로 작성
2. triggers는 최소 3개, 원본 쿼리 포함
3. keywords는 규정에서 실제로 사용되는 용어 포함
4. audience는 대상에 따라 설정 (학생=student, 교원=faculty, 직원=staff)"""


SYNONYM_GENERATION_PROMPT = """당신은 대학 규정 검색 시스템의 전문가입니다.
실패한 검색 쿼리를 분석하여 동의어 그룹을 생성합니다.

## 작업
주어진 쿼리에서 추출한 키워드와 기대 키워드 간의 동의어 관계를 찾으세요.

## 출력 형식 (반드시 JSON)
{
    "synonyms": [
        {
            "canonical": "대표 용어 (규정에서 사용하는 공식 용어)",
            "variants": ["일상 표현 1", "일상 표현 2", "줄임말"]
        }
    ]
}

## 규칙
1. canonical은 규정에서 실제로 사용하는 공식 용어
2. variants는 일상에서 사용하는 비격식 표현
3. 기존 동의어와 중복되지 않도록 확인"""


class FailureAnalyzer:
    """Analyzes evaluation results to find failed queries."""

    def __init__(
        self,
        evaluation_path: str = "data/config/evaluation_dataset.json",
        feedback_path: str = "data/feedback_log.jsonl",
    ):
        self._evaluation_path = Path(evaluation_path)
        self._feedback_path = Path(feedback_path)

    def load_evaluation_dataset(self) -> list[dict]:
        """Load evaluation dataset."""
        if not self._evaluation_path.exists():
            logger.warning(f"Evaluation dataset not found: {self._evaluation_path}")
            return []

        with open(self._evaluation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("queries", [])

    def load_feedback_log(self) -> list[dict]:
        """Load feedback log (negative feedback = potential failures)."""
        if not self._feedback_path.exists():
            return []

        entries = []
        with open(self._feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def find_failed_queries(
        self,
        min_score: float = 0.5,
        max_results: int = 10,
    ) -> list[FailedQuery]:
        """
        Find queries that failed in evaluation.

        Args:
            min_score: Minimum score threshold (queries below this are failures).
            max_results: Maximum number of failures to return.

        Returns:
            List of FailedQuery objects.
        """
        # Load evaluation results (assuming auto_evaluate.py output format)
        results_path = Path("data/evaluation_results.json")
        if not results_path.exists():
            logger.info("No evaluation results found. Run auto_evaluate.py first.")
            return []

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        failures = []
        for result in results.get("query_results", []):
            score = result.get("score", 0)
            if score < min_score:
                failures.append(
                    FailedQuery(
                        query=result.get("query", ""),
                        expected_keywords=result.get("expected_keywords", []),
                        expected_regulation=result.get("expected_regulation"),
                        actual_results=result.get("results", []),
                        failure_reason=result.get("failure_reason", "unknown"),
                        score=score,
                    )
                )

        # Sort by score (worst first)
        failures.sort(key=lambda x: x.score)
        return failures[:max_results]

    def find_negative_feedback(self) -> list[FailedQuery]:
        """Find queries with negative user feedback."""
        feedback = self.load_feedback_log()
        failures = []

        for entry in feedback:
            if entry.get("rating") == "negative" or entry.get("helpful") is False:
                failures.append(
                    FailedQuery(
                        query=entry.get("query", ""),
                        expected_keywords=[],
                        expected_regulation=entry.get("regulation"),
                        actual_results=[],
                        failure_reason="negative_feedback",
                        score=0.0,
                    )
                )

        return failures


class IntentGenerator:
    """Generates intents from failed queries using LLM."""

    def __init__(self, llm_client: Optional[LLMClientAdapter] = None):
        self._llm = llm_client
        self._existing_intents: dict[str, Any] = {}
        self._existing_synonyms: dict[str, list[str]] = {}

    def load_existing_intents(
        self, path: str = "data/config/intents.json"
    ) -> None:
        """Load existing intents to avoid duplicates."""
        intents_path = Path(path)
        if intents_path.exists():
            with open(intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for intent in data.get("intents", []):
                    self._existing_intents[intent["id"]] = intent

    def load_existing_synonyms(
        self, path: str = "data/config/synonyms.json"
    ) -> None:
        """Load existing synonyms to avoid duplicates."""
        synonyms_path = Path(path)
        if synonyms_path.exists():
            with open(synonyms_path, "r", encoding="utf-8") as f:
                self._existing_synonyms = json.load(f)

    def _is_duplicate_intent(self, intent: GeneratedIntent) -> bool:
        """Check if intent already exists or is too similar."""
        if intent.id in self._existing_intents:
            return True

        # Check if any triggers overlap significantly
        for existing in self._existing_intents.values():
            existing_triggers = set(t.lower() for t in existing.get("triggers", []))
            new_triggers = set(t.lower() for t in intent.triggers)
            overlap = existing_triggers & new_triggers
            if len(overlap) >= 2:  # More than 1 trigger overlaps
                return True

        return False

    def generate_intent(self, failed: FailedQuery) -> Optional[GeneratedIntent]:
        """
        Generate a new intent from a failed query.

        Args:
            failed: Failed query information.

        Returns:
            GeneratedIntent or None if generation failed.
        """
        if not self._llm:
            return self._generate_intent_heuristic(failed)

        try:
            prompt = f"""실패한 쿼리: {failed.query}
기대 키워드: {', '.join(failed.expected_keywords)}
기대 규정: {failed.expected_regulation or '없음'}
실패 이유: {failed.failure_reason}

위 정보를 바탕으로 새 인텐트를 생성하세요."""

            response = self._llm.generate(
                system_prompt=INTENT_GENERATION_PROMPT,
                user_message=prompt,
                temperature=0.3,
            )

            # Parse response
            intent_data = self._parse_json_response(response)
            if not intent_data:
                return None

            intent = GeneratedIntent(
                id=intent_data.get("id", "unknown"),
                label=intent_data.get("label", ""),
                triggers=intent_data.get("triggers", [failed.query]),
                patterns=intent_data.get("patterns", []),
                keywords=intent_data.get("keywords", failed.expected_keywords),
                audience=intent_data.get("audience", "all"),
                source_query=failed.query,
                confidence=intent_data.get("confidence", 0.7),
            )

            # Check for duplicates
            if self._is_duplicate_intent(intent):
                logger.info(f"Skipping duplicate intent: {intent.id}")
                return None

            return intent

        except Exception as e:
            logger.warning(f"LLM intent generation failed: {e}")
            return self._generate_intent_heuristic(failed)

    def _generate_intent_heuristic(self, failed: FailedQuery) -> Optional[GeneratedIntent]:
        """Generate intent using simple heuristics (no LLM)."""
        # Extract key terms from query
        query = failed.query
        words = re.findall(r"[가-힣A-Za-z0-9]+", query)
        keywords = [w for w in words if len(w) >= 2]

        if not keywords:
            return None

        # Generate ID from keywords
        intent_id = "_".join(keywords[:2]).lower()
        if len(intent_id) < 3:
            intent_id = f"auto_{intent_id}"

        intent = GeneratedIntent(
            id=intent_id,
            label=f"자동생성: {query[:20]}",
            triggers=[query],
            patterns=[],
            keywords=failed.expected_keywords or keywords,
            audience="all",
            source_query=query,
            confidence=0.5,
        )

        if self._is_duplicate_intent(intent):
            return None

        return intent

    def generate_synonyms(self, failed: FailedQuery) -> list[GeneratedSynonym]:
        """
        Generate synonym groups from failed query.

        Args:
            failed: Failed query information.

        Returns:
            List of GeneratedSynonym objects.
        """
        if not self._llm:
            return []

        try:
            prompt = f"""실패한 쿼리: {failed.query}
기대 키워드: {', '.join(failed.expected_keywords)}

쿼리에서 사용된 표현과 기대 키워드 간의 동의어 관계를 찾으세요."""

            response = self._llm.generate(
                system_prompt=SYNONYM_GENERATION_PROMPT,
                user_message=prompt,
                temperature=0.3,
            )

            data = self._parse_json_response(response)
            if not data:
                return []

            synonyms = []
            for syn in data.get("synonyms", []):
                canonical = syn.get("canonical", "")
                variants = syn.get("variants", [])

                # Skip if already exists
                if canonical in self._existing_synonyms:
                    continue

                synonyms.append(
                    GeneratedSynonym(
                        canonical=canonical,
                        synonyms=variants,
                        source_query=failed.query,
                    )
                )

            return synonyms

        except Exception as e:
            logger.warning(f"Synonym generation failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from text
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None


class ConfigUpdater:
    """Updates configuration files with generated intents/synonyms."""

    def __init__(
        self,
        intents_path: str = "data/config/intents.json",
        synonyms_path: str = "data/config/synonyms.json",
    ):
        self._intents_path = Path(intents_path)
        self._synonyms_path = Path(synonyms_path)

    def backup_config(self, path: Path) -> Path:
        """Create backup of config file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".{timestamp}.backup.json")
        if path.exists():
            import shutil
            shutil.copy(path, backup_path)
            logger.info(f"Backed up {path} to {backup_path}")
        return backup_path

    def add_intents(
        self,
        intents: list[GeneratedIntent],
        dry_run: bool = True,
    ) -> int:
        """
        Add generated intents to intents.json.

        Args:
            intents: List of intents to add.
            dry_run: If True, only preview changes without writing.

        Returns:
            Number of intents added.
        """
        if not intents:
            return 0

        # Load existing
        with open(self._intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        existing_ids = {i["id"] for i in data.get("intents", [])}
        added = 0

        for intent in intents:
            if intent.id in existing_ids:
                logger.info(f"Skipping existing intent: {intent.id}")
                continue

            new_intent = {
                "id": intent.id,
                "label": intent.label,
                "triggers": intent.triggers,
                "patterns": intent.patterns,
                "keywords": intent.keywords,
                "audience": intent.audience,
                "weight": intent.weight,
                "_auto_generated": True,
                "_source_query": intent.source_query,
                "_confidence": intent.confidence,
            }

            if dry_run:
                logger.info(f"[DRY RUN] Would add intent: {intent.id}")
                print(json.dumps(new_intent, ensure_ascii=False, indent=2))
            else:
                data["intents"].append(new_intent)
                existing_ids.add(intent.id)

            added += 1

        if not dry_run and added > 0:
            # Backup before writing
            self.backup_config(self._intents_path)

            # Update metadata
            data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

            with open(self._intents_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Added {added} intents to {self._intents_path}")

        return added

    def add_synonyms(
        self,
        synonyms: list[GeneratedSynonym],
        dry_run: bool = True,
    ) -> int:
        """
        Add generated synonyms to synonyms.json.

        Args:
            synonyms: List of synonyms to add.
            dry_run: If True, only preview changes without writing.

        Returns:
            Number of synonym groups added.
        """
        if not synonyms:
            return 0

        # Load existing
        with open(self._synonyms_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        added = 0

        for syn in synonyms:
            if syn.canonical in data:
                # Merge with existing
                existing = set(data[syn.canonical])
                new_variants = set(syn.synonyms) - existing
                if new_variants:
                    if dry_run:
                        logger.info(
                            f"[DRY RUN] Would add to '{syn.canonical}': {new_variants}"
                        )
                    else:
                        data[syn.canonical].extend(list(new_variants))
                    added += 1
            else:
                # New synonym group
                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would add synonym group: {syn.canonical} -> {syn.synonyms}"
                    )
                else:
                    data[syn.canonical] = syn.synonyms
                added += 1

        if not dry_run and added > 0:
            # Backup before writing
            self.backup_config(self._synonyms_path)

            with open(self._synonyms_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Added/updated {added} synonym groups in {self._synonyms_path}")

        return added


def main():
    parser = argparse.ArgumentParser(
        description="Learn from failed queries and generate intents/synonyms"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze evaluation results to find failed queries",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate intents/synonyms from failed queries (dry run)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated intents/synonyms to config files",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum score threshold for failures (default: 0.5)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="Maximum number of failures to process (default: 10)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for better intent generation (requires ollama/lmstudio)",
    )

    args = parser.parse_args()

    if not any([args.analyze, args.generate, args.apply]):
        parser.print_help()
        return

    # Initialize components
    analyzer = FailureAnalyzer()

    llm_client = None
    if args.use_llm:
        try:
            llm_client = LLMClientAdapter(provider="ollama")
            logger.info("LLM client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")

    generator = IntentGenerator(llm_client)
    generator.load_existing_intents()
    generator.load_existing_synonyms()

    updater = ConfigUpdater()

    # Step 1: Analyze
    if args.analyze or args.generate or args.apply:
        logger.info("Analyzing failed queries...")
        failures = analyzer.find_failed_queries(
            min_score=args.min_score,
            max_results=args.max_failures,
        )

        if not failures:
            # Try loading from evaluation dataset manually
            logger.info("No evaluation results found, checking feedback log...")
            failures = analyzer.find_negative_feedback()

        if failures:
            logger.info(f"Found {len(failures)} failed queries:")
            for i, f in enumerate(failures, 1):
                print(f"  {i}. [{f.score:.2f}] {f.query}")
                print(f"      Expected: {f.expected_keywords}")
                print(f"      Reason: {f.failure_reason}")
        else:
            logger.info("No failed queries found. System is working well!")
            return

    # Step 2: Generate
    if args.generate or args.apply:
        logger.info("\nGenerating intents from failures...")
        generated_intents = []
        generated_synonyms = []

        for failure in failures:
            # Generate intent
            intent = generator.generate_intent(failure)
            if intent:
                generated_intents.append(intent)
                logger.info(f"Generated intent: {intent.id} (confidence={intent.confidence:.2f})")

            # Generate synonyms
            synonyms = generator.generate_synonyms(failure)
            generated_synonyms.extend(synonyms)

        logger.info(
            f"\nGenerated {len(generated_intents)} intents, "
            f"{len(generated_synonyms)} synonym groups"
        )

    # Step 3: Apply
    if args.apply:
        logger.info("\nApplying changes to config files...")
        updater.add_intents(generated_intents, dry_run=False)
        updater.add_synonyms(generated_synonyms, dry_run=False)
        logger.info("Changes applied successfully!")
    elif args.generate:
        # Dry run preview
        logger.info("\n[DRY RUN] Preview of changes:")
        updater.add_intents(generated_intents, dry_run=True)
        updater.add_synonyms(generated_synonyms, dry_run=True)


if __name__ == "__main__":
    main()
