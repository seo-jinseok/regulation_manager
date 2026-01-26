"""
Dictionary Manager for RAG System.

Infrastructure layer for automated management of intents.json and synonyms.json
with LLM-based recommendations and conflict detection.

Clean Architecture: Infrastructure implements domain interfaces and manages
external resources (JSON files).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentEntry:
    """Represents an intent entry for intents.json."""

    id: str
    label: str
    triggers: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    audience: str = "all"
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format for JSON serialization."""
        d = {
            "id": self.id,
            "label": self.label,
            "triggers": self.triggers,
            "patterns": self.patterns,
            "keywords": self.keywords,
        }
        if self.audience != "all":
            d["audience"] = self.audience
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass(frozen=True)
class SynonymEntry:
    """Represents a synonym entry for synonyms.json."""

    term: str
    synonyms: List[str]
    context: str = "regulation_query"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "term": self.term,
            "synonyms": self.synonyms,
            "context": self.context,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""

    conflict_type: str  # "duplicate_id", "similar_triggers", "overlapping_keywords"
    existing_entry: Dict
    new_entry: Dict
    severity: str  # "error", "warning", "info"
    message: str


@dataclass
class RecommendationResult:
    """Result of LLM-based recommendation."""

    recommended_intents: List[IntentEntry] = field(default_factory=list)
    recommended_synonyms: List[SynonymEntry] = field(default_factory=list)
    conflicts: List[ConflictInfo] = field(default_factory=list)
    llm_used: bool = False
    processing_time_seconds: float = 0.0


class DictionaryManager:
    """
    Manages intents.json and synonyms.json with automated operations.

    Features:
    - Load/save dictionary files
    - LLM-based intent and synonym recommendations
    - Conflict detection and resolution
    - Automatic merging with existing entries
    - Version tracking and backup
    """

    # LLM prompt for intent recommendation
    INTENT_RECOMMENDATION_PROMPT = """
당신은 대학 규정 검색 시스템의 전문가입니다. 실패한 검색 쿼리를 분석하여
시스템 개선을 위한 새로운 인텐트(Intent)와 동의어(Synonym)를 추천해주세요.

## 실패한 쿼리 정보
- **쿼리**: {query}
- **원인**: {root_cause}
- **제안된 수정**: {suggested_fix}

## 기존 인텐트 목록 (참고용)
{existing_intents_summary}

## 추천 형식
다음 JSON 형식으로만 응답하세요:

```json
{{
  "intents": [
    {{
      "id": "unique_intent_id",
      "label": "인텐트 라벨 (한글)",
      "triggers": ["트리거1", "트리거2"],
      "patterns": ["정규식 패턴1", "정규식 패턴2"],
      "keywords": ["키워드1", "키워드2", "키워드3"],
      "audience": "student|faculty|staff|all",
      "weight": 1.0
    }}
  ],
  "synonyms": [
    {{
      "term": "기본 용어",
      "synonyms": ["동의어1", "동의어2"],
      "context": "regulation_query"
    }}
  ]
}}
```

## 주의사항
1. id는 snake_case로 작성하고 유일해야 합니다
2. triggers는 실제 사용자가 입력할 법한 자연스러운 표현
3. patterns는 정규식 형태 (예: "(휴학|쉬고).*싶")
4. keywords는 규정 검색에 사용할 핵심 용어
5. audience는 해당 인텐트의 주요 대상 (student, faculty, staff, all)
"""

    def __init__(
        self,
        intents_path: Optional[Path] = None,
        synonyms_path: Optional[Path] = None,
        llm_client: Optional["ILLMClient"] = None,
    ):
        """
        Initialize DictionaryManager.

        Args:
            intents_path: Path to intents.json file.
            synonyms_path: Path to synonyms.json file.
            llm_client: Optional LLM client for recommendations.
        """
        self.intents_path = intents_path or Path("data/config/intents.json")
        self.synonyms_path = synonyms_path or Path("data/config/synonyms.json")
        self._llm_client = llm_client

        # Load existing dictionaries
        self._intents_data: Dict = self._load_json(self.intents_path, {})
        self._synonyms_data: Dict = self._load_json(self.synonyms_path, {})

    def _load_json(self, path: Path, default: Dict) -> Dict:
        """Load JSON file with error handling."""
        if not path.exists():
            logger.warning(f"File not found: {path}, using defaults")
            return default

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return default

    def get_all_intents(self) -> List[Dict]:
        """Get all intent entries."""
        return self._intents_data.get("intents", [])

    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """Get all synonym entries."""
        return self._synonyms_data.get("terms", {})

    def intent_id_exists(self, intent_id: str) -> bool:
        """Check if intent ID already exists."""
        return any(
            intent.get("id") == intent_id for intent in self.get_all_intents()
        )

    def find_similar_intents(
        self, triggers: List[str], keywords: List[str], threshold: float = 0.5
    ) -> List[Dict]:
        """
        Find intents with similar triggers or keywords.

        Args:
            triggers: List of triggers to compare.
            keywords: List of keywords to compare.
            threshold: Similarity threshold (0.0 to 1.0).

        Returns:
            List of similar intents with similarity scores.
        """
        similar = []
        all_intents = self.get_all_intents()

        for intent in all_intents:
            intent_triggers = intent.get("triggers", [])
            intent_keywords = intent.get("keywords", [])

            # Calculate overlap ratio
            trigger_overlap = self._calculate_overlap(triggers, intent_triggers)
            keyword_overlap = self._calculate_overlap(keywords, intent_keywords)

            # Use maximum overlap as similarity score
            similarity = max(trigger_overlap, keyword_overlap)

            if similarity >= threshold:
                similar.append({**intent, "_similarity": similarity})

        return sorted(similar, key=lambda x: x["_similarity"], reverse=True)

    def _calculate_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate overlap ratio between two lists."""
        if not list1 or not list2:
            return 0.0

        set1 = set(list1)
        set2 = set(list2)
        intersection = set1 & set2

        return len(intersection) / min(len(set1), len(set2))

    def synonym_term_exists(self, term: str) -> bool:
        """Check if synonym term already exists."""
        return term in self.get_all_synonyms()

    def detect_conflicts(
        self, intent: IntentEntry, synonym_entries: List[SynonymEntry]
    ) -> List[ConflictInfo]:
        """
        Detect conflicts with existing entries.

        Args:
            intent: Intent entry to check.
            synonym_entries: Synonym entries to check.

        Returns:
            List of detected conflicts.
        """
        conflicts = []

        # Check for duplicate intent ID
        if self.intent_id_exists(intent.id):
            existing = next(
                (i for i in self.get_all_intents() if i.get("id") == intent.id), None
            )
            if existing:
                conflicts.append(
                    ConflictInfo(
                        conflict_type="duplicate_id",
                        existing_entry=existing,
                        new_entry=intent.to_dict(),
                        severity="error",
                        message=f"Intent ID '{intent.id}' already exists",
                    )
                )

        # Check for similar intents
        similar = self.find_similar_intents(intent.triggers, intent.keywords, threshold=0.3)
        for sim in similar:
            if sim.get("id") != intent.id:  # Exclude self
                conflicts.append(
                    ConflictInfo(
                        conflict_type="similar_triggers",
                        existing_entry=sim,
                        new_entry=intent.to_dict(),
                        severity="warning",
                        message=f"Similar intent exists: {sim.get('id')} "
                        f"(similarity: {sim.get('_similarity', 0):.2f})",
                    )
                )

        # Check synonym conflicts
        for syn_entry in synonym_entries:
            if self.synonym_term_exists(syn_entry.term):
                existing_synonyms = self.get_all_synonyms().get(syn_entry.term, [])
                conflicts.append(
                    ConflictInfo(
                        conflict_type="duplicate_synonym_term",
                        existing_entry={"term": syn_entry.term, "synonyms": existing_synonyms},
                        new_entry=syn_entry.to_dict(),
                        severity="warning",
                        message=f"Synonym term '{syn_entry.term}' already exists",
                    )
                )

        return conflicts

    def recommend_from_failure(
        self,
        query: str,
        root_cause: str,
        suggested_fix: str,
        use_llm: bool = True,
    ) -> RecommendationResult:
        """
        Generate recommendations from failed query analysis.

        Args:
            query: The failed query.
            root_cause: Root cause analysis.
            suggested_fix: Suggested fix.
            use_llm: Whether to use LLM for recommendations.

        Returns:
            RecommendationResult with recommendations and conflicts.
        """
        import time

        start_time = time.time()
        result = RecommendationResult()

        # Build existing intents summary for context
        existing_summary = self._build_intents_summary()

        if use_llm and self._llm_client:
            try:
                prompt = self.INTENT_RECOMMENDATION_PROMPT.format(
                    query=query,
                    root_cause=root_cause,
                    suggested_fix=suggested_fix,
                    existing_intents_summary=existing_summary,
                )

                response = self._llm_client.generate(
                    system_prompt="당신은 대학 규정 검색 시스템의 전문가입니다.",
                    user_message=prompt,
                    temperature=0.3,
                )

                recommendations = self._parse_llm_recommendations(response)

                # Detect conflicts
                all_conflicts = []
                for intent in recommendations.recommended_intents:
                    all_conflicts.extend(self.detect_conflicts(intent, []))

                for syn in recommendations.recommended_synonyms:
                    all_conflicts.extend(self.detect_conflicts(IntentEntry("", "", []), [syn]))

                result = RecommendationResult(
                    recommended_intents=recommendations.recommended_intents,
                    recommended_synonyms=recommendations.recommended_synonyms,
                    conflicts=all_conflicts,
                    llm_used=True,
                    processing_time_seconds=time.time() - start_time,
                )

            except Exception as e:
                logger.error(f"LLM recommendation failed: {e}")
                result.llm_used = False
                result.processing_time_seconds = time.time() - start_time
        else:
            result.llm_used = False
            result.processing_time_seconds = time.time() - start_time

        return result

    def _build_intents_summary(self) -> str:
        """Build a summary of existing intents for LLM context."""
        intents = self.get_all_intents()
        if not intents:
            return "기존 인텐트 없음"

        summary_lines = []
        for intent in intents[:50]:  # Limit to prevent context overflow
            id_val = intent.get("id", "")
            label = intent.get("label", "")
            triggers = intent.get("triggers", [])[:3]  # First 3 triggers
            summary_lines.append(f"- {id_val}: {label} (triggers: {', '.join(triggers)})")

        return "\n".join(summary_lines)

    def _parse_llm_recommendations(self, response: str) -> RecommendationResult:
        """Parse LLM response into RecommendationResult."""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)

            result = RecommendationResult()

            # Parse intents
            for intent_data in data.get("intents", []):
                try:
                    intent = IntentEntry(
                        id=intent_data.get("id", ""),
                        label=intent_data.get("label", ""),
                        triggers=intent_data.get("triggers", []),
                        patterns=intent_data.get("patterns", []),
                        keywords=intent_data.get("keywords", []),
                        audience=intent_data.get("audience", "all"),
                        weight=intent_data.get("weight", 1.0),
                        metadata=intent_data.get("metadata", {}),
                    )
                    result.recommended_intents.append(intent)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse intent: {e}")
                    continue

            # Parse synonyms
            for syn_data in data.get("synonyms", []):
                try:
                    synonym = SynonymEntry(
                        term=syn_data.get("term", ""),
                        synonyms=syn_data.get("synonyms", []),
                        context=syn_data.get("context", "regulation_query"),
                        metadata=syn_data.get("metadata", {}),
                    )
                    result.recommended_synonyms.append(synonym)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse synonym: {e}")
                    continue

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM recommendations JSON: {e}")
            return RecommendationResult()

    def add_intent(self, intent: IntentEntry, merge: bool = True) -> bool:
        """
        Add an intent to the dictionary.

        Args:
            intent: IntentEntry to add.
            merge: If True, merge with existing entry if ID exists.

        Returns:
            True if successfully added, False otherwise.
        """
        intents = self._intents_data.setdefault("intents", [])

        if merge:
            # Check for existing intent with same ID
            for i, existing in enumerate(intents):
                if existing.get("id") == intent.id:
                    # Merge with existing
                    merged = existing.copy()
                    merged["triggers"] = list(set(merged.get("triggers", []) + intent.triggers))
                    merged["patterns"] = list(set(merged.get("patterns", []) + intent.patterns))
                    merged["keywords"] = list(set(merged.get("keywords", []) + intent.keywords))

                    # Update metadata
                    if not merged.get("metadata"):
                        merged["metadata"] = {}
                    merged["metadata"].update(intent.metadata)
                    merged["metadata"]["last_updated"] = datetime.now().isoformat()

                    intents[i] = merged
                    return True

        # Add new intent
        intents.append(intent.to_dict())
        return True

    def add_synonym(self, synonym: SynonymEntry, merge: bool = True) -> bool:
        """
        Add a synonym to the dictionary.

        Args:
            synonym: SynonymEntry to add.
            merge: If True, merge with existing entry if term exists.

        Returns:
            True if successfully added, False otherwise.
        """
        terms = self._synonyms_data.setdefault("terms", {})

        if merge and synonym.term in terms:
            # Merge with existing
            existing = terms[synonym.term]
            merged = list(set(existing + synonym.synonyms))
            terms[synonym.term] = merged
        else:
            terms[synonym.term] = synonym.synonyms

        return True

    def save(self, create_backup: bool = True) -> Tuple[bool, bool]:
        """
        Save dictionaries to files.

        Args:
            create_backup: If True, create backup files before saving.

        Returns:
            Tuple of (intents_saved, synonyms_saved) success flags.
        """
        intents_saved = self._save_intents(create_backup)
        synonyms_saved = self._save_synonyms(create_backup)
        return intents_saved, synonyms_saved

    def _save_intents(self, create_backup: bool) -> bool:
        """Save intents.json file."""
        try:
            if create_backup and self.intents_path.exists():
                backup_path = self.intents_path.with_suffix(
                    f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.intents_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Update version and timestamp
            self._intents_data["version"] = self._increment_version(
                self._intents_data.get("version", "1.0.0")
            )
            self._intents_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

            with open(self.intents_path, "w", encoding="utf-8") as f:
                json.dump(self._intents_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved intents to {self.intents_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save intents: {e}")
            return False

    def _save_synonyms(self, create_backup: bool) -> bool:
        """Save synonyms.json file."""
        try:
            if create_backup and self.synonyms_path.exists():
                backup_path = self.synonyms_path.with_suffix(
                    f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.synonyms_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Update version and timestamp
            self._synonyms_data["version"] = self._increment_version(
                self._synonyms_data.get("version", "1.0.0")
            )
            self._synonyms_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

            with open(self.synonyms_path, "w", encoding="utf-8") as f:
                json.dump(self._synonyms_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved synonyms to {self.synonyms_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save synonyms: {e}")
            return False

    def _increment_version(self, version: str) -> str:
        """Increment version string (patch version)."""
        try:
            parts = version.split(".")
            if len(parts) >= 3:
                patch = int(parts[2]) + 1
                return f"{'.'.join(parts[:2])}.{patch}"
            return version
        except (ValueError, IndexError):
            return version

    def get_stats(self) -> Dict:
        """Get statistics about the dictionaries."""
        intents = self.get_all_intents()
        synonyms = self.get_all_synonyms()

        total_triggers = sum(len(intent.get("triggers", [])) for intent in intents)
        total_patterns = sum(len(intent.get("patterns", [])) for intent in intents)
        total_keywords = sum(len(intent.get("keywords", [])) for intent in intents)
        total_synonyms = sum(len(syn_list) for syn_list in synonyms.values())

        return {
            "intents": {
                "count": len(intents),
                "total_triggers": total_triggers,
                "total_patterns": total_patterns,
                "total_keywords": total_keywords,
            },
            "synonyms": {
                "count": len(synonyms),
                "total_synonyms": total_synonyms,
            },
            "versions": {
                "intents": self._intents_data.get("version", "unknown"),
                "synonyms": self._synonyms_data.get("version", "unknown"),
            },
        }
