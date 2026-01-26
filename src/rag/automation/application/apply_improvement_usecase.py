"""
Apply Improvement Use Case.

Application layer for applying automated improvements to the RAG system
based on test failure analysis.

Clean Architecture: Application layer orchestrates domain and infrastructure.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ...domain.repositories import ILLMClient
    from ..domain.entities import TestResult
    from ..domain.value_objects import FiveWhyAnalysis

logger = logging.getLogger(__name__)

# Dictionary manager for automated intent/synonym management
try:
    from ...infrastructure.dictionary_manager import (
        DictionaryManager,
        IntentEntry,
        RecommendationResult,
        SynonymEntry,
    )
except ImportError:
    DictionaryManager = None
    IntentEntry = None
    SynonymEntry = None
    RecommendationResult = None


class ApplyImprovementUseCase:
    """
    Use Case for applying automated improvements to the RAG system.

    Takes 5-Why analysis results and applies patches to configuration
    files or generates code change suggestions.

    Enhanced with LLM-based dictionary automation for intents.json and synonyms.json.
    """

    def __init__(
        self,
        intents_path: Optional[Path] = None,
        synonyms_path: Optional[Path] = None,
        llm_client: Optional["ILLMClient"] = None,
    ):
        """
        Initialize the use case.

        Args:
            intents_path: Path to intents.json file.
            synonyms_path: Path to synonyms.json file.
            llm_client: Optional LLM client for recommendations.
        """
        # Default paths (can be overridden)
        self.intents_path = intents_path or Path("data/config/intents.json")
        self.synonyms_path = synonyms_path or Path("data/config/synonyms.json")

        self.logger = logging.getLogger(__name__)

        # Initialize dictionary manager if available
        self._dict_manager: Optional[DictionaryManager] = None
        if DictionaryManager is not None:
            self._dict_manager = DictionaryManager(
                intents_path=self.intents_path,
                synonyms_path=self.synonyms_path,
                llm_client=llm_client,
            )
        self._llm_client = llm_client

    def apply_improvements(
        self,
        analyses: List["FiveWhyAnalysis"],
        test_results: List["TestResult"],
        dry_run: bool = True,
    ) -> Dict[str, any]:
        """
        Apply improvements based on 5-Why analyses.

        Args:
            analyses: List of 5-Why analysis results.
            test_results: Corresponding test results.
            dry_run: If True, preview changes without applying.

        Returns:
            Dictionary with applied improvements and statistics.
        """
        self.logger.info(
            f"Applying improvements for {len(analyses)} analyses (dry_run={dry_run})"
        )

        results = {
            "intents_patches": [],
            "synonyms_patches": [],
            "code_suggestions": [],
            "config_suggestions": [],
            "llm_recommendations": [],
            "statistics": {
                "total_analyses": len(analyses),
                "intents_patches": 0,
                "synonyms_patches": 0,
                "code_changes": 0,
                "config_changes": 0,
                "llm_recommendations": 0,
            },
        }

        # Handle potentially different length lists (Python 3.9 compatibility)
        min_len = min(len(analyses), len(test_results))
        for i in range(min_len):
            analysis = analyses[i]
            test_result = test_results[i]
            # Skip if no patch target and no code change required
            if not analysis.component_to_patch and not analysis.code_change_required:
                continue

            # Try LLM-based recommendation first if available
            if self._dict_manager and self._llm_client:
                llm_result = self._apply_llm_recommendation(
                    analysis, test_result, dry_run
                )
                if llm_result:
                    results["llm_recommendations"].append(llm_result)
                    results["statistics"]["llm_recommendations"] += 1

            # Apply patch based on target
            if analysis.component_to_patch == "intents.json":
                patch_result = self._patch_intents(analysis, test_result, dry_run)
                if patch_result:
                    results["intents_patches"].append(patch_result)
                    results["statistics"]["intents_patches"] += 1

            elif analysis.component_to_patch == "synonyms.json":
                patch_result = self._patch_synonyms(analysis, test_result, dry_run)
                if patch_result:
                    results["synonyms_patches"].append(patch_result)
                    results["statistics"]["synonyms_patches"] += 1

            elif analysis.component_to_patch == "config":
                suggestion = self._generate_config_suggestion(analysis, test_result)
                results["config_suggestions"].append(suggestion)
                results["statistics"]["config_changes"] += 1

            elif analysis.code_change_required:
                suggestion = self._generate_code_suggestion(analysis, test_result)
                results["code_suggestions"].append(suggestion)
                results["statistics"]["code_changes"] += 1

        self.logger.info(
            f"Improvements complete: "
            f"{results['statistics']['intents_patches']} intents, "
            f"{results['statistics']['synonyms_patches']} synonyms, "
            f"{results['statistics']['code_changes']} code, "
            f"{results['statistics']['config_changes']} config, "
            f"{results['statistics']['llm_recommendations']} llm"
        )

        return results

    def _apply_llm_recommendation(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "TestResult",
        dry_run: bool,
    ) -> Optional[Dict]:
        """
        Apply LLM-based recommendation for failed queries.

        Args:
            analysis: Five-Why analysis result.
            test_result: Corresponding test result.
            dry_run: If True, preview without applying.

        Returns:
            Recommendation result dictionary, or None if not applicable.
        """
        if not self._dict_manager:
            return None

        try:
            # Get LLM recommendations
            recommendation = self._dict_manager.recommend_from_failure(
                query=test_result.query,
                root_cause=analysis.root_cause,
                suggested_fix=analysis.suggested_fix,
                use_llm=True,
            )

            if not recommendation.recommended_intents and not recommendation.recommended_synonyms:
                return None

            result = {
                "test_case_id": test_result.test_case_id,
                "query": test_result.query,
                "recommended_intents": len(recommendation.recommended_intents),
                "recommended_synonyms": len(recommendation.recommended_synonyms),
                "conflicts": len(recommendation.conflicts),
                "llm_used": recommendation.llm_used,
                "processing_time_seconds": recommendation.processing_time_seconds,
                "applied": False,
                "conflicts_detail": [
                    {
                        "type": c.conflict_type,
                        "severity": c.severity,
                        "message": c.message,
                    }
                    for c in recommendation.conflicts
                ],
            }

            # Apply recommendations if no conflicts and not dry_run
            if not dry_run and not recommendation.conflicts:
                for intent in recommendation.recommended_intents:
                    if self._dict_manager.add_intent(intent, merge=True):
                        result["applied"] = True

                for synonym in recommendation.recommended_synonyms:
                    if self._dict_manager.add_synonym(synonym, merge=True):
                        result["applied"] = True

                # Save changes
                self._dict_manager.save(create_backup=True)

            return result

        except Exception as e:
            self.logger.error(f"LLM recommendation failed: {e}")
            return None

    def batch_update_from_failures(
        self,
        failure_data: List[Dict[str, str]],
        dry_run: bool = True,
        auto_apply: bool = False,
    ) -> Dict[str, any]:
        """
        Batch update dictionaries from failure analysis results.

        Args:
            failure_data: List of failure dictionaries with query, root_cause, suggested_fix.
            dry_run: If True, preview without applying.
            auto_apply: If True, automatically apply non-conflicting changes.

        Returns:
            Dictionary with batch update results.
        """
        if not self._dict_manager:
            return {
                "success": False,
                "error": "DictionaryManager not available",
                "statistics": {},
            }

        results = {
            "processed": 0,
            "intents_added": 0,
            "synonyms_added": 0,
            "conflicts_detected": 0,
            "recommendations": [],
            "statistics": {},
        }

        for failure in failure_data:
            query = failure.get("query", "")
            root_cause = failure.get("root_cause", "")
            suggested_fix = failure.get("suggested_fix", "")

            if not query:
                continue

            try:
                recommendation = self._dict_manager.recommend_from_failure(
                    query=query,
                    root_cause=root_cause,
                    suggested_fix=suggested_fix,
                    use_llm=True,
                )

                results["processed"] += 1
                results["intents_added"] += len(recommendation.recommended_intents)
                results["synonyms_added"] += len(recommendation.recommended_synonyms)
                results["conflicts_detected"] += len(recommendation.conflicts)

                results["recommendations"].append({
                    "query": query,
                    "intents": len(recommendation.recommended_intents),
                    "synonyms": len(recommendation.recommended_synonyms),
                    "conflicts": len(recommendation.conflicts),
                })

                # Auto-apply if requested and no conflicts
                if auto_apply and not dry_run and not recommendation.conflicts:
                    for intent in recommendation.recommended_intents:
                        self._dict_manager.add_intent(intent, merge=True)
                    for synonym in recommendation.recommended_synonyms:
                        self._dict_manager.add_synonym(synonym, merge=True)

            except Exception as e:
                self.logger.error(f"Failed to process failure '{query[:50]}...': {e}")

        # Save if auto_apply and not dry_run
        if auto_apply and not dry_run:
            self._dict_manager.save(create_backup=True)

        # Get final statistics
        if self._dict_manager:
            results["statistics"] = self._dict_manager.get_stats()

        return results

    def get_dictionary_stats(self) -> Dict:
        """Get current dictionary statistics."""
        if self._dict_manager:
            return self._dict_manager.get_stats()
        return {}

    def _patch_intents(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "TestResult",
        dry_run: bool,
    ) -> Optional[Dict]:
        """
        Patch intents.json with new intent pattern.

        Args:
            analysis: Five-Why analysis result.
            test_result: Corresponding test result.
            dry_run: If True, preview without applying.

        Returns:
            Patch result dictionary, or None if patch not applicable.
        """
        if not self.intents_path.exists():
            self.logger.warning(f"Intents file not found: {self.intents_path}")
            return None

        # Load existing intents
        try:
            with open(self.intents_path, "r", encoding="utf-8") as f:
                intents_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load intents: {e}")
            return None

        # Generate new intent entry
        new_intent = self._generate_intent_entry(test_result.query, analysis)

        # Check if intent already exists
        intent_exists = any(
            intent.get("intent") == new_intent["intent"]
            for intent in intents_data.get("intents", [])
        )

        patch_result = {
            "test_case_id": test_result.test_case_id,
            "query": test_result.query,
            "new_intent": new_intent,
            "exists": intent_exists,
            "applied": False,
        }

        if intent_exists:
            self.logger.info(f"Intent already exists: {new_intent['intent']}")
            return patch_result

        if not dry_run:
            # Apply patch
            intents_data.setdefault("intents", []).append(new_intent)

            try:
                with open(self.intents_path, "w", encoding="utf-8") as f:
                    json.dump(intents_data, f, ensure_ascii=False, indent=2)
                patch_result["applied"] = True
                self.logger.info(f"Applied intent patch: {new_intent['intent']}")
            except Exception as e:
                self.logger.error(f"Failed to apply intent patch: {e}")
        else:
            self.logger.info(
                f"Would apply intent patch (dry run): {new_intent['intent']}"
            )

        return patch_result

    def _patch_synonyms(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "TestResult",
        dry_run: bool,
    ) -> Optional[Dict]:
        """
        Patch synonyms.json with new synonyms.

        Args:
            analysis: Five-Why analysis result.
            test_result: Corresponding test result.
            dry_run: If True, preview without applying.

        Returns:
            Patch result dictionary, or None if patch not applicable.
        """
        if not self.synonyms_path.exists():
            self.logger.warning(f"Synonyms file not found: {self.synonyms_path}")
            return None

        # Load existing synonyms
        try:
            with open(self.synonyms_path, "r", encoding="utf-8") as f:
                synonyms_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load synonyms: {e}")
            return None

        # Generate new synonym entry
        new_synonym = self._generate_synonym_entry(test_result.query, analysis)

        # Check if synonym already exists
        synonym_exists = any(
            syn.get("term") == new_synonym["term"]
            for syn in synonyms_data.get("synonyms", [])
        )

        patch_result = {
            "test_case_id": test_result.test_case_id,
            "query": test_result.query,
            "new_synonym": new_synonym,
            "exists": synonym_exists,
            "applied": False,
        }

        if synonym_exists:
            self.logger.info(f"Synonym already exists: {new_synonym['term']}")
            return patch_result

        if not dry_run:
            # Apply patch
            synonyms_data.setdefault("synonyms", []).append(new_synonym)

            try:
                with open(self.synonyms_path, "w", encoding="utf-8") as f:
                    json.dump(synonyms_data, f, ensure_ascii=False, indent=2)
                patch_result["applied"] = True
                self.logger.info(f"Applied synonym patch: {new_synonym['term']}")
            except Exception as e:
                self.logger.error(f"Failed to apply synonym patch: {e}")
        else:
            self.logger.info(
                f"Would apply synonym patch (dry run): {new_synonym['term']}"
            )

        return patch_result

    def _generate_intent_entry(self, query: str, analysis: "FiveWhyAnalysis") -> Dict:
        """Generate a new intent entry from query."""
        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Generate intent name
        intent_name = f"query_{self._sanitize_string(query[:30])}"

        return {
            "id": intent_name,
            "label": f"Query: {query[:30]}...",
            "triggers": [query],
            "patterns": [],
            "keywords": keywords,
            "metadata": {
                "generated_from": "automated_testing",
                "root_cause": analysis.root_cause,
            },
        }

    def _generate_synonym_entry(self, query: str, analysis: "FiveWhyAnalysis") -> Dict:
        """Generate a new synonym entry from query."""
        keywords = self._extract_keywords(query)

        primary_term = (
            keywords[0] if keywords else self._sanitize_string(query.split()[0])
        )

        return {
            "term": primary_term,
            "synonyms": keywords[1:5] if len(keywords) > 1 else [],
            "context": "regulation_query",
            "metadata": {
                "generated_from": "automated_testing",
                "root_cause": analysis.root_cause,
            },
        }

    def _generate_code_suggestion(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "TestResult",
    ) -> Dict:
        """
        Generate code change suggestion.

        Args:
            analysis: Five-Why analysis result.
            test_result: Corresponding test result.

        Returns:
            Code suggestion dictionary.
        """
        suggestion = {
            "test_case_id": test_result.test_case_id,
            "root_cause": analysis.root_cause,
            "suggested_fix": analysis.suggested_fix,
            "priority": self._assess_priority(analysis),
            "file_hint": self._suggest_file_to_modify(analysis),
            "code_hint": self._generate_code_hint(analysis),
        }

        return suggestion

    def _generate_config_suggestion(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "TestResult",
    ) -> Dict:
        """
        Generate configuration change suggestion.

        Args:
            analysis: Five-Why analysis result.
            test_result: Corresponding test result.

        Returns:
            Config suggestion dictionary.
        """
        suggestion = {
            "test_case_id": test_result.test_case_id,
            "root_cause": analysis.root_cause,
            "suggested_fix": analysis.suggested_fix,
            "config_params": self._suggest_config_changes(analysis),
        }

        return suggestion

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple extraction: remove stop words, keep meaningful terms
        stop_words = {
            "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로",
        }

        # Split by whitespace and filter
        words = query.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        return keywords[:5]  # Return top 5 keywords

    def _sanitize_string(self, s: str) -> str:
        """Sanitize string for use as identifier."""
        # Remove special characters, replace with underscore
        import re

        sanitized = re.sub(r"[^\w\s]", "_", s)
        sanitized = re.sub(r"\s+", "_", sanitized)
        return sanitized[:30]  # Limit length

    def _assess_priority(self, analysis: "FiveWhyAnalysis") -> str:
        """Assess priority of the fix."""
        # High priority if it affects core functionality
        high_priority_keywords = ["retrieval", "search", "accuracy", "fact_check"]

        if any(
            keyword in analysis.root_cause.lower() for keyword in high_priority_keywords
        ):
            return "high"

        return "medium"

    def _suggest_file_to_modify(self, analysis: "FiveWhyAnalysis") -> Optional[str]:
        """Suggest which file to modify for code changes."""
        root_cause_lower = analysis.root_cause.lower()

        if "prompt" in root_cause_lower:
            return "src/rag/prompts/generation_prompt.txt"
        elif "retrieval" in root_cause_lower or "search" in root_cause_lower:
            return "src/rag/infrastructure/hybrid_search.py"
        elif "fact_check" in root_cause_lower:
            return "src/rag/infrastructure/fact_checker.py"
        else:
            return None

    def _generate_code_hint(self, analysis: "FiveWhyAnalysis") -> str:
        """Generate a hint for code modification."""
        root_cause_lower = analysis.root_cause.lower()

        if "prompt" in root_cause_lower:
            return "Update LLM prompt to enforce source-based generation and reduce hallucination"
        elif "parameters" in root_cause_lower:
            return "Adjust component hyperparameters (e.g., top_k, threshold, temperature)"
        elif "retrieval" in root_cause_lower:
            return "Review retrieval strategy and consider improving embeddings or indexing"
        else:
            return "Review and update the relevant component logic"

    def _suggest_config_changes(self, analysis: "FiveWhyAnalysis") -> Dict:
        """Suggest configuration parameter changes."""
        # Return suggested config changes based on root cause
        return {
            "suggested_changes": [
                {
                    "parameter": "top_k",
                    "current_value": 5,
                    "suggested_value": 7,
                    "reason": "Increase retrieved documents for better coverage",
                }
            ]
        }

    def preview_improvements(
        self,
        analyses: List["FiveWhyAnalysis"],
        test_results: List["TestResult"],
    ) -> str:
        """
        Generate a human-readable preview of improvements.

        Args:
            analyses: List of 5-Why analysis results.
            test_results: Corresponding test results.

        Returns:
            Markdown formatted preview string.
        """
        lines = []
        lines.append("# Improvement Preview\n")
        lines.append(f"Total analyses: {len(analyses)}\n")

        # Handle potentially different length lists (Python 3.9 compatibility)
        min_len = min(len(analyses), len(test_results))
        for i in range(min_len):
            analysis = analyses[i]
            test_result = test_results[i]
            lines.append(f"## Test: {test_result.test_case_id}")
            lines.append(f"**Query:** {test_result.query[:100]}...")
            lines.append(f"**Root Cause:** {analysis.root_cause}")
            lines.append(f"**Suggested Fix:** {analysis.suggested_fix}")

            if analysis.component_to_patch:
                lines.append(f"**Target:** {analysis.component_to_patch}")
                lines.append(f"**Code Change Required:** {analysis.code_change_required}")

            lines.append("")

        return "\n".join(lines)
