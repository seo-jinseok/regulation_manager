"""
Auto Learning for RAG System Improvement.

Analyzes feedback data and suggests improvements
for intents, synonyms, and query understanding.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient
    from ..infrastructure.feedback import FeedbackCollector, FeedbackEntry


@dataclass
class ImprovementSuggestion:
    """
    A suggested improvement based on feedback analysis.
    
    Types:
        - intent: intents.json에 트리거 추가
        - synonym: synonyms.json에 동의어 추가
        - rerank: Reranker 검토 필요
        - llm_expert: LLM 기반 제안
        - code_pattern: QueryAnalyzer 패턴 로직 개선 필요
        - code_weight: 가중치 프리셋 조정 필요
        - code_audience: 대상 감지 로직 개선 필요
        - architecture: 시스템 구조적 개선 필요
    """

    type: str
    priority: str  # "high", "medium", "low"
    description: str
    suggested_value: Dict[str, Any] = field(default_factory=dict)
    affected_queries: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of feedback analysis."""

    total_negative_feedback: int
    unique_problematic_queries: int
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)


class AutoLearnUseCase:
    """
    Use case for automated learning from feedback.

    Analyzes negative feedback patterns and suggests
    improvements to intents, synonyms, and prompts.
    """

    def __init__(
        self,
        feedback_collector: Optional["FeedbackCollector"] = None,
        llm_client: Optional["ILLMClient"] = None,
        intents_path: Optional[str] = None,
        synonyms_path: Optional[str] = None,
    ):
        """
        Initialize auto learning use case.

        Args:
            feedback_collector: FeedbackCollector instance.
            llm_client: Optional LLM client for generating suggestions.
            intents_path: Path to intents.json.
            synonyms_path: Path to synonyms.json.
        """
        self._feedback = feedback_collector
        self._llm_client = llm_client
        self._intents_path = intents_path or self._default_intents_path()
        self._synonyms_path = synonyms_path or self._default_synonyms_path()

    def _default_intents_path(self) -> str:
        from ..config import get_config
        return str(get_config().intents_path_resolved or "data/config/intents.json")

    def _default_synonyms_path(self) -> str:
        from ..config import get_config
        return str(get_config().synonyms_path_resolved or "data/config/synonyms.json")

    def analyze_feedback(self) -> AnalysisResult:
        """
        Analyze negative feedback and generate improvement suggestions.

        Returns:
            AnalysisResult with suggestions.
        """
        if self._feedback is None:
            return AnalysisResult(
                total_negative_feedback=0,
                unique_problematic_queries=0,
            )

        negative = self._feedback.get_negative_feedback()
        if not negative:
            return AnalysisResult(
                total_negative_feedback=0,
                unique_problematic_queries=0,
            )

        # Group by query
        query_groups: Dict[str, List["FeedbackEntry"]] = {}
        for entry in negative:
            if entry.query not in query_groups:
                query_groups[entry.query] = []
            query_groups[entry.query].append(entry)

        suggestions: List[ImprovementSuggestion] = []

        # Analyze patterns
        for query, entries in query_groups.items():
            if len(entries) >= 2:  # Multiple negative feedback
                suggestion = self._analyze_query_pattern(query, entries)
                if suggestion:
                    suggestions.append(suggestion)

        # Check for missing intents
        intent_suggestions = self._check_missing_intents(query_groups)
        suggestions.extend(intent_suggestions)

        # Check for missing synonyms
        synonym_suggestions = self._check_missing_synonyms(query_groups)
        suggestions.extend(synonym_suggestions)

        # Check for code-level improvements (patterns in failures)
        code_suggestions = self._check_code_improvements(query_groups)
        suggestions.extend(code_suggestions)

        # LLM-based suggestions (if available)
        if self._llm_client and query_groups:
            # Sort queries by negative count
            top_problematic = sorted(
                query_groups.items(), key=lambda x: len(x[1]), reverse=True
            )[:5]

            for query, _ in top_problematic:
                llm_sug = self.suggest_with_llm(query)
                if llm_sug:
                    suggestions.append(llm_sug)

        return AnalysisResult(
            total_negative_feedback=len(negative),
            unique_problematic_queries=len(query_groups),
            suggestions=suggestions,
        )

    def _analyze_query_pattern(
        self,
        query: str,
        entries: List["FeedbackEntry"],
    ) -> Optional[ImprovementSuggestion]:
        """Analyze a specific query with multiple negative feedback."""
        # Check if intents were matched
        all_intents = set()
        for e in entries:
            all_intents.update(e.matched_intents or [])

        if not all_intents:
            return ImprovementSuggestion(
                type="intent",
                priority="high",
                description=f"쿼리 '{query}'에 매칭되는 의도가 없음",
                suggested_value={
                    "action": "add_intent",
                    "query": query,
                },
                affected_queries=[query],
            )

        # Check if wrong results were returned
        wrong_codes = [e.rule_code for e in entries]
        return ImprovementSuggestion(
            type="rerank",
            priority="medium",
            description=f"쿼리 '{query}'에 잘못된 규정({wrong_codes[:3]})이 반환됨",
            suggested_value={
                "action": "review_search",
                "query": query,
                "wrong_rule_codes": wrong_codes,
            },
            affected_queries=[query],
        )

    def _check_missing_intents(
        self,
        query_groups: Dict[str, List["FeedbackEntry"]],
    ) -> List[ImprovementSuggestion]:
        """Check for queries that might need new intents."""
        suggestions = []

        # Load existing intents
        intents_path = Path(self._intents_path)
        if not intents_path.exists():
            return suggestions

        try:
            data = json.loads(intents_path.read_text(encoding="utf-8"))
            existing_triggers = set()
            for intent in data.get("intents", []):
                existing_triggers.update(intent.get("triggers", []))
        except Exception:
            return suggestions

        # Find queries not covered by existing triggers
        for query in query_groups.keys():
            query_lower = query.lower()
            covered = any(
                trigger.lower() in query_lower for trigger in existing_triggers
            )
            if not covered:
                suggestions.append(
                    ImprovementSuggestion(
                        type="intent",
                        priority="high",
                        description=f"새 인텐트 트리거 추가 권장: '{query}'",
                        suggested_value={
                            "action": "add_trigger",
                            "query": query,
                        },
                        affected_queries=[query],
                    )
                )

        return suggestions[:5]  # Limit suggestions

    def _check_missing_synonyms(
        self,
        query_groups: Dict[str, List["FeedbackEntry"]],
    ) -> List[ImprovementSuggestion]:
        """Check for terms that might need synonyms."""
        suggestions = []

        # Load existing synonyms
        synonyms_path = Path(self._synonyms_path)
        if not synonyms_path.exists():
            return suggestions

        try:
            data = json.loads(synonyms_path.read_text(encoding="utf-8"))
            terms = data.get("terms", data) if isinstance(data, dict) else {}
            existing_terms = set(terms.keys())
        except Exception:
            return suggestions

        # Extract common words from queries
        word_counts: Counter = Counter()
        for query in query_groups.keys():
            words = query.split()
            for word in words:
                if len(word) >= 2:
                    word_counts[word] += 1

        # Find frequent words not in synonyms
        for word, count in word_counts.most_common(10):
            if word not in existing_terms and count >= 2:
                suggestions.append(
                    ImprovementSuggestion(
                        type="synonym",
                        priority="medium",
                        description=f"동의어 추가 권장: '{word}'",
                        suggested_value={
                            "action": "add_synonym",
                            "term": word,
                            "count": count,
                        },
                        affected_queries=[q for q in query_groups.keys() if word in q],
                    )
                )

        return suggestions[:5]

    def _check_code_improvements(
        self,
        query_groups: Dict[str, List["FeedbackEntry"]],
    ) -> List[ImprovementSuggestion]:
        """
        Analyze failure patterns to suggest code-level improvements.
        
        Detects patterns that indicate need for:
        - QueryAnalyzer pattern logic changes
        - Weight preset adjustments
        - Audience detection improvements
        - Architectural changes
        """
        suggestions = []
        
        # Pattern 1: Intent triggers match but wrong results
        # → Indicates weight preset or reranker issues
        intent_matched_failures = [
            (q, entries) for q, entries in query_groups.items()
            if any(e.matched_intents for e in entries)
        ]
        if len(intent_matched_failures) >= 3:
            suggestions.append(
                ImprovementSuggestion(
                    type="code_weight",
                    priority="high",
                    description=(
                        f"인텐트 매칭됨에도 {len(intent_matched_failures)}개 쿼리에서 "
                        "잘못된 결과 반환 → WEIGHT_PRESETS 또는 Reranker 가중치 조정 필요"
                    ),
                    suggested_value={
                        "action": "adjust_weight_presets",
                        "file": "src/rag/infrastructure/query_analyzer.py",
                        "target": "WEIGHT_PRESETS",
                        "note": "Intent 쿼리의 BM25/Dense 비율 검토",
                    },
                    affected_queries=[q for q, _ in intent_matched_failures[:5]],
                )
            )
        
        # Pattern 2: Similar query patterns failing repeatedly
        # → Indicates missing pattern in QueryAnalyzer
        pattern_keywords = ["싶어", "하고", "어떻게", "뭐야", "알려줘"]
        keyword_failures: Dict[str, List[str]] = {}
        for query in query_groups.keys():
            for kw in pattern_keywords:
                if kw in query:
                    keyword_failures.setdefault(kw, []).append(query)
        
        for kw, queries in keyword_failures.items():
            if len(queries) >= 3:
                suggestions.append(
                    ImprovementSuggestion(
                        type="code_pattern",
                        priority="high",
                        description=(
                            f"'{kw}' 패턴 포함 쿼리 {len(queries)}개 반복 실패 → "
                            "QueryAnalyzer.INTENT_PATTERNS에 새 패턴 추가 필요"
                        ),
                        suggested_value={
                            "action": "add_intent_pattern",
                            "file": "src/rag/infrastructure/query_analyzer.py",
                            "pattern_keyword": kw,
                            "sample_queries": queries[:3],
                        },
                        affected_queries=queries[:5],
                    )
                )
        
        # Pattern 3: Audience-related failures
        # → Indicates audience detection logic issues
        audience_keywords = ["학생", "교수", "교원", "직원", "교직원"]
        audience_failures = []
        for query in query_groups.keys():
            for kw in audience_keywords:
                if kw in query:
                    audience_failures.append(query)
                    break
        
        if len(audience_failures) >= 2:
            suggestions.append(
                ImprovementSuggestion(
                    type="code_audience",
                    priority="medium",
                    description=(
                        f"대상 키워드 포함 쿼리 {len(audience_failures)}개 실패 → "
                        "detect_audience() 또는 AUDIENCE_KEYWORDS 확장 필요"
                    ),
                    suggested_value={
                        "action": "extend_audience_detection",
                        "file": "src/rag/infrastructure/query_analyzer.py",
                        "target": "FACULTY_KEYWORDS, STUDENT_KEYWORDS, STAFF_KEYWORDS",
                    },
                    affected_queries=audience_failures[:5],
                )
            )
        
        # Pattern 4: High failure rate overall
        # → Indicates potential architectural issues
        if len(query_groups) >= 10:
            suggestions.append(
                ImprovementSuggestion(
                    type="architecture",
                    priority="medium",
                    description=(
                        f"전체 {len(query_groups)}개 문제 쿼리 발생 → "
                        "검색 파이프라인 전반 검토 필요 (HybridSearcher, Reranker)"
                    ),
                    suggested_value={
                        "action": "review_pipeline",
                        "components": [
                            "HybridSearcher (BM25 + Dense 융합)",
                            "BGEReranker (Cross-Encoder 재정렬)",
                            "QueryAnalyzer (쿼리 분석/확장)",
                        ],
                    },
                    affected_queries=list(query_groups.keys())[:10],
                )
            )
        
        return suggestions[:5]

    def suggest_with_llm(self, query: str) -> Optional[ImprovementSuggestion]:
        """
        Use LLM to suggest improvements for a specific query.

        Args:
            query: The problematic query.

        Returns:
            ImprovementSuggestion or None.
        """
        if not self._llm_client:
            return None

        prompt = f"""다음 검색 쿼리에 대해 사용자가 부정적인 피드백을 남겼습니다.
검색 품질을 높이기 위해 어떤 인텐트(Intent), 동의어(Synonym), 또는 리라이팅 규칙이 필요할까요?

쿼리: "{query}"

## 제안 형식 (JSON)
{{
  "intent_trigger": "추가할 트리거 단어 (없으면 null)",
  "synonyms": ["단어1", "단어2"],
  "reason": "제안 이유"
}}"""

        try:
            response = self._llm_client.generate(
                system_prompt="당신은 대학 규정 검색 시스템 개선 전문가입니다.",
                user_message=prompt,
                temperature=0.3,
            )

            # Parse JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return ImprovementSuggestion(
                    type="llm_raw",
                    priority="high",
                    description=f"LLM 분석: {query}",
                    suggested_value={"llm_response": response},
                    affected_queries=[query],
                )

            data = json.loads(json_match.group())
            desc = f"LLM 제안 - {data.get('reason', '검색 품질 개선')}"

            return ImprovementSuggestion(
                type="llm_expert",
                priority="high",
                description=desc,
                suggested_value=data,
                affected_queries=[query],
            )
        except Exception:
            return None

    def format_suggestions(self, result: AnalysisResult) -> str:
        """Format analysis result as readable string."""
        lines = [
            "=" * 60,
            "자동 학습 분석 결과",
            "=" * 60,
            f"부정 피드백 수: {result.total_negative_feedback}",
            f"문제 쿼리 수: {result.unique_problematic_queries}",
            f"개선 제안 수: {len(result.suggestions)}",
            "-" * 60,
        ]

        if not result.suggestions:
            lines.append("\n✅ 현재 분석된 개선 사항이 없습니다.")
        else:
            for i, s in enumerate(result.suggestions, 1):
                priority_icon = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(s.priority, "⚪")
                lines.append(f"\n{priority_icon} [{i}] {s.type.upper()}")
                lines.append(f"   {s.description}")
                if s.affected_queries:
                    lines.append(f"   영향 쿼리: {s.affected_queries[:3]}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
