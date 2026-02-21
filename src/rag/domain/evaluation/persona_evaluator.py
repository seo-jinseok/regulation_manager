"""
Persona Evaluator for RAG Quality Evaluation.

Evaluates responses based on persona-specific requirements including
language level, citation preference, and key requirements.

SPEC-RAG-QUALITY-010 Milestone 6: Persona Evaluation System.

Clean Architecture: Domain layer contains evaluation logic and business rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import re

from .persona_definition import PersonaDefinition, DEFAULT_PERSONAS


@dataclass
class PersonaEvaluationScore:
    """
    Scores for persona-specific evaluation.

    Each score ranges from 0.0 to 1.0.

    Attributes:
        relevancy: How relevant the response is to the query
        clarity: How clear and appropriate the language is for the persona
        completeness: How complete the information is
        citation_quality: Quality of citations (appropriate for persona)
        overall: Weighted average of all scores
    """

    relevancy: float
    clarity: float
    completeness: float
    citation_quality: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "relevancy": round(self.relevancy, 3),
            "clarity": round(self.clarity, 3),
            "completeness": round(self.completeness, 3),
            "citation_quality": round(self.citation_quality, 3),
            "overall": round(self.overall, 3),
        }


@dataclass
class PersonaEvaluationResult:
    """
    Result of persona-specific evaluation.

    Contains scores, identified issues, and recommendations.
    """

    persona_id: str
    query: str
    response: str
    scores: PersonaEvaluationScore
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "persona_id": self.persona_id,
            "query": self.query,
            "response": self.response,
            "scores": self.scores.to_dict(),
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class PersonaDashboardData:
    """
    Dashboard data structure for persona evaluation results.

    Aggregates results across all personas with weak persona identification.
    """

    evaluation_date: str
    personas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weak_personas: List[str] = field(default_factory=list)
    recommendations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_date": self.evaluation_date,
            "personas": self.personas,
            "weak_personas": self.weak_personas,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_results(
        cls,
        results: List[PersonaEvaluationResult],
        evaluation_date: str,
        threshold: float = 0.65,
    ) -> "PersonaDashboardData":
        """
        Generate dashboard data from evaluation results.

        Args:
            results: List of PersonaEvaluationResult objects
            evaluation_date: Date of evaluation
            threshold: Threshold for weak persona identification

        Returns:
            PersonaDashboardData with aggregated metrics
        """
        # Group results by persona
        persona_results: Dict[str, List[PersonaEvaluationResult]] = defaultdict(list)
        for result in results:
            persona_results[result.persona_id].append(result)

        # Calculate aggregates for each persona
        personas_data = {}
        weak_personas = []
        recommendations = {}

        for persona_id, persona_result_list in persona_results.items():
            if not persona_result_list:
                continue

            # Calculate averages
            avg_relevancy = sum(r.scores.relevancy for r in persona_result_list) / len(persona_result_list)
            avg_clarity = sum(r.scores.clarity for r in persona_result_list) / len(persona_result_list)
            avg_completeness = sum(r.scores.completeness for r in persona_result_list) / len(persona_result_list)
            avg_citation = sum(r.scores.citation_quality for r in persona_result_list) / len(persona_result_list)
            avg_overall = sum(r.scores.overall for r in persona_result_list) / len(persona_result_list)

            # Count issues
            total_issues = sum(len(r.issues) for r in persona_result_list)

            personas_data[persona_id] = {
                "avg_overall": round(avg_overall, 3),
                "avg_relevancy": round(avg_relevancy, 3),
                "avg_clarity": round(avg_clarity, 3),
                "avg_completeness": round(avg_completeness, 3),
                "avg_citation": round(avg_citation, 3),
                "issue_count": total_issues,
            }

            # Identify weak personas
            if avg_overall < threshold:
                weak_personas.append(persona_id)

                # Collect all issues for recommendation
                all_issues = []
                for r in persona_result_list:
                    all_issues.extend(r.issues)

                # Generate recommendation based on common issues
                if all_issues:
                    recommendations[persona_id] = cls._generate_recommendation(persona_id, all_issues)

        return cls(
            evaluation_date=evaluation_date,
            personas=personas_data,
            weak_personas=weak_personas,
            recommendations=recommendations,
        )

    @staticmethod
    def _generate_recommendation(persona_id: str, issues: List[str]) -> str:
        """Generate improvement recommendation based on issues."""
        # Count issue occurrences
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue] += 1

        # Get most common issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        top_issues = sorted_issues[:3]

        # Generate recommendations based on persona and issues
        if persona_id == "international":
            if any("한국어" in issue for issue, _ in top_issues):
                return "간단한 한국어 사용 및 복잡한 용어 설명 강화 필요"
            return "언어 수준 조정 및 시각적 안내 추가 권장"

        elif persona_id == "freshman":
            if any("복잡" in issue for issue, _ in top_issues):
                return "간단명료한 표현으로 수정 및 친절한 설명 추가 권장"
            return "친절하고 쉬운 설명 방식 개선 필요"

        elif persona_id == "professor":
            if any("인용" in issue or "조항" in issue for issue, _ in top_issues):
                return "규정 조항 인용 강화 및 전문 용어 정확성 향상 필요"
            return "학술적 엄밀성 및 인용 체계 보완 권장"

        elif persona_id == "parent":
            if any("연락처" in issue for issue, _ in top_issues):
                return "연락처 정보 포함 및 부모 친화적 설명 추가 필요"
            return "이해하기 쉬운 용어 사용 및 친절한 안내 개선"

        elif persona_id == "staff":
            if any("기한" in issue or "부서" in issue for issue, _ in top_issues):
                return "처리 기한 및 담당 부서 정보 명시 강화 필요"
            return "행정 절차 안내 체계화 및 실무 정보 보완"

        else:
            # Default recommendation
            top_issue = top_issues[0][0] if top_issues else "일반적인 품질 개선"
            return f"주요 개선 사항: {top_issue}"


class PersonaEvaluator:
    """
    Evaluator for persona-specific response quality.

    Evaluates responses based on persona requirements including:
    - Language level appropriateness
    - Citation style preferences
    - Key requirement satisfaction
    """

    # Patterns for language analysis
    FORMAL_PATTERNS = [
        r"~합니다",
        r"~입니다",
        r"~하십시오",
        r"~바랍니다",
        r"따라서",
        r"또한",
        r"그러나",
    ]

    CASUAL_PATTERNS = [
        r"~해요",
        r"~이에요",
        r"~세요",
        r"~돼요",
        r"~어요",
    ]

    TECHNICAL_PATTERNS = [
        r"제\d+조",
        r"제\d+항",
        r"「[^」]+」",
        r"규정",
        r"조항",
        r"적용",
    ]

    def __init__(self):
        """Initialize the PersonaEvaluator."""
        self.personas = DEFAULT_PERSONAS

    def evaluate_persona(
        self,
        query: str,
        response: str,
        persona: PersonaDefinition,
    ) -> PersonaEvaluationResult:
        """
        Evaluate a response for a specific persona.

        Args:
            query: The user query
            response: The generated response
            persona: The target persona definition

        Returns:
            PersonaEvaluationResult with scores and recommendations
        """
        # Calculate individual scores
        relevancy = self._evaluate_relevancy(query, response)
        clarity = self._evaluate_clarity(response, persona)
        completeness = self._evaluate_completeness(query, response)
        citation_quality = self._evaluate_citation(response, persona)

        # Calculate overall score (weighted average)
        overall = (
            relevancy * 0.25
            + clarity * 0.30
            + completeness * 0.25
            + citation_quality * 0.20
        )

        scores = PersonaEvaluationScore(
            relevancy=relevancy,
            clarity=clarity,
            completeness=completeness,
            citation_quality=citation_quality,
            overall=overall,
        )

        # Identify issues
        issues = self._identify_issues(query, response, persona, scores)

        # Generate recommendations
        recommendations = self._generate_response_recommendations(issues, persona)

        return PersonaEvaluationResult(
            persona_id=persona.persona_id,
            query=query,
            response=response,
            scores=scores,
            issues=issues,
            recommendations=recommendations,
        )

    def evaluate_all_personas(
        self,
        queries: List[str],
        responses: List[str],
        personas: Optional[List[PersonaDefinition]] = None,
    ) -> Dict[str, List[PersonaEvaluationResult]]:
        """
        Evaluate responses for all personas.

        Args:
            queries: List of queries
            responses: List of responses
            personas: Optional list of personas (defaults to all)

        Returns:
            Dictionary mapping persona_id to list of results
        """
        if personas is None:
            personas = list(self.personas.values())

        results: Dict[str, List[PersonaEvaluationResult]] = defaultdict(list)

        for i, (query, response) in enumerate(zip(queries, responses)):
            # Cycle through personas if fewer personas than queries
            persona = personas[i % len(personas)]
            result = self.evaluate_persona(query, response, persona)
            results[persona.persona_id].append(result)

        return dict(results)

    def get_weak_personas(
        self,
        results: List[PersonaEvaluationResult],
        threshold: float = 0.65,
    ) -> List[str]:
        """
        Identify personas with overall score below threshold.

        Args:
            results: List of evaluation results
            threshold: Score threshold (default: 0.65)

        Returns:
            List of weak persona IDs
        """
        if not results:
            return []

        # Calculate average score per persona
        persona_scores: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            persona_scores[result.persona_id].append(result.scores.overall)

        weak_personas = []
        for persona_id, scores in persona_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < threshold:
                weak_personas.append(persona_id)

        return weak_personas

    def generate_recommendations(
        self,
        results: List[PersonaEvaluationResult],
    ) -> Dict[str, str]:
        """
        Generate improvement recommendations for each persona.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary mapping persona_id to recommendation
        """
        recommendations = {}

        # Group results by persona
        persona_results: Dict[str, List[PersonaEvaluationResult]] = defaultdict(list)
        for result in results:
            persona_results[result.persona_id].append(result)

        for persona_id, persona_result_list in persona_results.items():
            # Collect all issues
            all_issues = []
            for r in persona_result_list:
                all_issues.extend(r.issues)

            if all_issues:
                recommendations[persona_id] = PersonaDashboardData._generate_recommendation(
                    persona_id, all_issues
                )

        return recommendations

    def identify_common_issues(
        self,
        results: List[PersonaEvaluationResult],
        persona_id: str,
    ) -> List[str]:
        """
        Identify common issues for a specific persona.

        Args:
            results: List of evaluation results
            persona_id: Target persona ID

        Returns:
            List of common issues (appearing more than once)
        """
        issue_counts = defaultdict(int)

        for result in results:
            if result.persona_id == persona_id:
                for issue in result.issues:
                    issue_counts[issue] += 1

        # Return issues that appear more than once
        return [issue for issue, count in issue_counts.items() if count > 1]

    # Private scoring methods

    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query."""
        if not query or not response:
            return 0.5

        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if not query_words:
            return 0.5

        # Calculate word overlap
        overlap = len(query_words & response_words)
        score = overlap / len(query_words)

        # Scale to 0.5-1.0 range
        return min(0.95, max(0.5, 0.5 + score * 0.45))

    def _evaluate_clarity(self, response: str, persona: PersonaDefinition) -> float:
        """Evaluate clarity based on persona language level requirements."""
        if not response:
            return 0.5

        language_level = persona.language_level

        if language_level == "simple":
            # Simple language: prefer casual patterns, avoid formal/technical
            formal_count = sum(len(re.findall(p, response)) for p in self.FORMAL_PATTERNS)
            technical_count = sum(len(re.findall(p, response)) for p in self.TECHNICAL_PATTERNS)
            casual_count = sum(len(re.findall(p, response)) for p in self.CASUAL_PATTERNS)

            # Penalize formal/technical language for simple personas
            penalty = min(0.3, (formal_count + technical_count * 2) * 0.05)
            bonus = min(0.2, casual_count * 0.03)

            score = 0.75 + bonus - penalty
            return max(0.4, min(0.95, score))

        elif language_level == "technical":
            # Technical language: expect formal patterns and citations
            formal_count = sum(len(re.findall(p, response)) for p in self.FORMAL_PATTERNS)
            technical_count = sum(len(re.findall(p, response)) for p in self.TECHNICAL_PATTERNS)

            # Reward technical/formal language
            bonus = min(0.25, (formal_count + technical_count) * 0.03)

            score = 0.70 + bonus
            return max(0.5, min(0.95, score))

        elif language_level == "formal":
            # Formal language: expect formal patterns
            formal_count = sum(len(re.findall(p, response)) for p in self.FORMAL_PATTERNS)
            bonus = min(0.2, formal_count * 0.03)

            score = 0.70 + bonus
            return max(0.5, min(0.95, score))

        else:  # normal
            # Balanced language
            return 0.75

    def _evaluate_completeness(self, query: str, response: str) -> float:
        """Evaluate completeness of the response."""
        if not query or not response:
            return 0.5

        # Check for completeness indicators
        completeness_score = 0.7  # Base score

        # Check for procedure-related content
        if any(kw in query for kw in ["어떻게", "방법", "절차", "절차"]):
            if any(marker in response for marker in ["1.", "첫째", "우선", "단계"]):
                completeness_score += 0.1

        # Check for deadline-related content
        if any(kw in query for kw in ["언제", "기한", "기간", "까지"]):
            if any(marker in response for marker in ["까지", "기간", "일", "주"]):
                completeness_score += 0.1

        # Check for eligibility-related content
        if any(kw in query for kw in ["자격", "요건", "조건", "누가"]):
            if any(marker in response for marker in ["자격", "요건", "조건", "가능"]):
                completeness_score += 0.1

        # Check for contact-related content
        if any(kw in query for kw in ["연락", "문의", "어디서"]):
            if any(marker in response for marker in ["연락", "문의", "부서", "전화"]):
                completeness_score += 0.1

        # Penalize very short responses
        if len(response) < 50:
            completeness_score -= 0.2
        elif len(response) < 100:
            completeness_score -= 0.1

        return max(0.4, min(0.95, completeness_score))

    def _evaluate_citation(self, response: str, persona: PersonaDefinition) -> float:
        """Evaluate citation quality based on persona preferences."""
        if not response:
            return 0.5

        citation_pref = persona.citation_preference

        # Count citations
        citations = re.findall(r"「[^」]+」", response)
        article_refs = re.findall(r"제\d+조", response)

        citation_count = len(citations) + len(article_refs)

        if citation_pref == "minimal":
            # Minimal citations: 0-2 is ideal
            if citation_count == 0:
                return 0.85
            elif citation_count <= 2:
                return 0.90
            else:
                return max(0.6, 0.95 - citation_count * 0.05)

        elif citation_pref == "detailed":
            # Detailed citations: 2+ is ideal
            if citation_count >= 3:
                return 0.95
            elif citation_count >= 1:
                return 0.80
            else:
                return 0.55

        else:  # normal
            # Normal citations: 1-2 is ideal
            if 1 <= citation_count <= 2:
                return 0.90
            elif citation_count == 0:
                return 0.70
            else:
                return 0.85

    def _identify_issues(
        self,
        query: str,
        response: str,
        persona: PersonaDefinition,
        scores: PersonaEvaluationScore,
    ) -> List[str]:
        """Identify issues in the response for the given persona."""
        issues = []

        # Check clarity issues for simple language personas
        if persona.language_level == "simple":
            formal_count = sum(len(re.findall(p, response)) for p in self.FORMAL_PATTERNS)
            technical_count = sum(len(re.findall(p, response)) for p in self.TECHNICAL_PATTERNS)

            # If clarity is low or formal language is excessive, flag as issue
            if scores.clarity < 0.7 or formal_count >= 2 or technical_count >= 2:
                if formal_count >= 2 or technical_count >= 2:
                    issues.append("복잡한 표현 사용")

        # Check clarity for technical language personas
        elif persona.language_level == "technical":
            technical_count = sum(len(re.findall(p, response)) for p in self.TECHNICAL_PATTERNS)
            if scores.clarity < 0.7 and technical_count < 2:
                issues.append("전문 용어/조항 인용 부족")

        # Check citation issues
        if scores.citation_quality < 0.7:
            if persona.citation_preference == "detailed":
                issues.append("조항 인용 부족")
            elif persona.citation_preference == "minimal":
                issues.append("과도한 인용")

        # Check completeness issues
        if scores.completeness < 0.7:
            issues.append("정보 누락")

        # Check persona-specific requirements
        if persona.persona_id == "international":
            # Check for complex Korean
            if any(len(word) > 10 for word in response.split()):
                issues.append("복잡한 한국어 사용")

        elif persona.persona_id == "parent":
            # Check for contact info
            if "연락" in query or "문의" in query:
                if "연락" not in response and "전화" not in response:
                    issues.append("연락처 정보 누락")

        return issues

    def _generate_response_recommendations(
        self,
        issues: List[str],
        persona: PersonaDefinition,
    ) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []

        for issue in issues:
            if "복잡한" in issue:
                recommendations.append(f"간단명료한 표현으로 수정 권장 ({persona.name} 대상)")
            elif "인용 부족" in issue:
                recommendations.append("관련 조항 인용 추가 권장")
            elif "과도한 인용" in issue:
                recommendations.append("핵심 정보 위주로 간소화 권장")
            elif "정보 누락" in issue:
                recommendations.append("누락된 정보 보완 필요")
            elif "연락처" in issue:
                recommendations.append("담당 부서 연락처 정보 추가 권장")
            elif "한국어" in issue:
                recommendations.append("간단한 한국어 표현으로 수정 권장")

        return recommendations
