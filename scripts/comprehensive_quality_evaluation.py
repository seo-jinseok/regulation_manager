"""
Comprehensive Quality Evaluation for RAG System.

Evaluates the system using:
1. All 69 test cases from evaluation_dataset.json
2. Diverse persona simulation (Professor, Student, Staff, Administrator)
3. Query type analysis (Factual, Complex, Ambiguous, Multi-turn, Edge cases)
4. 6-dimension quality framework (Accuracy, Completeness, Relevance, Source Citation, Practicality, Actionability)

Output: Comprehensive quality report with actionable recommendations
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.infrastructure.reranker import BGEReranker
from src.rag.interface.query_handler import QueryContext, QueryHandler, QueryOptions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestCaseEvaluation:
    """Single test case evaluation result"""

    test_id: str
    query: str
    category: str

    # Quality dimensions (6-dimension framework)
    accuracy_score: float  # 1.0
    completeness_score: float  # 1.0
    relevance_score: float  # 1.0
    source_citation_score: float  # 1.0
    practicality_score: float  # 0.5
    actionability_score: float  # 0.5

    # Total quality score
    total_score: float  # 5.0 max

    # Additional metrics
    confidence: float
    execution_time_ms: int
    sources_count: int
    answer_length: int

    # Analysis
    intent_recognition: str
    quality_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # Persona context
    persona_type: Optional[str] = None
    query_type: Optional[str] = None

    # Pass/Fail (threshold: 4.0/5.0)
    passed: bool = False

    # Actual response
    answer: str = ""
    sources: List[str] = field(default_factory=list)


@dataclass
class CategoryAnalysis:
    """Analysis for a test category"""

    category_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_score: float
    failing_cases: List[str] = field(default_factory=list)


@dataclass
class PersonaAnalysis:
    """Analysis for a user persona"""

    persona_type: str
    total_tests: int
    avg_score: float
    avg_confidence: float
    common_issues: List[str] = field(default_factory=list)


@dataclass
class QueryTypeAnalysis:
    """Analysis for query types"""

    query_type: str
    total_tests: int
    pass_rate: float
    avg_score: float
    avg_execution_time: float


class ComprehensiveQualityEvaluator:
    """Comprehensive quality evaluator for RAG system"""

    # Query type classification
    QUERY_TYPES = {
        "factual": ["졸업 요건이 뭐야?", "승진 요건이 뭐야?", "휴학"],
        "complex_multi_step": [
            "연구년 신청하려면 승인 절차가 어떻게 되고",
            "복수전공하는데",
        ],
        "ambiguous": ["그거 신청서 어디서 받아?", "공부하기 싫어", "규정 바뀌었어?"],
        "multi_turn": ["졸업 요건이 뭐야?", "그럼 논문은 필수야?"],
        "edge_case": ["장학금 밫고 시퍼", "학교 그만두고 싶음"],
        "comparative": ["휴학과 자퇴 차이가 뭐야?", "정교수와 부교수 승진 요건 차이"],
        "conditional": ["학사경고 2번 받았을 때 휴학 가능해?"],
        "procedural": ["휴학 신청하려면?", "성적증명서 발급받으려면?"],
    }

    def __init__(self, db_path: str = "data/chroma_db"):
        """Initialize evaluator"""
        self.db_path = db_path
        self.query_handler: Optional[QueryHandler] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self._initialize_system()

    def _initialize_system(self):
        """Initialize RAG system"""
        logger.info("Initializing RAG system for comprehensive evaluation...")

        try:
            # ChromaDB
            store = ChromaVectorStore(persist_directory=self.db_path)
            logger.info(f"ChromaDB documents: {store.count()}")

            # LLM client
            llm_client = LLMClientAdapter(
                provider=os.getenv("LLM_PROVIDER", "openrouter"),
                model=os.getenv("LLM_MODEL", "z-ai/glm-4.7-flash"),
                base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            )

            # Reranker
            try:
                reranker = BGEReranker()
                use_reranker = True
                logger.info("BGE Reranker enabled")
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}")
                reranker = None
                use_reranker = False

            # Query Analyzer
            self.query_analyzer = QueryAnalyzer(llm_client=llm_client)

            # Query Handler
            self.query_handler = QueryHandler(
                store=store, llm_client=llm_client, use_reranker=use_reranker
            )

            logger.info("RAG system initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()

        for query_type, patterns in self.QUERY_TYPES.items():
            if any(pattern.lower() in query_lower for pattern in patterns):
                return query_type

        return "general"

    def determine_persona(self, query: str, category: str) -> str:
        """Determine likely user persona from query and category"""
        # Student-related keywords
        if any(
            kw in query
            for kw in [
                "졸업",
                "학점",
                "수강",
                "휴학",
                "장학금",
                "등록금",
                "성적",
                "전과",
                "복수전공",
            ]
        ):
            return "student"

        # Professor-related keywords
        if any(
            kw in query
            for kw in ["승진", "연구년", "안식년", "학술대회", "업적", "교원"]
        ):
            return "professor"

        # Staff-related keywords
        if any(kw in query for kw in ["퇴직", "사직", "연차", "복지", "호봉", "직원"]):
            return "staff"

        # Administrator-related keywords
        if any(kw in query for kw in ["징계", "인사", "시설", "예산", "규정"]):
            return "administrator"

        return "general"

    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        sources: List[str],
        execution_time_ms: int,
        confidence: float,
    ) -> TestCaseEvaluation:
        """Evaluate answer quality across 6 dimensions"""

        # 1. Accuracy (0-1.0): Based on citations and specificity
        accuracy = 0.0
        accuracy_reasons = []

        if len(answer) > 50:
            accuracy += 0.3
            accuracy_reasons.append("Adequate length")
        else:
            accuracy_reasons.append("Answer too short")

        # Check for regulation citations
        citation_patterns = ["제", "조", "규정", "학칙", "조항"]
        has_citation = any(pattern in answer for pattern in citation_patterns)
        if has_citation:
            accuracy += 0.4
            accuracy_reasons.append("Has regulation citations")
        else:
            accuracy_reasons.append("Missing regulation citations")

        # Check for specific information
        specific_indicators = [
            "기간",
            "신청",
            "방법",
            "조건",
            "제출",
            "서류",
            "담당",
            "연락",
        ]
        specific_count = sum(1 for ind in specific_indicators if ind in answer)
        accuracy += min(0.3, specific_count * 0.05)

        # 2. Completeness (0-1.0): Coverage of question aspects
        completeness = 0.0
        completeness_reasons = []

        # Check if answer addresses the core question
        if "?" in query or "어떻게" in query or "방법" in query:
            if "신청" in answer or "방법" in answer or "절차" in answer:
                completeness += 0.5
                completeness_reasons.append("Addresses how-to aspect")
            else:
                completeness_reasons.append("Missing procedural information")

        # Check for deadline information
        if "기간" in query or "언제" in query or "什么时候" in query:
            if any(kw in answer for kw in ["기간", "마감", "까지", "내"]):
                completeness += 0.3
                completeness_reasons.append("Includes deadline info")
            else:
                completeness_reasons.append("Missing deadline information")

        # Check for condition information
        if "조건" in query or "자격" in query or "requirement" in query.lower():
            if any(kw in answer for kw in ["조건", "자격", "해야", "필요"]):
                completeness += 0.2
                completeness_reasons.append("Includes condition info")
            else:
                completeness_reasons.append("Missing condition information")

        completeness = min(1.0, completeness + 0.2)  # Base score

        # 3. Relevance (0-1.0): Alignment with user intent
        relevance = 0.5  # Base score
        relevance_reasons = []

        # Check for generic responses
        generic_phrases = [
            "대학마다 다를 수 있습니다",
            "알 수 없습니다",
            "확인이 필요합니다",
        ]
        if any(phrase in answer for phrase in generic_phrases):
            relevance -= 0.3
            relevance_reasons.append("Generic response detected")
        else:
            relevance += 0.2
            relevance_reasons.append("Specific response")

        # Check if answer is on-topic
        query_words = set(query.split())
        answer_words = set(answer.split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        relevance += min(0.3, overlap)

        relevance = max(0.0, min(1.0, relevance))

        # 4. Source Citation (0-1.0): Proper regulation references
        source_citation = 0.0
        citation_reasons = []

        if has_citation:
            source_citation += 0.6
            citation_reasons.append("Has citations")

            # Check for specific citation format (e.g., "제15조")
            if any(char.isdigit() for char in answer) and "제" in answer:
                source_citation += 0.2
                citation_reasons.append("Has article number references")

            # Check for regulation names
            if any(reg in answer for reg in ["규정", "학칙", "시행세칙"]):
                source_citation += 0.2
                citation_reasons.append("Cites regulation names")
        else:
            citation_reasons.append("No citations")

        source_citation = min(1.0, source_citation)

        # 5. Practicality (0-0.5): Deadlines, requirements, contact info
        practicality = 0.0
        practicality_reasons = []

        practical_keywords = {
            "기간": 0.1,
            "마감": 0.1,
            "신청": 0.05,
            "방법": 0.05,
            "서류": 0.05,
            "제출": 0.05,
            "담당": 0.05,
            "연락": 0.05,
            "바로": 0.05,
            "즉시": 0.05,
        }

        for keyword, value in practical_keywords.items():
            if keyword in answer:
                practicality += value
                practicality_reasons.append(f"Has '{keyword}'")

        practicality = min(0.5, practicality)

        # 6. Actionability (0-0.5): Clear next steps
        actionability = 0.0
        actionability_reasons = []

        action_verbs = [
            "신청하세요",
            "제출하세요",
            "연락하세요",
            "확인하세요",
            "방문하세요",
            "준비하세요",
        ]
        if any(verb in answer for verb in action_verbs):
            actionability += 0.3
            actionability_reasons.append("Has action verbs")

        # Check for clear steps
        if "방법" in answer or "절차" in answer:
            actionability += 0.2
            actionability_reasons.append("Explains procedure")

        actionability = min(0.5, actionability)

        # Calculate total score
        total_score = (
            accuracy
            + completeness
            + relevance
            + source_citation
            + practicality
            + actionability
        )

        # Collect issues and strengths
        issues = []
        strengths = []

        if accuracy < 0.7:
            issues.append(
                f"Low accuracy ({accuracy:.2f}): {'; '.join(accuracy_reasons)}"
            )
        if completeness < 0.7:
            issues.append(
                f"Incomplete answer ({completeness:.2f}): {'; '.join(completeness_reasons)}"
            )
        if relevance < 0.7:
            issues.append(
                f"Low relevance ({relevance:.2f}): {'; '.join(relevance_reasons)}"
            )
        if source_citation < 0.7:
            issues.append(
                f"Poor citation ({source_citation:.2f}): {'; '.join(citation_reasons)}"
            )

        if accuracy >= 0.8:
            strengths.append(f"High accuracy ({accuracy:.2f})")
        if source_citation >= 0.8:
            strengths.append(f"Excellent citations ({source_citation:.2f})")
        if practicality >= 0.4:
            strengths.append(f"Practical information ({practicality:.2f})")
        if actionability >= 0.4:
            strengths.append(f"Actionable guidance ({actionability:.2f})")

        return TestCaseEvaluation(
            test_id="",
            query=query,
            category="",
            accuracy_score=accuracy,
            completeness_score=completeness,
            relevance_score=relevance,
            source_citation_score=source_citation,
            practicality_score=practicality,
            actionability_score=actionability,
            total_score=total_score,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            sources_count=len(sources),
            answer_length=len(answer),
            intent_recognition=f"Query type: {self.classify_query_type(query)}",
            quality_issues=issues,
            strengths=strengths,
            query_type=self.classify_query_type(query),
            answer=answer,
            sources=sources,
            passed=total_score >= 4.0,
        )

    def load_test_dataset(
        self, dataset_path: str = "data/config/evaluation_dataset.json"
    ) -> List[Dict]:
        """Load test dataset"""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data['test_cases'])} test cases from {dataset_path}")
        return data["test_cases"]

    def run_comprehensive_evaluation(
        self,
        dataset_path: str = "data/config/evaluation_dataset.json",
        output_dir: str = "test_results",
    ) -> Dict[str, Any]:
        """Run comprehensive quality evaluation"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE RAG QUALITY EVALUATION")
        logger.info("=" * 80)

        # Load test dataset
        test_cases = self.load_test_dataset(dataset_path)

        # Run evaluations
        results = []

        for idx, test_case in enumerate(test_cases, 1):
            test_id = test_case.get("id", f"test_{idx}")
            query = test_case.get("query", "")
            category = test_case.get("category", "general")

            logger.info(
                f"[{idx}/{len(test_cases)}] Evaluating: {test_id} - {query[:50]}..."
            )

            try:
                start_time = time.time()

                # Process query
                options = QueryOptions(top_k=5, use_rerank=True, show_debug=False)
                context = QueryContext()
                result = self.query_handler.process_query(
                    query=query, context=context, options=options
                )

                execution_time_ms = int((time.time() - start_time) * 1000)

                # Extract results
                answer = result.content
                sources = [r.get("name", "") for r in result.data.get("results", [])]
                confidence = result.data.get("confidence", 0.0)

                # Evaluate quality
                evaluation = self.evaluate_answer_quality(
                    query=query,
                    answer=answer,
                    sources=sources,
                    execution_time_ms=execution_time_ms,
                    confidence=confidence,
                )

                # Update metadata
                evaluation.test_id = test_id
                evaluation.category = category
                evaluation.persona_type = self.determine_persona(query, category)

                results.append(evaluation)

                status = "PASS" if evaluation.passed else "FAIL"
                logger.info(f"  {status} - Score: {evaluation.total_score:.2f}/5.0")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                # Create failed evaluation
                results.append(
                    TestCaseEvaluation(
                        test_id=test_id,
                        query=query,
                        category=category,
                        accuracy_score=0.0,
                        completeness_score=0.0,
                        relevance_score=0.0,
                        source_citation_score=0.0,
                        practicality_score=0.0,
                        actionability_score=0.0,
                        total_score=0.0,
                        confidence=0.0,
                        execution_time_ms=0,
                        sources_count=0,
                        answer_length=0,
                        intent_recognition="ERROR",
                        quality_issues=[f"System error: {str(e)}"],
                        passed=False,
                        answer=f"ERROR: {str(e)}",
                        persona_type=self.determine_persona(query, category),
                        query_type=self.classify_query_type(query),
                    )
                )

        # Generate analysis
        summary = self._generate_comprehensive_summary(results)

        # Save results
        self._save_comprehensive_results(summary, results, output_dir)

        # Generate report
        self._generate_comprehensive_report(summary, results, output_dir)

        return summary

    def _generate_comprehensive_summary(
        self, results: List[TestCaseEvaluation]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary"""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Dimension averages
        avg_accuracy = sum(r.accuracy_score for r in results) / total if total else 0
        avg_completeness = (
            sum(r.completeness_score for r in results) / total if total else 0
        )
        avg_relevance = sum(r.relevance_score for r in results) / total if total else 0
        avg_source_citation = (
            sum(r.source_citation_score for r in results) / total if total else 0
        )
        avg_practicality = (
            sum(r.practicality_score for r in results) / total if total else 0
        )
        avg_actionability = (
            sum(r.actionability_score for r in results) / total if total else 0
        )
        avg_total = sum(r.total_score for r in results) / total if total else 0

        # Category analysis
        by_category: Dict[str, List[TestCaseEvaluation]] = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)

        category_analyses = {}
        for category, cat_results in by_category.items():
            cat_total = len(cat_results)
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_avg = (
                sum(r.total_score for r in cat_results) / cat_total if cat_total else 0
            )
            cat_failed = [r.test_id for r in cat_results if not r.passed]

            category_analyses[category] = CategoryAnalysis(
                category_name=category,
                total_tests=cat_total,
                passed_tests=cat_passed,
                failed_tests=cat_total - cat_passed,
                pass_rate=cat_passed / cat_total if cat_total else 0,
                avg_score=cat_avg,
                failing_cases=cat_failed,
            )

        # Persona analysis
        by_persona: Dict[str, List[TestCaseEvaluation]] = {}
        for r in results:
            if r.persona_type not in by_persona:
                by_persona[r.persona_type] = []
            by_persona[r.persona_type].append(r)

        persona_analyses = {}
        for persona, persona_results in by_persona.items():
            p_total = len(persona_results)
            p_avg = (
                sum(r.total_score for r in persona_results) / p_total if p_total else 0
            )
            p_conf = (
                sum(r.confidence for r in persona_results) / p_total if p_total else 0
            )

            # Collect common issues
            all_issues = []
            for r in persona_results:
                all_issues.extend(r.quality_issues)
            from collections import Counter

            common_issues = [
                issue for issue, count in Counter(all_issues).most_common(5)
            ]

            persona_analyses[persona] = PersonaAnalysis(
                persona_type=persona,
                total_tests=p_total,
                avg_score=p_avg,
                avg_confidence=p_conf,
                common_issues=common_issues,
            )

        # Query type analysis
        by_type: Dict[str, List[TestCaseEvaluation]] = {}
        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = []
            by_type[r.query_type].append(r)

        query_type_analyses = {}
        for qtype, qtype_results in by_type.items():
            qt_total = len(qtype_results)
            qt_passed = sum(1 for r in qtype_results if r.passed)
            qt_avg = (
                sum(r.total_score for r in qtype_results) / qt_total if qt_total else 0
            )
            qt_time = (
                sum(r.execution_time_ms for r in qtype_results) / qt_total
                if qt_total
                else 0
            )

            query_type_analyses[qtype] = QueryTypeAnalysis(
                query_type=qtype,
                total_tests=qt_total,
                pass_rate=qt_passed / qt_total if qt_total else 0,
                avg_score=qt_avg,
                avg_execution_time=qt_time,
            )

        # Score distribution
        score_dist = {
            "excellent": sum(1 for r in results if r.total_score >= 4.5),
            "good": sum(1 for r in results if 4.0 <= r.total_score < 4.5),
            "acceptable": sum(1 for r in results if 3.5 <= r.total_score < 4.0),
            "poor": sum(1 for r in results if r.total_score < 3.5),
        }

        return {
            "evaluated_at": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total else 0,
            "dimension_averages": {
                "accuracy": avg_accuracy,
                "completeness": avg_completeness,
                "relevance": avg_relevance,
                "source_citation": avg_source_citation,
                "practicality": avg_practicality,
                "actionability": avg_actionability,
                "total": avg_total,
            },
            "category_analyses": {
                k: {
                    "category_name": v.category_name,
                    "total_tests": v.total_tests,
                    "passed_tests": v.passed_tests,
                    "failed_tests": v.failed_tests,
                    "pass_rate": v.pass_rate,
                    "avg_score": v.avg_score,
                    "failing_cases": v.failing_cases,
                }
                for k, v in category_analyses.items()
            },
            "persona_analyses": {
                k: {
                    "persona_type": v.persona_type,
                    "total_tests": v.total_tests,
                    "avg_score": v.avg_score,
                    "avg_confidence": v.avg_confidence,
                    "common_issues": v.common_issues,
                }
                for k, v in persona_analyses.items()
            },
            "query_type_analyses": {
                k: {
                    "query_type": v.query_type,
                    "total_tests": v.total_tests,
                    "pass_rate": v.pass_rate,
                    "avg_score": v.avg_score,
                    "avg_execution_time": v.avg_execution_time,
                }
                for k, v in query_type_analyses.items()
            },
            "score_distribution": score_dist,
        }

    def _save_comprehensive_results(
        self,
        summary: Dict[str, Any],
        results: List[TestCaseEvaluation],
        output_dir: str,
    ):
        """Save results to JSON"""
        os.makedirs(output_dir, exist_ok=True)

        output_path = (
            Path(output_dir)
            / f"comprehensive_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        output_data = {
            "summary": summary,
            "results": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "category": r.category,
                    "persona_type": r.persona_type,
                    "query_type": r.query_type,
                    "scores": {
                        "accuracy": r.accuracy_score,
                        "completeness": r.completeness_score,
                        "relevance": r.relevance_score,
                        "source_citation": r.source_citation_score,
                        "practicality": r.practicality_score,
                        "actionability": r.actionability_score,
                        "total": r.total_score,
                    },
                    "metrics": {
                        "confidence": r.confidence,
                        "execution_time_ms": r.execution_time_ms,
                        "sources_count": r.sources_count,
                        "answer_length": r.answer_length,
                    },
                    "passed": r.passed,
                    "issues": r.quality_issues,
                    "strengths": r.strengths,
                    "answer": r.answer[:500],  # Truncate for JSON
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved: {output_path}")

    def _generate_comprehensive_report(
        self,
        summary: Dict[str, Any],
        results: List[TestCaseEvaluation],
        output_dir: str,
    ):
        """Generate comprehensive markdown report"""
        report_path = (
            Path(output_dir)
            / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        lines = []
        lines.append("# RAG System Comprehensive Quality Evaluation Report")
        lines.append(f"\n**Generated:** {summary['evaluated_at']}")
        lines.append("\n## Executive Summary")
        lines.append(f"\n- **Total Test Cases:** {summary['total_tests']}")
        lines.append(f"- **Passed:** {summary['passed']} ({summary['pass_rate']:.1%})")
        lines.append(f"- **Failed:** {summary['failed']}")
        lines.append(
            f"- **Overall Quality Score:** {summary['dimension_averages']['total']:.2f}/5.0"
        )

        # Quality dimensions
        lines.append("\n## Quality Dimensions Analysis")
        lines.append("\n| Dimension | Average | Target | Status |")
        lines.append("|-----------|---------|--------|--------|")

        dims = summary["dimension_averages"]
        lines.append(
            f"| Accuracy | {dims['accuracy']:.2f} | 0.90 | {'✓' if dims['accuracy'] >= 0.90 else '✗'} |"
        )
        lines.append(
            f"| Completeness | {dims['completeness']:.2f} | 0.80 | {'✓' if dims['completeness'] >= 0.80 else '✗'} |"
        )
        lines.append(
            f"| Relevance | {dims['relevance']:.2f} | 0.80 | {'✓' if dims['relevance'] >= 0.80 else '✗'} |"
        )
        lines.append(
            f"| Source Citation | {dims['source_citation']:.2f} | 0.80 | {'✓' if dims['source_citation'] >= 0.80 else '✗'} |"
        )
        lines.append(
            f"| Practicality | {dims['practicality']:.2f} | 0.35 | {'✓' if dims['practicality'] >= 0.35 else '✗'} |"
        )
        lines.append(
            f"| Actionability | {dims['actionability']:.2f} | 0.35 | {'✓' if dims['actionability'] >= 0.35 else '✗'} |"
        )

        # Score distribution
        lines.append("\n## Score Distribution")
        dist = summary["score_distribution"]
        lines.append(
            f"- Excellent (4.5-5.0): {dist['excellent']} ({dist['excellent'] / summary['total_tests'] * 100:.1f}%)"
        )
        lines.append(
            f"- Good (4.0-4.4): {dist['good']} ({dist['good'] / summary['total_tests'] * 100:.1f}%)"
        )
        lines.append(
            f"- Acceptable (3.5-3.9): {dist['acceptable']} ({dist['acceptable'] / summary['total_tests'] * 100:.1f}%)"
        )
        lines.append(
            f"- Poor (<3.5): {dist['poor']} ({dist['poor'] / summary['total_tests'] * 100:.1f}%)"
        )

        # Category analysis
        lines.append("\n## Category Analysis")
        for cat_name, cat_data in summary["category_analyses"].items():
            status = "✓" if cat_data["pass_rate"] >= 0.8 else "✗"
            lines.append(f"\n### {cat_name.upper()} {status}")
            lines.append(f"- Pass Rate: {cat_data['pass_rate']:.1%}")
            lines.append(f"- Average Score: {cat_data['avg_score']:.2f}/5.0")
            if cat_data["failing_cases"]:
                lines.append(
                    f"- Failing Cases: {', '.join(cat_data['failing_cases'][:5])}"
                )

        # Persona analysis
        lines.append("\n## Persona-Based Performance")
        for persona, p_data in summary["persona_analyses"].items():
            lines.append(f"\n### {persona.upper().replace('_', ' ')}")
            lines.append(f"- Average Score: {p_data['avg_score']:.2f}/5.0")
            lines.append(f"- Average Confidence: {p_data['avg_confidence']:.2f}")
            if p_data["common_issues"]:
                lines.append("- Common Issues:")
                for issue in p_data["common_issues"][:3]:
                    lines.append(f"  - {issue}")

        # Query type analysis
        lines.append("\n## Query Type Analysis")
        for qtype, qt_data in summary["query_type_analyses"].items():
            lines.append(f"\n### {qtype.upper().replace('_', ' ')}")
            lines.append(f"- Pass Rate: {qt_data['pass_rate']:.1%}")
            lines.append(f"- Average Score: {qt_data['avg_score']:.2f}/5.0")
            lines.append(f"- Avg Execution Time: {qt_data['avg_execution_time']:.0f}ms")

        # Failing cases
        lines.append("\n## Failing Test Cases Analysis")
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            lines.append(f"\n### Top Issues ({len(failed_results)} total)")
            for r in sorted(failed_results, key=lambda x: x.total_score)[:20]:
                lines.append(f"\n#### {r.test_id}: {r.query[:60]}...")
                lines.append(f"- Score: {r.total_score:.2f}/5.0")
                lines.append(f"- Category: {r.category}")
                lines.append(f"- Persona: {r.persona_type}")
                lines.append(f"- Query Type: {r.query_type}")
                if r.quality_issues:
                    lines.append("- Issues:")
                    for issue in r.quality_issues[:3]:
                        lines.append(f"  - {issue}")
        else:
            lines.append("\nNo failing cases!")

        # Recommendations
        lines.append("\n## Recommendations")

        # Identify weak dimensions
        weak_dims = []
        if dims["accuracy"] < 0.8:
            weak_dims.append("accuracy")
        if dims["completeness"] < 0.8:
            weak_dims.append("completeness")
        if dims["source_citation"] < 0.8:
            weak_dims.append("source_citation")
        if dims["practicality"] < 0.3:
            weak_dims.append("practicality")
        if dims["actionability"] < 0.3:
            weak_dims.append("actionability")

        if weak_dims:
            lines.append("\n### Priority Improvements (Weak Dimensions)")
            lines.append(f"Target dimensions: {', '.join(weak_dims)}")

            if "accuracy" in weak_dims:
                lines.append("\n**Accuracy Improvements:**")
                lines.append(
                    "- Improve retrieval precision for regulation-specific queries"
                )
                lines.append(
                    "- Enhance entity recognition for rule codes and article numbers"
                )
                lines.append("- Add validation for factual claims against regulations")

            if "completeness" in weak_dims:
                lines.append("\n**Completeness Improvements:**")
                lines.append(
                    "- Ensure all aspects of multi-part questions are addressed"
                )
                lines.append("- Include procedural steps for process-oriented queries")
                lines.append("- Add deadline information when relevant")

            if "source_citation" in weak_dims:
                lines.append("\n**Source Citation Improvements:**")
                lines.append(
                    "- Always cite specific regulation articles (e.g., '제15조')"
                )
                lines.append("- Include regulation names in responses")
                lines.append("- Provide links or references to full regulations")

            if "practicality" in weak_dims or "actionability" in weak_dims:
                lines.append("\n**Practicality & Actionability Improvements:**")
                lines.append("- Include specific deadlines and timeframes")
                lines.append("- Provide contact information for relevant departments")
                lines.append("- List required documents and forms")
                lines.append("- Use clear action verbs (신청하세요, 제출하세요, etc.)")

        # Category-specific recommendations
        lines.append("\n### Category-Specific Recommendations")
        for cat_name, cat_data in summary["category_analyses"].items():
            if cat_data["pass_rate"] < 0.8:
                lines.append(f"\n**{cat_name}:**")
                lines.append(f"- Current pass rate: {cat_data['pass_rate']:.1%}")
                lines.append(
                    f"- Review failing cases: {', '.join(cat_data['failing_cases'][:3])}"
                )
                lines.append("- Consider adding more training data for this category")

        # Persona-specific recommendations
        lines.append("\n### Persona-Specific Recommendations")
        for persona, p_data in summary["persona_analyses"].items():
            if p_data["avg_score"] < 4.0:
                lines.append(f"\n**{persona}:**")
                lines.append(f"- Average score: {p_data['avg_score']:.2f}/5.0")
                if p_data["common_issues"]:
                    lines.append("- Address common issues:")
                    for issue in p_data["common_issues"][:2]:
                        lines.append(f"  - {issue}")

        lines.append("\n---")
        lines.append("\n*Report generated by Comprehensive Quality Evaluator*")
        lines.append(
            "*6-Dimension Quality Framework: Accuracy, Completeness, Relevance, Source Citation, Practicality, Actionability*"
        )

        report_text = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Report generated: {report_path}")


def main():
    """Main entry point"""
    from dotenv import load_dotenv

    load_dotenv()

    evaluator = ComprehensiveQualityEvaluator()
    summary = evaluator.run_comprehensive_evaluation()

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed: {summary['failed']}")
    print(f"Overall Quality: {summary['dimension_averages']['total']:.2f}/5.0")
    print("\nDimension Scores:")
    for dim, value in summary["dimension_averages"].items():
        if dim != "total":
            print(f"  {dim}: {value:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
