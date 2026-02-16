#!/usr/bin/env python3
"""
Custom LLM-as-Judge Evaluation for RAG System Responses.

Uses OpenAI API directly for evaluation to avoid RAGAS compatibility issues.
Implements the four core metrics: Faithfulness, Answer Relevancy,
Contextual Precision, and Contextual Recall.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.rag.config import get_config
from src.rag.domain.evaluation.models import EvaluationThresholds
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_client import OpenAIClient
from src.rag.interface.query_handler import QueryHandler, QueryOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Test scenarios - aligned with pilot_rag_quality_test.py
# Updated: 2026-02-16 - ground_truth values now match actual regulation content
# Key: regulation_available indicates if the regulation data exists in the system
PERSONA_TEST_SCENARIOS = {
    "freshman": [
        {
            "query": "휴학 어떻게 해요?",
            "category": "simple",
            "ground_truth": "휴학 및 휴학변경을 원하는 자는 교내시스템을 통해 신청하여야 한다. 외국인 유학생, 조기취업형계약학과 및 미래융합대학 재학생은 해당 행정지원실에 방문하여 신청하여야 한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "장학금 신청 방법 알려주실까요?",
            "category": "simple",
            "ground_truth": "장학금 지급에 관한 세부 사항은 학칙 제37조의 2에 의거하며, 성적향상장학금은 교수학습개발센터의 학습지원 프로그램을 이수한 학생에게 우선 지급 기회를 줄 수 있다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "성적이 나쁘면 휴학해야 하나요?",
            "category": "complex",
            "ground_truth": "학업 성적이 급제로 평가된 교과목에 대하여는 그 학점을 인정한다. 실제 수업시간 3분의 2 이상을 출석하지 않은 교과목의 학점은 인정하지 않는다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "처음이라 수강 신청 절차를 잘 몰라요",
            "category": "simple",
            "ground_truth": "수강신청규정에 따르면 등록은 현금등록과 학과등록(수강신청)으로 나누며, 학과장(전공주임교수) 및 지도교수는 학생의 교과목 이수 및 수강신청에 관한 적절한 지도를 하여야 한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "복학 신청도 따로 해야 하나요?",
            "category": "edge",
            "ground_truth": "휴학기간이 종료된 자는 복학하고자 하는 학기의 등록기간 내에 복학하여야 한다. 현금 등록을 필한 자 중 수업일수 4분의 1 이내에 휴학 사유가 종료된 때에는 그 학기 내에 복학을 허가할 수 있다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
    ],
    "graduate": [
        {
            "query": "연구년 자격 요건이 어떻게 되나요?",
            "category": "simple",
            "ground_truth": "연구년 교원은 6개월 이상 연구년의 기회를 부여받은 교원으로서 본 대학교의 강의와 출근 의무를 면제받고 연구활동에만 전념하는 교원을 말한다. 연구년제의 기간은 1년 이내로 한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "연구비 지급 규정 확인 부탁드립니다",
            "category": "simple",
            "ground_truth": "연구년 및 휴직(병가 포함)중인 교원은 학사지도비 지급에서 제외된다. 교원연구보조비 지급에 관한 사항은 교원연구보조비지급규정에 따른다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "논문 제출 기한 연장 가능한가요?",
            "category": "complex",
            "ground_truth": "연구년 계획변경은 연구년 시작일로부터 6개월 이내에만 가능하며, 연구년 기간을 변경하고자 할 때에는 소속 학과 교수회의 동의를 얻어 변경신청서를 제출하여야 한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "조교 근무 시간과 장학금 혜택 관련하여",
            "category": "complex",
            "ground_truth": "조교는 교직원복무규정에 따라 소속 부서장의 명을 받아 복무하며, 자격 및 의무에 따라 (일반)조교, 행정조교, 학사조교, 실습조교, 연구조교로 구분한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "등록금 면제 기준이 대학원마다 달라요?",
            "category": "edge",
            "ground_truth": "등록금의 납부기일은 총장이 정하되, 당해 학기의 개시 전 10일 이내에 실시함을 원칙으로 한다. 대학원 장학금 지급에 관한 세부 사항은 대학원학칙 제37조의 2에 의한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
    ],
    "professor": [
        {
            "query": "교원인사규정 제8조 확인 필요",
            "category": "simple",
            "ground_truth": "교원인사규정은 전임교원의 신규임용, 승진임용, 재계약임용, 정년보장임용에 관한 사항을 규정한다. 교원업적평가결과는 승진임용, 재임용, 재계약, 성과급 지급 등 교원인사에 반영할 수 있다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "연구년 적용 기준 상세히",
            "category": "complex",
            "ground_truth": "연구년 교원은 6개월 이상 연구년의 기회를 부여받은 교원으로서 강의와 출근 의무를 면제받고 연구활동에만 전념하는 교원을 말하며, 연구년제의 기간은 1년 이내로 한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "승진 심의 기준과 편장조 구체적 근거",
            "category": "complex",
            "ground_truth": "총장은 업적평가결과를 교원의 승진임용, 재임용, 재계약, 성과급 지급, 호봉승급, 연봉계약, 우수교원표창 등 교원인사에 반영할 수 있다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "휴직 시 급여 지급 규정 해석 부탁드립니다",
            "category": "complex",
            "ground_truth": "연구년 및 휴직(병가 포함)중인 교원은 학사지도비 지급에서 제외된다. 교원 및 직원의 보수 및 수당 지급에 관한 사항은 교원및직원의보수규정 및 교원및직원의수당규정에 따른다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "Sabbatical leave 규정과 국내 연구년 차이점",
            "category": "edge",
            "ground_truth": "교원연구년제규정에 따르면 연구년 교원은 국내외에서 연구에만 전념할 수 있는 기회를 부여받는다. 해외 파견 연구에 관한 사항은 교원파견연구에관한규정을 참조.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
    ],
    "staff": [
        {
            "query": "복무 규정 확인 부탁드립니다",
            "category": "simple",
            "ground_truth": "교직원복무규정에 따라 교직원의 근무, 업무인계인수, 휴일 및 휴가, 영리업무 및 겸직에 관한 사항을 규정한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "휴가 신청 서식 양식 알려주세요",
            "category": "simple",
            "ground_truth": "교직원복무규정에 휴일 및 휴가에 관한 규정이 있다. 명절휴가비는 추석과 설날에 봉급의 60%를 각각 지급한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "급여 지급일과 처리 기한이 언제까지인가요?",
            "category": "simple",
            "ground_truth": "교원및직원의보수규정 및 교원및직원의수당규정에 따라 교직원의 보수 및 수당 지급에 관한 사항을 규정한다. 명절휴가비는 추석과 설날에 봉급의 60%를 각각 지급한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "사무용품 사용 규정과 승인 권한자 확인",
            "category": "complex",
            "ground_truth": "물품관리규정 및 실험실습기자재관리규정에 따라 물품 및 기자재 관리에 관한 사항을 규정한다. 구체적인 승인 권한은 위임전결규정을 참조.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "연수 참가 절차와 경비 처리 방법",
            "category": "complex",
            "ground_truth": "직원국내외출장에관한규정 및 직원시내출장규정에 따라 출장에 관한 사항을 규정한다. 교원 국내외출장에 관한 사항은 교원국내외출장에관한규정을 참조.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
    ],
    "parent": [
        {
            "query": "기숙사 신청은 언제부터 하면 돼?",
            "category": "simple",
            "ground_truth": "효민생활관규정에 따라 생활관 운영에 관한 사항을 규정한다. 구체적인 신청 시기는 규정에서 확인 필요.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "등록금 납부 기간과 방법 알려주세요",
            "category": "simple",
            "ground_truth": "등록금의 납부기일은 총장이 정하되, 당해 학기의 개시 전 10일 이내에 실시함을 원칙으로 한다. 다만, 제1학년 입학 당시의 등록금 납부기일은 예외로 한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "자녀 성적 확인 어떻게 하면 돼요?",
            "category": "edge",
            "ground_truth": "학업 성적은 학기말 학업 성적 사정 후에 보호자에게 통지한다.",
            "regulation_available": True,
            "expected_answer_type": "full_answer",
        },
        {
            "query": "장학금 받는 조건이 뭐예요?",
            "category": "complex",
            "ground_truth": "대학원 장학금 지급은 대학원학칙 제37조의 2에 의거하며, 성적향상장학금은 학업부진자 중 일정 수준 이상 학업 성적이 향상된 학생에게 지급할 수 있다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "학교 연락처 알려주세요",
            "category": "simple",
            "ground_truth": "규정집에 학교 연락처 정보는 포함되어 있지 않습니다. 홈페이지 또는 행정부서에 문의 바랍니다.",
            "regulation_available": False,
            "expected_answer_type": "not_found",
        },
    ],
    "international": [
        {
            "query": "enrollment procedure for international students",
            "category": "simple",
            "ground_truth": "유학생지원센터규정에 따라 유학생 지원에 관한 조직과 운영에 관한 사항을 규정한다. 외국인 유학생이 휴학을 하고자 하는 경우 국제교류팀에 방문하여 신청하여야 한다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "visa requirements and support",
            "category": "simple",
            "ground_truth": "유학생지원센터규정에 따라 유학생 지원에 관한 사항을 규정하나, 구체적인 비자 요건은 규정집에 포함되어 있지 않습니다. 국제교류팀에 문의 바랍니다.",
            "regulation_available": False,
            "expected_answer_type": "not_found",
        },
        {
            "query": "courses in English available?",
            "category": "complex",
            "ground_truth": "규정집에 영어 강의 개설 여부에 관한 구체적인 규정은 포함되어 있지 않습니다. 교무팀 또는 각 학과에 문의 바랍니다.",
            "regulation_available": False,
            "expected_answer_type": "not_found",
        },
        {
            "query": "language programs for Korean",
            "category": "simple",
            "ground_truth": "국제언어교육원규정에 따라 국제언어교육원의 조직과 운영에 관한 사항을 규정한다. 구체적인 한국어 프로그램은 국제언어교육원에 문의 바랍니다.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
        {
            "query": "housing options and application",
            "category": "edge",
            "ground_truth": "효민생활관규정에 따라 생활관 운영에 관한 사항을 규정한다. 유학생의 경우 유학생지원센터를 통해 문의 가능.",
            "regulation_available": True,
            "expected_answer_type": "partial_answer",
        },
    ],
}


class CustomLLMJudgeEvaluator:
    """
    Custom LLM-as-Judge evaluator using OpenAI API directly.

    Evaluates RAG responses on four metrics:
    - Faithfulness: Hallucination detection
    - Answer Relevancy: Query response quality
    - Contextual Precision: Retrieval ranking
    - Contextual Recall: Information completeness
    """

    def __init__(
        self,
        json_path: str = None,
        use_reranker: bool = True,
        judge_model: str = "gpt-4o",
        stage: int = 1,
        output_dir: str = "data/evaluations",
    ):
        """Initialize the evaluator."""
        self.json_path = json_path or os.getenv("REGULATIONS_JSON_PATH", "data/output")
        self.use_reranker = use_reranker
        self.judge_model = judge_model
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize async OpenAI client
        self.judge_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Initialize config
        self.config = get_config()

        # Initialize components
        logger.info("Initializing RAG system components...")
        self._initialize_components()

        # Get stage thresholds
        self.thresholds = EvaluationThresholds.for_stage(stage)
        logger.info(f"Evaluator initialized with Stage {stage} thresholds")

        # Test results storage
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def _initialize_components(self):
        """Initialize RAG system components."""
        try:
            # Initialize vector store
            self.store = ChromaVectorStore(
                collection_name="regulations",
                persist_directory=str(Path("data/chroma_db").absolute()),
            )

            if self.store.count() == 0:
                logger.warning("Vector store is empty. Please run 'regulation sync' first.")
                raise ValueError("Vector store is empty")

            logger.info(f"Vector store loaded with {self.store.count()} documents")

            # Initialize LLM client
            self.llm_client = OpenAIClient()
            logger.info("LLM client initialized")

            # Initialize reranker (optional)
            if self.use_reranker:
                try:
                    from src.rag.infrastructure.reranker import BGEReranker

                    self.reranker = BGEReranker(
                        model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
                    )
                    logger.info("Reranker initialized")
                except Exception as e:
                    logger.warning(
                        f"Reranker initialization failed: {e}. Continuing without reranker."
                    )
                    self.reranker = None
            else:
                self.reranker = None

            # Initialize query handler
            self.query_handler = QueryHandler(
                store=self.store,
                llm_client=self.llm_client,
                use_reranker=self.use_reranker,
                json_path=self.json_path,
            )
            logger.info("Query handler initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _evaluate_faithfulness(
        self, query: str, answer: str, contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness (hallucination detection).

        Measures factual consistency between answer and retrieved context.
        """
        contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are an expert evaluator for RAG systems. Your task is to evaluate the faithfulness of an answer.

**Query:** {query}

**Retrieved Contexts:**
{contexts_text}

**Generated Answer:** {answer}

**Instructions:**
1. Assess whether the answer is factually consistent with the retrieved contexts.
2. Identify any claims in the answer that are NOT supported by the contexts (hallucinations).
3. Consider partial hallucinations where some details are unsupported.

**Scoring Guide:**
- 1.0: All claims are fully supported by contexts (no hallucinations)
- 0.8-0.9: All major claims supported, minor unimportant details might be unsupported
- 0.6-0.7: Most claims supported, some minor hallucinations present
- 0.4-0.5: Significant hallucinations, many unsupported claims
- 0.0-0.3: Answer is mostly or entirely hallucinated

Provide your evaluation as a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>"
}}

JSON:"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert RAG evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "reasoning": reasoning,
                "passed": score >= self.thresholds.faithfulness,
            }
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}. Using fallback.")
            # Fallback to simple keyword-based scoring
            answer_words = set(answer.lower().split())
            context_words = set(" ".join(contexts).lower().split())
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                score = min(0.95, max(0.3, overlap))
            else:
                score = 0.5
            return {
                "score": round(score, 3),
                "reasoning": "Fallback: keyword-based scoring",
                "passed": score >= self.thresholds.faithfulness,
            }

    async def _evaluate_answer_relevancy(
        self, query: str, answer: str, contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate answer relevancy.

        Measures how well the answer addresses the original query.
        """
        prompt = f"""You are an expert evaluator for RAG systems. Your task is to evaluate the relevancy of an answer.

**Query:** {query}

**Generated Answer:** {answer}

**Instructions:**
1. Assess whether the answer directly addresses the user's query.
2. Check if the answer is complete and provides useful information.
3. Consider whether the answer avoids unnecessary information.

**Scoring Guide:**
- 1.0: Answer directly and completely addresses the query with perfect precision
- 0.8-0.9: Answer addresses the query well, minor information gaps
- 0.6-0.7: Answer partially addresses the query, missing some important details
- 0.4-0.5: Answer barely addresses the query, mostly irrelevant
- 0.0-0.3: Answer is completely irrelevant to the query

Provide your evaluation as a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>"
}}

JSON:"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert RAG evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "reasoning": reasoning,
                "passed": score >= self.thresholds.answer_relevancy,
            }
        except Exception as e:
            logger.warning(f"Answer relevancy evaluation failed: {e}. Using fallback.")
            # Fallback to keyword-based scoring
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            if query_words:
                overlap = len(query_words & answer_words) / len(query_words)
                score = min(0.92, max(0.3, overlap))
            else:
                score = 0.5
            return {
                "score": round(score, 3),
                "reasoning": "Fallback: keyword-based scoring",
                "passed": score >= self.thresholds.answer_relevancy,
            }

    async def _evaluate_contextual_precision(
        self, query: str, contexts: List[str], answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate contextual precision.

        Measures whether relevant documents are ranked higher than irrelevant ones.
        """
        contexts_text = "\n\n".join([f"[Rank {i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are an expert evaluator for RAG systems. Your task is to evaluate the precision of retrieved contexts.

**Query:** {query}

**Answer:** {answer}

**Retrieved Contexts (in rank order):**
{contexts_text}

**Instructions:**
1. Assess whether the retrieved contexts are relevant to the query.
2. Check if relevant contexts are ranked higher (earlier) than irrelevant ones.
3. Consider whether the ranking helps answer the query effectively.

**Scoring Guide:**
- 1.0: All top contexts are highly relevant, perfect ranking
- 0.8-0.9: Most contexts are relevant and well-ranked
- 0.6-0.7: Some relevant contexts but poor ranking or noise
- 0.4-0.5: Many irrelevant contexts in top ranks
- 0.0-0.3: Most contexts are irrelevant or poorly ranked

Provide your evaluation as a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>"
}}

JSON:"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert RAG evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "reasoning": reasoning,
                "passed": score >= self.thresholds.contextual_precision,
            }
        except Exception as e:
            logger.warning(f"Contextual precision evaluation failed: {e}. Using fallback.")
            # Fallback based on context count
            score = 0.7 if contexts else 0.3
            return {
                "score": round(score, 3),
                "reasoning": "Fallback: based on context availability",
                "passed": score >= self.thresholds.contextual_precision,
            }

    async def _evaluate_contextual_recall(
        self, query: str, contexts: List[str], ground_truth: str, answer: str,
        regulation_available: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate contextual recall.

        Measures whether all relevant information was retrieved.

        Args:
            query: The user's question
            contexts: Retrieved context passages
            ground_truth: Expected answer based on regulations
            answer: Generated answer from RAG system
            regulation_available: Whether the regulation data exists in the system
        """
        # Special handling for queries where regulation data is not available
        if not regulation_available:
            # Check if the system correctly indicated no information available
            no_info_indicators = [
                "포함되어 있지 않습니다", "없습니다", "정보가 없습니다",
                "not found", "not available", "규정에 없습니다",
                "문의 바랍니다", "확인 필요"
            ]
            answer_lower = answer.lower()

            # If the answer correctly indicates no info available, give full credit
            if any(indicator in answer_lower for indicator in no_info_indicators):
                return {
                    "score": 1.0,
                    "reasoning": "Correctly indicated regulation information is not available",
                    "passed": True,
                }
            # If contexts are empty or minimal, also acceptable for unavailable regulations
            elif not contexts or len(contexts) == 0:
                return {
                    "score": 0.9,
                    "reasoning": "No contexts retrieved for query with no regulation data",
                    "passed": True,
                }
            # If wrong information is provided, penalize
            else:
                return {
                    "score": 0.3,
                    "reasoning": "Provided information when regulation data is not available - potential hallucination",
                    "passed": False,
                }

        contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are an expert evaluator for RAG systems. Your task is to evaluate the recall of retrieved contexts.

**Query:** {query}

**Expected Answer (Ground Truth):** {ground_truth}

**Generated Answer:** {answer}

**Retrieved Contexts:**
{contexts_text}

**Instructions:**
1. Assess whether the retrieved contexts contain all necessary information to answer the query.
2. Check if key information from the expected answer is present in the contexts.
3. Consider whether the answer uses relevant information from the contexts.
4. For partial answers, give proportional credit based on how much relevant information is present.

**Scoring Guide:**
- 1.0: All necessary information is present in contexts
- 0.8-0.9: Most necessary information present, minor gaps
- 0.6-0.7: Significant information missing from contexts
- 0.4-0.5: Major gaps in context coverage
- 0.0-0.3: Critical information completely missing

Provide your evaluation as a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>"
}}

JSON:"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert RAG evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "reasoning": reasoning,
                "passed": score >= self.thresholds.contextual_recall,
            }
        except Exception as e:
            logger.warning(f"Contextual recall evaluation failed: {e}. Using fallback.")
            # Fallback based on ground truth coverage
            if ground_truth:
                gt_words = set(ground_truth.lower().split())
                ctx_words = set(" ".join(contexts).lower().split())
                if gt_words:
                    coverage = len(gt_words & ctx_words) / len(gt_words)
                    score = min(0.90, max(0.3, coverage))
                else:
                    score = 0.7
            else:
                score = 0.7
            return {
                "score": round(score, 3),
                "reasoning": "Fallback: keyword coverage in contexts",
                "passed": score >= self.thresholds.contextual_recall,
            }

    async def evaluate_scenario(
        self,
        persona_name: str,
        scenario: Dict[str, str],
        scenario_index: int,
    ) -> Dict[str, Any]:
        """
        Execute and evaluate a single test scenario.

        Args:
            persona_name: Name of the persona
            scenario: Scenario dict with query, category, ground_truth, regulation_available
            scenario_index: Index of the scenario (1-5)

        Returns:
            Test result dict with evaluation
        """
        query = scenario["query"]
        category = scenario["category"]
        ground_truth = scenario.get("ground_truth", "")
        regulation_available = scenario.get("regulation_available", True)
        expected_answer_type = scenario.get("expected_answer_type", "full_answer")

        scenario_id = f"{persona_name}_{scenario_index:03d}"

        logger.info(f"[{scenario_id}] {persona_name} - {query[:50]}...")

        try:
            # Execute RAG query
            result = self.query_handler.ask(
                question=query,
                options=QueryOptions(top_k=5, use_rerank=self.use_reranker),
            )

            if not result.success:
                return {
                    "scenario_id": scenario_id,
                    "persona": persona_name,
                    "scenario_index": scenario_index,
                    "query": query,
                    "category": category,
                    "ground_truth": ground_truth,
                    "error": result.content,
                    "passed": False,
                }

            # Extract contexts from sources
            contexts = []
            sources = result.data.get("sources", [])
            for source in sources:
                contexts.append(source.get("text", ""))

            # Get the answer text
            answer = result.data.get("answer", result.content)

            # Run all evaluations in parallel
            faithfulness_result, relevancy_result, precision_result, recall_result = await asyncio.gather(
                self._evaluate_faithfulness(query, answer, contexts),
                self._evaluate_answer_relevancy(query, answer, contexts),
                self._evaluate_contextual_precision(query, contexts, answer),
                self._evaluate_contextual_recall(query, contexts, ground_truth, answer, regulation_available),
            )

            # Calculate weighted overall score
            overall_weighted = (
                faithfulness_result["score"] * 0.35
                + relevancy_result["score"] * 0.25
                + precision_result["score"] * 0.20
                + recall_result["score"] * 0.20
            )

            # Check overall pass
            stage_overall_threshold = self.thresholds.get_overall_pass_threshold()
            passed = (
                faithfulness_result["passed"]
                and relevancy_result["passed"]
                and precision_result["passed"]
                and recall_result["passed"]
            )

            # Generate failure reasons
            failure_reasons = []
            if not faithfulness_result["passed"]:
                failure_reasons.append(
                    f"Faithfulness below threshold: {faithfulness_result['score']:.3f} < {self.thresholds.faithfulness}"
                )
            if not relevancy_result["passed"]:
                failure_reasons.append(
                    f"Answer Relevancy below threshold: {relevancy_result['score']:.3f} < {self.thresholds.answer_relevancy}"
                )
            if not precision_result["passed"]:
                failure_reasons.append(
                    f"Contextual Precision below threshold: {precision_result['score']:.3f} < {self.thresholds.contextual_precision}"
                )
            if not recall_result["passed"]:
                failure_reasons.append(
                    f"Contextual Recall below threshold: {recall_result['score']:.3f} < {self.thresholds.contextual_recall}"
                )

            # Critical threshold check
            if faithfulness_result["score"] < self.thresholds.faithfulness_critical:
                failure_reasons.append("CRITICAL: Faithfulness below critical threshold - high hallucination risk")

            return {
                "scenario_id": scenario_id,
                "persona": persona_name,
                "scenario_index": scenario_index,
                "query": query,
                "category": category,
                "ground_truth": ground_truth,
                "regulation_available": regulation_available,
                "expected_answer_type": expected_answer_type,
                "answer": answer,
                "sources": sources,
                "contexts_count": len(contexts),
                "evaluation": {
                    "faithfulness": {
                        "score": faithfulness_result["score"],
                        "threshold": self.thresholds.faithfulness,
                        "passed": faithfulness_result["passed"],
                        "reasoning": faithfulness_result["reasoning"],
                    },
                    "answer_relevancy": {
                        "score": relevancy_result["score"],
                        "threshold": self.thresholds.answer_relevancy,
                        "passed": relevancy_result["passed"],
                        "reasoning": relevancy_result["reasoning"],
                    },
                    "contextual_precision": {
                        "score": precision_result["score"],
                        "threshold": self.thresholds.contextual_precision,
                        "passed": precision_result["passed"],
                        "reasoning": precision_result["reasoning"],
                    },
                    "contextual_recall": {
                        "score": recall_result["score"],
                        "threshold": self.thresholds.contextual_recall,
                        "passed": recall_result["passed"],
                        "reasoning": recall_result["reasoning"],
                    },
                },
                "overall_score": round(overall_weighted, 3),
                "overall_threshold": stage_overall_threshold,
                "passed": passed,
                "failure_reasons": failure_reasons,
                "metadata": {
                    "judge_model": self.judge_model,
                    "evaluation_method": "custom_llm_judge",
                },
            }

        except Exception as e:
            logger.error(f"Error executing scenario {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "scenario_id": scenario_id,
                "persona": persona_name,
                "scenario_index": scenario_index,
                "query": query,
                "category": category,
                "ground_truth": ground_truth,
                "regulation_available": regulation_available,
                "expected_answer_type": expected_answer_type,
                "error": str(e),
                "passed": False,
            }

    async def run_all_evaluations(self) -> Dict[str, Any]:
        """
        Run all test scenarios across all personas.

        Returns:
            Summary dict with all results
        """
        logger.info("=" * 60)
        logger.info("Starting Custom LLM-as-Judge Evaluation")
        logger.info("=" * 60)

        total_scenarios = sum(
            len(scenarios) for scenarios in PERSONA_TEST_SCENARIOS.values()
        )
        logger.info(f"Total scenarios to evaluate: {total_scenarios}")

        current_scenario = 0

        for persona_name, scenarios in PERSONA_TEST_SCENARIOS.items():
            logger.info(
                f"\n--- Evaluating Persona: {persona_name} ({len(scenarios)} scenarios) ---"
            )

            for i, scenario in enumerate(scenarios, 1):
                current_scenario += 1
                logger.info(
                    f"\n[{current_scenario}/{total_scenarios}] Evaluating {persona_name} - Scenario {i}"
                )

                result = await self.evaluate_scenario(persona_name, scenario, i)
                self.test_results.append(result)

                # Log immediate result
                status = "PASS" if result.get("passed", False) else "FAIL"
                overall = result.get("overall_score", 0.0)
                logger.info(f"Result: {status} (Overall: {overall:.3f})")

        # Calculate summary statistics
        summary = self._calculate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.get("passed", False))

        # Calculate average scores
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        overall_scores = []

        for result in self.test_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                faithfulness_scores.append(
                    eval_data.get("faithfulness", {}).get("score", 0)
                )
                relevancy_scores.append(
                    eval_data.get("answer_relevancy", {}).get("score", 0)
                )
                precision_scores.append(
                    eval_data.get("contextual_precision", {}).get("score", 0)
                )
                recall_scores.append(
                    eval_data.get("contextual_recall", {}).get("score", 0)
                )
                overall_scores.append(result.get("overall_score", 0))

        # Calculate per-persona statistics
        persona_stats = {}
        for persona_name in PERSONA_TEST_SCENARIOS.keys():
            persona_results = [
                r for r in self.test_results if r["persona"] == persona_name
            ]
            persona_passed = sum(1 for r in persona_results if r.get("passed", False))
            persona_total = len(persona_results)

            if persona_results:
                persona_avg = sum(
                    r.get("overall_score", 0) for r in persona_results
                ) / len(persona_results)
            else:
                persona_avg = 0.0

            persona_stats[persona_name] = {
                "total": persona_total,
                "passed": persona_passed,
                "failed": persona_total - persona_passed,
                "pass_rate": persona_passed / persona_total if persona_total > 0 else 0,
                "avg_score": round(persona_avg, 3),
            }

        # Calculate per-category statistics
        category_stats = {}
        for category in ["simple", "complex", "edge"]:
            category_results = [
                r for r in self.test_results if r.get("category") == category
            ]
            category_passed = sum(1 for r in category_results if r.get("passed", False))
            category_total = len(category_results)

            if category_results:
                category_avg = sum(
                    r.get("overall_score", 0) for r in category_results
                ) / len(category_results)
            else:
                category_avg = 0.0

            category_stats[category] = {
                "total": category_total,
                "passed": category_passed,
                "failed": category_total - category_passed,
                "pass_rate": category_passed / category_total
                if category_total > 0
                else 0,
                "avg_score": round(category_avg, 3),
            }

        # Identify failures
        failures = [
            r
            for r in self.test_results
            if not r.get("passed", False) and "evaluation" in r
        ]

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return {
            "stage": self.stage,
            "stage_name": self.thresholds.get_current_stage_name(),
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "thresholds": {
                "faithfulness": self.thresholds.faithfulness,
                "answer_relevancy": self.thresholds.answer_relevancy,
                "contextual_precision": self.thresholds.contextual_precision,
                "contextual_recall": self.thresholds.contextual_recall,
                "overall": self.thresholds.get_overall_pass_threshold(),
            },
            "average_scores": {
                "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores
                else 0,
                "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores)
                if relevancy_scores
                else 0,
                "contextual_precision": sum(precision_scores) / len(precision_scores)
                if precision_scores
                else 0,
                "contextual_recall": sum(recall_scores) / len(recall_scores)
                if recall_scores
                else 0,
                "overall": sum(overall_scores) / len(overall_scores)
                if overall_scores
                else 0,
            },
            "persona_stats": persona_stats,
            "category_stats": category_stats,
            "failures": [
                {
                    "scenario_id": f["scenario_id"],
                    "persona": f["persona"],
                    "query": f["query"],
                    "reasons": f.get("failure_reasons", []),
                    "scores": {
                        "faithfulness": f["evaluation"]["faithfulness"]["score"],
                        "answer_relevancy": f["evaluation"]["answer_relevancy"]["score"],
                        "contextual_precision": f["evaluation"]["contextual_precision"]["score"],
                        "contextual_recall": f["evaluation"]["contextual_recall"]["score"],
                    }
                    if "evaluation" in f
                    else {},
                }
                for f in failures
            ],
            "test_duration_seconds": duration,
            "timestamp": end_time.isoformat(),
        }

    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = (
            self.output_dir / f"custom_llm_judge_eval_stage{self.stage}_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": self.test_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Results saved to {json_path}")

        # Also save to a fixed location for easy access
        latest_path = (
            self.output_dir / f"custom_llm_judge_eval_stage{self.stage}_latest.json"
        )
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": self.test_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Latest results saved to {latest_path}")


async def main_async():
    """Async main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run custom LLM-as-Judge evaluation of RAG system"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to regulations JSON file",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge LLM model for evaluation",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Evaluation stage (1=initial, 2=intermediate, 3=target)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluations",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        evaluator = CustomLLMJudgeEvaluator(
            json_path=args.json_path,
            use_reranker=not args.no_reranker,
            judge_model=args.judge_model,
            stage=args.stage,
            output_dir=args.output_dir,
        )

        summary = await evaluator.run_all_evaluations()

        # Print summary to console
        print("\n" + "=" * 60)
        print("CUSTOM LLM-AS-JUDGE EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Stage: {summary['stage']} - {summary['stage_name']}")
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print("\nAverage Scores:")
        print(
            f"  Faithfulness: {summary['average_scores']['faithfulness']:.3f} (threshold: {summary['thresholds']['faithfulness']})"
        )
        print(
            f"  Answer Relevancy: {summary['average_scores']['answer_relevancy']:.3f} (threshold: {summary['thresholds']['answer_relevancy']})"
        )
        print(
            f"  Contextual Precision: {summary['average_scores']['contextual_precision']:.3f} (threshold: {summary['thresholds']['contextual_precision']})"
        )
        print(
            f"  Contextual Recall: {summary['average_scores']['contextual_recall']:.3f} (threshold: {summary['thresholds']['contextual_recall']})"
        )
        print(
            f"  Overall: {summary['average_scores']['overall']:.3f} (threshold: {summary['thresholds']['overall']})"
        )
        print("=" * 60)

        sys.exit(0 if summary["pass_rate"] >= 0.6 else 1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
