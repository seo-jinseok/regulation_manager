"""
Data Validator

RAG 데이터셋 품질 검증 시스템
- 답변 품질 검증 (RAGAS Faithfulness 기반)
- 질문 다양성 검증
- 중복 검증
- 카테고리 균형 검증
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataValidator:
    """
    RAG 데이터셋 품질 검증기

    검증 항목:
    1. 답변 품질 (관련성, 정확성, 충분성)
    2. 질문 다양성 (유형, 난이도, 카테고리)
    3. 중복 검증 (질문 간 유사도)
    4. 데이터 균형 (카테고리 분포)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,  # 검증 시에는 일관성을 위해 낮은 온도
    ):
        """
        DataValidator 초기화

        Args:
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    def validate_answer_quality(
        self,
        query: str,
        answer: str,
    ) -> dict[str, Any]:
        """
        답변 품질 검증

        Args:
            query: 질문
            answer: 답변

        Returns:
            검증 결과 딕셔너리
        """
        prompt = f"""
다음 질문-정답 쌍의 품질을 평가하세요.

질문: {query}
답변: {answer}

평가 기준:
1. 관련성 (Relevance): 답변이 질문과 직접적으로 관련이 있는가? (0-1)
2. 정확성 (Accuracy): 답변의 정보가 정확해 보이는가? (0-1)
3. 충분성 (Sufficiency): 답변이 질문에 충분히 응답하는가? (0-1)
4. 명확성 (Clarity): 답변이 이해하기 쉽게 작성되었는가? (0-1)

JSON 형식으로 응답하세요:
{{
    "relevance": 0.0 ~ 1.0,
    "accuracy": 0.0 ~ 1.0,
    "sufficiency": 0.0 ~ 1.0,
    "clarity": 0.0 ~ 1.0,
    "overall_score": 0.0 ~ 1.0,
    "issues": ["문제점1", "문제점2"],
    "suggestions": ["개선 제안1", "개선 제안2"]
}}
"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)

            # 전체 점수 계산
            if "overall_score" not in result:
                scores = [
                    result.get("relevance", 0),
                    result.get("accuracy", 0),
                    result.get("sufficiency", 0),
                    result.get("clarity", 0),
                ]
                result["overall_score"] = sum(scores) / len(scores)

            return result

        except Exception as e:
            logger.error(f"답변 품질 검증 실패: {e}")
            return {
                "relevance": 0.5,
                "accuracy": 0.5,
                "sufficiency": 0.5,
                "clarity": 0.5,
                "overall_score": 0.5,
                "issues": ["검증 실패"],
                "suggestions": [],
            }

    def validate_question_diversity(
        self,
        questions: list[str],
    ) -> dict[str, Any]:
        """
        질문 다양성 검증

        Args:
            questions: 질문 리스트

        Returns:
            다양성 분석 결과
        """
        analysis = {
            "total_questions": len(questions),
            "unique_questions": len(set(questions)),
            "duplicate_ratio": 0.0,
            "avg_question_length": 0,
            "length_distribution": {"short": 0, "medium": 0, "long": 0},
        }

        # 중복 비율
        unique = set(questions)
        analysis["duplicate_ratio"] = 1 - (len(unique) / len(questions))

        # 평균 길이
        lengths = [len(q) for q in questions]
        analysis["avg_question_length"] = sum(lengths) / len(lengths) if lengths else 0

        # 길이 분포
        for length in lengths:
            if length < 20:
                analysis["length_distribution"]["short"] += 1
            elif length < 50:
                analysis["length_distribution"]["medium"] += 1
            else:
                analysis["length_distribution"]["long"] += 1

        return analysis

    def validate_category_balance(
        self,
        pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        카테고리 균형 검증

        Args:
            pairs: 질문-정답 쌍 리스트

        Returns:
            카테고리 분포 분석
        """
        categories = [pair.get("category", "기타") for pair in pairs]
        category_counts = Counter(categories)

        analysis = {
            "total_categories": len(category_counts),
            "category_distribution": dict(category_counts),
            "min_count": min(category_counts.values()) if category_counts else 0,
            "max_count": max(category_counts.values()) if category_counts else 0,
            "balance_ratio": 0.0,
        }

        # 균형 비율 (최소/최대)
        if analysis["max_count"] > 0:
            analysis["balance_ratio"] = analysis["min_count"] / analysis["max_count"]

        return analysis

    def validate_difficulty_distribution(
        self,
        pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        난이도 분포 검증

        Args:
            pairs: 질문-정답 쌍 리스트

        Returns:
            난이도 분포 분석
        """
        difficulties = [pair.get("difficulty", "중급") for pair in pairs]
        difficulty_counts = Counter(difficulties)

        return {
            "difficulty_distribution": dict(difficulty_counts),
            "total_pairs": len(pairs),
        }

    def validate_query_types(
        self,
        pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        질문 유형 분포 검증

        Args:
            pairs: 질문-정답 쌍 리스트

        Returns:
            질문 유형 분포 분석
        """
        query_types = [pair.get("query_type", "정확한 쿼리") for pair in pairs]
        type_counts = Counter(query_types)

        return {
            "query_type_distribution": dict(type_counts),
            "total_types": len(type_counts),
            "type_diversity": len(type_counts) / len(pairs) if pairs else 0,
        }

    def check_answer_completeness(
        self,
        pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        답변 완전성 검증

        Args:
            pairs: 질문-정답 쌍 리스트

        Returns:
            완전성 검증 결과
        """
        empty_answers = 0
        short_answers = 0
        template_answers = 0

        for pair in pairs:
            answer = pair.get("answer", "")

            if not answer or answer.strip() == "":
                empty_answers += 1
            elif len(answer) < 50:
                short_answers += 1

            if pair.get("metadata", {}).get("requires_filling", False):
                template_answers += 1

        return {
            "total_pairs": len(pairs),
            "empty_answers": empty_answers,
            "short_answers": short_answers,
            "template_answers": template_answers,
            "completeness_ratio": (
                (len(pairs) - empty_answers) / len(pairs) if pairs else 0
            ),
        }

    def validate_dataset(
        self,
        pairs: list[dict[str, Any]],
        sample_size: int = 50,
    ) -> dict[str, Any]:
        """
        전체 데이터셋 검증

        Args:
            pairs: 질문-정답 쌍 리스트
            sample_size: 답변 품질 검증 샘플 수

        Returns:
            종합 검증 결과
        """
        logger.info(f"데이터셋 검증 시작: {len(pairs)}개 쌍")

        validation_results = {
            "total_pairs": len(pairs),
            "validation_timestamp": Path(__file__).stat().st_mtime,
        }

        # 1. 질문 다양성 검증
        questions = [pair["query"] for pair in pairs]
        validation_results["question_diversity"] = self.validate_question_diversity(
            questions
        )

        # 2. 카테고리 균형 검증
        validation_results["category_balance"] = self.validate_category_balance(pairs)

        # 3. 난이도 분포 검증
        validation_results["difficulty_distribution"] = (
            self.validate_difficulty_distribution(pairs)
        )

        # 4. 질문 유형 분포 검증
        validation_results["query_types"] = self.validate_query_types(pairs)

        # 5. 답변 완전성 검증
        validation_results["answer_completeness"] = self.check_answer_completeness(
            pairs
        )

        # 6. 답변 품질 검증 (샘플링)
        logger.info(f"답변 품질 검증 중 (샘플 {sample_size}개)...")
        quality_scores = []

        sample_indices = range(0, min(len(pairs), sample_size))
        for idx in tqdm(sample_indices, desc="Answer Quality"):
            pair = pairs[idx]
            quality = self.validate_answer_quality(pair["query"], pair["answer"])
            quality_scores.append(quality["overall_score"])

        validation_results["answer_quality"] = {
            "sample_size": len(quality_scores),
            "avg_score": sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0,
            "min_score": min(quality_scores) if quality_scores else 0,
            "max_score": max(quality_scores) if quality_scores else 0,
        }

        # 종합 평가
        validation_results["overall_assessment"] = self._generate_overall_assessment(
            validation_results
        )

        return validation_results

    def _generate_overall_assessment(
        self,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """종합 평가 생성"""
        assessment = {
            "status": "PENDING",
            "issues": [],
            "recommendations": [],
            "quality_score": 0.0,
        }

        scores = []

        # 질문 다양성 점수
        diversity = results.get("question_diversity", {})
        if diversity.get("duplicate_ratio", 1.0) > 0.3:
            assessment["issues"].append("높은 중복 비율")
        scores.append(1 - diversity.get("duplicate_ratio", 0))

        # 카테고리 균형 점수
        balance = results.get("category_balance", {})
        scores.append(balance.get("balance_ratio", 0))

        # 답변 품질 점수
        quality = results.get("answer_quality", {})
        scores.append(quality.get("avg_score", 0))

        # 답변 완전성 점수
        completeness = results.get("answer_completeness", {})
        scores.append(completeness.get("completeness_ratio", 0))

        # 전체 점수 계산
        assessment["quality_score"] = sum(scores) / len(scores) if scores else 0

        # 상태 결정
        if assessment["quality_score"] >= 0.8:
            assessment["status"] = "EXCELLENT"
        elif assessment["quality_score"] >= 0.6:
            assessment["status"] = "GOOD"
        elif assessment["quality_score"] >= 0.4:
            assessment["status"] = "FAIR"
            assessment["recommendations"].append("답변 품질 향상 필요")
        else:
            assessment["status"] = "POOR"
            assessment["recommendations"].extend(
                [
                    "데이터셋 재검토 필요",
                    "중복 질문 제거 권장",
                    "답변 품질 개선 필요",
                ]
            )

        return assessment

    def filter_low_quality_pairs(
        self,
        pairs: list[dict[str, Any]],
        min_quality_score: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        낮은 품질 쌍 필터링

        Args:
            pairs: 질문-정답 쌍 리스트
            min_quality_score: 최소 품질 점수

        Returns:
            필터링된 쌍 리스트
        """
        filtered = []

        for pair in tqdm(pairs, desc="Filtering Low Quality"):
            # 기본 필터링
            answer = pair.get("answer", "")

            if not answer or len(answer.strip()) < 20:
                continue

            # 질문 기본 필터링
            query = pair.get("query", "")
            if len(query.strip()) < 5:
                continue

            # 품질 점수 검증 (샘플링으로 전체 검증은 비용 과부하 방지)
            quality = self.validate_answer_quality(query, answer)

            if quality["overall_score"] >= min_quality_score:
                filtered.append(pair)

        removed = len(pairs) - len(filtered)
        logger.info(f"필터링 완료: {removed}개 제거됨 ({len(filtered)}개 유지)")

        return filtered

    def save_validation_report(
        self,
        validation_results: dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        검증 리포트 저장

        Args:
            validation_results: 검증 결과
            output_path: 출력 파일 경로
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        logger.info(f"검증 리포트 저장 완료: {output_path}")

    def print_summary(
        self,
        validation_results: dict[str, Any],
    ) -> None:
        """검증 결과 요약 출력"""
        assessment = validation_results.get("overall_assessment", {})

        print("\n" + "=" * 50)
        print("데이터셋 검증 결과 요약")
        print("=" * 50)

        print(f"\n전체 상태: {assessment.get('status', 'UNKNOWN')}")
        print(f"품질 점수: {assessment.get('quality_score', 0):.2f}/1.00")

        print("\n데이터셋 규모:")
        print(f"  - 전체 쌍: {validation_results.get('total_pairs', 0)}개")

        diversity = validation_results.get("question_diversity", {})
        print("\n질문 다양성:")
        print(f"  - 중복 비율: {diversity.get('duplicate_ratio', 0):.2%}")
        print(f"  - 평균 길이: {diversity.get('avg_question_length', 0):.1f}자")

        balance = validation_results.get("category_balance", {})
        print("\n카테고리 균형:")
        print(f"  - 카테고리 수: {balance.get('total_categories', 0)}개")
        print(f"  - 균형 비율: {balance.get('balance_ratio', 0):.2f}")

        quality = validation_results.get("answer_quality", {})
        print("\n답변 품질:")
        print(f"  - 평균 점수: {quality.get('avg_score', 0):.2f}/1.00")

        if assessment.get("issues"):
            print("\n발견된 문제점:")
            for issue in assessment["issues"]:
                print(f"  - {issue}")

        if assessment.get("recommendations"):
            print("\n개선 권장사항:")
            for rec in assessment["recommendations"]:
                print(f"  - {rec}")

        print("\n" + "=" * 50)


def main():
    """테스트 메인 함수"""
    validator = DataValidator()

    # 테스트용 가상 데이터
    test_pairs = [
        {
            "id": "test_001",
            "query": "졸업 요건은 어떻게 되나요?",
            "answer": "졸업 요건은 다음과 같습니다: 총 140학점 이수, 전공 60학점, 교양 30학점",
            "category": "졸업",
            "difficulty": "초급",
            "query_type": "정확한 쿼리",
        },
        {
            "id": "test_002",
            "query": "휴학 신청은 언제까지 가능한가요?",
            "answer": "휴학 신청 기간은 매 학기 시작 전 2주간입니다.",
            "category": "휴학",
            "difficulty": "초급",
            "query_type": "정확한 쿼리",
        },
    ]

    results = validator.validate_dataset(test_pairs)
    validator.print_summary(results)


if __name__ == "__main__":
    main()
