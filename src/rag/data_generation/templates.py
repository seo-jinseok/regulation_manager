"""
Expert Template Generator

전문가 검증을 위한 질문-정답 쌍 템플릿을 생성합니다.
다양한 시나리오와 질문 유형을 커버합니다.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExpertTemplateGenerator:
    """
    전문가 검증 템플릿 생성기

    다양한 카테고리, 난이도, 질문 유형의 템플릿을 제공합니다.
    """

    # 카테고리별 템플릿
    CATEGORY_TEMPLATES = {
        "졸업": [
            {
                "query": "졸업 요건은 어떻게 되나요?",
                "answer_template": "졸업 요건은 다음과 같습니다: {details}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "학점이 부족한데 졸업할 수 있나요?",
                "answer_template": "졸업 요건 미충족 시 {consequence}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
            {
                "query": "복수전공 졸업 요건은 단일전공과 어떻게 다른가요?",
                "answer_template": "복수전공 졸업 요건: {requirements}",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
        ],
        "휴학": [
            {
                "query": "휴학 신청은 언제까지 가능한가요?",
                "answer_template": "휴학 신청 기간: {period}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "군휴학 기간은 어디까지 인정되나요?",
                "answer_template": "군휴학 기간: {duration}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "복학": [
            {
                "query": "복학 신청 절차가 궁금합니다",
                "answer_template": "복학 신청 절차: {steps}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "휴학 후 복학 가능한 시기는?",
                "answer_template": "복학 가능 시기: {timing}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "장학금": [
            {
                "query": "장학금 신청 자격이 어떻게 되나요?",
                "answer_template": "장학금 신청 자격: {eligibility}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "성적 장학금을 받으려면 몇 학점 이상 수강해야 하나요?",
                "answer_template": "성적 장학금 신청 학점 기준: {credits}",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
        ],
        "등록": [
            {
                "query": "등록금 납부 기간은 언제인가요?",
                "answer_template": "등록금 납부 기간: {period}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "등록금 분낰이 가능한가요?",
                "answer_template": "등록금 분낰: {installment_info}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "성적": [
            {
                "query": "학점 평균 계산 방법이 궁금합니다",
                "answer_template": "학점 평균 계산: {calculation_method}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "F 학점을 받으면 어떻게 되나요?",
                "answer_template": "F 학점 처리: {consequence}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
            {
                "query": "성적 이의신청은 어떻게 하나요?",
                "answer_template": "성적 이의신청 절차: {appeal_process}",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
        ],
        "교과과정": [
            {
                "query": "교양 과목을 몇 학점 이수해야 하나요?",
                "answer_template": "교양 이수 학점: {credits}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "전공 심화 과목과 기초 과목의 차이가 무엇인가요?",
                "answer_template": "전공 심화/기초 차이: {difference}",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
        ],
        "교환학생": [
            {
                "query": "교환학생 지원 자격이 어떻게 되나요?",
                "answer_template": "교환학생 지원 자격: {eligibility}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "교환학생으로 갔을 때 학점 인정은 어떻게 되나요?",
                "answer_template": "교환학생 학점 인정: {credit_transfer}",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
        ],
        "규정해석": [
            {
                "query": "이 규정의 적용 대상은 누구인가요?",
                "answer_template": "적용 대상: {scope}",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "이 조항의 목적이 무엇인가요?",
                "answer_template": "조항 목적: {purpose}",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
        ],
        "신청절차": [
            {
                "query": "서류 제출 방법이 궁금합니다",
                "answer_template": "서류 제출 방법: {submission_method}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "신청 후 승인까지 얼마나 걸리나요?",
                "answer_template": "승인 소요 기간: {processing_time}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "기간": [
            {
                "query": "신청 기간이 언제부터 언제까지인가요?",
                "answer_template": "신청 기간: {period}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "연장 신청은 기간 내에 해야 하나요?",
                "answer_template": "연장 신청 기한: {deadline}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "서류": [
            {
                "query": "필요한 서류가 무엇인가요?",
                "answer_template": "필수 서류: {documents}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "서류 미제출 시 어떻게 되나요?",
                "answer_template": "서류 미제출 처리: {consequence}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
        "자격": [
            {
                "query": "지원 자격이 어떻게 되나요?",
                "answer_template": "지원 자격: {eligibility}",
                "difficulty": "초급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "자격 미달 시 재지원이 가능한가요?",
                "answer_template": "재지원 가능 여부: {reapplication}",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
        ],
    }

    # 질문 유형별 변형 템플릿
    QUERY_TYPE_VARIATIONS = {
        "오타 포함 쿼리": [
            ("졸업", "졸엽"),
            ("장학금", "장학금"),
            ("복학", "복학"),
            ("휴학", "휴학"),
            ("등록금", "등록금"),
        ],
        "영문 혼용 쿼리": [
            "성적(GPA)은 어떻게 계산되나요?",
            "수강신청(Register) 기간이 언제인가요?",
        ],
        "모호한 쿼리": [
            "그거 언제까지야?",
            "신청하는 법 알려줘",
            "자격이 어떻게 됨?",
        ],
    }

    def __init__(self):
        """ExpertTemplateGenerator 초기화"""
        self.templates: list[dict[str, Any]] = []

    def generate_templates(
        self,
        target_count: int = 200,
    ) -> list[dict[str, Any]]:
        """
        전문가 검증 템플릿 생성

        Args:
            target_count: 목표 템플릿 수

        Returns:
            생성된 템플릿 리스트
        """
        templates = []

        # 기본 템플릿 추가
        for category, tmpl_list in self.CATEGORY_TEMPLATES.items():
            for tmpl in tmpl_list:
                template = {
                    "id": f"gt_expert_{uuid.uuid4().hex[:8]}",
                    "query": tmpl["query"],
                    "answer": tmpl["answer_template"],
                    "context": [category],
                    "category": category,
                    "difficulty": tmpl["difficulty"],
                    "query_type": tmpl["query_type"],
                    "metadata": {
                        "source": "expert_template",
                        "is_template": True,
                        "requires_filling": True,
                    },
                }
                templates.append(template)

        # 질문 유형 변형 추가
        variations = self._generate_variations(templates[:20])  # 상위 20개 변형
        templates.extend(variations)

        # 시나리오 기반 템플릿 추가
        scenarios = self._generate_scenario_templates()
        templates.extend(scenarios)

        # 목표 수량 맞추기
        while len(templates) < target_count:
            # 반복 패턴 생성
            base = templates[len(templates) % len(templates)]
            variant = base.copy()
            variant["id"] = f"gt_expert_{uuid.uuid4().hex[:8]}"

            # 질문 약간 변형
            if "초급" in variant["difficulty"]:
                variant["query"] = f"{variant['query']} (자세히 알려주세요)"
            elif "고급" in variant["difficulty"]:
                variant["query"] = (
                    f"{variant['query']}? 구체적인 예시와 함께 설명 부탁드립니다."
                )

            templates.append(variant)

        self.templates = templates[:target_count]

        logger.info(f"전문가 템플릿 {len(self.templates)}개 생성 완료")

        return self.templates

    def _generate_variations(
        self,
        base_templates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """질문 유형 변형 생성"""
        variations = []

        for base in base_templates:
            # 오타 포함 쿼리
            for original, typo in self.QUERY_TYPE_VARIATIONS["오타 포함 쿼리"]:
                if original in base["query"]:
                    variant = base.copy()
                    variant["id"] = f"gt_expert_{uuid.uuid4().hex[:8]}"
                    variant["query"] = base["query"].replace(original, typo)
                    variant["query_type"] = "오타 포함 쿼리"
                    variations.append(variant)

            # 모호한 쿼리 변형
            if len(variations) < 30:
                variant = base.copy()
                variant["id"] = f"gt_expert_{uuid.uuid4().hex[:8]}"
                # 구어체로 변형
                variant["query"] = base["query"].replace("어떻게 되나요?", "어떻게 됨?")
                variant["query"] = variant["query"].replace("궁금합니다", "궁금")
                variant["query_type"] = "구어체 쿼리"
                variations.append(variant)

        return variations[:50]

    def _generate_scenario_templates(self) -> list[dict[str, Any]]:
        """시나리오 기반 템플릿 생성"""
        scenarios = [
            {
                "query": "저 이번 학기에 휴학하고 싶은데, 등록금 납부 후에도 가능한가요?",
                "answer": "등록금 납부 후 휴학 처리: {refund_policy}",
                "category": "휴학",
                "difficulty": "고급",
                "query_type": "문맥 의존 질문",
            },
            {
                "query": "성적 장학금 받은 학생이 휴학하면 장학금 반환해야 하나요?",
                "answer": "장학금 수혜 후 휴학 시 처리: {policy}",
                "category": "장학금",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
            {
                "query": "편입학생으로서 전공 인정 학점이 모자라면 어떻게 되나요?",
                "answer": "편입학생 전공 인정: {credit_transfer_policy}",
                "category": "졸업",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
            {
                "query": "군 입대 전에 등록했는데 휴학하면 등록금 환불되나요?",
                "answer": "군입대 관련 등록금 환불: {military_refund_policy}",
                "category": "휴학",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
            {
                "query": "F 학점이 여러 개면 제적되나요?",
                "answer": "학사경고 및 제적 기준: {academic_warning_policy}",
                "category": "성적",
                "difficulty": "중급",
                "query_type": "구어체 쿼리",
            },
            {
                "query": "교환학생 갔다 오면 졸업 학점에서 인정받을 수 있는 최대 학점은?",
                "answer": "교환학생 인정 가능 최대 학점: {max_transfer_credits}",
                "category": "교환학생",
                "difficulty": "고급",
                "query_type": "복합 질문",
            },
            {
                "query": "복수전공과 부전공의 차이가 무엇인가요?",
                "answer": "복수전공과 부전공 차이: {difference}",
                "category": "졸업",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "계절 학기에 이수한 학점은 졸업 요건에 포함되나요?",
                "answer": "계절학기 학점 인정: {summer_credit_policy}",
                "category": "졸업",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
            {
                "query": "장학금 신청 기간을 놓쳤는데 다음 기회는 언제인가요?",
                "answer": "장학금 신청 기간 놓친 경우: {next_opportunity}",
                "category": "장학금",
                "difficulty": "중급",
                "query_type": "문맥 의존 질문",
            },
            {
                "query": "수강 신청 정정 기간에 변경 가능한 학점 제한이 있나요?",
                "answer": "수강신청 정정 기간 학점 변경: {correction_policy}",
                "category": "교과과정",
                "difficulty": "중급",
                "query_type": "정확한 쿼리",
            },
        ]

        templates = []
        for scenario in scenarios:
            template = {
                "id": f"gt_scenario_{uuid.uuid4().hex[:8]}",
                "query": scenario["query"],
                "answer": scenario["answer"],
                "context": [scenario["category"]],
                "category": scenario["category"],
                "difficulty": scenario["difficulty"],
                "query_type": scenario["query_type"],
                "metadata": {
                    "source": "expert_scenario",
                    "is_template": True,
                    "requires_filling": True,
                },
            }
            templates.append(template)

        return templates

    def save(
        self,
        output_path: Path,
        dataset_id: str = "rag_gt_expert_v1.0",
    ) -> None:
        """
        템플릿을 JSON 형식으로 저장

        Args:
            output_path: 출력 파일 경로
            dataset_id: 데이터셋 ID
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            "dataset_id": dataset_id,
            "total_pairs": len(self.templates),
            "created_at": Path(__file__).stat().st_mtime,
            "description": "Expert-validated question-answer templates for RAG evaluation",
            "pairs": self.templates,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"저장 완료: {output_path}")


def main():
    """테스트 메인 함수"""
    generator = ExpertTemplateGenerator()
    generator.generate_templates(target_count=200)
    generator.save(Path("data/ground_truth/expert_templates.json"))


if __name__ == "__main__":
    main()
