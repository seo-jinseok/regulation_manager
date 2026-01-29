"""
Ground Truth Dataset Builder

RAG 시스템 평가를 위한 Ground Truth 데이터셋 생성 메인 스크립트

Workflow:
1. Flip-the-RAG Generator로 300개 자동 생성
2. Expert Template Generator로 200개 템플릿 생성
3. 데이터 검증 및 필터링
4. Train/Validation/Test 분리 (70/15/15)
5. JSONL 형식으로 저장
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

from .flip_the_rag_generator import FlipTheRAGGenerator
from .templates import ExpertTemplateGenerator
from .validator import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GroundTruthDatasetBuilder:
    """
    Ground Truth 데이터셋 빌더

    RAG 시스템 평가를 위한 질문-정답 쌍 데이터셋을 생성하고 검증합니다.
    """

    def __init__(
        self,
        regulation_dir: Path,
        output_dir: Path,
        target_total: int = 500,
        flip_ratio: float = 0.6,  # 60% Flip-the-RAG
    ):
        """
        GroundTruthDatasetBuilder 초기화

        Args:
            regulation_dir: 규정 파일이 있는 디렉토리
            output_dir: 출력 디렉토리
            target_total: 목표 전체 쌍 수
            flip_ratio: Flip-the-RAG 비율
        """
        self.regulation_dir = regulation_dir
        self.output_dir = output_dir
        self.target_total = target_total
        self.flip_ratio = flip_ratio

        self.flip_generator = FlipTheRAGGenerator()
        self.expert_generator = ExpertTemplateGenerator()
        self.validator = DataValidator()

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

    def build(self) -> dict[str, Any]:
        """
        전체 데이터셋 빌드

        Returns:
            빌드 결과 요약
        """
        logger.info("=" * 60)
        logger.info("Ground Truth Dataset Builder 시작")
        logger.info("=" * 60)

        # 1. Flip-the-RAG 생성
        logger.info(
            f"\n1단계: Flip-the-RAG 생성 (목표: {int(self.target_total * self.flip_ratio)}개)"
        )
        flip_pairs = self._generate_flip_pairs()

        # 2. Expert 템플릿 생성
        logger.info(
            f"\n2단계: Expert 템플릿 생성 (목표: {int(self.target_total * (1 - self.flip_ratio))}개)"
        )
        expert_pairs = self._generate_expert_pairs()

        # 3. 병합
        all_pairs = flip_pairs + expert_pairs
        random.shuffle(all_pairs)  # 셔플

        logger.info(f"\n3단계: 데이터 병합 완료 (총 {len(all_pairs)}개)")

        # 4. 검증
        logger.info("\n4단계: 데이터 검증")
        validation_results = self.validator.validate_dataset(all_pairs)
        self.validator.print_summary(validation_results)

        # 5. 필터링 (품질 기준)
        logger.info("\n5단계: 저품질 데이터 필터링")
        filtered_pairs = self.validator.filter_low_quality_pairs(
            all_pairs,
            min_quality_score=0.5,
        )

        # 6. 분리
        logger.info("\n6단계: Train/Validation/Test 분리")
        splits = self._split_dataset(filtered_pairs)

        # 7. 저장
        logger.info("\n7단계: 데이터셋 저장")
        self._save_splits(splits)
        self._save_metadata(splits, validation_results)

        # 결과 요약
        result = {
            "total_pairs": len(filtered_pairs),
            "train_pairs": len(splits["train"]),
            "val_pairs": len(splits["val"]),
            "test_pairs": len(splits["test"]),
            "validation_status": validation_results["overall_assessment"]["status"],
            "quality_score": validation_results["overall_assessment"]["quality_score"],
        }

        logger.info("\n" + "=" * 60)
        logger.info("데이터셋 빌드 완료!")
        logger.info(f"총 {len(filtered_pairs)}개 쌍 생성")
        logger.info(f"Train: {len(splits['train'])}개 (70%)")
        logger.info(f"Val: {len(splits['val'])}개 (15%)")
        logger.info(f"Test: {len(splits['test'])}개 (15%)")
        logger.info("=" * 60)

        return result

    def _generate_flip_pairs(self) -> list[dict[str, Any]]:
        """Flip-the-RAG로 질문-정답 쌍 생성"""
        target_count = int(self.target_total * self.flip_ratio)

        # 규정 파일 확인
        processed_dir = self.regulation_dir / "processed" / "regulations"

        if processed_dir.exists():
            pairs = self.flip_generator.generate(
                processed_dir, target_pairs=target_count
            )
        else:
            logger.warning(f"규정 파일을 찾을 수 없음: {processed_dir}")
            logger.info("데모 데이터 생성 중...")
            pairs = self._generate_demo_pairs(target_count)

        return pairs[:target_count]

    def _generate_expert_pairs(self) -> list[dict[str, Any]]:
        """Expert 템플릿으로 질문-정답 쌍 생성"""
        target_count = int(self.target_total * (1 - self.flip_ratio))
        templates = self.expert_generator.generate_templates(target_count)
        return templates

    def _generate_demo_pairs(self, count: int) -> list[dict[str, Any]]:
        """데모용 질문-정답 쌍 생성"""
        demo_pairs = []

        demo_data = [
            {
                "query": "졸업 요건은 어떻게 되나요?",
                "answer": "졸업 요건은 총 140학점 이수, 전공 60학점, 교양 30학점, 일반선택 50학점입니다.",
                "category": "졸업",
            },
            {
                "query": "휴학 신청은 언제까지 가능한가요?",
                "answer": "휴학 신청은 매 학기 개시일 20일 전부터 개시일 7일 전까지 가능합니다.",
                "category": "휴학",
            },
            {
                "query": "복학 절차가 궁금합니다",
                "answer": "복학 신청은 학기 개시 30일 전부터 20일 전까지 소속 학과에 신청서를 제출해야 합니다.",
                "category": "복학",
            },
            {
                "query": "장학금 신청 자격이 어떻게 되나요?",
                "answer": "성적 장학금은 직전 학기 15학점 이상 이수하고 평점 3.0 이상인 학생에게 지급됩니다.",
                "category": "장학금",
            },
            {
                "query": "등록금 납부 기간은 언제인가요?",
                "answer": "등록금 납부 기간은 매 학기 시작 2주 전부터 1주일 전까지입니다.",
                "category": "등록",
            },
            {
                "query": "F 학점을 받으면 어떻게 되나요?",
                "answer": "F 학점은 0.0 평점으로 처리되며, 해당 과목을 재수강해야 합니다.",
                "category": "성적",
            },
            {
                "query": "교양 과목을 몇 학점 이수해야 하나요?",
                "answer": "교양 이수 학점은 총 30학점 이상이며, 영역별로 최소 6학점 이상 이수해야 합니다.",
                "category": "교과과정",
            },
            {
                "query": "교환학생 지원 자격이 어떻게 되나요?",
                "answer": "교환학생 지원 자격은 2학년 이상으로 직전 학기 평점 3.0 이상이어야 합니다.",
                "category": "교환학생",
            },
        ]

        import uuid

        for i in range(count):
            demo = demo_data[i % len(demo_data)].copy()
            demo["id"] = f"gt_demo_{uuid.uuid4().hex[:8]}"
            demo["difficulty"] = ["초급", "중급", "고급"][i % 3]
            demo["query_type"] = ["정확한 쿼리", "구어체 쿼리", "복합 질문"][i % 3]
            demo["context"] = [demo["category"]]
            demo["metadata"] = {
                "source": "demo_data",
                "is_generated": True,
            }
            demo_pairs.append(demo)

        return demo_pairs

    def _split_dataset(
        self,
        pairs: list[dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        데이터셋을 Train/Validation/Test로 분리

        Args:
            pairs: 질문-정답 쌍 리스트
            train_ratio: Train 비율
            val_ratio: Validation 비율

        Returns:
            분리된 데이터셋 딕셔너리
        """
        # 카테고리 균형을 위해 층화 추출
        from collections import defaultdict

        by_category = defaultdict(list)
        for pair in pairs:
            category = pair.get("category", "기타")
            by_category[category].append(pair)

        train, val, test = [], [], []

        for category, category_pairs in by_category.items():
            random.shuffle(category_pairs)

            n_total = len(category_pairs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train.extend(category_pairs[:n_train])
            val.extend(category_pairs[n_train : n_train + n_val])
            test.extend(category_pairs[n_train + n_val :])

        # 최종 셔플
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return {
            "train": train,
            "val": val,
            "test": test,
        }

    def _save_splits(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        """분리된 데이터셋 저장"""
        for split_name, pairs in splits.items():
            output_path = self.output_dir / f"{split_name}" / f"{split_name}.jsonl"

            with open(output_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            logger.info(
                f"{split_name.upper()} 저장 완료: {output_path} ({len(pairs)}개)"
            )

    def _save_metadata(
        self,
        splits: dict[str, list[dict[str, Any]]],
        validation_results: dict[str, Any],
    ) -> None:
        """메타데이터 저장"""
        metadata = {
            "dataset_id": "rag_gt_v1.0",
            "description": "RAG 시스템 평가를 위한 Ground Truth 데이터셋",
            "total_pairs": sum(len(p) for p in splits.values()),
            "splits": {
                "train": {"count": len(splits["train"]), "ratio": 0.7},
                "val": {"count": len(splits["val"]), "ratio": 0.15},
                "test": {"count": len(splits["test"]), "ratio": 0.15},
            },
            "generation_method": {
                "flip_the_rag": {
                    "count": int(self.target_total * self.flip_ratio),
                    "ratio": self.flip_ratio,
                },
                "expert_templates": {
                    "count": int(self.target_total * (1 - self.flip_ratio)),
                    "ratio": 1 - self.flip_ratio,
                },
            },
            "validation": validation_results,
            "created_at": Path(__file__).stat().st_mtime,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"메타데이터 저장 완료: {metadata_path}")


def main():
    """메인 함수"""
    import sys

    # 경로 설정
    project_root = Path("/Users/truestone/Dropbox/repo/University/regulation_manager")
    regulation_dir = project_root / "data"
    output_dir = project_root / "data" / "ground_truth"

    # 빌더 생성
    builder = GroundTruthDatasetBuilder(
        regulation_dir=regulation_dir,
        output_dir=output_dir,
        target_total=500,
        flip_ratio=0.6,
    )

    # 데이터셋 빌드
    try:
        result = builder.build()

        # 결과 출력
        print("\n" + "=" * 60)
        print("빌드 결과 요약")
        print("=" * 60)
        print(f"전체 쌍: {result['total_pairs']}개")
        print(f"Train: {result['train_pairs']}개")
        print(f"Val: {result['val_pairs']}개")
        print(f"Test: {result['test_pairs']}개")
        print(f"품질 상태: {result['validation_status']}")
        print(f"품질 점수: {result['quality_score']:.2f}")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"데이터셋 빌드 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
