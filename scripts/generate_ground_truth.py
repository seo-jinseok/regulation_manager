#!/usr/bin/env python3
"""
Ground Truth Dataset Generation CLI

RAG 시스템 평가를 위한 Ground Truth 데이터셋 생성 커맨드라인 인터페이스

Usage:
    python scripts/generate_ground_truth.py --count 500 --output data/ground_truth
    python scripts/generate_ground_truth.py --validate-only --input data/ground_truth
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트 경로를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rag.data_generation.build_dataset import GroundTruthDatasetBuilder
from rag.data_generation.validator import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="RAG Ground Truth Dataset Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 500개 질문-정답 쌍 생성
  python scripts/generate_ground_truth.py --count 500

  # 기존 데이터셋 검증만
  python scripts/generate_ground_truth.py --validate-only --input data/ground_truth

  # 사용자 정의 규정 디렉토리 사용
  python scripts/generate_ground_truth.py --regulation-dir /path/to/regulations --count 1000

  # Flip-the-RAG 비율 조정
  python scripts/generate_ground_truth.py --flip-ratio 0.7 --count 500
        """,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="목표 전체 질문-정답 쌍 수 (기본값: 500)",
    )

    parser.add_argument(
        "--flip-ratio",
        type=float,
        default=0.6,
        help="Flip-the-RAG 생성 비율 (기본값: 0.6, 즉 60%%)",
    )

    parser.add_argument(
        "--regulation-dir",
        type=Path,
        default=None,
        help="규정 파일이 있는 디렉토리 (기본값: data/processed/regulations)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 디렉토리 (기본값: data/ground_truth)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="데이터 생성 없이 검증만 수행",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="검증할 데이터셋 경로 (--validate-only와 함께 사용)",
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="품질 필터링 임계값 (기본값: 0.5)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로깅 활성화",
    )

    return parser.parse_args()


def load_dataset(input_dir: Path) -> list[dict]:
    """데이터셋 로드"""
    import json

    pairs = []

    # Train, Val, Test 파일 로드
    for split in ["train", "val", "test"]:
        split_file = input_dir / f"{split}" / f"{split}.jsonl"

        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pairs.append(json.loads(line))

    return pairs


def generate_dataset(args):
    """데이터셋 생성"""
    project_root = Path(__file__).parent.parent

    regulation_dir = args.regulation_dir or (
        project_root / "data" / "processed" / "regulations"
    )
    output_dir = args.output or (project_root / "data" / "ground_truth")

    logger.info("=" * 70)
    logger.info("Ground Truth Dataset Generation")
    logger.info("=" * 70)
    logger.info(f"Regulation Directory: {regulation_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Target Count: {args.count}")
    logger.info(f"Flip-the-RAG Ratio: {args.flip_ratio:.1%}")
    logger.info("=" * 70)

    builder = GroundTruthDatasetBuilder(
        regulation_dir=regulation_dir,
        output_dir=output_dir,
        target_total=args.count,
        flip_ratio=args.flip_ratio,
    )

    try:
        result = builder.build()

        # 결과 출력
        print("\n" + "=" * 70)
        print("DATASET GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total Pairs: {result['total_pairs']}")
        print(f"  - Train: {result['train_pairs']} (70%)")
        print(f"  - Val: {result['val_pairs']} (15%)")
        print(f"  - Test: {result['test_pairs']} (15%)")
        print(f"Validation Status: {result['validation_status']}")
        print(f"Quality Score: {result['quality_score']:.2f}/1.00")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        return 1


def validate_dataset(args):
    """데이터셋 검증만 수행"""
    if not args.input:
        logger.error("--input argument required for validation")
        return 1

    logger.info(f"Loading dataset from: {args.input}")

    pairs = load_dataset(args.input)

    if not pairs:
        logger.error("No pairs found in dataset")
        return 1

    logger.info(f"Loaded {len(pairs)} pairs")

    validator = DataValidator()
    validation_results = validator.validate_dataset(pairs, sample_size=100)

    # 결과 출력
    validator.print_summary(validation_results)

    # 검증 리포트 저장
    report_path = args.input / "validation_report.json"
    validator.save_validation_report(validation_results, report_path)
    logger.info(f"Validation report saved: {report_path}")

    # 상태 코드
    status = validation_results["overall_assessment"]["status"]
    if status in ["EXCELLENT", "GOOD"]:
        return 0
    else:
        logger.warning(f"Dataset quality status: {status}")
        return 1


def main():
    """메인 함수"""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.validate_only:
        return validate_dataset(args)
    else:
        return generate_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
