#!/usr/bin/env python3
"""
RAG Quality Evaluation CLI - 새로운 RAG 평가 시스템 통합 CLI

RAGAS 기반 LLM-as-Judge 평가, 페르소나 시뮬레이션, 합성 데이터 생성을
단일 명령어로 실행할 수 있습니다.

Usage:
    # 전체 페르소나로 기준선 평가
    python scripts/run_rag_evaluation.py baseline

    # 특정 페르소나로 평가
    python scripts/run_rag_evaluation.py persona --id freshman

    # 합성 데이터 생성
    python scripts/run_rag_evaluation.py synthetic --count 50

    # 평가 통계 확인
    python scripts/run_rag_evaluation.py stats

    # 대시보드 실행
    python scripts/run_rag_evaluation.py dashboard
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.rag.domain.evaluation import RAGQualityEvaluator
from src.rag.domain.evaluation.personas import PersonaManager
from src.rag.domain.evaluation.synthetic_data import SyntheticDataGenerator
from src.rag.infrastructure.json_loader import JSONDocumentLoader
from src.rag.infrastructure.storage.evaluation_store import EvaluationStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_baseline_evaluation(args):
    """기준선 평가 실행 - 모든 페르소나로 테스트"""
    logger.info("기준선 평가 시작...")

    evaluator = RAGQualityEvaluator(
        judge_model=args.judge_model,
        use_ragas=not args.no_ragas,
    )
    store = EvaluationStore(storage_dir=args.output_dir)
    persona_mgr = PersonaManager()

    results = []

    for persona_id in persona_mgr.list_personas():
        logger.info(f"페르소나 {persona_id} 테스트 중...")
        queries = persona_mgr.generate_queries(
            persona_id, count=args.queries_per_persona, topic=args.topic
        )

        for query in queries:
            # RAG 시스템 실행 (실제 구현 필요)
            # 여기서는 모의 실행
            try:
                from src.rag.application.search_usecase import SearchUseCase
                from src.rag.infrastructure.json_loader import JSONDocumentLoader
                from src.rag.infrastructure.vector_store import ChromaVectorStore

                search_usecase = SearchUseCase(
                    vector_store=ChromaVectorStore(),
                    document_loader=JSONDocumentLoader(),
                )

                search_results = search_usecase.search(
                    query_text=query,
                    top_k=args.top_k,
                )

                contexts = [
                    r.chunk.text for r in search_results
                ] if search_results else []

                # 답변 생성 (실제 구현 필요)
                answer = f"검색된 {len(contexts)}개의 컨텍스트를 바탕으로 답변 생성..."

                # 평가 실행
                result = evaluator.evaluate_single_turn(query, contexts, answer)
                result.persona = persona_id
                results.append(result)
                store.save_evaluation(result)

                logger.info(
                    f"Query: {query[:50]}... Score: {result.overall_score:.2f}"
                )
            except Exception as e:
                logger.error(f"평가 실패: {e}")
                continue

    # 통계 출력
    stats = store.get_statistics()
    print("\n" + "=" * 60)
    print("기준선 평가 결과")
    print("=" * 60)
    print(f"전체 평가: {stats.total_evaluations}")
    print(f"평균 점수: {stats.avg_overall_score:.2f}")
    print(f"합격률: {stats.pass_rate:.1%}")
    print(f"추세: {stats.trend}")
    print("\n메트릭별 점수:")
    print(f"  Faithfulness: {stats.avg_faithfulness:.2f}")
    print(f"  Answer Relevancy: {stats.avg_answer_relevancy:.2f}")
    print(f"  Contextual Precision: {stats.avg_contextual_precision:.2f}")
    print(f"  Contextual Recall: {stats.avg_contextual_recall:.2f}")
    print("=" * 60)


def run_persona_evaluation(args):
    """특정 페르소나로 평가 실행"""
    logger.info(f"페르소나 {args.id} 테스트 시작...")

    evaluator = RAGQualityEvaluator(
        judge_model=args.judge_model,
        use_ragas=not args.no_ragas,
    )
    store = EvaluationStore(storage_dir=args.output_dir)
    persona_mgr = PersonaManager()

    queries = persona_mgr.generate_queries(
        args.id, count=args.count, topic=args.topic
    )

    logger.info(f"{len(queries)}개 쿼리 생성 완료")

    for query in queries:
        # RAG 시스템 실행
        try:
            from src.rag.application.search_usecase import SearchUseCase
            from src.rag.infrastructure.json_loader import JSONDocumentLoader
            from src.rag.infrastructure.vector_store import ChromaVectorStore

            search_usecase = SearchUseCase(
                vector_store=ChromaVectorStore(),
                document_loader=JSONDocumentLoader(),
            )

            search_results = search_usecase.search(query_text=query, top_k=args.top_k)
            contexts = [r.chunk.text for r in search_results] if search_results else []

            # 답변 생성
            answer = f"검색된 {len(contexts)}개의 컨텍스트를 바탕으로 답변..."

            # 평가 실행
            result = evaluator.evaluate_single_turn(query, contexts, answer)
            result.persona = args.id
            store.save_evaluation(result)

            print(f"Score: {result.overall_score:.2f} | Query: {query[:50]}...")
        except Exception as e:
            logger.error(f"평가 실패: {e}")


def run_synthetic_generation(args):
    """합성 테스트 데이터 생성"""
    logger.info("합성 테스트 데이터 생성 시작...")

    loader = JSONDocumentLoader("data/output/규정집.json")
    generator = SyntheticDataGenerator(loader)

    if args.scenarios:
        # 규정 기반 시나리오 생성
        scenarios = generator.generate_scenarios_from_regulations(
            regulation=args.regulation, num_scenarios=args.count
        )

        output_file = Path(args.output_dir) / f"scenarios_{args.regulation}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in scenarios],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"{len(scenarios)}개 시나리오 생성 완료: {output_file}")
    else:
        # 문서 기반 질문 생성
        queries = generator.generate_queries_from_documents(
            num_questions=args.count, difficulty=args.difficulty
        )

        output_file = Path(args.output_dir) / "synthetic_queries.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)

        logger.info(f"{len(queries)}개 질문 생성 완료: {output_file}")


def run_statistics(args):
    """평가 통계 확인"""
    store = EvaluationStore(storage_dir=args.input_dir)

    if args.days:
        stats = store.get_statistics(days=args.days)
    else:
        stats = store.get_statistics()

    print("\n" + "=" * 60)
    print("평가 통계")
    print("=" * 60)
    print(f"전체 평가: {stats.total_evaluations}")
    print(f"평균 점수: {stats.avg_overall_score:.2f}")
    print(f"합격률: {stats.pass_rate:.1%}")
    print(f"최저 점수: {stats.min_score:.2f}")
    print(f"최고 점수: {stats.max_score:.2f}")
    print(f"표준 편차: {stats.std_deviation:.2f}")
    print(f"추세: {stats.trend}")
    print("\n메트릭별 평균:")
    print(f"  Faithfulness: {stats.avg_faithfulness:.2f}")
    print(f"  Answer Relevancy: {stats.avg_answer_relevancy:.2f}")
    print(f"  Contextual Precision: {stats.avg_contextual_precision:.2f}")
    print(f"  Contextual Recall: {stats.avg_contextual_recall:.2f}")
    print("=" * 60)


def run_dashboard(args):
    """Gradio 대시보드 실행"""
    import subprocess

    logger.info("Gradio 대시보드 시작...")
    subprocess.run(["uv", "run", "gradio", "src.rag.interface.web.quality_dashboard:app"])


def main():
    parser = argparse.ArgumentParser(
        description="RAG Quality Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAG_JUDGE_MODEL", "gpt-4o"),
        help="Judge LLM 모델 (기본값: gpt-4o)",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="RAGAS 사용 안 함 (모의 평가)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluations",
        help="평가 결과 출력 디렉터리",
    )
    parser.add_argument(
        "--input-dir",
        default="data/evaluations",
        help="평가 결과 입력 디렉터리",
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # baseline 명령
    baseline_parser = subparsers.add_parser("baseline", help="기준선 평가 실행")
    baseline_parser.add_argument(
        "--queries-per-persona",
        type=int,
        default=5,
        help="페르소나당 쿼리 수 (기본값: 5)",
    )
    baseline_parser.add_argument("--topic", help="특정 주제로만 테스트")
    baseline_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색할 문서 수 (기본값: 5)",
    )

    # persona 명령
    persona_parser = subparsers.add_parser("persona", help="특정 페르소나로 평가")
    persona_parser.add_argument(
        "--id",
        required=True,
        choices=["freshman", "graduate", "professor", "staff", "parent", "international"],
        help="페르소나 ID",
    )
    persona_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="생성할 쿼리 수 (기본값: 10)",
    )
    persona_parser.add_argument("--topic", help="특정 주제")
    persona_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색할 문서 수",
    )

    # synthetic 명령
    synthetic_parser = subparsers.add_parser("synthetic", help="합성 데이터 생성")
    synthetic_parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="생성할 질문 수 (기본값: 50)",
    )
    synthetic_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="난이도 (기본값: mixed)",
    )
    synthetic_parser.add_argument(
        "--scenarios",
        action="store_true",
        help="시나리오 생성 모드",
    )
    synthetic_parser.add_argument(
        "--regulation",
        default="학칙",
        help="시나리오 생성할 규정 (기본값: 학칙)",
    )

    # stats 명령
    stats_parser = subparsers.add_parser("stats", help="평가 통계 확인")
    stats_parser.add_argument(
        "--days",
        type=int,
        help="최근 N일간 통계만",
    )

    # dashboard 명령
    subparsers.add_parser("dashboard", help="Gradio 대시보드 실행")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 명령 실행
    if args.command == "baseline":
        run_baseline_evaluation(args)
    elif args.command == "persona":
        run_persona_evaluation(args)
    elif args.command == "synthetic":
        run_synthetic_generation(args)
    elif args.command == "stats":
        run_statistics(args)
    elif args.command == "dashboard":
        run_dashboard(args)


if __name__ == "__main__":
    import os

    main()
