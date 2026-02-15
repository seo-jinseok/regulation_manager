"""
Unified CLI Interface for Regulation Manager.

Provides a single entry point for all regulation management tasks:
- convert: HWP → JSON conversion
- sync: Database synchronization
- search: Regulation search
- ask: LLM-powered Q&A
- status: Sync status check
- reset: Database reset
- serve: Start Web UI or MCP Server

Usage:
    uv run regulation convert "규정집.hwp"
    uv run regulation search "교원 연구년"
    uv run regulation ask "휴학 절차"
    uv run regulation serve --web
"""

import argparse
import os
import sys
from typing import Optional

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_default_llm_settings():
    """Get default LLM settings from centralized config."""
    from ..config import get_config

    config = get_config()
    return (
        config.llm_providers,
        config.llm_provider,
        config.llm_model,
        config.llm_base_url,
    )


def _add_convert_parser(subparsers):
    """Add convert subcommand parser."""
    convert_providers = [
        "openai",
        "gemini",
        "openrouter",
        "ollama",
        "lmstudio",
        "local",
        "mlx",
    ]
    default_provider = os.getenv("LLM_PROVIDER") or "openai"
    if default_provider not in convert_providers:
        default_provider = "openai"
    default_model = os.getenv("LLM_MODEL") or None
    default_base_url = os.getenv("LLM_BASE_URL") or None

    parser = subparsers.add_parser(
        "convert",
        help="HWPX 파일을 JSON으로 변환",
        description="HWPX 규정집을 구조화된 JSON으로 변환합니다.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="HWPX 파일 또는 디렉토리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="출력 디렉토리 (기본: data/output)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="LLM 전처리 활성화",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=convert_providers,
        help="LLM 프로바이더",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="LLM 모델",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="LLM API URL (로컬 서버용)",
    )
    parser.add_argument(
        "--allow-llm-fallback",
        action="store_true",
        help="LLM 실패 시 정규식 폴백 허용",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시 무시하고 강제 재변환",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="캐시 디렉토리",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--no-enhance-rag",
        action="store_false",
        dest="enhance_rag",
        help="RAG 최적화 비활성화",
    )
    parser.add_argument(
        "--hwpx",
        action="store_true",
        help="HWPX 직접 파싱 사용 (HTML/Markdown 변환 과정을 건너뛰어 정확도 향상)",
    )
    parser.set_defaults(enhance_rag=True)


def _add_sync_parser(subparsers):
    """Add sync subcommand parser."""
    parser = subparsers.add_parser(
        "sync",
        help="규정 데이터베이스 동기화",
        description="JSON 파일을 ChromaDB에 동기화합니다.",
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="규정집 JSON 파일 경로",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 재동기화 (기본: 증분 동기화)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--extract-keywords",
        action="store_true",
        help="동기화 후 키워드 자동 추출 (regulation_keywords.json 갱신)",
    )


def _add_search_parser(subparsers):
    """Add search subcommand parser."""

    # Get default settings
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "search",
        help="규정 검색 (자동으로 답변 생성 또는 문서 검색)",
        description="질문이면 AI 답변을, 키워드면 문서 검색 결과를 자동으로 보여줍니다. (혹은 -a/-q 옵션으로 강제)",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="검색 쿼리 또는 질문",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="결과 개수 (기본: 5)",
    )
    parser.add_argument(
        "--include-abolished",
        action="store_true",
        help="폐지 규정 포함 (검색 모드일 때만 유효)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기 (인터랙티브)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드로 연속 질의",
    )

    # Unified specific arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-a",
        "--answer",
        action="store_true",
        help="AI 답변 생성 강제 (Ask 모드)",
    )
    mode_group.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="문서 검색만 수행 (Search 모드)",
    )

    # LLM options (for answer mode)
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더 (답변 생성 시)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력 (답변 생성 시)",
    )

    # Tool calling is now DEFAULT (FunctionGemma for routing, base LLM for answers)
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Tool Calling 비활성화 (기존 방식 사용)",
    )
    parser.add_argument(
        "--tool-mode",
        type=str,
        choices=["auto", "mlx", "openai", "ollama"],
        default="auto",
        help="Tool Calling 백엔드 (auto: OpenAI API 우선, mlx: Apple Silicon Experimental)",
    )


def _add_ask_parser(subparsers):
    """Add ask subcommand parser (Legacy Wrapper)."""
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "ask",
        help="규정 질문 (search -a와 동일)",
        description="LLM을 사용하여 규정에 대한 질문에 답변합니다.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="질문",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="참고 규정 수",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기 (인터랙티브)",
    )


def _add_status_parser(subparsers):
    """Add status subcommand parser."""
    parser = subparsers.add_parser(
        "status",
        help="동기화 상태 확인",
        description="현재 데이터베이스 동기화 상태를 표시합니다.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )


def _add_reset_parser(subparsers):
    """Add reset subcommand parser."""
    parser = subparsers.add_parser(
        "reset",
        help="데이터베이스 초기화",
        description="모든 데이터를 삭제하고 데이터베이스를 초기화합니다.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="초기화 확인 (필수)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )


def _add_serve_parser(subparsers):
    """Add serve subcommand parser."""
    parser = subparsers.add_parser(
        "serve",
        help="서버 시작 (Web UI 또는 MCP)",
        description="Gradio Web UI 또는 MCP Server를 시작합니다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--web",
        action="store_true",
        help="Gradio Web UI 시작",
    )
    group.add_argument(
        "--mcp",
        action="store_true",
        help="MCP Server 시작",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Web UI 포트 (기본: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Web UI 공개 링크 생성 (Gradio share)",
    )


def _add_evaluate_parser(subparsers):
    """Add evaluate subcommand parser."""
    parser = subparsers.add_parser(
        "evaluate",
        help="RAG 시스템 품질 평가",
        description="테스트 데이터셋으로 검색 품질을 평가합니다.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/config/evaluation_dataset.json",
        help="평가 데이터셋 경로",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="특정 카테고리만 평가",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="검색 결과 수 (기본: 5)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_extract_keywords_parser(subparsers):
    """Add extract-keywords subcommand parser."""
    parser = subparsers.add_parser(
        "extract-keywords",
        help="규정에서 키워드 추출",
        description="규정 JSON에서 핵심 키워드를 자동으로 추출합니다.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/output/규정집.json",
        help="규정 JSON 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/config/regulation_keywords.json",
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="저장하지 않고 결과만 표시",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_feedback_parser(subparsers):
    """Add feedback subcommand parser."""
    parser = subparsers.add_parser(
        "feedback",
        help="피드백 통계 확인",
        description="수집된 피드백 통계를 표시합니다.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="모든 피드백 삭제",
    )


def _add_analyze_parser(subparsers):
    """Add analyze subcommand parser."""
    parser = subparsers.add_parser(
        "analyze",
        help="피드백 기반 개선 제안",
        description="피드백을 분석하여 개선 사항을 제안합니다.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_quality_parser(subparsers):
    """Add quality subcommand parser for RAGAS-based evaluation."""
    parser = subparsers.add_parser(
        "quality",
        help="RAG 시스템 품질 평가 (RAGAS LLM-as-Judge)",
        description="RAGAS 프레임워크를 사용한 LLM-as-Judge 평가를 실행합니다.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAG_JUDGE_MODEL", "gpt-4o"),
        help="Judge LLM 모델 (기본: gpt-4o)",
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
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )

    # Subcommands for quality
    quality_subparsers = parser.add_subparsers(dest="quality_cmd", title="평가 명령어")

    # quality baseline
    baseline_parser = quality_subparsers.add_parser(
        "baseline",
        help="기준선 평가 실행 (모든 페르소나)",
    )
    baseline_parser.add_argument(
        "--queries-per-persona",
        type=int,
        default=5,
        help="페르소나당 쿼리 수 (기본: 5)",
    )
    baseline_parser.add_argument(
        "--topic",
        help="특정 주제로만 테스트",
    )
    baseline_parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="검색할 문서 수 (기본: 5)",
    )

    # quality persona
    persona_parser = quality_subparsers.add_parser(
        "persona",
        help="특정 페르소나로 평가",
    )
    persona_parser.add_argument(
        "--id",
        required=True,
        choices=[
            "freshman",
            "graduate",
            "professor",
            "staff",
            "parent",
            "international",
        ],
        help="페르소나 ID",
    )
    persona_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="생성할 쿼리 수 (기본: 10)",
    )
    persona_parser.add_argument("--topic", help="특정 주제")
    persona_parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="검색할 문서 수",
    )

    # quality synthetic
    synthetic_parser = quality_subparsers.add_parser(
        "synthetic",
        help="합성 데이터 생성",
    )
    synthetic_parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="생성할 질문 수 (기본: 50)",
    )
    synthetic_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="난이도 (기본: mixed)",
    )
    synthetic_parser.add_argument(
        "--scenarios",
        action="store_true",
        help="시나리오 생성 모드",
    )
    synthetic_parser.add_argument(
        "--regulation",
        default="학칙",
        help="시나리오 생성할 규정 (기본: 학칙)",
    )

    # quality stats
    stats_parser = quality_subparsers.add_parser(
        "stats",
        help="평가 통계 확인",
    )
    stats_parser.add_argument(
        "--days",
        type=int,
        help="최근 N일간 통계만",
    )

    # quality dashboard
    quality_subparsers.add_parser(
        "dashboard",
        help="Gradio 품질 대시보드 실행",
    )

    # P2: quality run - 전체 평가 실행 (BatchEvaluationExecutor + ProgressReporter)
    run_parser = quality_subparsers.add_parser(
        "run",
        help="전체 평가 실행 (배치 처리, 진행률 추적)",
        description="BatchEvaluationExecutor를 사용한 전체 평가를 실행합니다.",
    )
    run_parser.add_argument(
        "--personas",
        "-p",
        nargs="+",
        help="특정 페르소나만 테스트 (복수 선택 가능)",
    )
    run_parser.add_argument(
        "--queries-per-persona",
        "-q",
        type=int,
        default=25,
        help="페르소나당 쿼리 수 (기본: 25)",
    )
    run_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5,
        help="API 배치 크기 (기본: 5)",
    )
    run_parser.add_argument(
        "--session-id",
        "-s",
        help="세션 ID (지정하면 해당 세션 재개)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        help="평가 보고서 출력 파일",
    )
    run_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="체크포인트 저장 비활성화",
    )

    # P2: quality resume - 중단된 세션 재개
    resume_parser = quality_subparsers.add_parser(
        "resume",
        help="중단된 평가 세션 재개",
        description="CheckpointManager를 사용하여 중단된 평가를 재개합니다.",
    )
    resume_parser.add_argument(
        "--session-id",
        "-s",
        help="재개할 세션 ID (생략 시 가장 최근 중단 세션)",
    )
    resume_parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="재개 가능한 세션 목록 표시",
    )

    # P2: quality generate-spec - 실패 패턴에서 SPEC 생성
    spec_parser = quality_subparsers.add_parser(
        "generate-spec",
        help="평가 실패 패턴에서 SPEC 문서 생성",
        description="FailureClassifier + SPECGenerator를 사용하여 개선 SPEC을 생성합니다.",
    )
    spec_parser.add_argument(
        "--session-id",
        "-s",
        help="특정 세션의 실패 패턴 사용",
    )
    spec_parser.add_argument(
        "--output",
        "-o",
        help="SPEC 문서 출력 파일",
    )
    spec_parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="실패로 간주할 점수 임계값 (기본: 0.6)",
    )

    # P2: quality status - 세션 상태 확인
    status_parser = quality_subparsers.add_parser(
        "status",
        help="평가 세션 상태 확인",
        description="진행 중인 평가 세션의 상태를 확인합니다.",
    )
    status_parser.add_argument(
        "--session-id",
        "-s",
        help="특정 세션 상태 확인",
    )
    status_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="모든 세션 표시",
    )
    status_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="오래된 완료 세션 정리",
    )


def _add_synonym_parser(subparsers):
    """Add synonym subcommand parser."""
    # Get LLM settings for suggest command
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "synonym",
        help="동의어 관리 (LLM 기반 자동 생성 및 수동 관리)",
        description="동의어 사전을 관리합니다. LLM으로 동의어를 자동 생성하거나 수동으로 추가/제거할 수 있습니다.",
    )
    synonym_subparsers = parser.add_subparsers(dest="synonym_cmd")

    # synonym suggest <term>
    suggest_parser = synonym_subparsers.add_parser(
        "suggest",
        help="LLM으로 동의어 후보 생성",
    )
    suggest_parser.add_argument("term", help="동의어를 생성할 용어")
    suggest_parser.add_argument(
        "--context",
        default="대학 규정",
        help="용어 맥락 (기본: 대학 규정)",
    )
    suggest_parser.add_argument(
        "--auto-add",
        action="store_true",
        help="검토 없이 바로 추가",
    )
    suggest_parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더",
    )
    suggest_parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델명",
    )
    suggest_parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )

    # synonym add <term> <synonym>
    add_parser = synonym_subparsers.add_parser("add", help="동의어 수동 추가")
    add_parser.add_argument("term", help="기준 용어")
    add_parser.add_argument("synonym", help="추가할 동의어")

    # synonym remove <term> <synonym>
    remove_parser = synonym_subparsers.add_parser("remove", help="동의어 제거")
    remove_parser.add_argument("term", help="기준 용어")
    remove_parser.add_argument("synonym", help="제거할 동의어")

    # synonym list [term]
    list_parser = synonym_subparsers.add_parser("list", help="동의어 목록 조회")
    list_parser.add_argument("term", nargs="?", help="특정 용어만 조회 (생략 시 전체)")


def _add_reparse_parser(subparsers):
    """Add reparse subcommand parser for HWPX full reparse."""
    parser = subparsers.add_parser(
        "reparse",
        help="HWPX 파일 일괄 재파싱 및 품질 분석",
        description="모든 HWPX 파일을 재파싱하고 품질 분석 리포트를 생성합니다.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="data/input",
        help="HWPX 파일 디렉토리 (기본: data/input)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/output",
        help="출력 디렉토리 (기본: data/output)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 출력",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 파일 생성 없이 미리보기",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="regulation",
        description="대학 규정 관리 시스템 - HWPX 변환, RAG 검색, AI Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  regulation convert "규정집.hwpx"      HWPX → JSON 변환
  regulation sync data/output/규정집.json  DB 동기화
  regulation search "교원 연구년"        규정 검색
  regulation ask "휴학 절차"             AI 질문
  regulation status                      상태 확인
  regulation serve --web                 Web UI 시작
""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="global_debug",
        help="전역 디버그 모드 활성화",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="사용 가능한 명령어",
        metavar="<command>",
    )

    # Add all subcommands
    _add_convert_parser(subparsers)
    _add_sync_parser(subparsers)
    _add_search_parser(subparsers)
    _add_ask_parser(subparsers)
    _add_status_parser(subparsers)
    _add_reset_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_extract_keywords_parser(subparsers)
    _add_feedback_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_quality_parser(subparsers)
    _add_synonym_parser(subparsers)
    _add_reparse_parser(subparsers)

    return parser


# =============================================================================
# Command Handlers
# =============================================================================


def cmd_convert(args) -> int:
    """Execute convert command - HWP to JSON conversion."""

    # Convert argument names to match main.py expectations
    # (unified CLI uses kebab-case, main.py uses snake_case)
    class ConvertArgs:
        def __init__(self, args):
            self.input_path = args.input_path
            self.output_dir = getattr(args, "output_dir", "data/output")
            self.use_llm = getattr(args, "use_llm", False)
            self.provider = getattr(args, "provider", "openai")
            self.model = getattr(args, "model", None)
            self.base_url = getattr(args, "base_url", None)
            self.allow_llm_fallback = getattr(args, "allow_llm_fallback", False)
            self.force = getattr(args, "force", False)
            self.cache_dir = getattr(args, "cache_dir", ".cache")
            self.verbose = getattr(args, "verbose", False)
            self.enhance_rag = getattr(args, "enhance_rag", True)
            self.hwpx = getattr(args, "hwpx", False)

    from ...main import run_pipeline

    return run_pipeline(ConvertArgs(args))


def cmd_sync(args) -> int:
    """Execute sync command."""
    from .cli import cmd_sync as _cmd_sync

    return _cmd_sync(args)


def cmd_search(args) -> int:
    """Execute search command."""
    from .cli import cmd_search as _cmd_search

    return _cmd_search(args)


def cmd_ask(args) -> int:
    """Execute ask command."""
    from .cli import cmd_ask as _cmd_ask

    return _cmd_ask(args)


def cmd_status(args) -> int:
    """Execute status command."""
    from .cli import cmd_status as _cmd_status

    return _cmd_status(args)


def cmd_reset(args) -> int:
    """Execute reset command."""
    from .cli import cmd_reset as _cmd_reset

    return _cmd_reset(args)


def cmd_reparse(args) -> int:
    """Execute reparse command - HWPX full reparse with quality analysis."""
    from ...commands.reparse_hwpx import main as reparse_main

    # Build argv for reparse_hwpx
    argv = []
    if args.input_dir:
        argv.extend(["--input-dir", args.input_dir])
    if args.output_dir:
        argv.extend(["--output-dir", args.output_dir])
    if args.verbose:
        argv.append("--verbose")
    if args.dry_run:
        argv.append("--dry-run")

    return reparse_main(argv)


def cmd_serve(args) -> int:
    """Execute serve command - start Web UI or MCP Server."""
    import os

    # Enable warmup for server modes
    os.environ["WARMUP_ON_INIT"] = "true"

    if args.web:
        import gradio as gr

        from .gradio_app import CUSTOM_CSS, create_app

        app = create_app(db_path=args.db_path)
        app.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue=gr.themes.colors.emerald,
                neutral_hue=gr.themes.colors.neutral,
            ).set(
                body_background_fill="#0f0f0f",
                body_background_fill_dark="#0f0f0f",
                block_background_fill="#1a1a1a",
                block_background_fill_dark="#1a1a1a",
                border_color_primary="rgba(255,255,255,0.06)",
                border_color_primary_dark="rgba(255,255,255,0.06)",
            ),
        )
        return 0
    elif args.mcp:
        from .mcp_server import mcp

        mcp.run()
        return 0
    return 1


def cmd_evaluate(args) -> int:
    """Execute evaluate command - run quality evaluation."""
    from rich.console import Console

    from ..application.evaluate import EvaluationUseCase
    from ..application.search_usecase import SearchUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.llm_adapter import LLMClientAdapter

    console = Console()

    # Initialize components
    store = ChromaVectorStore(persist_directory=args.db_path)

    # Get default settings for LLM
    _, provider, model, base_url = _get_default_llm_settings()

    llm_client = LLMClientAdapter(
        provider=provider,
        model=model,
        base_url=base_url,
    )
    search_usecase = SearchUseCase(
        store=store,
        llm_client=llm_client,
        use_reranker=True,
    )

    # Run evaluation
    eval_usecase = EvaluationUseCase(
        search_usecase=search_usecase,
        dataset_path=args.dataset,
    )

    console.print("[bold]🔍 평가 데이터셋 로드 중...[/bold]")
    test_cases = eval_usecase.load_dataset()
    console.print(f"[dim]총 {len(test_cases)}개 테스트 케이스[/dim]\n")

    console.print("[bold]🧪 평가 실행 중...[/bold]")
    summary = eval_usecase.run_evaluation(
        top_k=args.top_k,
        category=args.category,
    )

    # Print results
    console.print(eval_usecase.format_summary(summary))

    if args.verbose:
        console.print(eval_usecase.format_details(summary))

    return 0 if summary.pass_rate >= 0.8 else 1


def cmd_extract_keywords(args) -> int:
    """Execute extract-keywords command."""
    from rich.console import Console

    from ..infrastructure.keyword_extractor import KeywordExtractor

    console = Console()

    extractor = KeywordExtractor(
        json_path=args.json_path,
        output_path=args.output,
    )

    console.print("[bold]📚 규정 키워드 추출 중...[/bold]")
    result = extractor.extract_keywords()

    console.print(extractor.format_summary(result))

    if args.verbose:
        console.print(extractor.format_details(result))

    if not args.dry_run:
        output_path = extractor.save_keywords(result)
        console.print(f"\n[green]✅ 저장됨: {output_path}[/green]")

    return 0


def cmd_feedback(args) -> int:
    """Execute feedback command."""
    from rich.console import Console

    from ..infrastructure.feedback import FeedbackCollector

    console = Console()
    collector = FeedbackCollector()

    if args.clear:
        collector.clear_feedback()
        console.print("[yellow]🗑️ 모든 피드백이 삭제되었습니다.[/yellow]")
        return 0

    stats = collector.get_statistics()
    console.print(collector.format_statistics(stats))

    return 0


def cmd_analyze(args) -> int:
    """Execute analyze command - analyze feedback for improvements."""
    from rich.console import Console

    from ..application.auto_learn import AutoLearnUseCase
    from ..infrastructure.feedback import FeedbackCollector

    console = Console()

    collector = FeedbackCollector()
    auto_learn = AutoLearnUseCase(feedback_collector=collector)

    console.print("[bold]🧠 피드백 분석 중...[/bold]")
    result = auto_learn.analyze_feedback()

    console.print(auto_learn.format_suggestions(result))

    return 0


def cmd_synonym(args) -> int:
    """Execute synonym management commands."""
    from .cli import cmd_synonym as _cmd_synonym

    return _cmd_synonym(args)


def cmd_quality(args) -> int:
    """Execute quality command - RAGAS-based RAG quality evaluation."""
    from rich.console import Console

    from ..domain.evaluation import RAGQualityEvaluator
    from ..domain.evaluation.personas import PersonaManager
    from ..domain.evaluation.synthetic_data import SyntheticDataGenerator
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..infrastructure.storage.evaluation_store import EvaluationStore

    console = Console()

    # Initialize components
    evaluator = RAGQualityEvaluator(
        judge_model=args.judge_model,
        use_ragas=not args.no_ragas,
    )
    store = EvaluationStore(storage_dir=args.output_dir)
    persona_mgr = PersonaManager()
    loader = JSONDocumentLoader()

    # Initialize RAG system for answer generation
    from ..application.search_usecase import SearchUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.llm_adapter import LLMClientAdapter

    vector_store = ChromaVectorStore(persist_directory=args.db_path)
    _, provider, model, base_url = _get_default_llm_settings()

    # For quality evaluation, prefer local Ollama for reliability
    # Override with env var if explicitly set for evaluation
    import os

    eval_provider = os.getenv("EVAL_LLM_PROVIDER", "ollama")
    eval_model = os.getenv("EVAL_LLM_MODEL", "llama3.2:latest")
    eval_base_url = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:11434")

    console.print(
        f"[dim]Using LLM: {eval_provider} ({eval_model}) at {eval_base_url}[/dim]"
    )

    llm_client = LLMClientAdapter(
        provider=eval_provider, model=eval_model, base_url=eval_base_url
    )
    search_usecase = SearchUseCase(
        store=vector_store, llm_client=llm_client, use_reranker=True
    )

    # Subcommand handling
    if args.quality_cmd == "baseline":
        console.print("[bold]🔍 기준선 평가 시작...[/bold]")
        results = []

        for persona_id in persona_mgr.list_personas():
            console.print(f"[dim]페르소나 {persona_id} 테스트 중...[/dim]")
            queries = persona_mgr.generate_queries(
                persona_id,
                count=args.queries_per_persona,
                topics=[args.topic] if args.topic else None,
            )

            for query in queries:
                try:
                    # RAG 시스템 실행
                    search_results = search_usecase.search(
                        query_text=query,
                        top_k=args.top_k,
                    )
                    contexts = (
                        [r.chunk.text for r in search_results] if search_results else []
                    )

                    # 검색 결과가 없는 경우 스킵
                    if not contexts:
                        console.print(
                            f"[yellow]  ⚠ 검색 결과 없음: {query[:40]}...[/yellow]"
                        )
                        continue

                    # 답변 생성
                    from ..infrastructure.tool_executor import ToolExecutor

                    tool_executor = ToolExecutor(
                        search_usecase=search_usecase,
                        llm_client=llm_client,
                    )

                    # LLM 클라이언트 확인
                    if not llm_client:
                        console.print("[red]  ❌ LLM 클라이언트 초기화 실패[/red]")
                        continue

                    answer = tool_executor._handle_generate_answer(
                        {"question": query, "context": "\n\n".join(contexts)}
                    )

                    # 응답이 비어있는 경우 처리
                    if not answer or answer.strip() == "":
                        console.print(
                            f"[yellow]  ⚠ 응답 생성 실패: {query[:40]}...[/yellow]"
                        )
                        continue

                    # 평가 실행
                    result = evaluator.evaluate_single_turn(query, contexts, answer)
                    result.persona = persona_id
                    results.append(result)
                    store.save_evaluation(result)

                    console.print(
                        f"[dim]  Query: {query[:40]}... Score: {result.overall_score:.2f}[/dim]"
                    )
                except Exception as e:
                    console.print(f"[red]평가 실패: {e}[/red]")
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()[:200]}[/dim]")
                    continue

        # 통계 출력
        stats = store.get_statistics()
        console.print("\n[bold]기준선 평가 결과[/bold]")
        console.print(f"전체 평가: {stats.total_evaluations}")
        console.print(f"평균 점수: {stats.avg_overall_score:.2f}")
        console.print(f"합격률: {stats.pass_rate:.1%}")
        console.print(f"추세: {stats.trend}")
        console.print("\n[bold]메트릭별 점수:[/bold]")
        console.print(f"  Faithfulness: {stats.avg_faithfulness:.2f}")
        console.print(f"  Answer Relevancy: {stats.avg_answer_relevancy:.2f}")
        console.print(f"  Contextual Precision: {stats.avg_contextual_precision:.2f}")
        console.print(f"  Contextual Recall: {stats.avg_contextual_recall:.2f}")

    elif args.quality_cmd == "persona":
        console.print(f"[bold]🔍 페르소나 {args.id} 테스트 시작...[/bold]")
        queries = persona_mgr.generate_queries(
            args.id, count=args.count, topics=[args.topic] if args.topic else None
        )
        console.print(f"[dim]{len(queries)}개 쿼리 생성 완료[/dim]")

        for query in queries:
            try:
                search_results = search_usecase.search(
                    query_text=query, top_k=args.top_k
                )
                contexts = (
                    [r.chunk.text for r in search_results] if search_results else []
                )

                from ..infrastructure.tool_executor import ToolExecutor

                tool_executor = ToolExecutor(
                    search_usecase=search_usecase,
                    llm_client=llm_client,
                )
                answer = tool_executor._handle_generate_answer(
                    {"question": query, "context": "\n\n".join(contexts)}
                )

                result = evaluator.evaluate_single_turn(query, contexts, answer)
                result.persona = args.id
                store.save_evaluation(result)

                console.print(
                    f"Score: {result.overall_score:.2f} | Query: {query[:50]}..."
                )
            except Exception as e:
                console.print(f"[red]평가 실패: {e}[/red]")

    elif args.quality_cmd == "synthetic":
        console.print("[bold]📝 합성 테스트 데이터 생성 시작...[/bold]")
        generator = SyntheticDataGenerator(loader)

        if args.scenarios:
            scenarios = generator.generate_scenarios_from_regulations(
                regulation=args.regulation, num_scenarios=args.count
            )
            console.print(f"[green]✅ {len(scenarios)}개 시나리오 생성 완료[/green]")
        else:
            queries = generator.generate_queries_from_documents(
                num_questions=args.count, difficulty=args.difficulty
            )
            console.print(f"[green]✅ {len(queries)}개 질문 생성 완료[/green]")

    elif args.quality_cmd == "stats":
        if args.days:
            stats = store.get_statistics(days=args.days)
        else:
            stats = store.get_statistics()

        console.print("\n[bold]평가 통계[/bold]")
        console.print(f"전체 평가: {stats.total_evaluations}")
        console.print(f"평균 점수: {stats.avg_overall_score:.2f}")
        console.print(f"합격률: {stats.pass_rate:.1%}")
        console.print(f"최저 점수: {stats.min_score:.2f}")
        console.print(f"최고 점수: {stats.max_score:.2f}")
        console.print(f"표준 편차: {stats.std_deviation:.2f}")
        console.print(f"추세: {stats.trend}")
        console.print("\n[bold]메트릭별 평균:[/bold]")
        console.print(f"  Faithfulness: {stats.avg_faithfulness:.2f}")
        console.print(f"  Answer Relevancy: {stats.avg_answer_relevancy:.2f}")
        console.print(f"  Contextual Precision: {stats.avg_contextual_precision:.2f}")
        console.print(f"  Contextual Recall: {stats.avg_contextual_recall:.2f}")

    elif args.quality_cmd == "dashboard":
        console.print("[bold]🚀 Gradio 품질 대시보드 시작...[/bold]")

        from .web.quality_dashboard import app as quality_app

        quality_app.launch(
            server_port=7861,
            share=False,
            show_error=True,
        )

    # P2: quality run - 전체 평가 실행
    elif args.quality_cmd == "run":
        return _cmd_quality_run(args, console, evaluator, store, persona_mgr, search_usecase, llm_client)

    # P2: quality resume - 중단된 세션 재개
    elif args.quality_cmd == "resume":
        return _cmd_quality_resume(args, console, store)

    # P2: quality generate-spec - 실패 패턴에서 SPEC 생성
    elif args.quality_cmd == "generate-spec":
        return _cmd_quality_generate_spec(args, console, store)

    # P2: quality status - 세션 상태 확인
    elif args.quality_cmd == "status":
        return _cmd_quality_status(args, console, store)

    return 0


def _cmd_quality_run(args, console, evaluator, store, persona_mgr, search_usecase, llm_client) -> int:
    """Execute quality run command - batch evaluation with progress tracking."""
    from ..application.evaluation import (
        CheckpointManager,
        ProgressReporter,
    )

    checkpoint_dir = "data/checkpoints"
    if not args.no_checkpoint:
        checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
    else:
        checkpoint_mgr = None

    # Create session or resume existing
    session_id = args.session_id
    if session_id:
        progress_data = checkpoint_mgr.load_checkpoint(session_id) if checkpoint_mgr else None
        if progress_data is None:
            console.print(f"[red]세션 {session_id}을(를) 찾을 수 없습니다.[/red]")
            return 1
        console.print(f"[bold]세션 {session_id} 재개 중...[/bold]")
    else:
        # Create new session
        import uuid
        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        personas = list(args.personas) if args.personas else persona_mgr.list_personas()
        total_queries = len(personas) * args.queries_per_persona

        if checkpoint_mgr:
            checkpoint_mgr.create_session(
                session_id=session_id,
                total_queries=total_queries,
                personas=personas,
            )
        console.print(f"[bold]새 평가 세션 시작: {session_id}[/bold]")
        console.print(f"[dim]페르소나: {', '.join(personas)}[/dim]")
        console.print(f"[dim]총 쿼리 수: {total_queries}[/dim]")

    # Initialize progress reporter
    personas_for_progress = list(args.personas) if args.personas else persona_mgr.list_personas()
    persona_counts = {p: args.queries_per_persona for p in personas_for_progress}
    total_for_reporter = sum(persona_counts.values())
    reporter = ProgressReporter(total_queries=total_for_reporter, persona_counts=persona_counts)

    # Initialize batch executor
    # Note: BatchEvaluationExecutor requires an evaluator callable
    # For now, we'll skip the batch executor and process queries directly
    # batch_executor = BatchEvaluationExecutor(
    #     evaluator=evaluator.evaluate_single_turn,
    #     batch_size=args.batch_size,
    # )

    # Run evaluation
    results = []
    personas = list(args.personas) if args.personas else persona_mgr.list_personas()

    try:
        for persona_id in personas:
            queries = persona_mgr.generate_queries(
                persona_id,
                count=args.queries_per_persona,
            )

            for i, query in enumerate(queries):
                try:
                    # Search
                    search_results = search_usecase.search(query_text=query, top_k=5)
                    contexts = [r.chunk.text for r in search_results] if search_results else []

                    if not contexts:
                        console.print(f"[yellow]  ⚠ 검색 결과 없음: {query[:40]}...[/yellow]")
                        continue

                    # Generate answer
                    from ..infrastructure.tool_executor import ToolExecutor
                    tool_executor = ToolExecutor(
                        search_usecase=search_usecase,
                        llm_client=llm_client,
                    )
                    answer = tool_executor._handle_generate_answer(
                        {"question": query, "context": "\n\n".join(contexts)}
                    )

                    if not answer or answer.strip() == "":
                        continue

                    # Evaluate
                    result = evaluator.evaluate_single_turn(query, contexts, answer)
                    result.persona = persona_id
                    results.append(result)
                    store.save_evaluation(result)

                    # Update progress
                    reporter.update(
                        completed=1,
                        persona=persona_id,
                        query=query,
                        score=result.overall_score,
                    )

                    # Save checkpoint
                    if checkpoint_mgr and not args.no_checkpoint:
                        checkpoint_mgr.update_progress(
                            session_id=session_id,
                            persona=persona_id,
                            query_id=f"q_{i}",
                            result={"score": result.overall_score, "query": query},
                        )

                    # Show progress
                    progress_info = reporter.get_progress()
                    eta = reporter.get_eta()
                    console.print(
                        f"[dim]  [{progress_info.completed}/{progress_info.total}] "
                        f"Score: {result.overall_score:.2f} | ETA: {eta:.0f}s | {query[:30]}...[/dim]"
                    )

                except Exception as e:
                    console.print(f"[red]평가 실패: {e}[/red]")
                    if checkpoint_mgr and not args.no_checkpoint:
                        checkpoint_mgr.update_progress(
                            session_id=session_id,
                            persona=persona_id,
                            query_id=f"q_{i}",
                            error=str(e),
                        )

    except KeyboardInterrupt:
        console.print("\n[yellow]평가가 중단되었습니다.[/yellow]")
        if checkpoint_mgr and not args.no_checkpoint:
            checkpoint_mgr.pause_session(session_id)
            console.print(f"[yellow]세션 저장됨: {session_id}[/yellow]")
            console.print(f"[yellow]재개 명령: regulation quality resume -s {session_id}[/yellow]")
        return 130

    # Final statistics
    stats = store.get_statistics()
    console.print("\n[bold green]✅ 평가 완료![/bold green]")
    console.print(f"세션 ID: {session_id}")
    console.print(f"평가된 쿼리: {len(results)}")
    console.print(f"평균 점수: {stats.avg_overall_score:.2f}")
    console.print(f"합격률: {stats.pass_rate:.1%}")

    # Save report
    if args.output:
        import json
        report = {
            "session_id": session_id,
            "total_queries": len(results),
            "stats": {
                "avg_score": stats.avg_overall_score,
                "pass_rate": stats.pass_rate,
                "avg_faithfulness": stats.avg_faithfulness,
                "avg_answer_relevancy": stats.avg_answer_relevancy,
                "avg_contextual_precision": stats.avg_contextual_precision,
                "avg_contextual_recall": stats.avg_contextual_recall,
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]보고서 저장됨: {args.output}[/dim]")

    return 0


def _cmd_quality_resume(args, console, store) -> int:
    """Execute quality resume command - resume interrupted session."""
    from ..application.evaluation import CheckpointManager, ResumeController

    checkpoint_dir = "data/checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
    resume_ctrl = ResumeController(checkpoint_manager=checkpoint_mgr)

    # List sessions if requested
    if args.list:
        sessions = resume_ctrl.find_interrupted_sessions()
        if not sessions:
            console.print("[yellow]재개 가능한 세션이 없습니다.[/yellow]")
            return 0

        console.print("[bold]재개 가능한 세션:[/bold]")
        for session in sessions:
            console.print(
                f"  - {session['session_id']}: "
                f"{session['completion_rate']:.0f}% 완료, "
                f"업데이트: {session['updated_at']}"
            )
        return 0

    # Get session to resume
    session_id = args.session_id
    if not session_id:
        session_id = resume_ctrl.get_resume_recommendation()
        if not session_id:
            console.print("[yellow]재개할 세션이 없습니다.[/yellow]")
            return 1
        console.print(f"[dim]가장 최근 중단 세션 선택: {session_id}[/dim]")

    # Check if can resume
    can_resume, reason = resume_ctrl.can_resume(session_id)
    if not can_resume:
        console.print(f"[red]세션 {session_id}을(를) 재개할 수 없습니다: {reason}[/red]")
        return 1

    # Get resume context
    context = resume_ctrl.get_resume_context(session_id)
    if not context:
        console.print(f"[red]세션 {session_id}의 컨텍스트를 가져올 수 없습니다.[/red]")
        return 1

    console.print(f"[bold]세션 {session_id} 재개 정보:[/bold]")
    console.print(f"  완료율: {context.completion_rate:.1f}%")
    console.print(f"  완료된 쿼리: {context.completed_count}/{context.total_count}")
    console.print(f"  실패한 쿼리: {context.failed_count}")
    console.print(f"  남은 페르소나: {', '.join(context.remaining_personas) or '없음'}")

    console.print("\n[green]세션 재개 명령:[/green]")
    console.print(f"  regulation quality run -s {session_id}")

    return 0


def _cmd_quality_generate_spec(args, console, store) -> int:
    """Execute quality generate-spec command - generate SPEC from failures."""
    from ..domain.evaluation import (
        FailureClassifier,
        RecommendationEngine,
        SPECGenerator,
    )
    from ..infrastructure.storage.evaluation_store import EvaluationStore

    # Get evaluations from store
    eval_store = EvaluationStore(storage_dir=args.output_dir if hasattr(args, 'output_dir') else "data/evaluations")

    # Get recent evaluations below threshold
    evaluations = eval_store.get_evaluations(
        max_score=args.threshold,
        limit=100,
    )

    if not evaluations:
        console.print(f"[yellow]임계값 {args.threshold} 미만의 실패한 평가가 없습니다.[/yellow]")
        return 0

    console.print(f"[bold]분석 중: {len(evaluations)}개 실패 평가[/bold]")

    # Classify failures
    classifier = FailureClassifier()
    failure_summaries = classifier.classify_batch(evaluations)

    console.print("\n[bold]실패 유형 분석:[/bold]")
    for summary in failure_summaries:
        console.print(
            f"  - {summary.failure_type.value}: {summary.count}건 "
            f"(평균 점수: {summary.avg_score:.2f})"
        )

    # Generate recommendations
    engine = RecommendationEngine()
    failure_counts = {s.failure_type: s.count for s in failure_summaries}
    recommendations = engine.generate_recommendations(failure_counts, threshold=1)

    console.print(f"\n[bold]생성된 권장사항: {len(recommendations)}개[/bold]")

    # Generate SPEC
    spec_generator = SPECGenerator()
    spec = spec_generator.generate_spec(
        failures=failure_summaries,
        recommendations=recommendations,
    )

    # Output SPEC
    if args.output:
        spec_path = spec_generator.save_spec(spec, path=args.output)
        console.print(f"\n[green]✅ SPEC 문서 생성 완료: {spec_path}[/green]")
    else:
        # Print to console
        console.print("\n" + "=" * 60)
        console.print(spec.to_markdown())
        console.print("=" * 60)

    # Show action plan
    if recommendations:
        plan = engine.get_action_plan(recommendations)
        console.print("\n[bold]액션 플랜:[/bold]")
        console.print(f"  즉시 조치: {len(plan['immediate_actions'])}개")
        console.print(f"  단기 조치: {len(plan['short_term_actions'])}개")
        console.print(f"  장기 조치: {len(plan['long_term_actions'])}개")

    return 0


def _cmd_quality_status(args, console, store) -> int:
    """Execute quality status command - check session status."""
    from ..application.evaluation import CheckpointManager

    checkpoint_dir = "data/checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)

    # Cleanup if requested
    if args.cleanup:
        cleaned = checkpoint_mgr.cleanup_completed_sessions(keep_days=7)
        console.print(f"[green]정리된 세션: {cleaned}개[/green]")
        return 0

    # Show specific session
    if args.session_id:
        progress = checkpoint_mgr.load_checkpoint(args.session_id)
        if not progress:
            console.print(f"[red]세션 {args.session_id}을(를) 찾을 수 없습니다.[/red]")
            return 1

        console.print(f"[bold]세션: {progress.session_id}[/bold]")
        console.print(f"  상태: {progress.status}")
        console.print(f"  시작: {progress.started_at}")
        console.print(f"  업데이트: {progress.updated_at}")
        console.print(f"  진행률: {progress.completed_queries}/{progress.total_queries}")
        console.print(f"  완료율: {progress.completion_rate:.1f}%")

        console.print("\n[bold]페르소나별 진행:[/bold]")
        for persona, persona_progress in progress.personas.items():
            console.print(
                f"  - {persona}: {persona_progress.completed_queries}/{persona_progress.total_queries} "
                f"(실패: {persona_progress.failed_queries})"
            )
        return 0

    # Show all sessions
    sessions = checkpoint_mgr.list_sessions()

    if not sessions:
        console.print("[yellow]저장된 세션이 없습니다.[/yellow]")
        return 0

    if not args.all:
        # Show only recent/active sessions
        sessions = [s for s in sessions if s.get("status") != "completed"][:5]

    console.print(f"[bold]평가 세션 ({len(sessions)}개):[/bold]\n")

    for session in sessions:
        status_color = {
            "running": "green",
            "paused": "yellow",
            "completed": "blue",
            "failed": "red",
        }.get(session.get("status"), "white")

        console.print(
            f"  [{status_color}]{session['session_id']}[/{status_color}] "
            f"- {session['status']} "
            f"- {session['completion_rate']:.0f}% "
            f"- {session['updated_at']}"
        )

    return 0


# =============================================================================
# Entry Point
# =============================================================================


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Global debug flag handling
    if hasattr(args, "global_debug") and args.global_debug:
        args.debug = True

    # 커맨드 없이 실행하면 interactive 모드로 시작
    if not args.command:
        # 기본값 설정
        args.command = "search"
        args.query = None
        args.interactive = True
        args.top_k = 5
        args.include_abolished = False
        args.db_path = "data/chroma_db"
        args.no_rerank = False
        args.verbose = False
        args.debug = False
        args.feedback = False
        args.answer = False
        args.quick = False
        args.show_sources = False
        # LLM 기본값
        providers, provider, model, base_url = _get_default_llm_settings()
        args.provider = provider
        args.model = model
        args.base_url = base_url
        return cmd_search(args)

    commands = {
        "convert": cmd_convert,
        "sync": cmd_sync,
        "search": cmd_search,
        "ask": cmd_ask,
        "status": cmd_status,
        "reset": cmd_reset,
        "reparse": cmd_reparse,
        "serve": cmd_serve,
        "evaluate": cmd_evaluate,
        "extract-keywords": cmd_extract_keywords,
        "feedback": cmd_feedback,
        "analyze": cmd_analyze,
        "synonym": cmd_synonym,
        "quality": cmd_quality,
    }

    if args.command in commands:
        try:
            return commands[args.command](args)
        except KeyboardInterrupt:
            print("\nAborted.")
            return 130

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
