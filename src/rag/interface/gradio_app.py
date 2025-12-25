"""
Gradio Web UI for Regulation RAG System.

Provides a user-friendly web interface for:
- Searching regulations
- Asking questions with LLM-generated answers
- Viewing sync status

Usage:
    uv run python -m src.rag.interface.gradio_app
"""

import os
from typing import List, Optional, Tuple

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from ..infrastructure.chroma_store import ChromaVectorStore
from ..infrastructure.json_loader import JSONDocumentLoader
from ..infrastructure.llm_client import OpenAIClient, MockLLMClient
from ..application.sync_usecase import SyncUseCase
from ..application.search_usecase import SearchUseCase
from ..domain.value_objects import SearchFilter
from ..domain.entities import RegulationStatus


# Default paths
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_JSON_PATH = "data/output/규정집-test01.json"


def create_app(
    db_path: str = DEFAULT_DB_PATH,
    use_mock_llm: bool = False,
) -> "gr.Blocks":
    """
    Create Gradio app instance.

    Args:
        db_path: Path to ChromaDB storage.
        use_mock_llm: Use mock LLM for testing without API key.

    Returns:
        Gradio Blocks app.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("gradio is required. Install with: uv add gradio")

    # Initialize components
    store = ChromaVectorStore(persist_directory=db_path)
    loader = JSONDocumentLoader()

    # LLM client (use mock if no API key)
    llm_client = None
    llm_status = "❌ LLM 비활성 (OPENAI_API_KEY 미설정)"

    if not use_mock_llm and os.getenv("OPENAI_API_KEY"):
        try:
            llm_client = OpenAIClient()
            llm_status = "✅ OpenAI GPT-4o-mini 연결됨"
        except Exception as e:
            llm_status = f"❌ OpenAI 연결 실패: {e}"
    elif use_mock_llm:
        llm_client = MockLLMClient()
        llm_status = "⚠️ Mock LLM (테스트 모드)"

    search_usecase = SearchUseCase(store, llm_client)
    sync_usecase = SyncUseCase(loader, store)

    # Get initial status
    def get_status_text() -> str:
        status = sync_usecase.get_sync_status()
        return f"""**동기화 상태**
- 마지막 동기화: {status['last_sync'] or '없음'}
- JSON 파일: {status['json_file'] or '없음'}
- 인덱싱된 규정: {status['store_regulations']}개
- 청크 수: {status['store_chunks']}개
- LLM: {llm_status}
"""

    # Search function
    def search_regulations(
        query: str,
        top_k: int,
        include_abolished: bool,
    ) -> Tuple[str, str]:
        """Execute search and return formatted results."""
        if not query.strip():
            return "검색어를 입력해주세요.", ""

        if store.count() == 0:
            return "데이터베이스가 비어 있습니다. 먼저 동기화를 실행하세요.", ""

        results = search_usecase.search(
            query,
            top_k=top_k,
            include_abolished=include_abolished,
        )

        if not results:
            return "검색 결과가 없습니다.", ""

        # Format results as markdown table
        table_rows = ["| # | 규정 | 조항 | 점수 |", "|---|------|------|------|"]
        for i, r in enumerate(results, 1):
            path = " > ".join(r.chunk.parent_path[-2:]) if r.chunk.parent_path else r.chunk.title
            table_rows.append(f"| {i} | {r.chunk.rule_code} | {path[:30]} | {r.score:.2f} |")

        table = "\n".join(table_rows)

        # Top result detail
        top = results[0]
        detail = f"""### 1위 결과: {top.chunk.rule_code}
**경로:** {' > '.join(top.chunk.parent_path)}

{top.chunk.text[:500]}{'...' if len(top.chunk.text) > 500 else ''}
"""

        return table, detail

    # Ask function (with LLM)
    def ask_question(
        question: str,
        top_k: int,
        include_abolished: bool,
    ) -> Tuple[str, str]:
        """Ask question and get LLM answer."""
        if not question.strip():
            return "질문을 입력해주세요.", ""

        if not llm_client:
            return "LLM이 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정하세요.", ""

        if store.count() == 0:
            return "데이터베이스가 비어 있습니다.", ""

        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        answer = search_usecase.ask(
            question,
            filter=filter,
            top_k=top_k,
            include_abolished=include_abolished,
        )

        # Format sources
        sources = []
        for i, r in enumerate(answer.sources, 1):
            path = " > ".join(r.chunk.parent_path[-2:]) if r.chunk.parent_path else ""
            sources.append(f"{i}. [{r.chunk.rule_code}] {path}")

        sources_text = "\n".join(sources) if sources else "출처 없음"

        return answer.text, f"### 참고 규정\n{sources_text}\n\n*신뢰도: {answer.confidence:.0%}*"

    # Sync function
    def run_sync(json_path: str, full_sync: bool) -> str:
        """Run synchronization."""
        if not json_path.strip():
            return "JSON 파일 경로를 입력해주세요."

        try:
            if full_sync:
                result = sync_usecase.full_sync(json_path)
            else:
                result = sync_usecase.incremental_sync(json_path)

            if result.has_errors:
                return f"❌ 오류 발생:\n" + "\n".join(result.errors)

            return f"""✅ 동기화 완료!
- 추가: {result.added}개
- 수정: {result.modified}개
- 삭제: {result.removed}개
- 변경없음: {result.unchanged}개
- 총 청크: {store.count()}개
"""
        except Exception as e:
            return f"❌ 오류: {str(e)}"

    # Build UI
    with gr.Blocks(
        title="📚 대학 규정집 Q&A",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 📚 대학 규정집 Q&A 시스템")

        with gr.Tabs():
            # Tab 1: Search
            with gr.TabItem("🔍 검색"):
                with gr.Row():
                    with gr.Column(scale=3):
                        search_query = gr.Textbox(
                            label="검색어",
                            placeholder="예: 교원 연구년 자격",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        search_top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="결과 수",
                        )
                        search_abolished = gr.Checkbox(
                            label="폐지 규정 포함",
                            value=False,
                        )

                search_btn = gr.Button("검색", variant="primary")

                search_results = gr.Markdown(label="검색 결과")
                search_detail = gr.Markdown(label="상세 내용")

                search_btn.click(
                    fn=search_regulations,
                    inputs=[search_query, search_top_k, search_abolished],
                    outputs=[search_results, search_detail],
                )

            # Tab 2: Ask (Q&A)
            with gr.TabItem("💬 질문하기"):
                with gr.Row():
                    with gr.Column(scale=3):
                        ask_question_input = gr.Textbox(
                            label="질문",
                            placeholder="예: 교원 연구년 신청 자격은 무엇인가요?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        ask_top_k = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="참고 규정 수",
                        )
                        ask_abolished = gr.Checkbox(
                            label="폐지 규정 포함",
                            value=False,
                        )

                ask_btn = gr.Button("질문하기", variant="primary")

                ask_answer = gr.Markdown(label="답변")
                ask_sources = gr.Markdown(label="참고 규정")

                ask_btn.click(
                    fn=ask_question,
                    inputs=[ask_question_input, ask_top_k, ask_abolished],
                    outputs=[ask_answer, ask_sources],
                )

            # Tab 3: Sync
            with gr.TabItem("⚙️ 설정"):
                gr.Markdown(get_status_text())

                with gr.Row():
                    sync_json_path = gr.Textbox(
                        label="JSON 파일 경로",
                        value=DEFAULT_JSON_PATH,
                    )
                    sync_full = gr.Checkbox(
                        label="전체 동기화",
                        value=False,
                    )

                sync_btn = gr.Button("동기화 실행", variant="secondary")
                sync_result = gr.Markdown(label="결과")

                sync_btn.click(
                    fn=run_sync,
                    inputs=[sync_json_path, sync_full],
                    outputs=[sync_result],
                )

    return app


def main():
    """Launch Gradio app."""
    import argparse

    parser = argparse.ArgumentParser(description="규정집 RAG 웹 UI")
    parser.add_argument("--port", type=int, default=7860, help="서버 포트")
    parser.add_argument("--share", action="store_true", help="공개 링크 생성")
    parser.add_argument("--mock-llm", action="store_true", help="Mock LLM 사용")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="DB 경로")

    args = parser.parse_args()

    app = create_app(db_path=args.db_path, use_mock_llm=args.mock_llm)
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
