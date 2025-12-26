"""
Gradio Web UI for Regulation RAG System.

Provides a user-friendly web interface for:
- Searching regulations
- Asking questions with LLM-generated answers
- Viewing sync status

Usage:
    uv run python -m src.rag.interface.gradio_app
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from ..infrastructure.chroma_store import ChromaVectorStore
from ..infrastructure.json_loader import JSONDocumentLoader
from ...main import run_pipeline
from ..infrastructure.llm_adapter import LLMClientAdapter
from ..infrastructure.llm_client import MockLLMClient
from ..application.sync_usecase import SyncUseCase
from ..application.search_usecase import SearchUseCase
from ..domain.value_objects import SearchFilter
from ..domain.entities import RegulationStatus


# Default paths
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_JSON_PATH = "data/output/규정집-test01.json"
LLM_PROVIDERS = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "ollama"
if DEFAULT_LLM_PROVIDER not in LLM_PROVIDERS:
    DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or ""
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""


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

    llm_status = "ℹ️ 질문 탭에서 LLM 설정을 선택하세요."
    if use_mock_llm:
        llm_status = "⚠️ Mock LLM (테스트 모드)"

    search_usecase = SearchUseCase(store)
    sync_usecase = SyncUseCase(loader, store)

    data_input_dir = Path("data/input")
    data_output_dir = Path("data/output")
    data_input_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    def _find_latest_json(output_dir: Path) -> Optional[Path]:
        json_files = [
            p for p in output_dir.rglob("*.json")
            if not p.name.endswith("_metadata.json")
        ]
        if not json_files:
            return None
        return max(json_files, key=lambda p: p.stat().st_mtime)

    def _list_json_files(output_dir: Path) -> List[Path]:
        return sorted(
            [
                p for p in output_dir.rglob("*.json")
                if not p.name.endswith("_metadata.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _list_hwp_files(input_dir: Path) -> List[Path]:
        return sorted(input_dir.rglob("*.hwp"))

    def _render_status(target_db_path: str) -> str:
        db_path_value = target_db_path or db_path
        try:
            store_local = ChromaVectorStore(persist_directory=db_path_value)
        except Exception as e:
            return f"❌ DB 초기화 실패: {e}"

        sync_state_path = Path("data/sync_state.json")
        last_synced = None
        if sync_state_path.exists():
            try:
                import json
                data = json.loads(sync_state_path.read_text(encoding="utf-8"))
                last_synced = data.get("json_file")
            except Exception:
                last_synced = None

        json_files = _list_json_files(data_output_dir)
        hwp_files = _list_hwp_files(data_input_dir)
        json_by_stem = {p.stem: p for p in json_files}

        lines = []
        lines.append("## DB 상태")
        lines.append(f"- DB 경로: `{db_path_value}`")
        lines.append(f"- 청크 수: {store_local.count()}")
        lines.append(f"- 규정 수: {len(store_local.get_all_rule_codes())}")
        if last_synced:
            lines.append(f"- 마지막 동기화 JSON: `{last_synced}`")

        lines.append("\n## JSON 파일 목록 (`data/output`)")
        if json_files:
            lines.append("| 파일 | 수정 시각 | 크기 | 마지막 동기화 |")
            lines.append("|---|---|---|---|")
            for p in json_files:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size_kb = f"{p.stat().st_size / 1024:.1f} KB"
                is_synced = "✅" if last_synced and p.name == last_synced else ""
                lines.append(f"| `{p.name}` | {mtime} | {size_kb} | {is_synced} |")
        else:
            lines.append("- JSON 파일이 없습니다.")

        lines.append("\n## HWP 파일 목록 (`data/input`)")
        if hwp_files:
            lines.append("| 파일 | 변환 여부 | 대응 JSON |")
            lines.append("|---|---|---|")
            for p in hwp_files:
                json_path = json_by_stem.get(p.stem)
                converted = "✅" if json_path else "❌"
                json_name = f"`{json_path.name}`" if json_path else "-"
                lines.append(f"| `{p.name}` | {converted} | {json_name} |")
        else:
            lines.append("- HWP 파일이 없습니다.")

        return "\n".join(lines)

    def _json_choices() -> List[str]:
        return [str(p) for p in _list_json_files(data_output_dir)]

    auto_sync_message = ""
    if store.count() == 0:
        latest_json = _find_latest_json(data_output_dir)
        if latest_json:
            try:
                result = sync_usecase.incremental_sync(str(latest_json))
                auto_sync_message = f"자동 동기화: {latest_json.name} ({result})"
            except Exception as e:
                auto_sync_message = f"자동 동기화 실패: {e}"

    # Get initial status
    def get_status_text() -> str:
        status = sync_usecase.get_sync_status()
        auto_sync_note = f"\n- {auto_sync_message}" if auto_sync_message else ""
        return f"""**동기화 상태**
- 마지막 동기화: {status['last_sync'] or '없음'}
- JSON 파일: {status['json_file'] or '없음'}
- 인덱싱된 규정: {status['store_regulations']}개
- 청크 수: {status['store_chunks']}개
- LLM: {llm_status}{auto_sync_note}
"""

    def _persist_upload(file_path: str) -> Path:
        input_path = Path(file_path)
        data_input_dir = Path("data/input")
        data_input_dir.mkdir(parents=True, exist_ok=True)
        target_path = data_input_dir / input_path.name
        if input_path.resolve() != target_path.resolve():
            shutil.copy2(input_path, target_path)
        return target_path

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
        table_rows = ["| # | 규정명 | 조항 | 점수 |", "|---|------|------|------|"]
        for i, r in enumerate(results, 1):
            path = " > ".join(r.chunk.parent_path[-2:]) if r.chunk.parent_path else r.chunk.title
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.rule_code
            table_rows.append(f"| {i} | {reg_title} | {path[:30]} | {r.score:.2f} |")

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
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
    ) -> Tuple[str, str]:
        """Ask question and get LLM answer."""
        if not question.strip():
            return "질문을 입력해주세요.", ""

        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)

        if use_mock_llm:
            llm_client = MockLLMClient()
        else:
            try:
                llm_client = LLMClientAdapter(
                    provider=llm_provider,
                    model=llm_model or None,
                    base_url=llm_base_url or None,
                )
            except Exception as e:
                return f"LLM 초기화 실패: {e}", ""

        if store_for_ask.count() == 0:
            return "데이터베이스가 비어 있습니다.", ""

        search_with_llm = SearchUseCase(store_for_ask, llm_client)

        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        answer = search_with_llm.ask(
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

    def run_conversion_and_sync(
        hwp_file: str,
        use_llm: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        output_dir: str,
        target_db_path: str,
        full_sync: bool,
    ) -> Tuple[str, str, str]:
        if not hwp_file:
            return "HWP 파일을 업로드하세요.", "", ""

        output_dir_value = output_dir or "data/output"
        db_path_value = target_db_path or db_path

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        input_path = _persist_upload(hwp_file)

        args = argparse.Namespace(
            input_path=str(input_path),
            output_dir=output_dir_value,
            use_llm=use_llm,
            provider=llm_provider,
            model=llm_model or None,
            base_url=llm_base_url or None,
            allow_llm_fallback=True,
            force=False,
            cache_dir=".cache",
            verbose=True,
            enhance_rag=True,
        )

        from rich.console import Console
        console = Console(record=True)
        status = run_pipeline(args, console=console)
        log_text = console.export_text() or ""

        if status != 0:
            return log_text or "변환 실패", "", ""

        json_path = Path(output_dir_value) / f"{input_path.stem}.json"
        if not json_path.exists():
            return f"{log_text}\nJSON 파일을 찾을 수 없습니다: {json_path}", "", ""

        store_local = ChromaVectorStore(persist_directory=db_path_value)
        loader_local = JSONDocumentLoader()
        sync_local = SyncUseCase(loader_local, store_local)
        if full_sync:
            sync_result = sync_local.full_sync(str(json_path))
        else:
            sync_result = sync_local.incremental_sync(str(json_path))

        sync_lines = [str(sync_result), f"총 청크 수: {store_local.count()}"]
        if sync_result.has_errors:
            sync_lines.extend(sync_result.errors)

        status_text = "\n".join([log_text, "[SYNC]", *sync_lines]).strip()
        return status_text, str(json_path), db_path_value

    # Build UI
    with gr.Blocks(
        title="📚 대학 규정집 Q&A",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 📚 대학 규정집 Q&A 시스템")

        with gr.Tabs():
            # Tab 0: All-in-one
            with gr.TabItem("🧩 올인원"):
                gr.Markdown("HWP 업로드 → JSON 변환 → DB 동기화 → 질문까지 한 번에 진행합니다.")

                hwp_file = gr.File(
                    label="HWP 파일 업로드",
                    file_types=[".hwp"],
                    type="filepath",
                )
                use_llm_preprocess = gr.Checkbox(
                    label="LLM 전처리 사용 (문서 품질 낮은 경우 추천)",
                    value=False,
                )

                with gr.Accordion("LLM 설정", open=False):
                    llm_provider_easy = gr.Dropdown(
                        choices=LLM_PROVIDERS,
                        value=DEFAULT_LLM_PROVIDER,
                        label="프로바이더",
                    )
                    llm_model_easy = gr.Textbox(
                        value=DEFAULT_LLM_MODEL,
                        label="모델 (선택)",
                    )
                    llm_base_url_easy = gr.Textbox(
                        value=DEFAULT_LLM_BASE_URL,
                        label="Base URL (로컬용)",
                        placeholder="예: http://127.0.0.1:11434",
                    )

                with gr.Accordion("고급 설정", open=False):
                    output_dir = gr.Textbox(
                        value="data/output",
                        label="출력 폴더",
                    )
                    db_path_input = gr.Textbox(
                        value=db_path,
                        label="DB 경로",
                    )
                    full_sync_input = gr.Checkbox(
                        label="전체 동기화",
                        value=False,
                    )

                convert_btn = gr.Button("변환 + DB 동기화", variant="primary")
                pipeline_status = gr.Textbox(label="진행 로그", lines=12)
                output_json_path = gr.Textbox(label="생성된 JSON 경로")
                output_db_path = gr.Textbox(label="DB 경로")

                convert_btn.click(
                    fn=run_conversion_and_sync,
                    inputs=[
                        hwp_file,
                        use_llm_preprocess,
                        llm_provider_easy,
                        llm_model_easy,
                        llm_base_url_easy,
                        output_dir,
                        db_path_input,
                        full_sync_input,
                    ],
                    outputs=[pipeline_status, output_json_path, output_db_path],
                )

                gr.Markdown("---")
                ask_question_input_easy = gr.Textbox(
                    label="질문",
                    placeholder="예: 교원 연구년 신청 자격은 무엇인가요?",
                    lines=2,
                )
                ask_top_k_easy = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="참고 규정 수",
                )
                ask_abolished_easy = gr.Checkbox(
                    label="폐지 규정 포함",
                    value=False,
                )
                ask_btn_easy = gr.Button("질문하기", variant="secondary")
                ask_answer_easy = gr.Markdown(label="답변")
                ask_sources_easy = gr.Markdown(label="참고 규정")

                ask_btn_easy.click(
                    fn=ask_question,
                    inputs=[
                        ask_question_input_easy,
                        ask_top_k_easy,
                        ask_abolished_easy,
                        llm_provider_easy,
                        llm_model_easy,
                        llm_base_url_easy,
                        db_path_input,
                    ],
                    outputs=[ask_answer_easy, ask_sources_easy],
                )

            # Tab 0.5: Status
            with gr.TabItem("📂 데이터 현황"):
                status_db_path = gr.Textbox(
                    value=db_path,
                    label="DB 경로",
                )
                status_markdown = gr.Markdown(_render_status(db_path))
                refresh_btn = gr.Button("새로고침", variant="secondary")

                with gr.Row():
                    json_choices = _json_choices()
                    json_select = gr.Dropdown(
                        choices=json_choices,
                        value=json_choices[0] if json_choices else "",
                        label="동기화할 JSON 선택",
                    )
                    full_sync_select = gr.Checkbox(
                        label="전체 동기화",
                        value=False,
                    )
                    sync_btn = gr.Button("동기화 실행", variant="primary")
                sync_result = gr.Markdown()

                def _refresh_status(target_db_path: str):
                    updated_status = _render_status(target_db_path)
                    choices = _json_choices()
                    value = choices[0] if choices else ""
                    return updated_status, gr.update(choices=choices, value=value)

                refresh_btn.click(
                    fn=_refresh_status,
                    inputs=[status_db_path],
                    outputs=[status_markdown, json_select],
                )

                sync_btn.click(
                    fn=run_sync,
                    inputs=[json_select, full_sync_select],
                    outputs=[sync_result],
                )

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

                with gr.Accordion("LLM 설정", open=False):
                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            choices=LLM_PROVIDERS,
                            value=DEFAULT_LLM_PROVIDER,
                            label="프로바이더",
                        )
                        llm_model = gr.Textbox(
                            value=DEFAULT_LLM_MODEL,
                            label="모델 (선택)",
                        )
                        llm_base_url = gr.Textbox(
                            value=DEFAULT_LLM_BASE_URL,
                            label="Base URL (로컬용)",
                            placeholder="예: http://127.0.0.1:11434",
                        )

                ask_btn = gr.Button("질문하기", variant="primary")

                ask_answer = gr.Markdown(label="답변")
                ask_sources = gr.Markdown(label="참고 규정")

                ask_btn.click(
                    fn=ask_question,
                    inputs=[ask_question_input, ask_top_k, ask_abolished, llm_provider, llm_model, llm_base_url, gr.State(db_path)],
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
