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
from ..application.search_usecase import QueryRewriteInfo, SearchUseCase
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


def _format_query_rewrite_debug(info: Optional[QueryRewriteInfo]) -> str:
    if not info:
        return ""

    lines = ["### 🐞 디버그"]

    if not info.used:
        lines.append(f"- 쿼리 리라이팅: (적용 안됨) '{info.original}'")
        return "\n".join(lines)

    if info.method == "llm":
        method_label = "LLM"
    elif info.method == "rules":
        method_label = "규칙"
    else:
        method_label = "알수없음"

    extras = []
    if info.from_cache:
        extras.append("캐시")
    if info.fallback:
        extras.append("LLM 실패 폴백")
    extra_text = f" ({', '.join(extras)})" if extras else ""

    if info.original == info.rewritten:
        lines.append(
            f"- 쿼리 리라이팅[{method_label}]{extra_text}: (변경 없음) '{info.original}'"
        )
    else:
        lines.append(
            f"- 쿼리 리라이팅[{method_label}]{extra_text}: '{info.original}' -> '{info.rewritten}'"
        )

    if info.used_synonyms is not None:
        lines.append(f"- 동의어 사전: {'사용' if info.used_synonyms else '미사용'}")
    if info.used_intent is not None:
        lines.append(f"- 의도 키워드: {'사용' if info.used_intent else '미사용'}")

    return "\n".join(lines)


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

    search_usecase = SearchUseCase(store)  # use_reranker는 config 기본값 사용
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
            lines.append(f"- **규정집: `{last_synced}`**")

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
        show_debug: bool,
    ) -> Tuple[str, str, str]:
        """Execute search and return formatted results."""
        if not query.strip():
            return "검색어를 입력해주세요.", "", ""

        if store.count() == 0:
            return "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.", "", ""

        # SearchUseCase가 HybridSearcher를 자동 초기화
        search_with_hybrid = SearchUseCase(store)
        results = search_with_hybrid.search_unique(
            query,
            top_k=top_k,
            include_abolished=include_abolished,
        )

        if not results:
            debug_text = ""
            if show_debug:
                debug_text = _format_query_rewrite_debug(
                    search_with_hybrid.get_last_query_rewrite()
                )
            return "검색 결과가 없습니다.", "", debug_text

        # Format results as markdown table (CLI 수준)
        table_rows = ["| # | 규정명 | 코드 | 조항 | 점수 |", "|---|------|------|------|------|"]
        for i, r in enumerate(results, 1):
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = " > ".join(r.chunk.parent_path[-2:]) if r.chunk.parent_path else r.chunk.title
            table_rows.append(f"| {i} | {reg_title} | {r.chunk.rule_code} | {path[:40]} | {r.score:.2f} |")

        table = "\n".join(table_rows)

        # Top result detail (CLI 수준)
        top = results[0]
        full_path = ' > '.join(top.chunk.parent_path) if top.chunk.parent_path else top.chunk.title
        detail = f"""### 🏆 1위 결과: {top.chunk.rule_code}

**규정명:** {top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title}

**경로:** {full_path}

---

{top.chunk.text}
"""

        debug_text = ""
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_hybrid.get_last_query_rewrite()
            )

        return table, detail, debug_text

    # Ask function (with LLM) - Generator for streaming progress
    def ask_question(
        question: str,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        show_debug: bool,
    ):
        """Ask question and get LLM answer with progress updates."""
        if not question.strip():
            yield "질문을 입력해주세요.", "", ""
            return

        # Step 1: Initialize
        yield "⏳ 데이터베이스 연결 중...", "", ""
        
        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)

        if store_for_ask.count() == 0:
            yield "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.", "", ""
            return

        # Step 2: Initialize LLM
        yield "⏳ LLM 클라이언트 초기화 중...", "", ""
        
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
                yield f"LLM 초기화 실패: {e}", "", ""
                return

        # Step 3: Search
        yield "🔍 관련 규정 검색 중...", "", ""
        
        search_with_llm = SearchUseCase(store_for_ask, llm_client)

        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        # Step 4: Generate answer
        yield "🤖 AI 답변 생성 중... (10-30초 소요)", "", ""
        
        answer = search_with_llm.ask(
            question,
            filter=filter,
            top_k=top_k,
            include_abolished=include_abolished,
        )

        # Format sources (CLI 수준)
        # Relative normalization for display
        sources_list = answer.sources
        if sources_list:
            scores = [r.score for r in sources_list]
            max_s, min_s = max(scores), min(scores)
            if max_s == min_s:
                norm_scores = {r.chunk.id: 1.0 for r in sources_list}
            else:
                norm_scores = {r.chunk.id: (r.score - min_s) / (max_s - min_s) for r in sources_list}
        else:
            norm_scores = {}

        sources_md = ["### 📚 참고 규정\n"]
        for i, r in enumerate(answer.sources, 1):
            reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = " > ".join(r.chunk.parent_path) if r.chunk.parent_path else r.chunk.title
            norm_score = norm_scores.get(r.chunk.id, 0.0)
            rel_pct = int(norm_score * 100)
            
            if rel_pct >= 80:
                rel_label = "🟢 매우 높음"
            elif rel_pct >= 50:
                rel_label = "🟡 높음"
            elif rel_pct >= 30:
                rel_label = "🟠 보통"
            else:
                rel_label = "🔴 낮음"
            
            sources_md.append(f"""#### [{i}] {reg_name}
**경로:** {path}

{r.chunk.text[:300]}{'...' if len(r.chunk.text) > 300 else ''}

*규정번호: {r.chunk.rule_code} | 관련도: {rel_pct}% {rel_label}*

---
""")

        # Confidence description
        if answer.confidence >= 0.7:
            conf_desc = "🟢 답변 신뢰도 높음"
        elif answer.confidence >= 0.4:
            conf_desc = "🟡 답변 신뢰도 보통 - 원문 확인 권장"
        else:
            conf_desc = "🔴 답변 신뢰도 낮음 - 학교 행정실 문의 권장"

        sources_text = "\n".join(sources_md) + f"\n**{conf_desc}** (신뢰도 {answer.confidence:.0%})"

        debug_text = ""
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_llm.get_last_query_rewrite()
            )

        yield answer.text, sources_text, debug_text

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
                        search_debug_toggle = gr.Checkbox(
                            label="디버그 출력",
                            value=False,
                        )

                search_btn = gr.Button("검색", variant="primary")

                search_results = gr.Markdown(label="검색 결과")
                search_detail = gr.Markdown(label="상세 내용")
                with gr.Accordion("디버그", open=False):
                    search_debug = gr.Markdown()

                search_btn.click(
                    fn=search_regulations,
                    inputs=[search_query, search_top_k, search_abolished, search_debug_toggle],
                    outputs=[search_results, search_detail, search_debug],
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
                        ask_debug_toggle = gr.Checkbox(
                            label="디버그 출력",
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
                with gr.Accordion("디버그", open=False):
                    ask_debug = gr.Markdown()

                ask_btn.click(
                    fn=ask_question,
                    inputs=[
                        ask_question_input,
                        ask_top_k,
                        ask_abolished,
                        llm_provider,
                        llm_model,
                        llm_base_url,
                        gr.State(db_path),
                        ask_debug_toggle,
                    ],
                    outputs=[ask_answer, ask_sources, ask_debug],
                )

            # Tab 3: Status (Read-only)
            with gr.TabItem("📂 데이터 현황"):
                gr.Markdown("> DB 관리(동기화, 초기화)는 CLI에서 수행합니다: `regulation-rag sync`, `regulation-rag reset`")
                
                status_db_path = gr.Textbox(
                    value=db_path,
                    label="DB 경로",
                    interactive=False,
                )
                status_markdown = gr.Markdown(_render_status(db_path))
                refresh_btn = gr.Button("새로고침", variant="secondary")

                def _refresh_status_only(target_db_path: str):
                    return _render_status(target_db_path)

                refresh_btn.click(
                    fn=_refresh_status_only,
                    inputs=[status_db_path],
                    outputs=[status_markdown],
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
