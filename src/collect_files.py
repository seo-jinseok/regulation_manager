"""
HWP 파일 수집기

Enhanced version that:
- .hwpx와 .hwp 파일 모두 수집
- 파일 확장자 자동 감지
- args.hwpx 기본값에 따른 최적 파서 자동 선택 메시지
"""
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def collect_hwp_files(input_path: Path, console, use_hwpx: bool = True) -> Optional[List[Path]]:
    """
    HWP 파일 수집 (개선된 버전)

    Args:
        input_path: 입력 경로
        console: Rich console 객체
        use_hwpx: HWPX 파서 사용 여부 (기본값: True)

    Returns:
        수집된 파일 목록
    """
    if not input_path.exists():
        console.print(f"[red]입력 경로가 존재하지 않습니다: {input_path}[/red]")
        return None

    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        # .hwpx와 .hwp 파일 모두 수집
        hwpx_files = sorted(input_path.rglob("*.hwpx"))
        hwp_legacy = sorted(input_path.rglob("*.hwp"))
        files.extend(hwpx_files)
        files.extend(hwp_legacy)

    if not files:
        console.print("[red]처리할 파일이 없습니다.[/red]")
        console.print("[yellow]지원 형식: .hwpx (HWPX), .hwp (legacy)[/yellow]")
        return None

    # 파일 유형 감지 및 메시지 표시
    has_hwpx = len(hwpx_files) > 0
    has_hwp = len(hwp_legacy) > 0

    if use_hwpx and (has_hwpx or has_hwp):
        # 자동 모드: 최적의 파서 선택 메시지
        if has_hwpx and has_hwp:
            console.print("[dim]확장자 감지: .hwpx({len(hwpx_files)}개), .hwp({len(hwp_legacy)}개)[/dim]")
            console.print("[dim]파서 자동 선택: HWPXDirectParser 사용 (--hwpx 활성화됨)[/dim]")
        elif has_hwpx:
            console.print("[dim]확장자 감지: .hwpx({len(hwpx_files)}개) - HWPX 파서 사용[/dim]")
            console.print("[green]참고: --no-hwpx 옵션으로 기존 파서를 사용할 수 있습니다.[/green]")
        elif has_hwp:
            console.print("[dim]확장자 감지: .hwp({len(hwp_legacy)}개) - 기존 파서 사용[/dim]")
            console.print("[green]참고: --hwpx 옵션으로 HWPXDirectParser를 사용할 수 있습니다.[/green]")
    elif use_hwpx:
        # 자동 모드이지만 한 종류만 있을 때
        console.print("[dim]확장자 감지: .hwpx 또는 .hwp 파일만 발견됨[/dim]")
        console.print("[yellow]HWPX 파서를 사용하려면 --hwpx 옵션을 사용하세요.[/yellow]")
    else:
        # use_hwpx=False이면 수동 모드 메시지
        console.print("[dim]파서 수동 선택 모드 (--no-hwpx 지정됨)[/dim]")

    logger.info(f"수집된 파일: {len(files)}개 (.hwpx: {len(hwpx_files)}, .hwp: {len(hwp_legacy)})")
    return files
