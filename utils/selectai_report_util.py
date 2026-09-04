"""SelectAI execution report utilities."""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "画面名",
    "実行ID",
    "作成日時",
    "自然言語の質問",
    "実行に使用した質問",
    "カテゴリ予測",
    "Profile",
    "生成されたSQL",
    "ステータス",
    "試行回数",
    "実行開始時間",
    "Select AI開始時間",
    "Select AI終了時間",
    "Select AI経過時間",
    "SELECT開始時間",
    "SELECT終了時間",
    "SELECT経過時間",
    "全体経過時間",
    "結果件数",
    "エラー内容",
]

DEFAULT_HISTORY_PATH = (
    Path("runtime") / "selectai_reports" / "selectai_execution_history.jsonl"
)
_REPORT_LOCK = threading.Lock()


def now_local_iso() -> str:
    """Return a local timezone-aware timestamp for report display."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_execution_id() -> str:
    """Create a compact human-sortable execution id."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def format_elapsed_clock(ms: float | int | None) -> str:
    """Format elapsed milliseconds as mm:ss or h:mm:ss for stable UI layout."""
    total_seconds = max(0, int((ms or 0) // 1000))
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{total_minutes:02d}:{seconds:02d}"


def format_elapsed_duration(ms: float | int | None) -> str:
    """Format elapsed milliseconds for saved technical history."""
    if ms is None:
        return ""
    normalized = max(0.0, float(ms))
    if normalized < 1000:
        return f"{round(normalized)}ms"
    if normalized < 60000:
        return f"{normalized / 1000:.1f}秒"
    return format_elapsed_clock(normalized)


def _stringify_report_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def normalize_execution_report_record(record: dict[str, Any]) -> dict[str, str]:
    """Return a record with exactly the public report columns in order."""
    return {
        column: _stringify_report_value((record or {}).get(column, ""))
        for column in REPORT_COLUMNS
    }


def append_execution_report(
    record: dict[str, Any],
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> Path:
    """Append one execution record to the JSONL history."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_execution_report_record(record)
    line = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    with _REPORT_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return path


def load_execution_report_records(
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> list[dict[str, str]]:
    """Load execution records from JSONL history, skipping malformed lines."""
    path = Path(history_path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed SelectAI report line %s: %s",
                    line_number,
                    exc,
                )
                continue
            if isinstance(data, dict):
                records.append(normalize_execution_report_record(data))
    return records


def execution_report_dataframe(
    records: list[dict[str, Any]] | None = None,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    """Build the public report DataFrame with stable column order."""
    normalized_records = (
        [normalize_execution_report_record(record) for record in records]
        if records is not None
        else load_execution_report_records(history_path)
    )
    return pd.DataFrame(normalized_records, columns=REPORT_COLUMNS)


def create_execution_report_excel(
    output_path: Path | str | None = None,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> Path | None:
    """Create an Excel report from JSONL history and return its path."""
    df = execution_report_dataframe(history_path=history_path)
    if df.empty:
        return None
    if output_path is None:
        filename = (
            "selectai_execution_report_"
            f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        output_path = Path("/tmp") / filename
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="SelectAI Report", index=False)
    return path


def generate_execution_report_download(
    history_path: Path | str = DEFAULT_HISTORY_PATH,
):
    """Return Gradio updates for report generation status and download button."""
    try:
        report_path = create_execution_report_excel(history_path=history_path)
        if report_path is None:
            return (
                gr.Markdown(
                    visible=True,
                    value="⚠️ 実行履歴がありません",
                    elem_classes=["operation-status", "operation-status--warning"],
                ),
                gr.DownloadButton(value=None, visible=False),
            )
        return (
            gr.Markdown(
                visible=True,
                value=f"✅ レポートを生成しました: {report_path}",
                elem_classes=["operation-status", "operation-status--success"],
            ),
            gr.DownloadButton(
                label="レポートをダウンロード",
                value=str(report_path),
                visible=True,
                variant="secondary",
            ),
        )
    except Exception as exc:
        logger.error("SelectAI report generation failed: %s", exc)
        return (
            gr.Markdown(
                visible=True,
                value=f"❌ レポート生成に失敗しました: {exc}",
                elem_classes=["operation-status", "operation-status--error"],
            ),
            gr.DownloadButton(value=None, visible=False),
        )
