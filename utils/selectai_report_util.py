"""SelectAI execution report utilities."""

import json
import logging
import re
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
    "Select AI経過時間（秒）",
    "SELECT開始時間",
    "SELECT終了時間",
    "SELECT経過時間（秒）",
    "全体経過時間（秒）",
    "結果件数",
    "エラー内容",
]

DEFAULT_HISTORY_PATH = (
    Path("runtime") / "selectai_reports" / "selectai_execution_history.jsonl"
)
REPORT_SCREEN_FILENAME_PARTS = {
    "開発者機能: チャット・分析": "developer_chat_analysis",
    "ユーザー機能: 基本機能": "user_basic",
}
ELAPSED_SECONDS_COLUMNS = [
    "Select AI経過時間（秒）",
    "SELECT経過時間（秒）",
    "全体経過時間（秒）",
]
LEGACY_ELAPSED_COLUMNS = {
    "Select AI経過時間（秒）": "Select AI経過時間",
    "SELECT経過時間（秒）": "SELECT経過時間",
    "全体経過時間（秒）": "全体経過時間",
}
_REPORT_LOCK = threading.Lock()


def disabled_report_download_button():
    """Return the initial/inactive report download button state."""
    return gr.DownloadButton(
        label="レポートをダウンロード",
        value=None,
        visible=True,
        interactive=False,
        variant="secondary",
    )


def now_local_iso() -> str:
    """Return a local timezone-aware timestamp for report display."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_execution_id() -> str:
    """Create a compact human-sortable execution id."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def format_elapsed_seconds(ms: float | int | None) -> str:
    """Format elapsed milliseconds as seconds without a unit."""
    if ms is None:
        return ""
    return f"{max(0.0, float(ms)) / 1000:.3f}"


def _format_seconds_value(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    return f"{max(0.0, float(seconds)):.3f}"


def _elapsed_seconds_from_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return _format_seconds_value(value)

    text = str(value).strip()
    if not text:
        return ""
    normalized_text = text.replace(",", "")
    try:
        if normalized_text.endswith("ms"):
            return _format_seconds_value(float(normalized_text[:-2].strip()) / 1000)
        if normalized_text.endswith("秒"):
            return _format_seconds_value(float(normalized_text[:-1].strip()))
        if ":" in normalized_text:
            parts = [float(part) for part in normalized_text.split(":")]
            if len(parts) == 2:
                minutes, seconds = parts
                return _format_seconds_value((minutes * 60) + seconds)
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return _format_seconds_value((hours * 3600) + (minutes * 60) + seconds)
        return _format_seconds_value(float(normalized_text))
    except (TypeError, ValueError):
        return text


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
    source = record or {}
    normalized = {}
    for column in REPORT_COLUMNS:
        value = source.get(column, "")
        if column in ELAPSED_SECONDS_COLUMNS:
            legacy_column = LEGACY_ELAPSED_COLUMNS.get(column, "")
            if value in (None, "") and legacy_column:
                value = source.get(legacy_column, "")
            normalized[column] = _elapsed_seconds_from_value(value)
        else:
            normalized[column] = _stringify_report_value(value)
    return normalized


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


def _filter_records_by_screen(
    records: list[dict[str, str]],
    screen_name: str | None,
) -> list[dict[str, str]]:
    if not screen_name:
        return records
    return [record for record in records if record.get("画面名") == screen_name]


def _report_filename_screen_suffix(screen_name: str | None) -> str:
    if not screen_name:
        return ""
    configured = REPORT_SCREEN_FILENAME_PARTS.get(screen_name)
    if configured:
        return configured
    return re.sub(r"[^0-9A-Za-z]+", "_", str(screen_name)).strip("_").lower()[:80]


def execution_report_dataframe(
    records: list[dict[str, Any]] | None = None,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
    screen_name: str | None = None,
) -> pd.DataFrame:
    """Build the public report DataFrame with stable column order."""
    normalized_records = (
        [normalize_execution_report_record(record) for record in records]
        if records is not None
        else load_execution_report_records(history_path)
    )
    normalized_records = _filter_records_by_screen(normalized_records, screen_name)
    return pd.DataFrame(normalized_records, columns=REPORT_COLUMNS)


def _excel_report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    excel_df = df.copy()
    for column in ELAPSED_SECONDS_COLUMNS:
        excel_df[column] = pd.to_numeric(excel_df[column], errors="coerce")
    return excel_df


def create_execution_report_excel(
    output_path: Path | str | None = None,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
    screen_name: str | None = None,
) -> Path | None:
    """Create an Excel report from JSONL history and return its path."""
    df = execution_report_dataframe(history_path=history_path, screen_name=screen_name)
    if df.empty:
        return None
    if output_path is None:
        screen_suffix = _report_filename_screen_suffix(screen_name)
        screen_part = f"_{screen_suffix}" if screen_suffix else ""
        filename = (
            f"selectai_execution_report{screen_part}_"
            f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        output_path = Path("/tmp") / filename
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        excel_df = _excel_report_dataframe(df)
        excel_df.to_excel(writer, sheet_name="SelectAI Report", index=False)
        worksheet = writer.sheets["SelectAI Report"]
        for column_index, column_name in enumerate(excel_df.columns, start=1):
            if column_name in ELAPSED_SECONDS_COLUMNS:
                for row in worksheet.iter_rows(
                    min_row=2,
                    min_col=column_index,
                    max_col=column_index,
                ):
                    row[0].number_format = "0.000"
    return path


def generate_execution_report_download(
    history_path: Path | str = DEFAULT_HISTORY_PATH,
    screen_name: str | None = None,
):
    """Return Gradio updates for report generation status and download button."""
    try:
        report_path = create_execution_report_excel(
            history_path=history_path,
            screen_name=screen_name,
        )
        target_label = f"（{screen_name}）" if screen_name else ""
        if report_path is None:
            return (
                gr.Markdown(
                    visible=True,
                    value=f"⚠️ 実行履歴がありません{target_label}",
                ),
                disabled_report_download_button(),
            )
        return (
            gr.Markdown(
                visible=True,
                value=f"✅ レポートを生成しました{target_label}: {report_path}",
            ),
            gr.DownloadButton(
                label="レポートをダウンロード",
                value=str(report_path),
                visible=True,
                interactive=True,
                variant="secondary",
            ),
        )
    except Exception as exc:
        logger.error("SelectAI report generation failed: %s", exc)
        return (
            gr.Markdown(
                visible=True,
                value=f"❌ レポート生成に失敗しました: {exc}",
            ),
            disabled_report_download_button(),
        )
