"""Tests for SelectAI execution report utilities."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.selectai_report_util import (
    REPORT_COLUMNS,
    append_execution_report,
    create_execution_report_excel,
    execution_report_dataframe,
    format_elapsed_clock,
    format_elapsed_duration,
    load_execution_report_records,
)


class SelectAiReportUtilTest(unittest.TestCase):
    def test_format_elapsed_clock_matches_reference_shape(self):
        self.assertEqual(format_elapsed_clock(0), "00:00")
        self.assertEqual(format_elapsed_clock(1500), "00:01")
        self.assertEqual(format_elapsed_clock(65000), "01:05")
        self.assertEqual(format_elapsed_clock(3661000), "1:01:01")

    def test_format_elapsed_duration_keeps_short_precision(self):
        self.assertEqual(format_elapsed_duration(None), "")
        self.assertEqual(format_elapsed_duration(999), "999ms")
        self.assertEqual(format_elapsed_duration(1500), "1.5秒")
        self.assertEqual(format_elapsed_duration(65000), "01:05")

    def test_append_and_load_jsonl_keeps_non_ascii_and_column_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "reports" / "history.jsonl"

            append_execution_report(
                {
                    "画面名": "ユーザー機能: 基本機能",
                    "実行ID": "run-1",
                    "自然言語の質問": "大阪の顧客数を教えて",
                    "カテゴリ予測": "営業",
                    "生成されたSQL": "SELECT COUNT(*) FROM CUSTOMERS",
                    "試行回数": 2,
                },
                history_path=history_path,
            )
            history_path.write_text(
                history_path.read_text(encoding="utf-8")
                + "{malformed json}\n",
                encoding="utf-8",
            )

            records = load_execution_report_records(history_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(list(records[0].keys()), REPORT_COLUMNS)
            self.assertEqual(records[0]["自然言語の質問"], "大阪の顧客数を教えて")
            self.assertEqual(records[0]["試行回数"], "2")

            raw_line = history_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("大阪", raw_line)
            json.loads(raw_line)

    def test_execution_report_excel_uses_stable_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            history_path = temp_root / "history.jsonl"
            output_path = temp_root / "selectai_report.xlsx"
            append_execution_report(
                {
                    "画面名": "開発者機能: チャット・分析",
                    "実行ID": "run-2",
                    "ステータス": "✅ 取得完了",
                    "結果件数": 3,
                },
                history_path=history_path,
            )

            df = execution_report_dataframe(history_path=history_path)
            self.assertEqual(df.columns.tolist(), REPORT_COLUMNS)
            self.assertEqual(df.iloc[0]["結果件数"], "3")

            created_path = create_execution_report_excel(
                output_path=output_path,
                history_path=history_path,
            )
            self.assertEqual(created_path, output_path)
            self.assertTrue(output_path.exists())

            excel_df = pd.read_excel(output_path)
            self.assertEqual(excel_df.columns.tolist(), REPORT_COLUMNS)
            self.assertEqual(
                excel_df.iloc[0]["画面名"],
                "開発者機能: チャット・分析",
            )


if __name__ == "__main__":
    unittest.main()
