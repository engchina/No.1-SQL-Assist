"""Tests for SelectAI execution status display."""

import unittest

from utils.selectai_util import _format_selectai_execution_status


class SelectAiExecutionStatusTest(unittest.TestCase):
    def test_select_execution_phase_is_labeled_sql_execution(self):
        status = _format_selectai_execution_status(
            "✅ 完了",
            {
                "execution_id": "run-1",
                "execution_started_at": "2026-09-07T10:00:00+09:00",
                "execution_started_perf": 1.0,
                "execution_finished_perf": 2.0,
                "select_ai_started_at": "2026-09-07T10:00:00+09:00",
                "select_ai_finished_at": "2026-09-07T10:00:01+09:00",
                "select_ai_elapsed_ms": 1000.0,
                "select_started_at": "2026-09-07T10:00:01+09:00",
                "select_finished_at": "2026-09-07T10:00:02+09:00",
                "select_elapsed_ms": 62.0,
            },
        )

        self.assertIn("| SQL実行 |", status)
        self.assertNotIn("| SELECT |", status)


if __name__ == "__main__":
    unittest.main()
