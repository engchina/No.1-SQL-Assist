"""旧SelectAIカテゴリ移行スクリプトのテスト。"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "migrate_selectai_categories.sh"


class MigrateSelectAiCategoriesTest(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_script(self, legacy_path: Path, metadata_path: Path):
        return subprocess.run(
            [str(SCRIPT_PATH), str(legacy_path), str(metadata_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_migrates_categories_and_preserves_other_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_path = root / "profiles" / "selectai.json"
            metadata_path = root / "metadata_cache" / "list_metadata.json"
            self._write_json(
                legacy_path,
                [
                    {"profile": "PROFILE_A", "category": "営業"},
                    {"profile": " profile_b ", "category": " 人事 "},
                    {"profile": "PROFILE_EMPTY", "category": ""},
                    {"profile": "LEGACY_ONLY", "category": "旧分類"},
                ],
            )
            self._write_json(
                metadata_path,
                {
                    "version": 1,
                    "updated_at": "unchanged",
                    "tables": [{"name": "EMPLOYEE"}],
                    "views": [],
                    "profiles": [
                        {
                            "profile": "PROFILE_A",
                            "category": "",
                            "attributes": {"model": "model-a"},
                        },
                        {
                            "profile": "PROFILE_B",
                            "category": "old",
                            "attributes": {"model": "model-b"},
                        },
                        {
                            "profile": "PROFILE_C",
                            "category": "保守対象外",
                            "attributes": {"model": "model-c"},
                        },
                    ],
                },
            )

            result = self._run_script(legacy_path, metadata_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("更新したカテゴリ数: 2", result.stdout)
            self.assertIn(
                "移行先キャッシュに存在しないProfile: LEGACY_ONLY",
                result.stdout,
            )
            migrated = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["profiles"][0]["category"], "営業")
            self.assertEqual(migrated["profiles"][1]["category"], "人事")
            self.assertEqual(migrated["profiles"][2]["category"], "保守対象外")
            self.assertEqual(
                migrated["profiles"][0]["attributes"],
                {"model": "model-a"},
            )
            self.assertEqual(migrated["tables"], [{"name": "EMPLOYEE"}])
            self.assertEqual(migrated["updated_at"], "unchanged")

            backups = list(metadata_path.parent.glob("list_metadata.json.bak.*"))
            self.assertEqual(len(backups), 1)
            original = json.loads(backups[0].read_text(encoding="utf-8"))
            self.assertEqual(original["profiles"][0]["category"], "")

    def test_conflicting_legacy_categories_fail_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_path = root / "profiles" / "selectai.json"
            metadata_path = root / "metadata_cache" / "list_metadata.json"
            self._write_json(
                legacy_path,
                [
                    {"profile": "PROFILE_A", "category": "営業"},
                    {"profile": "profile_a", "category": "人事"},
                ],
            )
            original_metadata = {
                "profiles": [{"profile": "PROFILE_A", "category": ""}]
            }
            self._write_json(metadata_path, original_metadata)

            result = self._run_script(legacy_path, metadata_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("重複Profile", result.stderr)
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                original_metadata,
            )
            self.assertEqual(
                list(metadata_path.parent.glob("list_metadata.json.bak.*")),
                [],
            )

    def test_no_changes_does_not_create_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_path = root / "profiles" / "selectai.json"
            metadata_path = root / "metadata_cache" / "list_metadata.json"
            self._write_json(
                legacy_path,
                [{"profile": "PROFILE_A", "category": "営業"}],
            )
            original_metadata = {
                "profiles": [{"profile": "PROFILE_A", "category": "営業"}]
            }
            self._write_json(metadata_path, original_metadata)

            result = self._run_script(legacy_path, metadata_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("書き込む変更はありません", result.stdout)
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                original_metadata,
            )
            self.assertEqual(
                list(metadata_path.parent.glob("list_metadata.json.bak.*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
