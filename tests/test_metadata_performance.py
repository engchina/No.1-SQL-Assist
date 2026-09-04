"""Tests for metadata list queries used by management and SelectAI screens."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_dependency_stubs():
    if "utils.vpd_management_util" not in sys.modules:
        vpd_management_util = types.ModuleType("utils.vpd_management_util")
        vpd_management_util.build_vpd_management_tab = lambda *args, **kwargs: None
        sys.modules["utils.vpd_management_util"] = vpd_management_util
    if "utils.sql_learning_util" not in sys.modules:
        sql_learning_util = types.ModuleType("utils.sql_learning_util")
        sql_learning_util.build_sql_learning_tab = lambda *args, **kwargs: None
        sys.modules["utils.sql_learning_util"] = sql_learning_util
    if "oracledb" not in sys.modules:
        oracledb = types.ModuleType("oracledb")
        oracledb.DatabaseError = Exception
        sys.modules["oracledb"] = oracledb
    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        sklearn_linear_model = types.ModuleType("sklearn.linear_model")
        sklearn_linear_model.LogisticRegression = object
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.linear_model"] = sklearn_linear_model
    if "joblib" not in sys.modules:
        joblib = types.ModuleType("joblib")
        joblib.load = lambda *args, **kwargs: None
        sys.modules["joblib"] = joblib
    if "oci" not in sys.modules:
        class FakeGenerativeAiInferenceClient:
            def __init__(self, *args, **kwargs):
                pass

        oci = types.ModuleType("oci")
        oci.config = types.SimpleNamespace(from_file=lambda *args, **kwargs: {"region": "ap-tokyo-1"})
        oci.retry = types.SimpleNamespace(NoneRetryStrategy=lambda: None)
        genai = types.ModuleType("oci.generative_ai_inference")
        genai.GenerativeAiInferenceClient = FakeGenerativeAiInferenceClient
        genai_models = types.ModuleType("oci.generative_ai_inference.models")
        genai_models.EmbedTextDetails = object
        sys.modules["oci"] = oci
        sys.modules["oci.generative_ai_inference"] = genai
        sys.modules["oci.generative_ai_inference.models"] = genai_models


_install_dependency_stubs()

from utils.management_util import (
    execute_create_table,
    get_table_list,
    get_table_list_cached,
    get_table_list_for_data,
    get_table_list_for_upload,
    get_view_list,
    get_view_list_cached,
    invalidate_object_list_cache,
    refresh_table_view_cache_from_db,
)
from utils.metadata_cache_util import (
    get_profile_cache_entries,
    replace_profile_cache,
    replace_table_view_cache,
)
from utils.selectai_util import get_db_profiles, _save_profiles_to_json_stream
from utils.selectai_util import create_db_profile


class FakeLob:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class FakeCursor:
    def __init__(self, responses, executed):
        self._responses = responses
        self._current = []
        self.executed = executed
        self.arraysize = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, *args, **kwargs):
        self.executed.append((sql, args, kwargs))
        self._current = self._responses.pop(0) if self._responses else []

    def fetchall(self):
        return self._current


class FakeConnection:
    def __init__(self, responses, executed):
        self._responses = responses
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor(self._responses, self._executed)

    def commit(self):
        pass


class FakePool:
    def __init__(self, responses):
        self._responses = list(responses)
        self.executed = []

    def acquire(self):
        return FakeConnection(self._responses, self.executed)


def executed_sql(pool):
    return "\n".join(sql for sql, _args, _kwargs in pool.executed)


class MetadataPerformanceTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self._temp_dir.name)
        cache_dir = temp_root / "metadata_cache"
        legacy_dir = temp_root / "profiles"
        self._patchers = [
            patch("utils.metadata_cache_util._CACHE_DIR", cache_dir),
            patch(
                "utils.metadata_cache_util._METADATA_CACHE_FILE",
                cache_dir / "list_metadata.json",
            ),
            patch(
                "utils.metadata_cache_util._LEGACY_PROFILE_CACHE_FILE",
                legacy_dir / "selectai.json",
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
        invalidate_object_list_cache()

    def tearDown(self):
        invalidate_object_list_cache()
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._temp_dir.cleanup()

    def test_table_list_reads_local_json_without_querying_db(self):
        replace_table_view_cache(
            [
                {"name": "CUSTOMERS", "rows": 1200, "comments": "customers table"},
                {"name": "ORDERS", "rows": "", "comments": ""},
            ],
            [],
        )
        pool = FakePool([])

        df = get_table_list(pool)

        self.assertEqual(list(df.columns), ["Table Name", "Rows", "Comments"])
        self.assertEqual(df["Table Name"].tolist(), ["CUSTOMERS", "ORDERS"])
        self.assertEqual(df["Rows"].tolist(), [1200, ""])
        self.assertEqual(df["Comments"].tolist(), ["customers table", ""])
        self.assertEqual(len(pool.executed), 0)

    def test_table_list_cache_reuses_result_until_invalidated(self):
        pool = FakePool([])
        replace_table_view_cache(
            [{"name": "CUSTOMERS", "rows": 1200, "comments": "customers table"}],
            [],
        )

        first_df = get_table_list_cached(pool)
        replace_table_view_cache(
            [{"name": "ORDERS", "rows": 500, "comments": "orders table"}],
            [],
        )
        second_df = get_table_list_cached(pool)
        invalidate_object_list_cache("table")
        third_df = get_table_list_cached(pool)

        self.assertEqual(first_df["Table Name"].tolist(), ["CUSTOMERS"])
        self.assertEqual(second_df["Table Name"].tolist(), ["CUSTOMERS"])
        self.assertEqual(third_df["Table Name"].tolist(), ["ORDERS"])
        self.assertEqual(len(pool.executed), 0)

    def test_view_data_and_upload_lists_read_local_json(self):
        replace_table_view_cache(
            [
                {"name": "CUSTOMERS", "rows": 1200, "comments": "customers table"},
                {"name": "DR$INDEX", "rows": "", "comments": ""},
                {"name": "VECTOR$STORE", "rows": "", "comments": ""},
            ],
            [{"name": "V_CUSTOMERS", "comments": "customer view"}],
        )
        pool = FakePool([])

        view_df = get_view_list(pool)
        names = get_table_list_for_data(pool)
        upload_names = get_table_list_for_upload(pool)

        self.assertEqual(view_df["View Name"].tolist(), ["V_CUSTOMERS"])
        self.assertEqual(names, ["CUSTOMERS", "DR$INDEX", "VECTOR$STORE", "V_CUSTOMERS"])
        self.assertEqual(upload_names, ["CUSTOMERS"])
        self.assertEqual(len(pool.executed), 0)

    def test_table_view_cache_refresh_queries_db_once_per_list_and_saves_json(self):
        pool = FakePool([
            [
                ("CUSTOMERS", 1200, FakeLob("customers table")),
                ("ORDERS", None, None),
            ],
            [("V_CUSTOMERS", FakeLob("customer view"))],
        ])

        table_df, view_df, _cache_path = refresh_table_view_cache_from_db(pool)

        self.assertEqual(table_df["Table Name"].tolist(), ["CUSTOMERS", "ORDERS"])
        self.assertEqual(view_df["View Name"].tolist(), ["V_CUSTOMERS"])
        self.assertEqual(len(pool.executed), 2)
        sql_text = executed_sql(pool).upper()
        self.assertIn("ALL_OBJECTS", sql_text)
        self.assertIn("NUM_ROWS", sql_text)
        self.assertNotIn("COUNT(*)", sql_text)

        replace_table_view_cache(
            [{"name": "CHANGED_ON_DISK", "rows": 1, "comments": ""}],
            [{"name": "V_CHANGED_ON_DISK", "comments": ""}],
        )
        cached_table_df = get_table_list_cached(FakePool([]))
        cached_view_df = get_view_list_cached(FakePool([]))
        self.assertEqual(cached_table_df["Table Name"].tolist(), ["CUSTOMERS", "ORDERS"])
        self.assertEqual(cached_view_df["View Name"].tolist(), ["V_CUSTOMERS"])

    def test_create_table_syncs_single_cache_entry_after_db_success(self):
        pool = FakePool([
            [],
            [("NEW_TABLE", 10, "new table")],
        ])

        result = execute_create_table(pool, "CREATE TABLE new_table (id NUMBER)")
        df = get_table_list(FakePool([]))

        self.assertIn("✅ 成功", result)
        self.assertEqual(df["Table Name"].tolist(), ["NEW_TABLE"])
        self.assertEqual(df["Rows"].tolist(), [10])
        self.assertEqual(df["Comments"].tolist(), ["new table"])
        self.assertEqual(len(pool.executed), 2)
        self.assertNotIn("COUNT(*)", executed_sql(pool).upper())

    def test_get_db_profiles_reads_local_json_without_querying_db(self):
        replace_profile_cache([
            {
                "profile": "PROFILE_A",
                "category": "sales",
                "tables": ["CUSTOMERS"],
                "views": ["V_CUSTOMERS"],
                "region": "ap-tokyo-1",
                "model": "cohere.command-r-plus-08-2024",
                "embedding_model": "cohere.embed-v4.0",
                "attributes": {"region": "ap-tokyo-1"},
            }
        ])
        pool = FakePool([])

        df = get_db_profiles(pool)

        self.assertEqual(df["Profile Name"].tolist(), ["PROFILE_A"])
        self.assertEqual(df["Tables"].tolist(), ["CUSTOMERS"])
        self.assertEqual(df["Views"].tolist(), ["V_CUSTOMERS"])
        self.assertEqual(len(pool.executed), 0)

    def test_create_profile_syncs_local_json_after_db_success(self):
        pool = FakePool([[], []])

        create_db_profile(
            pool,
            "PROFILE_A",
            "ocid1.compartment.example",
            "ap-tokyo-1",
            "cohere.command-r-plus-08-2024",
            "cohere.embed-v4.0",
            1024,
            True,
            True,
            False,
            False,
            ["CUSTOMERS"],
            ["V_CUSTOMERS"],
            "sales",
        )
        df = get_db_profiles(FakePool([]))

        self.assertEqual(df["Profile Name"].tolist(), ["PROFILE_A"])
        self.assertEqual(df["Tables"].tolist(), ["CUSTOMERS"])
        self.assertEqual(df["Views"].tolist(), ["V_CUSTOMERS"])
        self.assertEqual(len(pool.executed), 2)

    def test_profile_json_stream_batches_attributes_for_all_profiles(self):
        attrs = json.dumps([
            {"owner": "ADMIN", "name": "CUSTOMERS"},
            {"owner": "ADMIN", "name": "V_CUSTOMERS"},
        ])
        pool = FakePool([
            [
                ("PROFILE_A", FakeLob("sales"), "ENABLED"),
                ("OCI_CRED$PROF", FakeLob("hidden"), "ENABLED"),
            ],
            [
                ("PROFILE_A", "OBJECT_LIST", attrs),
                ("PROFILE_A", "REGION", "ap-tokyo-1"),
                ("PROFILE_A", "MODEL", "cohere.command-r-plus-08-2024"),
                ("PROFILE_A", "EMBEDDING_MODEL", "cohere.embed-v4.0"),
            ],
        ])

        with patch("utils.selectai_util._get_table_names", return_value=["CUSTOMERS"]):
            with patch("utils.selectai_util._get_view_names", return_value=["V_CUSTOMERS"]):
                messages = list(_save_profiles_to_json_stream(pool))
        df = get_db_profiles(FakePool([]))

        self.assertTrue(messages[-1].startswith("✅ 1件のProfileを保存"))
        self.assertEqual(df["Profile Name"].tolist(), ["PROFILE_A"])
        self.assertEqual(df["Tables"].tolist(), ["CUSTOMERS"])
        self.assertEqual(df["Views"].tolist(), ["V_CUSTOMERS"])
        self.assertEqual(len(pool.executed), 2)
        sql_text = executed_sql(pool).upper()
        self.assertIn("USER_CLOUD_AI_PROFILES", sql_text)
        self.assertIn("USER_CLOUD_AI_PROFILE_ATTRIBUTES", sql_text)
        self.assertNotIn("WHERE PROFILE_NAME = :NAME", sql_text)

    def test_profile_json_stream_uses_single_metadata_cache_file(self):
        attrs = json.dumps([
            {"owner": "ADMIN", "name": "CUSTOMERS"},
            {"owner": "ADMIN", "name": "V_CUSTOMERS"},
        ])
        pool = FakePool([
            [("PROFILE_A", "sales", "ENABLED")],
            [("PROFILE_A", "OBJECT_LIST", attrs)],
        ])

        with patch("utils.selectai_util._get_table_names", return_value=["CUSTOMERS"]):
            with patch("utils.selectai_util._get_view_names", return_value=["V_CUSTOMERS"]):
                messages = list(_save_profiles_to_json_stream(pool))
        cache_path = Path(self._temp_dir.name) / "metadata_cache" / "list_metadata.json"
        legacy_path = Path(self._temp_dir.name) / "profiles" / "selectai.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertTrue(messages[-1].startswith("✅ 1件のProfileを保存"))
        self.assertFalse(legacy_path.exists())
        self.assertEqual(payload["profiles"][0]["profile"], "PROFILE_A")
        self.assertEqual(
            {
                key: payload["profiles"][0][key]
                for key in ("profile", "category", "tables", "views")
            },
            {
                "profile": "PROFILE_A",
                "category": "sales",
                "tables": ["CUSTOMERS"],
                "views": ["V_CUSTOMERS"],
            },
        )
        self.assertEqual(len(pool.executed), 2)
        self.assertEqual(executed_sql(pool).upper().count("USER_CLOUD_AI_PROFILE_ATTRIBUTES"), 1)

    def test_profile_cache_reads_legacy_selectai_json_for_migration_only(self):
        legacy_path = Path(self._temp_dir.name) / "profiles" / "selectai.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps([
                {
                    "profile": "PROFILE_LEGACY",
                    "category": "legacy",
                    "tables": ["CUSTOMERS"],
                    "views": ["V_CUSTOMERS"],
                }
            ]),
            encoding="utf-8",
        )

        profiles = get_profile_cache_entries()

        self.assertEqual(profiles[0]["profile"], "PROFILE_LEGACY")
        self.assertFalse((Path(self._temp_dir.name) / "metadata_cache" / "list_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
