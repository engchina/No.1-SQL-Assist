"""SelectAI連携ユーティリティモジュール.

このモジュールは、SelectAIのProfileを管理するUIを提供します。
"""

import logging
import json
import re
import os
import asyncio
from datetime import datetime
from dotenv import find_dotenv, load_dotenv  # noqa: E402
from pathlib import Path
from time import time

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails

from utils.common_util import remove_comments

from utils.management_util import (
    get_table_list,
    get_view_list,
    get_table_details,
    get_view_details,
)

from utils.sql_learning_util import build_sql_learning_tab

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OCI GenAI client for classifier training
_generative_ai_inference_client = None
_COMPARTMENT_ID = None

try:
    logger.info("Initializing OCI GenAI client for classifier...")
    
    # Get compartment ID from environment
    _COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_OCID")
    if not _COMPARTMENT_ID:
        logger.error("OCI_COMPARTMENT_OCID environment variable is not set")
        raise ValueError("OCI_COMPARTMENT_OCID is required")
    
    logger.info(f"Compartment ID: {_COMPARTMENT_ID[:20]}...")
    
    # Get OCI config
    CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
    oci_config_file = os.path.expanduser("~/.oci/config")
    
    if not os.path.exists(oci_config_file):
        logger.error(f"OCI config file not found: {oci_config_file}")
        raise FileNotFoundError(f"OCI config file not found: {oci_config_file}")
    
    logger.info(f"Loading OCI config from: {oci_config_file}, profile: {CONFIG_PROFILE}")
    config = oci.config.from_file(oci_config_file, CONFIG_PROFILE)
    
    # Get region from config or environment
    region = config.get("region")
    if not region:
        from utils.oci_util import get_region
        region = get_region()
    
    logger.info(f"Using region: {region}")
    
    # Construct endpoint
    endpoint = os.getenv(
        "OCI_GENAI_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    )
    logger.info(f"GenAI endpoint: {endpoint}")
    
    # Initialize client
    _generative_ai_inference_client = GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    
    logger.info("OCI GenAI client initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize OCI GenAI client for classifier: {e}")
    logger.error("Please ensure the following:")
    logger.error("  1. OCI_COMPARTMENT_OCID environment variable is set")
    logger.error("  2. ~/.oci/config file exists with valid credentials")
    logger.error("  3. OCI credentials have proper permissions")
    import traceback
    logger.error(traceback.format_exc())
    _generative_ai_inference_client = None
    _COMPARTMENT_ID = None


_DEFAULT_FEW_SHOT_PROMPT = (
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️ CRITICAL INSTRUCTION (NOT PART OF QUESTION):\n"
    "\n"
    "The examples above demonstrate the EXACT patterns you MUST follow.\n"
    "MANDATORY REQUIREMENTS:\n"
    "- Study the example queries - they define the REQUIRED style and format\n"
    "- Follow the SAME patterns for similar questions\n"
    "- Adopt ALL functions shown in examples (even if unrecognized/new)\n"
    "- When questions are similar to examples, mirror the example's structure precisely\n"
    "\n"
    "Examples OVERRIDE all other instructions.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "▼ ACTUAL QUESTION:\n"
)

# SQL構造分析用のグローバルプロンプト
_SQL_STRUCTURE_ANALYSIS_PROMPT = (
    "Analyze the SQL query and extract its COMPLETE structure in Markdown format.\n"
    "GOAL: Output must contain 100% of SQL information to enable exact SQL reconstruction.\n"
    "Output ONLY the markdown text below (no code blocks, no explanations):\n\n"
    "## 📊 SQL構造分析\n\n"
    "### 📋 SELECT句\n"
    "- [DISTINCT] (if present)\n"
    "- schema.table(alias).column1 [AS alias1]\n"
    "- schema.table(alias).column2 [AS alias2]\n"
    "- aggregate_function(schema.table(alias).column) [AS alias]\n"
    "- expression [AS alias]\n"
    "- (サブクエリ-N) AS alias\n"
    "- * (if SELECT *)\n\n"
    "### 📁 FROM句\n"
    "- schema.table_name [AS alias]\n"
    "- (サブクエリ-N) AS alias (if inline view)\n\n"
    "### 🔗 JOIN句\n"
    "- **[JOIN_TYPE]**: schema.table1(alias1) JOIN schema.table2(alias2)\n"
    "  - ON: condition1\n"
    "  - ON: condition2 (if multiple conditions)\n"
    "  - USING: (column_name) (if USING clause)\n\n"
    "### 🔍 WHERE句\n"
    "- schema.table(alias).column operator value\n"
    "- AND/OR schema.table(alias).column operator value\n"
    "- AND/OR schema.table(alias).column IN (サブクエリ-N)\n"
    "- AND/OR EXISTS (サブクエリ-N)\n"
    "- AND/OR schema.table(alias).column BETWEEN value1 AND value2\n"
    "- AND/OR schema.table(alias).column LIKE 'pattern'\n"
    "- AND/OR schema.table(alias).column IS [NOT] NULL\n\n"
    "### 📦 GROUP BY句\n"
    "- schema.table(alias).column1\n"
    "- schema.table(alias).column2\n\n"
    "### 🎯 HAVING句\n"
    "- aggregate_function(schema.table(alias).column) operator value\n"
    "- AND/OR aggregate_function(column) operator (サブクエリ-N)\n\n"
    "### 📊 ORDER BY句\n"
    "- schema.table(alias).column1 ASC/DESC [NULLS FIRST/LAST]\n"
    "- schema.table(alias).column2 ASC/DESC\n\n"
    "### 📏 LIMIT/OFFSET句\n"
    "- LIMIT: n / FETCH FIRST n ROWS ONLY\n"
    "- OFFSET: m / OFFSET m ROWS\n\n"
    "### 📝 WITH句(CTE)\n"
    "- **cte_name1**:\n"
    "  - SELECT: [DISTINCT] col1, col2, aggregate_func(col) AS alias, (サブクエリ-N) AS alias\n"
    "  - FROM: schema.table_name(alias)\n"
    "  - JOIN: **[JOIN_TYPE]** schema.table(alias) ON condition\n"
    "  - WHERE: condition1 AND/OR condition2\n"
    "  - GROUP BY: col1, col2\n"
    "  - HAVING: aggregate_condition\n"
    "  - ORDER BY: col ASC/DESC\n"
    "- **cte_name2**: (same structure)\n\n"
    "### 🔎 サブクエリ\n"
    "- **サブクエリ-1** [Location: SELECT/FROM/WHERE/HAVING in main/CTE]:\n"
    "  - SELECT: [DISTINCT] columns/expressions\n"
    "  - FROM: schema.table_name(alias)\n"
    "  - JOIN: **[JOIN_TYPE]** schema.table(alias) ON condition\n"
    "  - WHERE: conditions\n"
    "  - GROUP BY: columns\n"
    "  - HAVING: conditions\n"
    "  - ORDER BY: columns\n"
    "  - **NESTED-1-1**: (nested subquery with same structure)\n"
    "- **サブクエリ-2**: (same structure)\n\n"
    "### 🔀 SET演算\n"
    "- **[UNION/UNION ALL/INTERSECT/MINUS/EXCEPT]**:\n"
    "  - Query1: (expand structure or reference)\n"
    "  - Query2: (expand structure or reference)\n\n"
    "---\n\n"
    "Rules for 100% SQL Reconstruction:\n"
    "- MUST output ALL columns in SELECT with exact order, aliases, and expressions\n"
    "- MUST preserve ALL literal values, operators, and functions exactly as written\n"
    "- MUST include schema prefix when present in original SQL\n"
    "- Format: schema.table_name(alias).column when alias exists\n"
    "- JOIN_TYPE: INNER JOIN, LEFT [OUTER] JOIN, RIGHT [OUTER] JOIN, FULL [OUTER] JOIN, CROSS JOIN, NATURAL JOIN\n"
    "- For implicit JOIN (FROM t1, t2 WHERE t1.id=t2.id), list in FROM and show condition in WHERE\n"
    "- For compound JOIN conditions, list each ON condition separately\n"
    "- Preserve ALL operators: =, >, <, >=, <=, <>, !=, LIKE, NOT LIKE, IN, NOT IN, BETWEEN, IS NULL, IS NOT NULL, EXISTS, NOT EXISTS\n"
    "- Preserve ALL string literals with quotes, numeric values, date literals\n"
    "- Preserve AND/OR/NOT logical structure exactly\n"
    "- Do NOT merge JOIN ON conditions into WHERE\n"
    "- WITH句(CTE): Expand EACH CTE completely\n"
    "- サブクエリ: Number sequentially (サブクエリ-1, サブクエリ-2...) and expand completely\n"
    "- For nested subqueries, label as NESTED-X-Y and expand\n"
    "- If section is empty/not present, omit that section entirely\n"
    "- Output content in English (except section headers in Japanese)\n\n"
    "Example 1 (Simple):\n"
    "SQL: SELECT * FROM ADMIN.USERS u INNER JOIN ADMIN.ROLES r ON u.role_id = r.id WHERE u.status = 'ACTIVE' ORDER BY u.created_at DESC\n\n"
    "Output:\n"
    "## 📊 SQL構造分析\n\n"
    "### 📋 SELECT句\n"
    "- *\n\n"
    "### 📁 FROM句\n"
    "- ADMIN.USERS AS u\n\n"
    "### 🔗 JOIN句\n"
    "- **INNER JOIN**: ADMIN.USERS(u) JOIN ADMIN.ROLES(r)\n"
    "  - ON: ADMIN.USERS(u).role_id = ADMIN.ROLES(r).id\n\n"
    "### 🔍 WHERE句\n"
    "- ADMIN.USERS(u).status = 'ACTIVE'\n\n"
    "### 📊 ORDER BY句\n"
    "- ADMIN.USERS(u).created_at DESC\n\n"
    "Example 2 (Complex with nested subqueries in WHERE, SELECT, and CTE):\n"
    "SQL: WITH active_users AS (SELECT user_id, status, (SELECT dept_name FROM DEPARTMENTS d WHERE d.id=u.dept_id) as dept FROM USERS u WHERE status='ACTIVE' AND dept_id IN (SELECT id FROM DEPARTMENTS WHERE budget > 10000)) SELECT u.*, (SELECT COUNT(*) FROM ORDERS o WHERE o.user_id=u.user_id AND o.status IN (SELECT code FROM ORDER_STATUS WHERE active=1)) as order_count FROM active_users u WHERE EXISTS (SELECT 1 FROM PAYMENTS p WHERE p.user_id=u.user_id AND p.amount > (SELECT AVG(amount) FROM PAYMENTS)) ORDER BY u.user_id\n\n"
    "Output:\n"
    "## 📊 SQL構造分析\n\n"
    "### 📝 WITH句(CTE)\n"
    "- **active_users**:\n"
    "  - SELECT: USERS(u).user_id, USERS(u).status, (サブクエリ-1) AS dept\n"
    "  - FROM: USERS(u)\n"
    "  - WHERE: \n"
    "    - USERS(u).status = 'ACTIVE'\n"
    "    - AND USERS(u).dept_id IN (サブクエリ-2)\n\n"
    "### 📋 SELECT句\n"
    "- active_users(u).*\n"
    "- (サブクエリ-4) AS order_count\n\n"
    "### 📁 FROM句\n"
    "- active_users AS u\n\n"
    "### 🔍 WHERE句\n"
    "- EXISTS (サブクエリ-3)\n\n"
    "### 📊 ORDER BY句\n"
    "- active_users(u).user_id ASC\n\n"
    "### 🔎 サブクエリ\n"
    "- **サブクエリ-1** [Location: SELECT in CTE active_users]:\n"
    "  - SELECT: DEPARTMENTS(d).dept_name\n"
    "  - FROM: DEPARTMENTS(d)\n"
    "  - WHERE: DEPARTMENTS(d).id = USERS(u).dept_id\n"
    "- **サブクエリ-2** [Location: WHERE in CTE active_users]:\n"
    "  - SELECT: DEPARTMENTS.id\n"
    "  - FROM: DEPARTMENTS\n"
    "  - WHERE: DEPARTMENTS.budget > 10000\n"
    "- **サブクエリ-3** [Location: WHERE in main query]:\n"
    "  - SELECT: 1\n"
    "  - FROM: PAYMENTS(p)\n"
    "  - WHERE: \n"
    "    - PAYMENTS(p).user_id = active_users(u).user_id\n"
    "    - AND PAYMENTS(p).amount > (NESTED-3-1)\n"
    "  - **NESTED-3-1**:\n"
    "    - SELECT: AVG(PAYMENTS.amount)\n"
    "    - FROM: PAYMENTS\n"
    "- **サブクエリ-4** [Location: SELECT in main query]:\n"
    "  - SELECT: COUNT(*)\n"
    "  - FROM: ORDERS(o)\n"
    "  - WHERE: \n"
    "    - ORDERS(o).user_id = active_users(u).user_id\n"
    "    - AND ORDERS(o).status IN (NESTED-4-1)\n"
    "  - **NESTED-4-1**:\n"
    "    - SELECT: ORDER_STATUS.code\n"
    "    - FROM: ORDER_STATUS\n"
    "    - WHERE: ORDER_STATUS.active = 1\n"
)

_TABLE_DF_CACHE = {"df": None, "ts": 0.0}
_VIEW_DF_CACHE = {"df": None, "ts": 0.0}

def _get_table_df_cached(pool, force: bool = False, ttl: int = 120) -> pd.DataFrame:
    try:
        now = time()
        if (not force) and _TABLE_DF_CACHE.get("df") is not None and now - float(_TABLE_DF_CACHE.get("ts", 0.0)) < ttl:
            return _TABLE_DF_CACHE["df"]
        df = get_table_list(pool)
        _TABLE_DF_CACHE["df"] = df
        _TABLE_DF_CACHE["ts"] = now
        return df
    except Exception as e:
        logger.error(f"_get_table_df_cached error: {e}")
        return pd.DataFrame(columns=["Table Name"])  

def _get_view_df_cached(pool, force: bool = False, ttl: int = 120) -> pd.DataFrame:
    try:
        now = time()
        if (not force) and _VIEW_DF_CACHE.get("df") is not None and now - float(_VIEW_DF_CACHE.get("ts", 0.0)) < ttl:
            return _VIEW_DF_CACHE["df"]
        df = get_view_list(pool)
        _VIEW_DF_CACHE["df"] = df
        _VIEW_DF_CACHE["ts"] = now
        return df
    except Exception as e:
        logger.error(f"_get_view_df_cached error: {e}")
        return pd.DataFrame(columns=["View Name"])  

def _get_table_names(pool):
    try:
        df = _get_table_df_cached(pool)
        if not df.empty and "Table Name" in df.columns:
            return df["Table Name"].tolist()
    except Exception as e:
        logger.error(f"_get_table_names error: {e}")
    return []


def _get_view_names(pool):
    try:
        df = _get_view_df_cached(pool)
        if not df.empty and "View Name" in df.columns:
            return df["View Name"].tolist()
    except Exception as e:
        logger.error(f"_get_view_names error: {e}")
    return []


def _profiles_dir() -> Path:
    d = Path("profiles")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize_name(name: str) -> str:
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-ぁ-んァ-ヶ一-龥々ー０-９Ａ-Ｚａ-ｚ]", "", s)
    return s or f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _profile_path(name: str) -> Path:
    return _profiles_dir() / f"{_sanitize_name(name)}.json"


def _save_profiles_to_json(pool):
    """プロファイル情報をselectai.jsonファイルに保存する"""
    try:
        start_ts = time()
        logger.info("プロファイルJSON保存を開始")
        profiles_data = []
        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
        logger.info(f"キャッシュ済みテーブル: {len(table_names)}件 / ビュー: {len(view_names)}件")
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, DESCRIPTION FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
                logger.info(f"DBから{len(rows)}件のProfileを取得")
                for r in rows:
                    try:
                        name = r[0]
                        if str(name).strip().upper() == "OCI_CRED$PROF":
                            continue
                        desc_val = r[1]
                        desc = desc_val.read() if hasattr(desc_val, "read") else str(desc_val or "")
                        logger.info(f"解析中: {name}")
                        attrs = _get_profile_attributes(pool, str(name)) or {}
                        obj_list = attrs.get("object_list") or []
                        logger.info(f"対象オブジェクト: {len(obj_list)}件")
                        tables = []
                        views = []
                        for o in obj_list:
                            try:
                                obj_name = str((o or {}).get("name") or "")
                                if not obj_name:
                                    continue
                                if obj_name in table_names:
                                    tables.append(obj_name)
                                elif obj_name in view_names:
                                    views.append(obj_name)
                            except Exception as e:
                                logger.error(f"_save_profiles_to_json object error: {e}")
                        logger.info(f"解決結果: テーブル{len(tables)}件 / ビュー{len(views)}件")
                        profiles_data.append({
                            "profile": str(name),
                            "category": str(desc),
                            "tables": sorted(set(tables)),
                            "views": sorted(set(views)),
                        })
                    except Exception as e:
                        logger.error(f"_save_profiles_to_json row error: {e}")
        if not profiles_data:
            logger.info("Profileが見つかりません。プレースホルダーを出力")
            profiles_data = [{
                "profile": "",
                "category": "",
                "tables": [],
                "views": [],
            }]
        json_path = _profiles_dir() / "selectai.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        elapsed = time() - start_ts
        logger.info(f"{len(profiles_data)}件のProfileを {json_path} に保存（経過: {elapsed:.3f}s）")
    except Exception as e:
        logger.error(f"_save_profiles_to_json error: {e}")

def _save_profiles_to_json_stream(pool):
    """プロファイル情報保存の進捗を逐次返すジェネレーター"""
    try:
        start_ts = time()
        yield "⏳ プロファイルJSON保存を開始"
        profiles_data = []
        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
        yield f"ℹ️ キャッシュ済みテーブル: {len(table_names)}件 / ビュー: {len(view_names)}件"
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, DESCRIPTION FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
                yield f"ℹ️ DBから{len(rows)}件のProfileを取得"
                for r in rows:
                    try:
                        name = r[0]
                        if str(name).strip().upper() == "OCI_CRED$PROF":
                            continue
                        desc_val = r[1]
                        desc = desc_val.read() if hasattr(desc_val, "read") else str(desc_val or "")
                        yield f"⏳ 解析中: {name}"
                        attrs = _get_profile_attributes(pool, str(name)) or {}
                        obj_list = attrs.get("object_list") or []
                        yield f"ℹ️ 対象オブジェクト: {len(obj_list)}件"
                        tables = []
                        views = []
                        for o in obj_list:
                            try:
                                obj_name = str((o or {}).get("name") or "")
                                if not obj_name:
                                    continue
                                if obj_name in table_names:
                                    tables.append(obj_name)
                                elif obj_name in view_names:
                                    views.append(obj_name)
                            except Exception as e:
                                logger.error(f"_save_profiles_to_json_stream object error: {e}")
                        yield f"✅ {name}: テーブル{len(tables)} / ビュー{len(views)}"
                        profiles_data.append({
                            "profile": str(name),
                            "category": str(desc),
                            "tables": sorted(set(tables)),
                            "views": sorted(set(views)),
                        })
                    except Exception as e:
                        logger.error(f"_save_profiles_to_json_stream row error: {e}")
        if not profiles_data:
            yield "ℹ️ Profileが見つかりません。プレースホルダーを出力"
            profiles_data = [{
                "profile": "",
                "category": "",
                "tables": [],
                "views": [],
            }]
        json_path = _profiles_dir() / "selectai.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        elapsed = time() - start_ts
        yield f"✅ {len(profiles_data)}件のProfileを保存（経過: {elapsed:.1f}s）"
    except Exception as e:
        logger.error(f"_save_profiles_to_json_stream error: {e}")
        yield f"❌ エラー: {e}"


def _save_profile_to_json(pool, name: str, category: str, original_name: str = ""):
    try:
        json_path = _profiles_dir() / "selectai.json"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                profiles_data = json.load(f) or []
        else:
            profiles_data = []
        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
        attrs = _get_profile_attributes(pool, str(name)) or {}
        obj_list = attrs.get("object_list") or []
        tables = []
        views = []
        for o in obj_list:
            obj_name = str((o or {}).get("name") or "")
            if not obj_name:
                continue
            if obj_name in table_names:
                tables.append(obj_name)
            elif obj_name in view_names:
                views.append(obj_name)
        updated = {
            "profile": str(name),
            "category": str(category or ""),
            "tables": sorted(set(tables)),
            "views": sorted(set(views)),
        }
        out = []
        orig = str(original_name or "").strip()
        for p in profiles_data:
            pf = str((p or {}).get("profile", "") or "").strip()
            if pf and pf and pf != str(name).strip() and (not orig or pf != orig):
                out.append(p)
        out.append(updated)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"_save_profile_to_json error: {e}")

def _remove_profile_from_json(name: str):
    try:
        json_path = _profiles_dir() / "selectai.json"
        if not json_path.exists():
            return
        with json_path.open("r", encoding="utf-8") as f:
            profiles_data = json.load(f) or []
        target = str(name or "").strip().lower()
        out = []
        for p in profiles_data:
            pf = str((p or {}).get("profile", "") or "").strip().lower()
            if pf != target:
                out.append(p)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"_remove_profile_from_json error: {e}")


def _load_profiles_from_json():
    try:
        json_path = _profiles_dir() / "selectai.json"
        if not json_path.exists():
            return [("", "")]
        with json_path.open("r", encoding="utf-8") as f:
            profiles_data = json.load(f)
        result = []
        for p in profiles_data:
            bd = str((p or {}).get("category", "") or "").strip()
            pf = str((p or {}).get("profile", "") or "").strip()
            result.append((bd, pf))
        if not result:
            return [("", "")]
        return result
    except Exception as e:
        logger.error(f"_load_profiles_from_json error: {e}")
        return [("", "")]

def _get_profile_json_entry(display_or_name: str) -> dict:
    try:
        s = str(display_or_name or "").strip()
        json_path = _profiles_dir() / "selectai.json"
        if not json_path.exists():
            return {}
        with json_path.open("r", encoding="utf-8") as f:
            profiles = json.load(f) or []
        for p in profiles:
            if str((p or {}).get("profile", "")).strip() == s:
                return p
        for p in profiles:
            if str((p or {}).get("category", "")).strip() == s:
                return p
        return {}
    except Exception:
        return {}

def _get_profile_objects_from_json(display_or_name: str) -> tuple:
    try:
        p = _get_profile_json_entry(display_or_name)
        raw_tables = p.get("tables") or []
        raw_views = p.get("views") or []
        tables = []
        views = []
        for x in raw_tables:
            try:
                if isinstance(x, dict) and x.get("name"):
                    tables.append(str(x.get("name")))
                elif isinstance(x, str):
                    tables.append(str(x))
            except Exception:
                pass
        for x in raw_views:
            try:
                if isinstance(x, dict) and x.get("name"):
                    views.append(str(x.get("name")))
                elif isinstance(x, str):
                    views.append(str(x))
            except Exception:
                pass
        return sorted(set(tables)), sorted(set(views))
    except Exception:
        return [], []

def _get_profile_schema_text_from_json(display_or_name: str) -> str:
    try:
        p = _get_profile_json_entry(display_or_name)
        if not p:
            return ""
        # 仅保存名称时无法生成列级描述，返回空以触发DB回退
        return ""
    except Exception:
        return ""

def _get_profile_context_ddl_from_json(display_or_name: str) -> str:
    try:
        p = _get_profile_json_entry(display_or_name)
        if not p:
            return ""
        # 仅保存名称时不包含DDL，返回空以触发DB回退
        return ""
    except Exception:
        return ""

def _dev_profile_names():
    try:
        pairs = _load_profiles_from_json()
        return [(str(bd), str(pf)) for bd, pf in pairs]
    except Exception:
        return [("", "")]

def _profile_names():
    try:
        pairs = _load_profiles_from_json()
        return [(str(bd), str(pf)) for bd, pf in pairs]
    except Exception:
        return [("", "")]

def _predict_category_label(text):
    try:
        mname = "category"
        sp_root = Path("./models")
        model_path = sp_root / f"{mname}.joblib"
        meta_path = sp_root / f"{mname}.meta.json"
        if not model_path.exists() or not meta_path.exists():
            return ""
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        embed_model = (
            str(meta.get("embed_model", "cohere.embed-v4.0"))
            if isinstance(meta, dict)
            else "cohere.embed-v4.0"
        )
        classifier = joblib.load(model_path)
        embed_text_detail = EmbedTextDetails(
            compartment_id=_COMPARTMENT_ID,
            inputs=[str(text or "")],
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=embed_model),
            truncate="END",
            input_type="CLASSIFICATION",
        )
        embed_text_response = _generative_ai_inference_client.embed_text(embed_text_detail)
        embedding = np.array(embed_text_response.data.embeddings[0])
        prediction = classifier.predict([embedding])
        return str(prediction[0]).strip().lower()
    except Exception:
        return ""

def _map_domain_to_profile(predicted_domain, choices):
    try:
        profile_json_path = Path("./profiles/selectai.json")
        if not predicted_domain or not profile_json_path.exists():
            return gr.Dropdown(choices=choices, value=choices[0][1])
        with profile_json_path.open("r", encoding="utf-8") as f:
            profiles = json.load(f)
        matched_profile = ""
        for item in profiles:
            bd_item = str(item.get("category", "")).strip().lower()
            if bd_item == predicted_domain:
                matched_profile = str(item.get("profile", "")).strip()
                break
        if matched_profile:
            val_lower = matched_profile.strip().lower()
            exists = any(str(val).strip().lower() == val_lower for _, val in choices)
            if not exists:
                bd_display = ""
                for item in profiles:
                    if str(item.get("profile", "")).strip().lower() == val_lower:
                        bd_display = str(item.get("category", "")).strip()
                        break
                choices.append((bd_display, matched_profile))
            return gr.Dropdown(choices=choices, value=matched_profile)
        return gr.Dropdown(choices=choices, value=choices[0][1])
    except Exception:
        return gr.Dropdown(choices=choices, value=choices[0][1])

def get_db_profiles(pool) -> pd.DataFrame:
    try:
        logger.info("DBプロファイル一覧の取得を開始")
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, DESCRIPTION, STATUS FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
                logger.info(f"RAW取得件数: {len(rows)}")
                plain_rows = []
                for r in rows:
                    try:
                        name = r[0]
                        desc_val = r[1]
                        st = r[2]
                        desc = desc_val.read() if hasattr(desc_val, "read") else str(desc_val or "")
                        plain_rows.append([name, desc, st])
                    except Exception:
                        plain_rows.append([str(r[0]), str(r[1] or ""), str(r[2] or "")])
                if plain_rows:
                    df = pd.DataFrame(plain_rows, columns=["Profile Name", "Category", "Status"]).sort_values("Profile Name")
                else:
                    df = pd.DataFrame(columns=["Profile Name", "Category", "Status"]).sort_values("Profile Name")
                df = df[df["Profile Name"].astype(str).str.strip().str.upper() != "OCI_CRED$PROF"]

        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
        logger.info(f"付加情報の解決: tables={len(table_names)}, views={len(view_names)}")
        tables_col = []
        views_col = []
        regions_col = []
        models_col = []
        embed_models_col = []
        for _, r in df.iterrows():
            name = str(r["Profile Name"]) if "Profile Name" in df.columns else str(r.iloc[0])
            attrs = _get_profile_attributes(pool, name) or {}
            obj_list = attrs.get("object_list") or []
            t_list = sorted([o.get("name") for o in obj_list if o.get("name") in table_names])
            v_list = sorted([o.get("name") for o in obj_list if o.get("name") in view_names])
            tables_col.append(", ".join(t_list))
            views_col.append(", ".join(v_list))
            regions_col.append(str(attrs.get("region") or ""))
            models_col.append(str(attrs.get("model") or ""))
            embed_models_col.append(str(attrs.get("embedding_model") or ""))
        if len(df) > 0:
            df.insert(2, "Tables", tables_col)
            df.insert(3, "Views", views_col)
            df.insert(4, "Region", regions_col)
            df.insert(5, "Model", models_col)
            df.insert(6, "Embedding Model", embed_models_col)
            if "Status" in df.columns:
                df = df.drop(columns=["Status"])
        else:
            df = pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])  
        logger.info(f"DBプロファイル一覧の取得完了: {len(df)}件")
        return df
    except Exception as e:
        logger.error(f"get_db_profiles error: {e}")
        return pd.DataFrame(columns=["Profile Name", "Tables", "Views", "Region", "Model", "Embedding Model", "Status"]) 


def _get_profile_attributes(pool, name: str) -> dict:
    attrs = {}
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT ATTRIBUTE_NAME, ATTRIBUTE_VALUE FROM USER_CLOUD_AI_PROFILE_ATTRIBUTES WHERE PROFILE_NAME = :name",
                    name=name,
                )
                rows = cursor.fetchall() or []
                for k, v in rows:
                    try:
                        s = v.read() if hasattr(v, "read") else str(v)
                    except Exception:
                        s = str(v)
                    try:
                        attrs[k.lower()] = json.loads(s)
                    except Exception:
                        attrs[k.lower()] = s
    except Exception as e:
        logger.error(f"_get_profile_attributes error: {e}")
    return attrs


def _resolve_profile_name(pool, display_name: str) -> str:
    try:
        df = get_db_profiles(pool)
        s = str(display_name or "").strip()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "Category" in df.columns:
                m = df[df["Category"].astype(str) == s]
                if len(m) > 0:
                    return str(m.iloc[0]["Profile Name"]) if "Profile Name" in m.columns else str(m.iloc[0][0])
            if "Profile Name" in df.columns:
                m2 = df[df["Profile Name"].astype(str) == s]
                if len(m2) > 0:
                    return s
        return s
    except Exception:
        return str(display_name or "")

def _resolve_profile_name_from_json(pool, display_or_name: str):
    try:
        p = _get_profile_json_entry(display_or_name)
        if p:
            pf = str((p or {}).get("profile", "") or "").strip()
            bd = str((p or {}).get("category", "") or "").strip()
            tables, views = _get_profile_objects_from_json(pf or bd)
            if pf:
                return pf, tables, views, bd
        s = _resolve_profile_name(pool, display_or_name)
        tables, views = _get_profile_objects_from_json(s)
        bd = ""
        return s, tables, views, bd
    except Exception:
        s = _resolve_profile_name(pool, display_or_name)
        return s, [], [], ""


def _generate_create_sql_from_attrs(name: str, attrs: dict, description: str = "") -> str:
    try:
        attr_str = json.dumps(attrs, ensure_ascii=False)
    except Exception as e:
        logger.error(f"_generate_create_sql_from_attrs serialize error: {e}")
        attr_str = "{}"
    desc_str = str(description or "").replace("'", "''")
    sql = (
        f"BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => '{name}'); EXCEPTION WHEN OTHERS THEN NULL; END;\n"
        f"BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => '{name}', attributes => '{attr_str}', description => '{desc_str}'); END;"
    )
    return sql


def delete_profile(name: str) -> None:
    try:
        p = _profile_path(name)
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.error(f"delete_profile error: {e}")


def build_selectai_profile(pool, name, tables, views):
    profile = {
        "profile_name": name or f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tables": [],
        "views": [],
    }

    for t in tables or []:
        try:
            col_df, ddl = get_table_details(pool, t)
            cols = []
            if not col_df.empty:
                for _, row in col_df.iterrows():
                    cols.append({
                        "name": row.get("Column Name"),
                        "type": row.get("Data Type"),
                        "nullable": row.get("Nullable"),
                        "comments": row.get("Comments"),
                    })
            profile["tables"].append({
                "name": t,
                "columns": cols,
                "ddl": ddl,
            })
        except Exception as e:
            logger.warning(f"Failed to load table {t}: {e}")

    for v in views or []:
        try:
            col_df, ddl = get_view_details(pool, v)
            cols = []
            if not col_df.empty:
                for _, row in col_df.iterrows():
                    cols.append({
                        "name": row.get("Column Name"),
                        "type": row.get("Data Type"),
                        "nullable": row.get("Nullable"),
                        "comments": row.get("Comments"),
                    })
            profile["views"].append({
                "name": v,
                "columns": cols,
                "ddl": ddl,
            })
        except Exception as e:
            logger.warning(f"Failed to load view {v}: {e}")

    return json.dumps(profile, ensure_ascii=False, indent=2)


def create_db_profile(
    pool,
    name: str,
    compartment_id: str,
    region: str,
    model: str,
    embedding_model: str,
    max_tokens: int,
    enforce_object_list: bool,
    comments: bool,
    annotations: bool,
    constraints: bool,
    tables: list,
    views: list,
    category: str,
):
    attrs = {
        "provider": "oci",
        "credential_name": "OCI_CRED",
        "oci_compartment_id": compartment_id,
        "region": region,
        "model": model,
        "embedding_model": embedding_model,
        "max_tokens": int(max_tokens) if max_tokens is not None else 1024,
        "enforce_object_list": enforce_object_list,
        "comments": "true" if comments else "false",
        "annotations": "true" if annotations else "false",
        "constraints": "true" if constraints else "false",
        "temperature": 0.0,
        "object_list": [],
    }

    # gpt-* モデルの場合は provider と credential_name を変更
    if str(model).startswith("gpt-"):
        if "provider" in attrs:
            del attrs["provider"]
        # attrs["provider"] = "openai"
        env_path = find_dotenv()
        load_dotenv(env_path, override=True)
        base_url = os.getenv("OPENAI_BASE_URL", "")
        
        # provider_endpoint の整形: プロトコルと /v1 サフィックスを除去
        endpoint = base_url.replace("https://", "").replace("http://", "")
        if endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
            
        if ":" in endpoint:
             endpoint = endpoint.split(":")[0]

        attrs["provider_endpoint"] = endpoint
        attrs["credential_name"] = "OPENAI_CRED"
        # openai provider usually doesn't need region/compartment in attributes 
        # but keeping them might not hurt or might be ignored.
        # However, for safety and cleaner config:
        if "oci_compartment_id" in attrs:
            del attrs["oci_compartment_id"]
        if "region" in attrs:
            del attrs["region"]

    for t in tables or []:
        attrs["object_list"].append({"owner": "ADMIN", "name": t})
    for v in views or []:
        attrs["object_list"].append({"owner": "ADMIN", "name": v})

    attr_str = json.dumps(attrs, ensure_ascii=False)
    
    if "provider_endpoint" in attrs:
        logger.info(f"provider_endpoint: {attrs['provider_endpoint']}")

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    "BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => :name); EXCEPTION WHEN OTHERS THEN NULL; END;",
                    name=name,
                )
            except Exception as e:
                logger.warning(f"DROP_PROFILE failed: {e}")
            cursor.execute(
                "BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => :name, attributes => :attrs, description => :desc); END;",
                name=name,
                attrs=attr_str,
                desc=str(category or ""),
            )
            logger.info(f"Created profile: {name}")


def _predict_domain_and_set_profile(text):
    try:
        ch = _load_profiles_from_json() or [("", "")]
        def _predict_category(text_input: str):
            mname = "category"
            sp_root = Path("./models")
            model_path = sp_root / f"{mname}.joblib"
            meta_path = sp_root / f"{mname}.meta.json"
            if not model_path.exists() or not meta_path.exists():
                return ""
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            embed_model = (
                str(meta.get("embed_model", "cohere.embed-v4.0"))
                if isinstance(meta, dict)
                else "cohere.embed-v4.0"
            )
            classifier = joblib.load(model_path)
            embed_text_detail = EmbedTextDetails(
                compartment_id=_COMPARTMENT_ID,
                inputs=[str(text_input or "")],
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=embed_model),
                truncate="END",
                input_type="CLASSIFICATION",
            )
            embed_text_response = _generative_ai_inference_client.embed_text(embed_text_detail)
            embedding = np.array(embed_text_response.data.embeddings[0])
            prediction = classifier.predict([embedding])
            return str(prediction[0]).strip().lower()

        def _map_domain_to_profile(predicted_domain: str, choices):
            profile_json_path = Path("./profiles/selectai.json")
            if not predicted_domain or not profile_json_path.exists():
                return gr.Dropdown(choices=choices, value=choices[0][1])
            with profile_json_path.open("r", encoding="utf-8") as f:
                profiles = json.load(f)
            matched_profile = ""
            for item in profiles:
                bd_item = str(item.get("category", "")).strip().lower()
                if bd_item == predicted_domain:
                    matched_profile = str(item.get("profile", "")).strip()
                    break
            if matched_profile:
                val_lower = matched_profile.strip().lower()
                exists = any(str(val).strip().lower() == val_lower for _, val in choices)
                if not exists:
                    bd_display = ""
                    for item in profiles:
                        if str(item.get("profile", "")).strip().lower() == val_lower:
                            bd_display = str(item.get("category", "")).strip()
                            break
                    choices.append((bd_display, matched_profile))
                return gr.Dropdown(choices=choices, value=matched_profile)
            return gr.Dropdown(choices=choices, value=choices[0][1])

        pdomain = _predict_category(text)
        return _map_domain_to_profile(pdomain, ch)
    except Exception:
        ch = _load_profiles_from_json() or [("", "")]
        return gr.Dropdown(choices=ch, value=ch[0][1])

def build_selectai_tab(pool):
    with gr.Tabs():
        with gr.TabItem(label="開発者機能"):
            with gr.Tabs():
                with gr.TabItem(label="プロファイル管理"):
                    with gr.Accordion(label="1. プロファイル一覧", open=True):
                        profile_refresh_btn = gr.Button("プロファイル一覧を取得（時間がかかる場合があります）", variant="primary")
                        profile_refresh_status = gr.Markdown(visible=False)
                        profile_list_df = gr.Dataframe(
                            label="プロファイル一覧（件数: 0）",
                            interactive=False,
                            wrap=True,
                            value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"]),
                            headers=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"],
                            visible=False,
                            elem_id="profile_list_df",
                        )
                        profile_list_style = gr.HTML(visible=False)

                    with gr.Accordion(label="2. プロファイル詳細・変更", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("選択されたProfile名*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        selected_profile_name = gr.Textbox(show_label=False, interactive=True, container=False, autoscroll=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("カテゴリ*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        category_text = gr.Textbox(show_label=False, value="", interactive=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Profile 作成SQL", elem_classes="input-label")
                            with gr.Column(scale=9):
                                profile_json_text = gr.Textbox(
                                    show_label=False,
                                    lines=5,
                                    max_lines=10,
                                    show_copy_button=True,
                                    container=False,
                                    autoscroll=False,
                                )
                        selected_profile_original_name = gr.State("")
                        with gr.Row():
                            profile_update_btn = gr.Button("変更を保存", variant="primary")
                            profile_delete_btn = gr.Button("選択したProfileを削除", variant="stop")
                        with gr.Row():
                            profile_action_status = gr.Markdown(visible=False)

                    with gr.Accordion(label="3. プロファイル作成", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Profile名*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        profile_name = gr.Textbox(
                                            show_label=False,
                                            value=f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("カテゴリ*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        category_input = gr.Textbox(show_label=False, placeholder="例: 顧客管理、売上分析 等", container=False, autoscroll=False)

                        with gr.Row():
                            refresh_btn = gr.Button("テーブル・ビュー一覧を取得（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            refresh_status = gr.Markdown(visible=False)

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択*")
                                tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択*")
                                views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)

                        with gr.Row(visible=False):
                            with gr.Column(scale=1):
                                gr.Markdown("OCI Compartment OCID*", elem_classes="input-label")
                            with gr.Column(scale=9):
                                compartment_id_input = gr.Textbox(show_label=False, placeholder="ocid1.compartment.oc1...", value=os.environ.get("OCI_COMPARTMENT_OCID", ""), container=False, autoscroll=False)

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Region*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        region_input = gr.Dropdown(
                                            show_label=False,
                                            choices=["ap-osaka-1", "us-chicago-1"],
                                            value="us-chicago-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Model*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "xai.grok-code-fast-1",
                                                "xai.grok-3",
                                                "xai.grok-3-fast",
                                                "xai.grok-4",
                                                "xai.grok-4-fast-non-reasoning",
                                                "meta.llama-4-scout-17b-16e-instruct",
                                                "gpt-4o",
                                                "gpt-5.1",
                                            ],
                                            value="xai.grok-code-fast-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Max Tokens*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        max_tokens_input = gr.Slider(
                                            show_label=False,
                                            minimum=1000,
                                            maximum=32000,
                                            step=1000,
                                            value=32000,
                                            interactive=True,
                                            container=False,
                                        )

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Embedding Model*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        embedding_model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "cohere.embed-v4.0",
                                            ],
                                            value="cohere.embed-v4.0",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Enforce Object List*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        enforce_object_list_input = gr.Dropdown(
                                            show_label=False,
                                            choices=["true", "false"],
                                            value="true",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Comments*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        comments_input = gr.Dropdown(
                                            show_label=False,
                                            choices=["true", "false"],
                                            value="true",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Annotations*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        annotations_input = gr.Dropdown(
                                            show_label=False,
                                            choices=["true", "false"],
                                            value="true",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Constraints*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        constraints_input = gr.Dropdown(
                                            show_label=False,
                                            choices=["true", "false"],
                                            value="false",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")

                        with gr.Row():
                            build_btn = gr.Button("作成", variant="primary")

                        with gr.Row():
                            create_info = gr.Markdown(visible=False)               

                def refresh_profiles():
                    try:
                        yield gr.Markdown(value="⏳ プロファイル一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])), gr.HTML(visible=False)
                        yield gr.Markdown(value="⏳ DBのプロファイルメタデータを取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])), gr.HTML(visible=False)
                        df = get_db_profiles(pool)
                        yield gr.Markdown(value=f"✅ DBプロファイル取得完了（件数: {0 if df is None else len(df)}）", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])), gr.HTML(visible=False)
                        for msg in _save_profiles_to_json_stream(pool):
                            yield gr.Markdown(value=msg, visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])), gr.HTML(visible=False)
                        if df is None or df.empty:
                            empty_df = pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])
                            count = 0
                            label_text = f"プロファイル一覧（件数: {count}）"
                            yield gr.Markdown(value="✅ 取得完了（データなし）", visible=True), gr.Dataframe(value=empty_df, visible=True, label=label_text), gr.HTML(visible=False)
                            return
                        sample = df.head(5)
                        widths = []
                        for col in sample.columns:
                            series = sample[col].astype(str)
                            row_max = series.map(len).max() if len(series) > 0 else 0
                            length = max(len(str(col)), row_max)
                            widths.append(length)
                        total = sum(widths) if widths else 0
                        style_value = ""
                        if total > 0:
                            col_widths = [max(5, int(100 * w / total)) for w in widths]
                            diff = 100 - sum(col_widths)
                            if diff != 0 and len(col_widths) > 0:
                                col_widths[0] = max(5, col_widths[0] + diff)
                            rules = []
                            rules.append("#profile_list_df { width: 100% !important; }")
                            rules.append("#profile_list_df .wrap { overflow-x: auto !important; }")
                            rules.append("#profile_list_df table { table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; }")
                            for idx, pct in enumerate(col_widths, start=1):
                                rules.append(f"#profile_list_df table th:nth-child({idx}), #profile_list_df table td:nth-child({idx}) {{ width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; }}")
                            style_value = "<style>" + "\n".join(rules) + "</style>"
                        count = len(df)
                        label_text = f"プロファイル一覧（件数: {count}）"
                        yield gr.Markdown(visible=True, value="✅ 取得完了"), gr.Dataframe(value=df, visible=True, label=label_text), gr.HTML(visible=bool(style_value), value=style_value)
                    except Exception as e:
                        logger.error(f"refresh_profiles error: {e}")
                        yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Category", "Tables", "Views", "Region", "Model", "Embedding Model"])), gr.HTML(visible=False)
                
                def on_profile_select(evt: gr.SelectData, current_df, compartment_id):
                    try:
                        logger.info("on_profile_select: invoked")
                        logger.info(f"on_profile_select: event index={evt.index}, value={evt.value}")
                        logger.info(f"on_profile_select: current_df type={type(current_df)}")
                        if isinstance(current_df, dict):
                            try:
                                logger.info("on_profile_select: converting Gradio dict to DataFrame (orient='tight')")
                                current_df = pd.DataFrame.from_dict(current_df, orient='tight')
                            except Exception:
                                logger.info("on_profile_select: fallback converting dict to DataFrame")
                                current_df = pd.DataFrame(current_df)
                        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        logger.info(f"on_profile_select: resolved row_index={row_index}")
                        if len(current_df) > row_index:
                            name = str(current_df.iloc[row_index, 0])
                            logger.info(f"on_profile_select: selected profile name column value='{name}'")
                            attrs = _get_profile_attributes(pool, name) or {}
                            logger.info(f"on_profile_select: fetched attributes keys={list(attrs.keys())}")
                            if compartment_id:
                                attrs.setdefault("oci_compartment_id", compartment_id)
                                logger.info("on_profile_select: set oci_compartment_id from UI state")
                            desc = ""
                            try:
                                with pool.acquire() as conn2:
                                    with conn2.cursor() as cursor2:
                                        logger.info("on_profile_select: querying DESCRIPTION from USER_CLOUD_AI_PROFILES")
                                        cursor2.execute("SELECT DESCRIPTION FROM USER_CLOUD_AI_PROFILES WHERE PROFILE_NAME = :name", name=name)
                                        r2 = cursor2.fetchone()
                                        if r2:
                                            v = r2[0]
                                            desc = v.read() if hasattr(v, "read") else str(v)
                                        logger.info(f"on_profile_select: resolved description length={len(str(desc or ''))}")
                            except Exception:
                                desc = ""
                                logger.info("on_profile_select: description query failed; using empty string")
                            sql = _generate_create_sql_from_attrs(name, attrs, desc)
                            bdn = str(desc or "")
                            logger.info("on_profile_select: returning details for UI update")
                            return name, bdn, sql, name
                    except Exception as e:
                        logger.error(f"on_profile_select error: {e}")
                        return "", "", f"❌ 読み込みエラー: {str(e)}", ""
                    return "", "", "", ""

                def delete_selected_profile(name):
                    try:
                        # DB側も削除
                        with pool.acquire() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => :name); END;", name=name)
                        # JSONファイルから対象のエントリのみ削除
                        _remove_profile_from_json(name)
                        return gr.Markdown(visible=True, value=f"🗑️ 削除しました: {name}"), "", "", ""
                    except Exception as e:
                        logger.error(f"delete_selected_profile error: {e}")
                        return gr.Markdown(visible=True, value=f"❌ 削除に失敗しました: {str(e)}"), name, "", ""

                def update_selected_profile(original_name, edited_name, category):
                    try:
                        orig = str(original_name or "").strip()
                        new = str(edited_name or "").strip()
                        bd = str(category or "").strip()
                        if not orig:
                            attrs = {}
                            sql = _generate_create_sql_from_attrs(new or orig, attrs, bd)
                            return gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), edited_name, gr.Textbox(value=bd, autoscroll=False), sql, (new or orig or "")
                        if not new:
                            new = orig
                        if not bd:
                            attrs = _get_profile_attributes(pool, orig) or {}
                            sql = _generate_create_sql_from_attrs(orig, attrs, "")
                            return gr.Markdown(visible=True, value="⚠️ カテゴリを入力してください"), new, gr.Textbox(value=bd, autoscroll=False), sql, orig
                        attrs = _get_profile_attributes(pool, orig) or {}
                        attr_str = json.dumps(attrs, ensure_ascii=False)
                        with pool.acquire() as conn:
                            with conn.cursor() as cursor:
                                try:
                                    cursor.execute("BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => :name); EXCEPTION WHEN OTHERS THEN NULL; END;", name=new)
                                except Exception as e:
                                    logger.error(f"_am_generate sanitize error: {e}")
                                cursor.execute("BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => :name, attributes => :attrs, description => :desc); END;", name=new, attrs=attr_str, desc=bd)
                                if new != orig:
                                    cursor.execute("BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => :name); END;", name=orig)
                        # JSONファイルを更新（対象のprofileのみ）
                        _save_profile_to_json(pool, new, bd, original_name=orig)
                        sql = _generate_create_sql_from_attrs(new, attrs, bd)
                        return gr.Markdown(visible=True, value=f"✅ 更新しました: {new}"), new, gr.Textbox(value=bd, autoscroll=False), sql, new
                    except Exception as e:
                        logger.error(f"update_selected_profile error: {e}")
                        attrs = _get_profile_attributes(pool, orig or edited_name) or {}
                        sql = _generate_create_sql_from_attrs(new or orig, attrs, bd)
                        return gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {str(e)}"), edited_name, gr.Textbox(value=bd, autoscroll=False), sql, (new or orig or "")

                def build_profile(name, tables, views, compartment_id, region, model, embedding_model, max_tokens, enforce_object_list, comments, annotations, constraints, category):
                    if not tables and not views:
                        yield gr.Markdown(visible=True, value="⚠️ テーブルまたはビューを選択してください")
                        return
                    bd = str(category or "").strip()
                    if not bd:
                        yield gr.Markdown(visible=True, value="⚠️ カテゴリを入力してください")
                        return
                    try:
                        yield gr.Markdown(visible=True, value="⏳ 作成中...")
                        bool_map = {"true": True, "false": False}
                        eol = bool_map.get(str(enforce_object_list).lower(), True)
                        com = bool_map.get(str(comments).lower(), True)
                        ann = bool_map.get(str(annotations).lower(), True)
                        con = bool_map.get(str(constraints).lower(), True)
                        create_db_profile(
                            pool,
                            name,
                            compartment_id,
                            region,
                            model,
                            embedding_model,
                            int(max_tokens) if max_tokens is not None else 1024,
                            eol,
                            com,
                            ann,
                            con,
                            tables or [],
                            views or [],
                            str(category or ""),
                        )
                        # JSONファイルを更新
                        _save_profiles_to_json(pool)
                        yield gr.Markdown(visible=True, value=f"✅ 作成しました: {name}")
                    except Exception as e:
                        msg = f"❌ 作成に失敗しました: {str(e)}"
                        # gpt-* モデルで provider_endpoint が原因のエラーの場合、値を表示
                        if str(model).startswith("gpt-") and "provider_endpoint" in str(e):
                            env_path = find_dotenv()
                            load_dotenv(env_path, override=True)
                            base_url = os.getenv("OPENAI_BASE_URL", "")
                            
                            endpoint = base_url.replace("https://", "").replace("http://", "")
                            if endpoint.endswith("/v1"):
                                endpoint = endpoint[:-3]
                            if endpoint.endswith("/"):
                                endpoint = endpoint[:-1]
                            msg += f"\n\nprovider_endpoint: {endpoint}"
                        
                        yield gr.Markdown(visible=True, value=msg)

                profile_refresh_btn.click(
                    fn=refresh_profiles,
                    outputs=[profile_refresh_status, profile_list_df, profile_list_style],
                )

                profile_list_df.select(
                    fn=on_profile_select,
                    inputs=[profile_list_df, compartment_id_input],
                    outputs=[selected_profile_name, category_text, profile_json_text, selected_profile_original_name],
                )

                def _delete_profile_handler(name):
                    try:
                        logger.info(f"_delete_profile_handler: invoked name='{name}'")
                        yield gr.Markdown(visible=True, value="⏳ 削除中..."), name, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                        logger.info("_delete_profile_handler: calling delete_selected_profile")
                        md, sel_name, bd_text, json_text = delete_selected_profile(name)
                        logger.info(f"_delete_profile_handler: delete done sel_name='{sel_name}'")
                        yield md, sel_name, bd_text, json_text
                    except Exception as e:
                        logger.error(f"_delete_profile_handler error: {e}")
                        yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), name, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)

                def _update_profile_handler(original_name, edited_name, category):
                    try:
                        logger.info(f"_update_profile_handler: invoked original='{original_name}', edited='{edited_name}'")
                        yield gr.Markdown(visible=True, value="⏳ 更新中..."), edited_name, gr.Textbox(value=category), gr.Textbox(value="", autoscroll=False), original_name
                        logger.info("_update_profile_handler: calling update_selected_profile")
                        md, sel_name, bd_text, sql_text, orig_out = update_selected_profile(original_name, edited_name, category)
                        logger.info(f"_update_profile_handler: update done sel_name='{sel_name}', orig_out='{orig_out}'")
                        yield md, sel_name, bd_text, sql_text, orig_out
                    except Exception as e:
                        logger.error(f"_update_profile_handler error: {e}")
                        yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), edited_name, gr.Textbox(value=category), gr.Textbox(value="", autoscroll=False), original_name

                profile_delete_btn.click(
                    fn=_delete_profile_handler,
                    inputs=[selected_profile_name],
                    outputs=[profile_action_status, selected_profile_name, category_text, profile_json_text],
                ).then(
                    fn=refresh_profiles,
                    outputs=[profile_refresh_status, profile_list_df, profile_list_style],
                )

                profile_update_btn.click(
                    fn=_update_profile_handler,
                    inputs=[selected_profile_original_name, selected_profile_name, category_text],
                    outputs=[profile_action_status, selected_profile_name, category_text, profile_json_text, selected_profile_original_name],
                ).then(
                    fn=refresh_profiles,
                    outputs=[profile_refresh_status, profile_list_df, profile_list_style],
                )

                def refresh_sources_handler():
                    try:
                        yield gr.Markdown(visible=True, value="⏳ テーブル・ビュー一覧を取得中..."), gr.CheckboxGroup(visible=False, choices=[]), gr.CheckboxGroup(visible=False, choices=[])
                        t = _get_table_names(pool)
                        v = _get_view_names(pool)
                        status_text = "✅ 取得完了（データなし）" if (not t and not v) else "✅ 取得完了"
                        yield gr.Markdown(visible=True, value=status_text), gr.CheckboxGroup(choices=t, visible=True), gr.CheckboxGroup(choices=v, visible=True)
                    except Exception as e:
                        logger.error(f"refresh_sources_handler error: {e}")
                        yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

                refresh_btn.click(
                    fn=refresh_sources_handler,
                    outputs=[refresh_status, tables_input, views_input],
                )

                build_btn.click(
                    fn=build_profile,
                    inputs=[
                        profile_name,
                        tables_input,
                        views_input,
                        compartment_id_input,
                        region_input,
                        model_input,
                        embedding_model_input,
                        max_tokens_input,
                        enforce_object_list_input,
                        comments_input,
                        annotations_input,
                        constraints_input,
                        category_input,
                    ],
                    outputs=[create_info],
                ).then(
                    fn=refresh_profiles,
                    outputs=[profile_refresh_status, profile_list_df, profile_list_style],
                )

                def _on_model_change(m):
                    s = str(m or "")
                    if s.startswith("gpt-"):
                        choices = [
                            "text-embedding-3-large",
                        ]
                        v = "text-embedding-3-large"
                    else:
                        choices = [
                            "cohere.embed-v4.0",
                        ]
                        v = "cohere.embed-v4.0"
                    return gr.Dropdown(choices=choices, value=v)

                model_input.change(
                    _on_model_change,
                    inputs=model_input,
                    outputs=embedding_model_input,
                )

                def _profile_names():
                    try:
                        df = get_db_profiles(pool)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            return [str(x) for x in df["Profile Name"].tolist()]
                    except Exception as e:
                        logger.error(f"_profile_names error: {e}")
                    return []

                def _td_list():
                    try:
                        p = Path("uploads") / "training_data.xlsx"
                        if not p.exists():
                            return pd.DataFrame(columns=["CATEGORY","TEXT"])
                        df = pd.read_excel(str(p))
                        cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                        bd_col = cols_map.get("CATEGORY")
                        tx_col = cols_map.get("TEXT")
                        if not bd_col or not tx_col:
                            return pd.DataFrame(columns=["CATEGORY","TEXT"])
                        out = pd.DataFrame({"CATEGORY": df[bd_col].astype(str), "TEXT": df[tx_col].astype(str)})
                        return out
                    except Exception as e:
                        logger.error(f"訓練データ一覧の取得に失敗しました: {e}")
                        return pd.DataFrame(columns=["CATEGORY","TEXT"])                    

                def _td_refresh():
                    try:
                        yield gr.Markdown(visible=True, value="⏳ 訓練データ一覧を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame())
                        df = _td_list()
                        if df is None or df.empty:
                            count = 0
                            label_text = f"訓練データ一覧（件数: {count}）"
                            yield gr.Markdown(visible=True, value="✅ 取得完了（データなし）"), gr.Dataframe(visible=True, value=pd.DataFrame(columns=["CATEGORY","TEXT"]), label=label_text)
                            return
                        try:
                            df_disp = df.copy()
                            df_disp["TEXT"] = df_disp["TEXT"].astype(str).map(lambda s: s if len(s) <= 200 else (s[:200] + " ..."))
                        except Exception as e:
                            logger.error(f"build training data preview failed: {e}")
                            df_disp = df
                        count = len(df_disp)
                        label_text = f"訓練データ一覧（件数: {count}）"
                        yield gr.Markdown(visible=True, value="✅ 取得完了"), gr.Dataframe(visible=True, value=df_disp, label=label_text)
                    except Exception as e:
                        yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame())

                def _td_train(embed_model):
                    """参照コード(No.1-Classifier)に基づいた分類器訓練関数"""
                    try:
                        # 固定のモデル名を使用
                        model_name = "category"
                        iterations = 1000  # デフォルト値
                        
                        logger.info("="*50)
                        logger.info("Starting classifier training...")
                        logger.info(f"Embed model: {embed_model}")
                        logger.info(f"Model name: {model_name}")
                        logger.info(f"Iterations: {iterations}")
                        
                        # OCI GenAI クライアントの確認
                        if not _generative_ai_inference_client or not _COMPARTMENT_ID:
                            error_msg = "OCI GenAI クライアントが初期化されていません。環境変数を確認してください"
                            logger.error(error_msg)
                            logger.error(f"Client initialized: {_generative_ai_inference_client is not None}")
                            logger.error(f"Compartment ID set: {_COMPARTMENT_ID is not None}")
                            yield gr.Markdown(visible=True, value=f"❌ {error_msg}")
                            return
                        
                        logger.info("OCI GenAI client check passed")
                        yield gr.Markdown(visible=True, value="⏳ 学習開始")
                        
                        # 訓練データの読み込み
                        p = Path("uploads") / "training_data.xlsx"
                        logger.info(f"Loading training data from: {p}")
                        
                        if not p.exists():
                            error_msg = "訓練データファイルが存在しません"
                            logger.error(f"{error_msg}: {p}")
                            yield gr.Markdown(visible=True, value=f"⚠️ {error_msg}")
                            return
                        
                        logger.info("Reading Excel file...")
                        df = pd.read_excel(str(p))
                        logger.info(f"Excel file loaded, shape: {df.shape}")
                        
                        cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                        logger.info(f"Columns found: {list(cols_map.keys())}")
                        
                        bd_col = cols_map.get("CATEGORY")
                        tx_col = cols_map.get("TEXT")
                        
                        if not bd_col or not tx_col:
                            error_msg = "必須列(CATEGORY, TEXT)が見つかりません"
                            logger.error(error_msg)
                            logger.error(f"Available columns: {list(cols_map.keys())}")
                            yield gr.Markdown(visible=True, value=f"⚠️ {error_msg}")
                            return
                        
                        logger.info(f"Using columns - CATEGORY: {bd_col}, TEXT: {tx_col}")
                        
                        texts = []
                        labels = []
                        for _, r in df.iterrows():
                            s_txt = str(r.get(tx_col, "") or "")
                            s_bd = str(r.get(bd_col, "") or "")
                            if s_txt:
                                texts.append(s_txt)
                                labels.append(s_bd)
                        
                        if not texts or not labels:
                            error_msg = "訓練データがありません"
                            logger.error(error_msg)
                            yield gr.Markdown(visible=True, value=f"⚠️ {error_msg}")
                            return
                        
                        unique_labels = list(set(labels))
                        logger.info(f"Training data loaded: {len(texts)} samples, {len(unique_labels)} unique labels")
                        logger.info(f"Labels: {unique_labels}")
                        
                        yield gr.Markdown(visible=True, value=f"⏳ 訓練データ読み込み完了: {len(texts)}件")
                        
                        # モデル保存パスの準備
                        sp_root = Path("./models")
                        sp_root.mkdir(parents=True, exist_ok=True)
                        mname = str(model_name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}").strip()
                        model_path = sp_root / f"{mname}.joblib"
                        
                        logger.info(f"Model will be saved to: {model_path}")
                        
                        yield gr.Markdown(visible=True, value="⏳ 埋め込みベクトルを取得中...")
                        
                        # 埋め込みベクトルの取得(参照コードに基づく)
                        logger.info("Creating embedding request...")
                        logger.info(f"Using compartment ID: {_COMPARTMENT_ID[:20]}...")
                        logger.info(f"Using model: {embed_model or 'cohere.embed-v4.0'}")
                        logger.info(f"Number of texts to embed: {len(texts)}")
                        
                        embed_text_detail = EmbedTextDetails(
                            compartment_id=_COMPARTMENT_ID,
                            inputs=texts,
                            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                                model_id=str(embed_model or "cohere.embed-v4.0")
                            ),
                            truncate="END",
                            input_type="CLASSIFICATION"
                        )
                        
                        logger.info("Sending embedding request to OCI GenAI...")
                        embed_text_response = _generative_ai_inference_client.embed_text(embed_text_detail)
                        logger.info("Embedding response received")
                        
                        embeddings = np.array(embed_text_response.data.embeddings)
                        logger.info(f"Embeddings shape: {embeddings.shape}")
                        
                        yield gr.Markdown(visible=True, value=f"⏳ 埋め込み取得完了: {embeddings.shape}")
                        
                        # 学習回数の処理
                        try:
                            iters = int(iterations or 1)
                        except Exception:
                            iters = 1
                        
                        logger.info(f"Training iterations: {iters}")
                        
                        # LogisticRegressionによる訓練(参照コードに基づく)
                        max_iter = max(1000, iters * 100)
                        logger.info(f"Training LogisticRegression classifier with max_iter={max_iter}")
                        yield gr.Markdown(visible=True, value=f"⏳ 分類器を訓練中(max_iter={max_iter})...")
                        
                        classifier = LogisticRegression(max_iter=max_iter)
                        classifier.fit(embeddings, labels)
                        
                        logger.info("Classifier training completed")
                        logger.info(f"Classifier classes: {classifier.classes_}")
                        
                        # モデルの保存
                        logger.info(f"Saving model to: {model_path}")
                        joblib.dump(classifier, model_path)
                        logger.info("Model saved successfully")
                        
                        # メタ情報の保存
                        meta_path = sp_root / f"{mname}.meta.json"
                        logger.info(f"Saving metadata to: {meta_path}")
                        meta_info = {
                            "model_name": mname,
                            "labels": sorted(list(set(labels))),
                            "samples": len(texts),
                            "embed_model": str(embed_model or "cohere.embed-v4.0"),
                            "iterations": iters,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "algorithm": "LogisticRegression"
                        }
                        with meta_path.open("w", encoding="utf-8") as f:
                            json.dump(meta_info, f, ensure_ascii=False, indent=2)
                        logger.info("Metadata saved")
                        
                        success_msg = "✅ 学習が完了しました。モデルを保存しました。"
                        logger.info(success_msg)
                        logger.info("="*50)
                        yield gr.Markdown(visible=True, value=success_msg)
                        
                    except Exception as e:
                        error_msg = f"学習に失敗しました: {e}"
                        logger.error(error_msg)
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.info("="*50)
                        yield gr.Markdown(visible=True, value=f"❌ {error_msg}")

                async def _mt_test_async(text, trained_model_name):
                    """参照コード(No.1-Classifier)に基づいた予測関数"""
                    try:
                        # 固定のモデル名を使用
                        trained_model_name = "category"
                                        
                        logger.info("="*50)
                        logger.info("Starting model prediction...")
                        logger.info(f"Model name: {trained_model_name}")
                        logger.info(f"Input text length: {len(str(text or ''))}")
                        
                        # OCI GenAI クライアントの確認
                        if not _generative_ai_inference_client or not _COMPARTMENT_ID:
                            error_msg = "OCI GenAI クライアントが初期化されていません。環境変数を確認してください"
                            logger.error(error_msg)
                            return gr.Markdown(visible=True, value=f"❌ {error_msg}"), gr.Textbox(value="", autoscroll=False)
                        
                        logger.info("OCI GenAI client check passed")
                        
                        sp_root = Path("./models")
                        mname = str(trained_model_name or "").strip()
                        if not mname:
                            logger.warning("モデルが選択されていません")
                            return gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="", autoscroll=False)
                        
                        logger.info(f"Using model: {mname}")
                        
                        model_path = sp_root / f"{mname}.joblib"
                        meta_path = sp_root / f"{mname}.meta.json"
                        
                        logger.info(f"Model path: {model_path}")
                        logger.info(f"Meta path: {meta_path}")
                        
                        if not model_path.exists() or not meta_path.exists():
                            error_msg = f"モデルファイルが見つかりません (model: {model_path.exists()}, meta: {meta_path.exists()})"
                            logger.error(error_msg)
                            return gr.Markdown(visible=True, value="ℹ️ モデルが未学習です。まず『学習を実行』してください"), gr.Textbox(value="", autoscroll=False)
                        
                        # メタ情報を読み込み
                        logger.info("Loading model metadata...")
                        with meta_path.open("r", encoding="utf-8") as f:
                            meta = json.load(f)
                        
                        embed_model = str(meta.get("embed_model", "cohere.embed-v4.0"))
                        logger.info(f"Embed model: {embed_model}")
                        logger.info(f"Model labels: {meta.get('labels', [])}")
                        
                        # モデルを読み込み
                        logger.info("Loading classifier model...")
                        classifier = joblib.load(model_path)
                        logger.info(f"Classifier loaded, classes: {classifier.classes_}")
                        
                        # テキストの埋め込みベクトルを取得(参照コードに基づく)
                        logger.info("Creating embedding request for input text...")
                        embed_text_detail = EmbedTextDetails(
                            compartment_id=_COMPARTMENT_ID,
                            inputs=[str(text or "")],
                            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                                model_id=embed_model
                            ),
                            truncate="END",
                            input_type="CLASSIFICATION"
                        )
                        
                        logger.info("Sending embedding request to OCI GenAI...")
                        embed_text_response = _generative_ai_inference_client.embed_text(embed_text_detail)
                        logger.info("Embedding response received")
                        
                        embedding = np.array(embed_text_response.data.embeddings[0])
                        logger.info(f"Embedding shape: {embedding.shape}")
                        
                        # 予測を実行(参照コードに基づく)
                        logger.info("Making prediction...")
                        prediction = classifier.predict([embedding])
                        probabilities = classifier.predict_proba([embedding])
                        
                        # 結果を整形
                        pred = prediction[0]
                        probs = dict(zip(classifier.classes_, probabilities[0].round(3).tolist()))
                        
                        logger.info(f"Prediction: {pred}")
                        logger.info(f"Probabilities: {probs}")
                        
                        lines = [
                            f"prediction: {pred}",
                            "probabilities: " + json.dumps({k: round(v, 3) for k, v in probs.items()}, ensure_ascii=False),
                        ]
                        
                        logger.info("Prediction completed successfully")
                        logger.info("="*50)
                        return gr.Markdown(visible=True, value="\n".join(lines)), gr.Textbox(value=pred, autoscroll=False)
                        
                    except Exception as e:
                        error_msg = f"テストに失敗しました: {e}"
                        logger.error(error_msg)
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.info("="*50)
                        return gr.Markdown(visible=True, value=f"❌ {error_msg}"), gr.Textbox(value="", autoscroll=False)

                def _mt_test(text):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # 固定のモデル名を渡す
                        return loop.run_until_complete(_mt_test_async(text, "category"))
                    finally:
                        loop.close()

                def _td_upload_excel(file_path):
                    try:
                        if not file_path:
                            return gr.Textbox(visible=True, value="ファイルを選択してください", autoscroll=False)
                        try:
                            df = pd.read_excel(str(file_path))
                        except Exception:
                            return gr.Textbox(visible=True, value="Excel読み込みに失敗しました", autoscroll=False)
                        cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                        required = {"CATEGORY","TEXT"}
                        if not required.issubset(set(cols_map.keys())):
                            return gr.Textbox(visible=True, value="列名は CATEGORY, TEXT が必要です", autoscroll=False)
                        out_df = pd.DataFrame({
                            "CATEGORY": df[cols_map["CATEGORY"]],
                            "TEXT": df[cols_map["TEXT"]],
                        })
                        up_dir = Path("uploads")
                        up_dir.mkdir(parents=True, exist_ok=True)
                        dest = up_dir / "training_data.xlsx"
                        if dest.exists():
                            dest.unlink()
                        with pd.ExcelWriter(dest) as writer:
                            out_df.to_excel(writer, sheet_name="training_data", index=False)
                        return gr.Textbox(visible=True, value=f"✅ アップロード完了: {len(out_df)} 件", autoscroll=False)
                    except Exception as e:
                        logger.error(f"Excelアップロードに失敗しました: {e}")
                        return gr.Textbox(visible=True, value=f"❌ エラー: {e}", autoscroll=False)

                with gr.TabItem(label="モデル管理"):
                    with gr.Accordion(label="0. モデル学習の概要", open=False):
                        gr.Markdown(
                            """
                            目的: 文章から最適な「カテゴリ」を自動判定できるようにします。

                            手順:
                            - 訓練データアップロード: Excelに `CATEGORY` と `TEXT` の2列を用意し、アップロードします。
                            - 埋め込みモデル選択: `cohere.embed-v4.0` で文章を数値化します。
                            - 学習実行: 数値化したデータで分類器を作成し、モデルを保存します。
                              仕組み: 文章→埋め込みベクトル（OCI）→ロジスティック回帰（scikit-learn）でカテゴリを判定します。
                            - テスト: 文章を入力して、予測カテゴリと確率を表示します。

                            注意:
                            - OCI設定（`OCI_COMPARTMENT_OCID`）が必要です。
                            - 各カテゴリの件数は偏りなく十分に用意してください（目安: 10件以上）。
                            - 個人情報や機密情報は含めないでください。
                            """
                        )
                    with gr.Accordion(label="1. 訓練データ一覧（必須列: CATEGORY, TEXT）", open=True):
                        with gr.Row():
                            td_refresh_btn = gr.Button("訓練データ一覧を取得", variant="primary")
                        with gr.Row():
                            td_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            td_list_df = gr.Dataframe(label="訓練データ一覧", interactive=False, wrap=True, visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Excelファイル*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                td_upload_excel_file = gr.File(show_label=False, file_types=[".xlsx"], type="filepath")
                        with gr.Row():
                            with gr.Column():
                                gr.DownloadButton(label="Excelダウンロード", value="./uploads/training_data.xlsx", variant="secondary")
                            with gr.Column():
                                td_upload_excel_btn = gr.Button("Excelアップロード(全削除&挿入)", variant="stop")
                        with gr.Row():
                            td_upload_result = gr.Textbox(visible=False)
                    with gr.Accordion(label="2. モデル学習", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("埋め込みモデル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        td_embed_model = gr.Dropdown(
                                            show_label=False,
                                            choices=["cohere.embed-v4.0"],
                                            value="cohere.embed-v4.0",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            td_train_btn = gr.Button("学習を実行", variant="primary")
                        with gr.Row():
                            td_train_status = gr.Markdown(visible=False)
                    with gr.Accordion(label="3. モデルテスト", open=True):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("テキスト*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                mt_text_input = gr.Textbox(show_label=False, lines=4, max_lines=8, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("カテゴリ", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        mt_label_text = gr.Textbox(show_label=False, interactive=False, container=False, autoscroll=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            mt_test_btn = gr.Button("テスト", variant="primary")
                        with gr.Row():                            
                            mt_test_result = gr.Markdown(visible=False)

                    td_refresh_btn.click(
                        fn=_td_refresh,
                        outputs=[td_refresh_status, td_list_df],
                    )
                    td_upload_excel_btn.click(
                        fn=_td_upload_excel,
                        inputs=[td_upload_excel_file],
                        outputs=[td_upload_result],
                    )
                    td_train_btn.click(
                        fn=_td_train,
                        inputs=[td_embed_model],
                        outputs=[td_train_status],
                    )
                    mt_test_btn.click(
                        fn=_mt_test,
                        inputs=[mt_text_input],
                        outputs=[mt_test_result, mt_label_text],
                    )

                with gr.TabItem(label="用語集管理"):
                    with gr.Accordion(label="0. 用語集の概要", open=False):
                        gr.Markdown(
                            """
                            目的: 組織で使う用語を一元管理し、チャット/分析で参照できるようにします。

                            手順:
                            - 用語集Excelをダウンロードし、`TERM` と `DEFINITION` の2列を記入します。
                            - 用語集Excelをアップロードすると、`uploads/terms.xlsx` に保存されます。
                            - 「用語集をプレビュー」で内容と件数を確認します。

                            注意:
                            - 列名は必ず `TERM`, `DEFINITION` を使用してください。
                            - 文字列以外の値は保存時に文字列化されます。
                            - 個人情報や機密情報は含めないでください。
                            """
                        )
                    with gr.Accordion(label="1. 用語集", open=True):
                        # 用語集Excelのテンプレートファイルを事前作成し、そのままダウンロード可能にする
                        up_dir = Path("uploads")
                        up_dir.mkdir(parents=True, exist_ok=True)
                        _p = up_dir / "terms.xlsx"
                        if not _p.exists():
                            _df = pd.DataFrame(columns=["TERM", "DEFINITION"])
                            with pd.ExcelWriter(_p) as _writer:
                                _df.to_excel(_writer, sheet_name="terms", index=False)
    
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("ℹ️ ファイルをドロップすると自動的にアップロードされます")
                            
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("用語集Excelをアップロード*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                term_upload_file = gr.File(show_label=False, file_types=[".xlsx"], type="filepath", container=True)
                        with gr.Row():
                            term_upload_result = gr.Textbox(label="アップロード結果", interactive=False, visible=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column():
                                gr.DownloadButton(label="用語集Excelをダウンロード", value=str(_p), variant="secondary")
                            with gr.Column():
                                term_preview_btn = gr.Button("用語集をプレビュー", variant="primary")
                        with gr.Row():
                            term_preview_status = gr.Markdown(visible=False)
                        with gr.Row():
                            term_preview_df = gr.Dataframe(
                                label="用語集プレビュー（件数: 0）",
                                interactive=False,
                                wrap=True,
                                visible=False,
                                value=pd.DataFrame(columns=["TERM", "DEFINITION"]),
                            )

                    def _term_list():
                        try:
                            p = Path("uploads") / "terms.xlsx"
                            if not p.exists():
                                return pd.DataFrame(columns=["TERM", "DEFINITION"])
                            df = pd.read_excel(str(p))
                            cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                            t_col = cols_map.get("TERM")
                            d_col = cols_map.get("DEFINITION")
                            if not t_col or not d_col:
                                return pd.DataFrame(columns=["TERM", "DEFINITION"])
                            out = pd.DataFrame({
                                "TERM": df[t_col].astype(str),
                                "DEFINITION": df[d_col].astype(str),
                            })
                            return out
                        except Exception as e:
                            logger.error(f"用語集一覧の取得に失敗しました: {e}")
                            return pd.DataFrame(columns=["TERM", "DEFINITION"])

                    def _term_refresh():
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 用語集を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame())
                            df = _term_list()
                            if df is None or df.empty:
                                yield gr.Markdown(visible=True, value="✅ 取得完了（データなし）"), gr.Dataframe(visible=True, value=pd.DataFrame(columns=["TERM", "DEFINITION"]), label="用語集プレビュー（件数: 0）")
                                return
                            yield gr.Markdown(visible=True, value="✅ 取得完了"), gr.Dataframe(visible=True, value=df, label=f"用語集プレビュー（件数: {len(df)}）")
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame())

                    def _term_upload_excel(file_path):
                        try:
                            if not file_path:
                                return gr.Textbox(visible=True, value="ファイルを選択してください", autoscroll=False)
                            try:
                                df = pd.read_excel(str(file_path))
                            except Exception:
                                return gr.Textbox(visible=True, value="Excel読み込みに失敗しました", autoscroll=False)
                            cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                            required = {"TERM", "DEFINITION"}
                            if not required.issubset(set(cols_map.keys())):
                                return gr.Textbox(visible=True, value="列名は TERM, DESCRIPTION が必要です", autoscroll=False)
                            out_df = pd.DataFrame({
                                "TERM": df[cols_map["TERM"]],
                                "DEFINITION": df[cols_map["DEFINITION"]],
                            })
                            up_dir = Path("uploads")
                            up_dir.mkdir(parents=True, exist_ok=True)
                            dest = up_dir / "terms.xlsx"
                            if dest.exists():
                                dest.unlink()
                            with pd.ExcelWriter(dest) as writer:
                                out_df.to_excel(writer, sheet_name="terms", index=False)
                            return gr.Textbox(visible=True, value=f"✅ アップロード完了: {len(out_df)} 件", autoscroll=False)
                        except Exception as e:
                            logger.error(f"用語集Excelアップロードに失敗しました: {e}")
                            return gr.Textbox(visible=True, value=f"❌ エラー: {e}", autoscroll=False)

                    term_preview_btn.click(
                        fn=_term_refresh,
                        outputs=[term_preview_status, term_preview_df],
                    )

                    # ダウンロードはボタン自体で実行（クリックハンドラ不要）
                    term_upload_file.change(
                        fn=_term_upload_excel,
                        inputs=[term_upload_file],
                        outputs=[term_upload_result],
                    )

                with gr.TabItem(label="チャット・分析") as dev_chat_tab:
                    with gr.Accordion(label="1. チャット", open=True):
                        def _dev_profile_names():
                            try:
                                pairs = _load_profiles_from_json()
                                [str(bd) for bd, _ in pairs]
                                # Gradio Dropdown supports label/value pairs via choices=[(label,value),...]
                                # We return pairs so that display is category, value is profile
                                return [(str(bd), str(pf)) for bd, pf in pairs]
                            except Exception as e:
                                logger.error(f"_dev_profile_names error: {e}")
                            return [("", "")]

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("自然言語の質問*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_prompt_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="例: 東京の顧客数を教えて",
                                    lines=3,
                                    max_lines=10,
                                    show_copy_button=True,
                                    container=False,
                                    autoscroll=False,
                                )

                        with gr.Row():
                            with gr.Column(scale=5):
                                dev_predict_domain_btn = gr.Button("カテゴリ予測 ⇒", variant="secondary")
                            with gr.Column(scale=5):
                                # プロフィール選択肢を取得し、空の場合は空文字列を含むリストを設定
                                _dev_initial_choices = _dev_profile_names()
                                if not _dev_initial_choices:
                                    _dev_initial_choices = [("", "")]
                                dev_profile_select = gr.Dropdown(
                                    show_label=False,
                                    choices=_dev_initial_choices,
                                    value=_dev_initial_choices[0][1] if _dev_initial_choices and isinstance(_dev_initial_choices[0], tuple) else "",
                                    interactive=True,
                                    container=False,
                                )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("クエリ書き換えを有効化", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_enable_query_rewrite = gr.Checkbox(label="", value=False, container=False)
                        
                        with gr.Row():
                            with gr.Accordion(label="クエリ書き換え設定", open=True, visible=False) as dev_query_rewrite_section:
                                with gr.Row():
                                    with gr.Column(scale=5):
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                gr.Markdown("書き換え用モデル*", elem_classes="input-label")
                                            with gr.Column(scale=5):
                                                dev_rewrite_model_select = gr.Dropdown(
                                                    show_label=False,
                                                    choices=[
                                                        "xai.grok-code-fast-1",
                                                        "xai.grok-3",
                                                        "xai.grok-3-fast",
                                                        "xai.grok-4",
                                                        "xai.grok-4-fast-non-reasoning",
                                                        "meta.llama-4-scout-17b-16e-instruct",
                                                        "gpt-4o",
                                                        "gpt-5.1",
                                                    ],
                                                    value="xai.grok-code-fast-1",
                                                    interactive=True,
                                                    container=False,
                                                )
                                    with gr.Column(scale=5):
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                gr.Markdown("")
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("ステップ1: 用語集を利用", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        dev_rewrite_use_glossary = gr.Checkbox(label="", value=True, container=False)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("ステップ2: スキーマ情報を利用", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        dev_rewrite_use_schema = gr.Checkbox(label="", value=False, container=False)
                                with gr.Row():
                                    gr.Markdown(
                                        "ℹ️ 「ステップ1: 用語集を利用」または「ステップ2: スキーマ情報を利用」のいずれかをONにしてください。両方OFFの場合、書き換えは実行されません。",
                                        elem_classes="input-hint",
                                    )
                                with gr.Row():
                                    dev_rewrite_btn = gr.Button("書き換え実行", variant="primary")
                                with gr.Row():
                                    dev_rewrite_status = gr.Markdown(visible=False)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("書き換え後の質問", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        dev_rewritten_query = gr.Textbox(
                                            show_label=False,
                                            lines=5,
                                            max_lines=10,
                                            interactive=True,
                                            show_copy_button=True,
                                            container=False,
                                            autoscroll=False,
                                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("追加指示・例示を使用", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_include_extra_prompt = gr.Checkbox(label="", value=False, container=False)

                        with gr.Row():
                            with gr.Accordion(label="追加指示・例示を設定", open=True, visible=False) as dev_extra_prompt_section:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        dev_extra_prompt = gr.Textbox(
                                            show_label=False,
                                            value=_DEFAULT_FEW_SHOT_PROMPT,
                                            lines=15,
                                            max_lines=15,
                                            show_copy_button=True,
                                            autoscroll=False,
                                            container=False,
                                        )
                            dev_include_extra_prompt.change(lambda v: gr.Accordion(visible=v), inputs=dev_include_extra_prompt, outputs=dev_extra_prompt_section)
                        
                        # Query転写のCheckbox変更ハンドラ
                        dev_enable_query_rewrite.change(lambda v: gr.Accordion(visible=v), inputs=dev_enable_query_rewrite, outputs=dev_query_rewrite_section)

                        with gr.Row():
                            with gr.Column():
                                dev_chat_clear_btn = gr.Button("クリア", variant="secondary")
                            with gr.Column():
                                dev_chat_execute_btn = gr.Button("実行（時間がかかる場合があります）", variant="primary")

                        with gr.Row():
                            dev_chat_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. 生成SQL・分析", open=True):
                        dev_generated_sql_text = gr.Textbox(
                            label="生成されたSQL*",
                            lines=8,
                            max_lines=15,
                            interactive=True,
                            show_copy_button=True,
                            autoscroll=False,
                        )

                        with gr.Accordion(label="AI分析", open=True):
                            with gr.Row():
                                with gr.Column(scale=5):
                                    dev_prompt_text = gr.Textbox(label="開発者向けのプロンプト", lines=4, max_lines=10, visible=True, show_copy_button=True, value="以下のSQLを技術的観点で短く要約してください。目的、主要テーブル/結合、主なフィルタ、出力の概要を含めてください。")
                                with gr.Column(scale=5):
                                    user_prompt_text = gr.Textbox(label="ユーザー向けのプロンプト", lines=4, max_lines=10, visible=True, show_copy_button=True, value="以下のSQLが何をしているか、非技術的なユーザー向けに1〜3文で説明してください。専門用語はできるだけ避けてください。")
                            with gr.Row():
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            gr.Markdown("モデル*", elem_classes="input-label")
                                        with gr.Column(scale=5):
                                            dev_analysis_model_input = gr.Dropdown(
                                                show_label=False,
                                                choices=[
                                                    "xai.grok-code-fast-1",
                                                    "xai.grok-3",
                                                    "xai.grok-3-fast",
                                                    "xai.grok-4",
                                                    "xai.grok-4-fast-non-reasoning",
                                                    "meta.llama-4-scout-17b-16e-instruct",
                                                    "gpt-4o",
                                                    "gpt-5.1",
                                                ],
                                                value="xai.grok-code-fast-1",
                                                interactive=True,
                                                container=False,
                                            )
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            dev_ai_analyze_btn = gr.Button("AI分析（時間がかかる場合があります）", variant="primary")

                            with gr.Row():
                                dev_ai_analyze_status = gr.Markdown(visible=False)

                            with gr.Row():
                                with gr.Column(scale=5):
                                    dev_sql_summary_text = gr.Textbox(label="開発者向け SQLの概要説明", lines=6, max_lines=12, interactive=False, show_copy_button=True, autoscroll=False)
                                with gr.Column(scale=5):
                                    user_sql_summary_text = gr.Textbox(label="ユーザー向け SQLの概要説明", lines=6, max_lines=12, interactive=False, show_copy_button=True, autoscroll=False)

                            with gr.Row():
                                dev_sql_structure_text = gr.Textbox(
                                    label="SQL構造分析",
                                    lines=15,
                                    max_lines=30,
                                    interactive=False,
                                    show_copy_button=True,
                                    visible=True,
                                    autoscroll=False,
                                )

                    with gr.Accordion(label="3. 実行結果", open=True):
                        dev_chat_result_df = gr.Dataframe(
                            label="実行結果",
                            interactive=False,
                            wrap=True,
                            visible=False,
                            value=pd.DataFrame(),
                            elem_id="selectai_dev_chat_result_df",
                        )
                        dev_chat_result_style = gr.HTML(visible=False)

                    with gr.Accordion(label="4. クエリのフィードバック", open=False):
                        gr.Markdown("ℹ️ positiveで登録するとsql_id取得のために再実行が必要なため、同等の効果を効率的に得られるnegativeとして保存し、修正SQL(response)には「生成されたSQL*」を自動入力します。")
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("種類*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        dev_feedback_type_select = gr.Dropdown(
                                            show_label=False,
                                            choices=["positive", "negative"],
                                            value="positive",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("修正SQL(response)*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_feedback_response_text = gr.Textbox(
                                    show_label=False,
                                    placeholder="期待する正しいSQLを入力",
                                    lines=4,
                                    max_lines=12,
                                    show_copy_button=True,
                                    interactive=False,
                                    container=False,
                                    autoscroll=False,
                                )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("コメント(feedback_content)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_feedback_content_text = gr.Textbox(
                                    show_label=False,
                                    placeholder="自然言語で改善点や条件などを入力",
                                    lines=4,
                                    max_lines=12,
                                    show_copy_button=True,
                                    interactive=False,
                                    container=False,
                                    autoscroll=False,
                                )


                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("使用されたDBMS_CLOUD_AI.FEEDBACK", elem_classes="input-label")
                            with gr.Column(scale=5):
                                dev_feedback_used_sql_text = gr.Textbox(
                                    show_label=False,
                                    lines=8,
                                    max_lines=15,
                                    interactive=False,
                                    show_copy_button=True,
                                    container=False,
                                    autoscroll=False,
                                )

                        with gr.Row():
                            dev_feedback_send_btn = gr.Button("フィードバック送信", variant="primary")
                        with gr.Row():
                            dev_feedback_status = gr.Markdown(visible=False)
                        with gr.Row():
                            dev_feedback_result = gr.Markdown(visible=False)

                    def _build_showsql_stmt(prompt: str) -> str:
                        s = str(prompt or "")
                        # singles = ["!", "~", "^", "@", "#", "$", "%", "&", ";", ":"]
                        # for d in singles:
                        #     if d not in s:
                        #         return f"select ai showsql q'{d}{s}{d}'"
                        # pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
                        # for o, c in pairs:
                        #     if o not in s and c not in s:
                        #         return f"select ai showsql q'{o}{s}{c}'"
                        # esc = s.replace("'", "''")
                        # return f"select ai showsql '{esc}'"
                        return f"select ai showsql {s}"
                    
                    def _get_profile_schema_info(profile_name: str) -> str:
                        """指定されたProfileのテーブル/ビュー情報(コメント含む)を取得する."""
                        try:
                            s = _get_profile_schema_text_from_json(profile_name)
                            if str(s).strip():
                                return s
                            prof_name = _resolve_profile_name(pool, profile_name)
                            attrs = _get_profile_attributes(pool, prof_name) or {}
                            obj_list = attrs.get("object_list") or []
                            if not obj_list:
                                return ""
                            schema_parts = []
                            schema_parts.append("=== データベーススキーマ情報 ===")
                            for obj in obj_list:
                                obj_name = str((obj or {}).get("name") or "")
                                if not obj_name:
                                    continue
                                try:
                                    table_df = get_table_details(pool, obj_name)
                                    if table_df is not None and not table_df.empty:
                                        schema_parts.append(f"\n--- テーブル: {obj_name} ---")
                                        for _, row in table_df.iterrows():
                                            col_name = row.get("Column Name", "")
                                            col_type = row.get("Data Type", "")
                                            col_comment = row.get("Comments", "")
                                            schema_parts.append(f"  - {col_name} ({col_type}): {col_comment}")
                                        continue
                                except Exception as e:
                                    logger.error(f"Failed to get table details for {obj_name}: {e}")
                                try:
                                    view_df, _ = get_view_details(pool, obj_name)
                                    if view_df is not None and not view_df.empty:
                                        schema_parts.append(f"\n--- VIEW: {obj_name} ---")
                                        for _, row in view_df.iterrows():
                                            col_name = row.get("Column Name", "")
                                            col_type = row.get("Data Type", "")
                                            col_comment = row.get("Comments", "")
                                            schema_parts.append(f"  - {col_name} ({col_type}): {col_comment}")
                                except Exception as e:
                                    logger.error(f"Failed to get view details for {obj_name}: {e}")
                            return "\n".join(schema_parts)
                        except Exception as e:
                            logger.error(f"_get_profile_schema_info error: {e}")
                            return ""
                    
                    def _load_terminology() -> dict:
                        """用語集を読み込む.
                        
                        Returns:
                            dict: {TERM: DESCRIPTION}の辞書
                        """
                        try:
                            p = Path("uploads") / "terms.xlsx"
                            if not p.exists():
                                return {}
                            df = pd.read_excel(str(p))
                            cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                            t_col = cols_map.get("TERM")
                            d_col = cols_map.get("DEFINITION")
                            if not t_col or not d_col:
                                return {}
                            terms = {}
                            for _, row in df.iterrows():
                                term = str(row[t_col]).strip()
                                desc = str(row[d_col]).strip()
                                if term and desc:
                                    terms[term] = desc
                            return terms
                        except Exception as e:
                            logger.error(f"_load_terminology error: {e}")
                            return {}
                    
                    def _dev_rewrite_query(model_name, profile_name, original_query, use_glossary, use_schema):
                        """開発者向けクエリ書き換え処理.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            profile_name: Profile名
                            original_query: 元の自然言語の質問
                            use_glossary: 第1ステップ（用語集）を実行するか
                            use_schema: 第2ステップを実行するか
                        
                        Yields:
                            tuple: (status_md, rewritten_text)
                        """
                        from utils.chat_util import get_oci_region, get_compartment_id
                        from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                        
                        try:
                            # 入力チェック
                            if not model_name or not str(model_name).strip():
                                yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="", autoscroll=False)
                                return
                            if not original_query or not str(original_query).strip():
                                yield gr.Markdown(visible=True, value="⚠️ 元の質問を入力してください"), gr.Textbox(value="", autoscroll=False)
                                return
                            
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                yield gr.Markdown(visible=True, value="❌ OCI設定が不足しています"), gr.Textbox(value="", autoscroll=False)
                                return
                            
                            # ステップ1/2が両方OFFの場合は警告して終了
                            if (not use_glossary) and (not use_schema):
                                yield gr.Markdown(visible=True, value="⚠️ ステップ1（用語集）とステップ2（スキーマ）がOFFです。少なくとも1つをONにしてください"), gr.Textbox(value="", autoscroll=False)
                                return
                            
                            step1_result = str(original_query).strip()
                            
                            # 第1ステップ: 用語集で分析・置換（ONの場合のみ）
                            if use_glossary:
                                yield gr.Markdown(visible=True, value="⏳ 第1ステップ: 用語集で分析・置換中..."), gr.Textbox(value="", autoscroll=False)
                                
                                terms = _load_terminology()
                                if terms:
                                    # 用語集を使ってLLMで分析
                                    terms_text = "\n".join([f"- {k}: {v}" for k, v in terms.items()])
                                    step1_prompt = f"""あなたはデータベースクエリの専門家です。以下の用語集は「A（TERM）→B（定義・推奨表現）」の最適化指針です。本ステップでは正方向の最適化を行い、元の質問に含まれるA側の用語をB側の推奨表現へ明確化・正規化してください。

用語集:
{terms_text}

元の質問:
{original_query}

指示:
1. TERM（A側）が含まれる場合は、その定義・推奨表現（B側）に置換し、意味を明確化してください。
2. 曖昧な表現は、対象・条件・期間などを可能な限り具体的な言い回しに整えてください。
3. 質問の意図・条件・対象は維持し、不要な追加・削除は行わないでください。
4. 数値・日付・範囲などの具体値は変更しないでください。
5. 出力は修正後の質問文のみ。説明や前置きは不要です。

修正後の質問:"""
                                    
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        messages = [{"role": "user", "content": step1_prompt}]
                                        if str(model_name).startswith("gpt-"):
                                            from openai import AsyncOpenAI
                                            client = AsyncOpenAI()
                                            resp = loop.run_until_complete(
                                                client.chat.completions.create(model=model_name, messages=messages)
                                            )
                                        else:
                                            client = AsyncOciOpenAI(
                                                service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                                auth=OciUserPrincipalAuth(),
                                                compartment_id=compartment_id,
                                            )
                                            resp = loop.run_until_complete(
                                                client.chat.completions.create(model=model_name, messages=messages)
                                            )
                                        if resp.choices and len(resp.choices) > 0:
                                            step1_result = resp.choices[0].message.content.strip()
                                    finally:
                                        loop.close()
                            
                            # 第2ステップが無効ならここで終了
                            if not use_schema:
                                yield gr.Markdown(visible=True, value="✅ 完了（第1ステップのみ）"), gr.Textbox(value=step1_result)
                                return
                            
                            yield gr.Markdown(visible=True, value="⏳ ステップ2: スキーマ情報を取り込み、自然言語へ書き換え中..."), gr.Textbox(value=step1_result)
                            
                            # ステップ2: スキーマ情報を取り込み、自然言語へ書き換え
                            if not profile_name or not str(profile_name).strip():
                                yield gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Textbox(value=step1_result)
                                return
                            
                            schema_info = _get_profile_schema_info(profile_name)
                            if not schema_info:
                                yield gr.Markdown(visible=True, value="⚠️ スキーマ情報が取得できませんでした"), gr.Textbox(value=step1_result)
                                return
                            
                            step2_prompt = f"""あなたはデータベースクエリの専門家です。以下のデータベーススキーマ情報を参照し、元の質問をデータベースがより正確に解釈できる自然言語へ変換してください。

=== 参考スキーマ情報 ===
{schema_info}

=== 元の質問 ===
{step1_result}

指示:
1. 利用可能なテーブル名・カラム名・VIEW名を自然言語の中で明確にし、曖昧な用語はスキーマに合わせて具体化してください。
2. 条件・期間・集計などが含まれる場合は、自然言語で明確に記述してください。
3. 質問の元の意図を保ちつつ、データベースにとって解釈しやすい表現にしてください。
4. SQLやコードは絶対に出力せず、自然言語のみを使用してください。
5. 出力は変換後の自然言語の質問文のみとし、説明や前置きは不要です。

変換後の自然言語:"""
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                messages = [{"role": "user", "content": step2_prompt}]
                                if str(model_name).startswith("gpt-"):
                                    from openai import AsyncOpenAI
                                    client = AsyncOpenAI()
                                    resp = loop.run_until_complete(
                                        client.chat.completions.create(model=model_name, messages=messages)
                                    )
                                else:
                                    client = AsyncOciOpenAI(
                                        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                        auth=OciUserPrincipalAuth(),
                                        compartment_id=compartment_id,
                                    )
                                    resp = loop.run_until_complete(
                                        client.chat.completions.create(model=model_name, messages=messages)
                                    )
                                if resp.choices and len(resp.choices) > 0:
                                    final_result = str(step1_result) + "\n\n" + resp.choices[0].message.content.strip()
                                else:
                                    final_result = step1_result
                            finally:
                                loop.close()

                            
                            yield gr.Markdown(visible=True, value="✅ 書き換え完了"), gr.Textbox(value=final_result)
                            
                        except Exception as e:
                            logger.error(f"_dev_rewrite_query error: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _common_step_generate(profile, prompt, extra_prompt, include_extra, enable_rewrite, rewritten_query):
                        if enable_rewrite and rewritten_query and str(rewritten_query).strip():
                            s = str(rewritten_query).strip()
                        else:
                            s = str(prompt or "").strip()
                        ep = str(extra_prompt or "").strip()
                        inc = bool(include_extra)
                        final = s if not inc or not ep else (ep + "\n\n" + s)
                        if not profile or not str(profile).strip():
                            yield gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Textbox(value="", autoscroll=False)
                            return
                        if not final:
                            yield gr.Markdown(visible=True, value="⚠️ 質問を入力してください"), gr.Textbox(value="", autoscroll=False)
                            return
                        q = final
                        if q.endswith(";"):
                            q = q[:-1]
                        try:
                            yield gr.Markdown(visible=True, value="⏳ SQL生成中..."), gr.Textbox(value="", autoscroll=False)
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    try:
                                        prof, _tables, _views, _bd = _resolve_profile_name_from_json(pool, str(profile or ""))
                                        cursor.execute("BEGIN DBMS_CLOUD_AI.SET_PROFILE(profile_name => :name); END;", name=prof)
                                    except Exception as e:
                                        logger.error(f"set profile error: {e}")
                                    gen_stmt = "select dbms_cloud_ai.generate(prompt=> :q, profile_name=> :name, action=> :a)"
                                    showsql_stmt = _build_showsql_stmt(q)
                                    show_text = ""
                                    show_cells = []
                                    try:
                                        cursor.execute(gen_stmt, q=showsql_stmt, name=prof, a="showsql")
                                        rows = cursor.fetchmany(size=200)
                                        if rows:
                                            for r in rows:
                                                for v in r:
                                                    try:
                                                        s2 = v.read() if hasattr(v, "read") else str(v)
                                                    except Exception:
                                                        s2 = str(v)
                                                    if s2:
                                                        show_cells.append(s2)
                                            show_text = "\n".join(show_cells)
                                    except Exception as e:
                                        err_msg = str(e)
                                        try:
                                            import re as _re
                                            m = _re.search(r'Error response - ({.*})', err_msg)
                                            if m:
                                                err_json = json.loads(m.group(1))
                                                if "message" in err_json:
                                                    inner_msg = err_json["message"]
                                                    try:
                                                        inner_json = json.loads(inner_msg)
                                                        if "error" in inner_json:
                                                            err_msg = inner_json["error"]
                                                        elif "code" in inner_json and "message" in inner_json:
                                                            err_msg = f"{inner_json['code']}: {inner_json['message']}"
                                                    except Exception:
                                                        err_msg = inner_msg
                                        except Exception as _inner_err:
                                            logger.error(f"inner error parse failed: {_inner_err}")
                                        yield gr.Markdown(visible=True, value=f"❌ エラー: {err_msg}"), gr.Textbox(value="", autoscroll=False)
                                        show_text = ""
                                        return
                                    # try:
                                    #     cursor.execute(showsql_stmt)
                                    # except Exception as e:
                                    #     yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="", autoscroll=False)
                                    #     return
                                    # _ = _get_sql_id_for_text(showsql_stmt)
                                    def _extract_sql(text: str) -> str:
                                        if not text:
                                            return ""
                                        m = re.search(r"```sql\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
                                        if m:
                                            s3 = m.group(1).strip()
                                            return s3
                                        m2 = re.search(r"SQL\s*:([\s\S]+)$", text, flags=re.IGNORECASE)
                                        if m2:
                                            s3 = m2.group(1).strip()
                                            return s3
                                        m3 = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
                                        if m3:
                                            s3 = m3.group(0).strip()
                                            return s3
                                        return ""
                                    generated_sql = _extract_sql(show_text)
                                    if not generated_sql and show_cells:
                                        for cell in show_cells:
                                            c = str(cell)
                                            try:
                                                obj = json.loads(c)
                                                if isinstance(obj, dict):
                                                    for k in ["sql", "SQL", "generated_sql", "query", "Query"]:
                                                        if k in obj and obj[k]:
                                                            generated_sql = str(obj[k]).strip()
                                                            break
                                                if generated_sql:
                                                    break
                                            except Exception:
                                                pass
                                            m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", c, flags=re.IGNORECASE)
                                            if m:
                                                generated_sql = m.group(0).strip()
                                                break
                                    gen_sql_display = generated_sql
                                    if gen_sql_display and not gen_sql_display.endswith(";"):
                                        gen_sql_display = gen_sql_display
                                    yield gr.Markdown(visible=True, value="✅ SQL生成完了"), gr.Textbox(value=gen_sql_display)
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _dev_step_generate(profile, prompt, extra_prompt, include_extra, enable_rewrite, rewritten_query):
                        yield from _common_step_generate(profile, prompt, extra_prompt, include_extra, enable_rewrite, rewritten_query)

                    def _run_sql_common(sql_text, elem_id):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", interactive=False, wrap=True, elem_id=elem_id), gr.HTML(visible=False, value="")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    s = str(sql_text or "").strip()
                                    if not s or not re.match(r"^\s*(select|with)\b", s, flags=re.IGNORECASE):
                                        yield gr.Markdown(visible=True, value="✅ 表示完了（データなし）"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", interactive=False, wrap=True, elem_id=elem_id), gr.HTML(visible=False, value="")
                                        return
                                    run_sql = s
                                    if run_sql.endswith(";"):
                                        run_sql = run_sql[:-1]
                                    cursor.execute(run_sql)
                                    exec_rows = cursor.fetchmany(size=10000)
                                    exec_cols = [d[0] for d in cursor.description] if cursor.description else []
                                    if exec_rows:
                                        cleaned_rows = []
                                        for r in exec_rows:
                                            cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                                        df = pd.DataFrame(cleaned_rows, columns=exec_cols)
                                        widths = []
                                        if len(df) > 0:
                                            sample = df.head(5)
                                            for col in df.columns:
                                                series = sample[col].astype(str)
                                                row_max = series.map(len).max() if len(series) > 0 else 0
                                                length = max(len(str(col)), row_max)
                                                widths.append(length)
                                        else:
                                            widths = [len(str(c)) for c in df.columns]
                                        total = sum(widths) if widths else 0
                                        if total <= 0:
                                            col_widths = None
                                        else:
                                            col_widths = [max(5, int(100 * w / total)) for w in widths]
                                            diff = 100 - sum(col_widths)
                                            if diff != 0 and len(col_widths) > 0:
                                                col_widths[0] = max(5, col_widths[0] + diff)
                                        df_component = gr.Dataframe(
                                            visible=True,
                                            value=df,
                                            label=f"実行結果（件数: {len(df)}）",
                                            interactive=False,
                                            wrap=True,
                                            elem_id=elem_id,
                                        )
                                        style_value = ""
                                        if col_widths:
                                            rules = []
                                            rules.append(f"#{elem_id} { '{' } width: 100% !important; { '}' }")
                                            rules.append(f"#{elem_id} .wrap { '{' } overflow-x: auto !important; { '}' }")
                                            rules.append(f"#{elem_id} table { '{' } table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; { '}' }")
                                            for idx, pct in enumerate(col_widths, start=1):
                                                rules.append(
                                                    f"#{elem_id} table th:nth-child({idx}), #{elem_id} table td:nth-child({idx}) { '{' } width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; { '}' }"
                                                )
                                            style_value = "<style>" + "\n".join(rules) + "</style>"
                                        style_component = gr.HTML(visible=bool(style_value), value=style_value)
                                        yield gr.Markdown(visible=True, value="✅ 取得完了"), df_component, style_component
                                        return
                                    yield gr.Markdown(visible=True, value="✅ 表示完了（データなし）"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", interactive=False, wrap=True, elem_id=elem_id), gr.HTML(visible=False, value="")
                        except Exception as e:
                            logger.error(f"_run_sql_common error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（エラー）", interactive=False, wrap=True, elem_id=elem_id), gr.HTML(visible=False, value="")

                    def _dev_step_run_sql(generated_sql, status_text=None):
                        if status_text and "❌" in str(status_text):
                            return
                        yield from _run_sql_common(generated_sql, "selectai_dev_chat_result_df")

                    async def _dev_ai_analyze_async(model_name, sql_text, dev_prompt, user_prompt):
                        try:
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Markdown(visible=True, value="⚠️ OCI設定が不足しています"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            s = str(sql_text or "").strip()
                            if not s:
                                return gr.Markdown(visible=True, value="⚠️ SQLが空です"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            
                            if str(model_name).startswith("gpt-"):
                                from openai import AsyncOpenAI
                                client = AsyncOpenAI()
                            else:
                                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                                client = AsyncOciOpenAI(
                                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                    auth=OciUserPrincipalAuth(),
                                    compartment_id=compartment_id,
                                )

                            # グローバルプロンプトを使用
                            prompt = _SQL_STRUCTURE_ANALYSIS_PROMPT + "SQL:\n```sql\n" + s + "\n```"

                            messages = [
                                {
                                    "role": "system", 
                                    "content": "You are a SQL parser. Output ONLY the requested format. No explanations."
                                },
                                {
                                    "role": "user", 
                                    "content": prompt
                                },
                            ]

                            resp = await client.chat.completions.create(model=model_name, messages=messages)
                            sql_structure_md = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                out = msg.content if hasattr(msg, "content") else ""
                                sql_structure_md = str(out or "").strip()
                                # マークダウンコードブロックを削除
                                sql_structure_md = re.sub(r"```+markdown\s*", "", sql_structure_md)
                                sql_structure_md = re.sub(r"```+\s*$", "", sql_structure_md)
                                sql_structure_md = sql_structure_md.strip()
                            if not sql_structure_md:
                                sql_structure_md = "## 📊 SQL構造分析\n\n情報を抽出できませんでした。"
                            dev_summary = ""
                            user_summary = ""
                            dev_enable=True
                            user_enable=True
                            if bool(dev_enable):
                                dp = str(dev_prompt or "").strip()
                                dmsg = [
                                    {"role": "system", "content": "あなたはSQLの要約に特化したアシスタントです。回答は簡潔な日本語テキストのみ。"},
                                    {"role": "user", "content": ((dp + "\n") if dp else "") + "SQL:\n```sql\n" + str(sql_text) + "\n```"}
                                ]
                                dresp = await client.chat.completions.create(model=model_name, messages=dmsg)
                                if getattr(dresp, "choices", None):
                                    dmsg0 = dresp.choices[0].message
                                    dout = dmsg0.content if hasattr(dmsg0, "content") else ""
                                    dev_summary = re.sub(r"```+\w*", "", str(dout or "")).strip()
                            if bool(user_enable):
                                up = str(user_prompt or "").strip()
                                umsg = [
                                    {"role": "system", "content": "あなたは非技術ユーザー向けに分かりやすく説明するアシスタントです。回答は日本語の平易な文章のみ。"},
                                    {"role": "user", "content": ((up + "\n") if up else "") + "SQL:\n```sql\n" + str(sql_text) + "\n```"}
                                ]
                                uresp = await client.chat.completions.create(model=model_name, messages=umsg)
                                if getattr(uresp, "choices", None):
                                    umsg0 = uresp.choices[0].message
                                    uout = umsg0.content if hasattr(umsg0, "content") else ""
                                    user_summary = re.sub(r"```+\w*", "", str(uout or "")).strip()
                            return gr.Markdown(visible=True, value="✅ AI分析完了"), gr.Textbox(value=sql_structure_md), gr.Textbox(value=dev_summary), gr.Textbox(value=user_summary)
                        except Exception as e:
                            logger.error(f"_dev_ai_analyze_async error: {e}")
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)

                    def _dev_ai_analyze(model_name, sql_text, dev_prompt, user_prompt):
                        import asyncio
                        # 必須入力項目のチェック
                        if not model_name or not str(model_name).strip():
                            yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            return
                        if not sql_text or not str(sql_text).strip():
                            yield gr.Markdown(visible=True, value="⚠️ SQLが空です。先にSQLを生成してください"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            return
                        yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Textbox(value="## 📊 SQL構造分析\n\n解析中..."), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(_dev_ai_analyze_async(model_name, sql_text, dev_prompt, user_prompt))
                            yield result
                        finally:
                            loop.close()

                    def _on_dev_chat_clear():
                        ch = _dev_profile_names() or [("", "")]
                        return "", gr.Dropdown(choices=ch, value=ch[0][1])

                    def _predict_domain_and_set_profile(text):
                        try:
                            ch = _dev_profile_names() or [("", "")]
                            pdomain = _predict_category_label(text)
                            return _map_domain_to_profile(pdomain, ch)
                        except Exception:
                            ch = _dev_profile_names() or [("", "")]
                            return gr.Dropdown(choices=ch, value=ch[0][1])

                    def _append_comment(current_text: str, template: str):
                        s = str(current_text or "").strip()
                        t = str(template or "").strip()
                        if not s:
                            return t
                        if s.endswith("\n"):
                            return s + t
                        return s + "\n" + t

                    def _get_sql_id_for_text(sql_text: str) -> str:
                            try:
                                s = str(sql_text or "").strip()
                                if not s:
                                    return ""
                                with pool.acquire() as conn:
                                    with conn.cursor() as cursor:
                                        # Try match against SQL_TEXT in v$mapped_sql
                                        try:
                                            cursor.execute(
                                                """
                                                select sql_id
                                                from v$mapped_sql
                                                where regexp_replace(sql_text,'\\s+',' ') = regexp_replace(:t,'\\s+',' ')
                                                order by translation_timestamp desc nulls last, use_count desc
                                                fetch first 1 rows only
                                                """,
                                                t=s,
                                            )
                                            row = cursor.fetchone()
                                            if row and row[0]:
                                                return str(row[0])
                                        except Exception as e:
                                            logger.error(f"_get_sql_id_for_text primary error: {e}")

                                        # Fallback: match against SQL_FULLTEXT (CLOB)
                                        try:
                                            cursor.execute(
                                                """
                                                select sql_id
                                                from v$mapped_sql
                                                where regexp_replace(dbms_lob.substr(sql_fulltext, 4000),'\\s+',' ') = regexp_replace(:t,'\\s+',' ')
                                                order by translation_timestamp desc nulls last, use_count desc
                                                fetch first 1 rows only
                                                """,
                                                t=s,
                                            )
                                            row = cursor.fetchone()
                                            if row and row[0]:
                                                return str(row[0])
                                        except Exception as e:
                                            logger.error(f"_get_sql_id_for_text fallback error: {e}")
                            except Exception as e:
                                logger.error(f"_get_sql_id_for_text outer error: {e}")
                                return ""
                            return ""

                    def _send_feedback(fb_type, response_text, content_text, prompt_text, profile_name):
                        plsql = ""
                        try:
                            yield gr.Markdown(visible=True, value="⏳ フィードバック送信中..."), gr.Markdown(visible=False), gr.Textbox(value="", autoscroll=False)
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof, _t, _v, _bd = _resolve_profile_name_from_json(pool, str(profile_name or ""))
                                    q = str(prompt_text or "").strip()
                                    if q.endswith(";"):
                                        q = q[:-1]
                                    if not q:
                                        yield gr.Markdown(visible=False), gr.Markdown(visible=True, value="⚠️ 質問が未入力のため、フィードバックを送信できませんでした"), gr.Textbox(value="", autoscroll=False)
                                        return
                                    prompt_text = f"select ai showsql {q}"
                                    # gen_stmt = "select dbms_cloud_ai.generate(prompt=> :q, profile_name => :name, action=> :a)"
                                    showsql_stmt = _build_showsql_stmt(q)
                                    logger.info(f"_send_feedback: q={q}, showsql_stmt={showsql_stmt}")
                                    # try:
                                    #     cursor.execute(gen_stmt, q=showsql_stmt, name=prof, a="showsql")
                                    # except Exception as e:
                                    #     logger.error(f"_send_feedback generate showsql error: {e}")
                                    # try:
                                    #     cursor.execute(showsql_stmt)
                                    # except Exception as e:
                                    #     logger.error(f"_send_feedback execute showsql error: {e}")
                                    t = str(fb_type or "").lower()
                                    resp = ""
                                    fc = ""
                                    ft_val = str(fb_type or "").upper()
                                    if t == "negative":
                                        resp = str(response_text or "").strip()
                                        fc = str(content_text or "")
                                        if not resp:
                                            yield gr.Markdown(visible=False), gr.Markdown(visible=True, value="⚠️ 修正SQLが未入力のため、ネガティブ・フィードバックを送信できませんでした"), gr.Textbox(value="", autoscroll=False)
                                            return
                                    elif t == "positive":
                                        resp = str(response_text or "").strip()
                                        fc = str(content_text or "")
                                        ft_val = "NEGATIVE"
                                        if not resp:
                                            yield gr.Markdown(visible=False), gr.Markdown(visible=True, value="⚠️ 生成されたSQLが未生成のため、ポジティブ・フィードバックを送信できませんでした"), gr.Textbox(value="", autoscroll=False)
                                            return
                                    # Build PL/SQL text regardless of execution result
                                    def _lit(x):
                                        s = str(x or "")
                                        return "'" + s.replace("'", "''") + "'"
                                    plsql = (
                                        "BEGIN\n"
                                        "  DBMS_CLOUD_AI.FEEDBACK(\n"
                                        f"    profile_name => {_lit(prof)},\n"
                                        f"    sql_text => {_lit(showsql_stmt)},\n"
                                        f"    feedback_type => {_lit(ft_val)},\n"
                                        "    response => " + ("NULL" if not resp else _lit(resp)) + ",\n"
                                        "    feedback_content => " + ("NULL" if not fc else _lit(fc)) + ",\n"
                                        "    operation => 'ADD'\n"
                                        "  );\n"
                                        "END;"
                                    )
                                    cursor.execute(
                                        """
                                        BEGIN
                                        DBMS_CLOUD_AI.FEEDBACK(
                                            profile_name => :p,
                                            sql_text => :st,
                                            feedback_type => :ft,
                                            response => :resp,
                                            feedback_content => :fc,
                                            operation => 'ADD'
                                        );
                                        END;
                                        """,
                                        p=prof,
                                        st=showsql_stmt,
                                        ft=ft_val,
                                        resp=resp,
                                        fc=fc,
                                    )
                                    yield gr.Markdown(visible=False), gr.Markdown(visible=True, value="✅ クエリに対するフィードバックを送信しました"), gr.Textbox(value=plsql)
                        except Exception as e:
                            yield gr.Markdown(visible=False), gr.Markdown(visible=True, value=f"❌ フィードバック送信に失敗しました: {str(e)}"), gr.Textbox(value=plsql)

                    dev_chat_execute_btn.click(
                        fn=_dev_step_generate,
                        inputs=[dev_profile_select, dev_prompt_input, dev_extra_prompt, dev_include_extra_prompt, dev_enable_query_rewrite, dev_rewritten_query],
                        outputs=[dev_chat_status_md, dev_generated_sql_text],
                    ).then(
                        fn=_dev_step_run_sql,
                        inputs=[dev_generated_sql_text, dev_chat_status_md],
                        outputs=[dev_chat_status_md, dev_chat_result_df, dev_chat_result_style],
                    ).then(
                        fn=lambda x: gr.Textbox(value=str(x or "")),
                        inputs=[dev_generated_sql_text],
                        outputs=[dev_feedback_response_text]
                    )

                    dev_ai_analyze_btn.click(
                        fn=_dev_ai_analyze,
                        inputs=[dev_analysis_model_input, dev_generated_sql_text, dev_prompt_text, user_prompt_text],
                        outputs=[dev_ai_analyze_status, dev_sql_structure_text, dev_sql_summary_text, user_sql_summary_text],
                    )

                    dev_chat_clear_btn.click(
                        fn=_on_dev_chat_clear,
                        outputs=[dev_prompt_input, dev_profile_select],
                    )

                    dev_predict_domain_btn.click(
                        fn=_predict_domain_and_set_profile,
                        inputs=[dev_prompt_input],
                        outputs=[dev_profile_select],
                    )
                    
                    # Query転写ボタンのイベントハンドラ
                    dev_rewrite_btn.click(
                        fn=_dev_rewrite_query,
                        inputs=[dev_rewrite_model_select, dev_profile_select, dev_prompt_input, dev_rewrite_use_glossary, dev_rewrite_use_schema],
                        outputs=[dev_rewrite_status, dev_rewritten_query],
                    )

                with gr.TabItem(label="フィードバック管理") as feedback_tab:
                    def _global_profile_names():
                        try:
                            # JSONファイルから読み込む
                            return _load_profiles_from_json()
                        except Exception as e:
                            logger.error(f"_global_profile_names error: {e}")
                        return []

                    with gr.Accordion(label="1. フィードバック一覧", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Profile", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        # プロフィール選択肢を取得し、空の場合は空文字列を含むリストを設定
                                        _global_initial_choices = _global_profile_names()
                                        if not _global_initial_choices:
                                            _global_initial_choices = [("", "")]
                                        global_profile_select = gr.Dropdown(
                                            show_label=False,
                                            choices=_global_initial_choices,
                                            value=(
                                                _global_initial_choices[0][1]
                                                if (_global_initial_choices and isinstance(_global_initial_choices[0], tuple))
                                                else (_global_initial_choices[0] if _global_initial_choices else "")
                                            ),
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            gr.Markdown(visible=True, value="ℹ️ Profile選択後は『最新エントリを取得』をクリックしてください")
                        with gr.Row():
                            global_feedback_index_refresh_btn = gr.Button("最新エントリを取得", variant="primary")
                        with gr.Row():
                            global_feedback_index_refresh_status = gr.Markdown(visible=False)
                        # Removed: global_feedback_index_info, use global_feedback_index_refresh_status for status
                        with gr.Row():
                            global_feedback_index_df = gr.Dataframe(
                                label="フィードバック索引の最新エントリ",
                                interactive=False,
                                wrap=True,
                                visible=False,
                                value=pd.DataFrame(),
                            )

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("選択されたSQL_TEXT", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        selected_sql_text = gr.Textbox(show_label=False, interactive=False, container=False, autoscroll=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        selected_feedback_delete_btn = gr.Button("選択したフィードバックを削除", variant="stop")
                            
                        with gr.Row():
                            selected_feedback_delete_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. ベクトルインデックス", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Similarity_Threshold", elem_classes="input-label")
                                    with gr.Column(scale=4):
                                        vec_similarity_threshold_input = gr.Slider(
                                            show_label=False,
                                            minimum=0.10,
                                            maximum=0.95,
                                            step=0.05,
                                            value=0.90,
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("Match_Limit", elem_classes="input-label")
                                    with gr.Column(scale=4):
                                        vec_match_limit_input = gr.Slider(
                                            show_label=False,
                                            minimum=1,
                                            maximum=5,
                                            step=1,
                                            value=3,
                                            interactive=True,
                                            container=False,
                                        )

                        with gr.Row():
                            vec_update_btn = gr.Button("ベクトルインデックスを更新", variant="primary")

                    def _view_feedback_index_global(profile_name: str):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ フィードバック索引を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame())
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof, _t, _v, _bd = _resolve_profile_name_from_json(pool, str(profile_name or ""))
                                    tab = f"{str(prof).upper()}_FEEDBACK_VECINDEX$VECTAB"
                                    q_no_ctx = (
                                        f'SELECT CONTENT, '
                                        f"JSON_VALUE(ATTRIBUTES, '$.sql_id' RETURNING VARCHAR2(128)) AS SQL_ID, "
                                        f"JSON_VALUE(ATTRIBUTES, '$.sql_text' RETURNING CLOB) AS SQL_TEXT, "
                                        f'ATTRIBUTES FROM "{tab}" FETCH FIRST 50 ROWS ONLY'
                                    )
                                    rows = []
                                    cols = []
                                    cursor.execute(q_no_ctx)
                                    rows = cursor.fetchall() or []
                                    cols = [d[0] for d in cursor.description] if cursor.description else []
                                    def _to_plain(val):
                                        v = val.read() if hasattr(val, "read") else val
                                        if isinstance(v, bytes):
                                            try:
                                                v = v.decode("utf-8")
                                            except Exception:
                                                v = v.decode("latin1", errors="ignore")
                                        s = v
                                        if not isinstance(s, str):
                                            s = str(s)
                                        t = s.strip()
                                        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                                            try:
                                                obj = json.loads(t)
                                                s = json.dumps(obj, ensure_ascii=False)
                                            except Exception:
                                                pass
                                        return s

                                    cleaned_rows = []
                                    for r in rows:
                                        cleaned_rows.append([_to_plain(v) for v in r])
                                    df = pd.DataFrame(cleaned_rows, columns=cols)
                                    if df.empty:
                                        yield gr.Markdown(visible=True, value="ℹ️ まだフィードバック索引がありません"), gr.Dataframe(visible=False, value=pd.DataFrame())
                                        return
                                    yield gr.Markdown(visible=False), gr.Dataframe(visible=True, value=df)
                        except Exception as e:
                            logger.error(f"_view_feedback_index_global error: {e}")
                            yield gr.Markdown(visible=True, value="ℹ️ まだフィードバック索引がありません"), gr.Dataframe(visible=False, value=pd.DataFrame())

                    global_feedback_index_refresh_btn.click(
                        fn=_view_feedback_index_global,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_refresh_status, global_feedback_index_df],
                    )

                    def on_index_row_select(evt: gr.SelectData, current_df):
                        try:
                            row_index = evt.index[0]
                            df = current_df
                            if isinstance(df, dict) and "data" in df:
                                df = pd.DataFrame(df["data"], columns=df.get("headers", []))
                            if isinstance(df, pd.DataFrame) and not df.empty and row_index >= 0:
                                row = df.iloc[row_index]
                                sql_text = str(row.get("SQL_TEXT", ""))
                                return sql_text
                        except Exception as e:
                            logger.error(f"on_index_row_select error: {e}")
                        return ""

                    global_feedback_index_df.select(
                        fn=on_index_row_select,
                        inputs=[global_feedback_index_df],
                        outputs=[selected_sql_text],
                    )

                    def _delete_by_sql_text(profile_name: str, sql_text: str):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 削除中..."), gr.Textbox(value=str(sql_text or ""))
                            if not sql_text:
                                yield gr.Markdown(visible=True, value="❌ 失敗: SQL_TEXTが選択されていません"), gr.Textbox(value="", autoscroll=False)
                                return
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof, _t, _v, _bd = _resolve_profile_name_from_json(pool, str(profile_name or ""))
                                    cursor.execute(
                                        """
                                        BEGIN
                                        DBMS_CLOUD_AI.FEEDBACK(
                                            profile_name => :p,
                                            sql_text => :st,
                                            operation => 'DELETE'
                                        );
                                        END;
                                        """,
                                        p=str(prof),
                                        st=str(sql_text),
                                    )
                            yield gr.Markdown(visible=True, value="✅ 成功"), gr.Textbox(value="", autoscroll=False)
                        except Exception as e:
                            logger.error(f"_delete_by_sql_text error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 失敗: {str(e)}"), gr.Textbox(value="", autoscroll=False)

                    selected_feedback_delete_btn.click(
                        fn=_delete_by_sql_text,
                        inputs=[global_profile_select, selected_sql_text],
                        outputs=[selected_feedback_delete_status_md, selected_sql_text],
                    ).then(
                        fn=_view_feedback_index_global,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_refresh_status, global_feedback_index_df],
                    )

                    def _update_vector_index(profile_name: str, similarity_threshold: float, match_limit: int):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 更新中...")
                            prof, _t, _v, _bd = _resolve_profile_name_from_json(pool, str(profile_name or ""))
                            idx_name = f"{str(prof).upper()}_FEEDBACK_VECINDEX"
                            tab_name = f"{str(prof).upper()}_FEEDBACK_VECINDEX$VECTAB"
                            logger.info(f"Update vector index: profile={profile_name}, index={idx_name}, table={tab_name}, threshold={similarity_threshold}, limit={match_limit}")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    try:
                                        cursor.execute(f'SELECT 1 FROM "{tab_name}" FETCH FIRST 1 ROWS ONLY')
                                        _ = cursor.fetchall()
                                    except Exception as e:
                                        logger.error(f"Index table not found: {tab_name}: {e}")
                                        yield gr.Markdown(visible=True, value=f"❌ 索引が存在しません: {tab_name}")
                                        return

                                    vec_attrs = json.dumps({
                                        "similarity_threshold": float(similarity_threshold),
                                        "match_limit": int(match_limit),
                                    }, ensure_ascii=False)
                                    logger.info(f"Calling UPDATE_VECTOR_INDEX with attrs={vec_attrs}")
                                    try:
                                        cursor.execute(
                                            """
                                            BEGIN
                                            DBMS_CLOUD_AI.UPDATE_VECTOR_INDEX(
                                                index_name => :idx,
                                                attributes => :vattrs
                                            );
                                            END;
                                            """,
                                            idx=idx_name,
                                            vattrs=vec_attrs,
                                        )
                                    except Exception as e:
                                        logger.error(f"UPDATE_VECTOR_INDEX failed: index={idx_name}, table={tab_name}, error={e}")
                                        yield gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {str(e)}")
                                        return
                                    logger.info("UPDATE_VECTOR_INDEX succeeded")
                                    yield gr.Markdown(visible=True, value="✅ 更新完了")
                        except Exception as e:
                            logger.error(f"Unexpected error in _update_vector_index: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {str(e)}")

                    vec_update_btn.click(
                        fn=_update_vector_index,
                        inputs=[global_profile_select, vec_similarity_threshold_input, vec_match_limit_input],
                        outputs=[global_feedback_index_refresh_status],
                    ).then(
                        fn=_view_feedback_index_global,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_refresh_status, global_feedback_index_df],
                    )

                with gr.TabItem(label="コメント管理"):
                    with gr.Accordion(label="1. オブジェクト選択", open=True):
                        with gr.Row():
                            with gr.Column():                        
                                cm_refresh_btn = gr.Button("テーブル・ビュー一覧を取得（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                cm_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択*")
                                cm_tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択*")
                                cm_views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("サンプル件数*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        cm_sample_limit = gr.Number(show_label=False, minimum=0, maximum=100, value=10, interactive=True, container=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                cm_fetch_btn = gr.Button("情報を取得", variant="primary")
                        with gr.Row():
                            cm_fetch_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. 入力確認", open=True) as cm_input_confirm_acc:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("構造情報*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_structure_text = gr.Textbox(show_label=False, lines=8, max_lines=16, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("主キー情報(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_pk_text = gr.Textbox(show_label=False, lines=4, max_lines=10, interactive=True, show_copy_button=True, container=False, autoscroll=False)    
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("外部キー情報(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_fk_text = gr.Textbox(show_label=False, lines=6, max_lines=14, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("サンプルデータ(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_samples_text = gr.Textbox(show_label=False, lines=8, max_lines=16, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("追加入力(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_extra_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="追加で考慮してほしい説明や条件を記入",
                                    value=(""),
                                    lines=8,
                                    max_lines=16,
                                    container=False,
                                    autoscroll=False,
                                )

                    with gr.Accordion(label="3. コメント自動生成", open=False):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("モデル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        cm_model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "xai.grok-code-fast-1",
                                                "xai.grok-3",
                                                "xai.grok-3-fast",
                                                "xai.grok-4",
                                                "xai.grok-4-fast-non-reasoning",
                                                "meta.llama-4-scout-17b-16e-instruct",
                                                "gpt-4o",
                                                "gpt-5.1",
                                            ],
                                            value="xai.grok-code-fast-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        cm_generate_btn = gr.Button("生成（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            cm_generate_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("生成されたSQL*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                cm_generated_sql = gr.Textbox(show_label=False, lines=15, max_lines=15, interactive=True, show_copy_button=True, container=False, autoscroll=False)

                    with gr.Accordion(label="4. 実行", open=False):
                        with gr.Row():
                            cm_execute_btn = gr.Button("一括実行", variant="primary")
                        with gr.Row():
                            cm_execute_result = gr.Markdown(visible=False)

                        with gr.Accordion(label="AI分析と処理", open=True):
                            with gr.Row():
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            gr.Markdown("モデル*", elem_classes="input-label")
                                        with gr.Column(scale=5):
                                            cm_ai_model_input = gr.Dropdown(
                                                show_label=False,
                                                choices=[
                                                    "xai.grok-code-fast-1",
                                                    "xai.grok-3",
                                                    "xai.grok-3-fast",
                                                    "xai.grok-4",
                                                    "xai.grok-4-fast-non-reasoning",
                                                    "meta.llama-4-scout-17b-16e-instruct",
                                                    "gpt-4o",
                                                    "gpt-5.1",
                                                ],
                                                value="xai.grok-code-fast-1",
                                                interactive=True,
                                                container=False,
                                            )
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            cm_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                            with gr.Row():
                                cm_ai_status_md = gr.Markdown(visible=False)
                            with gr.Row():
                                cm_ai_result_md = gr.Markdown(visible=False)

                    def _cm_refresh_objects():
                        try:
                            yield gr.Markdown(visible=True, value="⏳ テーブル・ビュー一覧を取得中..."), gr.CheckboxGroup(visible=False, choices=[]), gr.CheckboxGroup(visible=False, choices=[])
                            df_tab = _get_table_df_cached(pool, force=True)
                            df_view = _get_view_df_cached(pool, force=True)
                            names = []
                            if not df_tab.empty and "Table Name" in df_tab.columns:
                                names.extend([str(x) for x in df_tab["Table Name"].tolist()])
                            if not df_view.empty and "View Name" in df_view.columns:
                                names.extend([str(x) for x in df_view["View Name"].tolist()])
                            table_names = sorted(set([str(x) for x in (df_tab["Table Name"].tolist() if (not df_tab.empty and "Table Name" in df_tab.columns) else [])]))
                            view_names = sorted(set([str(x) for x in (df_view["View Name"].tolist() if (not df_view.empty and "View Name" in df_view.columns) else [])]))
                            status_text = "✅ 取得完了（データなし）" if (not table_names and not view_names) else "✅ 取得完了"
                            yield gr.Markdown(visible=True, value=status_text), gr.CheckboxGroup(choices=table_names, visible=True), gr.CheckboxGroup(choices=view_names, visible=True)
                        except Exception as e:
                            logger.error(f"_cm_refresh_objects error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

                    def _cm_fetch_stream(tables_selected, views_selected, sample_limit):
                        try:
                            tbls = tables_selected or []
                            vws = views_selected or []
                            lim = int(sample_limit) if sample_limit is not None else 10
                            if lim < 0:
                                lim = 0
                            if not tbls and not vws:
                                yield gr.Markdown(visible=True, value="⚠️ 対象を選択してください"), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ 取得中..."), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            struct = _cm_fetch_structure(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 構造情報取得完了"), gr.Accordion(open=True), struct, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            pk = _cm_fetch_pk(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 主キー情報取得完了"), gr.Accordion(open=True), struct, pk, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            fk = _cm_fetch_fk(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 外部キー情報取得完了"), gr.Accordion(open=True), struct, pk, fk, gr.Textbox(value="", autoscroll=False)
                            samples = _cm_fetch_samples(tbls, vws, lim)
                            yield gr.Markdown(visible=True, value="✅ サンプル取得完了"), gr.Accordion(open=True), struct, pk, fk, samples
                        except Exception as e:
                            logger.error(f"_cm_fetch_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)

                    def _cm_build_prompt(struct_text, pk_text, fk_text, samples_text, extra_text):
                        try:
                            prompt = (
                                "あなたはOracleデータベース専門家です。以下の情報に基づき、COMMENT文を生成してください。\n"
                                "出力はSQLのCOMMENT文のみ。\n"
                                "表・ビューはA-Zの順で、列はCREATE文の定義順で出力してください。\n\n"
                                "<構造>\n" + str(struct_text or "") + "\n\n"
                                "<主キー>\n" + str(pk_text or "") + "\n\n"
                                "<外部キー>\n" + str(fk_text or "") + "\n\n"
                                "<サンプル>\n" + str(samples_text or "") + "\n\n"
                                + (str(extra_text or "") if extra_text else "")
                            )
                            return prompt
                        except Exception as e:
                            logger.error(f"_cm_build_prompt error: {e}")
                            return str(e)

                    async def _cm_generate_async(obj_name, model_name, extra_text, struct_text, pk_text, fk_text, samples_text):
                        try:
                            prompt = _cm_build_prompt(struct_text, pk_text, fk_text, samples_text, extra_text)
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています", autoscroll=False)
                            from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                            client = AsyncOciOpenAI(
                                service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                auth=OciUserPrincipalAuth(),
                                compartment_id=compartment_id,
                            )
                            messages = [
                                {"role": "system", "content": "OracleのCOMMENT文のみを出力。説明文は200字以内。"},
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages)
                            text = ""
                            if resp.choices and len(resp.choices) > 0:
                                msg = resp.choices[0].message
                                text = msg.content if hasattr(msg, 'content') else ''
                            return gr.Textbox(value=text, autoscroll=False)
                        except Exception as e:
                            logger.error(f"_cm_generate_async error: {e}")
                            return gr.Textbox(value=f"❌ エラー: {e}", autoscroll=False)

                    def _cm_generate(obj_name, model_name, extra_text, struct_text, pk_text, fk_text, samples_text):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(_cm_generate_async(obj_name, model_name, extra_text, struct_text, pk_text, fk_text, samples_text))
                            return result
                        finally:
                            loop.close()

                    def _cm_generate_stream(model_name, struct_text, pk_text, fk_text, samples_text, extra_text, obj_tables):
                        try:
                            missing = []
                            if not model_name or not str(model_name).strip():
                                missing.append("モデル")
                            if not struct_text or not str(struct_text).strip():
                                missing.append("構造情報")
                            if missing:
                                msg = "⚠️ 必須入力が不足しています: " + ", ".join(missing)
                                yield gr.Markdown(visible=True, value=msg), gr.Textbox(value="", autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ 生成中..."), gr.Textbox(value="", autoscroll=False)
                            result = _cm_generate(obj_tables, model_name, extra_text, struct_text, pk_text, fk_text, samples_text)
                            yield gr.Markdown(visible=True, value="✅ 生成完了"), result
                        except Exception as e:
                            logger.error(f"_cm_generate_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 生成に失敗しました: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _cm_execute(sql_text):
                        from utils.management_util import execute_comment_sql
                        try:
                            res = execute_comment_sql(pool, sql_text)
                            return gr.Markdown(visible=True, value=str(res or ""))
                        except Exception as e:
                            logger.error(f"_cm_execute error: {e}")
                            return gr.Markdown(visible=True, value=f"❌ エラー: {str(e)}")

                    async def _cm_ai_analyze_async(model_name, sql_text, exec_result_text):
                        from utils.chat_util import get_oci_region, get_compartment_id
                        region = get_oci_region()
                        compartment_id = get_compartment_id()
                        if not region or not compartment_id:
                            return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                        try:
                            from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                            s = str(sql_text or "").strip()
                            r = str(exec_result_text or "").strip()
                            prompt = (
                                "以下のCOMMENT文の一括実行内容と実行結果を分析してください。出力は次の3点に限定します。\n"
                                "1) エラー原因（該当する場合）\n"
                                "2) 解決方法（修正案や具体的手順）\n"
                                "3) 簡潔な結論\n\n"
                                + ("SQL:\n```sql\n" + s + "\n```\n" if s else "")
                                + ("実行結果:\n" + r + "\n" if r else "")
                            )
                            client = AsyncOciOpenAI(
                                service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                auth=OciUserPrincipalAuth(),
                                compartment_id=compartment_id,
                            )
                            messages = [
                                {"role": "system", "content": "あなたはシニアDBエンジニアです。COMMENT ON TABLE/COLUMN の診断に特化し、必要最小限の要点のみを簡潔に提示してください。"},
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages)
                            text = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                text = msg.content if hasattr(msg, "content") else ""
                            return gr.Markdown(visible=True, value=text or "分析結果が空です")
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

                    def _cm_ai_analyze(model_name, sql_text, exec_result_text):
                        import asyncio
                        # 必須入力項目のチェック
                        if not model_name or not str(model_name).strip():
                            yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
                            return
                        if not sql_text or not str(sql_text).strip():
                            yield gr.Markdown(visible=True, value="⚠️ SQLを入力してください"), gr.Markdown(visible=False)
                            return
                        if not exec_result_text or not str(exec_result_text).strip():
                            yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先に一括実行を実行してください"), gr.Markdown(visible=False)
                            return
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                            result_md = loop.run_until_complete(_cm_ai_analyze_async(model_name, sql_text, exec_result_text))
                            yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                        finally:
                            loop.close()

                    cm_refresh_btn.click(
                        fn=_cm_refresh_objects,
                        outputs=[cm_refresh_status, cm_tables_input, cm_views_input],
                    )

                    def _cm_fetch_structure(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        struct_chunks = []
                        for kind, name in targets:
                            if kind == "VIEW":
                                cols_df, _ddl = get_view_details(pool, name)
                            else:
                                cols_df, _ddl = get_table_details(pool, name)
                            lines = [f"OBJECT: {name}", "COLUMNS:"]
                            if isinstance(cols_df, pd.DataFrame) and not cols_df.empty:
                                for _, row in cols_df.iterrows():
                                    lines.append(f"- {row['Column Name']}: {row['Data Type']} NULLABLE={row['Nullable']}")
                            struct_chunks.append("\n".join(lines))
                        struct_text = "\n\n".join(struct_chunks)
                        return gr.Textbox(value=struct_text, interactive=True, autoscroll=False)

                    def _cm_fetch_pk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import get_primary_key_info
                        pk_chunks = []
                        for _kind, name in targets:
                            pk_info = get_primary_key_info(pool, name) or ""
                            if pk_info:
                                pk_chunks.append(f"OBJECT: {name}\n{pk_info}")
                        pk_text = "\n\n".join(pk_chunks) if pk_chunks else ""
                        return gr.Textbox(value=pk_text, interactive=True, autoscroll=False)

                    def _cm_fetch_fk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import get_foreign_key_info
                        fk_chunks = []
                        for _kind, name in targets:
                            fk_info = get_foreign_key_info(pool, name) or ""
                            if fk_info:
                                fk_chunks.append(f"OBJECT: {name}\n{fk_info}")
                        fk_text = "\n\n".join(fk_chunks) if fk_chunks else ""
                        return gr.Textbox(value=fk_text, interactive=True, autoscroll=False)

                    def _cm_fetch_samples(tables_selected, views_selected, sample_limit):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import display_table_data
                        lim = int(sample_limit)
                        samples_chunks = []
                        if lim > 0:
                            for _kind, name in targets:
                                df = display_table_data(pool, name, lim)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    samples_chunks.append(f"OBJECT: {name}\n" + df.to_csv(index=False))
                        samples_text = "\n\n".join(samples_chunks) if samples_chunks else ""
                        return gr.Textbox(value=samples_text, interactive=True, autoscroll=False)

                    cm_fetch_btn.click(
                        fn=_cm_fetch_stream,
                        inputs=[cm_tables_input, cm_views_input, cm_sample_limit],
                        outputs=[cm_fetch_status_md, cm_input_confirm_acc, cm_structure_text, cm_pk_text, cm_fk_text, cm_samples_text],
                    )

                    cm_generate_btn.click(
                        fn=_cm_generate_stream,
                        inputs=[cm_model_input, cm_structure_text, cm_pk_text, cm_fk_text, cm_samples_text, cm_extra_input, cm_tables_input],
                        outputs=[cm_generate_status_md, cm_generated_sql],
                    )

                    cm_execute_btn.click(
                        fn=_cm_execute,
                        inputs=[cm_generated_sql],
                        outputs=[cm_execute_result],
                    )

                    cm_ai_analyze_btn.click(
                        fn=_cm_ai_analyze,
                        inputs=[cm_ai_model_input, cm_generated_sql, cm_execute_result],
                        outputs=[cm_ai_status_md, cm_ai_result_md],
                    )

                    def _on_feedback_type_change(fb_type, gen_sql_text):
                        t = str(fb_type or "").lower()
                        if t == "positive":
                            return gr.Textbox(value=str(gen_sql_text or ""), interactive=False, autoscroll=False), gr.Textbox(interactive=True, autoscroll=False)
                        return gr.Textbox(interactive=True, autoscroll=False), gr.Textbox(interactive=True, autoscroll=False)

                    dev_feedback_type_select.change(
                        fn=_on_feedback_type_change,
                        inputs=[dev_feedback_type_select, dev_generated_sql_text],
                        outputs=[dev_feedback_response_text, dev_feedback_content_text],
                    )

                    dev_feedback_send_btn.click(
                        fn=_send_feedback,
                        inputs=[dev_feedback_type_select, dev_feedback_response_text, dev_feedback_content_text, dev_prompt_input, dev_profile_select],
                        outputs=[dev_feedback_status, dev_feedback_result, dev_feedback_used_sql_text],
                    )

                with gr.TabItem(label="アノテーション管理"):
                    with gr.Accordion(label="1. オブジェクト選択", open=True):
                        with gr.Row():
                            am_refresh_btn = gr.Button("テーブル・ビュー一覧を取得（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            am_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択*")
                                am_tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択*")
                                am_views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("サンプル件数", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        am_sample_limit = gr.Number(show_label=False, minimum=0, maximum=100, value=10, interactive=True, container=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                am_fetch_btn = gr.Button("情報を取得", variant="primary")
                        with gr.Row():
                            am_fetch_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. 入力確認", open=False) as am_input_confirm_acc:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("構造情報*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_structure_text = gr.Textbox(show_label=False, lines=8, max_lines=16, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("主キー情報(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_pk_text = gr.Textbox(show_label=False, lines=4, max_lines=10, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("外部キー情報(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_fk_text = gr.Textbox(show_label=False, lines=6, max_lines=14, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("サンプルデータ(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_samples_text = gr.Textbox(show_label=False, lines=8, max_lines=16, interactive=True, show_copy_button=True, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("追加入力(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_extra_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="追加で考慮してほしい説明や条件を記入",
                                    value=(
                                        "ANNOTATIONSの安全な適用ガイド:\n"
                                        "- DROPとADDは同一文で混在させず、別々のALTER文に分割\n"
                                        "- 一括実行では重複名(DROP/ADD同時指定)がORA-11562の原因、順次個別に実行\n"
                                        "- 可能ならDROP後はADD IF NOT EXISTSで再追加、重複を回避\n"
                                        "- 値の'は''へエスケープ、予約語/空白は注釈名を二重引用符\n"
                                        "例(表): ALTER TABLE USERS ANNOTATIONS (DROP IF EXISTS sample_header);\n"
                                        "例(列): ALTER TABLE USERS MODIFY (ID ANNOTATIONS (ADD IF NOT EXISTS ui_display 'ID'));\n"
                                        "再追加例: ALTER TABLE USERS ANNOTATIONS (ADD sample_data 'value');\n"
                                    ),
                                    lines=8,
                                    max_lines=16,
                                    container=False,
                                    autoscroll=False,
                                )

                    with gr.Accordion(label="3. アノテーション自動生成", open=False):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("モデル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        am_model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "xai.grok-code-fast-1",
                                                "xai.grok-3",
                                                "xai.grok-3-fast",
                                                "xai.grok-4",
                                                "xai.grok-4-fast-non-reasoning",
                                                "meta.llama-4-scout-17b-16e-instruct",
                                                "gpt-4o",
                                                "gpt-5.1",
                                            ],
                                            value="xai.grok-code-fast-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        am_generate_btn = gr.Button("生成（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            am_generate_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("生成されたSQL*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                am_generated_sql = gr.Textbox(show_label=False, lines=15, max_lines=15, interactive=True, show_copy_button=True, container=False, autoscroll=False)

                    with gr.Accordion(label="4. 実行", open=False):
                        with gr.Row():
                            am_execute_btn = gr.Button("一括実行", variant="primary")
                        with gr.Row():
                            am_execute_result = gr.Markdown(visible=False)

                        with gr.Accordion(label="AI分析と処理", open=True):
                            with gr.Row():
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            gr.Markdown("モデル*", elem_classes="input-label")
                                        with gr.Column(scale=5):
                                            am_ai_model_input = gr.Dropdown(
                                                show_label=False,
                                                choices=[
                                                    "xai.grok-code-fast-1",
                                                    "xai.grok-3",
                                                    "xai.grok-3-fast",
                                                    "xai.grok-4",
                                                    "xai.grok-4-fast-non-reasoning",
                                                    "meta.llama-4-scout-17b-16e-instruct",
                                                    "gpt-4o",
                                                    "gpt-5.1",
                                                ],
                                                value="xai.grok-code-fast-1",
                                                interactive=True,
                                                container=False,
                                            )
                                with gr.Column(scale=5):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            am_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                            with gr.Row():
                                am_ai_status_md = gr.Markdown(visible=False)
                            with gr.Row():
                                am_ai_result_md = gr.Markdown(visible=False)

                    def _am_refresh_objects():
                        try:
                            yield gr.Markdown(visible=True, value="⏳ テーブル・ビュー一覧を取得中..."), gr.CheckboxGroup(visible=False, choices=[]), gr.CheckboxGroup(visible=False, choices=[])
                            df_tab = _get_table_df_cached(pool, force=True)
                            df_view = _get_view_df_cached(pool, force=True)
                            table_names = sorted(set([str(x) for x in (df_tab["Table Name"].tolist() if (not df_tab.empty and "Table Name" in df_tab.columns) else [])]))
                            view_names = sorted(set([str(x) for x in (df_view["View Name"].tolist() if (not df_view.empty and "View Name" in df_view.columns) else [])]))
                            status_text = "✅ 取得完了（データなし）" if (not table_names and not view_names) else "✅ 取得完了"
                            yield gr.Markdown(visible=True, value=status_text), gr.CheckboxGroup(choices=table_names, visible=True), gr.CheckboxGroup(choices=view_names, visible=True)
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

                    def _am_fetch_structure(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        struct_chunks = []
                        for kind, name in targets:
                            if kind == "VIEW":
                                cols_df, _ddl = get_view_details(pool, name)
                            else:
                                cols_df, _ddl = get_table_details(pool, name)
                            lines = [f"OBJECT: {name}", "COLUMNS:"]
                            if isinstance(cols_df, pd.DataFrame) and not cols_df.empty:
                                for _, row in cols_df.iterrows():
                                    lines.append(f"- {row['Column Name']}: {row['Data Type']} NULLABLE={row['Nullable']}")
                            struct_chunks.append("\n".join(lines))
                        struct_text = "\n\n".join(struct_chunks)
                        return gr.Textbox(value=struct_text, interactive=True, autoscroll=False)

                    def _am_fetch_pk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import get_primary_key_info
                        pk_chunks = []
                        for _kind, name in targets:
                            pk_info = get_primary_key_info(pool, name) or ""
                            if pk_info:
                                pk_chunks.append(f"OBJECT: {name}\n{pk_info}")
                        pk_text = "\n\n".join(pk_chunks) if pk_chunks else ""
                        return gr.Textbox(value=pk_text, interactive=True, autoscroll=False)

                    def _am_fetch_fk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import get_foreign_key_info
                        fk_chunks = []
                        for _kind, name in targets:
                            fk_info = get_foreign_key_info(pool, name) or ""
                            if fk_info:
                                fk_chunks.append(f"OBJECT: {name}\n{fk_info}")
                        fk_text = "\n\n".join(fk_chunks) if fk_chunks else ""
                        return gr.Textbox(value=fk_text, interactive=True, autoscroll=False)

                    def _am_fetch_samples(tables_selected, views_selected, sample_limit):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True, autoscroll=False)
                        from utils.management_util import display_table_data
                        lim = int(sample_limit)
                        samples_chunks = []
                        if lim > 0:
                            for _kind, name in targets:
                                df = display_table_data(pool, name, lim)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    samples_chunks.append(f"OBJECT: {name}\n" + df.to_csv(index=False))
                        samples_text = "\n\n".join(samples_chunks) if samples_chunks else ""
                        return gr.Textbox(value=samples_text, interactive=True, autoscroll=False)

                    def _am_build_prompt(struct_text, pk_text, fk_text, samples_text, extra_text):
                        has_samples = bool(str(samples_text or "").strip())
                        prompt = (
                            "あなたはOracleデータベース専門家です。以下の情報に基づき、ALTER TABLE/ALTER VIEW の ANNOTATIONS 文のみを生成してください。\n"
                            "出力はSQLのアノテーション文のみ。説明や余計な文は出力しないでください。\n"
                            "テーブル・ビューはA-Zの順、列は定義順で出力してください。\n"
                            "ビューの列レベルのアノテーションは生成しないでください（列はテーブル列に対してのみ生成）。\n\n"
                            "参考構文とルール:\n"
                            "- 対象: TABLE / VIEW / MATERIALIZED VIEW / INDEX（本ツールでは TABLE 列と VIEW 本体を対象）\n"
                            "- 操作: ADD / DROP / REPLACE（CREATE 時は ADD/ADD IF NOT EXISTS のみ）\n"
                            "- 注釈名: 英数字と $, _, # を無引用で許容。予約語や空白を含む場合は二重引用符。最大1024文字。\n"
                            "- 値: 最大4000文字。単一引用符は '' にエスケープ。\n"
                            "- 複数注釈は同一文で列挙可能。\n"
                            + ("- サンプルが取得できた場合のみ 'sample_header' と 'sample_data' を生成する。\n" if has_samples else "- サンプルが無い場合は 'sample_header' と 'sample_data' を生成しない。\n")
                            + "例:\n"
                            + "  ALTER TABLE T1 ANNOTATIONS (Operations '[\"Sort\", \"Group\"]', Hidden);\n"
                            + "  ALTER TABLE T1 MODIFY (ID ANNOTATIONS (UI_Display 'ID', Classification 'Doc Info'));\n"
                            + "  ALTER VIEW SALES_V ANNOTATIONS (UI_Display 'Sales View');\n\n"
                            + "<構造>\n" + str(struct_text or "") + "\n\n"
                            + "<主キー>\n" + str(pk_text or "") + "\n\n"
                            + "<外部キー>\n" + str(fk_text or "") + "\n\n"
                            + "<サンプル>\n" + str(samples_text or "") + "\n\n"
                            + (str(extra_text or "") if extra_text else "")
                        )
                        return prompt

                    async def _am_generate_async(model_name, struct_text, pk_text, fk_text, samples_text, extra_text):
                        try:
                            prompt = _am_build_prompt(struct_text, pk_text, fk_text, samples_text, extra_text)
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                logger.error("_am_generate_async missing OCI configuration: region or compartment_id is empty")
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています", autoscroll=False)
                            
                            if str(model_name).startswith("gpt-"):
                                from openai import AsyncOpenAI
                                client = AsyncOpenAI()
                            else:
                                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                                client = AsyncOciOpenAI(
                                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                    auth=OciUserPrincipalAuth(),
                                    compartment_id=compartment_id,
                                )
                            messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "出力は次の形式のみ: \n"
                                        "- テーブル: ALTER TABLE <表> ANNOTATIONS (<name> '<value>'[, ...]);\n"
                                        "- 列: ALTER TABLE <表> MODIFY (<列> ANNOTATIONS (<name> '<value>'[, ...]));\n"
                                        "- ビュー: ALTER VIEW <ビュー> ANNOTATIONS (<name> '<value>'[, ...]);\n"
                                        "制約: ビュー列のアノテーションは生成しない。'data_type' と 'nullable' を優先的に使用。'sample_header' と 'sample_data' はサンプルが存在する場合のみ生成。'type' は使用しない。値内の単一引用符は '' にエスケープ。余計な説明は出力しない。\n\n"
                                        "Oracle公式の annotations_clause ルール:\n"
                                        "- ADD / DROP / REPLACE をサポート（CREATE は ADD/ADD IF NOT EXISTS）。\n"
                                        "- 注釈名は識別子。予約語や空白を含む場合は二重引用符。\n"
                                        "- 値は最大4000文字。複数注釈は同一文で列挙可能。\n"
                                        "例: ALTER TABLE T1 ANNOTATIONS (Operations '[\"Sort\", \"Group\"]', Hidden);\n"
                                        "例: ALTER TABLE T1 MODIFY (ID ANNOTATIONS (UI_Display 'ID'));\n"
                                        "例: ALTER VIEW V1 ANNOTATIONS (UI_Display 'Sales View');"
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages, temperature=0.0)
                            text = ""
                            if resp.choices and len(resp.choices) > 0:
                                msg = resp.choices[0].message
                                text = msg.content if hasattr(msg, "content") else ""
                            # サンプルが無い場合は、出力から sample_header / sample_data を除去
                            if not str(samples_text or "").strip():
                                try:
                                    s = str(text or "")
                                    def _split_items(inner):
                                        items = []
                                        current = []
                                        in_quote = False
                                        quote = ''
                                        i = 0
                                        n = len(inner)
                                        while i < n:
                                            ch = inner[i]
                                            if in_quote:
                                                current.append(ch)
                                                if ch == quote:
                                                    if quote == "'" and i + 1 < n and inner[i + 1] == "'":
                                                        current.append("'")
                                                        i += 1
                                                    else:
                                                        in_quote = False
                                                        quote = ''
                                            else:
                                                if ch == "'" or ch == '"':
                                                    in_quote = True
                                                    quote = ch
                                                    current.append(ch)
                                                elif ch == ',':
                                                    items.append(''.join(current).strip())
                                                    current = []
                                                else:
                                                    current.append(ch)
                                            i += 1
                                        items.append(''.join(current).strip())
                                        return [it for it in items if it]
                                    def _extract_name(part):
                                        m = re.match(r'^\s*("([^"]+)"|([A-Za-z0-9_\$#]+))', part)
                                        if not m:
                                            return ''
                                        return (m.group(2) or m.group(3) or '').strip()
                                    out_lines = []
                                    for ln in s.splitlines():
                                        up = ln.upper()
                                        if 'ANNOTATIONS' in up:
                                            m = re.search(r'ANNOTATIONS\s*\((.*)\)', ln, flags=re.IGNORECASE)
                                            if m:
                                                inner = m.group(1)
                                                items = _split_items(inner)
                                                kept = []
                                                for it in items:
                                                    nm = _extract_name(it)
                                                    if nm.lower() in ('sample_header', 'sample_data'):
                                                        continue
                                                    kept.append(it)
                                                if kept:
                                                    new_inner = ', '.join(kept)
                                                    new_ln = re.sub(r'(ANNOTATIONS\s*)\((.*)\)', r"\1(" + new_inner + ")", ln, flags=re.IGNORECASE)
                                                    out_lines.append(new_ln)
                                                else:
                                                    continue
                                            else:
                                                out_lines.append(ln)
                                        else:
                                            out_lines.append(ln)
                                    text = "\n".join(out_lines)
                                except Exception:
                                    pass
                            return gr.Textbox(value=text, autoscroll=False)
                        except Exception as e:
                            logger.error(f"_am_generate_async error: {e}")
                            return gr.Textbox(value=f"❌ エラー: {e}", autoscroll=False)

                    async def _am_ai_analyze_async(model_name, sql_text, exec_result_text):
                        from utils.chat_util import get_oci_region, get_compartment_id
                        region = get_oci_region()
                        compartment_id = get_compartment_id()
                        if not region or not compartment_id:
                            return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                        try:
                            s = str(sql_text or "").strip()
                            r = str(exec_result_text or "").strip()
                            prompt = (
                                "以下のアノテーション文の一括実行内容と実行結果を分析してください。出力は次の3点に限定します。\n"
                                "1) エラー原因（該当する場合）\n"
                                "2) 解決方法（修正案や具体的手順）\n"
                                "3) 簡潔な結論\n\n"
                                + ("SQL:\n```sql\n" + s + "\n```\n" if s else "")
                                + ("実行結果:\n" + r + "\n" if r else "")
                            )
                            
                            if str(model_name).startswith("gpt-"):
                                from openai import AsyncOpenAI
                                client = AsyncOpenAI()
                            else:
                                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                                client = AsyncOciOpenAI(
                                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                    auth=OciUserPrincipalAuth(),
                                    compartment_id=compartment_id,
                                )
                            
                            messages = [
                                {"role": "system", "content": "あなたはシニアDBエンジニアです。ALTER ... ANNOTATIONS の診断に特化し、必要最小限の要点のみを簡潔に提示してください。"},
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages)
                            text = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                text = msg.content if hasattr(msg, "content") else ""
                            return gr.Markdown(visible=True, value=text or "分析結果が空です")
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

                    def _am_ai_analyze(model_name, sql_text, exec_result_text):
                        import asyncio
                        # 必須入力項目のチェック
                        if not model_name or not str(model_name).strip():
                            yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
                            return
                        if not sql_text or not str(sql_text).strip():
                            yield gr.Markdown(visible=True, value="⚠️ SQLを入力してください"), gr.Markdown(visible=False)
                            return
                        if not exec_result_text or not str(exec_result_text).strip():
                            yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先に一括実行を実行してください"), gr.Markdown(visible=False)
                            return
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                            result_md = loop.run_until_complete(_am_ai_analyze_async(model_name, sql_text, exec_result_text))
                            yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                        finally:
                            loop.close()

                    def _am_generate(model_name, struct_text, pk_text, fk_text, samples_text, extra_text):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(_am_generate_async(model_name, struct_text, pk_text, fk_text, samples_text, extra_text))
                            return result
                        finally:
                            loop.close()

                    def _am_execute(sql_text):
                        def _prep(s):
                            txt = str(s or "")
                            parts = [p.strip() for p in txt.split(';') if p.strip()]
                            out = []
                            for p in parts:
                                out.append(p)
                            return ";\n".join(out)
                        from utils.management_util import execute_annotation_sql
                        try:
                            res = execute_annotation_sql(pool, _prep(sql_text))
                            return gr.Markdown(visible=True, value=str(res or ""))
                        except Exception as e:
                            logger.error(f"_am_execute error: {e}")
                            return gr.Markdown(visible=True, value=f"❌ エラー: {str(e)}")

                    am_refresh_btn.click(
                        fn=_am_refresh_objects,
                        outputs=[am_refresh_status, am_tables_input, am_views_input],
                    )

                    def _am_fetch_stream(tables_selected, views_selected, sample_limit):
                        try:
                            tbls = tables_selected or []
                            vws = views_selected or []
                            lim = int(sample_limit) if sample_limit is not None else 10
                            if lim < 0:
                                lim = 0
                            if not tbls and not vws:
                                yield gr.Markdown(visible=True, value="⚠️ 対象を選択してください"), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ 取得中..."), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            struct = _am_fetch_structure(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 構造情報取得完了"), gr.Accordion(open=True), struct, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            pk = _am_fetch_pk(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 主キー情報取得完了"), gr.Accordion(open=True), struct, pk, gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                            fk = _am_fetch_fk(tbls, vws)
                            yield gr.Markdown(visible=True, value="✅ 外部キー情報取得完了"), gr.Accordion(open=True), struct, pk, fk, gr.Textbox(value="", autoscroll=False)
                            samples = _am_fetch_samples(tbls, vws, lim)
                            yield gr.Markdown(visible=True, value="✅ サンプル取得完了"), gr.Accordion(open=True), struct, pk, fk, samples
                        except Exception as e:
                            logger.error(f"_am_fetch_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Accordion(open=True), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)

                    am_fetch_btn.click(
                        fn=_am_fetch_stream,
                        inputs=[am_tables_input, am_views_input, am_sample_limit],
                        outputs=[am_fetch_status_md, am_input_confirm_acc, am_structure_text, am_pk_text, am_fk_text, am_samples_text],
                    )

                    def _am_generate_stream(model_name, struct_text, pk_text, fk_text, samples_text, extra_text):
                        try:
                            missing = []
                            if not model_name or not str(model_name).strip():
                                missing.append("モデル")
                            if not struct_text or not str(struct_text).strip():
                                missing.append("構造情報")
                            if missing:
                                msg = "⚠️ 必須入力が不足しています: " + ", ".join(missing)
                                yield gr.Markdown(visible=True, value=msg), gr.Textbox(value="", autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ 生成中..."), gr.Textbox(value="", autoscroll=False)
                            result = _am_generate(model_name, struct_text, pk_text, fk_text, samples_text, extra_text)
                            yield gr.Markdown(visible=True, value="✅ 生成完了"), result
                        except Exception as e:
                            logger.error(f"_am_generate_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 生成に失敗しました: {e}"), gr.Textbox(value="", autoscroll=False)

                    am_generate_btn.click(
                        fn=_am_generate_stream,
                        inputs=[am_model_input, am_structure_text, am_pk_text, am_fk_text, am_samples_text, am_extra_input],
                        outputs=[am_generate_status_md, am_generated_sql],
                    )

                    am_execute_btn.click(
                        fn=_am_execute,
                        inputs=[am_generated_sql],
                        outputs=[am_execute_result],
                    )

                    am_ai_analyze_btn.click(
                        fn=_am_ai_analyze,
                        inputs=[am_ai_model_input, am_generated_sql, am_execute_result],
                        outputs=[am_ai_status_md, am_ai_result_md],
                    )

                with gr.TabItem(label="合成データ生成") as synthetic_tab:
                    with gr.Accordion(label="1. 対象選択", open=True):
                        with gr.Row():
                            gr.Markdown(visible=True, value="ℹ️ Profileと対象テーブルを選択し、生成開始を押下してください")
                        with gr.Row():
                            with gr.Column(scale=5):
                                # プロフィール選択肢を取得し、空の場合は空文字列を含むリストを設定
                                _syn_initial_choices = _load_profiles_from_json()
                                if not _syn_initial_choices:
                                    _syn_initial_choices = [("", "")]
                                syn_profile_select = gr.Dropdown(
                                    show_label=False,
                                    choices=_syn_initial_choices,
                                    value=(
                                        _syn_initial_choices[0][1]
                                        if (_syn_initial_choices and isinstance(_syn_initial_choices[0], tuple))
                                        else (_syn_initial_choices[0] if _syn_initial_choices else "")
                                    ),
                                    interactive=True,
                                    container=False
                                )
                            with gr.Column(scale=5):
                                syn_refresh_btn = gr.Button("テーブル一覧を取得（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                syn_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        syn_tables_input = gr.CheckboxGroup(label="テーブル選択*", choices=[], visible=True)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("各テーブルの生成件数*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_rows_per_table = gr.Number(show_label=False, minimum=1, maximum=100, value=1, interactive=True, container=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("生成の指示(オプション)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                syn_prompt_input = gr.Textbox(show_label=False, placeholder="スキーマ特性や分布、制約などを自然言語で記述", lines=4, max_lines=10, container=False, autoscroll=False)
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("サンプル行数(sample_rows)*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_sample_rows = gr.Number(show_label=False, minimum=0, maximum=100, value=5, interactive=True, container=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("コメントを考慮(comments)", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_comments = gr.Checkbox(label="", value=True, container=False)

                        with gr.Row():
                            syn_generate_btn = gr.Button("生成開始（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            syn_generate_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. 進捗と状態", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("オペレーションID*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_operation_id_text = gr.Textbox(show_label=False, interactive=False, container=False, autoscroll=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        syn_status_update_btn = gr.Button("ステータスを更新", variant="primary", visible=False)
                        with gr.Row():
                            syn_status_update_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            syn_status_df = gr.Dataframe(label="ステータス", interactive=False, wrap=True, visible=False, value=pd.DataFrame())
                        with gr.Row():
                            syn_status_style = gr.HTML(visible=False)

                    with gr.Accordion(label="3. 結果確認", open=True):
                        with gr.Row():
                            gr.Markdown(visible=True, value="ℹ️ 生成済みテーブルからデータを表示します")
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("テーブル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_result_table_select = gr.Dropdown(show_label=False, choices=[], interactive=True, container=False)
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("取得件数*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        syn_result_limit = gr.Number(show_label=False, value=100, minimum=1, maximum=10000, container=False)
                        with gr.Row():
                            syn_result_btn = gr.Button("データを表示", variant="primary")
                        with gr.Row():
                            syn_result_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            syn_result_df = gr.Dataframe(label="データ表示", interactive=False, wrap=True, visible=False, value=pd.DataFrame(), elem_id="synthetic_data_result_df")
                        with gr.Row():
                            syn_result_style = gr.HTML(visible=False)

                    def _syn_refresh_objects(profile_name):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ テーブル一覧を取得中..."), gr.CheckboxGroup(visible=False, choices=[], value=[]), gr.Dropdown(visible=False, choices=[], value=None)
                            tables, _ = _get_profile_objects_from_json(profile_name)
                            if not tables:
                                prof = _resolve_profile_name(pool, str(profile_name or ""))
                                df_tab = _get_table_df_cached(pool, force=True)
                                all_table_names = [str(x) for x in (df_tab["Table Name"].tolist() if (not df_tab.empty and "Table Name" in df_tab.columns) else [])]
                                table_names = sorted(set(all_table_names))
                                try:
                                    attrs = _get_profile_attributes(pool, prof) or {}
                                    obj_list = attrs.get("object_list") or []
                                    prof_tables = sorted(set([str(o.get("name")) for o in obj_list if o and o.get("name")]))
                                    if prof_tables:
                                        table_names = [t for t in table_names if t in prof_tables]
                                except Exception as e:
                                    logger.error(f"_syn_refresh_objects filter by profile error: {e}")
                                tables = table_names
                            status_text = "✅ 取得完了(データなし)" if (not tables) else "✅ 取得完了"
                            # テーブル選択を空にリセット
                            yield gr.Markdown(visible=True, value=status_text), gr.CheckboxGroup(choices=tables, visible=True, value=[]), gr.Dropdown(choices=tables, visible=True, value=None)
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[], value=[]), gr.Dropdown(choices=[], value=None)

                    def _syn_generate(profile_name, tables_selected, rows_per_table, extra_text, sample_rows, comments):
                        """合成データ生成処理.
                        
                        Args:
                            profile_name: プロファイル名
                            tables_selected: 選択されたテーブルリスト
                            rows_per_table: テーブルあたりの行数
                            extra_text: 追加プロンプト
                            sample_rows: サンプル行数
                            comments: コメントを考慮するか
                        
                        Yields:
                            gr.Markdown: ステータスメッセージ
                        
                        Returns:
                            str or None: オペレーションID(成功時)、None(失敗時)
                        """
                        op_id = None
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 合成データ生成を準備中...")
                            pj = _get_profile_json_entry(str(profile_name or ""))
                            prof = str((pj or {}).get("profile") or "").strip() or str(profile_name or "").strip()
                            yield gr.Markdown(visible=True, value="⏳ データベースに接続中...")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    try:
                                        p_base = {
                                            "comments": bool(comments),
                                        }
                                    except Exception:
                                        p_base = {"comments": False}
                                    try:
                                        sel = list(tables_selected or [])
                                        if len(sel) == 1:
                                            yield gr.Markdown(visible=True, value=f"⏳ テーブル '{sel[0]}' の合成データ生成中...")
                                            obj_name = str(sel[0])
                                            rc = int(rows_per_table or 0)
                                            p_single = dict(p_base)
                                            p_single["sample_rows"] = int(sample_rows or 0)
                                            p_json = json.dumps(p_single, ensure_ascii=False)
                                            cursor.execute(
                                                "BEGIN DBMS_CLOUD_AI.GENERATE_SYNTHETIC_DATA(profile_name => :name, object_name => :obj, owner_name => :owner, record_count => :rc, user_prompt => :up, params => :p); END;",
                                                name=prof,
                                                obj=obj_name,
                                                owner="ADMIN",
                                                rc=rc,
                                                up=str(extra_text or ""),
                                                p=p_json,
                                            )
                                        else:
                                            yield gr.Markdown(visible=True, value=f"⏳ {len(sel)} 件のテーブルの合成データ生成中...")
                                            rc = int(rows_per_table or 0)
                                            sr = int(sample_rows or 0)
                                            obj_list = []
                                            for t in sel:
                                                obj_list.append({"owner": "ADMIN", "name": str(t), "record_count": rc})
                                            obj_json = json.dumps(obj_list, ensure_ascii=False)
                                            # sample_rowsはparamsに含める
                                            p_multi = dict(p_base)
                                            p_multi["sample_rows"] = sr
                                            p_json = json.dumps(p_multi, ensure_ascii=False)
                                            cursor.execute(
                                                "BEGIN DBMS_CLOUD_AI.GENERATE_SYNTHETIC_DATA(profile_name => :name, object_list => :objlist, params => :p); END;",
                                                name=prof,
                                                objlist=obj_json,
                                                p=p_json,
                                            )
                                        yield gr.Markdown(visible=True, value="⏳ オペレーションIDを取得中...")
                                        cursor.execute("SELECT max(id) FROM user_load_operations")
                                        rid = cursor.fetchall() or []
                                        if rid and len(rid) > 0:
                                            try:
                                                v0 = rid[0][0]
                                                op_id = str(v0.read() if hasattr(v0, "read") else v0)
                                            except Exception:
                                                try:
                                                    op_id = str(rid[0][0])
                                                except Exception:
                                                    op_id = None
                                    except Exception as e:
                                        # データベースエラーを詳細にログ出力
                                        error_msg = str(e)
                                        logger.error(f"合成データ生成のデータベースエラー: {error_msg}")
                                        
                                        # ORA-20003エラーの場合、ステータステーブルからエラー詳細を取得
                                        if "ORA-20003" in error_msg or "Operation failed" in error_msg:
                                            try:
                                                # エラーメッセージからテーブル名を抽出
                                                import re
                                                match = re.search(r'SYNTHETIC_DATA\$(\d+)_STATUS', error_msg)
                                                if match:
                                                    extracted_op_id = match.group(1)
                                                    logger.info(f"エラーメッセージからオペレーションID抽出: {extracted_op_id}")
                                                    
                                                    # ステータステーブルからERROR_MESSAGEを取得
                                                    status_table = f'"SYNTHETIC_DATA${extracted_op_id}_STATUS"'
                                                    status_sql = f"SELECT ERROR_MESSAGE FROM ADMIN.{status_table} WHERE ERROR_MESSAGE IS NOT NULL FETCH FIRST 1 ROWS ONLY"
                                                    cursor.execute(status_sql)
                                                    error_rows = cursor.fetchall() or []
                                                    
                                                    if error_rows and len(error_rows) > 0:
                                                        db_error_msg = error_rows[0][0]
                                                        if hasattr(db_error_msg, 'read'):
                                                            db_error_msg = db_error_msg.read()
                                                        db_error_msg = str(db_error_msg).strip()
                                                        
                                                        if db_error_msg:
                                                            logger.info(f"ステータステーブルからエラー詳細取得: {db_error_msg}")
                                                            yield gr.Markdown(visible=True, value=f"❌ 合成データ生成エラー: {db_error_msg}")
                                                            return None
                                            except Exception as status_err:
                                                logger.error(f"ステータステーブル照会エラー: {status_err}")
                                            
                                            # ステータステーブルからエラーが取得できなかった場合
                                            yield gr.Markdown(visible=True, value="❌ 合成データ生成中にエラーが発生しました。詳細は「ステータスを更新」ボタンで確認してください。")
                                        elif "ORA-" in error_msg:
                                            # その他のOracleエラー
                                            yield gr.Markdown(visible=True, value="❌ データベースエラーが発生しました。設定内容やProfile設定を確認してください。")
                                        else:
                                            # 一般的なエラー
                                            yield gr.Markdown(visible=True, value="❌ 合成データ生成に失敗しました。入力内容を確認してください。")
                                        return None
                                    
                                    if not op_id:
                                        yield gr.Markdown(visible=True, value="❌ 合成データ生成に失敗しました: オペレーションIDを取得できませんでした")
                                        return None
                                    yield gr.Markdown(visible=True, value="✅ 合成データ生成を開始しました")
                                    return op_id
                        except Exception as e:
                            logger.error(f"合成データ生成処理エラー: {e}")
                            yield gr.Markdown(visible=True, value="❌ 予期しないエラーが発生しました。")
                            return None

                    def _syn_update_status(op_id):
                        op = str(op_id or "").strip()
                        if not op:
                            return gr.Markdown(visible=True, value="⚠️ オペレーションIDを入力/取得してください"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    tab = f"\"SYNTHETIC_DATA${op.upper()}_STATUS\""
                                    sql = f"SELECT * FROM ADMIN.{tab} FETCH FIRST 200 ROWS ONLY"
                                    cursor.execute(sql)
                                    rows = cursor.fetchall() or []
                                    cols = [d[0] for d in cursor.description] if cursor.description else []
                                    df = pd.DataFrame(rows, columns=cols)
                                    keep = [
                                        "ID",
                                        "NAME",
                                        "ROWS_LOADED",
                                        "STATUS",
                                    ]
                                    show_cols = [c for c in keep if c in df.columns]
                                    if show_cols:
                                        df = df[show_cols]
                                    df_component = gr.Dataframe(visible=True, value=df, label=f"ステータス（件数: {len(df)}）", elem_id="synthetic_data_status_df")
                                    style_value = ""
                                    if len(cols) > 0:
                                        sample = df.head(5)
                                        widths = []
                                        for col in df.columns:
                                            series = sample[col].astype(str) if not sample.empty else pd.Series([], dtype=str)
                                            row_max = series.map(len).max() if len(series) > 0 else 0
                                            length = max(len(str(col)), row_max)
                                            widths.append(length)
                                        total = sum(widths) if widths else 0
                                        if total > 0:
                                            col_widths = [max(5, int(100 * w / total)) for w in widths]
                                            diff = 100 - sum(col_widths)
                                            if diff != 0 and len(col_widths) > 0:
                                                col_widths[0] = max(5, col_widths[0] + diff)
                                            rules = []
                                            rules.append("#synthetic_data_status_df { width: 100% !important; }")
                                            rules.append("#synthetic_data_status_df .wrap { overflow-x: auto !important; }")
                                            rules.append("#synthetic_data_status_df table { table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; }")
                                            for idx, pct in enumerate(col_widths, start=1):
                                                rules.append(f"#synthetic_data_status_df table th:nth-child({idx}), #synthetic_data_status_df table td:nth-child({idx}) {{ width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; }}")
                                            style_value = "<style>" + "\n".join(rules) + "</style>"
                                    return gr.Markdown(visible=True, value="✅ ステータス更新完了"), df_component, gr.HTML(visible=bool(style_value), value=style_value)
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)

                    def _syn_display_result(table_name, limit_value):
                        try:
                            from utils.management_util import display_table_data
                            try:
                                lv = int(limit_value)
                            except Exception:
                                lv = 50
                            if lv < 0:
                                lv = 0
                            df = display_table_data(pool, table_name, lv)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                widths = []
                                cols = df.columns.tolist()
                                sample = df.head(5)
                                for col in cols:
                                    series = sample[col].astype(str) if not sample.empty else pd.Series([], dtype=str)
                                    row_max = series.map(len).max() if len(series) > 0 else 0
                                    length = max(len(str(col)), row_max)
                                    widths.append(length)
                                total = sum(widths) if widths else 0
                                style_value = ""
                                if total > 0:
                                    col_widths = [max(5, int(100 * w / total)) for w in widths]
                                    diff = 100 - sum(col_widths)
                                    if diff != 0 and len(col_widths) > 0:
                                        col_widths[0] = max(5, col_widths[0] + diff)
                                    rules = []
                                    rules.append("#synthetic_data_result_df { width: 100% !important; }")
                                    rules.append("#synthetic_data_result_df .wrap { overflow-x: auto !important; }")
                                    rules.append("#synthetic_data_result_df table { table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; }")
                                    for idx, pct in enumerate(col_widths, start=1):
                                        rules.append(f"#synthetic_data_result_df table th:nth-child({idx}), #synthetic_data_result_df table td:nth-child({idx}) {{ width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; }}")
                                    style_value = "<style>" + "\n".join(rules) + "</style>"
                                return gr.Dataframe(visible=True, value=df, label=f"データ表示（件数: {len(df)}）", elem_id="synthetic_data_result_df"), gr.HTML(visible=bool(style_value), value=style_value)
                            else:
                                return gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示（件数: 0）", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")
                        except Exception:
                            return gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")

                    def _syn_display_result_stream(table_name, limit_value):
                        try:
                            t = str(table_name or "").strip()
                            if not t:
                                yield gr.Markdown(visible=True, value="⚠️ テーブルを選択してください"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")
                                return
                            try:
                                lv = int(limit_value)
                            except Exception:
                                lv = 50
                            if lv < 0:
                                lv = 0
                            yield gr.Markdown(visible=True, value="⏳ データ表示中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")
                            df_comp, style_comp = _syn_display_result(table_name, lv)
                            yield gr.Markdown(visible=True, value="✅ 表示完了"), df_comp, style_comp
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ 表示に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")

                    syn_refresh_btn.click(
                        fn=_syn_refresh_objects,
                        inputs=[syn_profile_select],
                        outputs=[syn_refresh_status, syn_tables_input, syn_result_table_select],
                    )

                    # プロファイル変更時にテーブル選択をリセット
                    syn_profile_select.change(
                        fn=lambda: gr.CheckboxGroup(value=[]),
                        inputs=[],
                        outputs=[syn_tables_input],
                    )

                    def _syn_generate_stream(profile_name, tables_selected, rows_per_table, extra_text, sample_rows, comments):
                        try:
                            missing = []
                            if not profile_name or not str(profile_name).strip():
                                missing.append("Profile")
                            if not tables_selected:
                                missing.append("テーブル選択")
                            if missing:
                                msg = "⚠️ 必須入力が不足しています: " + ", ".join(missing)
                                yield gr.Markdown(visible=True, value=msg), gr.Textbox(value="", autoscroll=False)
                                return
                            
                            # ジェネレーターからステータスを順次取得
                            op_id_value = None
                            gen = _syn_generate(profile_name, tables_selected, rows_per_table, extra_text, sample_rows, comments)
                            
                            # ジェネレータを順次処理し、return値も取得
                            while True:
                                try:
                                    item = next(gen)
                                    # itemが文字列ならオペレーションID、そうでなければgr.Markdown
                                    if isinstance(item, str):
                                        # オペレーションIDを取得
                                        op_id_value = item
                                        yield gr.Markdown(visible=True, value="✅ 合成データ生成を開始しました"), gr.Textbox(value=str(op_id_value), autoscroll=False)
                                    elif item is None:
                                        # エラー：IDが取得できなかった
                                        yield gr.Markdown(visible=True, value="❌ オペレーションIDの取得に失敗しました"), gr.Textbox(value="", autoscroll=False)
                                    else:
                                        # ステータス更新を出力
                                        yield item, gr.Textbox(value=str(op_id_value or ""), autoscroll=False)
                                except StopIteration as e:
                                    # ジェネレータの終了時、return値を取得
                                    returned_value = e.value
                                    if returned_value is not None and isinstance(returned_value, str) and returned_value.strip():
                                        op_id_value = returned_value
                                        yield gr.Markdown(visible=True, value="✅ 合成データ生成を開始しました"), gr.Textbox(value=str(op_id_value), autoscroll=False)
                                    elif op_id_value is None:
                                        # オペレーションIDが取得できなかった場合
                                        # 最後のyieldがエラーメッセージの場合はそれが表示されているはず
                                        pass
                                    break
                                
                        except Exception as e:
                            logger.error(f"_syn_generate_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 生成に失敗しました: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _syn_update_status_stream(op_id):
                        try:
                            oid = str(op_id or "").strip()
                            if not oid:
                                yield gr.Markdown(visible=True, value="⚠️ オペレーションIDを入力/取得してください"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ ステータス更新中..."), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                            info_md, df_comp, style_comp = _syn_update_status(op_id)
                            yield info_md, df_comp, style_comp
                        except Exception as e:
                            logger.error(f"_syn_update_status_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)

                    syn_generate_btn.click(
                        fn=_syn_generate_stream,
                        inputs=[syn_profile_select, syn_tables_input, syn_rows_per_table, syn_prompt_input, syn_sample_rows, syn_comments],
                        outputs=[syn_generate_status_md, syn_operation_id_text],
                    ).then(
                        fn=_syn_update_status_stream,
                        inputs=[syn_operation_id_text],
                        outputs=[syn_status_update_status_md, syn_status_df, syn_status_style],
                    )

                    syn_status_update_btn.click(
                        fn=_syn_update_status_stream,
                        inputs=[syn_operation_id_text],
                        outputs=[syn_status_update_status_md, syn_status_df, syn_status_style],
                    )

                    syn_result_btn.click(
                        fn=_syn_display_result_stream,
                        inputs=[syn_result_table_select, syn_result_limit],
                        outputs=[syn_result_status_md, syn_result_df, syn_result_style],
                    )

                with gr.TabItem(label="SQL分析→質問 逆生成") as reverse_tab:
                    with gr.Accordion(label="1. 入力", open=True):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("対象SQL*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                rev_sql_input = gr.Textbox(show_label=False, lines=8, max_lines=15, show_copy_button=True, container=False, autoscroll=False)

                    with gr.Accordion(label="SQL構造分析", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("モデル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        rev_analysis_model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "xai.grok-code-fast-1",
                                                "xai.grok-3",
                                                "xai.grok-3-fast",
                                                "xai.grok-4",
                                                "xai.grok-4-fast-non-reasoning",
                                                "meta.llama-4-scout-17b-16e-instruct",
                                                "gpt-4o",
                                                "gpt-5.1",
                                            ],
                                            value="xai.grok-code-fast-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        rev_analysis_btn = gr.Button("AI分析（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            rev_analysis_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("SQL構造分析", elem_classes="input-label")
                            with gr.Column(scale=5):
                                rev_sql_structure_output = gr.Textbox(label=" ", show_label=True, lines=15, max_lines=20, interactive=True, show_copy_button=True, container=True, autoscroll=False)

                    with gr.Accordion(label="2. 参照コンテキスト", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                # プロフィール選択肢を取得し、空の場合は空文字列を含むリストを設定
                                _rev_initial_choices = _load_profiles_from_json()
                                if not _rev_initial_choices:
                                    _rev_initial_choices = [("", "")]
                                rev_profile_select = gr.Dropdown(
                                    show_label=False,
                                    choices=_rev_initial_choices,
                                    value=(
                                        _rev_initial_choices[0][1]
                                        if (_rev_initial_choices and isinstance(_rev_initial_choices[0], tuple))
                                        else (_rev_initial_choices[0] if _rev_initial_choices else "")
                                    ),
                                    interactive=True,
                                    container=False
                                )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        rev_context_meta_btn = gr.Button("メタ情報を取得（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            rev_context_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("送信するメタ情報*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                rev_context_text = gr.Textbox(show_label=False, lines=15, max_lines=15, interactive=True, show_copy_button=True, autoscroll=False, container=True)

                    with gr.Accordion(label="3. 生成", open=True):
                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("モデル*", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        rev_model_input = gr.Dropdown(
                                            show_label=False,
                                            choices=[
                                                "xai.grok-code-fast-1",
                                                "xai.grok-3",
                                                "xai.grok-3-fast",
                                                "xai.grok-4",
                                                "xai.grok-4-fast-non-reasoning",
                                                "meta.llama-4-scout-17b-16e-instruct",
                                                "gpt-4o",
                                                "gpt-5.1",
                                            ],
                                            value="xai.grok-code-fast-1",
                                            interactive=True,
                                            container=False,
                                        )
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("用語集を利用", elem_classes="input-label")
                            with gr.Column(scale=5):
                                rev_use_glossary = gr.Checkbox(label="", value=False, container=False)
                        with gr.Row():
                            rev_generate_btn = gr.Button("自然言語を生成", variant="primary")
                        with gr.Row():
                            rev_generate_status_md = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("推奨質問(日本語)", elem_classes="input-label")
                            with gr.Column(scale=5):
                                rev_question_output = gr.Textbox(
                                    label=" ",
                                    show_label=True,
                                    lines=4,
                                    max_lines=10,
                                    interactive=False,
                                    show_copy_button=True,
                                    container=True,
                                    autoscroll=False,
                                )

                    def _rev_build_context_text(profile_name):
                        try:
                            s = _get_profile_context_ddl_from_json(profile_name)
                            if str(s).strip():
                                return s
                            tables, views = _get_profile_objects_from_json(profile_name)
                            chunks = []
                            for name in tables:
                                if not name:
                                    continue
                                try:
                                    _, ddl_t = get_table_details(pool, name)
                                except Exception:
                                    ddl_t = ""
                                if ddl_t:
                                    chunks.append(str(ddl_t).strip())
                            for name in views:
                                if not name:
                                    continue
                                try:
                                    _, ddl_v = get_view_details(pool, name)
                                except Exception:
                                    ddl_v = ""
                                if ddl_v:
                                    chunks.append(str(ddl_v).strip())
                            return "\n\n".join([c for c in chunks if c]) or ""
                        except Exception as e:
                            logger.error(f"_rev_build_context error: {e}")
                            return f"❌ エラー: {e}"

                    def _rev_build_context(profile_name):
                        try:
                            txt = _rev_build_context_text(profile_name)
                            return gr.Textbox(value=txt, autoscroll=False)
                        except Exception as e:
                            return gr.Textbox(value=f"❌ エラー: {e}", autoscroll=False)

                    def _on_profile_change_set_context_stream(p):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ メタ情報取得中..."), gr.Textbox(value="", interactive=True, autoscroll=False)
                            txt = _rev_build_context_text(p)
                            status_text = "✅ 取得完了" if str(txt).strip() else "✅ 取得完了（メタ情報なし）"
                            yield gr.Markdown(visible=True, value=status_text), gr.Textbox(value=txt, interactive=True, autoscroll=False)
                        except Exception as e:
                            logger.error(f"_on_profile_change_set_context_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Textbox(value="", interactive=True, autoscroll=False)

                    async def _rev_generate_async(model_name, sql_structure_text, context_text, sql_text, use_glossary):
                        """SQL→質問逆生成処理.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            sql_structure_text: SQL構造分析結果
                            context_text: スキーマやDDLのコンテキスト
                            sql_text: 対象SQL
                            use_glossary: 用語集を利用するか
                        
                        Returns:
                            gr.Textbox: 生成された質問文
                        """
                        try:
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています", autoscroll=False)
                            ctx_comp = str(context_text or "")
                            
                            # コメントを除去
                            s = remove_comments(str(sql_text or "").strip())
                            
                            # SQL構造分析の情報を利用
                            sql_structure = str(sql_structure_text or "").strip()
                            
                            prompt = (
                                "Convert the SQL structure analysis from physical names to business terms (COMMENT).\n"
                                "GOAL: Output must contain 100% of SQL information using business terms to enable exact SQL reconstruction.\n"
                                "Output ONLY the markdown text below (no code blocks, no explanations):\n\n"
                                
                                "## 📋 SQL論理構造 (業務用語版)\n\n"
                                
                                "### 📋 SELECT句\n"
                                "- [DISTINCT] (if present)\n"
                                "- [テーブル業務用語](別名).[列業務用語] [AS alias]\n"
                                "- aggregate_function([テーブル業務用語](別名).[列業務用語]) [AS alias]\n"
                                "- CASE WHEN [condition] THEN [result1] ELSE [result2] END [AS alias]\n"
                                "- expression [AS alias] (preserve function structure exactly)\n"
                                "- (サブクエリ-N) AS alias\n"
                                "- * (if SELECT *)\n\n"
                                
                                "### 📁 FROM句\n"
                                "- [テーブル業務用語] [AS alias]\n"
                                "- [結合方式]: EXPLICIT_JOIN (JOIN...ON) / IMPLICIT_JOIN (FROM t1, t2 WHERE)\n"
                                "- (サブクエリ-N) AS alias (if inline view)\n\n"
                                
                                "### 🔗 JOIN句\n"
                                "- **[JOIN_TYPE]**: [テーブルA業務用語](aliasA) JOIN [テーブルB業務用語](aliasB)\n"
                                "  - ON: [テーブルA業務用語](aliasA).[列A業務用語] = [テーブルB業務用語](aliasB).[列B業務用語]\n"
                                "  - ON: condition2 (if multiple conditions)\n"
                                "  - USING: ([列業務用語]) (if USING clause)\n"
                                "- **IMPLICIT**: [テーブルA業務用語](aliasA), [テーブルB業務用語](aliasB) - condition in WHERE\n\n"
                                
                                "### 🔍 WHERE句\n"
                                "- [テーブル業務用語](alias).[列業務用語] operator value\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] operator value\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] IN (サブクエリ-N)\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] = (サブクエリ-N)\n"
                                "- AND/OR EXISTS (サブクエリ-N)\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] BETWEEN value1 AND value2\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] LIKE 'pattern'\n"
                                "- AND/OR [テーブル業務用語](alias).[列業務用語] IS [NOT] NULL\n"
                                "- Complex expressions: preserve function structure exactly\n\n"
                                
                                "### 📦 GROUP BY句\n"
                                "- [テーブル業務用語](alias).[列業務用語1]\n"
                                "- [テーブル業務用語](alias).[列業務用語2]\n\n"
                                
                                "### 🎯 HAVING句\n"
                                "- aggregate_function([テーブル業務用語](alias).[列業務用語]) operator value\n"
                                "- AND/OR aggregate_function([列業務用語]) operator (サブクエリ-N)\n\n"
                                
                                "### 📊 ORDER BY句\n"
                                "- [テーブル業務用語](alias).[列業務用語1] ASC/DESC [NULLS FIRST/LAST]\n"
                                "- [テーブル業務用語](alias).[列業務用語2] ASC/DESC\n\n"
                                
                                "### 📏 LIMIT/OFFSET句\n"
                                "- LIMIT: n / FETCH FIRST n ROWS ONLY\n"
                                "- OFFSET: m / OFFSET m ROWS\n\n"
                                
                                "### 📝 WITH句(CTE)\n"
                                "- **cte_name**:\n"
                                "  - SELECT: [DISTINCT] [列業務用語1], [列業務用語2], aggregate_func([列業務用語]) AS alias, (サブクエリ-N) AS alias\n"
                                "  - FROM: [テーブル業務用語](alias)\n"
                                "  - JOIN: **[JOIN_TYPE]** [テーブル業務用語](alias) ON condition\n"
                                "  - WHERE: condition1 AND/OR condition2\n"
                                "  - GROUP BY: [列業務用語1], [列業務用語2]\n"
                                "  - HAVING: aggregate_condition\n"
                                "  - ORDER BY: [列業務用語] ASC/DESC\n\n"
                                
                                "### 🔎 サブクエリ\n"
                                "- **サブクエリ-N** [Location: SELECT/FROM/WHERE/HAVING in main/CTE]:\n"
                                "  - SELECT: [DISTINCT] [columns/expressions using 業務用語]\n"
                                "  - FROM: [テーブル業務用語](alias)\n"
                                "  - JOIN: **[JOIN_TYPE]** [テーブル業務用語](alias) ON condition\n"
                                "  - WHERE: conditions (use 業務用語)\n"
                                "  - Correlation: [内部テーブル業務用語](alias).[列業務用語] = [外部テーブル業務用語](alias).[列業務用語]\n"
                                "  - **NESTED-N-M**: (nested subquery with same structure)\n\n"
                                
                                "### 🔀 SET演算\n"
                                "- **[UNION/UNION ALL/INTERSECT/MINUS/EXCEPT]**:\n"
                                "  - Query1: (expand structure)\n"
                                "  - Query2: (expand structure)\n\n"
                                
                                "---\n\n"
                                
                                "CRITICAL RULES for 100% SQL Reconstruction:\n"
                                "- Replace physical table/column names with business terms from COMMENT\n"
                                "- Preserve alias exactly as in original SQL (e.g., ZHR, ZER, ADR)\n"
                                "- Format: [テーブル業務用語](alias).[列業務用語]\n"
                                "- MUST preserve ALL literal values exactly: strings with quotes, numbers, date literals\n"
                                "- MUST preserve ALL operators exactly: =, >, <, >=, <=, <>, !=, LIKE, IN, BETWEEN, IS NULL\n"
                                "- MUST preserve function calls with EXACT parameters: SUBSTR(col,4,6) ≠ SUBSTR(col,1,6)\n"
                                "- MUST preserve CASE expression structure completely with WHEN/THEN/ELSE\n"
                                "- MUST preserve nested function structure: TO_NUMBER(TO_CHAR(TO_DATE(...)))\n"
                                "- MUST preserve AND/OR/NOT logical structure exactly\n"
                                "- MUST distinguish EXPLICIT JOIN (JOIN...ON) vs IMPLICIT JOIN (FROM t1, t2 WHERE)\n"
                                "- For IMPLICIT JOIN: list tables with comma in FROM, show join condition in WHERE\n"
                                "- サブクエリ: Number sequentially (サブクエリ-1, サブクエリ-2...) and expand completely\n"
                                "- For correlated subqueries: show Correlation with inner.col = outer.col\n"
                                "- For nested subqueries: label as NESTED-X-Y and expand\n"
                                "- If section is empty/not present, omit that section entirely\n\n"
                                
                                "Example:\n"
                                "SQL: SELECT ZHR.EMPLID, CASE WHEN TO_NUMBER(TO_CHAR(TO_DATE(substr(ZER.CAL_ID,4,6)||'01','YYYYMMDD'),'MM'))>=4 THEN TO_NUMBER(TO_CHAR(TO_DATE(substr(ZER.CAL_ID,4,6)||'01','YYYYMMDD'),'YYYY')) ELSE TO_NUMBER(TO_CHAR(TO_DATE(substr(ZER.CAL_ID,4,6)||'01','YYYYMMDD'),'YYYY'))-1 END AS FiscalYear FROM PS_Z_IF_HRBASE_VW ZHR, PS_Z_GP_WA_SAL_ER ZER WHERE ZHR.EMPLID=ZER.EMPLID AND ZHR.EFFDT=(SELECT MAX(ZHR1.EFFDT) FROM PS_Z_IF_HRBASE_VW ZHR1 WHERE ZHR1.EMPLID=ZHR.EMPLID AND ZHR1.EFFDT<=SYSDATE)\n\n"
                                
                                "Output:\n"
                                "## 📋 SQL論理構造 (業務用語版)\n\n"
                                
                                "### 📋 SELECT句\n"
                                "- [人事基本](ZHR).[社員番号]\n"
                                "- CASE WHEN TO_NUMBER(TO_CHAR(TO_DATE(SUBSTR([給与計算結果(支給)](ZER).[カレンダーID], 4, 6) || '01', 'YYYYMMDD'), 'MM')) >= 4 THEN TO_NUMBER(TO_CHAR(TO_DATE(SUBSTR([給与計算結果(支給)](ZER).[カレンダーID], 4, 6) || '01', 'YYYYMMDD'), 'YYYY')) ELSE TO_NUMBER(TO_CHAR(TO_DATE(SUBSTR([給与計算結果(支給)](ZER).[カレンダーID], 4, 6) || '01', 'YYYYMMDD'), 'YYYY')) - 1 END AS FiscalYear\n\n"
                                
                                "### 📁 FROM句\n"
                                "- [人事基本] AS ZHR\n"
                                "- [給与計算結果(支給)] AS ZER\n"
                                "- [結合方式]: IMPLICIT_JOIN\n\n"
                                
                                "### 🔍 WHERE句\n"
                                "- [人事基本](ZHR).[社員番号] = [給与計算結果(支給)](ZER).[社員番号]\n"
                                "- AND [人事基本](ZHR).[有効日] = (サブクエリ-1)\n\n"
                                
                                "### 🔎 サブクエリ\n"
                                "- **サブクエリ-1** [Location: WHERE in main query]:\n"
                                "  - SELECT: MAX([人事基本](ZHR1).[有効日])\n"
                                "  - FROM: [人事基本](ZHR1)\n"
                                "  - WHERE:\n"
                                "    - [人事基本](ZHR1).[社員番号] = [人事基本](ZHR).[社員番号]\n"
                                "    - AND [人事基本](ZHR1).[有効日] <= SYSDATE\n"
                                "  - Correlation: [人事基本](ZHR1).[社員番号] = [人事基本](ZHR).[社員番号]\n\n"
                                
                                "===SQL構造分析===\n" + (sql_structure if sql_structure else "(未分析)") + "\n\n"
                                "===データベース定義===\n" + str(ctx_comp or "") + "\n\n"
                                "===対象SQL===\n```sql\n" + s + "\n```"
                            )
                            
                            if str(model_name).startswith("gpt-"):
                                from openai import AsyncOpenAI
                                client = AsyncOpenAI()
                            else:
                                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                                client = AsyncOciOpenAI(
                                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                    auth=OciUserPrincipalAuth(),
                                    compartment_id=compartment_id,
                                )
                            
                            messages = [
                                {
                                    "role": "system", 
                                    "content": "You are a Text-to-SQL reverse engineer. Generate STRUCTURED natural language specifications that preserve 100% of SQL semantics for exact reconstruction. Output the specified format ONLY."
                                },
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages, temperature=0.0)
                            out_text = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                out_text = msg.content if hasattr(msg, "content") else ""
                            import re as _re
                            out_text = _re.sub(r"^```.*?\n|\n```$", "", str(out_text or ""), flags=_re.DOTALL).strip()
                            
                            # 用語集を利用する場合は逆処理を適用
                            if use_glossary:
                                terms = _load_terminology()
                                if terms:
                                    # 用語集を使ってLLMで書き換え（逆処理）
                                    terms_text = "\n".join([f"- {k}: {v}" for k, v in terms.items()])
                                    glossary_prompt = f"""あなたはデータベースクエリの専門家です。以下の用語集は通常「A（TERM）→B（定義・推奨表現）」の最適化指針です。本タスクでは逆最適化を行い、元の質問に含まれるB側の表現をA側の正式用語（TERM）へ正規化してください。

用語集:
{terms_text}

元の質問:
{out_text}

指示:
1. 定義や推奨表現、別名、略称などB側に該当する語句は対応する正式用語（A/TERM）に置換してください。
2. 意図・条件・対象は維持し、語彙のみを正規化してください。
3. 数値・日付・範囲などの具体値は変更しないでください。
4. 出力は正規化後の質問文のみ。説明や前置きは不要です。

正規化後の質問:"""
                                    
                                    messages = [{"role": "user", "content": glossary_prompt}]
                                    glossary_resp = await client.chat.completions.create(model=model_name, messages=messages)
                                    if glossary_resp.choices and len(glossary_resp.choices) > 0:
                                        glossary_result = glossary_resp.choices[0].message.content.strip()
                                        # 元の質問と用語集適用後の質問を\n\nで連結
                                        out_text = str(out_text) + "\n\n" + glossary_result
                            
                            return gr.Textbox(value=out_text, autoscroll=False)
                        except Exception as e:
                            logger.error(f"_rev_generate_async error: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            return gr.Textbox(value=f"❌ エラー: {e}", autoscroll=False)

                    def _rev_generate(model_name, sql_structure_text, context_text, sql_text, use_glossary):
                        """SQL→質問逆生成のラッパー関数.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            sql_structure_text: SQL構造分析結果
                            context_text: スキーマやDDLのコンテキスト
                            sql_text: 対象SQL
                            use_glossary: 用語集を利用するか
                        
                        Returns:
                            gr.Textbox: 生成された質問文
                        """
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(_rev_generate_async(model_name, sql_structure_text, context_text, sql_text, use_glossary))
                        finally:
                            loop.close()

                    def _rev_generate_stream(model_name, sql_structure_text, context_text, sql_text, use_glossary):
                        try:
                            ctx = str(context_text or "").strip()
                            sql = str(sql_text or "").strip()
                            missing = []
                            if not ctx:
                                missing.append("送信するメタ情報")
                            if not sql:
                                missing.append("対象SQL")
                            if missing:
                                msg = "⚠️ 必須入力が不足しています: " + ", ".join(missing)
                                yield gr.Markdown(visible=True, value=msg), gr.Textbox(value="", interactive=False, autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ 生成中..."), gr.Textbox(value="", interactive=False, autoscroll=False)
                            out = _rev_generate(model_name, sql_structure_text, context_text, sql_text, use_glossary)
                            yield gr.Markdown(visible=True, value="✅ 生成完了"), out
                        except Exception as e:
                            logger.error(f"_rev_generate_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 生成に失敗しました: {e}"), gr.Textbox(value="", interactive=False, autoscroll=False)

                    async def _rev_ai_analyze_async(model_name, sql_text):
                        """逆生成タブ用のAI分析処理.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            sql_text: 対象SQL
                        
                        Returns:
                            tuple: (status_md, structure_output)
                        """
                        try:
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Markdown(visible=True, value="⚠️ OCI設定が不足しています"), gr.Textbox(value="", autoscroll=False)
                            
                            s = str(sql_text or "").strip()
                            if not s:
                                return gr.Markdown(visible=True, value="⚠️ SQLが空です"), gr.Textbox(value="", autoscroll=False)
                            
                            # グローバルプロンプトを使用
                            prompt = _SQL_STRUCTURE_ANALYSIS_PROMPT + "SQL:\n```sql\n" + s + "\n```"
                            
                            if str(model_name).startswith("gpt-"):
                                from openai import AsyncOpenAI
                                client = AsyncOpenAI()
                            else:
                                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                                client = AsyncOciOpenAI(
                                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                    auth=OciUserPrincipalAuth(),
                                    compartment_id=compartment_id,
                                )
                            
                            messages = [
                                {
                                    "role": "system", 
                                    "content": "You are a SQL parser. Output ONLY the requested format. No explanations."
                                },
                                {
                                    "role": "user", 
                                    "content": prompt
                                },
                            ]
                            
                            resp = await client.chat.completions.create(model=model_name, messages=messages)
                            sql_structure_md = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                out = msg.content if hasattr(msg, "content") else ""
                                sql_structure_md = str(out or "").strip()
                                # マークダウンコードブロックを削除
                                sql_structure_md = re.sub(r"```+markdown\s*", "", sql_structure_md)
                                sql_structure_md = re.sub(r"```+\s*$", "", sql_structure_md)
                                sql_structure_md = sql_structure_md.strip()
                            
                            if not sql_structure_md:
                                sql_structure_md = "## 📊 SQL構造分析\n\n情報を抽出できませんでした。"
                            
                            return gr.Markdown(visible=True, value="✅ AI分析完了"), gr.Textbox(value=sql_structure_md, autoscroll=False)
                        except Exception as e:
                            logger.error(f"_rev_ai_analyze_async error: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _rev_ai_analyze(model_name, sql_text):
                        """逆生成タブ用のAI分析ラッパー関数.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            sql_text: 対象SQL
                        
                        Returns:
                            tuple: (status_md, structure_output)
                        """
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(_rev_ai_analyze_async(model_name, sql_text))
                        finally:
                            loop.close()

                    def _rev_ai_analyze_stream(model_name, sql_text):
                        """逆生成タブ用のAI分析ストリーム関数.
                        
                        Args:
                            model_name: 使用するLLMモデル
                            sql_text: 対象SQL
                        
                        Yields:
                            tuple: (status_md, structure_output)
                        """
                        try:
                            if not model_name or not str(model_name).strip():
                                yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="", autoscroll=False)
                                return
                            if not sql_text or not str(sql_text).strip():
                                yield gr.Markdown(visible=True, value="⚠️ SQLが空です。先にSQLを入力してください"), gr.Textbox(value="", autoscroll=False)
                                return
                            yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Textbox(value="## 📊 SQL構造分析\n\n解析中...", autoscroll=False)
                            result = _rev_ai_analyze(model_name, sql_text)
                            yield result
                        except Exception as e:
                            logger.error(f"_rev_ai_analyze_stream error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ 分析に失敗しました: {e}"), gr.Textbox(value="", autoscroll=False)

                    def _on_profile_change_set_context(p):
                        return _rev_build_context(p)

                    rev_analysis_btn.click(
                        fn=_rev_ai_analyze_stream,
                        inputs=[rev_analysis_model_input, rev_sql_input],
                        outputs=[rev_analysis_status_md, rev_sql_structure_output],
                    )

                    rev_context_meta_btn.click(
                        fn=_on_profile_change_set_context_stream,
                        inputs=[rev_profile_select],
                        outputs=[rev_context_status_md, rev_context_text],
                    )

                    rev_generate_btn.click(
                        fn=_rev_generate_stream,
                        inputs=[rev_model_input, rev_sql_structure_output, rev_context_text, rev_sql_input, rev_use_glossary],
                        outputs=[rev_generate_status_md, rev_question_output],
                    )

        with gr.TabItem(label="ユーザー機能"):
            with gr.Tabs():
                with gr.TabItem(label="基本機能") as user_basic_tab:
                    with gr.Accordion(label="1. チャット", open=True):
                        def _profile_names():
                            try:
                                pairs = _load_profiles_from_json()
                                return [(str(bd), str(pf)) for bd, pf in pairs]
                            except Exception as e:
                                logger.error(f"_profile_names error: {e}")
                            return [("", "")]

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("自然言語の質問*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                prompt_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="例: 大阪の顧客数を教えて",
                                    lines=3,
                                    max_lines=10,
                                    show_copy_button=True,
                                    container=False,
                                )

                        with gr.Row():
                            with gr.Column(scale=5):
                                user_predict_domain_btn = gr.Button("カテゴリ予測 ⇒", variant="secondary")
                            with gr.Column(scale=5):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        # プロフィール選択肢を取得し、空の場合は空文字列を含むリストを設定
                                        _initial_choices = _profile_names()
                                        if not _initial_choices:
                                            _initial_choices = [("", "")]
                                        profile_select = gr.Dropdown(
                                            show_label=False,
                                            choices=_initial_choices,
                                            value=_initial_choices[0][1] if _initial_choices and isinstance(_initial_choices[0], tuple) else "",
                                            interactive=True,
                                            container=False,
                                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("クエリ書き換えを有効化", elem_classes="input-label")
                            with gr.Column(scale=5):
                                enable_query_rewrite = gr.Checkbox(label="", value=False, container=False)
                        
                        with gr.Row():
                            with gr.Accordion(label="", open=True, visible=False) as query_rewrite_section:
                                with gr.Row():
                                    with gr.Column(scale=5):
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                gr.Markdown("書き換え用モデル*", elem_classes="input-label")
                                            with gr.Column(scale=5):
                                                rewrite_model_select = gr.Dropdown(
                                                    show_label=False,
                                                    choices=[
                                                        "xai.grok-code-fast-1",
                                                        "xai.grok-3",
                                                        "xai.grok-3-fast",
                                                        "xai.grok-4",
                                                        "xai.grok-4-fast-non-reasoning",
                                                        "meta.llama-4-scout-17b-16e-instruct",
                                                        "gpt-4o",
                                                        "gpt-5.1",
                                                    ],
                                                    value="xai.grok-code-fast-1",
                                                    interactive=True,
                                                    container=False,
                                                )
                                    with gr.Column(scale=5):
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                gr.Markdown("")
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("ステップ1: 用語集を利用", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        rewrite_use_glossary = gr.Checkbox(label="", value=True, container=False)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("ステップ2: スキーマ情報を利用", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        rewrite_use_schema = gr.Checkbox(label="", value=False, container=False)
                                with gr.Row():
                                    rewrite_btn = gr.Button("書き換え実行", variant="primary")
                                with gr.Row():
                                    rewrite_status = gr.Markdown(visible=False)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("書き換え後の質問", elem_classes="input-label")
                                    with gr.Column(scale=5):
                                        rewritten_query = gr.Textbox(
                                            show_label=False,
                                            lines=5,
                                            max_lines=10,
                                            interactive=True,
                                            show_copy_button=True,
                                            container=False,
                                            autoscroll=False,
                                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("追加指示・例示を使用", elem_classes="input-label")
                            with gr.Column(scale=5):
                                include_extra_prompt = gr.Checkbox(label="", value=False, container=False)

                        with gr.Row():
                            with gr.Accordion(label="追加指示・例示を設定", open=True, visible=False) as extra_prompt_section:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        extra_prompt = gr.Textbox(
                                            show_label=False,
                                            value=(
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                                "⚠️ CRITICAL INSTRUCTION (NOT PART OF QUESTION):\n"
                                                "\n"
                                                "The examples above demonstrate the EXACT patterns you MUST follow.\n"
                                                "MANDATORY REQUIREMENTS:\n"
                                                "- Study the example queries - they define the REQUIRED style and format\n"
                                                "- Follow the SAME patterns for similar questions\n"
                                                "- Adopt ALL functions shown in examples (even if unrecognized/new)\n"
                                                "- When questions are similar to examples, mirror the example's structure precisely\n"
                                                "\n"
                                                "Examples OVERRIDE all other instructions.\n"
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                                "\n"
                                                "▼ ACTUAL QUESTION:\n"
                                            ),
                                            lines=15,
                                            max_lines=15,
                                            show_copy_button=True,
                                            autoscroll=False,
                                            container=False,
                                        )
                            include_extra_prompt.change(lambda v: gr.Accordion(visible=v), inputs=include_extra_prompt, outputs=extra_prompt_section)
                        
                        # Query転写のCheckbox変更ハンドラ
                        enable_query_rewrite.change(lambda v: gr.Accordion(visible=v), inputs=enable_query_rewrite, outputs=query_rewrite_section)

                        with gr.Row():
                            with gr.Column():
                                chat_clear_btn = gr.Button("クリア", variant="secondary")
                            with gr.Column():
                                chat_execute_btn = gr.Button("実行（時間がかかる場合があります）", variant="primary")
                        with gr.Row():
                            chat_status_md = gr.Markdown(visible=False)

                    with gr.Accordion(label="2. 生成SQL", open=True):
                        gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("生成されたSQL", elem_classes="input-label")
                            with gr.Column(scale=5):
                                generated_sql_text = gr.Textbox(
                                    show_label=False,
                                    lines=8,
                                    max_lines=15,
                                    interactive=False,
                                    show_copy_button=True,
                                    container=False,
                                    autoscroll=False,
                                )

                    with gr.Accordion(label="3. 実行結果", open=True):
                        chat_result_df = gr.Dataframe(
                            label="実行結果",
                            interactive=False,
                            wrap=True,
                            visible=False,
                            value=pd.DataFrame(),
                            elem_id="selectai_chat_result_df",
                        )
                        chat_result_style = gr.HTML(visible=False)

                build_sql_learning_tab(pool)

            def _user_step_generate(profile, prompt, extra_prompt, include_extra, enable_rewrite, rewritten_query):
                yield from _common_step_generate(profile, prompt, extra_prompt, include_extra, enable_rewrite, rewritten_query)

            def _user_step_run_sql(sql_text, status_text=None):
                if status_text and "❌" in str(status_text):
                    return
                yield from _run_sql_common(sql_text, "selectai_chat_result_df")

            def _on_chat_clear():
                ch = _profile_names() or [("", "")]
                return "", gr.Dropdown(choices=ch, value=ch[0][1]), gr.Textbox(value="", autoscroll=False)
            
            def _user_rewrite_query(model_name, profile_name, original_query, use_glossary, use_schema):
                """ユーザー向けクエリ書き換え処理.
                
                Args:
                    model_name: 使用するLLMモデル
                    profile_name: Profile名
                    original_query: 元の自然言語の質問
                    use_schema: 第2ステップを実行するか
                
                Yields:
                    tuple: (status_md, rewritten_text)
                """
                # 開発者機能と同じロジックを使用
                yield from _dev_rewrite_query(model_name, profile_name, original_query, use_glossary, use_schema)

            def _user_predict_domain_and_set_profile(text):
                try:
                    ch = _load_profiles_from_json() or [("", "")]
                    pdomain = _predict_category_label(text)
                    return _map_domain_to_profile(pdomain, ch)
                except Exception:
                    ch = _profile_names() or [("", "")]
                    return gr.Dropdown(choices=ch, value=ch[0][1])

            chat_execute_btn.click(
                fn=_user_step_generate,
                inputs=[profile_select, prompt_input, extra_prompt, include_extra_prompt, enable_query_rewrite, rewritten_query],
                outputs=[chat_status_md, generated_sql_text],
            ).then(
                fn=_user_step_run_sql,
                inputs=[generated_sql_text, chat_status_md],
                outputs=[chat_status_md, chat_result_df, chat_result_style],
            )

            chat_clear_btn.click(
                fn=_on_chat_clear,
                outputs=[prompt_input, profile_select, generated_sql_text],
            )

            user_predict_domain_btn.click(
                fn=_user_predict_domain_and_set_profile,
                inputs=[prompt_input],
                outputs=[profile_select],
            )
            
            # Query転写ボタンのイベントハンドラ
            rewrite_btn.click(
                fn=_user_rewrite_query,
                inputs=[rewrite_model_select, profile_select, prompt_input, rewrite_use_glossary, rewrite_use_schema],
                outputs=[rewrite_status, rewritten_query],
            )

        # 各タブ選択時のProfileドロップダウン更新イベントハンドラー
        def _update_dropdown_from_json(current_value):
            choices = _load_profiles_from_json() or [("", "")]
            if choices and isinstance(choices[0], tuple):
                values = [c[1] for c in choices]
                if current_value in values:
                    return gr.Dropdown(choices=choices, value=current_value)
                return gr.Dropdown(choices=choices, value=values[0])
            if not choices:
                choices = [""]
            if current_value in choices:
                return gr.Dropdown(choices=choices, value=current_value)
            return gr.Dropdown(choices=choices, value=choices[0])

        # チャット・分析タブ
        dev_chat_tab.select(
            fn=_update_dropdown_from_json,
            inputs=[dev_profile_select],
            outputs=[dev_profile_select],
        )

        # フィードバック管理タブ
        feedback_tab.select(
            fn=_update_dropdown_from_json,
            inputs=[global_profile_select],
            outputs=[global_profile_select],
        )

        # 合成データ生成タブ
        synthetic_tab.select(
            fn=_update_dropdown_from_json,
            inputs=[syn_profile_select],
            outputs=[syn_profile_select],
        )

        # SQL→質問 逆生成タブ
        reverse_tab.select(
            fn=_update_dropdown_from_json,
            inputs=[rev_profile_select],
            outputs=[rev_profile_select],
        )

        # ユーザー機能 → 基本機能タブ
        user_basic_tab.select(
            fn=_update_dropdown_from_json,
            inputs=[profile_select],
            outputs=[profile_select],
        )
