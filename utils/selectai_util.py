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
import oracledb
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails

from utils.management_util import (
    get_table_list,
    get_view_list,
    get_table_details,
    get_view_details,
)

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
        profiles_data = [
            {
                "profile": "",
                "business_domain": ""
            }
        ]
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, DESCRIPTION FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
                for r in rows:
                    try:
                        name = r[0]
                        if str(name).strip().upper() == "OCI_CRED$PROF":
                            continue
                        desc_val = r[1]
                        desc = desc_val.read() if hasattr(desc_val, "read") else str(desc_val or "")
                        profiles_data.append({
                            "profile": str(name),
                            "business_domain": str(desc)
                        })
                    except Exception as e:
                        logger.error(f"_save_profiles_to_json row error: {e}")
        
        # profiles/selectai.json に保存
        json_path = _profiles_dir() / "selectai.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(profiles_data)} profiles to {json_path}")
    except Exception as e:
        logger.error(f"_save_profiles_to_json error: {e}")


def _load_profiles_from_json():
    """保存されたselectai.jsonからプロファイル一覧を読み込む"""
    try:
        json_path = _profiles_dir() / "selectai.json"
        if not json_path.exists():
            return []
        with json_path.open("r", encoding="utf-8") as f:
            profiles_data = json.load(f)
        # business_domainを優先して返す
        result = []
        for p in profiles_data:
            bd = str(p.get("business_domain", "") or "").strip()
            if bd:
                result.append(bd)
            else:
                result.append(str(p.get("profile", "")))
        return result
    except Exception as e:
        logger.error(f"_load_profiles_from_json error: {e}")
        return []
        
def get_db_profiles(pool) -> pd.DataFrame:
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, DESCRIPTION, STATUS FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
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
                    df = pd.DataFrame(plain_rows, columns=["Profile Name", "Business Domain", "Status"]).sort_values("Profile Name")
                else:
                    df = pd.DataFrame(columns=["Profile Name", "Business Domain", "Status"]).sort_values("Profile Name")
                df = df[df["Profile Name"].astype(str).str.strip().str.upper() != "OCI_CRED$PROF"]

        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
        # business_domain_col is already populated from DESCRIPTION
        tables_col = []
        views_col = []
        regions_col = []
        models_col = []
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
        if len(df) > 0:
            df.insert(2, "Tables", tables_col)
            df.insert(3, "Views", views_col)
            df.insert(4, "Region", regions_col)
            df.insert(5, "Model", models_col)
        else:
            df = pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"])  
        return df
    except Exception as e:
        logger.error(f"get_db_profiles error: {e}")
        return pd.DataFrame(columns=["Profile Name", "Tables", "Views", "Region", "Model", "Status"]) 


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
            if "Business Domain" in df.columns:
                m = df[df["Business Domain"].astype(str) == s]
                if len(m) > 0:
                    return str(m.iloc[0]["Profile Name"]) if "Profile Name" in m.columns else str(m.iloc[0][0])
            if "Profile Name" in df.columns:
                m2 = df[df["Profile Name"].astype(str) == s]
                if len(m2) > 0:
                    return s
        return s
    except Exception:
        return str(display_name or "")


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
    tables: list,
    views: list,
    business_domain: str,
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
        "temperature": 0.0,
        "object_list": [],
    }

    for t in tables or []:
        attrs["object_list"].append({"owner": "ADMIN", "name": t})
    for v in views or []:
        attrs["object_list"].append({"owner": "ADMIN", "name": v})

    attr_str = json.dumps(attrs, ensure_ascii=False)

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
                desc=str(business_domain or ""),
            )
            logger.info(f"Created profile: {name}")


def build_selectai_tab(pool):
    with gr.Tabs():
        with gr.TabItem(label="開発者機能"):
            with gr.Tabs():
                with gr.TabItem(label="プロファイル管理"):
                    with gr.Accordion(label="1. プロファイル一覧", open=True):
                        profile_refresh_btn = gr.Button("プロファイル一覧を取得", variant="primary")
                        profile_refresh_status = gr.Markdown(visible=False)
                        profile_list_df = gr.Dataframe(
                            label="プロファイル一覧(行をクリックして詳細を表示)",
                            interactive=False,
                            wrap=True,
                            value=pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"]),
                            headers=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"],
                            visible=False,
                            elem_id="profile_list_df",
                        )
                        profile_list_style = gr.HTML(visible=False)

                    with gr.Accordion(label="2. プロファイル詳細・変更", open=True):
                        with gr.Row():
                            with gr.Column():
                                selected_profile_name = gr.Textbox(label="選択されたProfile名", interactive=True)
                            with gr.Column():
                                business_domain_text = gr.Textbox(label="業務ドメイン名", value="", interactive=True)
                        with gr.Row():
                            with gr.Column():
                                profile_json_text = gr.Textbox(
                                    label="Profile 作成SQL",
                                    lines=5,
                                    max_lines=10,
                                    show_copy_button=True,
                                )
                        selected_profile_original_name = gr.State("")
                        with gr.Row():
                            profile_update_btn = gr.Button("変更を保存", variant="primary")
                            profile_delete_btn = gr.Button("選択したProfileを削除", variant="stop")

                    with gr.Accordion(label="3. プロファイル作成", open=False):
                        with gr.Row():
                            profile_name = gr.Textbox(
                                label="Profile名",
                                value=f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            )
                            business_domain_input = gr.Textbox(label="業務ドメイン名", placeholder="例: 顧客管理、売上分析 等")

                        with gr.Row():
                            refresh_btn = gr.Button("テーブル・ビュー一覧を取得", variant="primary")

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択")
                                tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択")
                                views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)

                        with gr.Row():
                            compartment_id_input = gr.Textbox(label="OCI Compartment OCID", placeholder="ocid1.compartment.oc1...", value=os.environ.get("OCI_COMPARTMENT_OCID", ""))

                        with gr.Row():
                            region_input = gr.Dropdown(
                                label="Region",
                                choices=["ap-osaka-1", "us-chicago-1"],
                                value="us-chicago-1",
                                interactive=True,
                            )

                        with gr.Row():
                            with gr.Column():
                                embedding_model_input = gr.Dropdown(
                                    label="Embedding_Model",
                                    choices=[
                                        "cohere.embed-v4.0",
                                    ],
                                    value="cohere.embed-v4.0",
                                    interactive=True,
                                )

                            with gr.Column():
                                model_input = gr.Dropdown(
                                    label="Model",
                                    choices=[
                                        "xai.grok-code-fast-1",
                                        "xai.grok-3",
                                        "xai.grok-3-fast",
                                        "xai.grok-4",
                                        "xai.grok-4-fast-non-reasoning",
                                        "meta.llama-4-scout-17b-16e-instruct",
                                    ],
                                    value="xai.grok-code-fast-1",
                                    interactive=True,
                                )

                            with gr.Column():
                                max_tokens_input = gr.Slider(
                                    label="Max Tokens",
                                    minimum=1024,
                                    maximum=16384,
                                    step=1024,
                                    value=4096,
                                    interactive=True,
                                )

                        with gr.Row():
                            with gr.Column():
                                enforce_object_list_input = gr.Dropdown(
                                    label="Enforce_Object_List",
                                    choices=["true", "false"],
                                    value="true",
                                    interactive=True,
                                )

                            with gr.Column():
                                comments_input = gr.Dropdown(
                                    label="Comments",
                                    choices=["true", "false"],
                                    value="true",
                                    interactive=True,
                                )

                            with gr.Column():
                                annotations_input = gr.Dropdown(
                                    label="Annotations",
                                    choices=["true", "false"],
                                    value="true",
                                    interactive=True,
                                )

                        with gr.Row():
                            build_btn = gr.Button("作成", variant="primary")

                        create_info = gr.Markdown(visible=False)               

                def refresh_profiles():
                    try:
                        yield gr.Markdown(value="⏳ プロファイル一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"])), gr.HTML(visible=False)
                        df = get_db_profiles(pool)
                        # JSONファイルに保存
                        _save_profiles_to_json(pool)
                        if df is None or df.empty:
                            empty_df = pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"])
                            yield gr.Markdown(value="✅ 取得完了（データなし）", visible=True), gr.Dataframe(value=empty_df, visible=True), gr.HTML(visible=False)
                            return
                        sample = df.head(5)
                        widths = []
                        columns = max(1, len(df.columns))
                        for col in sample.columns:
                            series = sample[col].astype(str)
                            row_max = series.map(len).max() if len(series) > 0 else 0
                            length = max(len(str(col)), row_max)
                            widths.append(min(100 / columns, length))
                        total = sum(widths) if widths else 0
                        style_value = ""
                        if total > 0:
                            col_widths = [max(5, int(100 * w / total)) for w in widths]
                            diff = 100 - sum(col_widths)
                            if diff != 0 and len(col_widths) > 0:
                                col_widths[0] = max(5, col_widths[0] + diff)
                            rules = ["#profile_list_df table { table-layout: fixed; width: 100%; }"]
                            for idx, pct in enumerate(col_widths, start=1):
                                rules.append(f"#profile_list_df table th:nth-child({idx}), #profile_list_df table td:nth-child({idx}) {{ width: {pct}%; }}")
                            style_value = "<style>" + "\n".join(rules) + "</style>"
                        yield gr.Markdown(visible=False), gr.Dataframe(value=df, visible=True), gr.HTML(visible=bool(style_value), value=style_value)
                    except Exception as e:
                        logger.error(f"refresh_profiles error: {e}")
                        yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"])), gr.HTML(visible=False)
                
                def on_profile_select(evt: gr.SelectData, current_df, compartment_id):
                    try:
                        if isinstance(current_df, dict):
                            try:
                                current_df = pd.DataFrame.from_dict(current_df, orient='tight')
                            except Exception:
                                current_df = pd.DataFrame(current_df)
                        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        if len(current_df) > row_index:
                            name = str(current_df.iloc[row_index, 0])
                            attrs = _get_profile_attributes(pool, name) or {}
                            if compartment_id:
                                attrs.setdefault("oci_compartment_id", compartment_id)
                            desc = ""
                            try:
                                with pool.acquire() as conn2:
                                    with conn2.cursor() as cursor2:
                                        cursor2.execute("SELECT DESCRIPTION FROM USER_CLOUD_AI_PROFILES WHERE PROFILE_NAME = :name", name=name)
                                        r2 = cursor2.fetchone()
                                        if r2:
                                            v = r2[0]
                                            desc = v.read() if hasattr(v, "read") else str(v)
                            except Exception:
                                desc = ""
                            sql = _generate_create_sql_from_attrs(name, attrs, desc)
                            bdn = str(desc or "")
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
                        # JSONファイルを更新
                        _save_profiles_to_json(pool)
                        return gr.Markdown(visible=True, value=f"🗑️ 削除しました: {name}"), gr.Dataframe(value=get_db_profiles(pool)), "", "", ""
                    except Exception as e:
                        logger.error(f"delete_selected_profile error: {e}")
                        return gr.Markdown(visible=True, value=f"❌ 削除に失敗しました: {str(e)}"), gr.Dataframe(value=get_db_profiles(pool)), name, "", ""

                def update_selected_profile(original_name, edited_name, business_domain):
                    try:
                        orig = str(original_name or "").strip()
                        new = str(edited_name or "").strip()
                        bd = str(business_domain or "").strip()
                        if not orig:
                            attrs = {}
                            sql = _generate_create_sql_from_attrs(new or orig, attrs, bd)
                            return gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Dataframe(value=get_db_profiles(pool)), edited_name, gr.Textbox(value=bd), sql, (new or orig or "")
                        if not new:
                            new = orig
                        if not bd:
                            attrs = _get_profile_attributes(pool, orig) or {}
                            sql = _generate_create_sql_from_attrs(orig, attrs, "")
                            return gr.Markdown(visible=True, value="⚠️ 業務ドメイン名を入力してください"), gr.Dataframe(value=get_db_profiles(pool)), new, gr.Textbox(value=bd), sql, orig
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
                        # JSONファイルを更新
                        _save_profiles_to_json(pool)
                        sql = _generate_create_sql_from_attrs(new, attrs, bd)
                        return gr.Markdown(visible=True, value=f"✅ 更新しました: {new}"), gr.Dataframe(value=get_db_profiles(pool)), new, gr.Textbox(value=bd), sql, new
                    except Exception as e:
                        logger.error(f"update_selected_profile error: {e}")
                        attrs = _get_profile_attributes(pool, orig or edited_name) or {}
                        sql = _generate_create_sql_from_attrs(new or orig, attrs, bd)
                        return gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {str(e)}"), gr.Dataframe(value=get_db_profiles(pool)), edited_name, gr.Textbox(value=bd), sql, (new or orig or "")

                def refresh_sources():
                    return gr.CheckboxGroup(choices=_get_table_names(pool), visible=True), gr.CheckboxGroup(choices=_get_view_names(pool), visible=True)

                def build_profile(name, tables, views, compartment_id, region, model, embedding_model, max_tokens, enforce_object_list, comments, annotations, business_domain):
                    if not tables and not views:
                        yield gr.Markdown(visible=True, value="⚠️ テーブルまたはビューを選択してください"), gr.Dataframe(value=get_db_profiles(pool)), gr.Textbox(value=str(name or "")), gr.Textbox(value=str(business_domain or "")), gr.Textbox(value="")
                        return
                    bd = str(business_domain or "").strip()
                    if not bd:
                        yield gr.Markdown(visible=True, value="⚠️ 業務ドメイン名を入力してください"), gr.Dataframe(value=get_db_profiles(pool)), gr.Textbox(value=str(name or "")), gr.Textbox(value=str(business_domain or "")), gr.Textbox(value="")
                        return
                    try:
                        yield gr.Markdown(visible=True, value="⏳ 作成中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Profile Name", "Business Domain", "Tables", "Views", "Region", "Model", "Status"])), gr.Textbox(value=str(name or "")), gr.Textbox(value=bd), gr.Textbox(value="")
                        bool_map = {"true": True, "false": False}
                        eol = bool_map.get(str(enforce_object_list).lower(), True)
                        com = bool_map.get(str(comments).lower(), True)
                        ann = bool_map.get(str(annotations).lower(), True)
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
                            tables or [],
                            views or [],
                            str(business_domain or ""),
                        )
                        attrs = _get_profile_attributes(pool, name) or {}
                        desc = str(bd)
                        # JSONファイルを更新
                        _save_profiles_to_json(pool)
                        sql = _generate_create_sql_from_attrs(name, attrs, desc)
                        yield gr.Markdown(visible=True, value=f"✅ 作成しました: {name}"), gr.Dataframe(value=get_db_profiles(pool), visible=True), gr.Textbox(value=str(name or "")), gr.Textbox(value=desc), gr.Textbox(value=sql)
                    except Exception as e:
                        yield gr.Markdown(visible=True, value=f"❌ 作成に失敗しました: {str(e)}"), gr.Dataframe(value=get_db_profiles(pool), visible=False), gr.Textbox(value=str(name or "")), gr.Textbox(value=str(business_domain or "")), gr.Textbox(value="")

                profile_refresh_btn.click(
                    fn=refresh_profiles,
                    outputs=[profile_refresh_status, profile_list_df, profile_list_style],
                )

                profile_list_df.select(
                    fn=on_profile_select,
                    inputs=[profile_list_df, compartment_id_input],
                    outputs=[selected_profile_name, business_domain_text, profile_json_text, selected_profile_original_name],
                )

                profile_delete_btn.click(
                    fn=delete_selected_profile,
                    inputs=[selected_profile_name],
                    outputs=[create_info, profile_list_df, selected_profile_name, business_domain_text, profile_json_text],
                )

                profile_update_btn.click(
                    fn=update_selected_profile,
                    inputs=[selected_profile_original_name, selected_profile_name, business_domain_text],
                    outputs=[create_info, profile_list_df, selected_profile_name, business_domain_text, profile_json_text, selected_profile_original_name],
                )

                def refresh_sources_handler():
                    try:
                        t = _get_table_names(pool)
                        v = _get_view_names(pool)
                        return gr.CheckboxGroup(choices=t, visible=True), gr.CheckboxGroup(choices=v, visible=True)
                    except Exception as e:
                        logger.error(f"refresh_sources_handler error: {e}")
                        return gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

                refresh_btn.click(
                    fn=refresh_sources_handler,
                    outputs=[tables_input, views_input],
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
                        business_domain_input,
                    ],
                    outputs=[create_info, profile_list_df, selected_profile_name, business_domain_text, profile_json_text],
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
                            return pd.DataFrame(columns=["BUSINESS_DOMAIN","TEXT"])
                        df = pd.read_excel(str(p))
                        cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                        bd_col = cols_map.get("BUSINESS_DOMAIN")
                        tx_col = cols_map.get("TEXT")
                        if not bd_col or not tx_col:
                            return pd.DataFrame(columns=["BUSINESS_DOMAIN","TEXT"])
                        out = pd.DataFrame({"BUSINESS_DOMAIN": df[bd_col].astype(str), "TEXT": df[tx_col].astype(str)})
                        return out
                    except Exception as e:
                        logger.error(f"訓練データ一覧の取得に失敗しました: {e}")
                        return pd.DataFrame(columns=["BUSINESS_DOMAIN","TEXT"])

                def _td_refresh():
                    try:
                        yield gr.Markdown(visible=True, value="⏳ 訓練データ一覧を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame())
                        df = _td_list()
                        if df is None or df.empty:
                            yield gr.Markdown(visible=True, value="✅ 取得完了（データなし）"), gr.Dataframe(visible=True, value=pd.DataFrame(columns=["BUSINESS_DOMAIN","TEXT"]))
                            return
                        # Display TEXT as a 200-char preview with ellipsis
                        try:
                            df_disp = df.copy()
                            df_disp["TEXT"] = df_disp["TEXT"].astype(str).map(lambda s: s if len(s) <= 200 else (s[:200] + " ..."))
                        except Exception as e:
                            logger.error(f"build training data preview failed: {e}")
                            df_disp = df
                        yield gr.Markdown(visible=False), gr.Dataframe(visible=True, value=df_disp)
                    except Exception as e:
                        yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame())

                # 選択時の詳細取得は不要

                def _td_create(business_domain, text):
                    try:
                        with pool.acquire() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("INSERT INTO ADMIN.TRAINING_DATA (BUSINESS_DOMAIN, TEXT) VALUES (:bd, :txt)", bd=business_domain, txt=str(text or ""))
                                conn.commit()
                        return gr.Markdown(visible=True, value="✅ 登録しました"), gr.Dataframe(value=_td_list(), visible=True)
                    except Exception as e:
                        logger.error(f"登録に失敗しました: {e}")
                        return gr.Markdown(visible=True, value=f"❌ 登録に失敗しました: {e}"), gr.Dataframe(value=_td_list(), visible=True)

                def _td_update(record_id, business_domain, text):
                    try:
                        with pool.acquire() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("UPDATE ADMIN.TRAINING_DATA SET BUSINESS_DOMAIN=:bd, TEXT=:txt WHERE RECORD_ID=:id", bd=business_domain, txt=str(text or ""), id=int(record_id or 0))
                                conn.commit()
                        return gr.Markdown(visible=True, value="✅ 更新しました"), gr.Dataframe(value=_td_list(), visible=True)
                    except Exception as e:
                        logger.error(f"更新に失敗しました: {e}")
                        return gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {e}"), gr.Dataframe(value=_td_list(), visible=True)

                def _td_delete(record_id):
                    try:
                        with pool.acquire() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("DELETE FROM ADMIN.TRAINING_DATA WHERE RECORD_ID=:id", id=int(record_id or 0))
                                conn.commit()
                        return gr.Markdown(visible=True, value="🗑️ 削除しました"), gr.Dataframe(value=_td_list(), visible=True), "", "", ""
                    except Exception as e:
                        logger.error(f"削除に失敗しました: {e}")
                        return gr.Markdown(visible=True, value=f"❌ 削除に失敗しました: {e}"), gr.Dataframe(value=_td_list(), visible=True), "", "", ""

                def _td_train(embed_model, model_name, iterations):
                    """参照コード(No.1-Classifier)に基づいた分類器訓練関数"""
                    try:
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
                        
                        bd_col = cols_map.get("BUSINESS_DOMAIN")
                        tx_col = cols_map.get("TEXT")
                        
                        if not bd_col or not tx_col:
                            error_msg = "必須列(BUSINESS_DOMAIN, TEXT)が見つかりません"
                            logger.error(error_msg)
                            logger.error(f"Available columns: {list(cols_map.keys())}")
                            yield gr.Markdown(visible=True, value=f"⚠️ {error_msg}")
                            return
                        
                        logger.info(f"Using columns - BUSINESS_DOMAIN: {bd_col}, TEXT: {tx_col}")
                        
                        texts = []
                        labels = []
                        for idx, r in df.iterrows():
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
                        
                        # 訓練データをJSONL形式で保存
                        td_path = sp_root / f"{mname}_training_data.jsonl"
                        logger.info(f"Saving training data to: {td_path}")
                        with td_path.open("w", encoding="utf-8") as f:
                            for txt, lab in zip(texts, labels):
                                f.write(json.dumps({"text": txt, "label": lab}, ensure_ascii=False) + "\n")
                        logger.info("Training data saved")
                        
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
                        
                        # インデックスの更新
                        index_path = sp_root / "models.index.json"
                        logger.info(f"Updating model index: {index_path}")
                        try:
                            idx = []
                            if index_path.exists():
                                with index_path.open("r", encoding="utf-8") as f:
                                    idx = json.load(f) or []
                            idx = [x for x in idx if str(x.get("model_name")) != mname]
                            idx.append({
                                "model_name": mname,
                                "labels": sorted(list(set(labels))),
                                "samples": len(texts),
                                "created_at": datetime.now().isoformat(timespec="seconds")
                            })
                            with index_path.open("w", encoding="utf-8") as f:
                                json.dump(idx, f, ensure_ascii=False, indent=2)
                            logger.info("Model index updated")
                        except Exception as e:
                            logger.error(f"インデックス更新エラー: {e}")
                        
                        success_msg = f"✅ 学習完了: モデル '{mname}' を保存しました({len(texts)}件、ラベル: {', '.join(sorted(list(set(labels))))})"
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

                # ラベル候補の更新は削除

                def _list_models():
                    try:
                        sp_root = Path("./models")
                        out = []
                        if sp_root.exists():
                            # .joblibファイルからモデル名を取得
                            for p in sp_root.glob("*.joblib"):
                                model_name = p.stem
                                out.append(model_name)
                        return gr.Dropdown(choices=sorted(out))
                    except Exception as e:
                        logger.error(f"_list_models error: {e}")
                        return gr.Dropdown(choices=[])

                async def _mt_test_async(text, trained_model_name):
                    """参照コード(No.1-Classifier)に基づいた予測関数"""
                    try:
                        logger.info("="*50)
                        logger.info("Starting model prediction...")
                        logger.info(f"Model name: {trained_model_name}")
                        logger.info(f"Input text length: {len(str(text or ''))}")
                        
                        # OCI GenAI クライアントの確認
                        if not _generative_ai_inference_client or not _COMPARTMENT_ID:
                            error_msg = "OCI GenAI クライアントが初期化されていません。環境変数を確認してください"
                            logger.error(error_msg)
                            return gr.Markdown(visible=True, value=f"❌ {error_msg}"), gr.Textbox(value="")
                        
                        logger.info("OCI GenAI client check passed")
                        
                        sp_root = Path("./models")
                        mname = str(trained_model_name or "").strip()
                        if not mname:
                            logger.warning("モデルが選択されていません")
                            return gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="")
                        
                        logger.info(f"Using model: {mname}")
                        
                        model_path = sp_root / f"{mname}.joblib"
                        meta_path = sp_root / f"{mname}.meta.json"
                        
                        logger.info(f"Model path: {model_path}")
                        logger.info(f"Meta path: {meta_path}")
                        
                        if not model_path.exists() or not meta_path.exists():
                            error_msg = f"モデルファイルが見つかりません (model: {model_path.exists()}, meta: {meta_path.exists()})"
                            logger.error(error_msg)
                            return gr.Markdown(visible=True, value="ℹ️ モデルが未学習です。まず『学習を実行』してください"), gr.Textbox(value="")
                        
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
                        return gr.Markdown(visible=True, value="\n".join(lines)), gr.Textbox(value=pred)
                        
                    except Exception as e:
                        error_msg = f"テストに失敗しました: {e}"
                        logger.error(error_msg)
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.info("="*50)
                        return gr.Markdown(visible=True, value=f"❌ {error_msg}"), gr.Textbox(value="")

                def _mt_test(text, trained_model_name):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(_mt_test_async(text, trained_model_name))
                    finally:
                        loop.close()

                # 訓練データ行選択の編集機能は削除
                def _td_download_excel():
                    try:
                        p = Path("uploads") / "training_data.xlsx"
                        if p.exists():
                            return gr.DownloadButton(value=str(p), visible=True)
                        df = pd.DataFrame(columns=["BUSINESS_DOMAIN","TEXT"])
                        tmp = Path("/tmp") / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        with pd.ExcelWriter(tmp) as writer:
                            df.to_excel(writer, sheet_name="training_data", index=False)
                        return gr.DownloadButton(value=str(tmp), visible=True)
                    except Exception:
                        return gr.DownloadButton(visible=False)

                # テンプレートは固定ファイルパスを使用

                def _td_upload_excel(file_path):
                    try:
                        if not file_path:
                            return gr.Textbox(visible=True, value="ファイルを選択してください")
                        try:
                            df = pd.read_excel(str(file_path))
                        except Exception:
                            return gr.Textbox(visible=True, value="Excel読み込みに失敗しました")
                        cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                        required = {"BUSINESS_DOMAIN","TEXT"}
                        if not required.issubset(set(cols_map.keys())):
                            return gr.Textbox(visible=True, value="列名は BUSINESS_DOMAIN, TEXT が必要です")
                        out_df = pd.DataFrame({
                            "BUSINESS_DOMAIN": df[cols_map["BUSINESS_DOMAIN"]],
                            "TEXT": df[cols_map["TEXT"]],
                        })
                        up_dir = Path("uploads")
                        up_dir.mkdir(parents=True, exist_ok=True)
                        dest = up_dir / "training_data.xlsx"
                        if dest.exists():
                            dest.unlink()
                        with pd.ExcelWriter(dest) as writer:
                            out_df.to_excel(writer, sheet_name="training_data", index=False)
                        return gr.Textbox(visible=True, value=f"✅ アップロード完了: {len(out_df)} 件")
                    except Exception as e:
                        logger.error(f"Excelアップロードに失敗しました: {e}")
                        return gr.Textbox(visible=True, value=f"❌ エラー: {e}")

                # 削除: ダウンロードボタンのクリック処理は不要（直接ファイルを提供）
                # 直接固定テンプレートをダウンロード（クリック処理不要）
                def _delete_model(trained_model_name):
                    try:
                        sp_root = Path("./models")
                        mname = str(trained_model_name or "").strip()
                        if not mname:
                            return _list_models()
                        
                        # .joblibファイルと関連ファイルを削除
                        model_path = sp_root / f"{mname}.joblib"
                        meta_path = sp_root / f"{mname}.meta.json"
                        td_path = sp_root / f"{mname}_training_data.jsonl"
                        
                        if model_path.exists():
                            model_path.unlink(missing_ok=True)
                        if meta_path.exists():
                            meta_path.unlink(missing_ok=True)
                        if td_path.exists():
                            td_path.unlink(missing_ok=True)
                        
                        # インデックスから削除
                        index_path = sp_root / "models.index.json"
                        try:
                            if index_path.exists():
                                with index_path.open("r", encoding="utf-8") as f:
                                    idx = json.load(f) or []
                                idx = [x for x in idx if str(x.get("model_name")) != mname]
                                with index_path.open("w", encoding="utf-8") as f:
                                    json.dump(idx, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logger.error(f"_delete_model_meta error: {e}")
                        
                        return _list_models()
                    except Exception as e:
                        logger.error(f"_delete_model error: {e}")
                        return _list_models()

                with gr.TabItem(label="モデル管理"):
                    with gr.Accordion(label="1. 訓練データ一覧", open=True):
                        with gr.Row():
                            td_refresh_btn = gr.Button("訓練データ一覧を取得", variant="primary")
                        with gr.Row():
                            td_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            td_list_df = gr.Dataframe(label="訓練データ一覧", interactive=False, wrap=True, visible=False)
                        with gr.Row():
                            td_upload_excel_file = gr.File(label="Excelファイル", file_types=[".xlsx"], type="filepath")
                        with gr.Row():
                            with gr.Column():
                                gr.DownloadButton(label="Excelダウンロード", value="./uploads/training_data.xlsx", variant="secondary")
                            with gr.Column():
                                td_upload_excel_btn = gr.Button("Excelアップロード(全削除&挿入)", variant="stop")
                        with gr.Row():
                            td_upload_result = gr.Textbox(visible=False)
                    with gr.Accordion(label="2. モデル学習", open=True):
                        with gr.Row():
                            td_embed_model = gr.Dropdown(
                                label="埋め込みモデル",
                                choices=["cohere.embed-v4.0"],
                                value="cohere.embed-v4.0",
                                interactive=True,
                            )
                        with gr.Row():
                            td_model_name = gr.Textbox(label="モデル名", value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", interactive=True)
                        with gr.Row():
                            td_train_iterations = gr.Slider(label="学習回数", minimum=1, maximum=10, step=1, value=5, interactive=True)
                        with gr.Row():
                            td_train_btn = gr.Button("学習を実行", variant="primary")
                        with gr.Row():
                            td_train_status = gr.Markdown(visible=False)
                    with gr.Accordion(label="3. モデルテスト", open=True):
                        with gr.Row():
                            mt_refresh_models_btn = gr.Button("モデル一覧を取得", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                mt_trained_model_select = gr.Dropdown(label="モデル名", show_label=False, container=False, choices=[], interactive=True)
                            with gr.Column():
                                mt_delete_model_btn = gr.Button("選択モデルを削除", variant="stop")
                        with gr.Row():
                            mt_text_input = gr.Textbox(label="テキスト", lines=4, max_lines=8)
                        with gr.Row():
                            mt_label_text = gr.Textbox(label="業務ドメイン(=ラベル)", interactive=False)
                        with gr.Row():
                            mt_test_btn = gr.Button("テスト", variant="primary")
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
                        inputs=[td_embed_model, td_model_name, td_train_iterations],
                        outputs=[td_train_status],
                    )
                    mt_refresh_models_btn.click(
                        fn=_list_models,
                        inputs=[],
                        outputs=[mt_trained_model_select],
                    )
                    mt_delete_model_btn.click(
                        fn=_delete_model,
                        inputs=[mt_trained_model_select],
                        outputs=[mt_trained_model_select],
                    )
                    mt_test_btn.click(
                        fn=_mt_test,
                        inputs=[mt_text_input, mt_trained_model_select],
                        outputs=[mt_test_result, mt_label_text],
                    )

                with gr.TabItem(label="用語集管理"):
                    with gr.Accordion(label="1. 用語集", open=True):
                        # テンプレートファイルを事前作成し、そのままダウンロード可能にする
                        up_dir = Path("uploads")
                        up_dir.mkdir(parents=True, exist_ok=True)
                        _p = up_dir / "terms.xlsx"
                        if not _p.exists():
                            _df = pd.DataFrame(columns=["Term", "Description", "English"])
                            with pd.ExcelWriter(_p) as _writer:
                                _df.to_excel(_writer, sheet_name="terms", index=False)
    
                        with gr.Row():
                            term_upload_file = gr.File(label="用語集Excelをアップロード", file_types=[".xlsx"], type="filepath")
                        with gr.Row():
                            term_upload_result = gr.Textbox(label="アップロード結果", interactive=False, visible=False)
                        with gr.Row():
                            with gr.Column():
                                term_download_btn = gr.DownloadButton(label="テンプレートをダウンロード", value=str(_p), variant="secondary")
                            with gr.Column():
                                term_preview_btn = gr.Button("用語集をプレビュー", variant="primary")
                        with gr.Row():
                            term_preview_status = gr.Markdown(visible=False)
                        with gr.Row():
                            term_preview_df = gr.Dataframe(
                                label="用語集プレビュー",
                                interactive=False,
                                wrap=True,
                                visible=False,
                                value=pd.DataFrame(columns=["TERM", "DESCRIPTION"]),
                            )

                    def _term_list():
                        try:
                            p = Path("uploads") / "terms.xlsx"
                            if not p.exists():
                                return pd.DataFrame(columns=["TERM", "DESCRIPTION"])
                            df = pd.read_excel(str(p))
                            cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                            t_col = cols_map.get("TERM")
                            d_col = cols_map.get("DESCRIPTION")
                            if not t_col or not d_col:
                                return pd.DataFrame(columns=["TERM", "DESCRIPTION"])
                            out = pd.DataFrame({
                                "TERM": df[t_col].astype(str),
                                "DESCRIPTION": df[d_col].astype(str),
                            })
                            return out
                        except Exception as e:
                            logger.error(f"用語集一覧の取得に失敗しました: {e}")
                            return pd.DataFrame(columns=["TERM", "DESCRIPTION"])

                    def _term_refresh():
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 用語集を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame())
                            df = _term_list()
                            if df is None or df.empty:
                                yield gr.Markdown(visible=True, value="✅ 取得完了（データなし）"), gr.Dataframe(visible=True, value=pd.DataFrame(columns=["TERM", "DESCRIPTION"]))
                                return
                            yield gr.Markdown(visible=False), gr.Dataframe(visible=True, value=df)
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame())

                    def _term_download_excel():
                        try:
                            up_dir = Path("uploads")
                            up_dir.mkdir(parents=True, exist_ok=True)
                            p = up_dir / "terms.xlsx"
                            if not p.exists():
                                df = pd.DataFrame(columns=["TERM", "DESCRIPTION"])
                                with pd.ExcelWriter(p) as writer:
                                    df.to_excel(writer, sheet_name="terms", index=False)
                            return gr.DownloadButton(value=str(p), visible=True)
                        except Exception:
                            return gr.DownloadButton(visible=False)

                    def _term_upload_excel(file_path):
                        try:
                            if not file_path:
                                return gr.Textbox(visible=True, value="ファイルを選択してください")
                            try:
                                df = pd.read_excel(str(file_path))
                            except Exception:
                                return gr.Textbox(visible=True, value="Excel読み込みに失敗しました")
                            cols_map = {str(c).upper(): c for c in df.columns.tolist()}
                            required = {"TERM", "DESCRIPTION"}
                            if not required.issubset(set(cols_map.keys())):
                                return gr.Textbox(visible=True, value="列名は TERM, DESCRIPTION が必要です")
                            out_df = pd.DataFrame({
                                "TERM": df[cols_map["TERM"]],
                                "DESCRIPTION": df[cols_map["DESCRIPTION"]],
                            })
                            up_dir = Path("uploads")
                            up_dir.mkdir(parents=True, exist_ok=True)
                            dest = up_dir / "terms.xlsx"
                            if dest.exists():
                                dest.unlink()
                            with pd.ExcelWriter(dest) as writer:
                                out_df.to_excel(writer, sheet_name="terms", index=False)
                            return gr.Textbox(visible=True, value=f"✅ アップロード完了: {len(out_df)} 件")
                        except Exception as e:
                            logger.error(f"用語集Excelアップロードに失敗しました: {e}")
                            return gr.Textbox(visible=True, value=f"❌ エラー: {e}")

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
                                # JSONファイルから読み込む
                                return _load_profiles_from_json()
                            except Exception as e:
                                logger.error(f"_dev_profile_names error: {e}")
                            return []

                        with gr.Row():
                            dev_profile_select = gr.Dropdown(
                                label="Profile",
                                choices=_dev_profile_names(),
                                interactive=True,
                            )

                        with gr.Row():
                            dev_include_extra_prompt = gr.Checkbox(label="追加Promptを結合して送信", value=False)

                        with gr.Row():
                            with gr.Accordion(label="追加プロンプト", open=False):
                                dev_extra_prompt = gr.Textbox(
                                    label="追加のPrompt",
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
                                    autoscroll=True,
                                    container=False,
                                )

                        with gr.Row():
                            dev_prompt_input = gr.Textbox(
                                label="自然言語の質問",
                                placeholder="例: 東京の顧客数を教えて",
                                lines=3,
                                max_lines=10,
                                show_copy_button=True,
                            )

                        with gr.Row():
                            dev_chat_clear_btn = gr.Button("クリア", variant="secondary")
                            dev_chat_execute_btn = gr.Button("実行", variant="primary")

                    with gr.Accordion(label="2. 実行結果", open=True):
                        dev_chat_result_info = gr.Markdown(
                            value="ℹ️ Profile を選択し、自然言語の質問を入力して「実行」をクリックしてください",
                            visible=True,
                        )

                        dev_chat_result_df = gr.Dataframe(
                            label="実行結果",
                            interactive=False,
                            wrap=True,
                            visible=False,
                            value=pd.DataFrame(),
                            elem_id="selectai_dev_chat_result_df",
                        )
                        dev_chat_result_style = gr.HTML(visible=False)

                    with gr.Accordion(label="3. 生成SQL・分析", open=True):
                        dev_generated_sql_text = gr.Textbox(
                            label="生成されたSQL文",
                            lines=8,
                            max_lines=15,
                            interactive=True,
                            show_copy_button=True,
                        )

                        gr.Dataframe(
                            label="使用オブジェクト一覧",
                            interactive=False,
                            wrap=True,
                            visible=False,
                            value=pd.DataFrame(columns=["Name", "Type"]),
                        )

                        with gr.Row():
                            dev_analysis_model_input = gr.Dropdown(
                                label="モデル",
                                choices=[
                                    "xai.grok-code-fast-1",
                                    "xai.grok-3",
                                    "xai.grok-3-fast",
                                    "xai.grok-4",
                                    "xai.grok-4-fast-non-reasoning",
                                    "meta.llama-4-scout-17b-16e-instruct",
                                ],
                                value="xai.grok-code-fast-1",
                                interactive=True,
                            )

                        with gr.Row():
                            dev_ai_analyze_btn = gr.Button("AI分析", variant="primary")

                        with gr.Row():
                            dev_join_conditions_text = gr.Textbox(
                                label="結合条件",
                                lines=6,
                                max_lines=15,
                                interactive=False,
                                show_copy_button=True,
                            )
                            dev_where_conditions_text = gr.Textbox(
                                label="Where条件",
                                lines=6,
                                max_lines=15,
                                interactive=False,
                                show_copy_button=True,
                            )

                    with gr.Accordion(label="4. クエリのフィードバック", open=False):
                        with gr.Row():
                            dev_feedback_type_select = gr.Dropdown(
                                label="種類",
                                choices=["positive", "negative"],
                                value="positive",
                                interactive=True,
                            )

                        dev_feedback_response_text = gr.Textbox(
                            label="修正SQL(response)",
                            placeholder="期待する正しいSQLを入力",
                            lines=4,
                            max_lines=12,
                            show_copy_button=True,
                            interactive=False,
                        )

                        dev_feedback_content_text = gr.Textbox(
                            label="コメント(feedback_content)",
                            placeholder="自然言語で改善点や条件などを入力",
                            lines=4,
                            max_lines=12,
                            show_copy_button=True,
                            interactive=False,
                        )

                        with gr.Row():
                            tpl_btn_null_filter = gr.Button("NULLは除外", variant="secondary")
                            tpl_btn_change_sum = gr.Button("sumを使用", variant="secondary")
                            tpl_btn_add_distinct = gr.Button("重複は除外(distinct)", variant="secondary")
                            tpl_btn_add_date_filter = gr.Button("期間条件を追加", variant="secondary")

                        dev_feedback_result = gr.Markdown(visible=False)
                        dev_feedback_used_sql_text = gr.Textbox(
                            label="使用されたDBMS_CLOUD_AI.FEEDBACK",
                            lines=8,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )

                        dev_feedback_send_btn = gr.Button("フィードバック送信", variant="primary")

                    def _build_showsql_stmt(prompt: str) -> str:
                        s = str(prompt or "")
                        singles = ["!", "~", "^", "@", "#", "$", "%", "&", ";", ":"]
                        for d in singles:
                            if d not in s:
                                return f"select ai showsql q'{d}{s}{d}'"
                        pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
                        for o, c in pairs:
                            if o not in s and c not in s:
                                return f"select ai showsql q'{o}{s}{c}'"
                        esc = s.replace("'", "''")
                        return f"select ai showsql '{esc}'"

                    def _dev_step_generate(profile, prompt, extra_prompt, include_extra):
                        s = str(prompt or "").strip()
                        ep = str(extra_prompt or "").strip()
                        inc = bool(include_extra)
                        final = s if not inc or not ep else (ep + "\n\n" + s)
                        if not profile or not str(profile).strip():
                            logger.error("Profileが未選択です")
                            return gr.Textbox(value="")
                        if not final:
                            logger.error("質問が未入力です")
                            return gr.Textbox(value="")
                        q = final
                        if q.endswith(";"):
                            q = q[:-1]
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    try:
                                        prof = _resolve_profile_name(pool, str(profile or ""))
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
                                                        s = v.read() if hasattr(v, "read") else str(v)
                                                    except Exception:
                                                        s = str(v)
                                                    if s:
                                                        show_cells.append(s)
                                            show_text = "\n".join(show_cells)
                                    except Exception as e:
                                        logger.error(f"dev showsql generate error: {e}")
                                        show_text = ""
                                    try:
                                        cursor.execute(showsql_stmt)
                                    except Exception as e:
                                        logger.error(f"dev showsql execute error: {e}")
                                    _ = _get_sql_id_for_text(showsql_stmt)
                                    def _extract_sql(text: str) -> str:
                                        if not text:
                                            return ""
                                        m = re.search(r"```sql\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
                                        if m:
                                            s = m.group(1).strip()
                                            return s
                                        m2 = re.search(r"SQL\s*:([\s\S]+)$", text, flags=re.IGNORECASE)
                                        if m2:
                                            s = m2.group(1).strip()
                                            return s
                                        m3 = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
                                        if m3:
                                            s = m3.group(0).strip()
                                            return s
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
                                            except Exception as e:
                                                logger.error(f"_to_plain JSON decode error: {e}")
                                            m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", c, flags=re.IGNORECASE)
                                            if m:
                                                generated_sql = m.group(0).strip()
                                                break
                                    gen_sql_display = generated_sql
                                    if gen_sql_display and not gen_sql_display.endswith(";"):
                                        gen_sql_display = gen_sql_display
                                    return gr.Textbox(value=gen_sql_display)
                        except Exception as e:
                            logger.error(f"_dev_step_generate error: {e}")
                            return gr.Textbox(value="")

                    def _dev_step_run_sql(profile, generated_sql):
                        try:
                            yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_dev_chat_result_df"), gr.HTML(visible=False, value="")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    s = str(generated_sql or "").strip()
                                    if not s or not re.match(r"^\s*(select|with)\b", s, flags=re.IGNORECASE):
                                        yield gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_dev_chat_result_df"), gr.HTML(visible=False, value="")
                                        return
                                    run_sql = s
                                    if run_sql.endswith(";"):
                                        run_sql = run_sql[:-1]
                                    cursor.execute(run_sql)
                                    exec_rows = cursor.fetchmany(size=100)
                                    exec_cols = [d[0] for d in cursor.description] if cursor.description else []
                                    if exec_rows:
                                        cleaned_rows = []
                                        for r in exec_rows:
                                            cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                                        df = pd.DataFrame(cleaned_rows, columns=exec_cols)
                                        widths = []
                                        if len(df) > 0:
                                            sample = df.head(5)
                                            columns = max(1, len(df.columns))
                                            for col in df.columns:
                                                series = sample[col].astype(str)
                                                row_max = series.map(len).max() if len(series) > 0 else 0
                                                length = max(len(str(col)), row_max)
                                                widths.append(min(100 / columns, length))
                                        else:
                                            columns = max(1, len(df.columns))
                                            widths = [min(100 / columns, len(c)) for c in df.columns]
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
                                            elem_id="selectai_dev_chat_result_df",
                                        )
                                        style_value = ""
                                        if col_widths:
                                            rules = []
                                            rules.append("#selectai_dev_chat_result_df table { table-layout: fixed; width: 100%; }")
                                            for idx, pct in enumerate(col_widths, start=1):
                                                rules.append(
                                                    f"#selectai_dev_chat_result_df table th:nth-child({idx}), #selectai_dev_chat_result_df table td:nth-child({idx}) {{ width: {pct}%; }}"
                                                )
                                            style_value = "<style>" + "\n".join(rules) + "</style>"
                                        style_component = gr.HTML(visible=bool(style_value), value=style_value)
                                        yield gr.Markdown(visible=True, value=f"✅ {len(df)}件のデータを取得しました"), df_component, style_component
                                        return
                                    yield gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_dev_chat_result_df"), gr.HTML(visible=False, value="")
                        except Exception as e:
                            logger.error(f"_dev_step_run_sql error: {e}")
                            yield gr.Markdown(visible=True, value=f"❌ エラー: {str(e)}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_dev_chat_result_df"), gr.HTML(visible=False, value="")

                    def _dev_step_explain(profile, generated_sql, current_summary):
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    s = str(generated_sql or "").strip()
                                    if not s:
                                        return gr.Markdown(visible=True, value=current_summary)
                                    analysis_text = ""
                                    try:
                                        ex_stmt = f"SELECT AI EXPLAINSQL <sql>\n{s}\n</sql>。\n日本語で解説してください。"
                                        cursor.execute(ex_stmt)
                                        arows = cursor.fetchmany(size=200)
                                        if arows:
                                            parts = []
                                            for r in arows:
                                                for v in r:
                                                    try:
                                                        t = v.read() if hasattr(v, "read") else str(v)
                                                    except Exception as e:
                                                        logger.error(f"_dev_step_explain value read error: {e}")
                                                        t = str(v)
                                                    if t:
                                                        parts.append(t)
                                            analysis_text = "\n".join(parts)
                                    except Exception as e:
                                        logger.error(f"_dev_step_explain analysis fetch error: {e}")
                                        analysis_text = ""
                                    base = str(current_summary or "")
                                    if analysis_text:
                                        base = base + f"\n\n### EXPLAINSQL\n````\n{analysis_text}\n````"
                                    return gr.Markdown(visible=True, value=base)
                        except Exception as e:
                            logger.error(f"_dev_step_explain error: {e}")
                            return gr.Markdown(visible=True, value=f"❌ エラー: {str(e)}")

                    async def _dev_ai_analyze_async(model_name, sql_text):
                        try:
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Textbox(value=""), gr.Textbox(value="")
                            s = str(sql_text or "").strip()
                            if not s:
                                return gr.Textbox(value=""), gr.Textbox(value="")
                            from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                            client = AsyncOciOpenAI(
                                service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                auth=OciUserPrincipalAuth(),
                                compartment_id=compartment_id,
                            )
                            # prompt = (
                            #     "次のSQLからJOIN条件とWHERE条件のみを抽出して出力。形式は厳密に:\n"
                            #     "JOIN:\n<JOIN条件を1行ずつ>\n\nWHERE:\n<WHERE条件を1行ずつ>\n\n"
                            #     "```sql\n" + s + "\n```"
                            # )
                            # messages = [
                            #     {"role": "system", "content": "追加説明は不要。指定形式のみ。"},
                            #     {"role": "user", "content": prompt},
                            # ]

                            prompt = (
                                "Extract ONLY JOIN and WHERE conditions from the SQL query below.\n"
                                "Output in STRICT format (no explanations, no markdown, no extra text):\n\n"
                                "JOIN:\n"
                                "[JOIN_TYPE] alias1(schema.table1).column1 = alias2(schema.table2).column2\n"
                                "[JOIN_TYPE] alias3(schema.table3).column3 = alias4(schema.table4).column4\n\n"
                                "WHERE:\n"
                                "alias(schema.table).column operator value\n\n"
                                "Rules:\n"
                                "- Format: alias(schema.table_name).column or schema.table_name.column (if no alias)\n"
                                "- JOIN_TYPE must be one of: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, CROSS JOIN, JOIN\n"
                                "- Include schema name if present (e.g., ADMIN.USER_ROLE)\n"
                                "- One condition per line\n"
                                "- Keep original operators (=, >, <, LIKE, IN, etc.)\n"
                                "- Preserve exact column names and values with quotes\n"
                                "- If no JOIN/WHERE exists, output 'JOIN:\\nNone' or 'WHERE:\\nNone'\n\n"
                                "SQL:\n```sql\n" + s + "\n```"
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
                            join_text = ""
                            where_text = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                out = msg.content if hasattr(msg, "content") else ""
                                s = str(out or "")
                                s = re.sub(r"```+\w*", "", s)
                                m = re.search(r"JOIN:\s*([\s\S]*?)\n\s*WHERE:\s*([\s\S]*)$", s, flags=re.IGNORECASE)
                                if m:
                                    join_text = m.group(1).strip()
                                    where_text = m.group(2).strip()
                            if not join_text:
                                join_text = "None"
                            if not where_text:
                                where_text = "None"
                            return gr.Textbox(value=join_text), gr.Textbox(value=where_text)
                        except Exception as e:
                            logger.error(f"_dev_ai_analyze_async error: {e}")
                            return gr.Textbox(value="None"), gr.Textbox(value="None")

                    def _dev_ai_analyze(model_name, sql_text):
                        import asyncio
                        # 必須入力項目のチェック
                        if not model_name or not str(model_name).strip():
                            return gr.Textbox(value="⚠️ モデルを選択してください"), gr.Textbox(value="")
                        if not sql_text or not str(sql_text).strip():
                            return gr.Textbox(value="⚠️ SQL文が空です。先にSQL文を生成してください"), gr.Textbox(value="")
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(_dev_ai_analyze_async(model_name, sql_text))
                        finally:
                            loop.close()

                    def _on_dev_chat_clear():
                        return "", gr.Dropdown(choices=_dev_profile_names())

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
                        try:
                            yield gr.Markdown(visible=True, value="⏳ フィードバック送信中..."), gr.Textbox(value="")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof = _resolve_profile_name(pool, str(profile_name or ""))
                                    q = str(prompt_text or "").strip()
                                    if q.endswith(";"):
                                        q = q[:-1]
                                    if not q:
                                        yield gr.Markdown(visible=True, value="⚠️ 質問が未入力のため、フィードバックを送信できませんでした")
                                        return
                                    prompt_text = f"select ai showsql {q}"
                                    gen_stmt = "select dbms_cloud_ai.generate(prompt=> :q, profile_name => :name, action=> :a)"
                                    showsql_stmt = _build_showsql_stmt(q)
                                    try:
                                        cursor.execute(gen_stmt, q=showsql_stmt, name=prof, a="showsql")
                                    except Exception as e:
                                        logger.error(f"_send_feedback generate showsql error: {e}")
                                    try:
                                        cursor.execute(showsql_stmt)
                                    except Exception as e:
                                        logger.error(f"_send_feedback execute showsql error: {e}")
                                    t = str(fb_type or "").lower()
                                    resp = ""
                                    fc = ""
                                    if t == "negative":
                                        resp = str(response_text or "").strip()
                                        fc = str(content_text or "")
                                        if not resp:
                                            yield gr.Markdown(visible=True, value="⚠️ 修正SQLが未入力のため、ネガティブ・フィードバックを送信できませんでした"), gr.Textbox(value="")
                                            return
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
                                        ft=str(fb_type or "").upper(),
                                        resp=resp,
                                        fc=fc,
                                    )
                                    def _lit(x):
                                        s = str(x or "")
                                        return "'" + s.replace("'", "''") + "'"
                                    plsql = (
                                        "BEGIN\n"
                                        "  DBMS_CLOUD_AI.FEEDBACK(\n"
                                        f"    profile_name => {_lit(prof)},\n"
                                        f"    sql_text => {_lit(showsql_stmt)},\n"
                                        f"    feedback_type => {_lit(str(fb_type or '').upper())},\n"
                                        "    response => " + ("NULL" if not resp else _lit(resp)) + ",\n"
                                        "    feedback_content => " + ("NULL" if not fc else _lit(fc)) + ",\n"
                                        "    operation => 'ADD'\n"
                                        "  );\n"
                                        "END;"
                                    )
                                    yield gr.Markdown(visible=True, value="✅ クエリに対するフィードバックを送信しました"), gr.Textbox(value=plsql)
                        except Exception as e:
                            yield gr.Markdown(visible=True, value=f"❌ フィードバック送信に失敗しました: {str(e)}"), gr.Textbox(value=str(e))

                    dev_chat_execute_btn.click(
                        fn=_dev_step_generate,
                        inputs=[dev_profile_select, dev_prompt_input, dev_extra_prompt, dev_include_extra_prompt],
                        outputs=[dev_generated_sql_text],
                    ).then(
                        fn=_dev_step_run_sql,
                        inputs=[dev_profile_select, dev_generated_sql_text],
                        outputs=[dev_chat_result_info, dev_chat_result_df, dev_chat_result_style],
                    )

                    dev_ai_analyze_btn.click(
                        fn=_dev_ai_analyze,
                        inputs=[dev_analysis_model_input, dev_generated_sql_text],
                        outputs=[dev_join_conditions_text, dev_where_conditions_text],
                    )

                    dev_chat_clear_btn.click(
                        fn=_on_dev_chat_clear,
                        outputs=[dev_prompt_input, dev_profile_select],
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
                            global_profile_select = gr.Dropdown(
                                label="Profile",
                                choices=_global_profile_names(),
                                interactive=True,
                            )

                        with gr.Row():
                            global_feedback_index_refresh_btn = gr.Button("最新エントリを取得", variant="primary")

                        with gr.Row():
                            global_feedback_index_df = gr.Dataframe(
                                label="フィードバック索引の最新エントリ",
                                interactive=False,
                                wrap=True,
                                visible=False,
                                value=pd.DataFrame(),
                            )

                        with gr.Row():
                            global_feedback_index_info = gr.Markdown(visible=False)

                        with gr.Row():
                            selected_sql_id = gr.Textbox(label="選択されたSQL_ID", interactive=False)

                        with gr.Row():
                            selected_feedback_delete_btn = gr.Button("選択したフィードバックを削除", variant="stop")
                            
                        with gr.Row():
                            selected_feedback_delete_result = gr.Textbox(label="削除結果", interactive=False, lines=2, max_lines=5)

                    with gr.Accordion(label="2. ベクトルインデックス", open=True):
                        with gr.Row():
                            vec_similarity_threshold_input = gr.Slider(
                                label="Similarity_Threshold",
                                minimum=0.10,
                                maximum=0.95,
                                step=0.05,
                                value=0.90,
                                interactive=True,
                            )
                            vec_match_limit_input = gr.Slider(
                                label="Match_Limit",
                                minimum=1,
                                maximum=5,
                                step=1,
                                value=3,
                                interactive=True,
                            )

                        with gr.Row():
                            vec_update_btn = gr.Button("ベクトルインデックスを更新", variant="primary")

                    def _view_feedback_index_global(profile_name: str):
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof = _resolve_profile_name(pool, str(profile_name or ""))
                                    tab = f"{str(prof).upper()}_FEEDBACK_VECINDEX$VECTAB"
                                    q_no_ctx = (
                                        f'SELECT CONTENT, '
                                        f"JSON_VALUE(ATTRIBUTES, '$.sql_id' RETURNING VARCHAR2(128)) AS SQL_ID, "
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
                                        return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value="ℹ️ まだフィードバック索引がありません")
                                    return gr.Dataframe(visible=True, value=df), gr.Markdown(visible=False)
                        except Exception as e:
                            logger.error(f"_view_feedback_index_global error: {e}")
                            return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value="ℹ️ まだフィードバック索引がありません")

                    def _on_profile_select_change(profile_name: str):
                        try:
                            return (
                                gr.Dataframe(visible=False, value=pd.DataFrame()),
                                gr.Markdown(visible=True, value="ℹ️ Profile選択後は『最新エントリを取得』をクリックしてください"),
                            )
                        except Exception:
                            return (
                                gr.Dataframe(visible=False, value=pd.DataFrame()),
                                gr.Markdown(visible=True, value="ℹ️ Profile選択後は『最新エントリを取得』をクリックしてください"),
                            )

                    global_profile_select.change(
                        fn=_on_profile_select_change,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_df, global_feedback_index_info],
                    )

                    global_feedback_index_refresh_btn.click(
                        fn=_view_feedback_index_global,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_df, global_feedback_index_info],
                    )

                    def on_index_row_select(evt: gr.SelectData, current_df):
                        try:
                            row_index = evt.index[0]
                            df = current_df
                            if isinstance(df, dict) and "data" in df:
                                df = pd.DataFrame(df["data"], columns=df.get("headers", []))
                            if isinstance(df, pd.DataFrame) and not df.empty and row_index >= 0:
                                row = df.iloc[row_index]
                                sql_id = str(row.get("SQL_ID", ""))
                                return sql_id
                        except Exception as e:
                            logger.error(f"on_index_row_select error: {e}")
                        return ""

                    global_feedback_index_df.select(
                        fn=on_index_row_select,
                        inputs=[global_feedback_index_df],
                        outputs=[selected_sql_id],
                    )

                    def _delete_by_sql_id(profile_name: str, sql_id: str):
                        try:
                            if not sql_id:
                                return _view_feedback_index_global(profile_name)[0], "❌ 失敗: SQL_IDが選択されていません"
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    prof = _resolve_profile_name(pool, str(profile_name or ""))
                                    cursor.execute(
                                        """
                                        BEGIN
                                        DBMS_CLOUD_AI.FEEDBACK(
                                            profile_name => :p,
                                            sql_id => :sid,
                                            operation => 'DELETE'
                                        );
                                        END;
                                        """,
                                        p=str(prof),
                                        sid=str(sql_id),
                                    )
                            return _view_feedback_index_global(profile_name)[0], "✅ 成功"
                        except Exception as e:
                            return gr.Dataframe(visible=False, value=pd.DataFrame()), f"❌ 失敗: {str(e)}"

                    selected_feedback_delete_btn.click(
                        fn=_delete_by_sql_id,
                        inputs=[global_profile_select, selected_sql_id],
                        outputs=[global_feedback_index_df, selected_feedback_delete_result],
                    )

                    def _update_vector_index(profile_name: str, similarity_threshold: float, match_limit: int):
                        try:
                            prof = _resolve_profile_name(pool, str(profile_name or ""))
                            idx_name = f"{str(prof).upper()}_FEEDBACK_VECINDEX"
                            tab_name = f"{str(prof).upper()}_FEEDBACK_VECINDEX$VECTAB"
                            logger.info(f"Update vector index: profile={profile_name}, index={idx_name}, table={tab_name}, threshold={similarity_threshold}, limit={match_limit}")
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    # Verify index table exists
                                    try:
                                        cursor.execute(f'SELECT 1 FROM "{tab_name}" FETCH FIRST 1 ROWS ONLY')
                                        _ = cursor.fetchall()
                                    except Exception as e:
                                        logger.error(f"Index table not found: {tab_name}: {e}")
                                        return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value=f"❌ 索引が存在しません: {tab_name}")

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
                                        return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {str(e)}")
                                    logger.info("UPDATE_VECTOR_INDEX succeeded")
                                    return _view_feedback_index_global(profile_name)
                        except Exception as e:
                            logger.error(f"Unexpected error in _update_vector_index: {e}")
                            return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value=f"❌ 更新に失敗しました: {str(e)}")

                    vec_update_btn.click(
                        fn=_update_vector_index,
                        inputs=[global_profile_select, vec_similarity_threshold_input, vec_match_limit_input],
                        outputs=[global_feedback_index_df, global_feedback_index_info],
                    )

                with gr.TabItem(label="コメント管理"):
                    with gr.Accordion(label="1. オブジェクト選択", open=True):
                        with gr.Row():
                            with gr.Column():
                                cm_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column():                        
                                cm_refresh_btn = gr.Button("テーブル・ビュー一覧を取得", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択")
                                cm_tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択")
                                cm_views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)
                        with gr.Row():
                            with gr.Column():
                                cm_sample_limit = gr.Slider(label="サンプル件数", minimum=0, maximum=100, step=1, value=10, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                cm_fetch_btn = gr.Button("情報を取得", variant="primary")

                    with gr.Accordion(label="2. 入力確認", open=False):
                        with gr.Row():
                            with gr.Column():
                                cm_structure_text = gr.Textbox(label="構造情報", lines=8, max_lines=16, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                cm_pk_text = gr.Textbox(label="主キー情報", lines=4, max_lines=10, interactive=True, show_copy_button=True)    
                        with gr.Row():
                            with gr.Column():
                                cm_fk_text = gr.Textbox(label="外部キー情報", lines=6, max_lines=14, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                cm_samples_text = gr.Textbox(label="サンプルデータ", lines=8, max_lines=16, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                cm_extra_input = gr.Textbox(
                                    label="追加入力(Optional)",
                                    placeholder="追加で考慮してほしい説明や条件を記入",
                                    value=(""),
                                    lines=8,
                                    max_lines=16,
                                )

                    with gr.Accordion(label="3. コメント自動生成", open=False):
                        cm_model_input = gr.Dropdown(
                            label="モデル",
                            choices=[
                                "xai.grok-code-fast-1",
                                "xai.grok-3",
                                "xai.grok-3-fast",
                                "xai.grok-4",
                                "xai.grok-4-fast-non-reasoning",
                                "meta.llama-4-scout-17b-16e-instruct",
                            ],
                            value="xai.grok-code-fast-1",
                            interactive=True,
                        )
                        cm_generate_btn = gr.Button("生成", variant="primary")
                        cm_generated_sql = gr.Textbox(label="生成されたSQL文", lines=15, max_lines=15, interactive=True, show_copy_button=True)

                    with gr.Accordion(label="4. 実行", open=False):
                        cm_execute_btn = gr.Button("一括実行", variant="primary")
                        cm_execute_result = gr.Textbox(label="実行結果", interactive=False, lines=5, max_lines=8)

                        with gr.Accordion(label="AI分析と処理", open=False):
                            with gr.Row():
                                cm_ai_model_input = gr.Dropdown(
                                    label="モデル",
                                    choices=[
                                        "xai.grok-code-fast-1",
                                        "xai.grok-3",
                                        "xai.grok-3-fast",
                                        "xai.grok-4",
                                        "xai.grok-4-fast-non-reasoning",
                                        "meta.llama-4-scout-17b-16e-instruct",
                                    ],
                                    value="xai.grok-code-fast-1",
                                    interactive=True,
                                )
                            with gr.Row():
                                cm_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                            with gr.Row():
                                cm_ai_status_md = gr.Markdown(visible=False)
                            with gr.Row():
                                cm_ai_result_md = gr.Markdown(visible=False)

                    def _cm_refresh_objects():
                        try:
                            df_tab = _get_table_df_cached(pool, force=True)
                            df_view = _get_view_df_cached(pool, force=True)
                            names = []
                            if not df_tab.empty and "Table Name" in df_tab.columns:
                                names.extend([str(x) for x in df_tab["Table Name"].tolist()])
                            if not df_view.empty and "View Name" in df_view.columns:
                                names.extend([str(x) for x in df_view["View Name"].tolist()])
                            table_names = sorted(set([str(x) for x in (df_tab["Table Name"].tolist() if (not df_tab.empty and "Table Name" in df_tab.columns) else [])]))
                            view_names = sorted(set([str(x) for x in (df_view["View Name"].tolist() if (not df_view.empty and "View Name" in df_view.columns) else [])]))
                            return gr.Markdown(visible=True, value="✅ 取得完了"), gr.CheckboxGroup(choices=table_names, visible=True), gr.CheckboxGroup(choices=view_names, visible=True)
                        except Exception as e:
                            logger.error(f"_cm_refresh_objects error: {e}")
                            return gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

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
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています")
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
                            return gr.Textbox(value=text)
                        except Exception as e:
                            logger.error(f"_cm_generate_async error: {e}")
                            return gr.Textbox(value=f"❌ エラー: {e}")

                    def _cm_generate(obj_name, model_name, extra_text, struct_text, pk_text, fk_text, samples_text):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(_cm_generate_async(obj_name, model_name, extra_text, struct_text, pk_text, fk_text, samples_text))
                            return result
                        finally:
                            loop.close()

                    def _cm_execute(sql_text):
                        from utils.management_util import execute_comment_sql
                        return execute_comment_sql(pool, sql_text)

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
                            yield gr.Markdown(visible=True, value="⚠️ SQL文を入力してください"), gr.Markdown(visible=False)
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
                            return gr.Textbox(value="", interactive=True)
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
                        return gr.Textbox(value=struct_text, interactive=True)

                    def _cm_fetch_pk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import get_primary_key_info
                        pk_chunks = []
                        for _kind, name in targets:
                            pk_info = get_primary_key_info(pool, name) or ""
                            if pk_info:
                                pk_chunks.append(f"OBJECT: {name}\n{pk_info}")
                        pk_text = "\n\n".join(pk_chunks) if pk_chunks else ""
                        return gr.Textbox(value=pk_text, interactive=True)

                    def _cm_fetch_fk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import get_foreign_key_info
                        fk_chunks = []
                        for _kind, name in targets:
                            fk_info = get_foreign_key_info(pool, name) or ""
                            if fk_info:
                                fk_chunks.append(f"OBJECT: {name}\n{fk_info}")
                        fk_text = "\n\n".join(fk_chunks) if fk_chunks else ""
                        return gr.Textbox(value=fk_text, interactive=True)

                    def _cm_fetch_samples(tables_selected, views_selected, sample_limit):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import display_table_data
                        lim = int(sample_limit)
                        samples_chunks = []
                        if lim > 0:
                            for _kind, name in targets:
                                df = display_table_data(pool, name, lim)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    samples_chunks.append(f"OBJECT: {name}\n" + df.to_csv(index=False))
                        samples_text = "\n\n".join(samples_chunks) if samples_chunks else ""
                        return gr.Textbox(value=samples_text, interactive=True)

                    cm_fetch_btn.click(
                        fn=_cm_fetch_structure,
                        inputs=[cm_tables_input, cm_views_input],
                        outputs=[cm_structure_text],
                    ).then(
                        fn=_cm_fetch_pk,
                        inputs=[cm_tables_input, cm_views_input],
                        outputs=[cm_pk_text],
                    ).then(
                        fn=_cm_fetch_fk,
                        inputs=[cm_tables_input, cm_views_input],
                        outputs=[cm_fk_text],
                    ).then(
                        fn=_cm_fetch_samples,
                        inputs=[cm_tables_input, cm_views_input, cm_sample_limit],
                        outputs=[cm_samples_text],
                    )

                    cm_generate_btn.click(
                        fn=_cm_generate,
                        inputs=[cm_tables_input, cm_model_input, cm_extra_input, cm_structure_text, cm_pk_text, cm_fk_text, cm_samples_text],
                        outputs=[cm_generated_sql],
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

                    def _on_feedback_type_change(fb_type):
                        t = str(fb_type or "").lower()
                        if t == "positive":
                            return gr.Textbox(value="", interactive=False), gr.Textbox(value="", interactive=False)
                        return gr.Textbox(interactive=True), gr.Textbox(interactive=True)

                    dev_feedback_type_select.change(
                        fn=_on_feedback_type_change,
                        inputs=[dev_feedback_type_select],
                        outputs=[dev_feedback_response_text, dev_feedback_content_text],
                    )

                    dev_feedback_send_btn.click(
                        fn=_send_feedback,
                        inputs=[dev_feedback_type_select, dev_feedback_response_text, dev_feedback_content_text, dev_prompt_input, dev_profile_select],
                        outputs=[dev_feedback_result, dev_feedback_used_sql_text],
                    )

                    tpl_btn_null_filter.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_content_text, gr.State("倍率を計算するときにNULL値がある場合は除外してください")],
                        outputs=[dev_feedback_content_text],
                    )
                    tpl_btn_change_sum.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_content_text, gr.State("集計にはCOUNTではなくSUMを使用してください")],
                        outputs=[dev_feedback_content_text],
                    )
                    tpl_btn_add_distinct.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_content_text, gr.State("重複を除外するためDISTINCTを追加してください")],
                        outputs=[dev_feedback_content_text],
                    )
                    tpl_btn_add_date_filter.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_content_text, gr.State("対象期間条件を追加してください（例: 2024年以降）")],
                        outputs=[dev_feedback_content_text],
                    )

                with gr.TabItem(label="アノテーション管理"):
                    with gr.Accordion(label="1. オブジェクト選択", open=True):
                        with gr.Row():
                            with gr.Column():
                                am_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column():
                                am_refresh_btn = gr.Button("テーブル・ビュー一覧を取得", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("###### テーブル選択")
                                am_tables_input = gr.CheckboxGroup(label="テーブル選択", show_label=False, choices=[], visible=False)
                            with gr.Column():
                                gr.Markdown("###### ビュー選択")
                                am_views_input = gr.CheckboxGroup(label="ビュー選択", show_label=False, choices=[], visible=False)
                        with gr.Row():
                            with gr.Column():
                                am_sample_limit = gr.Slider(label="サンプル件数", minimum=0, maximum=100, step=1, value=10, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                am_fetch_btn = gr.Button("情報を取得", variant="primary")

                    with gr.Accordion(label="2. 入力確認", open=False):
                        with gr.Row():
                            with gr.Column():
                                am_structure_text = gr.Textbox(label="構造情報", lines=8, max_lines=16, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                am_pk_text = gr.Textbox(label="主キー情報", lines=4, max_lines=10, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                am_fk_text = gr.Textbox(label="外部キー情報", lines=6, max_lines=14, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                am_samples_text = gr.Textbox(label="サンプルデータ", lines=8, max_lines=16, interactive=True, show_copy_button=True)
                        with gr.Row():
                            with gr.Column():
                                am_extra_input = gr.Textbox(
                                    label="追加入力(Optional)",
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
                                )

                    with gr.Accordion(label="3. アノテーション自動生成", open=False):
                        am_model_input = gr.Dropdown(
                            label="モデル",
                            choices=[
                                "xai.grok-code-fast-1",
                                "xai.grok-3",
                                "xai.grok-3-fast",
                                "xai.grok-4",
                                "xai.grok-4-fast-non-reasoning",
                                "meta.llama-4-scout-17b-16e-instruct",
                            ],
                            value="xai.grok-code-fast-1",
                            interactive=True,
                        )
                        am_generate_btn = gr.Button("生成", variant="primary")
                        am_generated_sql = gr.Textbox(label="生成されたSQL文", lines=15, max_lines=15, interactive=True, show_copy_button=True)

                    with gr.Accordion(label="4. 実行", open=False):
                        am_execute_btn = gr.Button("一括実行", variant="primary")
                        am_execute_result = gr.Textbox(label="実行結果", interactive=False, lines=5, max_lines=8)

                        with gr.Accordion(label="AI分析と処理", open=False):
                            am_ai_model_input = gr.Dropdown(
                                label="モデル",
                                choices=[
                                    "xai.grok-code-fast-1",
                                    "xai.grok-3",
                                    "xai.grok-3-fast",
                                    "xai.grok-4",
                                    "xai.grok-4-fast-non-reasoning",
                                    "meta.llama-4-scout-17b-16e-instruct",
                                ],
                                value="xai.grok-code-fast-1",
                                interactive=True,
                            )
                            am_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                            am_ai_status_md = gr.Markdown(visible=False)
                            am_ai_result_md = gr.Markdown(visible=False)

                    def _am_refresh_objects():
                        try:
                            df_tab = _get_table_df_cached(pool, force=True)
                            df_view = _get_view_df_cached(pool, force=True)
                            table_names = sorted(set([str(x) for x in (df_tab["Table Name"].tolist() if (not df_tab.empty and "Table Name" in df_tab.columns) else [])]))
                            view_names = sorted(set([str(x) for x in (df_view["View Name"].tolist() if (not df_view.empty and "View Name" in df_view.columns) else [])]))
                            return gr.Markdown(visible=True, value="✅ 取得完了"), gr.CheckboxGroup(choices=table_names, visible=True), gr.CheckboxGroup(choices=view_names, visible=True)
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[])

                    def _am_fetch_structure(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
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
                        return gr.Textbox(value=struct_text, interactive=True)

                    def _am_fetch_pk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import get_primary_key_info
                        pk_chunks = []
                        for _kind, name in targets:
                            pk_info = get_primary_key_info(pool, name) or ""
                            if pk_info:
                                pk_chunks.append(f"OBJECT: {name}\n{pk_info}")
                        pk_text = "\n\n".join(pk_chunks) if pk_chunks else ""
                        return gr.Textbox(value=pk_text, interactive=True)

                    def _am_fetch_fk(tables_selected, views_selected):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import get_foreign_key_info
                        fk_chunks = []
                        for _kind, name in targets:
                            fk_info = get_foreign_key_info(pool, name) or ""
                            if fk_info:
                                fk_chunks.append(f"OBJECT: {name}\n{fk_info}")
                        fk_text = "\n\n".join(fk_chunks) if fk_chunks else ""
                        return gr.Textbox(value=fk_text, interactive=True)

                    def _am_fetch_samples(tables_selected, views_selected, sample_limit):
                        tables_selected = tables_selected or []
                        views_selected = views_selected or []
                        targets = []
                        targets.extend([("TABLE", t) for t in tables_selected])
                        targets.extend([("VIEW", v) for v in views_selected])
                        if not targets:
                            return gr.Textbox(value="", interactive=True)
                        from utils.management_util import display_table_data
                        lim = int(sample_limit)
                        samples_chunks = []
                        if lim > 0:
                            for _kind, name in targets:
                                df = display_table_data(pool, name, lim)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    samples_chunks.append(f"OBJECT: {name}\n" + df.to_csv(index=False))
                        samples_text = "\n\n".join(samples_chunks) if samples_chunks else ""
                        return gr.Textbox(value=samples_text, interactive=True)

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
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています")
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
                            return gr.Textbox(value=text)
                        except Exception as e:
                            logger.error(f"_am_generate_async error: {e}")
                            return gr.Textbox(value=f"❌ エラー: {e}")

                    async def _am_ai_analyze_async(model_name, sql_text, exec_result_text):
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
                                "以下のアノテーション文の一括実行内容と実行結果を分析してください。出力は次の3点に限定します。\n"
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
                            yield gr.Markdown(visible=True, value="⚠️ SQL文を入力してください"), gr.Markdown(visible=False)
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
                            return execute_annotation_sql(pool, _prep(sql_text))
                        except Exception as e:
                            logger.error(f"_am_execute error: {e}")
                            return f"❌ エラー: {str(e)}"

                    am_refresh_btn.click(
                        fn=_am_refresh_objects,
                        outputs=[am_refresh_status, am_tables_input, am_views_input],
                    )

                    am_fetch_btn.click(
                        fn=_am_fetch_structure,
                        inputs=[am_tables_input, am_views_input],
                        outputs=[am_structure_text],
                    ).then(
                        fn=_am_fetch_pk,
                        inputs=[am_tables_input, am_views_input],
                        outputs=[am_pk_text],
                    ).then(
                        fn=_am_fetch_fk,
                        inputs=[am_tables_input, am_views_input],
                        outputs=[am_fk_text],
                    ).then(
                        fn=_am_fetch_samples,
                        inputs=[am_tables_input, am_views_input, am_sample_limit],
                        outputs=[am_samples_text],
                    )

                    am_generate_btn.click(
                        fn=_am_generate,
                        inputs=[am_model_input, am_structure_text, am_pk_text, am_fk_text, am_samples_text, am_extra_input],
                        outputs=[am_generated_sql],
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
                            with gr.Column():
                                syn_profile_select = gr.Dropdown(label="Profile", choices=_load_profiles_from_json(), interactive=True)

                        with gr.Row():
                            with gr.Column():
                                syn_refresh_status = gr.Markdown(visible=False)
                        with gr.Row():
                            with gr.Column():
                                syn_refresh_btn = gr.Button("テーブル一覧を取得", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                syn_tables_input = gr.CheckboxGroup(label="テーブル選択", choices=[], visible=False)
                            with gr.Column():
                                syn_rows_per_table = gr.Slider(label="各テーブルの生成件数", minimum=0, maximum=10000, step=1, value=1, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                syn_prompt_input = gr.Textbox(label="生成の指示(オプション)", placeholder="スキーマ特性や分布、制約などを自然言語で記述", lines=4, max_lines=10)
                        with gr.Row():
                            with gr.Column():
                                syn_sample_rows = gr.Slider(label="サンプル行数(sample_rows)", minimum=0, maximum=1000, step=1, value=5, interactive=True)
                            with gr.Column(visible=False):
                                syn_table_statistics = gr.Checkbox(label="テーブル統計を収集(table_statistics)", value=True)
                            with gr.Column():
                                syn_comments = gr.Checkbox(label="コメントを考慮(comments)", value=True)

                        with gr.Row():
                            syn_generate_btn = gr.Button("生成開始", variant="primary")

                    with gr.Accordion(label="2. 進捗と状態", open=True):
                        syn_generate_info = gr.Markdown(visible=True, value="ℹ️ Profileと対象テーブルを選択し、生成開始を押下してください")
                        syn_operation_id_text = gr.Textbox(label="オペレーションID", interactive=False)
                        syn_status_update_btn = gr.Button("ステータスを更新", variant="secondary")
                        syn_status_df = gr.Dataframe(label="ステータス", interactive=False, wrap=True, visible=False, value=pd.DataFrame())
                        syn_status_style = gr.HTML(visible=False)

                    with gr.Accordion(label="3. 結果確認", open=False):
                        with gr.Row():
                            syn_result_table_select = gr.Dropdown(label="テーブル", choices=[], interactive=True)
                            syn_result_limit = gr.Number(label="取得件数", value=50, minimum=0, maximum=10000)
                        syn_result_btn = gr.Button("データを表示", variant="primary")
                        syn_result_info = gr.Markdown(visible=True, value="ℹ️ 生成済みテーブルからデータを表示します")
                        syn_result_df = gr.Dataframe(label="データ表示", interactive=False, wrap=True, visible=False, value=pd.DataFrame(), elem_id="synthetic_data_result_df")
                        syn_result_style = gr.HTML(visible=False)

                    def _syn_profile_names():
                        try:
                            # JSONファイルから読み込む
                            return _load_profiles_from_json()
                        except Exception as e:
                            logger.error(f"_syn_profile_names error: {e}")
                        return []

                    def _syn_refresh_objects(profile_name):
                        try:
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
                            return gr.Markdown(visible=True, value="✅ 取得完了"), gr.CheckboxGroup(choices=table_names, visible=True), gr.Dropdown(choices=table_names)
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ 失敗: {e}"), gr.CheckboxGroup(choices=[]), gr.Dropdown(choices=[])

                    def _syn_build_prompt(tables_selected, rows_per_table, extra_text):
                        tbls = [str(t) for t in (tables_selected or []) if str(t).strip()]
                        rp = int(rows_per_table or 0)
                        base = (
                            "以下のテーブルに対して合成データを生成してください。行数は各テーブルで指定値に近づけ、スキーマの制約と自然な分布を考慮してください。\n"
                            + f"対象テーブル: {', '.join(tbls)}\n"
                            + f"行数目安: {rp} 行/テーブル\n"
                        )
                        if str(extra_text or "").strip():
                            base += "\n追加指示:\n" + str(extra_text).strip()
                        return base

                    def _syn_generate(profile_name, tables_selected, rows_per_table, extra_text, sample_rows, table_statistics, comments):
                        if not profile_name or not str(profile_name).strip():
                            return gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Textbox(value=""), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                        if not tables_selected:
                            return gr.Markdown(visible=True, value="⚠️ テーブルを選択してください"), gr.Textbox(value=""), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                        try:
                            prof = _resolve_profile_name(pool, str(profile_name or ""))
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    try:
                                        p_base = {
                                            "comments": bool(comments),
                                        }
                                    except Exception:
                                        p_base = {"comments": False}
                                    op_id = None
                                    try:
                                        sel = list(tables_selected or [])
                                        if len(sel) == 1:
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
                                            rc = int(rows_per_table or 0)
                                            sr = int(sample_rows or 0)
                                            obj_list = []
                                            for t in sel:
                                                obj_list.append({"owner": "ADMIN", "name": str(t), "record_count": rc, "sample_rows": sr})
                                            obj_json = json.dumps(obj_list, ensure_ascii=False)
                                            p_json = json.dumps(p_base, ensure_ascii=False)
                                            cursor.execute(
                                                "BEGIN DBMS_CLOUD_AI.GENERATE_SYNTHETIC_DATA(profile_name => :name, object_list => :objlist, params => :p); END;",
                                                name=prof,
                                                objlist=obj_json,
                                                p=p_json,
                                            )
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
                                        op_id = None
                                    info_text = "✅ 合成データ生成を開始しました" if op_id else "⚠️ 合成データ生成を開始しました(オペレーションIDの取得に失敗)"
                                    return gr.Markdown(visible=True, value=info_text), gr.Textbox(value=str(op_id or "")), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value=""), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)

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
                                        "BYTES",
                                        "ROWS_LOADED",
                                        "STATUS",
                                        "LAST_MODIFIED",
                                    ]
                                    show_cols = [c for c in keep if c in df.columns]
                                    if show_cols:
                                        df = df[show_cols]
                                    df_component = gr.Dataframe(visible=True, value=df, label=f"ステータス（件数: {len(df)}）", elem_id="synthetic_data_status_df")
                                    style_value = ""
                                    if len(cols) > 0:
                                        sample = df.head(5)
                                        widths = []
                                        columns = max(1, len(df.columns))
                                        for col in df.columns:
                                            series = sample[col].astype(str) if not sample.empty else pd.Series([], dtype=str)
                                            row_max = series.map(len).max() if len(series) > 0 else 0
                                            length = max(len(str(col)), row_max)
                                            widths.append(min(100 / columns, length))
                                        total = sum(widths) if widths else 0
                                        if total > 0:
                                            col_widths = [max(5, int(100 * w / total)) for w in widths]
                                            diff = 100 - sum(col_widths)
                                            if diff != 0 and len(col_widths) > 0:
                                                col_widths[0] = max(5, col_widths[0] + diff)
                                            rules = ["#synthetic_data_status_df table { table-layout: fixed; width: 100%; }"]
                                            for idx, pct in enumerate(col_widths, start=1):
                                                rules.append(f"#synthetic_data_status_df table th:nth-child({idx}), #synthetic_data_status_df table td:nth-child({idx}) {{ width: {pct}%; }}")
                                            style_value = "<style>" + "\n".join(rules) + "</style>"
                                    return gr.Markdown(visible=True, value="✅ ステータス更新完了"), df_component, gr.HTML(visible=bool(style_value), value=style_value)
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)

                    def _syn_display_result(table_name, limit_value):
                        try:
                            from utils.management_util import display_table_data
                            df = display_table_data(pool, table_name, int(limit_value))
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                widths = []
                                cols = df.columns.tolist()
                                sample = df.head(5)
                                columns = max(1, len(cols))
                                for col in cols:
                                    series = sample[col].astype(str)
                                    row_max = series.map(len).max() if len(series) > 0 else 0
                                    length = max(len(str(col)), row_max)
                                    widths.append(min(100 / columns, length))
                                total = sum(widths) if widths else 0
                                style_value = ""
                                if total > 0:
                                    col_widths = [max(5, int(100 * w / total)) for w in widths]
                                    diff = 100 - sum(col_widths)
                                    if diff != 0 and len(col_widths) > 0:
                                        col_widths[0] = max(5, col_widths[0] + diff)
                                    rules = ["#synthetic_data_result_df table { table-layout: fixed; width: 100%; }"]
                                    for idx, pct in enumerate(col_widths, start=1):
                                        rules.append(f"#synthetic_data_result_df table th:nth-child({idx}), #synthetic_data_result_df table td:nth-child({idx}) {{ width: {pct}%; }}")
                                    style_value = "<style>" + "\n".join(rules) + "</style>"
                                return gr.Markdown(visible=False), gr.Dataframe(visible=True, value=df, label=f"データ表示（件数: {len(df)}）", elem_id="synthetic_data_result_df"), gr.HTML(visible=bool(style_value), value=style_value)
                            else:
                                return gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示（件数: 0）", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="データ表示", elem_id="synthetic_data_result_df"), gr.HTML(visible=False, value="")

                    syn_refresh_btn.click(
                        fn=_syn_refresh_objects,
                        inputs=[syn_profile_select],
                        outputs=[syn_refresh_status, syn_tables_input, syn_result_table_select],
                    )

                    syn_generate_btn.click(
                        fn=_syn_generate,
                        inputs=[syn_profile_select, syn_tables_input, syn_rows_per_table, syn_prompt_input, syn_sample_rows, syn_table_statistics, syn_comments],
                        outputs=[syn_generate_info, syn_operation_id_text, syn_status_df, syn_status_style],
                    )

                    syn_status_update_btn.click(
                        fn=_syn_update_status,
                        inputs=[syn_operation_id_text],
                        outputs=[syn_generate_info, syn_status_df, syn_status_style],
                    )

                    syn_result_btn.click(
                        fn=_syn_display_result,
                        inputs=[syn_result_table_select, syn_result_limit],
                        outputs=[syn_result_info, syn_result_df, syn_result_style],
                    )

                # モデル管理タブは上へ移動しました

                with gr.TabItem(label="SQL→質問 逆生成") as reverse_tab:
                    with gr.Accordion(label="1. 入力", open=True):
                        with gr.Row():
                            rev_profile_select = gr.Dropdown(label="Profile", choices=_load_profiles_from_json(), interactive=True)
                        with gr.Row():
                            rev_model_input = gr.Dropdown(
                                label="モデル",
                                choices=[
                                    "xai.grok-code-fast-1",
                                    "xai.grok-3",
                                    "xai.grok-3-fast",
                                    "xai.grok-4",
                                    "xai.grok-4-fast-non-reasoning",
                                    "meta.llama-4-scout-17b-16e-instruct",
                                ],
                                value="xai.grok-code-fast-1",
                                interactive=True,
                            )
                        with gr.Row():
                            rev_sql_input = gr.Textbox(label="対象SQL", lines=8, max_lines=15, show_copy_button=True)

                    with gr.Accordion(label="2. 参照コンテキスト", open=False):
                        rev_context_text = gr.Textbox(label="送信するメタ情報", lines=15, max_lines=15, interactive=False, show_copy_button=True)

                    with gr.Accordion(label="3. 生成", open=True):
                        rev_generate_btn = gr.Button("自然言語を生成", variant="primary")
                        rev_question_output = gr.Textbox(label="推奨質問(日本語)", lines=4, max_lines=10, interactive=False, show_copy_button=True)

                    def _rev_profile_names():
                        try:
                            # JSONファイルから読み込む
                            return _load_profiles_from_json()
                        except Exception as e:
                            logger.error(f"_rev_profile_names error: {e}")
                        return []

                    def _rev_build_context_text(profile_name):
                        try:
                            prof = _resolve_profile_name(pool, str(profile_name or ""))
                            attrs = _get_profile_attributes(pool, prof) or {}
                            obj_list = attrs.get("object_list") or []
                            tables = []
                            views = []
                            try:
                                df_tab = _get_table_df_cached(pool)
                                df_view = _get_view_df_cached(pool)
                                tab_names = set(df_tab["Table Name"].tolist() if (isinstance(df_tab, pd.DataFrame) and "Table Name" in df_tab.columns) else [])
                                view_names = set(df_view["View Name"].tolist() if (isinstance(df_view, pd.DataFrame) and "View Name" in df_view.columns) else [])
                            except Exception:
                                view_names = set()
                            for o in obj_list:
                                name = str((o or {}).get("name") or "")
                                if not name:
                                    continue
                                if name in view_names:
                                    views.append(name)
                                else:
                                    tables.append(name)
                            chunks = []
                            # CREATE DDL + COMMENT statements (column level)
                            for t in sorted(set(tables)):
                                try:
                                    cols_df, ddl = get_table_details(pool, t)
                                except Exception:
                                    cols_df, ddl = pd.DataFrame(), ""
                                if ddl:
                                    chunks.append(str(ddl).strip())
                            for v in sorted(set(views)):
                                try:
                                    cols_df, ddl = get_view_details(pool, v)
                                except Exception:
                                    cols_df, ddl = pd.DataFrame(), ""
                                if ddl:
                                    chunks.append(str(ddl).strip())
                            return "\n\n".join([c for c in chunks if c]) or ""
                        except Exception as e:
                            logger.error(f"_rev_build_context error: {e}")
                            return f"❌ エラー: {e}"

                    def _rev_build_context(profile_name):
                        try:
                            txt = _rev_build_context_text(profile_name)
                            return gr.Textbox(value=txt)
                        except Exception as e:
                            return gr.Textbox(value=f"❌ エラー: {e}")

                    async def _rev_generate_async(model_name, profile_name, sql_text):
                        try:
                            from utils.chat_util import get_oci_region, get_compartment_id
                            region = get_oci_region()
                            compartment_id = get_compartment_id()
                            if not region or not compartment_id:
                                return gr.Textbox(value="ℹ️ OCI設定が不足しています")
                            ctx_comp = _rev_build_context_text(profile_name)
                            s = str(sql_text or "").strip()
                            prompt = (
                                "与えられたSQLとデータベースの文脈から、そのSQLが生成されるような最適な日本語の質問を1つだけ作成してください。\n"
                                "出力は質問文のみ。接頭辞や説明、コードブロック、Markdownは禁止。\n\n"
                                "前提コンテキスト:\n" + str(ctx_comp or "") + "\n\n"
                                "対象SQL:\n```sql\n" + s + "\n```"
                            )
                            from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                            client = AsyncOciOpenAI(
                                service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                                auth=OciUserPrincipalAuth(),
                                compartment_id=compartment_id,
                            )
                            messages = [
                                {"role": "system", "content": "あなたはBIアナリストです。ユーザーがSQL生成エージェントに投げる自然言語の質問文を短く具体的に作ることが仕事です。出力は質問文のみ。"},
                                {"role": "user", "content": prompt},
                            ]
                            resp = await client.chat.completions.create(model=model_name, messages=messages, temperature=0.0)
                            out_text = ""
                            if getattr(resp, "choices", None):
                                msg = resp.choices[0].message
                                out_text = msg.content if hasattr(msg, "content") else ""
                            import re as _re
                            out_text = _re.sub(r"^```.*?\n|\n```$", "", str(out_text or ""), flags=_re.DOTALL).strip()
                            return gr.Textbox(value=out_text)
                        except Exception as e:
                            logger.error(f"_rev_generate_async error: {e}")
                            return gr.Textbox(value=f"❌ エラー: {e}")

                    def _rev_generate(model_name, profile_name, sql_text):
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(_rev_generate_async(model_name, profile_name, sql_text))
                        finally:
                            loop.close()

                    def _on_profile_change_set_context(p):
                        return _rev_build_context(p)

                    rev_profile_select.change(
                        fn=_on_profile_change_set_context,
                        inputs=[rev_profile_select],
                        outputs=[rev_context_text],
                    )

                    rev_generate_btn.click(
                        fn=_rev_generate,
                        inputs=[rev_model_input, rev_profile_select, rev_sql_input],
                        outputs=[rev_question_output],
                    )

        with gr.TabItem(label="ユーザー機能") as user_function_tab:
            with gr.Tabs():
                with gr.TabItem(label="基本機能") as user_basic_tab:
                    with gr.Accordion(label="1. チャット", open=True):
                        def _profile_names():
                            try:
                                # JSONファイルから読み込む
                                return _load_profiles_from_json()
                            except Exception as e:
                                logger.error(f"_profile_names error: {e}")
                            return []

                        with gr.Row():
                            profile_select = gr.Dropdown(
                                label="Profile",
                                choices=_profile_names(),
                                interactive=True,
                            )

                        with gr.Row():
                            include_extra_prompt = gr.Checkbox(label="追加Promptを結合して送信", value=False)

                        with gr.Row():
                            with gr.Accordion(label="追加プロンプト", open=False):
                                extra_prompt = gr.Textbox(
                                label="追加のPrompt",
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
                                autoscroll=True,
                                container=False,
                            )

                        with gr.Row():
                            prompt_input = gr.Textbox(
                                label="自然言語の質問",
                                placeholder="例: 大阪の顧客数を教えて",
                                lines=3,
                                max_lines=10,
                                show_copy_button=True,
                            )

                        with gr.Row():
                            chat_clear_btn = gr.Button("クリア", variant="secondary")
                            chat_execute_btn = gr.Button("実行", variant="primary")

                        with gr.Accordion(label="2. 実行結果", open=True):
                            chat_result_info = gr.Markdown(
                                value="ℹ️ Profile を選択し、自然言語の質問を入力して「実行」をクリックしてください",
                                visible=True,
                            )
                            chat_result_df = gr.Dataframe(
                                label="実行結果",
                                interactive=False,
                                wrap=True,
                                visible=False,
                                value=pd.DataFrame(),
                                elem_id="selectai_chat_result_df",
                            )
                            chat_result_style = gr.HTML(visible=False)

                        with gr.Accordion(label="3. 生成SQL", open=False):
                            generated_sql_status = gr.Markdown(visible=False)
                            generated_sql_text = gr.Textbox(
                                label="生成されたSQL文",
                                lines=8,
                                max_lines=15,
                                interactive=False,
                                show_copy_button=True,
                            )

            def _user_step_generate(profile, prompt, extra_prompt, include_extra):
                s = str(prompt or "").strip()
                ep = str(extra_prompt or "").strip()
                inc = bool(include_extra)
                final = s if not inc or not ep else (ep + "\n\n" + s)
                if not profile or not str(profile).strip():
                    return gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Textbox(value="")
                if not final:
                    return gr.Markdown(visible=True, value="⚠️ 質問を入力してください"), gr.Textbox(value="")
                q = final
                if q.endswith(";"):
                    q = q[:-1]
                try:
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            try:
                                prof = _resolve_profile_name(pool, str(profile or ""))
                                cursor.execute("BEGIN DBMS_CLOUD_AI.SET_PROFILE(profile_name => :name); END;", name=prof)
                            except Exception as e:
                                logger.error(f"SET_PROFILE failed: {e}")
                            gen_stmt = "select dbms_cloud_ai.generate(prompt=> :q, profile_name => :name, action=> :a)"
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
                                logger.error(f"user showsql generate error: {e}")
                                show_text = ""
                            try:
                                cursor.execute(showsql_stmt)
                            except Exception as e:
                                logger.error(f"user showsql execute error: {e}")
                            _ = _get_sql_id_for_text(showsql_stmt)
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
                                    except Exception as e:
                                        logger.error(f"generated_sql JSON parse error: {e}")
                                    m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", c, flags=re.IGNORECASE)
                                    if m:
                                        generated_sql = m.group(0).strip()
                                        break
                            gen_sql_display = generated_sql
                            return gr.Markdown(visible=True, value="✅ SQL生成完了"), gr.Textbox(value=gen_sql_display)
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="")

            def _user_step_run_sql(profile, sql_text):
                if not profile or not str(profile).strip():
                    yield gr.Markdown(visible=True, value="⚠️ Profileを選択してください"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"), gr.HTML(visible=False, value="")
                    return
                try:
                    yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"), gr.HTML(visible=False, value="")
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            exec_rows = []
                            exec_cols = []
                            run_sql = str(sql_text or "").strip()
                            if run_sql and re.match(r"^\s*(select|with)\b", run_sql, flags=re.IGNORECASE):
                                if run_sql.endswith(";"):
                                    run_sql = run_sql[:-1]
                                cursor.execute(run_sql)
                                exec_rows = cursor.fetchmany(size=100)
                                exec_cols = [d[0] for d in cursor.description] if cursor.description else []
                            if exec_rows:
                                cleaned_rows = []
                                for r in exec_rows:
                                    cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                                df = pd.DataFrame(cleaned_rows, columns=exec_cols)
                                widths = []
                                if len(df) > 0:
                                    sample = df.head(5)
                                    columns = max(1, len(df.columns))
                                    for col in df.columns:
                                        series = sample[col].astype(str)
                                        row_max = series.map(len).max() if len(series) > 0 else 0
                                        length = max(len(str(col)), row_max)
                                        widths.append(min(100 / columns, length))
                                else:
                                    columns = max(1, len(df.columns))
                                    widths = [min(100 / columns, len(c)) for c in df.columns]
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
                                    elem_id="selectai_chat_result_df",
                                )
                                style_value = ""
                                if col_widths:
                                    rules = []
                                    rules.append("#selectai_chat_result_df table { table-layout: fixed; width: 100%; }")
                                    for idx, pct in enumerate(col_widths, start=1):
                                        rules.append(
                                            f"#selectai_chat_result_df table th:nth-child({idx}), #selectai_chat_result_df table td:nth-child({idx}) {{ width: {pct}%; }}"
                                        )
                                    style_value = "<style>" + "\n".join(rules) + "</style>"
                                style_component = gr.HTML(visible=bool(style_value), value=style_value)
                                yield gr.Markdown(visible=True, value=f"✅ {len(df)}件のデータを取得しました"), df_component, style_component
                                return
                            yield gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_chat_result_df"), gr.HTML(visible=False, value="")
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {str(e)}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"), gr.HTML(visible=False, value="")

            def _on_chat_clear():
                return "", gr.Dropdown(choices=_profile_names()), gr.Textbox(value="")

            chat_execute_btn.click(
                fn=_user_step_generate,
                inputs=[profile_select, prompt_input, extra_prompt, include_extra_prompt],
                outputs=[generated_sql_status, generated_sql_text],
            ).then(
                fn=_user_step_run_sql,
                inputs=[profile_select, generated_sql_text],
                outputs=[chat_result_info, chat_result_df, chat_result_style],
            )

            chat_clear_btn.click(
                fn=_on_chat_clear,
                outputs=[prompt_input, profile_select, generated_sql_text],
            )

        # 各タブ選択時のProfileドロップダウン更新イベントハンドラー
        def _update_dropdown_from_json(current_value):
            """
            JSONファイルから読み込んでドロップダウンを更新。
            現在の値がリストにない場合は空文字列に設定。
            """
            choices = _load_profiles_from_json()
            if not choices:
                choices = [""]
            # 現在の値がリストに存在するか確認
            if current_value and current_value in choices:
                return gr.Dropdown(choices=choices, value=current_value)
            else:
                # リストにない場合は空文字列に設定
                return gr.Dropdown(choices=choices, value="")

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
