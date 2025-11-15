"""SelectAI連携ユーティリティモジュール.

このモジュールは、SelectAIのProfileを管理するUIを提供します。
"""

import logging
import json
import re
import os
from datetime import datetime
from dotenv import find_dotenv, load_dotenv  # noqa: E402
from pathlib import Path

import gradio as gr
import pandas as pd

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

def _get_table_names(pool):
    try:
        df = get_table_list(pool)
        if not df.empty and "Table Name" in df.columns:
            return df["Table Name"].tolist()
    except Exception:
        pass
    return []


def _get_view_names(pool):
    try:
        df = get_view_list(pool)
        if not df.empty and "View Name" in df.columns:
            return df["View Name"].tolist()
    except Exception:
        pass
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


def list_profiles() -> pd.DataFrame:
    rows = []
    for p in _profiles_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                (
                    data.get("profile_name") or p.stem,
                    len(data.get("tables") or []),
                    len(data.get("views") or []),
                    data.get("generated_at") or "",
                )
            )
        except Exception:
            rows.append((p.stem, 0, 0, ""))
    return pd.DataFrame(rows, columns=["Profile Name", "Tables", "Views", "Generated At"]).sort_values("Profile Name")


def load_profile(name: str) -> str:
    p = _profile_path(name)
    if p.exists():
        return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Profile '{name}' が見つかりません")


def save_profile_text(name: str, text: str) -> str:
    data = json.loads(text)
    final_name = data.get("profile_name") or name
    path = _profile_path(final_name)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_name


def delete_profile(name: str) -> None:
    p = _profile_path(name)
    if p.exists():
        p.unlink()


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


def create_db_profile(pool, name: str, compartment_id: str, region: str, model: str, tables: list, views: list):
    attrs = {
        "provider": "oci",
        "credential_name": "OCI_CRED",
        "oci_compartment_id": compartment_id,
        "region": region,
        "model": model,
        "comments": True,
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
            except Exception:
                pass
            cursor.execute(
                "BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => :name, attributes => :attrs); END;",
                name=name,
                attrs=attr_str,
            )


def build_selectai_tab(pool):
    with gr.Tabs():
        with gr.TabItem(label="Profileの管理"):
            with gr.Accordion(label="1. プロファイル一覧", open=True):
                profile_refresh_btn = gr.Button("一覧を更新", variant="primary")
                profile_list_df = gr.Dataframe(
                    label="プロファイル一覧(行をクリックして詳細を表示)",
                    interactive=False,
                    wrap=True,
                    value=list_profiles(),
                )

            with gr.Accordion(label="2. プロファイル詳細と変更", open=True):
                selected_profile_name = gr.Textbox(label="選択されたProfile名", interactive=False)
                profile_json_text = gr.Textbox(
                    label="Profile 作成SQL",
                    lines=5,
                    max_lines=10,
                    show_copy_button=True,
                )
                with gr.Row():
                    profile_delete_btn = gr.Button("選択したProfileを削除", variant="stop")

            with gr.Accordion(label="3. プロファイルの作成", open=False):
                with gr.Row():
                    refresh_btn = gr.Button("一覧を更新", variant="primary")

                with gr.Row():
                    table_choices = _get_table_names(pool)
                    view_choices = _get_view_names(pool)
                    tables_input = gr.CheckboxGroup(label="テーブル選択", choices=table_choices)
                    views_input = gr.CheckboxGroup(label="ビュー選択", choices=view_choices)

                with gr.Row():
                    profile_name = gr.Textbox(
                        label="Profile名",
                        value=f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    )

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
                    model_input = gr.Dropdown(
                        label="Model",
                        choices=["xai.grok-code-fast-1"],
                        value="xai.grok-code-fast-1",
                        interactive=True,
                    )

                with gr.Row():
                    build_btn = gr.Button("作成", variant="primary")

                create_info = gr.Markdown(visible=False)

            def refresh_profiles():
                return gr.Dataframe(value=list_profiles())

            def _generate_create_sql(name: str, compartment_id: str, tables: list, views: list) -> str:
                attrs = {
                    "provider": "oci",
                    "credential_name": "OCI_CRED",
                    "oci_compartment_id": compartment_id or "",
                    "region": "us-chicago-1",
                    "model": "xai.grok-code-fast-1",
                    "comments": True,
                    "object_list": [],
                }
                for t in tables or []:
                    attrs["object_list"].append({"owner": "ADMIN", "name": t})
                for v in views or []:
                    attrs["object_list"].append({"owner": "ADMIN", "name": v})
                attr_str = json.dumps(attrs, ensure_ascii=False)
                sql = (
                    f"BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => '{name}'); EXCEPTION WHEN OTHERS THEN NULL; END;\n"
                    f"BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => '{name}', attributes => '{attr_str}'); END;"
                )
                return sql

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
                        try:
                            data = json.loads(load_profile(name))
                        except Exception:
                            data = {}
                        tables = [t.get("name") for t in (data.get("tables") or []) if t.get("name")]
                        views = [v.get("name") for v in (data.get("views") or []) if v.get("name")]
                        sql = _generate_create_sql(name, compartment_id, tables, views)
                        return name, sql
                except Exception as e:
                    return "", f"❌ 読み込みエラー: {str(e)}"
                return "", ""

            

            def delete_selected_profile(name):
                try:
                    # DB側も削除
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => :name); END;", name=name)
                    delete_profile(name)
                    return gr.Markdown(visible=True, value=f"🗑️ 削除しました: {name}"), gr.Dataframe(value=list_profiles()), "", ""
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ 削除に失敗しました: {str(e)}"), gr.Dataframe(value=list_profiles()), name, ""

            def refresh_sources():
                return gr.CheckboxGroup(choices=_get_table_names(pool)), gr.CheckboxGroup(choices=_get_view_names(pool))

            def build_profile(name, tables, views, compartment_id, region, model):
                if not tables and not views:
                    return gr.Markdown(visible=True, value="⚠️ テーブルまたはビューを選択してください"), gr.Dataframe(value=list_profiles())
                try:
                    # DBに作成
                    create_db_profile(pool, name, compartment_id, region, model, tables or [], views or [])
                    # 参考用JSONを保存
                    text = build_selectai_profile(pool, name, tables or [], views or [])
                    final_name = save_profile_text(name, text)
                    return gr.Markdown(visible=True, value=f"✅ 作成しました: {final_name}"), gr.Dataframe(value=list_profiles())
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ 作成に失敗しました: {str(e)}"), gr.Dataframe(value=list_profiles())

            

            profile_refresh_btn.click(
                fn=refresh_profiles,
                outputs=[profile_list_df],
            )

            profile_list_df.select(
                fn=on_profile_select,
                inputs=[profile_list_df, compartment_id_input],
                outputs=[selected_profile_name, profile_json_text],
            )

            

            profile_delete_btn.click(
                fn=delete_selected_profile,
                inputs=[selected_profile_name],
                outputs=[create_info, profile_list_df, selected_profile_name, profile_json_text],
            )

            refresh_btn.click(
                fn=refresh_sources,
                outputs=[tables_input, views_input],
            )

            build_btn.click(
                fn=build_profile,
                inputs=[profile_name, tables_input, views_input, compartment_id_input, region_input, model_input],
                outputs=[create_info, profile_list_df],
            )