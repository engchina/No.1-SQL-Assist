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


def get_db_profiles(pool) -> pd.DataFrame:
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT PROFILE_NAME, STATUS FROM USER_CLOUD_AI_PROFILES ORDER BY PROFILE_NAME"
                )
                rows = cursor.fetchall() or []
                return pd.DataFrame(rows, columns=["Profile Name", "Status"]).sort_values("Profile Name")
    except Exception:
        return pd.DataFrame(columns=["Profile Name", "Status"]) 


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
    except Exception:
        pass
    return attrs


def _generate_create_sql_from_attrs(name: str, attrs: dict) -> str:
    try:
        attr_str = json.dumps(attrs, ensure_ascii=False)
    except Exception:
        attr_str = "{}"
    sql = (
        f"BEGIN DBMS_CLOUD_AI.DROP_PROFILE(profile_name => '{name}'); EXCEPTION WHEN OTHERS THEN NULL; END;\n"
        f"BEGIN DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => '{name}', attributes => '{attr_str}'); END;"
    )
    return sql


def delete_profile(name: str) -> None:
    pass


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
        with gr.TabItem(label="管理機能"):
            with gr.Tabs():
                with gr.TabItem(label="Profileの管理"):
                    with gr.Accordion(label="1. プロファイル一覧", open=True):
                        profile_refresh_btn = gr.Button("一覧を更新", variant="primary")
                        profile_list_df = gr.Dataframe(
                            label="プロファイル一覧(行をクリックして詳細を表示)",
                            interactive=False,
                            wrap=True,
                            value=get_db_profiles(pool),
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
                        build_btn = gr.Button("作成", variant="primary")

                    create_info = gr.Markdown(visible=False)

                def refresh_profiles():
                    return gr.Dataframe(value=get_db_profiles(pool))

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
                            attrs = _get_profile_attributes(pool, name) or {}
                            if compartment_id:
                                attrs.setdefault("oci_compartment_id", compartment_id)
                            sql = _generate_create_sql_from_attrs(name, attrs)
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
                        return gr.Markdown(visible=True, value=f"🗑️ 削除しました: {name}"), gr.Dataframe(value=get_db_profiles(pool)), "", ""
                    except Exception as e:
                        return gr.Markdown(visible=True, value=f"❌ 削除に失敗しました: {str(e)}"), gr.Dataframe(value=get_db_profiles(pool)), name, ""

                def refresh_sources():
                    return gr.CheckboxGroup(choices=_get_table_names(pool)), gr.CheckboxGroup(choices=_get_view_names(pool))

                def build_profile(name, tables, views, compartment_id, region, model):
                    if not tables and not views:
                        return gr.Markdown(visible=True, value="⚠️ テーブルまたはビューを選択してください"), gr.Dataframe(value=get_db_profiles(pool))
                    try:
                        # DBに作成
                        create_db_profile(pool, name, compartment_id, region, model, tables or [], views or [])
                        return gr.Markdown(visible=True, value=f"✅ 作成しました: {name}"), gr.Dataframe(value=get_db_profiles(pool))
                    except Exception as e:
                        return gr.Markdown(visible=True, value=f"❌ 作成に失敗しました: {str(e)}"), gr.Dataframe(value=get_db_profiles(pool))

                

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

        with gr.TabItem(label="開発者向け機能"):
            gr.Markdown(value="準備中")

        with gr.TabItem(label="ユーザー向け機能"):
            with gr.Accordion(label="1. チャット", open=True):
                def _profile_names():
                    try:
                        df = get_db_profiles(pool)
                        if isinstance(df, pd.DataFrame) and not df.empty and df.columns.tolist():
                            c0 = df.columns[0]
                            return [str(x) for x in df[c0].tolist()]
                    except Exception:
                        pass
                    return []

                with gr.Row():
                    user_profile_refresh_btn = gr.Button("一覧を更新", variant="primary")

                profile_select = gr.Dropdown(
                    label="Profile",
                    choices=_profile_names(),
                    interactive=True,
                )

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

            with gr.Accordion(label="2. 実行結果の表示", open=True):
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

            def _execute_select_ai(selected_profile: str, prompt: str):
                if not selected_profile or not str(selected_profile).strip():
                    gr.Warning("Profileを選択してください")
                    return (
                        gr.Markdown(visible=True, value="⚠️ Profileを選択してください"),
                        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"),
                        gr.HTML(visible=False, value=""),
                    )
                if not prompt or not str(prompt).strip():
                    gr.Warning("質問を入力してください")
                    return (
                        gr.Markdown(visible=True, value="ℹ️ 質問を入力してください"),
                        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"),
                        gr.HTML(visible=False, value=""),
                    )

                q = str(prompt).strip()
                if q.endswith(";"):
                    q = q[:-1]

                try:
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            try:
                                cursor.execute("BEGIN DBMS_CLOUD_AI.SET_PROFILE(profile_name => :name); END;", name=selected_profile)
                            except Exception:
                                pass

                            stmt = f"SELECT AI {q}"
                            cursor.execute(stmt)
                            rows = cursor.fetchmany(size=100)
                            cols = [d[0] for d in cursor.description] if cursor.description else []
                            if rows:
                                cleaned_rows = []
                                for r in rows:
                                    cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                                df = pd.DataFrame(cleaned_rows, columns=cols)
                                gr.Info(f"{len(df)}件のデータを取得しました")

                                widths = []
                                if len(df) > 0:
                                    first_row = df.iloc[0]
                                    for col in df.columns:
                                        cell = str(first_row[col])
                                        length = max(len(str(col)), len(cell))
                                        widths.append(min(100, length))
                                else:
                                    widths = [min(100, len(c)) for c in df.columns]

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
                                return (
                                    gr.Markdown(visible=False),
                                    df_component,
                                    style_component,
                                )
                            else:
                                return (
                                    gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"),
                                    gr.Dataframe(visible=True, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_chat_result_df"),
                                    gr.HTML(visible=False, value=""),
                                )
                except Exception as e:
                    ui_msg = f"❌ エラー: {str(e)}"
                    return (
                        gr.Markdown(visible=True, value=ui_msg),
                        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_chat_result_df"),
                        gr.HTML(visible=False, value=""),
                    )

            def _on_chat_execute(profile, prompt):
                return _execute_select_ai(profile, prompt)

            def _on_chat_clear():
                return "", gr.Dropdown(choices=_profile_names())

            def _on_user_profile_refresh():
                return gr.Dropdown(choices=_profile_names())

            chat_execute_btn.click(
                fn=_on_chat_execute,
                inputs=[profile_select, prompt_input],
                outputs=[chat_result_info, chat_result_df, chat_result_style],
            )

            chat_clear_btn.click(
                fn=_on_chat_clear,
                outputs=[prompt_input, profile_select],
            )

            user_profile_refresh_btn.click(
                fn=_on_user_profile_refresh,
                outputs=[profile_select],
            )