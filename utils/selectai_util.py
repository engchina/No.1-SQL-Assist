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
                df = pd.DataFrame(rows, columns=["Profile Name", "Status"]).sort_values("Profile Name")

        table_names = set(_get_table_names(pool))
        view_names = set(_get_view_names(pool))
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
            df.insert(1, "Tables", tables_col)
            df.insert(2, "Views", views_col)
            df.insert(3, "Region", regions_col)
            df.insert(4, "Model", models_col)
        else:
            df = pd.DataFrame(columns=["Profile Name", "Tables", "Views", "Region", "Model", "Status"])  
        return df
    except Exception:
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
        with gr.TabItem(label="管理者向け機能"):
            with gr.Tabs():
                with gr.TabItem(label="Profileの管理"):
                    with gr.Accordion(label="1. プロファイル一覧", open=True):
                        profile_refresh_btn = gr.Button("一覧を更新", variant="primary")
                        profile_list_df = gr.Dataframe(
                            label="プロファイル一覧(行をクリックして詳細を表示)",
                            interactive=False,
                            wrap=True,
                            value=pd.DataFrame(columns=["Profile Name", "Tables", "Views", "Region", "Model", "Status"]),
                            headers=["Profile Name", "Tables", "Views", "Region", "Model", "Status"],
                            visible=False,
                            elem_id="profile_list_df",
                        )
                        profile_list_style = gr.HTML(visible=False)

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
                    df = get_db_profiles(pool)
                    if df is None or df.empty:
                        empty_df = pd.DataFrame(columns=["Profile Name", "Tables", "Views", "Region", "Model", "Status"])
                        return gr.Dataframe(value=empty_df, visible=True), gr.HTML(visible=False)
                    sample = df.head(5)
                    widths = []
                    for col in sample.columns:
                        series = sample[col].astype(str)
                        row_max = series.map(len).max() if len(series) > 0 else 0
                        length = max(len(str(col)), row_max)
                        widths.append(min(20, length))
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
                    return gr.Dataframe(value=df, visible=True), gr.HTML(visible=bool(style_value), value=style_value)

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
                    outputs=[profile_list_df, profile_list_style],
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
            with gr.Tabs():
                with gr.TabItem(label="チャット・分析"):
                    with gr.Accordion(label="1. チャット", open=True):
                        def _dev_profile_names():
                            try:
                                df = get_db_profiles(pool)
                                if isinstance(df, pd.DataFrame) and not df.empty and df.columns.tolist():
                                    c0 = df.columns[0]
                                    return [str(x) for x in df[c0].tolist()]
                            except Exception:
                                pass
                            return []

                    with gr.Row():
                        dev_profile_refresh_btn = gr.Button("一覧を更新", variant="primary")

                    dev_profile_select = gr.Dropdown(
                        label="Profile",
                        choices=[],
                        interactive=True,
                    )

                    dev_prompt_input = gr.Textbox(
                        label="自然言語の質問",
                        placeholder="例: 大阪の顧客数を教えて",
                        lines=3,
                        max_lines=10,
                        show_copy_button=True,
                    )

                    with gr.Row():
                        dev_chat_clear_btn = gr.Button("クリア", variant="secondary")
                        dev_chat_execute_btn = gr.Button("実行", variant="primary")

                    with gr.Accordion(label="2. 実行結果の表示", open=True):
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

                    with gr.Accordion(label="3. 生成されたSQLと分析", open=True):
                        dev_generated_sql_text = gr.Textbox(
                            label="生成されたSQL文",
                            lines=8,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True,
                        )

                        dev_used_objects_df = gr.Dataframe(
                            label="使用オブジェクト一覧",
                            interactive=False,
                            wrap=True,
                            visible=False,
                            value=pd.DataFrame(columns=["Name", "Type"]),
                        )

                        with gr.Row():
                            dev_join_conditions_text = gr.Textbox(
                                label="結合条件",
                                lines=6,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )
                            dev_where_conditions_text = gr.Textbox(
                                label="Where条件",
                                lines=6,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )

                        dev_sql_summary = gr.Markdown(visible=False)

                    with gr.Accordion(label="4. クエリのフィードバック", open=False):
                        with gr.Row():
                            dev_feedback_type_select = gr.Dropdown(
                                label="種類",
                                choices=["positive", "negative"],
                                value="positive",
                                interactive=True,
                            )

                        dev_feedback_comment = gr.Textbox(
                            label="コメント/修正SQL",
                            placeholder="コメント（例: countではなくsumを使用してください）や修正SQLを入力",
                            lines=4,
                            max_lines=12,
                            show_copy_button=True,
                        )

                        with gr.Row():
                            tpl_btn_null_filter = gr.Button("NULLは除外", variant="secondary")
                            tpl_btn_change_sum = gr.Button("sumを使用", variant="secondary")
                            tpl_btn_add_distinct = gr.Button("重複は除外(distinct)", variant="secondary")
                            tpl_btn_add_date_filter = gr.Button("期間条件を追加", variant="secondary")

                        dev_feedback_result = gr.Markdown(visible=False)

                        dev_feedback_send_btn = gr.Button("フィードバック送信", variant="primary")
 
                    def _execute_select_ai_dev(selected_profile: str, prompt: str):
                        if not selected_profile or not str(selected_profile).strip():
                            gr.Warning("Profileを選択してください")
                            return (
                                gr.Markdown(visible=True, value="⚠️ Profileを選択してください"),
                                gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_dev_chat_result_df"),
                                gr.HTML(visible=False, value=""),
                                gr.Textbox(value=""),
                                gr.Markdown(visible=False),
                                gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Name", "Type"])),
                                gr.Textbox(value=""),
                                gr.Textbox(value=""),
                            )
                        if not prompt or not str(prompt).strip():
                            gr.Warning("質問を入力してください")
                            return (
                                gr.Markdown(visible=True, value="ℹ️ 質問を入力してください"),
                                gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_dev_chat_result_df"),
                                gr.HTML(visible=False, value=""),
                                gr.Textbox(value=""),
                                gr.Markdown(visible=False),
                                gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Name", "Type"])),
                                gr.Textbox(value=""),
                                gr.Textbox(value=""),
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

                                    showsql_stmt = f"SELECT AI SHOWSQL {q}"
                                    show_text = ""
                                    show_cells = []
                                    try:
                                        cursor.execute(showsql_stmt)
                                        rows = cursor.fetchmany(size=200)
                                        cols = [d[0] for d in cursor.description] if cursor.description else []
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
                                    except Exception:
                                        show_text = ""

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
                                            # Try JSON parse
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
                                            # Fallback: find SQL pattern in cell
                                            m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", c, flags=re.IGNORECASE)
                                            if m:
                                                generated_sql = m.group(0).strip()
                                                break
                                    gen_sql_display = generated_sql
                                    if gen_sql_display and not gen_sql_display.endswith(";"):
                                        gen_sql_display = gen_sql_display

                                    def _parse_sql(sql_text: str):
                                        info = {"tables": [], "views": [], "joins": [], "where": "", "object_names": []}
                                        if not sql_text:
                                            return info
                                        s = sql_text
                                        s1 = re.sub(r"--.*", "", s)
                                        s1 = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", "", s1)
                                        from_match = re.search(r"\bFROM\b([\s\S]*?)(\bWHERE\b|\bGROUP\b|\bORDER\b|$)", s1, flags=re.IGNORECASE)
                                        if from_match:
                                            from_part = from_match.group(1)
                                            toks = re.findall(r"\bJOIN\b\s+([\w\.\$]+)", from_part, flags=re.IGNORECASE)
                                            base = re.findall(r"\bFROM\b\s+([\w\.\$]+)", s1, flags=re.IGNORECASE)
                                            names = []
                                            if base:
                                                names.append(base[0])
                                            names.extend(toks)
                                            all_names = []
                                            for n in names:
                                                nn = n.split(".")[-1]
                                                all_names.append(nn)
                                            info["object_names"] = list(dict.fromkeys(all_names))
                                            join_parts = re.findall(r"\bJOIN\b[\s\S]*?\bON\b\s*([\s\S]*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|$)", s1, flags=re.IGNORECASE)
                                            info["joins"] = [jp.strip() for jp in join_parts if jp.strip()]
                                        where_match = re.search(r"\bWHERE\b([\s\S]*?)(?=\bGROUP\b|\bORDER\b|$)", s1, flags=re.IGNORECASE)
                                        if where_match:
                                            info["where"] = where_match.group(1).strip()
                                        try:
                                            table_df = get_table_list(pool)
                                            view_df = get_view_list(pool)
                                            table_set = set([str(x).upper() for x in (table_df["Table Name"].tolist() if "Table Name" in table_df.columns else [])])
                                            view_set = set([str(x).upper() for x in (view_df["View Name"].tolist() if "View Name" in view_df.columns else [])])
                                            t_list = []
                                            v_list = []
                                            for nm in info.get("object_names", []):
                                                up = str(nm).upper()
                                                if up in table_set:
                                                    t_list.append(nm)
                                                elif up in view_set:
                                                    v_list.append(nm)
                                            info["tables"] = t_list
                                            info["views"] = v_list
                                        except Exception:
                                            pass
                                        return info

                                    parsed = _parse_sql(generated_sql)
                                    if not parsed.get("tables") and not parsed.get("views"):
                                        try:
                                            attrs = _get_profile_attributes(pool, selected_profile) or {}
                                            obj_list = attrs.get("object_list") or []
                                            names = [o.get("name") for o in obj_list if o.get("name")]
                                            table_df = get_table_list(pool)
                                            view_df = get_view_list(pool)
                                            table_set = set([str(x).upper() for x in (table_df["Table Name"].tolist() if "Table Name" in table_df.columns else [])])
                                            view_set = set([str(x).upper() for x in (view_df["View Name"].tolist() if "View Name" in view_df.columns else [])])
                                            t_list = []
                                            v_list = []
                                            for nm in names:
                                                up = str(nm).upper()
                                                if up in table_set:
                                                    t_list.append(nm)
                                                elif up in view_set:
                                                    v_list.append(nm)
                                            parsed["tables"] = t_list
                                            parsed["views"] = v_list
                                        except Exception:
                                            pass

                                    analysis_text = ""
                                    try:
                                        if generated_sql:
                                            ex_stmt = f"SELECT AI EXPLAINSQL <sql>\n{generated_sql}\n</sql>。\n日本語で解説してください。"
                                            cursor.execute(ex_stmt)
                                            arows = cursor.fetchmany(size=200)
                                            if arows:
                                                parts = []
                                                for r in arows:
                                                    for v in r:
                                                        try:
                                                            s = v.read() if hasattr(v, "read") else str(v)
                                                        except Exception:
                                                            s = str(v)
                                                        if s:
                                                            parts.append(s)
                                                analysis_text = "\n".join(parts)
                                    except Exception:
                                        analysis_text = ""

                                    exec_rows = []
                                    exec_cols = []
                                    if generated_sql and re.match(r"^\s*(select|with)\b", generated_sql, flags=re.IGNORECASE):
                                        run_sql = generated_sql.strip()
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
                                        gr.Info(f"{len(df)}件のデータを取得しました")

                                        widths = []
                                        if len(df) > 0:
                                            sample = df.head(5)
                                            for col in df.columns:
                                                series = sample[col].astype(str)
                                                row_max = series.map(len).max() if len(series) > 0 else 0
                                                length = max(len(str(col)), row_max)
                                                widths.append(min(20, length))
                                        else:
                                            widths = [min(20, len(c)) for c in df.columns]

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
                                        used_objs = []
                                        for n in parsed.get("tables", []):
                                            used_objs.append((n, "TABLE"))
                                        for n in parsed.get("views", []):
                                            used_objs.append((n, "VIEW"))
                                        used_df = pd.DataFrame(used_objs, columns=["Name", "Type"])

                                        summary_lines = []
                                        summary_lines.append(f"生成SQLの有無: {'あり' if bool(generated_sql) else 'なし'}")
                                        summary_lines.append(f"テーブル数: {len(parsed.get('tables', []))} / ビュー数: {len(parsed.get('views', []))}")
                                        summary_lines.append(f"JOIN条件数: {len(parsed.get('joins', []))}")
                                        summary_lines.append(f"Where条件: {'あり' if bool(parsed.get('where')) else 'なし'}")
                                        src_hint = "SHOWSQL由来" if bool(show_text) else "解析推定"
                                        summary_lines.append(f"情報ソース: {src_hint}")
                                        base_md = "\n".join(["### SQL分析", *[f"- {x}" for x in summary_lines]])
                                        summary_md = base_md + (f"\n\n### EXPLAINSQL\n````\n{analysis_text}\n````" if analysis_text else "")

                                        return (
                                            gr.Markdown(visible=False),
                                            df_component,
                                            style_component,
                                            gr.Textbox(value=gen_sql_display),
                                            gr.Markdown(visible=True, value=summary_md),
                                            gr.Dataframe(visible=True, value=used_df),
                                            gr.Textbox(value="\n\n".join(parsed.get("joins", []))),
                                            gr.Textbox(value=parsed.get("where", "")),
                                        )
                                    else:
                                        used_objs = []
                                        for n in parsed.get("tables", []):
                                            used_objs.append((n, "TABLE"))
                                        for n in parsed.get("views", []):
                                            used_objs.append((n, "VIEW"))
                                        used_df = pd.DataFrame(used_objs, columns=["Name", "Type"])
                                        summary_lines = []
                                        summary_lines.append(f"生成SQLの有無: {'あり' if bool(generated_sql) else 'なし'}")
                                        summary_lines.append(f"テーブル数: {len(parsed.get('tables', []))} / ビュー数: {len(parsed.get('views', []))}")
                                        summary_lines.append(f"JOIN条件数: {len(parsed.get('joins', []))}")
                                        summary_lines.append(f"Where条件: {'あり' if bool(parsed.get('where')) else 'なし'}")
                                        src_hint = "SHOWSQL由来" if bool(show_text) else "解析推定"
                                        summary_lines.append(f"情報ソース: {src_hint}")
                                        base_md = "\n".join(["### SQL分析", *[f"- {x}" for x in summary_lines]])
                                        summary_md = base_md + (f"\n\n### EXPLAINSQL\n````\n{analysis_text}\n````" if analysis_text else "")
                                        return (
                                            gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"),
                                            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_dev_chat_result_df"),
                                            gr.HTML(visible=False, value=""),
                                            gr.Textbox(value=gen_sql_display),
                                            gr.Markdown(visible=True, value=summary_md),
                                            gr.Dataframe(visible=True, value=used_df),
                                            gr.Textbox(value="\n\n".join(parsed.get("joins", []))),
                                            gr.Textbox(value=parsed.get("where", "")),
                                        )
                        except Exception as e:
                            ui_msg = f"❌ エラー: {str(e)}"
                            return (
                                gr.Markdown(visible=True, value=ui_msg),
                                gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="selectai_dev_chat_result_df"),
                                gr.HTML(visible=False, value=""),
                                gr.Textbox(value=""),
                                gr.Markdown(visible=False),
                                gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Name", "Type"])),
                                gr.Textbox(value=""),
                                gr.Textbox(value=""),
                            )

                    def _on_dev_chat_execute(profile, prompt):
                        return _execute_select_ai_dev(profile, prompt)

                    def _on_dev_chat_clear():
                        return "", gr.Dropdown(choices=_dev_profile_names())

                    def _on_dev_profile_refresh():
                        return gr.Dropdown(choices=_dev_profile_names())

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
                                    cursor.execute(
                                        """
                                        select sql_id
                                        from v$sql
                                        where regexp_replace(sql_text,'\\s+',' ') = regexp_replace(:t,'\\s+',' ')
                                        order by last_active_time desc
                                        fetch first 1 rows only
                                        """,
                                        t=s,
                                    )
                                    row = cursor.fetchone()
                                    return str(row[0]) if row and row[0] else ""
                        except Exception:
                            return ""

                    def _send_feedback(fb_type, comment, generated_sql_text, prompt_text, profile_name):
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    cursor.execute("BEGIN DBMS_CLOUD_AI.SET_PROFILE(profile_name => :name); END;", name=profile_name)
                                    q = str(prompt_text or "").strip()
                                    if q.endswith(";"):
                                        q = q[:-1]
                                    if not q:
                                        return gr.Markdown(visible=True, value="⚠️ 質問が未入力のため、フィードバックを送信できませんでした")
                                    showsql_stmt = f"select ai showsql {q}"
                                    try:
                                        cursor.execute(showsql_stmt)
                                    except Exception:
                                        pass
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
                                        p=profile_name,
                                        st=showsql_stmt,
                                        ft=str(fb_type or "").upper(),
                                        resp=str(generated_sql_text or "").strip(),
                                        fc=str(comment or ""),
                                    )
                                    return gr.Markdown(visible=True, value="✅ クエリに対するフィードバックを送信しました")
                        except Exception as e:
                            return gr.Markdown(visible=True, value=f"❌ フィードバック送信に失敗しました: {str(e)}")

                    dev_chat_execute_btn.click(
                        fn=_on_dev_chat_execute,
                        inputs=[dev_profile_select, dev_prompt_input],
                        outputs=[dev_chat_result_info, dev_chat_result_df, dev_chat_result_style, dev_generated_sql_text, dev_sql_summary, dev_used_objects_df, dev_join_conditions_text, dev_where_conditions_text],
                    )

                    dev_chat_clear_btn.click(
                        fn=_on_dev_chat_clear,
                        outputs=[dev_prompt_input, dev_profile_select],
                    )

                    dev_profile_refresh_btn.click(
                        fn=_on_dev_profile_refresh,
                        outputs=[dev_profile_select],
                    )

                    dev_feedback_send_btn.click(
                        fn=_send_feedback,
                        inputs=[dev_feedback_type_select, dev_feedback_comment, dev_generated_sql_text, dev_prompt_input, dev_profile_select],
                        outputs=[dev_feedback_result],
                    )

                    tpl_btn_null_filter.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_comment, gr.State("倍率を計算するときにNULL値がある場合は除外してください")],
                        outputs=[dev_feedback_comment],
                    )
                    tpl_btn_change_sum.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_comment, gr.State("集計にはCOUNTではなくSUMを使用してください")],
                        outputs=[dev_feedback_comment],
                    )
                    tpl_btn_add_distinct.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_comment, gr.State("重複を除外するためDISTINCTを追加してください")],
                        outputs=[dev_feedback_comment],
                    )
                    tpl_btn_add_date_filter.click(
                        fn=_append_comment,
                        inputs=[dev_feedback_comment, gr.State("対象期間条件を追加してください（例: 2024年以降）")],
                        outputs=[dev_feedback_comment],
                    )

                with gr.TabItem(label="フィードバック管理"):
                    def _global_profile_names():
                        try:
                            df = get_db_profiles(pool)
                            if isinstance(df, pd.DataFrame) and not df.empty and df.columns.tolist():
                                c0 = df.columns[0]
                                return [str(x) for x in df[c0].tolist()]
                        except Exception:
                            pass
                        return []

                    with gr.Row():
                        global_profile_refresh_btn = gr.Button("一覧を更新", variant="primary")

                    global_profile_select = gr.Dropdown(
                        label="Profile",
                        choices=[],
                        interactive=True,
                    )

                    global_feedback_index_df = gr.Dataframe(
                        label="フィードバック索引の最新エントリ",
                        interactive=False,
                        wrap=True,
                        visible=False,
                        value=pd.DataFrame(),
                    )

                    with gr.Row():
                        global_feedback_type_select = gr.Dropdown(
                            label="種類",
                            choices=["positive", "negative"],
                            value="positive",
                            interactive=True,
                        )
                        global_feedback_comment = gr.Textbox(
                            label="コメント（検索/削除に使用）",
                            placeholder="キーワードやコメント（完全一致削除を推奨）",
                            lines=2,
                            max_lines=6,
                        )
                        global_feedback_op_select = gr.Dropdown(
                            label="操作",
                            choices=["LIST", "DELETE"],
                            value="LIST",
                            interactive=True,
                        )
                        global_feedback_action_btn = gr.Button("実行", variant="primary")

                    def _view_feedback_index_global(profile_name: str):
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    tab = f"{str(profile_name).upper()}_FEEDBACK_VECINDEX$VECTAB"
                                    q = f'SELECT CONTENT, ATTRIBUTES FROM "{tab}" FETCH FIRST 50 ROWS ONLY'
                                    cursor.execute(q)
                                    rows = cursor.fetchall() or []
                                    cols = [d[0] for d in cursor.description] if cursor.description else []
                                    df = pd.DataFrame(rows, columns=cols)
                                    return gr.Dataframe(visible=True, value=df)
                        except Exception:
                            return gr.Dataframe(visible=True, value=pd.DataFrame())

                    def _on_global_profile_refresh():
                        return gr.Dropdown(choices=_global_profile_names())

                    global_profile_refresh_btn.click(
                        fn=_on_global_profile_refresh,
                        outputs=[global_profile_select],
                    )

                    global_profile_select.change(
                        fn=_view_feedback_index_global,
                        inputs=[global_profile_select],
                        outputs=[global_feedback_index_df],
                    )

                    def _manage_feedback(profile_name: str, fb_type: str, fb_comment: str, op: str):
                        try:
                            with pool.acquire() as conn:
                                with conn.cursor() as cursor:
                                    cursor.execute("BEGIN DBMS_CLOUD_AI.SET_PROFILE(profile_name => :name); END;", name=profile_name)
                                    if op == "LIST":
                                        return _view_feedback_index_global(profile_name)
                                    elif op == "DELETE":
                                        cursor.execute(
                                            """
                                            BEGIN
                                              DBMS_CLOUD_AI.FEEDBACK(
                                                profile_name => :p,
                                                sql_text => :st,
                                                feedback_type => :ft,
                                                feedback_content => :fc,
                                                operation => 'DELETE'
                                              );
                                            END;
                                            """,
                                            p=profile_name,
                                            st=str(fb_comment or "").strip(),
                                            ft=str(fb_type or "").upper(),
                                            fc=str(fb_comment or "").strip(),
                                        )
                                        return _view_feedback_index_global(profile_name)
                        except Exception:
                            return gr.Dataframe(visible=True, value=pd.DataFrame())

                    global_feedback_action_btn.click(
                        fn=_manage_feedback,
                        inputs=[global_profile_select, global_feedback_type_select, global_feedback_comment, global_feedback_op_select],
                        outputs=[global_feedback_index_df],
                    )

        with gr.TabItem(label="ユーザー向け機能"):
            with gr.Tabs():
                with gr.TabItem(label="基本機能"):
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
                            choices=[],
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
                                    sample = df.head(5)
                                    for col in df.columns:
                                        series = sample[col].astype(str)
                                        row_max = series.map(len).max() if len(series) > 0 else 0
                                        length = max(len(str(col)), row_max)
                                        widths.append(min(20, length))
                                else:
                                    widths = [min(20, len(c)) for c in df.columns]

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
                                    gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="selectai_chat_result_df"),
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