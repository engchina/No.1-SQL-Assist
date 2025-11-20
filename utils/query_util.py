"""クエリ実行ユーティリティモジュール.

このモジュールは、Select文のみを安全に実行し、結果を表形式で
表示するためのGradio UIコンポーネントを提供します。
"""

import logging
import json
import re
import traceback

import gradio as gr
import pandas as pd
from oracledb import DatabaseError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _is_select_sql(sql: str) -> bool:
    if not sql:
        return False
    s = sql.strip()
    if not re.match(r"^\s*(select|with)\b", s, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(insert|update|delete|merge|create|drop|alter|truncate|grant|revoke)\b", s, flags=re.IGNORECASE):
        return False
    sc = s.count(";")
    if sc > 1:
        return False
    if sc == 1 and not s.endswith(";"):
        return False
    return True


def execute_select_sql(pool, sql: str, limit: int):
    if not sql or not sql.strip():
        gr.Warning("SQLを入力してください")
        return (
            gr.Markdown(visible=True),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果"),
        )

    if not _is_select_sql(sql):
        gr.Error("SELECT文のみ実行可能です")
        return (
            gr.Markdown(visible=True),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果"),
        )

    q = sql.strip()
    if q.endswith(";"):
        q = q[:-1]

    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(q)
                rows = cursor.fetchmany(size=int(limit) if limit and int(limit) > 0 else 100)
                cols = [d[0] for d in cursor.description] if cursor.description else []
                if rows:
                    cleaned_rows = []
                    for r in rows:
                        row_vals = []
                        for v in r:
                            val = v.read() if hasattr(v, "read") else v
                            if isinstance(val, (bytes, bytearray)):
                                try:
                                    val = val.decode("utf-8")
                                except Exception:
                                    try:
                                        val = val.decode("latin1")
                                    except Exception:
                                        val = str(val)
                            if isinstance(val, (dict, list)):
                                try:
                                    val = json.dumps(val, ensure_ascii=False)
                                except Exception:
                                    val = str(val)
                            elif isinstance(val, str):
                                s = val.strip()
                                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                                    try:
                                        obj = json.loads(s)
                                        disp = json.dumps(obj, ensure_ascii=False, indent=2)
                                        disp = disp.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '')
                                        disp = disp.replace('\\"', '"')
                                        val = disp
                                    except Exception:
                                        val = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '').replace('\\"', '"')
                                else:
                                    val = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '').replace('\\"', '"')
                            row_vals.append(val)
                        cleaned_rows.append(row_vals)
                    df = pd.DataFrame(cleaned_rows, columns=cols)
                    gr.Info(f"{len(df)}件のデータを取得しました")
                    widths = []
                    if len(df) > 0:
                        sample = df.head(5)
                        columns = max(1, len(df.columns))
                        for i, col in enumerate(df.columns):
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
                        elem_id="query_result_df",
                    )
                    style_value = ""
                    if col_widths:
                        rules = []
                        rules.append("#query_result_df table { table-layout: fixed; width: 100%; }")
                        for idx, pct in enumerate(col_widths, start=1):
                            rules.append(
                                f"#query_result_df table th:nth-child({idx}), #query_result_df table td:nth-child({idx}) {{ width: {pct}%; }}"
                            )
                        style_value = "<style>" + "\n".join(rules) + "</style>"
                    style_component = gr.HTML(visible=bool(style_value), value=style_value)
                    return (
                        gr.Markdown(visible=False),
                        df_component,
                        style_component,
                    )
                else:
                    logger.info("No rows returned")
                    return (
                        gr.Markdown(visible=True, value="ℹ️ データは返却されませんでした"),
                        gr.Dataframe(visible=True, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="query_result_df"),
                        gr.HTML(visible=False, value=""),
                    )
    except DatabaseError as e:
        logger.error(f"Oracleエラー: {e}")
        logger.error(traceback.format_exc())
        s = str(e)
        m = re.search(r"ORA-(\d{5})", s)
        code = m.group(0) if m else None
        hint = "SQLと権限、スキーマを確認してください"
        if code == "ORA-00942":
            hint = "対象の表またはビューが存在しません。スキーマやオブジェクト名を確認してください"
        ui_msg = f"❌ エラー: {s}\n\n👉 ヒント: {hint}"
        return (
            gr.Markdown(visible=True, value=ui_msg),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"クエリ実行エラー: {str(e)}")

    return (
        gr.Markdown(visible=True),
        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
        gr.HTML(visible=False, value=""),
    )


def build_query_tab(pool):
    """クエリ実行タブのUIを構築する."""
    with gr.TabItem(label="SQLの実行") as tab_query:
        with gr.Accordion(label="1. SQLの入力", open=True):
            sql_input = gr.Textbox(
                label="SQL文（SELECTのみ）\n注意: 重いSQLを実行すると、処理が遅くなり画面が一時的に固まる場合があります",
                placeholder="SELECT 文を入力してください（INSERT/UPDATE等は禁止）",
                lines=8,
                max_lines=15,
                show_copy_button=True,
            )

            with gr.Row():
                limit_input = gr.Number(
                    label="取得件数",
                    value=100,
                    minimum=1,
                    maximum=10000,
                )

            with gr.Row():
                clear_btn = gr.Button("クリア", variant="secondary")
                execute_btn = gr.Button("実行", variant="primary")

        with gr.Accordion(label="2. 実行結果の表示", open=True):
            result_info = gr.Markdown(
                value="ℹ️ SELECT文を入力して「実行」をクリックしてください",
                visible=True,
            )

            result_df = gr.Dataframe(
                label="実行結果",
                interactive=False,
                wrap=True,
                visible=False,
                value=pd.DataFrame(),
                elem_id="query_result_df",
            )
            result_style = gr.HTML(visible=False)

        with gr.Accordion(label="3. AI分析と処理", open=False):
            ai_model_input = gr.Dropdown(
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
            ai_analyze_btn = gr.Button("AI分析", variant="primary")
            ai_status_md = gr.Markdown(visible=False)
            ai_result_md = gr.Markdown(visible=False)

        async def _ai_analyze_async(model_name, sql_text, limit_value, result_df_input):
            from utils.chat_util import get_oci_region, get_compartment_id
            region = get_oci_region()
            compartment_id = get_compartment_id()
            if not region or not compartment_id:
                return gr.Markdown(visible=True, value="OCI設定が不足しています")
            try:
                import pandas as pd
                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                if isinstance(result_df_input, dict) and "data" in result_df_input:
                    headers = result_df_input.get("headers", [])
                    df = pd.DataFrame(result_df_input["data"], columns=headers)
                elif isinstance(result_df_input, pd.DataFrame):
                    df = result_df_input
                else:
                    df = pd.DataFrame()
                preview = df.head(20).to_markdown(index=False) if not df.empty else ""
                q = (sql_text or "").strip()
                if q.endswith(";"):
                    q = q[:-1]
                prompt = (
                    "以下のSELECT文とその結果を分析し、問題点や最適化の提案、次に取るべき対応案を提示してください。\n\n"
                    + ("SQL:\n```sql\n" + q + "\n```\n" if q else "")
                    + ("取得件数:\n" + str(limit_value) + "\n" if limit_value else "")
                    + ("結果プレビュー:\n" + preview + "\n" if preview else "")
                )
                client = AsyncOciOpenAI(
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    auth=OciUserPrincipalAuth(),
                    compartment_id=compartment_id,
                )
                messages = [
                    {"role": "system", "content": "あなたは熟練のデータベースエンジニア兼SQL最適化の専門家です。結果とSQLを精確に分析し、性能、可読性、正確性の観点で改善提案を出してください。"},
                    {"role": "user", "content": prompt},
                ]
                resp = await client.chat.completions.create(model=model_name, messages=messages)
                text = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    text = msg.content if hasattr(msg, "content") else ""
                return gr.Markdown(visible=True, value=text or "分析結果が空です")
            except Exception as e:
                return gr.Markdown(visible=True, value=f"エラー: {e}")

        def ai_analyze(model_name, sql_text, limit_value, result_df_input):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                result_md = loop.run_until_complete(_ai_analyze_async(model_name, sql_text, limit_value, result_df_input))
                yield gr.Markdown(visible=True, value="✅ 完了"), result_md
            except Exception as e:
                yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
            finally:
                loop.close()

        def on_execute(sql, limit):
            try:
                yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"), gr.HTML(visible=False, value="")
                result_info, result_df, result_style = execute_select_sql(pool, sql, limit)
                yield result_info, result_df, result_style
            except Exception as e:
                yield gr.Markdown(visible=True, value=f"❌ 実行に失敗しました: {str(e)}"), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"), gr.HTML(visible=False, value="")

        def on_clear():
            return ""

        execute_btn.click(
            fn=on_execute,
            inputs=[sql_input, limit_input],
            outputs=[result_info, result_df, result_style],
        )

        ai_analyze_btn.click(
            fn=ai_analyze,
            inputs=[ai_model_input, sql_input, limit_input, result_df],
            outputs=[ai_status_md, ai_result_md],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[sql_input],
        )

    return tab_query
