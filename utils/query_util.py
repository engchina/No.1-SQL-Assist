"""クエリ実行ユーティリティモジュール.

SELECTを1文のみ、その他のデータ操作/DDL/PL/SQLを複数文同時に安全に実行し、
SELECTはデータフレーム表示、非SELECTはサマリー表示を行うUIコンポーネントを提供します。
"""

import logging
import json
import re
import traceback

import gradio as gr
import pandas as pd
import oracledb
from oracledb import DatabaseError
from utils.common_util import CHAT_MODEL_CHOICES, DEFAULT_CHAT_MODEL
from utils.oracle_sql_util import (
    OracleScriptError,
    created_program,
    is_single_select,
    parse_oracle_script,
)
from utils.vpd_util import (
    request_username,
    user_role,
    vpd_runtime_connection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_ADMIN_QUERY_NOTICE = (
    "ℹ️ SELECT/WITHは1文のみ実行可能です。"
    "複数実行時はSELECTを含めないでください。\n\n"
    "ℹ️ 通常SQLはセミコロンで終了します。package、function、procedure、"
    "trigger、type、匿名PL/SQLブロックはSQL*Plus標準どおり、末尾に独占行の "
    "`/` が必須です。\n\n"
    "ℹ️ `SET`、`SPOOL`、`PROMPT`、`WHENEVER`、`@script.sql` は未対応です。"
)

_READ_ONLY_QUERY_NOTICE = (
    "ℹ️ ADMIN以外のユーザーは、SELECT/WITHを1文のみ実行できます。\n\n"
    "ℹ️ INSERT、UPDATE、DELETE、MERGE、DDL、PL/SQLなど、"
    "データを変更する文は実行できません。\n\n"
    "ℹ️ SQLファイルから読み込んだ内容にも同じ制限が適用されます。"
)


def _is_select_sql(sql: str) -> bool:
    return is_single_select(sql)


def _query_access_notice(role: str | None) -> str:
    """Return the SQL execution guidance for an authenticated role."""
    if role == "admin":
        return _ADMIN_QUERY_NOTICE
    if role == "vpd":
        return _READ_ONLY_QUERY_NOTICE
    return ""


def _query_error_result(message: str):
    return (
        gr.Markdown(visible=True, value=message),
        gr.Dataframe(
            visible=False,
            value=pd.DataFrame(),
            label="実行結果",
            elem_id="query_result_df",
        ),
        gr.HTML(visible=False, value=""),
    )


def _read_only_validation_error(sql: str) -> str | None:
    if not sql or not str(sql).strip():
        return "❌ エラー: SQLを入力してください"
    if not _is_select_sql(str(sql)):
        return "❌ エラー: SELECT/WITHは1文のみ実行可能です"
    return None


def execute_select_sql(pool, sql: str, limit: int, login_user: str | None = None):
    validation_error = _read_only_validation_error(sql)
    if validation_error:
        logger.error(validation_error)
        return _query_error_result(validation_error)

    try:
        q = parse_oracle_script(sql)[0].text
        connection_scope = (
            vpd_runtime_connection(pool, login_user)
            if login_user
            else pool.acquire()
        )
        with connection_scope as conn:
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
                        elem_id="query_result_df",
                    )
                    style_value = ""
                    if col_widths:
                        rules = []
                        rules.append("#query_result_df { width: 100% !important; }")
                        rules.append("#query_result_df .wrap { overflow-x: auto !important; }")
                        rules.append("#query_result_df table { table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; }")
                        for idx, pct in enumerate(col_widths, start=1):
                            rules.append(
                                f"#query_result_df table th:nth-child({idx}), #query_result_df table td:nth-child({idx}) {{ width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; }}"
                            )
                        style_value = "<style>" + "\n".join(rules) + "</style>"
                    style_component = gr.HTML(visible=bool(style_value), value=style_value)
                    return (
                        gr.Markdown(visible=True, value=f"✅ {len(df)}件のデータを取得しました"),
                        df_component,
                        style_component,
                    )
                else:
                    logger.info("No rows returned")
                    return (
                        gr.Markdown(visible=True, value="✅ 表示完了（データなし）"),
                        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果（件数: 0）", elem_id="query_result_df"),
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
            hint = (
                "対象の表またはビューが存在しないか、参照権限がありません。"
                "スキーマ、オブジェクト名、データアクセスルールを"
                "確認してください"
            )
        ui_msg = f"❌ エラー: {s}\n\n👉 ヒント: {hint}"
        return (
            gr.Markdown(visible=True, value=ui_msg),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        ui_msg = f"❌ クエリ実行エラー: {str(e)}"

    return (
        gr.Markdown(visible=True, value=ui_msg if 'ui_msg' in locals() else "❌ クエリ実行エラー"),
        gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
        gr.HTML(visible=False, value=""),
    )


def _normalize_exec(stmt: str) -> str:
    s = str(stmt or '').strip()
    if re.match(r"^(exec|execute)\b", s, flags=re.IGNORECASE):
        body = re.sub(r"^(exec|execute)\s+", "", s, flags=re.IGNORECASE).strip()
        if body.endswith(';'):
            body = body[:-1]
        return f"BEGIN {body}; END;"
    return s


def _compilation_errors(cursor, sql: str) -> str:
    program = created_program(sql)
    if not program:
        return ""
    object_type, object_name = program
    cursor.execute(
        """
        SELECT line, position, text
        FROM user_errors
        WHERE name = :name AND type = :object_type
        ORDER BY sequence
        """,
        name=object_name,
        object_type=object_type,
    )
    errors = cursor.fetchall()
    return "\n".join(
        f"line {line}, position {position}: {text}"
        for line, position, text in errors
    )


def execute_sql_general(pool, sql: str, limit: int):
    if not sql or not str(sql).strip():
        logger.error("SQLが未入力です")
        return (
            gr.Markdown(visible=True, value="❌ エラー: SQLを入力してください"),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    try:
        parsed = parse_oracle_script(sql)
    except OracleScriptError as exc:
        return (
            gr.Markdown(visible=True, value=f"❌ SQL形式エラー: {exc}"),
            gr.Dataframe(
                visible=False,
                value=pd.DataFrame(),
                label="実行結果",
                elem_id="query_result_df",
            ),
            gr.HTML(visible=False, value=""),
        )
    statements = [statement for statement in parsed if statement.text.strip()]
    if not statements:
        logger.error("分割後のSQLが空です")
        return (
            gr.Markdown(visible=True, value="❌ エラー: SQLを入力してください"),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    types = [statement.statement_type for statement in statements]
    sel_count = sum(1 for t in types if t == 'SELECT')
    if len(statements) == 1 and sel_count == 1:
        return execute_select_sql(pool, statements[0].text, limit)
    if len(statements) > 1 and sel_count > 0:
        return (
            gr.Markdown(visible=True, value="❌ エラー: 複数実行にSELECTは含められません"),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    import time
    rows = []
    ok = True
    dml_types = {"INSERT", "UPDATE", "DELETE", "MERGE"}
    ddl_types = {
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "COMMENT",
    }
    pure_dml = all(statement.statement_type in dml_types for statement in statements)
    implicit_commit_seen = False
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.callproc("dbms_output.enable")
                except Exception as e:
                    logger.error(f"dbms_output.enable failed: {e}")
                for idx, statement in enumerate(statements, start=1):
                    typ = statement.statement_type
                    run = _normalize_exec(statement.text)
                    t0 = time.perf_counter()
                    execution_reached = False
                    try:
                        if typ in ddl_types:
                            # Oracle commits the current transaction immediately
                            # before attempting a DDL statement.
                            implicit_commit_seen = True
                            for previous in rows:
                                if previous[2] == "成功":
                                    previous[5] = "DDL実行前COMMITにより確定"
                        cursor.execute(run)
                        execution_reached = True
                        compile_errors = _compilation_errors(cursor, run)
                        if compile_errors:
                            raise RuntimeError(
                                "Oracleコンパイルエラー:\n" + compile_errors
                            )
                        rc = cursor.rowcount if hasattr(cursor, 'rowcount') else None
                        dur = int((time.perf_counter() - t0) * 1000)
                        is_dml = typ in ('INSERT', 'UPDATE', 'DELETE', 'MERGE')
                        is_plsql = statement.is_plsql_unit or typ == 'PLSQL'
                        is_comment = (typ == 'COMMENT')
                        msg = _fetch_dbms_output(cursor)
                        if is_dml:
                            msg = msg or f"RowsAffected={rc if rc is not None else 0}"
                        elif is_plsql:
                            msg = msg or 'PL/SQL executed'
                        elif is_comment:
                            msg = msg or 'Comment applied'
                        else:
                            msg = msg or 'OK'
                        if typ in ddl_types:
                            permanence = "DDLにより確定"
                        elif is_plsql:
                            permanence = "完了時にCOMMIT（内部COMMITは要確認）"
                        else:
                            permanence = "未確定"
                        rows.append(
                            [
                                idx,
                                typ,
                                "成功",
                                rc if rc is not None else -1,
                                msg,
                                permanence,
                                dur,
                            ]
                        )
                    except Exception as e:
                        ok = False
                        dur = int((time.perf_counter() - t0) * 1000)
                        msg = str(e)
                        logger.error(f"Statement #{idx} failed: {e}")
                        logger.error(traceback.format_exc())
                        if typ in ddl_types and execution_reached:
                            permanence = (
                                "DDLは確定済み（コンパイル状態を確認）"
                            )
                        elif typ in ddl_types:
                            permanence = (
                                "DDL自身は未反映（直前までの処理は確定）"
                            )
                        else:
                            permanence = "未反映"
                        rows.append(
                            [idx, typ, "失敗", -1, msg, permanence, dur]
                        )
                        break
                if ok:
                    conn.commit()
                    for row in rows:
                        if row[2] == "成功":
                            row[5] = "確定"
                else:
                    conn.rollback()
                    for row in rows:
                        if row[2] != "成功":
                            continue
                        if row[5] in {
                            "DDLにより確定",
                            "DDL実行前COMMITにより確定",
                        }:
                            continue
                        if row[1] == "PLSQL":
                            row[5] = "ロールバック要求済み（内部COMMITは要確認）"
                        else:
                            row[5] = "ロールバック済み"
    except Exception as e:
        logger.error(f"SQL実行に失敗しました: {e}")
        logger.error(traceback.format_exc())
        s = str(e)
        df = pd.DataFrame(
            rows,
            columns=[
                "No",
                "Type",
                "Status",
                "RowsAffected",
                "Message",
                "Permanent",
                "Duration_ms",
            ],
        ) if rows else pd.DataFrame()
        info = f"❌ エラー: {s}"
        return (
            gr.Markdown(visible=True, value=info),
            gr.Dataframe(visible=True, value=df, label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "No",
            "Type",
            "Status",
            "RowsAffected",
            "Message",
            "Permanent",
            "Duration_ms",
        ],
    ) if rows else pd.DataFrame()
    succ = sum(1 for r in rows if r[2] == '成功')
    fail = sum(1 for r in rows if r[2] == '失敗')
    if ok:
        tx = "全成功・コミット済み"
    elif pure_dml:
        tx = "全体ロールバック済み"
    elif implicit_commit_seen:
        tx = "失敗箇所で停止（DDL確定分はロールバック不可）"
    else:
        tx = "失敗箇所で停止・ロールバック要求済み"
    summary = f"成功: {succ}件 / 失敗: {fail}件 ({tx})"
    if not pure_dml:
        summary = (
            "⚠️ このバッチにはDDL/PLSQLが含まれます。Oracleの暗黙COMMIT、"
            "またはPL/SQL内の明示COMMITはロールバックできません。\n\n"
            + summary
        )
    icon = "✅" if fail == 0 else "⚠️"
    return (
        gr.Markdown(visible=True, value=f"{icon} {summary}"),
        gr.Dataframe(visible=True, value=df, label="実行サマリー", elem_id="query_result_df"),
        gr.HTML(visible=False, value=""),
    )


def _execute_query_for_user(
    admin_pool,
    vpd_pool,
    username: str,
    sql: str,
    limit: int,
):
    """Route SQL through the execution policy for the authenticated user."""
    role = user_role(username)
    if role == "admin":
        return execute_sql_general(admin_pool, sql, limit)
    if role != "vpd":
        raise PermissionError("ログインユーザーを確認できません")

    validation_error = _read_only_validation_error(sql)
    if validation_error:
        return _query_error_result(validation_error)
    if vpd_pool is None:
        raise RuntimeError("VPD実行プールが設定されていません")
    return execute_select_sql(vpd_pool, sql, limit, login_user=username)


def build_query_tab(pool, vpd_pool=None):
    """クエリ実行タブのUIを構築する."""
    with gr.Accordion(label="1. SQLの入力", open=True):
        query_notice = gr.Markdown(value="", visible=False)
        with gr.Accordion(label="SQLファイル（.sql / .txt 形式をサポート）", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(" ", elem_classes="input-label")
                with gr.Column(scale=5):
                    sql_file_input = gr.File(
                        show_label=False,
                        file_types=[".sql", ".txt"],
                        type="filepath",
                    )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("SQL*", elem_classes="input-label")
            with gr.Column(scale=5):
                sql_input = gr.Textbox(
                    show_label=False,
                    placeholder="",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True,
                    container=False,
                    autoscroll=False,
                )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("取得件数*", elem_classes="input-label")
                    with gr.Column(scale=5):
                        limit_input = gr.Number(
                            show_label=False,
                            value=100,
                            minimum=1,
                            maximum=10000,
                            container=False,
                        )
            with gr.Column(scale=5):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("")

        with gr.Row():
            clear_btn = gr.Button("クリア", variant="secondary")
            execute_btn = gr.Button("実行", variant="primary")
        with gr.Row():
            result_info = gr.Markdown(visible=False)

    with gr.Accordion(label="2. 実行結果の表示", open=True):
        with gr.Row():
            result_df = gr.Dataframe(
                label="実行結果",
                interactive=False,
                wrap=True,
                visible=False,
                value=pd.DataFrame(),
                elem_id="query_result_df",
            )

        with gr.Row():
            result_style = gr.HTML(visible=False)

        with gr.Accordion(label="AI分析と処理", open=True):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("モデル*", elem_classes="input-label")
                        with gr.Column(scale=5):
                            ai_model_input = gr.Dropdown(
                                show_label=False,
                                choices=CHAT_MODEL_CHOICES,
                                value=DEFAULT_CHAT_MODEL,
                                interactive=True,
                                container=False,
                            )
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ai_analyze_btn = gr.Button("AI分析", variant="primary")
            with gr.Row():
                ai_status_md = gr.Markdown(visible=False)
            with gr.Row():
                ai_result_md = gr.Markdown(visible=False)

    async def _ai_analyze_async(model_name, sql_text, result_info_text, result_df_val=None):
        if not model_name.startswith("gpt-"):
            from utils.chat_util import get_oci_region, get_compartment_id
            region = get_oci_region()
            compartment_id = get_compartment_id()
            if not region or not compartment_id:
                return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
        try:
            import pandas as pd
            
            q = (sql_text or "").strip()
            if q.endswith(";"):
                q = q[:-1]
            info_text = str(result_info_text or "").strip()
            
            # DataFrameの内容をテキスト化
            df_text = ""
            if result_df_val is not None and isinstance(result_df_val, pd.DataFrame) and not result_df_val.empty:
                # 行数が多い場合は先頭のみを表示するなどの制限を入れる
                df_text = result_df_val.to_markdown(index=False)
            
            prompt = (
                "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                "1) エラー原因(該当する場合)\n"
                "2) 解決方法(修正案や具体的手順)\n"
                "3) 簡潔な結論(不要な詳細は省略)\n\n"
                + ("SQL:\n```sql\n" + q + "\n```\n" if q else "")
                + ("実行結果メッセージ:\n" + info_text + "\n" if info_text else "")
                + ("実行結果データ:\n" + df_text + "\n" if df_text else "")
            )
            
            messages = [
                {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                {"role": "user", "content": prompt},
            ]
            
            if model_name.startswith("gpt-"):
                from openai import AsyncOpenAI
                client = AsyncOpenAI()
                # Use Chat Completions API instead of Responses API to avoid 404 errors
                resp = await client.chat.completions.create(model=model_name, messages=messages)
                text = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    text = msg.content if hasattr(msg, "content") else ""
            elif model_name in {"google.gemini-2.5-flash", "google.gemini-2.5-pro"}:
                # Geminiモデルの場合はOCI Native SDKを使用
                from utils.chat_util import _stream_oci_genai_chat_gemini
                text = ""
                async for delta in _stream_oci_genai_chat_gemini(
                    region=region,
                    compartment_id=compartment_id,
                    model_id=model_name,
                    messages=messages,
                ):
                    text += delta
            else:
                from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                client = AsyncOciOpenAI(
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    auth=OciUserPrincipalAuth(),
                    compartment_id=compartment_id,
                )
                resp = await client.chat.completions.create(model=model_name, messages=messages)
                text = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    text = msg.content if hasattr(msg, "content") else ""
                    
            return gr.Markdown(visible=True, value=text or "分析結果が空です")
        except Exception as e:
            return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

    def ai_analyze(model_name, sql_text, result_info_text, result_df_val=None):
        import asyncio
        # 必須入力項目のチェック
        if not model_name or not str(model_name).strip():
            yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
            return
        if not sql_text or not str(sql_text).strip():
            yield gr.Markdown(visible=True, value="⚠️ SQLを入力してください"), gr.Markdown(visible=False)
            return
        
        # 実行結果メッセージもデータフレームも無い場合はエラーとする
        has_info = result_info_text and str(result_info_text).strip()
        has_df = result_df_val is not None and isinstance(result_df_val, pd.DataFrame) and not result_df_val.empty
        if not has_info and not has_df:
            yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先にSQLを実行してください"), gr.Markdown(visible=False)
            return
        
        logger.info(f"AI分析を開始します: model={model_name}, sql_length={len(str(sql_text or ''))}, result_info_length={len(str(result_info_text or ''))}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
            result_md = loop.run_until_complete(_ai_analyze_async(model_name, sql_text, result_info_text, result_df_val))
            yield gr.Markdown(visible=True, value="✅ 完了"), result_md
        except Exception as e:
            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
        finally:
            loop.close()

    def on_execute(sql, limit, request: gr.Request):
        try:
            yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"), gr.HTML(visible=False, value="")
            username = request_username(request)
            result_info, result_df, result_style = _execute_query_for_user(
                pool,
                vpd_pool,
                username,
                sql,
                limit,
            )
            yield result_info, result_df, result_style
        except Exception as e:
            yield gr.Markdown(visible=True, value=f"❌ 実行に失敗しました: {str(e)}"), gr.Dataframe(visible=False), gr.HTML(visible=False, value="")

    def on_clear():
        return ""
    
    def load_sql_file(file_path):
        """
        SQLファイルを読み込み、テキストボックスに表示する.
        
        Args:
            file_path: アップロードされたファイルのパス
        
        Returns:
            ファイルの内容(文字列)
        """
        if not file_path:
            return ""
        
        try:
            # 複数のエンコーディングで試行
            encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1', 'euc-jp']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"SQLファイルを{encoding}で読み込みました: {file_path}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                logger.error(f"SQLファイルの読み込みに失敗しました: {file_path}")
                return "❌ エラー: ファイルの読み込みに失敗しました。エンコーディングを確認してください。"
            
            return content
            
        except Exception as e:
            logger.error(f"SQLファイルの読み込み中にエラーが発生しました: {e}")
            logger.error(traceback.format_exc())
            return f"❌ エラー: {str(e)}"

    execute_btn.click(
        fn=on_execute,
        inputs=[sql_input, limit_input],
        outputs=[result_info, result_df, result_style],
    )

    sql_file_input.change(
        fn=load_sql_file,
        inputs=[sql_file_input],
        outputs=[sql_input],
    )

    ai_analyze_btn.click(
        fn=ai_analyze,
        inputs=[ai_model_input, sql_input, result_info, result_df],
        outputs=[ai_status_md, ai_result_md],
    )

    clear_btn.click(
        fn=on_clear,
        outputs=[sql_input],
    )

    return query_notice


def _fetch_dbms_output(cursor, batch: int = 1000) -> str:
    try:
        lines = []
        while True:
            lv = cursor.arrayvar(oracledb.STRING, batch)
            nv = cursor.var(oracledb.NUMBER)
            nv.setvalue(0, batch)
            cursor.callproc("dbms_output.get_lines", [lv, nv])
            n = int(nv.getvalue(0) or 0)
            arr = lv.getvalue() or []
            if n > 0:
                lines.extend([str(x) for x in arr[:n] if x])
                if n < batch:
                    break
            else:
                break
        return "\n".join(lines)
    except Exception:
        return ""
