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


def _split_sql_statements(sql: str):
    if not sql:
        return []
    s = str(sql)
    stmts = []
    buf = []
    in_s = False
    in_d = False
    in_lc = False
    in_bc = False
    pl = 0
    i = 0
    L = len(s)
    def ahead_word(j):
        k = j
        while k < L and s[k].isspace():
            k += 1
        w = []
        while k < L and (s[k].isalpha() or s[k] == '_'):
            w.append(s[k])
            k += 1
        return ''.join(w).lower(), k
    while i < L:
        ch = s[i]
        nxt = s[i+1] if i + 1 < L else ''
        if in_lc:
            buf.append(ch)
            if ch == '\n':
                in_lc = False
            i += 1
            continue
        if in_bc:
            buf.append(ch)
            if ch == '*' and nxt == '/':
                buf.append(nxt)
                in_bc = False
                i += 2
            else:
                i += 1
            continue
        if not in_s and not in_d:
            if ch == '-' and nxt == '-':
                buf.append(ch)
                buf.append(nxt)
                in_lc = True
                i += 2
                continue
            if ch == '/' and nxt == '*':
                buf.append(ch)
                buf.append(nxt)
                in_bc = True
                i += 2
                continue
        if ch == "'" and not in_d:
            buf.append(ch)
            if in_s:
                pk = s[i+1] if i + 1 < L else ''
                if pk == "'":
                    buf.append(pk)
                    i += 2
                    continue
                in_s = False
                i += 1
            else:
                in_s = True
                i += 1
            continue
        if ch == '"' and not in_s:
            buf.append(ch)
            in_d = not in_d
            i += 1
            continue
        if not in_s and not in_d:
            if ch.isalpha():
                w, k = ahead_word(i)
                if w in ('begin', 'declare'):
                    pl += 1
                elif w == 'end':
                    pass
                i = k
                buf.append(s[i-len(w):i])
                continue
            if ch == ';' and pl == 0:
                st = ''.join(buf).strip()
                if st:
                    stmts.append(st)
                buf = []
                i += 1
                continue
            if ch == ';' and pl > 0:
                js = ''.join(buf)
                m = re.search(r"\bend\s*$", js, flags=re.IGNORECASE)
                if m:
                    pl = max(0, pl - 1)
                buf.append(ch)
                i += 1
                continue
        buf.append(ch)
        i += 1
    tail = ''.join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def _normalize_exec(stmt: str) -> str:
    s = str(stmt or '').strip()
    if re.match(r"^(exec|execute)\b", s, flags=re.IGNORECASE):
        body = re.sub(r"^(exec|execute)\s+", "", s, flags=re.IGNORECASE).strip()
        if body.endswith(';'):
            body = body[:-1]
        return f"BEGIN {body}; END;"
    return s


def _stmt_type(stmt: str) -> str:
    s = str(stmt or '').strip()
    def strip_comments(x: str) -> str:
        i = 0
        L = len(x)
        while True:
            while i < L and x[i].isspace():
                i += 1
            if i + 1 < L and x[i] == '-' and x[i+1] == '-':
                i += 2
                while i < L and x[i] != '\n':
                    i += 1
                continue
            if i + 1 < L and x[i] == '/' and x[i+1] == '*':
                i += 2
                while i + 1 < L and not (x[i] == '*' and x[i+1] == '/'):
                    i += 1
                i = i + 2 if i + 1 < L else L
                continue
            break
        return x[i:]
    s = strip_comments(s)
    m = re.match(r"^comment\s+on\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", s, flags=re.IGNORECASE)
    if m:
        # tgt = m.group(1).upper()
        return f"COMMENT"
    if re.match(r"^(select|with)\b", s, flags=re.IGNORECASE):
        return 'SELECT'
    for k in ('insert', 'update', 'delete', 'merge', 'create', 'drop', 'alter', 'truncate', 'grant', 'revoke'):
        if re.match(rf"^{k}\b", s, flags=re.IGNORECASE):
            return k.upper()
    if re.match(r"^(begin|declare)\b", s, flags=re.IGNORECASE):
        return 'PLSQL'
    if re.match(r"^(exec|execute)\b", s, flags=re.IGNORECASE):
        return 'PLSQL'
    return 'UNKNOWN'


def execute_sql_general(pool, sql: str, limit: int):
    if not sql or not str(sql).strip():
        gr.Warning("SQLを入力してください")
        return (
            gr.Markdown(visible=True),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    statements = _split_sql_statements(sql)
    statements = [s for s in statements if s and s.strip()]
    if not statements:
        gr.Warning("SQLを入力してください")
        return (
            gr.Markdown(visible=True),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    types = [_stmt_type(s) for s in statements]
    sel_count = sum(1 for t in types if t == 'SELECT')
    if len(statements) == 1 and sel_count == 1:
        return execute_select_sql(pool, statements[0], limit)
    if len(statements) > 1 and sel_count > 0:
        gr.Error("複数実行時はSELECT文を含めることはできません")
        return (
            gr.Markdown(visible=True, value="❌ エラー: 複数実行にSELECTは含められません"),
            gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    import time
    rows = []
    ok = True
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.callproc("dbms_output.enable")
                except Exception:
                    pass
                for idx, st in enumerate(statements, start=1):
                    typ = _stmt_type(st)
                    run = _normalize_exec(st)
                    t0 = time.perf_counter()
                    try:
                        cursor.execute(run)
                        rc = cursor.rowcount if hasattr(cursor, 'rowcount') else None
                        dur = int((time.perf_counter() - t0) * 1000)
                        is_dml = typ in ('INSERT', 'UPDATE', 'DELETE', 'MERGE')
                        is_plsql = typ == 'PLSQL'
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
                        rows.append([idx, typ, '成功', rc if rc is not None else -1, msg, dur])
                    except Exception as e:
                        ok = False
                        dur = int((time.perf_counter() - t0) * 1000)
                        msg = str(e)
                        rows.append([idx, typ, '失敗', -1, msg, dur])
                        break
                if ok:
                    conn.commit()
                else:
                    conn.rollback()
    except Exception as e:
        s = str(e)
        df = pd.DataFrame(rows, columns=["No", "Type", "Status", "RowsAffected", "Message", "Duration_ms"]) if rows else pd.DataFrame()
        info = f"❌ エラー: {s}"
        return (
            gr.Markdown(visible=True, value=info),
            gr.Dataframe(visible=True, value=df, label="実行結果", elem_id="query_result_df"),
            gr.HTML(visible=False, value=""),
        )
    df = pd.DataFrame(rows, columns=["No", "Type", "Status", "RowsAffected", "Message", "Duration_ms"]) if rows else pd.DataFrame()
    succ = sum(1 for r in rows if r[2] == '成功')
    fail = sum(1 for r in rows if r[2] == '失敗')
    tx = "コミット済み" if ok else "ロールバック済み"
    gr.Info(f"成功: {succ}件 / 失敗: {fail}件 ({tx})")
    return (
        gr.Markdown(visible=False),
        gr.Dataframe(visible=True, value=df, label="実行サマリー", elem_id="query_result_df"),
        gr.HTML(visible=False, value=""),
    )


def build_query_tab(pool):
    """クエリ実行タブのUIを構築する."""
    with gr.TabItem(label="SQLの実行") as tab_query:
        with gr.Accordion(label="1. SQLの入力", open=True):
            sql_input = gr.Textbox(
                label="SQL文（SELECTは1文のみ、その他は複数文同時実行可）\n注意: 複数実行時はSELECTを含めないでください",
                placeholder="複数の文はセミコロンで区切って入力できます。\n例: INSERT/UPDATE/DELETE/MERGE/CREATE/COMMENT/BEGIN..END/EXEC など。SELECTは1回に1文のみ",
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
                value="ℹ️ SELECTは1文のみ実行可能。INSERT/UPDATE/DELETE/MERGE/CREATE/COMMENT/PL/SQL/EXECは複数文をセミコロンで区切って同時実行可能。複数実行時はSELECTを含めないでください",
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

            with gr.Accordion(label="AI分析と処理", open=False):
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

        async def _ai_analyze_async(model_name, sql_text, result_info_text, result_df_input):
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
                info_text = str(result_info_text or "").strip()
                prompt = (
                    "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                    "1) エラー原因（該当する場合）\n"
                    "2) 解決方法（修正案や具体的手順）\n"
                    "3) 簡潔な結論（不要な詳細は省略）\n\n"
                    + ("SQL:\n```sql\n" + q + "\n```\n" if q else "")
                    + ("実行メッセージ:\n" + info_text + "\n" if info_text else "")
                    + ("結果プレビュー:\n" + preview + "\n" if preview else "")
                )
                client = AsyncOciOpenAI(
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    auth=OciUserPrincipalAuth(),
                    compartment_id=compartment_id,
                )
                messages = [
                    {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
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

        def ai_analyze(model_name, sql_text, result_info_text, result_df_input):
            import asyncio
            logger.info(f"AI分析を開始します: model={model_name}, sql_length={len(str(sql_text or ''))}, result_info_length={len(str(result_info_text or ''))}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                result_md = loop.run_until_complete(_ai_analyze_async(model_name, sql_text, result_info_text, result_df_input))
                yield gr.Markdown(visible=True, value="✅ 完了"), result_md
            except Exception as e:
                yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
            finally:
                loop.close()

        def on_execute(sql, limit):
            try:
                yield gr.Markdown(visible=True, value="⏳ 実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(), label="実行結果", elem_id="query_result_df"), gr.HTML(visible=False, value="")
                result_info, result_df, result_style = execute_sql_general(pool, sql, limit)
                yield result_info, result_df, result_style
            except Exception as e:
                yield gr.Markdown(visible=True, value=f"❌ 実行に失敗しました: {str(e)}"), gr.Dataframe(visible=False), gr.HTML(visible=False, value="")

        def on_clear():
            return ""

        execute_btn.click(
            fn=on_execute,
            inputs=[sql_input, limit_input],
            outputs=[result_info, result_df, result_style],
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

    return tab_query
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
