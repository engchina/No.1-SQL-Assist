import logging
import json
import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def _to_plain(v):
    try:
        return v.read() if hasattr(v, "read") else str(v)
    except Exception:
        return str(v)

def _maybe_json_text(s: str) -> str:
    t = str(s or "").strip()
    if not t:
        return ""
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            if "reply" in obj and isinstance(obj["reply"], str):
                return obj["reply"]
            if "content" in obj and isinstance(obj["content"], str):
                return obj["content"]
        if isinstance(obj, list):
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return t

def build_selectai_agent_tab(pool):
    with gr.Tabs():
        with gr.TabItem(label="エージェント実行"):
            with gr.Accordion(label="1. 入力", open=True):
                team_name_input = gr.Textbox(label="Team 名", placeholder="例: RETURNS_TEAM")
                prompt_input = gr.Textbox(label="プロンプト", lines=3, max_lines=10, show_copy_button=True)
                execute_btn = gr.Button("実行", variant="primary")

            with gr.Accordion(label="2. 応答", open=True):
                agent_reply_md = gr.Markdown(visible=False)
                raw_output_text = gr.Textbox(label="RAW", visible=False, lines=8, max_lines=15, interactive=False)

            def _run_team(team_name, prompt):
                tn = str(team_name or "").strip()
                q = str(prompt or "").strip()
                if not tn:
                    return gr.Markdown(visible=True, value="⚠️ Team 名を入力してください"), gr.Textbox(visible=False, value="")
                if not q:
                    return gr.Markdown(visible=True, value="⚠️ プロンプトを入力してください"), gr.Textbox(visible=False, value="")
                try:
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT DBMS_CLOUD_AI_AGENT.RUN_TEAM(team_name => :tm, prompt => :p) FROM DUAL", tm=tn, p=q)
                            rows = cursor.fetchmany(size=200) or []
                            cells = []
                            for r in rows:
                                for v in r:
                                    s = _to_plain(v)
                                    if s:
                                        cells.append(s)
                            text = "\n".join(cells)
                            reply = _maybe_json_text(text)
                            return gr.Markdown(visible=True, value=reply or ""), gr.Textbox(visible=bool(text), value=text)
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ 実行に失敗しました: {e}"), gr.Textbox(visible=True, value=str(e))

            execute_btn.click(fn=_run_team, inputs=[team_name_input, prompt_input], outputs=[agent_reply_md, raw_output_text])

        with gr.TabItem(label="会話履歴"):
            with gr.Accordion(label="1. 取得", open=True):
                hist_team_name_input = gr.Textbox(label="Team 名", placeholder="例: RETURNS_TEAM")
                hist_fetch_btn = gr.Button("最新を取得", variant="primary")
            with gr.Accordion(label="2. 表示", open=True):
                hist_df = gr.Dataframe(label="USER_CLOUD_AI_CONVERSATION_PROMPTS", interactive=False, wrap=True, visible=False, value=pd.DataFrame())
                hist_info = gr.Markdown(visible=False)

            def _fetch_history(team_name):
                tn = str(team_name or "").strip()
                try:
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            stmt = "SELECT * FROM USER_CLOUD_AI_CONVERSATION_PROMPTS" + (" WHERE TEAM_NAME = :tm" if tn else "") + " ORDER BY PROMPT_ID DESC FETCH FIRST 50 ROWS ONLY"
                            if tn:
                                cursor.execute(stmt, tm=tn)
                            else:
                                cursor.execute(stmt)
                            rows = cursor.fetchall() or []
                            cols = [d[0] for d in cursor.description] if cursor.description else []
                            def _plain_row(row):
                                return [_to_plain(v) for v in row]
                            df = pd.DataFrame([_plain_row(r) for r in rows], columns=cols)
                            return gr.Dataframe(visible=True, value=df), gr.Markdown(visible=False)
                except Exception as e:
                    return gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}")

            hist_fetch_btn.click(fn=_fetch_history, inputs=[hist_team_name_input], outputs=[hist_df, hist_info])

        with gr.TabItem(label="権限チェック"):
            with gr.Accordion(label="1. 実行", open=True):
                priv_check_btn = gr.Button("権限を確認", variant="primary")
                priv_result_md = gr.Markdown(visible=False)

            def _check_priv():
                try:
                    with pool.acquire() as conn:
                        with conn.cursor() as cursor:
                            ok = True
                            try:
                                cursor.execute("DECLARE x CLOB; BEGIN x := DBMS_CLOUD_AI_AGENT.RUN_TEAM(team_name => '___', prompt => 'ping'); END;")
                            except Exception as e:
                                msg = str(e)
                                if "PLS-00201" in msg or "ORA-06550" in msg:
                                    ok = False
                            if ok:
                                return gr.Markdown(visible=True, value="✅ DBMS_CLOUD_AI_AGENT を実行可能です")
                            return gr.Markdown(visible=True, value="⚠️ DBMS_CLOUD_AI_AGENT の権限が不足している可能性があります。管理者に EXECUTE 権限の付与を依頼してください")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ チェックに失敗しました: {e}")

            priv_check_btn.click(fn=_check_priv, outputs=[priv_result_md])

