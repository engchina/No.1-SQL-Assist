"""SQL学習タブユーティリティ.

初心者〜中級者向けに、Oracle SQLのSELECT中心の学習をステップバイステップで行えるUIを提供します。
表・ビューの作成、初期データ投入、サブクエリ/CTE(WITH)/JOIN/WHERE/集約関数などの実行と結果表示を一つの画面で体験できます。

Args:
    pool: 遅延初期化されたOracle接続プール
"""

import logging
from typing import List, Dict, Tuple

import gradio as gr
import pandas as pd

from utils.query_util import execute_sql_general, execute_select_sql

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _schema_sql() -> Tuple[str, str]:
    """学習用の表作成SQLとビュー作成SQLを返す.

    Returns:
        tuple[str, str]: (tables_sql, views_sql)
    """
    tables_sql = (
"""
-- 部門テーブル（日本の企業で分かりやすい業務ドメイン）
CREATE TABLE DEPARTMENT (
    DEPARTMENT_ID NUMBER PRIMARY KEY,
    DEPARTMENT_NAME VARCHAR2(50) NOT NULL,
    LOCATION VARCHAR2(50),
    CREATED_AT DATE DEFAULT SYSDATE
);

-- 社員テーブル（社員は部門に所属）
CREATE TABLE EMPLOYEE (
    EMPLOYEE_ID NUMBER PRIMARY KEY,
    DEPARTMENT_ID NUMBER NOT NULL,
    EMPLOYEE_NAME VARCHAR2(100) NOT NULL,
    EMAIL VARCHAR2(100),
    HIRE_DATE DATE DEFAULT SYSDATE,
    SALARY NUMBER(10,2),
    CONSTRAINT FK_EMP_DEPT FOREIGN KEY (DEPARTMENT_ID)
        REFERENCES DEPARTMENT(DEPARTMENT_ID)
);

-- プロジェクトテーブル（プロジェクトは部門が担当）
CREATE TABLE PROJECT (
    PROJECT_ID NUMBER PRIMARY KEY,
    DEPARTMENT_ID NUMBER NOT NULL,
    PROJECT_NAME VARCHAR2(100) NOT NULL,
    START_DATE DATE,
    BUDGET NUMBER(12,2),
    CONSTRAINT FK_PROJ_DEPT FOREIGN KEY (DEPARTMENT_ID)
        REFERENCES DEPARTMENT(DEPARTMENT_ID)
);
"""
    ).strip()

    views_sql = (
"""
-- 社員と部門のビュー
CREATE OR REPLACE VIEW V_EMP_DEPT AS
SELECT e.EMPLOYEE_ID, e.EMPLOYEE_NAME, e.SALARY,
        d.DEPARTMENT_NAME, d.LOCATION
    FROM EMPLOYEE e
    JOIN DEPARTMENT d
    ON e.DEPARTMENT_ID = d.DEPARTMENT_ID;

-- 部門とプロジェクトのビュー
CREATE OR REPLACE VIEW V_DEPT_PROJECT AS
SELECT p.PROJECT_ID, p.PROJECT_NAME, p.BUDGET,
        d.DEPARTMENT_NAME
    FROM PROJECT p
    JOIN DEPARTMENT d
    ON p.DEPARTMENT_ID = d.DEPARTMENT_ID;
"""
    ).strip()

    return tables_sql, views_sql


def _insert_sql() -> Dict[str, str]:
    """各テーブルの初期データ投入用INSERT SQLを返す.

    Returns:
        dict[str, str]: {table_name: inserts_sql}
    """
    dep_inserts = (
"""
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (10, '総務', '東京');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (20, '経理', '東京');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (30, '人事', '大阪');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (40, '営業', '名古屋');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (50, '開発', '福岡');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (60, 'サポート', '札幌');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (70, '企画', '仙台');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (80, 'マーケティング', '神戸');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (90, '品質保証', '京都');
INSERT INTO DEPARTMENT (DEPARTMENT_ID, DEPARTMENT_NAME, LOCATION) VALUES (100, '法務', '横浜');
"""
    ).strip()

    emp_inserts = (
"""
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (1, 10, '佐藤 太郎', 'taro.sato@example.com', 420000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (2, 20, '鈴木 花子', 'hanako.suzuki@example.com', 510000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (3, 30, '高橋 健', 'ken.takahashi@example.com', 480000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (4, 40, '田中 美咲', 'misaki.tanaka@example.com', 550000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (5, 50, '伊藤 直樹', 'naoki.ito@example.com', 600000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (6, 50, '渡辺 真央', 'mao.watanabe@example.com', 620000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (7, 40, '山本 大輔', 'daisuke.yamamoto@example.com', 530000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (8, 30, '中村 さくら', 'sakura.nakamura@example.com', 470000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (9, 20, '小林 翔', 'sho.kobayashi@example.com', 520000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (10, 10, '加藤 恵', 'megumi.kato@example.com', 450000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (11, 60, '吉田 光', 'hikari.yoshida@example.com', 400000);
INSERT INTO EMPLOYEE (EMPLOYEE_ID, DEPARTMENT_ID, EMPLOYEE_NAME, EMAIL, SALARY) VALUES (12, 70, '佐々木 蓮', 'ren.sasaki@example.com', 460000);
"""
    ).strip()

    proj_inserts = (
"""
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (101, 50, '受注管理システム', DATE '2025-04-01', 15000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (102, 40, '新製品販売強化', DATE '2025-05-01', 8000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (103, 20, '会計自動化', DATE '2025-02-01', 6000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (104, 10, '社内ポータル刷新', DATE '2025-01-15', 3000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (105, 60, '顧客サポート改善', DATE '2025-03-10', 5000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (106, 70, '市場調査強化', DATE '2025-06-01', 4000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (107, 90, '品質監査強化', DATE '2025-07-01', 3500000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (108, 50, '開発基盤整備', DATE '2025-03-01', 12000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (109, 40, '大口顧客開拓', DATE '2025-04-15', 7000000);
INSERT INTO PROJECT (PROJECT_ID, DEPARTMENT_ID, PROJECT_NAME, START_DATE, BUDGET) VALUES (110, 30, '採用強化', DATE '2025-02-20', 2000000);
"""
    ).strip()

    return {
        "DEPARTMENT": dep_inserts,
        "EMPLOYEE": emp_inserts,
        "PROJECT": proj_inserts,
    }


def _lessons() -> List[Dict[str, str]]:
    """SELECT学習用のレッスン定義を返す.

    Returns:
        list[dict]: 各レッスンの {id, title, desc, sql}
    """
    lessons: List[Dict[str, str]] = [
        {
            "id": "L01",
            "title": "SELECTの基本（1表）",
            "desc": "社員テーブルから基本的な列を取得します。",
            "sql": "SELECT EMPLOYEE_ID, EMPLOYEE_NAME, SALARY FROM EMPLOYEE ORDER BY EMPLOYEE_ID FETCH FIRST 10 ROWS ONLY;",
        },
        {
            "id": "L02",
            "title": "WHERE条件",
            "desc": "給与が50万円以上の社員を抽出します。",
            "sql": "SELECT EMPLOYEE_NAME, SALARY FROM EMPLOYEE WHERE SALARY >= 500000 ORDER BY SALARY DESC;",
        },
        {
            "id": "L03",
            "title": "LIKEとUPPER",
            "desc": "社員名に'佐'を含むレコード（大文字小文字を正規化）。",
            "sql": "SELECT EMPLOYEE_NAME FROM EMPLOYEE WHERE UPPER(EMPLOYEE_NAME) LIKE UPPER('%佐%');",
        },
        {
            "id": "L04",
            "title": "JOIN（社員×部門）",
            "desc": "社員と部門を結合し、部署名付きで表示します。",
            "sql": "SELECT e.EMPLOYEE_NAME, d.DEPARTMENT_NAME, e.SALARY FROM EMPLOYEE e JOIN DEPARTMENT d ON e.DEPARTMENT_ID = d.DEPARTMENT_ID ORDER BY d.DEPARTMENT_ID, e.EMPLOYEE_ID;",
        },
        {
            "id": "L05",
            "title": "集約（COUNT/SUM）",
            "desc": "部門ごとの社員数と総給与を集計します。",
            "sql": "SELECT d.DEPARTMENT_NAME, COUNT(*) AS 人数, SUM(e.SALARY) AS 総給与 FROM EMPLOYEE e JOIN DEPARTMENT d ON e.DEPARTMENT_ID = d.DEPARTMENT_ID GROUP BY d.DEPARTMENT_NAME ORDER BY 人数 DESC;",
        },
        {
            "id": "L06",
            "title": "AVGとHAVING",
            "desc": "平均給与が50万円以上の部門のみ表示します。",
            "sql": "SELECT d.DEPARTMENT_NAME, AVG(e.SALARY) AS 平均給与 FROM EMPLOYEE e JOIN DEPARTMENT d ON e.DEPARTMENT_ID = d.DEPARTMENT_ID GROUP BY d.DEPARTMENT_NAME HAVING AVG(e.SALARY) >= 500000 ORDER BY 平均給与 DESC;",
        },
        {
            "id": "L07",
            "title": "ビューの利用（V_EMP_DEPT）",
            "desc": "ビューを使って社員と部門を簡潔に参照します。",
            "sql": "SELECT EMPLOYEE_NAME, DEPARTMENT_NAME, SALARY FROM V_EMP_DEPT ORDER BY DEPARTMENT_NAME, EMPLOYEE_NAME;",
        },
        {
            "id": "L08",
            "title": "サブクエリ（平均より高い給与）",
            "desc": "全体の平均給与より高い社員を抽出します。",
            "sql": "SELECT EMPLOYEE_NAME, SALARY FROM EMPLOYEE WHERE SALARY > (SELECT AVG(SALARY) FROM EMPLOYEE) ORDER BY SALARY DESC;",
        },
        {
            "id": "L09",
            "title": "相関サブクエリ（部門平均より高い）",
            "desc": "各社員の所属部門の平均給与より高い人を抽出します。",
            "sql": "SELECT e.EMPLOYEE_NAME, e.SALARY, d.DEPARTMENT_NAME FROM EMPLOYEE e JOIN DEPARTMENT d ON e.DEPARTMENT_ID = d.DEPARTMENT_ID WHERE e.SALARY > (SELECT AVG(e2.SALARY) FROM EMPLOYEE e2 WHERE e2.DEPARTMENT_ID = e.DEPARTMENT_ID) ORDER BY e.SALARY DESC;",
        },
        {
            "id": "L10",
            "title": "EXISTS（プロジェクトを持つ部門）",
            "desc": "少なくとも1件のプロジェクトがある部門を抽出します。",
            "sql": "SELECT d.DEPARTMENT_NAME FROM DEPARTMENT d WHERE EXISTS (SELECT 1 FROM PROJECT p WHERE p.DEPARTMENT_ID = d.DEPARTMENT_ID) ORDER BY d.DEPARTMENT_NAME;",
        },
        {
            "id": "L11",
            "title": "LEFT JOIN（プロジェクトがない部門）",
            "desc": "プロジェクトが存在しない部門を抽出します。",
            "sql": "SELECT d.DEPARTMENT_NAME FROM DEPARTMENT d LEFT JOIN PROJECT p ON p.DEPARTMENT_ID = d.DEPARTMENT_ID WHERE p.PROJECT_ID IS NULL ORDER BY d.DEPARTMENT_NAME;",
        },
        {
            "id": "L12",
            "title": "WITH句（CTE）",
            "desc": "CTEで部門平均給与を計算して社員と結合します。",
            "sql": (
                "WITH dept_avg AS (\n"
                "  SELECT DEPARTMENT_ID, AVG(SALARY) AS AVG_SAL\n"
                "    FROM EMPLOYEE\n"
                "    GROUP BY DEPARTMENT_ID\n"
                ")\n"
                "SELECT e.EMPLOYEE_NAME, d.DEPARTMENT_NAME, e.SALARY, da.AVG_SAL\n"
                "  FROM EMPLOYEE e\n"
                "  JOIN DEPARTMENT d ON e.DEPARTMENT_ID = d.DEPARTMENT_ID\n"
                "  LEFT JOIN dept_avg da ON da.DEPARTMENT_ID = d.DEPARTMENT_ID\n"
                "  ORDER BY d.DEPARTMENT_NAME, e.SALARY DESC;"
            ),
        },
        {
            "id": "L13",
            "title": "ORDER BYとFETCH",
            "desc": "給与が高い上位5名のみ取得します。",
            "sql": "SELECT EMPLOYEE_NAME, SALARY FROM EMPLOYEE ORDER BY SALARY DESC FETCH FIRST 5 ROWS ONLY;",
        },
        {
            "id": "L14",
            "title": "DISTINCT",
            "desc": "部門の一覧（重複なし）を取得します。",
            "sql": "SELECT DISTINCT DEPARTMENT_NAME FROM DEPARTMENT ORDER BY DEPARTMENT_NAME;",
        },
        {
            "id": "L15",
            "title": "ビュー活用（V_DEPT_PROJECT）",
            "desc": "部門とプロジェクトの一覧を表示します。",
            "sql": "SELECT DEPARTMENT_NAME, PROJECT_NAME, BUDGET FROM V_DEPT_PROJECT ORDER BY DEPARTMENT_NAME, PROJECT_NAME;",
        },
        {
            "id": "L16",
            "title": "日付関数（今年開始のプロジェクト）",
            "desc": "当年開始のプロジェクトのみを抽出します。",
            "sql": "SELECT PROJECT_NAME, START_DATE, BUDGET FROM PROJECT WHERE EXTRACT(YEAR FROM START_DATE) = EXTRACT(YEAR FROM SYSDATE) ORDER BY START_DATE;",
        },
        {
            "id": "L17",
            "title": "CASE式（給与帯ラベル）",
            "desc": "給与額に応じてS/M/Lのラベルを付けます。",
            "sql": "SELECT EMPLOYEE_NAME, SALARY, CASE WHEN SALARY >= 600000 THEN 'S' WHEN SALARY >= 500000 THEN 'M' ELSE 'L' END AS 給与帯 FROM EMPLOYEE ORDER BY SALARY DESC;",
        },
        {
            "id": "L18",
            "title": "日本拠点のみ（LOCATIONでフィルタ）",
            "desc": "東京や大阪など主要拠点の部門のみを抽出します。",
            "sql": "SELECT DEPARTMENT_NAME, LOCATION FROM DEPARTMENT WHERE LOCATION IN ('東京','大阪','名古屋','福岡') ORDER BY LOCATION, DEPARTMENT_NAME;",
        },
        {
            "id": "L19",
            "title": "部門別予算合計（プロジェクト集約）",
            "desc": "部門ごとにプロジェクト予算の合計を確認します。",
            "sql": "SELECT d.DEPARTMENT_NAME, SUM(p.BUDGET) AS 部門予算合計 FROM DEPARTMENT d JOIN PROJECT p ON d.DEPARTMENT_ID = p.DEPARTMENT_ID GROUP BY d.DEPARTMENT_NAME ORDER BY 部門予算合計 DESC;",
        },
        {
            "id": "L20",
            "title": "WITH句＋サブクエリの組み合わせ",
            "desc": "高コストプロジェクトを持つ部門と、部門内平均給与を同時に確認します。",
            "sql": (
                "WITH high_budget AS (\n"
                "  SELECT DEPARTMENT_ID, SUM(BUDGET) AS TOTAL_BUDGET\n"
                "    FROM PROJECT\n"
                "    GROUP BY DEPARTMENT_ID\n"
                "),\n"
                "dept_avg AS (\n"
                "  SELECT DEPARTMENT_ID, AVG(SALARY) AS AVG_SAL\n"
                "    FROM EMPLOYEE\n"
                "    GROUP BY DEPARTMENT_ID\n"
                ")\n"
                "SELECT d.DEPARTMENT_NAME, hb.TOTAL_BUDGET, da.AVG_SAL\n"
                "  FROM DEPARTMENT d\n"
                "  LEFT JOIN high_budget hb ON d.DEPARTMENT_ID = hb.DEPARTMENT_ID\n"
                "  LEFT JOIN dept_avg da ON d.DEPARTMENT_ID = da.DEPARTMENT_ID\n"
                "  WHERE hb.TOTAL_BUDGET IS NOT NULL\n"
                "  ORDER BY hb.TOTAL_BUDGET DESC;"
            ),
        },
    ]
    return lessons


def build_sql_learning_tab(pool):
    """SQL学習タブのUIを構築する.

    Args:
        pool: 遅延初期化されたOracle接続プール
    """
    tables_sql, views_sql = _schema_sql()
    inserts = _insert_sql()
    lessons = _lessons()

    with gr.TabItem(label="SQL学習"):
        with gr.Accordion(label="1. 学習用スキーマの準備", open=True):
            schema_help_md = gr.Markdown(
                value=(
                    "ℹ️ このセクションでは学習用の3つの表（DEPARTMENT/EMPLOYEE/PROJECT）と2つのビュー（V_EMP_DEPT/V_DEPT_PROJECT）を作成し、\n\n"
                    "ℹ️ サンプルデータを投入します。各ステップは『SQLの表示』→『実行』の2段階です。"
                ),
                visible=True,
            )

            with gr.Row():
                with gr.Column():
                    show_tables_btn = gr.Button("表のSQLを表示", variant="secondary")
                with gr.Column():
                    exec_tables_btn = gr.Button("表を作成", variant="primary")
            # 表示状態を追跡するState変数
            tables_sql_visible_state = gr.State(value=False)
            with gr.Row():
                tables_sql_text = gr.Textbox(label="表作成SQL", value=tables_sql, lines=10, max_lines=20, interactive=False, show_copy_button=True, visible=False, elem_id="sql_learning_tables_sql")
            with gr.Row():
                    tables_result_md = gr.Markdown(visible=False)

            with gr.Row():
                with gr.Column():
                    show_views_btn = gr.Button("ビューのSQLを表示", variant="secondary")
                with gr.Column():
                    exec_views_btn = gr.Button("ビューを作成", variant="primary")
            # 表示状態を追跡するState変数
            views_sql_visible_state = gr.State(value=False)
            with gr.Row():
                views_sql_text = gr.Textbox(label="ビュー作成SQL", value=views_sql, lines=8, max_lines=20, interactive=False, show_copy_button=True, visible=False, elem_id="sql_learning_views_sql")
            with gr.Row():
                views_result_md = gr.Markdown(visible=False)

            with gr.Row():
                with gr.Column():
                    show_inserts_btn = gr.Button("INSERTのSQLを表示", variant="secondary")
                with gr.Column():
                    exec_inserts_btn = gr.Button("データを投入", variant="primary")
            # 表示状態を追跡するState変数
            inserts_sql_visible_state = gr.State(value=False)
            with gr.Row():
                inserts_sql_text = gr.Textbox(label="INSERT SQL", value="", lines=8, max_lines=20, interactive=False, show_copy_button=True, visible=False, elem_id="sql_learning_insert_sql")
            with gr.Row():
                inserts_result_md = gr.Markdown(visible=False)

            with gr.Row():
                reset_btn = gr.Button("初期化（ドロップ）", variant="primary")
            with gr.Row():
                reset_result_md = gr.Markdown(visible=False)

        with gr.Accordion(label="2. SELECTの学習（ステップ）", open=True):
            lessons_df = pd.DataFrame([{k: l[k] for k in ("id", "title", "desc")} for l in lessons])
            # デフォルトレッスン（L01）の情報を取得
            default_lesson = lessons[0]
            default_lesson_text = f"【{default_lesson['id']}】{default_lesson['title']}\n\n{default_lesson['desc']}"
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("レッスン", elem_classes="input-label")
                with gr.Column(scale=5):
                    lesson_select = gr.Dropdown(
                        show_label=False,
                        choices=[f"{l['id']} - {l['title']}" for l in lessons],
                        value=f"{lessons[0]['id']} - {lessons[0]['title']}",
                        container=False,
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("レッスンの説明", elem_classes="input-label")
                with gr.Column(scale=5):
                    lesson_desc_md = gr.Markdown(visible=True, value=default_lesson_text)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("学習SQL", elem_classes="input-label")
                with gr.Column(scale=5):
                    lesson_sql_text = gr.Textbox(show_label=False, lines=6, max_lines=15, interactive=True, show_copy_button=True, autoscroll=False, value=default_lesson['sql'], container=False)

            with gr.Row():
                run_lesson_btn = gr.Button("このSQLを実行", variant="primary")
            lesson_result_info = gr.Markdown(visible=False)
            lesson_result_df = gr.Dataframe(label="実行結果", interactive=False, wrap=True, visible=False, value=pd.DataFrame())
            lesson_result_style = gr.HTML(visible=False)

        def _show_tables(current_visible):
            """表のSQL表示を切り替える."""
            new_visible = not current_visible
            new_label = "表のSQLを非表示" if new_visible else "表のSQLを表示"
            return gr.Button(value=new_label), gr.Textbox(visible=new_visible), new_visible

        def _exec_tables():
            try:
                info_md, df_comp, style_html = execute_sql_general(pool, tables_sql, limit=0)
                return info_md
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ 表作成に失敗しました: {e}")

        def _show_views(current_visible):
            """ビューのSQL表示を切り替える."""
            new_visible = not current_visible
            new_label = "ビューのSQLを非表示" if new_visible else "ビューのSQLを表示"
            return gr.Button(value=new_label), gr.Textbox(visible=new_visible), new_visible

        def _exec_views():
            try:
                info_md, df_comp, style_html = execute_sql_general(pool, views_sql, limit=0)
                return info_md
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ ビュー作成に失敗しました: {e}")

        def _show_inserts(current_visible):
            """INSERTのSQL表示を切り替える."""
            sql = (inserts["DEPARTMENT"] + "\n" + inserts["EMPLOYEE"] + "\n" + inserts["PROJECT"]).strip()
            new_visible = not current_visible
            new_label = "INSERTのSQLを非表示" if new_visible else "INSERTのSQLを表示"
            return gr.Button(value=new_label), gr.Textbox(visible=new_visible, value=sql if new_visible else ""), new_visible

        def _exec_inserts():
            try:
                sql = (inserts["DEPARTMENT"] + "\n" + inserts["EMPLOYEE"] + "\n" + inserts["PROJECT"]).strip()
                info_md, df_comp, style_html = execute_sql_general(pool, sql, limit=0)
                return info_md
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ データ投入に失敗しました: {e}")

        def _reset_all():
            try:
                drop_sql = (
                    "DROP VIEW V_DEPT_PROJECT;\n"
                    "DROP VIEW V_EMP_DEPT;\n"
                    "DROP TABLE PROJECT PURGE;\n"
                    "DROP TABLE EMPLOYEE PURGE;\n"
                    "DROP TABLE DEPARTMENT PURGE;\n"
                )
                # ドロップは失敗しても続行できるように個別実行
                try:
                    execute_sql_general(pool, drop_sql, limit=0)
                except Exception as e:
                    logger.warning(f"Drop errors ignored: {e}")
                # 再作成
                # info1, df1, _ = execute_sql_general(pool, tables_sql, limit=0)
                # info2, df2, _ = execute_sql_general(pool, views_sql, limit=0)
                # info3, df3, _ = execute_sql_general(pool, (inserts["DEPARTMENT"] + "\n" + inserts["EMPLOYEE"] + "\n" + inserts["PROJECT"]).strip(), limit=0)
                return gr.Markdown(visible=True, value="✅ 初期化が完了しました")
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ 初期化に失敗しました: {e}")

        def _on_lesson_change(choice: str):
            try:
                sel_id = (choice or "").split(" - ")[0]
                lmap = {l["id"]: l for l in lessons}
                lesson = lmap.get(sel_id)
                if not lesson:
                    return gr.Markdown(visible=True, value="⚠️ レッスンが見つかりません"), gr.Textbox(value=""),
                return gr.Markdown(visible=True, value=f"【{lesson['id']}】{lesson['title']}\n\n{lesson['desc']}"), gr.Textbox(value=lesson["sql"]) 
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ 取得に失敗しました: {e}"), gr.Textbox(value="")

        def _run_lesson(sql_text: str):
            try:
                info_md, df_comp, style_html = execute_select_sql(pool, sql_text, limit=1000)
                # データが返却されなかった場合はDataFrameを非表示にする
                df_value = df_comp.value if hasattr(df_comp, 'value') else df_comp
                if df_value is None or (isinstance(df_value, pd.DataFrame) and df_value.empty):
                    return info_md, gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)
                return info_md, df_comp, style_html
            except Exception as e:
                return gr.Markdown(visible=True, value=f"❌ 実行に失敗しました: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame()), gr.HTML(visible=False)

        # イベントハンドラの接続
        show_tables_btn.click(fn=_show_tables, inputs=[tables_sql_visible_state], outputs=[show_tables_btn, tables_sql_text, tables_sql_visible_state])
        exec_tables_btn.click(fn=_exec_tables, outputs=[tables_result_md])

        show_views_btn.click(fn=_show_views, inputs=[views_sql_visible_state], outputs=[show_views_btn, views_sql_text, views_sql_visible_state])
        exec_views_btn.click(fn=_exec_views, outputs=[views_result_md])

        show_inserts_btn.click(fn=_show_inserts, inputs=[inserts_sql_visible_state], outputs=[show_inserts_btn, inserts_sql_text, inserts_sql_visible_state])
        exec_inserts_btn.click(fn=_exec_inserts, outputs=[inserts_result_md])

        reset_btn.click(fn=_reset_all, outputs=[reset_result_md])

        lesson_select.change(fn=_on_lesson_change, inputs=[lesson_select], outputs=[lesson_desc_md, lesson_sql_text])
        run_lesson_btn.click(fn=_run_lesson, inputs=[lesson_sql_text], outputs=[lesson_result_info, lesson_result_df, lesson_result_style])
