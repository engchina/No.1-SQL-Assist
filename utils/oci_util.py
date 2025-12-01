"""OCI GenAI設定ユーティリティモジュール.

このモジュールは、OCI GenAIの認証情報を設定・テストするための
Gradio UIコンポーネントを提供します。
"""

import json
import logging
import os
import shutil
from pathlib import Path

import gradio as gr
import oracledb
import pandas as pd
import oci
from dotenv import find_dotenv, get_key, load_dotenv, set_key
from oracledb import DatabaseError

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress verbose logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# OCI設定ファイルのパス定数
OCI_CONFIG_PATH = "/root/.oci/config"
OCI_KEY_FILE_PATH = "/root/.oci/oci_api_key.pem"


def get_region():
    """OCI設定ファイルからリージョン情報を取得する共通関数.

    Returns:
        str: OCIリージョン名、取得できない場合はNone
    """
    try:
        oci_config_path = find_dotenv(OCI_CONFIG_PATH)
        if not oci_config_path:
            # find_dotenvが失敗した場合は直接パスを使用
            oci_config_path = OCI_CONFIG_PATH
        
        # ファイルが存在するか確認
        if not Path(oci_config_path).exists():
            logger.warning(f"OCI設定ファイルが存在しません: {oci_config_path}")
            return None
            
        region = get_key(oci_config_path, "region")
        return region
    except Exception as e:
        logger.error(f"リージョン取得エラー: {e}")
        return None


def update_oci_config(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
    """OCI設定ファイルを更新する.
    
    Args:
        user_ocid: ユーザーOCID
        tenancy_ocid: テナンシーOCID
        fingerprint: APIキーのフィンガープリント
        private_key_file: プライベートキーファイル（Gradio File型）
        region: OCIリージョン
        
    Returns:
        gr.Markdown: 更新結果メッセージ
    """
    has_error = False
    error_messages = []
    
    if not user_ocid:
        has_error = True
        error_messages.append("User OCID")
        logger.error("User OCIDが未入力です")
    if not tenancy_ocid:
        has_error = True
        error_messages.append("Tenancy OCID")
        logger.error("Tenancy OCIDが未入力です")
    if not fingerprint:
        has_error = True
        error_messages.append("Fingerprint")
        logger.error("Fingerprintが未入力です")
    if not private_key_file or (hasattr(private_key_file, 'name') and not private_key_file.name):
        has_error = True
        error_messages.append("Private Key")
        logger.error("Private Keyが未入力です")
    if not region:
        has_error = True
        error_messages.append("Region")
        logger.error("Regionが未選択です")

    if has_error:
        missing_fields = "、".join(error_messages)
        return gr.Markdown(visible=True, value=f"❌ 入力不足です: {missing_fields}が未入力です")

    user_ocid = str(user_ocid).strip()
    tenancy_ocid = str(tenancy_ocid).strip()
    fingerprint = str(fingerprint).strip()
    region = str(region).strip()

    try:
        # プライベートキーファイルの存在確認
        if not hasattr(private_key_file, 'name') or not private_key_file.name:
            return gr.Markdown(visible=True, value="❌ プライベートキーファイルが選択されていません")
            
        base_dir = Path(__file__).resolve().parent.parent
        oci_dir = Path("/root/.oci")
        oci_dir.mkdir(parents=True, exist_ok=True)

        if not (oci_dir / "config").exists():
            config_src = base_dir / ".oci" / "config"
            if config_src.exists():
                shutil.copy(str(config_src), str(oci_dir / "config"))
            else:
                # configファイルが存在しない場合は新規作成
                (oci_dir / "config").write_text("[DEFAULT]\n")

        # OCI設定ファイルのパスを取得
        oci_config_path = find_dotenv(OCI_CONFIG_PATH)
        if not oci_config_path:
            # find_dotenvが失敗した場合は直接パスを使用
            oci_config_path = OCI_CONFIG_PATH
            logger.info(f"find_dotenvが失敗したため、直接パスを使用: {oci_config_path}")
        
        # 設定ファイルが存在することを確認
        if not Path(oci_config_path).exists():
            logger.error(f"OCI設定ファイルが見つかりません: {oci_config_path}")
            return gr.Markdown(visible=True, value=f"❌ OCI設定ファイルが見つかりません: {oci_config_path}")
            
        key_file_path = OCI_KEY_FILE_PATH

        set_key(oci_config_path, "user", user_ocid, quote_mode="never")
        set_key(oci_config_path, "tenancy", tenancy_ocid, quote_mode="never")
        set_key(oci_config_path, "region", region, quote_mode="never")
        set_key(oci_config_path, "fingerprint", fingerprint, quote_mode="never")
        set_key(oci_config_path, "key_file", key_file_path, quote_mode="never")
        
        # プライベートキーファイルをコピー
        shutil.copy(private_key_file.name, key_file_path)
        # パーミッション設定（セキュリティのため）
        os.chmod(key_file_path, 0o600)
        
        load_dotenv(oci_config_path)
        logger.info("OCI設定ファイルを正常に更新しました")

        return gr.Markdown(visible=True, value="✅ OCI設定ファイルを更新しました")
    except Exception as e:
        logger.error(f"Error updating OCI config: {e}")
        logger.exception("Full traceback:")
        return gr.Markdown(visible=True, value=f"❌ OCI設定ファイルの更新に失敗しました: {e}")


def create_oci_db_credential(user_ocid, tenancy_ocid, fingerprint, private_key_file, region, pool=None):
    def process_private_key(private_key_file_path):
        with open(private_key_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return "".join(line.strip() for line in lines if not line.startswith("--"))

    has_error = False
    if not user_ocid:
        has_error = True
        logger.error("User OCIDが未入力です")
    if not tenancy_ocid:
        has_error = True
        logger.error("Tenancy OCIDが未入力です")
    if not fingerprint:
        has_error = True
        logger.error("Fingerprintが未入力です")
    if not private_key_file:
        has_error = True
        logger.error("Private Keyが未入力です")
    if has_error:
        return gr.Accordion(), gr.Textbox()

    compartment_ocid = os.environ.get("OCI_COMPARTMENT_OCID", "")
    logger.info(f"compartment_ocid: {compartment_ocid}")
    if not compartment_ocid:
        logger.error("OCI_COMPARTMENT_OCID環境変数が設定されていません")
        return gr.Accordion(), gr.Textbox()

    if pool is None:
        logger.error("データベース接続プールが初期化されていません")
        return gr.Accordion(), gr.Textbox()

    try:
        private_key = process_private_key(private_key_file.name)
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        """
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;"""
                    )
                except DatabaseError as e:
                    logger.error(f"ACL append failed: {e}")

                try:
                    cursor.execute("BEGIN dbms_vector.drop_credential('OCI_CRED'); END;")
                except DatabaseError as e:
                    logger.error(f"Drop credential failed: {e}")

                oci_cred = {
                    "user_ocid": str(user_ocid).strip(),
                    "tenancy_ocid": str(tenancy_ocid).strip(),
                    "compartment_ocid": compartment_ocid,
                    "private_key": private_key.strip(),
                    "fingerprint": str(fingerprint).strip(),
                }

                cursor.execute(
                    """
BEGIN
   dbms_vector.create_credential(
       credential_name => 'OCI_CRED',
       params => json(:json_params)
   );
END;""",
                    json_params=json.dumps(oci_cred),
                )
                conn.commit()

        create_sql_preview = f"""
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;

BEGIN dbms_vector.drop_credential('OCI_CRED'); END;

BEGIN
    dbms_vector.create_credential(
        credential_name => 'OCI_CRED',
        params => json('{json.dumps(oci_cred)}')
    );
END;"""

        return gr.Accordion(), gr.Textbox(value=create_sql_preview.strip())
    except Exception as e:
        logger.error(f"Error creating OCI credential: {e}")
        return gr.Accordion(), gr.Textbox()


def test_oci_cred(test_query_text, embed_model, pool):
    """OCI認証情報をテストする.

    Args:
        test_query_text: テスト用のクエリテキスト
        pool: データベース接続プール

    Returns:
        gr.Textbox: ベクトル結果を含むTextbox
    """
    logger.info("=" * 50)
    logger.info(f"Starting OCI credential test with text: '{test_query_text}'")
    logger.info(f"Embedding model: {embed_model}")
    logger.info(f"Input length: {len(str(test_query_text or ''))}")
    
    test_query_vector = ""
    
    try:
        region = get_region()
        logger.info(f"Region: {region}")
        
        embed_genai_params = {
            "provider": "ocigenai",
            "credential_name": "OCI_CRED",
            "url": f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText",
            "model": embed_model,
        }

        logger.info("Executing database query...")
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                plsql = """
DECLARE
    l_embed_genai_params CLOB := :embed_genai_params;
    l_result SYS_REFCURSOR;
BEGIN
    OPEN l_result FOR
        SELECT et.*
        FROM dbms_vector_chain.utl_to_embeddings(:text_to_embed, JSON(l_embed_genai_params)) et;
    :result := l_result;
END;"""

                result_cursor = cursor.var(oracledb.CURSOR)

                cursor.execute(
                    plsql,
                    embed_genai_params=json.dumps(embed_genai_params),
                    text_to_embed=test_query_text,
                    result=result_cursor,
                )

                # Fetch the results from the ref cursor
                with result_cursor.getvalue() as ref_cursor:
                    result_rows = ref_cursor.fetchall()
                    logger.info(f"Fetched {len(result_rows)} row(s)")
                    
                    for idx, row in enumerate(result_rows, 1):
                        if isinstance(row, tuple) and len(row) > 0:
                            # Handle both LOB and string types
                            if isinstance(row[0], oracledb.LOB):
                                logger.info(f"Row {idx}: Processing LOB data")
                                lob_content = row[0].read()
                                lob_json = json.loads(lob_content)
                                
                                if "embed_vector" in lob_json:
                                    vector_length = len(lob_json["embed_vector"])
                                    logger.info(f"  Vector length: {vector_length}")
                                    test_query_vector += str(lob_json["embed_vector"]) + "\n"
                                else:
                                    logger.warning("  'embed_vector' key not found in JSON")
                            elif isinstance(row[0], str):
                                logger.info(f"Row {idx}: Processing string data (length: {len(row[0])})")
                                
                                try:
                                    result_json = json.loads(row[0])
                                    
                                    if "embed_vector" in result_json:
                                        embed_vector = result_json["embed_vector"]
                                        # Handle embed_vector which might be a string or list
                                        if isinstance(embed_vector, str):
                                            vector_list = json.loads(embed_vector)
                                            logger.info(f"  Vector length: {len(vector_list)}")
                                            test_query_vector += str(vector_list) + "\n"
                                        else:
                                            logger.info(f"  Vector length: {len(embed_vector)}")
                                            test_query_vector += str(embed_vector) + "\n"
                                    else:
                                        logger.warning(f"  'embed_vector' not found. Available keys: {list(result_json.keys())}")
                                except json.JSONDecodeError as je:
                                    logger.error(f"  JSON parse error: {je}")
                            else:
                                logger.warning(f"Row {idx}: Unexpected type {type(row[0])}")
                        else:
                            logger.warning(f"Row {idx}: Empty or invalid row")

        if test_query_vector:
            logger.info(f"✓ Test completed successfully. Vector length: {len(test_query_vector)} chars")
        else:
            logger.warning("⚠ Test completed but vector is empty!")
        logger.info("=" * 50)
        
    except DatabaseError as de:
        error_code = de.args[0].code if de.args else 'N/A'
        error_msg = de.args[0].message if de.args else str(de)
        logger.error(f"Database error [{error_code}]: {error_msg}")
        test_query_vector = f"❌ エラー: {error_msg}"
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        test_query_vector = f"❌ エラー: {e}"
        logger.info("=" * 50)

    return gr.Textbox(value=test_query_vector)


def build_oci_genai_tab(pool):
    """OCI GenAI設定タブのUIを構築する.

    Args:
        pool: データベース接続プール

    Returns:
        gr.TabItem: OCI GenAI設定タブ
    """
    
    # ラッパー関数の定義
    def create_oci_cred_wrapper(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
        return create_oci_db_credential(user_ocid, tenancy_ocid, fingerprint, private_key_file, region, pool)
    def update_oci_config_wrapper(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
        return update_oci_config(user_ocid, tenancy_ocid, fingerprint, private_key_file, region)
    
    # UIコンポーネントの構築
    with gr.TabItem(label="OCI 認証情報設定") as tab_create_oci_cred:
        with gr.Accordion(label="OCI 認証情報設定", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                     gr.Markdown("User OCID*", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_create_oci_cred_user_ocid_text = gr.Textbox(
                        show_label=False,
                        lines=1,
                        interactive=True,
                        container=False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                     gr.Markdown("Tenancy OCID*", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_create_oci_cred_tenancy_ocid_text = gr.Textbox(
                        show_label=False,
                        lines=1,
                        interactive=True,
                        container=False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                     gr.Markdown("Fingerprint*", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_create_oci_cred_fingerprint_text = gr.Textbox(
                        show_label=False,
                        lines=1,
                        interactive=True,
                        container=False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                     gr.Markdown("Private Key*", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_create_oci_cred_private_key_file = gr.File(
                        show_label=False,
                        file_types=[".pem"],
                        type="filepath",
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("Region*", elem_classes="input-label")
                        with gr.Column(scale=5):
                            tab_create_oci_cred_region_text = gr.Dropdown(
                                choices=["ap-osaka-1", "us-chicago-1"],
                                show_label=False,
                                interactive=True,
                                value="us-chicago-1",
                                container=False,
                            )
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("")

            with gr.Row():
                with gr.Column():
                    tab_create_oci_clear_button = gr.ClearButton(value="クリア")
                with gr.Column():
                    tab_update_oci_config_button = gr.Button(value="OCI設定ファイルを更新", variant="primary")

            with gr.Row():
                with gr.Column():
                    tab_create_oci_config_status_md = gr.Markdown(visible=False)

        # イベントハンドラーの設定
        tab_create_oci_clear_button.add(
            components=[
                tab_create_oci_cred_user_ocid_text,
                tab_create_oci_cred_tenancy_ocid_text,
                tab_create_oci_cred_fingerprint_text,
                tab_create_oci_cred_private_key_file,
            ]
        )

        tab_update_oci_config_button.click(
            update_oci_config_wrapper,
            inputs=[
                tab_create_oci_cred_user_ocid_text,
                tab_create_oci_cred_tenancy_ocid_text,
                tab_create_oci_cred_fingerprint_text,
                tab_create_oci_cred_private_key_file,
                tab_create_oci_cred_region_text,
            ],
            outputs=[tab_create_oci_config_status_md],
        )

        return tab_create_oci_cred


def build_oci_embedding_test_tab(pool):
    """OCI GenAI EmbeddingテストタブのUIを構築する.

    Args:
        pool: データベース接続プール

    Returns:
        gr.TabItem: OCI GenAI Embeddingテストタブ
    """
    
    # ラッパー関数の定義
    def test_oci_cred_wrapper(test_query_text, embed_model):
        logger.info("Embedding test button clicked")
        logger.info(f"Model selected: {embed_model}")
        logger.info(f"Text preview: {str(test_query_text)[:80]}")
        try:
            yield gr.Markdown(visible=True, value="⏳ テスト中..."), gr.Textbox(value="")
            res = test_oci_cred(test_query_text, embed_model, pool)
            logger.info("Embedding test completed")
            yield gr.Markdown(visible=True, value="✅ 完了"), res
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="")
    
    # UIコンポーネントの構築
    with gr.TabItem(label="Embeddingテスト") as tab_test_oci_cred:
        with gr.Accordion(label="OCI Credentialの作成", open=True):
            with gr.Accordion(label="生成されたSQL", open=False):
                with gr.Column():
                    tab_auto_create_sql_text = gr.Textbox(
                        label="SQL",
                        show_label=False,
                        lines=15,
                        max_lines=15,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                        container=False,
                    )
            with gr.Row():
                tab_auto_create_btn = gr.Button(value="OCI Credentialを作成", variant="primary")
            with gr.Row():
                tab_auto_create_status_md = gr.Markdown(visible=False)
            
        with gr.Accordion(label="Embedding生成テスト", open=True):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("Embeddingモデル", elem_classes="input-label")
                        with gr.Column(scale=5):
                            tab_test_oci_cred_model_input = gr.Dropdown(
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
                with gr.Column(scale=1):
                    gr.Markdown("入力テキスト", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_test_oci_cred_query_text = gr.Textbox(
                        show_label=False,
                        placeholder="Embeddingベクトルに変換するテキストを入力してください...",
                        lines=2,
                        max_lines=5,
                        value="こんにちわ",
                        container=False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("ベクトル結果", elem_classes="input-label")
                with gr.Column(scale=5):
                    tab_test_oci_cred_vector_text = gr.Textbox(
                        show_label=False,
                        lines=8,
                        max_lines=8,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                        container=False,
                    )

            with gr.Row():
                with gr.Column():
                    tab_test_clear_button = gr.ClearButton(
                        value="クリア",
                        components=[]
                    )
                with gr.Column():
                    tab_test_oci_cred_button = gr.Button(value="テスト", variant="primary")
            with gr.Row():
                tab_test_status_md = gr.Markdown(visible=False)

        # イベントハンドラーの設定
        tab_test_clear_button.add(
            components=[tab_test_oci_cred_query_text, tab_test_oci_cred_vector_text]
        )
        
        tab_test_oci_cred_button.click(
            test_oci_cred_wrapper,
            inputs=[tab_test_oci_cred_query_text, tab_test_oci_cred_model_input],
            outputs=[tab_test_status_md, tab_test_oci_cred_vector_text],
        )

        def create_oci_cred_from_config_wrapper():
            for _vals in create_oci_db_credential_from_config(pool):
                yield _vals

        tab_auto_create_btn.click(
            create_oci_cred_from_config_wrapper,
            outputs=[tab_auto_create_btn, tab_auto_create_status_md, tab_auto_create_sql_text],
        )

    return tab_test_oci_cred

def create_oci_db_credential_from_config(pool=None):
    try:
        yield gr.Button(value="作成中...", interactive=False), gr.Markdown(visible=True, value="⏳ OCI_CRED作成中..."), gr.Textbox(value="")
        oci_config_path = find_dotenv("/root/.oci/config")
        user_ocid = get_key(oci_config_path, "user")
        tenancy_ocid = get_key(oci_config_path, "tenancy")
        fingerprint = get_key(oci_config_path, "fingerprint")
        key_file_path = get_key(oci_config_path, "key_file")
        region = get_key(oci_config_path, "region")
        compartment_ocid = os.environ.get("OCI_COMPARTMENT_OCID", "")
        logger.info(f"compartment_ocid: {compartment_ocid}")
        if not all([user_ocid, tenancy_ocid, fingerprint, key_file_path, region]):
            yield gr.Button(value="OCI_CREDを作成", interactive=True), gr.Markdown(visible=True, value="❌ OCI設定ファイルが不完全です"), gr.Textbox(value="")
            return
        if not compartment_ocid:
            yield gr.Button(value="OCI_CREDを作成", interactive=True), gr.Markdown(visible=True, value="❌ Compartment OCIDが見つかりません"), gr.Textbox(value="")
            return
        def _proc_key(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(line.strip() for line in lines if not line.startswith("--"))
        private_key = _proc_key(key_file_path)
        if pool is None:
            yield gr.Button(value="OCI_CREDを作成", interactive=True), gr.Markdown(visible=True, value="❌ データベース接続プール未初期化"), gr.Textbox(value="")
            return
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        """
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;"""
                    )
                except DatabaseError as e:
                    logger.error(f"ACL append failed: {e}")
                try:
                    genai_host = f"inference.generativeai.{region}.oci.oraclecloud.com"
                    cursor.execute(
                        """
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => :host,
    ace  => xs$ace_type(privilege_list => xs$name_list('http'),
                        principal_name => 'admin',
                        principal_type => xs_acl.ptype_db));
END;""",
                        host=genai_host,
                    )
                except DatabaseError as e:
                    logger.error(f"GenAI ACL append failed: {e}")
                try:
                    cursor.execute("BEGIN dbms_vector.drop_credential('OCI_CRED'); END;")
                except DatabaseError as e:
                    logger.error(f"Drop credential failed: {e}")
                oci_cred = {
                    "user_ocid": str(user_ocid).strip(),
                    "tenancy_ocid": str(tenancy_ocid).strip(),
                    "compartment_ocid": compartment_ocid,
                    "private_key": private_key.strip(),
                    "fingerprint": str(fingerprint).strip(),
                }
                cursor.execute(
                    """
BEGIN
   dbms_vector.create_credential(
       credential_name => 'OCI_CRED',
       params => json(:json_params)
   );
END;""",
                    json_params=json.dumps(oci_cred),
                )
                conn.commit()
        preview = f"""
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;

BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => 'inference.generativeai.{region}.oci.oraclecloud.com',
    ace  => xs$ace_type(privilege_list => xs$name_list('http'),
                        principal_name => 'admin',
                        principal_type => xs_acl.ptype_db));
END;

BEGIN dbms_vector.drop_credential('OCI_CRED'); END;

BEGIN
    dbms_vector.create_credential(
        credential_name => 'OCI_CRED',
        params => json('{json.dumps(oci_cred)}')
    );
END;"""
        yield gr.Button(value="OCI_CREDを作成", interactive=True), gr.Markdown(visible=True, value="✅ OCI_CREDを作成しました"), gr.Textbox(value=preview.strip())
    except Exception as e:
        logger.error(f"Error creating OCI_CRED from config: {e}")
        yield gr.Button(value="OCI_CREDを作成", interactive=True), gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="")
    
def _oci_config_with_region(region: str) -> dict:
    cfg_path = find_dotenv("/root/.oci/config")
    cfg = oci.config.from_file(file_location=cfg_path) if cfg_path else {}
    if region:
        cfg["region"] = region
    return cfg


def _list_adb(region: str, compartment_id: str):
    cfg = _oci_config_with_region(region)
    client = oci.database.DatabaseClient(cfg)
    resp = client.list_autonomous_databases(compartment_id=compartment_id)
    return resp.data


def _get_adb(client: oci.database.DatabaseClient, adb_id: str):
    return client.get_autonomous_database(autonomous_database_id=adb_id).data


def _start_adb(region: str, adb_id: str):
    cfg = _oci_config_with_region(region)
    client = oci.database.DatabaseClient(cfg)
    client.start_autonomous_database(autonomous_database_id=adb_id)
    try:
        return _get_adb(client, adb_id)
    except Exception as e:
        logger.error(f"_start_adb status fetch error: {e}")
        return oci.database.models.AutonomousDatabase(lifecycle_state="STARTING")


def _stop_adb(region: str, adb_id: str):
    cfg = _oci_config_with_region(region)
    client = oci.database.DatabaseClient(cfg)
    client.stop_autonomous_database(autonomous_database_id=adb_id)
    try:
        return _get_adb(client, adb_id)
    except Exception as e:
        logger.error(f"_stop_adb status fetch error: {e}")
        return oci.database.models.AutonomousDatabase(lifecycle_state="STOPPING")


def build_oracle_ai_database_tab(pool=None):
    with gr.TabItem(label="Oracle AI Database") as tab_adb:
        with gr.Accordion(label="Oracle AI Database", open=True):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("リージョン", elem_classes="input-label")
                        with gr.Column(scale=5):
                            region_input = gr.Dropdown(show_label=False, choices=["ap-osaka-1", "us-chicago-1"], value="ap-osaka-1", interactive=True, container=False)
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=5):
                            gr.Markdown("")
            with gr.Row():
                fetch_btn = gr.Button(value="ADB一覧を取得", variant="primary")
            with gr.Row():
                adb_status_md = gr.Markdown(visible=False)
            with gr.Row():
                adb_list_df = gr.Dataframe(label="ADB一覧", interactive=False, wrap=True, visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"]))
            with gr.Row():
                start_btn = gr.Button(value="起動", interactive=False, variant="primary")
                stop_btn = gr.Button(value="停止", interactive=False, variant="primary")
            with gr.Row():
                btn_status_md = gr.Markdown(visible=False)
        adb_map_state = gr.State({})
        adb_selected_id = gr.State("")

        def _state_to_val(x):
            try:
                if isinstance(x, dict):
                    return x
                v = getattr(x, "value", None)
                if isinstance(v, dict):
                    return v
            except Exception as e:
                logger.error(f"_state_to_val error: {e}")
            return {}

        def _fetch(region):
            comp = os.environ.get("OCI_COMPARTMENT_OCID", "")
            region_code = region
            if not comp:
                yield gr.Markdown(visible=True, value="⏳ ADB一覧を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"])), {}, ""
                yield gr.Markdown(visible=True, value="❌ OCI_COMPARTMENT_OCIDが見つかりません"), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"])), {}, ""
                return
            try:
                yield gr.Markdown(visible=True, value="⏳ ADB一覧を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"])), {}, ""
                items = _list_adb(region_code, comp)
                rows = []
                mp = {}
                for it in items:
                    st = it.lifecycle_state
                    if str(st).upper() == "TERMINATED":
                        continue
                    name = it.display_name
                    oid = it.id
                    rows.append([name, st, oid])
                    mp[oid] = {"name": name, "state": st}
                df = pd.DataFrame(rows, columns=["表示名", "状態", "OCID"]) if rows else pd.DataFrame(columns=["表示名", "状態", "OCID"]) 
                status_lines = []
                status_lines.append(f"リージョン: {region_code}")
                status_lines.append(f"取得件数: {len(rows)}")
                status_md = "\n".join(status_lines)
                yield gr.Markdown(visible=True, value=status_md), gr.Dataframe(visible=True, value=df), mp, ""
            except Exception as e:
                logger.error(f"_fetch error: {e}")
                yield gr.Markdown(visible=True, value="⏳ ADB一覧を取得中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"])), {}, ""
                yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["表示名", "状態", "OCID"])), {}, ""

        def _on_row_select(evt: gr.SelectData, current_df, mp):
            try:
                if isinstance(current_df, dict) and "data" in current_df:
                    headers = current_df.get("headers") or current_df.get("column_names") or []
                    data = current_df.get("data") or []
                    df = pd.DataFrame(data, columns=headers if headers else None)
                elif isinstance(current_df, pd.DataFrame):
                    df = current_df
                else:
                    df = pd.DataFrame(current_df)
                row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if len(df) > row_index:
                    # 取得した行からOCID/表示名/状態を抽出
                    ocid = None
                    name = None
                    st_df = None
                    try:
                        if "OCID" in df.columns:
                            ocid = str(df.iloc[row_index]["OCID"])
                        else:
                            ocid = str(df.iloc[row_index, 2])
                    except Exception:
                        ocid = None
                    try:
                        # 優先: 日本語表示名列
                        name = str(df.iloc[row_index]["表示名"]) if "表示名" in df.columns else (
                            str(df.iloc[row_index]["名称"]) if "名称" in df.columns else str(df.iloc[row_index, 0])
                        )
                    except Exception:
                        name = None
                    try:
                        st_df = str(df.iloc[row_index]["状態"]) if "状態" in df.columns else str(df.iloc[row_index, 1])
                    except Exception:
                        st_df = None

                    mpv = _state_to_val(mp)
                    info = (mpv or {}).get(ocid) or (mpv or {}).get(name)
                    st = info.get("state") if isinstance(info, dict) else (st_df or "")
                    if not st:
                        return gr.Markdown(visible=True, value="ℹ️ 行をクリックしてADBを選択してください"), gr.Button(interactive=False), gr.Button(interactive=False), ""
                    can_start = st in ("STOPPED", "INACTIVE")
                    can_stop = st in ("AVAILABLE", "RUNNING", "STARTING")
                    status_text = f"選択: {name or ''} / 状態: {st}"
                    return gr.Markdown(visible=True, value=status_text), gr.Button(interactive=can_start), gr.Button(interactive=can_stop), (ocid or "")
            except Exception as e:
                logger.error(f"_on_row_select エラー: {e}")
                return gr.Markdown(visible=True, value="ℹ️ 行をクリックしてADBを選択してください"), gr.Button(interactive=False), gr.Button(interactive=False), ""

        def _mp_to_df(mp):
            rows = []
            for ocid, v in (mp or {}).items():
                rows.append([v.get("name"), v.get("state"), ocid])
            return pd.DataFrame(rows, columns=["表示名", "状態", "OCID"]) if rows else pd.DataFrame(columns=["表示名", "状態", "OCID"]) 

        def _start(region, selected_id, mp):
            region_code = region
            mpv = _state_to_val(mp)
            if not selected_id:
                yield gr.Markdown(visible=True, value="❌ ADBが選択されていません"), gr.Button(interactive=False), gr.Button(interactive=False), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
                return
            yield gr.Markdown(visible=True, value="⏳ 起動をリクエスト中..."), gr.Button(interactive=False), gr.Button(interactive=False), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
            try:
                _start_adb(region_code, selected_id)
                st = "STARTING"
                if selected_id in mpv:
                    mpv[selected_id]["state"] = st
                else:
                    mpv[selected_id] = {"name": selected_id, "state": st}
                can_start = False
                can_stop = True
                msg = "⏳ 起動リクエストを送信しました。数分後に『ADB一覧を取得』で最新状態を確認してください"
                yield gr.Markdown(visible=True, value=msg), gr.Button(interactive=can_start), gr.Button(interactive=can_stop), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
            except Exception as e:
                logger.error(f"_start エラー: {e}")
                yield gr.Markdown(visible=True, value=f"❌ 起動エラー: {e}"), gr.Button(interactive=True), gr.Button(interactive=False), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))

        def _stop(region, selected_id, mp):
            region_code = region
            mpv = _state_to_val(mp)
            if not selected_id:
                yield gr.Markdown(visible=True, value="❌ ADBが選択されていません"), gr.Button(interactive=False), gr.Button(interactive=False), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
                return
            yield gr.Markdown(visible=True, value="⏳ 停止をリクエスト中..."), gr.Button(interactive=False), gr.Button(interactive=False), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
            try:
                try:
                    if pool is not None:
                        try:
                            pool.close()
                        except Exception as e:
                            logger.error(f"pool.close during stop error: {e}")
                    _stop_adb(region_code, selected_id)
                finally:
                    _ = True
                st = "STOPPING"
                if selected_id in mpv:
                    mpv[selected_id]["state"] = st
                else:
                    mpv[selected_id] = {"name": selected_id, "state": st}
                can_start = False
                can_stop = False
                msg = "⏳ 停止リクエストを送信しました。数分後に『ADB一覧を取得』で最新状態を確認してください"
                yield gr.Markdown(visible=True, value=msg), gr.Button(interactive=can_start), gr.Button(interactive=can_stop), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))
            except Exception as e:
                logger.error(f"_stop エラー: {e}")
                yield gr.Markdown(visible=True, value=f"❌ 停止エラー: {e}"), gr.Button(interactive=False), gr.Button(interactive=True), mpv, gr.Dataframe(visible=True, value=_mp_to_df(mpv))

        fetch_btn.click(
            _fetch,
            inputs=[region_input],
            outputs=[adb_status_md, adb_list_df, adb_map_state, adb_selected_id],
        )
        adb_list_df.select(
            _on_row_select,
            inputs=[adb_list_df, adb_map_state],
            outputs=[adb_status_md, start_btn, stop_btn, adb_selected_id],
        )
        start_btn.click(
            _start,
            inputs=[region_input, adb_selected_id, adb_map_state],
            outputs=[btn_status_md, start_btn, stop_btn, adb_map_state, adb_list_df],
        )
        stop_btn.click(
            _stop,
            inputs=[region_input, adb_selected_id, adb_map_state],
            outputs=[btn_status_md, start_btn, stop_btn, adb_map_state, adb_list_df],
        )

    return tab_adb
