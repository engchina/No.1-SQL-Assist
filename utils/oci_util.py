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


def get_region():
    """OCI設定ファイルからリージョン情報を取得する共通関数.

    Returns:
        str: OCIリージョン名
    """
    oci_config_path = find_dotenv("/root/.oci/config")
    region = get_key(oci_config_path, "region")
    return region


def update_oci_config(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
    has_error = False
    if not user_ocid:
        has_error = True
        gr.Warning("User OCIDを入力してください")
    if not tenancy_ocid:
        has_error = True
        gr.Warning("Tenancy OCIDを入力してください")
    if not fingerprint:
        has_error = True
        gr.Warning("Fingerprintを入力してください")
    if not private_key_file:
        has_error = True
        gr.Warning("Private Keyを入力してください")
    if not region:
        has_error = True
        gr.Warning("Regionを選択してください")

    if has_error:
        return gr.Markdown(visible=True, value="❌ 入力不足です")

    user_ocid = str(user_ocid).strip()
    tenancy_ocid = str(tenancy_ocid).strip()
    fingerprint = str(fingerprint).strip()
    region = str(region).strip()

    try:
        base_dir = Path(__file__).resolve().parent.parent
        oci_dir = Path("/root/.oci")
        oci_dir.mkdir(parents=True, exist_ok=True)

        if not (oci_dir / "config").exists():
            config_src = base_dir / ".oci" / "config"
            if config_src.exists():
                shutil.copy(str(config_src), str(oci_dir / "config"))
            else:
                pass

        oci_config_path = find_dotenv("/root/.oci/config")
        key_file_path = "/root/.oci/oci_api_key.pem"

        set_key(oci_config_path, "user", user_ocid, quote_mode="never")
        set_key(oci_config_path, "tenancy", tenancy_ocid, quote_mode="never")
        set_key(oci_config_path, "region", region, quote_mode="never")
        set_key(oci_config_path, "fingerprint", fingerprint, quote_mode="never")
        set_key(oci_config_path, "key_file", key_file_path, quote_mode="never")
        shutil.copy(private_key_file.name, key_file_path)
        load_dotenv(oci_config_path)

        return gr.Markdown(visible=True, value="✅ OCI設定ファイルを更新しました")
    except Exception as e:
        logger.error(f"Error updating OCI config: {e}")
        return gr.Markdown(visible=True, value=f"❌ OCI設定ファイルの更新に失敗しました: {e}")


def create_oci_db_credential(user_ocid, tenancy_ocid, fingerprint, private_key_file, region, pool=None):
    def process_private_key(private_key_file_path):
        with open(private_key_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return "".join(line.strip() for line in lines if not line.startswith("--"))

    has_error = False
    if not user_ocid:
        has_error = True
        gr.Warning("User OCIDを入力してください")
    if not tenancy_ocid:
        has_error = True
        gr.Warning("Tenancy OCIDを入力してください")
    if not fingerprint:
        has_error = True
        gr.Warning("Fingerprintを入力してください")
    if not private_key_file:
        has_error = True
        gr.Warning("Private Keyを入力してください")
    if has_error:
        return gr.Accordion(), gr.Textbox()

    compartment_ocid = os.environ.get("OCI_COMPARTMENT_OCID", "")
    if not compartment_ocid:
        gr.Error("OCI_COMPARTMENT_OCID環境変数が設定されていません")
        return gr.Accordion(), gr.Textbox()

    if pool is None:
        gr.Error("データベース接続プールが初期化されていません")
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
                except DatabaseError:
                    pass

                try:
                    cursor.execute("BEGIN dbms_vector.drop_credential('OCI_CRED'); END;")
                except DatabaseError:
                    pass

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

        gr.Info("OCI_CREDの作成が完了しました")
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
        gr.Error(f"データベースエラー: {error_msg}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        gr.Error(f"エラーが発生しました: {e}")
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
    with gr.TabItem(label="OCI GenAI 設定") as tab_create_oci_cred:
        with gr.Accordion(label="OCI GenAI 設定", open=True):
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_user_ocid_text = gr.Textbox(
                        label="User OCID*",
                        lines=1,
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_tenancy_ocid_text = gr.Textbox(
                        label="Tenancy OCID*",
                        lines=1,
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_fingerprint_text = gr.Textbox(
                        label="Fingerprint*",
                        lines=1,
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_private_key_file = gr.File(
                        label="Private Key*",
                        file_types=[".pem"],
                        type="filepath",
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_region_text = gr.Dropdown(
                        choices=["ap-osaka-1", "us-chicago-1"],
                        label="Region*",
                        interactive=True,
                        value="us-chicago-1",
                    )

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
        return test_oci_cred(test_query_text, embed_model, pool)
    
    # UIコンポーネントの構築
    with gr.TabItem(label="OCI GenAI 埋め込みテスト") as tab_test_oci_cred:
        with gr.Accordion(label="OCI 認証", open=True):
            with gr.Row():
                with gr.Column():
                    tab_auto_create_status_md = gr.Markdown(visible=False)
            with gr.Accordion(label="SQL", open=False):
                with gr.Column():
                    tab_auto_create_sql_text = gr.Textbox(
                        label="SQL",
                        show_label=False,
                        lines=15,
                        max_lines=15,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                    )
            with gr.Row():
                tab_auto_create_btn = gr.Button(value="OCI_CREDを作成", variant="primary")

        with gr.Accordion(label="埋め込みモデル", open=True):
            with gr.Row():
                with gr.Column():
                    tab_test_oci_cred_vector_text = gr.Textbox(
                        label="ベクトル結果",
                        lines=8,
                        max_lines=8,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_test_oci_cred_query_text = gr.Textbox(
                        label="テキスト",
                        placeholder="埋め込みベクトルに変換するテキストを入力してください...",
                        lines=2,
                        max_lines=5,
                        value="こんにちわ",
                    )
            
            with gr.Row():
                with gr.Column():
                    tab_test_oci_cred_model_input = gr.Dropdown(
                        label="モデル",
                        choices=["cohere.embed-v4.0"],
                        value="cohere.embed-v4.0",
                        interactive=True,
                    )

            with gr.Row():
                with gr.Column():
                    tab_test_clear_button = gr.ClearButton(
                        value="クリア",
                        components=[]
                    )
                with gr.Column():
                    tab_test_oci_cred_button = gr.Button(value="テスト", variant="primary")

        # イベントハンドラーの設定
        tab_test_clear_button.add(
            components=[tab_test_oci_cred_query_text, tab_test_oci_cred_vector_text]
        )
        
        tab_test_oci_cred_button.click(
            test_oci_cred_wrapper,
            inputs=[tab_test_oci_cred_query_text, tab_test_oci_cred_model_input],
            outputs=[tab_test_oci_cred_vector_text],
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
        compartment_ocid = get_key(oci_config_path, "compartment_ocid") or os.environ.get("OCI_COMPARTMENT_OCID", "")
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
                except DatabaseError:
                    pass
                try:
                    cursor.execute("BEGIN dbms_vector.drop_credential('OCI_CRED'); END;")
                except DatabaseError:
                    pass
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
