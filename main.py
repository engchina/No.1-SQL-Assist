"""SQL Assist - Main application entry point.

This module initializes the Gradio web application for SQL assistance,
including database connection pooling and OCI GenAI integration.
"""

import argparse
import os
import platform
import warnings

import gradio as gr
import oracledb
from dotenv import find_dotenv, load_dotenv
from gradio.themes import Default, GoogleFont

from utils.css_util import custom_css
from utils.oci_util import build_oci_genai_tab, build_oci_embedding_test_tab, build_oracle_ai_database_tab
from utils.chat_util import build_oci_chat_test_tab
from utils.management_util import build_management_tab
from utils.selectai_util import build_selectai_tab
from utils.query_util import build_query_tab

# Suppress NumPy warnings about longdouble on certain platforms
warnings.filterwarnings("ignore", message=".*does not match any known type.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Oracle client for Linux
if platform.system() == "Linux":
    oracledb.init_oracle_client(
        lib_dir=os.environ.get("ORACLE_CLIENT_LIB_DIR", "/u01/aipoc/instantclient_23_8")
    )

# Lazy database connection pool
class LazyPool:
    def __init__(self, **kwargs):
        self._pool = None
        self._kwargs = kwargs

    def _ensure(self):
        if self._pool is None:
            self._pool = oracledb.create_pool(**self._kwargs)

    def acquire(self):
        self._ensure()
        return self._pool.acquire()

    def close(self):
        if self._pool is not None:
            self._pool.close()

    def __getattr__(self, name):
        self._ensure()
        return getattr(self._pool, name)

pool = LazyPool(
    dsn=os.environ.get("ORACLE_26AI_CONNECTION_STRING", ""),
    min=5,
    max=20,
    increment=2,
    timeout=30,
    getmode=oracledb.POOL_GETMODE_WAIT,
)

# Configure Gradio theme
theme = Default(
    spacing_size="sm",
    font=[
        GoogleFont(name="Noto Sans JP"),
        GoogleFont(name="Noto Sans SC"),
        GoogleFont(name="Roboto"),
    ],
).set()

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=theme, title="SQL Assist") as app:
    gr.Markdown(value="# No.1 SQL Assist", elem_classes="main_Header")
    gr.Markdown(
        value="### 開発者がSQLクエリを簡単に生成し、SQLの理解を深めるためのツール",
        elem_classes="sub_Header",
    )

    with gr.Tabs():
        with gr.TabItem(label="環境設定"):
            # OCI GenAI設定タブを構築
            build_oci_genai_tab(pool)

            # Oracle AI Database タブを追加
            build_oracle_ai_database_tab()

            # OCI GenAI Embeddingテストタブを構築
            build_oci_embedding_test_tab(pool)

        with gr.TabItem(label="データベース管理機能"):
            # 管理機能タブを構築
            build_management_tab(pool)

        build_query_tab(pool)

        with gr.TabItem(label="SelectAI 連携"):
            build_selectai_tab(pool)

        build_oci_chat_test_tab(pool)

    gr.Markdown(
        value="### 本ソフトウェアは検証評価用です。日常利用のための基本機能は備えていない点につきましてご理解をよろしくお願い申し上げます。",
        elem_classes="sub_Header",
    )
    gr.Markdown(value="### Developed by Oracle Japan", elem_classes="sub_Header")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SQL Assist web application")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to run the server (default: 8080)",
    )
    args = parser.parse_args()

    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        max_threads=200,
        show_api=False,
    )
