"""SQL Assist - Main application entry point.

This module initializes the Gradio web application for SQL assistance,
including database connection pooling and OCI GenAI integration.
"""

import argparse
import os
import platform
import warnings
import threading
from typing import Optional
import socket

import gradio as gr
import oracledb
from dotenv import find_dotenv, load_dotenv
from gradio.themes import Default, GoogleFont
import logging

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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.info("Environment variables loaded")

# Initialize Oracle client for Linux
if platform.system() == "Linux":
    lib_dir = os.environ.get("ORACLE_CLIENT_LIB_DIR", "/u01/aipoc/instantclient_23_8")
    config_dir = os.environ.get("TNS_ADMIN")
    if config_dir:
        logger.info(f"Initializing Oracle client lib_dir={lib_dir}, config_dir={config_dir}")
        oracledb.init_oracle_client(lib_dir=lib_dir, config_dir=config_dir)
    else:
        logger.info(f"Initializing Oracle client lib_dir={lib_dir}")
        oracledb.init_oracle_client(lib_dir=lib_dir)

# Lazy database connection pool
class LazyPool:
    def __init__(self, **kwargs):
        self._pool = None
        self._kwargs = kwargs
        self._lock = threading.RLock()

    def _ensure(self):
        with self._lock:
            if self._pool is None:
                dsn = self._kwargs.get("dsn")
                if not dsn or not str(dsn).strip():
                    logger.warning("DSN is empty; skip creating DB connection pool")
                    raise RuntimeError("ORACLE_26AI_CONNECTION_STRING is not set")
                if str(os.environ.get("DB_CONNECT_PRECHECK_ENABLED", "false")).lower() in ("1", "true", "yes"):
                    try:
                        self._precheck_connectivity()
                    except socket.gaierror as e:
                        logger.warning(f"DB connectivity precheck name resolution failed: {e}")
                    except Exception as e:
                        logger.warning(f"DB connectivity precheck failed: {e}")
                logger.info("Creating DB connection pool")
                self._pool = oracledb.create_pool(**self._kwargs)

    def _precheck_connectivity(self):
        dsn = str(self._kwargs.get("dsn") or "")
        host = None
        port = None
        try:
            if "@" in dsn:
                after = dsn.split("@", 1)[1]
            else:
                after = dsn
            hp = after.split("/")[0]
            parts = hp.split(":")
            host = parts[0] if parts else None
            if len(parts) > 1:
                try:
                    port = int(parts[1])
                except Exception:
                    port = None
        except Exception:
            pass
        t = float(os.environ.get("DB_CONNECT_PRECHECK_TIMEOUT", "3") or "3")
        if host:
            p = port or 1521
            with socket.create_connection((host, p), timeout=t):
                pass

    def acquire(self):
        self._ensure()
        try:
            conn = self._pool.acquire()
            try:
                conn.ping()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                self.reset()
                conn = self._pool.acquire()
                conn.ping()
            return conn
        except Exception:
            self.reset()
            conn = self._pool.acquire()
            conn.ping()
            return conn

    def close(self):
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                finally:
                    self._pool = None

    def reset(self):
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception:
                    pass
            self._pool = None
            logger.info("Recreating DB connection pool")
            self._pool = oracledb.create_pool(**self._kwargs)

    def warmup(self, sessions: int = 1, test_query: Optional[str] = "SELECT 1 FROM DUAL"):
        self._ensure()
        n = max(1, int(sessions or 1))
        for _ in range(n):
            with self.acquire() as conn:
                if test_query:
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(test_query)
                            _ = cursor.fetchmany(size=1)
                    except Exception:
                        raise

    def healthy(self) -> bool:
        try:
            with self.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
                    _ = cursor.fetchmany(size=1)
            return True
        except Exception:
            return False

    def __getattr__(self, name):
        self._ensure()
        return getattr(self._pool, name)

pool = LazyPool(
    dsn=os.environ.get("ORACLE_26AI_CONNECTION_STRING", ""),
    min=0,
    max=8,
    increment=1,
    timeout=30,
    getmode=oracledb.POOL_GETMODE_WAIT,
)
logger.info("LazyPool configured")


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
            build_oracle_ai_database_tab(pool)

            # OCI GenAI Embeddingテストタブを構築
            build_oci_embedding_test_tab(pool)

        with gr.TabItem(label="データベース管理"):
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
