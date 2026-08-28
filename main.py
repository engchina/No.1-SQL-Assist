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

from utils.auth_util import access_navigation_for_role, do_auth
from utils.css_util import custom_css
from utils.llm_model_util import (
    bind_llm_model_settings_events,
    reset_model_dropdown_registry,
)

# from utils.oci_util import build_oci_genai_tab, build_oci_embedding_test_tab, build_oracle_ai_database_tab, build_openai_settings_tab
from utils.chat_util import build_oci_chat_test_tab
from utils.settings_tab import build_settings_tab
from utils.management_util import build_management_tab
from utils.selectai_util import build_selectai_tab
from utils.query_util import _query_access_notice, build_query_tab
from utils.vpd_util import (
    VpdConfigurationError,
    normalize_vpd_login_users,
    parse_oracle_connection_string,
    request_username,
    user_role,
)

# Suppress NumPy warnings about longdouble on certain platforms
warnings.filterwarnings("ignore", message=".*does not match any known type.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Load environment variables
load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger.info("Environment variables loaded")

# Initialize Oracle client for Linux
if platform.system() == "Linux":
    from pathlib import Path

    os.environ.pop("TNS_ADMIN", None)
    lib_dir = os.environ.get("ORACLE_CLIENT_LIB_DIR", "/u01/aipoc/instantclient_23_26")
    config_dir = str(Path(lib_dir) / "network" / "admin")

    config_path = Path(config_dir)
    if not config_path.exists():
        config_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Oracle network admin directory created: {config_dir}")

    logger.info(
        f"Initializing Oracle client lib_dir={lib_dir}, config_dir={config_dir}"
    )
    oracledb.init_oracle_client(lib_dir=lib_dir, config_dir=config_dir)


# Lazy database connection pool
class LazyPool:
    def __init__(self, config_name="ORACLE_26AI_CONNECTION_STRING", **kwargs):
        self._pool = None
        self._kwargs = kwargs
        self._config_name = config_name
        self._lock = threading.RLock()

    def _ensure(self):
        with self._lock:
            if self._pool is None:
                dsn = self._kwargs.get("dsn")
                if not dsn or not str(dsn).strip():
                    logger.warning("DSN is empty; skip creating DB connection pool")
                    raise RuntimeError(f"{self._config_name} is not set")
                if str(
                    os.environ.get("DB_CONNECT_PRECHECK_ENABLED", "false")
                ).lower() in ("1", "true", "yes"):
                    try:
                        self._precheck_connectivity()
                    except socket.gaierror as e:
                        logger.warning(
                            f"DB connectivity precheck name resolution failed: {e}"
                        )
                    except Exception as e:
                        logger.warning(f"DB connectivity precheck failed: {e}")
                logger.info("Creating DB connection pool")
                self._pool = oracledb.create_pool(**self._kwargs)

    def _precheck_connectivity(self):
        dsn = str(self._kwargs.get("dsn") or "")
        host = None
        port = None
        try:
            after = parse_oracle_connection_string(dsn).dsn
            hp = after.split("/")[0]
            parts = hp.split(":")
            host = parts[0] if parts else None
            if len(parts) > 1:
                try:
                    port = int(parts[1])
                except Exception:
                    port = None
        except Exception as e:
            logger.error(f"_precheck_connectivity dsn parse error: {e}")
        t = float(os.environ.get("DB_CONNECT_PRECHECK_TIMEOUT", "3") or "3")
        if host:
            p = port or 1521
            try:
                with socket.create_connection((host, p), timeout=t):
                    _ = True
            except Exception as e:
                logger.error(f"DB precheck socket connect failed: {e}")

    def acquire(self):
        self._ensure()
        try:
            conn = self._pool.acquire()
            try:
                conn.ping()
            except Exception as e:
                logger.error(f"conn.ping failed, resetting pool: {e}")
                try:
                    conn.close()
                except Exception as e2:
                    logger.error(f"conn.close error: {e2}")
                self.reset()
                conn = self._pool.acquire()
                conn.ping()
            return conn
        except Exception as e:
            logger.error(f"acquire failed, resetting pool: {e}")
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
                except Exception as e:
                    logger.error(f"pool.close error: {e}")
            self._pool = None
            logger.info("Recreating DB connection pool")
            self._pool = oracledb.create_pool(**self._kwargs)

    def warmup(
        self, sessions: int = 1, test_query: Optional[str] = "SELECT 1 FROM DUAL"
    ):
        self._ensure()
        n = max(1, int(sessions or 1))
        for _ in range(n):
            with self.acquire() as conn:
                if test_query:
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(test_query)
                            _ = cursor.fetchmany(size=1)
                    except Exception as e:
                        logger.error(f"warmup test_query failed: {e}")
                        raise

    def healthy(self) -> bool:
        try:
            with self.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
                    _ = cursor.fetchmany(size=1)
            return True
        except Exception as e:
            logger.error(f"healthy check failed: {e}")
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
try:
    configured_vpd_users = normalize_vpd_login_users()
except VpdConfigurationError as exc:
    configured_vpd_users = ()
    logger.error("VPD login-user configuration is invalid: %s", exc)

vpd_pool = (
    LazyPool(
        config_name="ORACLE_VPD_RUNTIME_CONNECTION_STRING",
        dsn=os.environ.get("ORACLE_VPD_RUNTIME_CONNECTION_STRING", ""),
        min=0,
        max=8,
        increment=1,
        timeout=30,
        getmode=oracledb.POOL_GETMODE_WAIT,
    )
    if configured_vpd_users
    else None
)
logger.info(
    "Database pools configured (VPD runtime pool enabled=%s)",
    vpd_pool is not None,
)


# Configure Gradio theme
theme = Default(
    spacing_size="sm",
    font=[
        GoogleFont(name="Noto Sans JP"),
        GoogleFont(name="Roboto"),
        "Arial",
        "sans-serif",
    ],
).set()

# Create Gradio interface
reset_model_dropdown_registry()
with gr.Blocks(css=custom_css, theme=theme, title="クエリできすぎくん") as app:
    gr.Markdown(value="# クエリできすぎくん ", elem_classes="main_Header")
    gr.Markdown(
        value="### 開発者がSQLクエリを簡単に生成し、SQLの理解を深めるためのツール",
        elem_classes="sub_Header",
    )

    with gr.Tabs() as primary_tabs:
        with gr.TabItem(label="環境設定") as settings_tab:
            # 環境設定関連のタブを構築
            llm_model_settings_controls = build_settings_tab(pool)

        with gr.TabItem(label="データベース管理") as management_tab:
            # 管理機能タブを構築
            build_management_tab(pool, vpd_pool)

        with gr.TabItem(label="SQLの実行") as query_tab:
            # SQLの実行タブを構築
            query_notice = build_query_tab(pool, vpd_pool)

        with gr.TabItem(label="SelectAI 連携") as selectai_tab:
            (
                developer_features_tab,
                user_features_tab,
                selectai_feature_tabs,
                user_function_tabs,
                user_basic_tab,
                sql_learning_schema_setup,
                sql_learning_select_lessons,
            ) = build_selectai_tab(pool, vpd_pool)

        with gr.TabItem(label="AI チャット") as chat_tab:
            build_oci_chat_test_tab(pool)

    bind_llm_model_settings_events(llm_model_settings_controls)

    gr.Markdown(
        value="### 本ソフトウェアは検証評価用です。日常利用のための基本機能は備えていない点につきましてご理解をよろしくお願い申し上げます。",
        elem_classes="sub_Header",
    )
    gr.Markdown(value="### Developed by Oracle Japan", elem_classes="sub_Header")

    def _apply_access_visibility(request: gr.Request):
        role = user_role(request_username(request))
        navigation = access_navigation_for_role(role)
        primary_targets = {
            "settings": settings_tab.id,
            "selectai": selectai_tab.id,
        }
        selectai_targets = {
            "developer": developer_features_tab.id,
            "user": user_features_tab.id,
        }
        user_targets = {"basic": user_basic_tab.id}
        return (
            gr.Tabs(
                selected=primary_targets.get(navigation.primary_selection)
            ),
            gr.TabItem(visible=navigation.settings_visible),
            gr.TabItem(visible=navigation.management_visible),
            gr.TabItem(visible=navigation.query_visible),
            gr.TabItem(visible=navigation.selectai_visible),
            gr.TabItem(visible=navigation.chat_visible),
            gr.Tabs(
                selected=selectai_targets.get(
                    navigation.selectai_selection
                )
            ),
            gr.TabItem(visible=navigation.developer_features_visible),
            gr.TabItem(visible=navigation.user_features_visible),
            gr.Tabs(
                selected=user_targets.get(navigation.user_selection)
            ),
            gr.Markdown(
                value=_query_access_notice(role),
                visible=navigation.query_visible,
            ),
            gr.Accordion(
                visible=navigation.sql_learning_schema_setup_visible
            ),
            gr.Accordion(
                label=navigation.sql_learning_select_label
            ),
        )

    app.load(
        fn=_apply_access_visibility,
        outputs=[
            primary_tabs,
            settings_tab,
            management_tab,
            query_tab,
            selectai_tab,
            chat_tab,
            selectai_feature_tabs,
            developer_features_tab,
            user_features_tab,
            user_function_tabs,
            query_notice,
            sql_learning_schema_setup,
            sql_learning_select_lessons,
        ],
    )

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
        auth=do_auth,
    )
