"""環境設定タブモジュール

このモジュールは、環境設定関連のすべてのタブを統合的に管理します。
"""

import gradio as gr
import logging
from utils.oci_util import (
    build_oci_genai_tab,
    build_oracle_ai_database_tab,
    build_oci_embedding_test_tab,
    build_openai_settings_tab,
    load_openai_settings,
)

logger = logging.getLogger(__name__)

def build_settings_tab(pool):
    """環境設定タブの内容を構築する
    
    Args:
        pool: データベース接続プール
    """
    with gr.Tabs():
        with gr.TabItem(label="OCI 認証情報設定"):
            # OCI GenAI設定タブを構築
            build_oci_genai_tab(pool)
        
        with gr.TabItem(label="Autonomous AI Database"):
            # Oracle AI Database タブを追加
            build_oracle_ai_database_tab(pool)
        
        with gr.TabItem(label="Embeddingテスト"):
            # OCI GenAI Embeddingテストタブを構築
            build_oci_embedding_test_tab(pool)
        
        with gr.TabItem(label="OpenAI設定") as openai_tab:
            # OpenAI設定タブを構築
            openai_base_url_input, openai_api_key_input = build_openai_settings_tab(pool)
    
    # OpenAI設定タブ選択時に設定を読み込む
    openai_tab.select(load_openai_settings, outputs=[openai_base_url_input, openai_api_key_input])
