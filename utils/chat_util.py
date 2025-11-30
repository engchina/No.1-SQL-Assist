"""OCI Chat Model ユーティリティモジュール.

このモジュールは、OCI GenAI Chat Modelとの対話機能を提供する
Gradio UIコンポーネントを実装します。
"""

import logging
from typing import List

import gradio as gr
from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
from dotenv import find_dotenv, get_key
import asyncio
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_oci_region():
    """OCI設定ファイルからリージョン情報を取得する.

    Returns:
        str: OCIリージョン名
    """
    try:
        oci_config_path = find_dotenv("/root/.oci/config")
        region = get_key(oci_config_path, "region") if oci_config_path else None
        return region
    except Exception as e:
        logger.error(f"Error reading OCI config: {e}")
        return None


def get_compartment_id():
    """環境変数からCompartment IDを取得する.

    Returns:
        str: OCI Compartment OCID
    """
    compartment_id = os.environ.get("OCI_COMPARTMENT_OCID", "")
    if not compartment_id:
        logger.error("OCI_COMPARTMENT_OCID environment variable is not set")
    return compartment_id


async def send_chat_message_async(
    message: str,
    history: List[dict],
    chat_model: str,
):
    """OCI Chat Modelにメッセージを送信してストリーミング応答を取得する（非同期）.

    Args:
        message: ユーザーメッセージ
        history: 会話履歴 (messages形式)

    Yields:
        tuple: (更新された会話履歴, 空文字列（入力クリア用）)
    """
    logger.info("=" * 50)
    logger.info(f"Sending chat message to model: {chat_model}")
    logger.info(f"Message: {message[:100]}..." if len(message) > 100 else f"Message: {message}")

    if not message.strip():
        logger.error("メッセージが未入力です")
        logger.warning("Empty message received")
        logger.info("=" * 50)
        yield history, ""
        return

    try:
        # Get region and compartment ID
        region = get_oci_region()
        compartment_id = get_compartment_id()
        
        if not region:
            logger.error("OCI region not found")
            logger.info("=" * 50)
            yield history, message
            return
        
        if not compartment_id:
            logger.error("Compartment ID not found")
            logger.info("=" * 50)
            yield history, message
            return

        logger.info(f"Using region: {region}")
        logger.info(f"Using compartment: {compartment_id[:20]}...")

        # Build messages for the API from history
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})

        # Create OCI OpenAI client
        client = AsyncOciOpenAI(
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            auth=OciUserPrincipalAuth(),
            compartment_id=compartment_id,
        )

        logger.info("Calling OCI GenAI API with streaming...")
        
        # Call the API with streaming
        stream = await client.chat.completions.create(
            model=chat_model,
            messages=messages,
            stream=True,
        )

        # Add user message to history first
        history.append({"role": "user", "content": message})
        # Then add empty assistant message for streaming
        history.append({"role": "assistant", "content": ""})
        
        # Collect streaming response
        response_text = ""
        
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    response_text += delta.content
                    # Update the last message in history with accumulated response
                    history[-1] = {"role": "assistant", "content": response_text}
                    yield history, ""
        
        if not response_text:
            logger.warning("Empty response from API")
            response_text = "応答を取得できませんでした。"
            history[-1] = {"role": "assistant", "content": response_text}
            yield history, ""

        logger.info(f"Response length: {len(response_text)} characters")
        logger.info("✓ Chat message processed successfully")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        logger.exception("Full traceback:")
        logger.info("=" * 50)
        yield history, message


def send_chat_message(
    message: str,
    history: List[dict],
    chat_model: str,
):
    """OCI Chat Modelにメッセージを送信して応答を取得する（同期ラッパー）.

    Args:
        message: ユーザーメッセージ
        history: 会話履歴 (messages形式)

    Yields:
        tuple: (更新された会話履歴, 空文字列（入力クリア用）)
    """
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async_gen = send_chat_message_async(message, history, chat_model)
        while True:
            try:
                result = loop.run_until_complete(async_gen.__anext__())
                yield result
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def clear_chat():
    """チャット履歴をクリアする.

    Returns:
        tuple: (空の履歴, 空のメッセージ)
    """
    logger.info("Chat history cleared")
    return [], ""


def build_oci_chat_test_tab(pool):
    """OCI Chat Model タブのUIを構築する.

    Args:
        pool: データベース接続プール（使用しない）

    Returns:
        gr.TabItem: Chat UIタブ
    """

    with gr.TabItem(label="AI チャット") as tab_chat:
        with gr.Accordion(label="", open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("モデル", elem_classes="input-label")
                                with gr.Column(scale=5):
                                    chat_model_input = gr.Dropdown(
                                        show_label=False,
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
                                        container=False,
                                    )
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("")
                    with gr.Row():
                        with gr.Column(scale=1):
                            chatbot = gr.Chatbot(
                                label="会話履歴",
                                height=350,
                                show_copy_button=True,
                                avatar_images=(None, None),
                                type='messages',
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            msg_input = gr.Textbox(
                                value="こんにちわ",
                                label="💬 メッセージ",
                                placeholder="メッセージを入力してください（Enterで改行、Shift＋Enterで送信）",
                                lines=2,
                                max_lines=8,
                                container=False,
                            )

                    with gr.Row():
                        clear_btn = gr.Button("クリア", scale=1)
                        send_btn = gr.Button("送信", variant="primary", scale=1)
                    with gr.Row():
                        chat_status_md = gr.Markdown(visible=False)

            # Event handlers with status markdown under the button
            def send_chat_with_status(message, history, chat_model):
                try:
                    from utils.chat_util import get_oci_region, get_compartment_id
                    if not str(message or "").strip():
                        # empty message
                        yield gr.Markdown(visible=True, value="⚠️ メッセージを入力してください"), history, message
                        return
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        yield gr.Markdown(visible=True, value="❌ OCI設定が不足しています"), history, message
                        return
                    yield gr.Markdown(visible=True, value="⏳ 送信中..."), history, message
                    last_hist, last_msg = history, message
                    for h, m in send_chat_message(message, history, chat_model):
                        last_hist, last_msg = h, m
                        yield gr.Markdown(visible=True, value="⏳ 送信中..."), h, m
                    yield gr.Markdown(visible=True, value="✅ 完了"), last_hist, last_msg
                except Exception as e:
                    logger.error(f"send_chat_with_status error: {e}")
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), history, message

            def clear_chat_with_status():
                h, m = clear_chat()
                return gr.Markdown(visible=False), h, m

            send_btn.click(
                send_chat_with_status,
                inputs=[msg_input, chatbot, chat_model_input],
                outputs=[chat_status_md, chatbot, msg_input],
            )

            msg_input.submit(
                send_chat_with_status,
                inputs=[msg_input, chatbot, chat_model_input],
                outputs=[chat_status_md, chatbot, msg_input],
            )

            clear_btn.click(
                clear_chat_with_status,
                inputs=[],
                outputs=[chat_status_md, chatbot, msg_input],
            )

    return tab_chat
