"""OCI Chat Model ユーティリティモジュール.

このモジュールは、OCI GenAI Chat Modelとの対話機能を提供する
Gradio UIコンポーネントを実装します。
"""

import logging
import json
from typing import List, Optional

import gradio as gr
from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
from openai import AsyncOpenAI
from dotenv import find_dotenv, get_key
import asyncio
import os
import oci

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_GEMINI_MODELS = {
    "google.gemini-2.5-flash",
    "google.gemini-2.5-pro",
}


def _extract_text_from_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text:
            return text
        content = value.get("content")
        if content is not None:
            return _extract_text_from_content(content)
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            t = _extract_text_from_content(item)
            if t:
                parts.append(t)
        return "".join(parts)
    return ""


def _extract_stream_delta_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                text = _extract_text_from_content(content)
                if text:
                    return text

            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                text = _extract_text_from_content(content)
                if text:
                    return text

            content = choice.get("content")
            text = _extract_text_from_content(content)
            if text:
                return text

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = _extract_text_from_content(content)
        if text:
            return text

    content = payload.get("content")
    text = _extract_text_from_content(content)
    if text:
        return text

    text = payload.get("text")
    if isinstance(text, str) and text:
        return text

    return ""


async def _stream_oci_genai_chat_gemini(
    *,
    region: str,
    compartment_id: str,
    model_id: str,
    messages: List[dict],
):
    config = oci.config.from_file(file_location="/root/.oci/config")
    config["region"] = region

    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
    )

    from oci.generative_ai_inference.models import (
        AssistantMessage,
        ChatDetails,
        GenericChatRequest,
        OnDemandServingMode,
        SystemMessage,
        TextContent,
        UserMessage,
    )

    oci_messages = []
    for msg in messages:
        role = str(msg.get("role", "") or "").strip().lower()
        content_text = str(msg.get("content", "") or "")
        content = [TextContent(text=content_text)]
        if role == "system":
            oci_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            oci_messages.append(AssistantMessage(content=content))
        else:
            oci_messages.append(UserMessage(content=content))

    chat_request = GenericChatRequest(
        api_format=GenericChatRequest.API_FORMAT_GENERIC,
        messages=oci_messages,
        is_stream=True,
    )
    chat_details = ChatDetails(
        compartment_id=compartment_id,
        serving_mode=OnDemandServingMode(model_id=model_id),
        chat_request=chat_request,
    )

    q: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    exc: dict = {"value": None}

    def _push(item: Optional[str]):
        loop.call_soon_threadsafe(q.put_nowait, item)

    def _run():
        try:
            resp = client.chat(chat_details)
            data = getattr(resp, "data", None)
            if hasattr(data, "events"):
                logged_format = False
                for event in data.events():
                    event_data = getattr(event, "data", None)
                    if not event_data:
                        continue
                    if event_data == "[DONE]":
                        break
                    event_data_str = str(event_data)
                    chunk_text = ""
                    try:
                        payload = json.loads(event_data_str)
                        chunk_text = _extract_stream_delta_text(payload)
                        if not chunk_text and not logged_format:
                            logger.info(
                                "OCI GenAI stream event payload keys: %s",
                                sorted(list(payload.keys()))
                                if isinstance(payload, dict)
                                else type(payload),
                            )
                            logged_format = True
                    except Exception:
                        chunk_text = event_data_str

                    if isinstance(chunk_text, str):
                        chunk_text = chunk_text.strip("\r\n")
                    if chunk_text and chunk_text != "[DONE]":
                        _push(chunk_text)
            else:
                response_text = ""
                chat_response = getattr(data, "chat_response", None)
                choices = getattr(chat_response, "choices", None)
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        # Handle list of TextContent objects
                        for item in content:
                            if hasattr(item, "text"):
                                response_text += item.text
                if response_text:
                    _push(response_text)
        except Exception as e:
            # Gemini権限エラーの検出と改善
            error_msg = str(e)
            if "PERMISSION_DENIED" in error_msg or "aiplatform.endpoints.predict" in error_msg:
                logger.error(f"Gemini model access denied: {error_msg}")
                user_friendly_msg = (
                    "Geminiモデルへのアクセスが拒否されました。\n\n"
                    "原因: OCIテナントでGeminiモデルの利用が許可されていない可能性があります。\n\n"
                    "対処方法:\n"
                    "1. 別のモデル（GPTまたは他のOCIモデル）を選択して再度お試しください\n"
                    "2. Geminiモデルの利用が必要な場合は、OCI管理者にアクセス権限を確認してください\n"
                    "3. 一時的なエラーの場合は、少し時間をおいて再度お試しください"
                )
                exc["value"] = Exception(user_friendly_msg)
            else:
                exc["value"] = e
        finally:
            _push(None)

    bg_task = asyncio.create_task(asyncio.to_thread(_run))

    while True:
        item = await q.get()
        if item is None:
            break
        yield item
    await bg_task
    if exc["value"] is not None:
        raise exc["value"]


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
    logger.info(
        f"Message: {message[:100]}..." if len(message) > 100 else f"Message: {message}"
    )

    if not message.strip():
        logger.error("メッセージが未入力です")
        logger.warning("Empty message received")
        logger.info("=" * 50)
        yield history, ""
        return

    try:
        # Initialize region and compartment_id for non-GPT models
        region = None
        compartment_id = None
        
        if not chat_model.startswith("gpt-"):
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

        # Add user message to history first
        history.append({"role": "user", "content": message})
        # Then add empty assistant message for streaming
        history.append({"role": "assistant", "content": ""})

        if chat_model.startswith("gpt-"):
            logger.info(f"Using OpenAI client for model: {chat_model}")
            client = AsyncOpenAI()

            # Use standard Chat Completions API which is stable and widely supported
            # Responses API (v1/responses) is in preview and may not be available in all regions/accounts
            # causing 404 Not Found errors.
            logger.info("Calling OpenAI Chat Completions API...")
            stream = await client.chat.completions.create(
                model=chat_model, messages=messages, stream=True
            )

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

        elif chat_model in _GEMINI_MODELS:
            logger.info("Calling OCI GenAI API with streaming (native SDK)...")

            response_text = ""
            async for delta in _stream_oci_genai_chat_gemini(
                region=region,
                compartment_id=compartment_id,
                model_id=chat_model,
                messages=messages,
            ):
                response_text += delta
                history[-1] = {"role": "assistant", "content": response_text}
                yield history, ""

        else:
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
    with gr.Accordion(
        label="ℹ️ AI は不正確な情報を表示することがあるため、生成された回答を再確認するようにしてください。",
        open=True,
    ):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("モデル*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                chat_model_input = gr.Dropdown(
                                    show_label=False,
                                    choices=[
                                        "xai.grok-code-fast-1",
                                        "xai.grok-3",
                                        "xai.grok-3-fast",
                                        "xai.grok-4",
                                        "xai.grok-4-fast-non-reasoning",
                                        "xai.grok-4.20-non-reasoning",
                                        "google.gemini-2.5-flash",
                                        "google.gemini-2.5-pro",
                                        "meta.llama-4-scout-17b-16e-instruct",
                                        "gpt-4o",
                                        "gpt-5.1",
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
                            height=400,
                            show_copy_button=True,
                            avatar_images=(None, None),
                            type="messages",
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
                    yield (
                        gr.Markdown(
                            visible=True, value="⚠️ メッセージを入力してください"
                        ),
                        history,
                        message,
                    )
                    return

                if not chat_model.startswith("gpt-"):
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        yield (
                            gr.Markdown(
                                visible=True, value="❌ OCI設定が不足しています"
                            ),
                            history,
                            message,
                        )
                        return

                yield gr.Markdown(visible=True, value="⏳ 送信中..."), history, message
                last_hist, last_msg = history, message
                # コピーを渡して元の履歴が変更されないようにする
                history_copy = history.copy()
                for h, m in send_chat_message(message, history_copy, chat_model):
                    last_hist, last_msg = h, m
                    yield gr.Markdown(visible=True, value="⏳ 送信中..."), h, m
                yield gr.Markdown(visible=True, value="✅ 完了"), last_hist, last_msg
            except Exception as e:
                logger.error(f"send_chat_with_status error: {e}")
                yield (
                    gr.Markdown(visible=True, value=f"❌ エラー: {e}"),
                    history,
                    message,
                )

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
