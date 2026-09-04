"""LLM model settings and shared Gradio model dropdown helpers."""

import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from dotenv import dotenv_values, find_dotenv, set_key

from utils.common_util import (
    LLM_DEFAULT_MODEL_ENV,
    LLM_SHOW_OPENAI_MODELS_ENV,
    LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
    LlmModelSettings,
    BASE_PROFILE_DEFAULT_REGION,
    CHICAGO_PROFILE_DEFAULT_REGION,
    get_chat_model_choices,
    get_effective_default_model,
    get_llm_model_settings,
    get_profile_default_region,
    validate_explicit_default_model,
)
from utils.vpd_util import require_admin

logger = logging.getLogger(__name__)

_MODEL_DROPDOWNS = []
_PROFILE_REGION_DROPDOWNS = []
_PROFILE_REGION_CHOICES = [
    BASE_PROFILE_DEFAULT_REGION,
    CHICAGO_PROFILE_DEFAULT_REGION,
]


@dataclass(frozen=True)
class LlmModelSettingsControls:
    """Gradio controls required to bind the settings events after UI creation."""

    tab: object
    show_chicago_checkbox: object
    show_openai_checkbox: object
    default_model_input: object
    save_button: object
    status_markdown: object


def reset_model_dropdown_registry():
    """Clear model-related component registrations before constructing an app."""
    _MODEL_DROPDOWNS.clear()
    _PROFILE_REGION_DROPDOWNS.clear()


def create_chat_model_dropdown(**kwargs):
    """Create and register a dropdown using the current LLM settings."""
    settings = get_llm_model_settings()
    dropdown = gr.Dropdown(
        choices=get_chat_model_choices(settings),
        value=get_effective_default_model(settings),
        **kwargs,
    )
    _MODEL_DROPDOWNS.append(dropdown)
    return dropdown


def create_profile_region_dropdown(**kwargs):
    """Create and register a SelectAI profile region dropdown."""
    settings = get_llm_model_settings()
    dropdown = gr.Dropdown(
        choices=_PROFILE_REGION_CHOICES,
        value=get_profile_default_region(settings),
        **kwargs,
    )
    _PROFILE_REGION_DROPDOWNS.append(dropdown)
    return dropdown


def get_registered_model_dropdowns():
    """Return a stable snapshot of all registered model dropdowns."""
    return tuple(_MODEL_DROPDOWNS)


def get_registered_profile_region_dropdowns():
    """Return a stable snapshot of registered SelectAI profile region dropdowns."""
    return tuple(_PROFILE_REGION_DROPDOWNS)


def create_model_dropdown_updates(settings, count):
    """Create identical Gradio updates for every registered model selector."""
    choices = get_chat_model_choices(settings)
    default_model = get_effective_default_model(settings)
    return [
        gr.Dropdown(choices=choices, value=default_model) for _ in range(count)
    ]


def create_profile_region_dropdown_updates(settings, count):
    """Create identical Gradio updates for every registered profile region."""
    default_region = get_profile_default_region(settings)
    return [
        gr.Dropdown(choices=_PROFILE_REGION_CHOICES, value=default_region)
        for _ in range(count)
    ]


def _dotenv_path(env_path=None):
    if env_path is not None:
        return Path(env_path)
    discovered = find_dotenv()
    return Path(discovered) if discovered else Path.cwd() / ".env"


def load_persisted_llm_model_settings(env_path=None):
    """Read settings from dotenv, with process values as a fallback."""
    path = _dotenv_path(env_path)
    values = dict(os.environ)
    if path.exists():
        persisted_values = dotenv_values(path)
        for key in (
            LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
            LLM_SHOW_OPENAI_MODELS_ENV,
            LLM_DEFAULT_MODEL_ENV,
        ):
            if key in persisted_values and persisted_values[key] is not None:
                values[key] = persisted_values[key]
    return get_llm_model_settings(values)


def create_persisted_llm_ui_updates(
    dropdown_count, env_path=None, profile_region_count=0
):
    """Create page-load updates from the latest persisted LLM settings."""
    settings = load_persisted_llm_model_settings(env_path)
    return [
        settings.show_us_chicago_1_models,
        settings.show_openai_models,
        settings.explicit_default_model,
        *create_model_dropdown_updates(settings, dropdown_count),
        *create_profile_region_dropdown_updates(settings, profile_region_count),
    ]


def persist_llm_model_settings(settings, env_path=None):
    """Atomically persist settings and refresh the current process environment."""
    path = _dotenv_path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_content = path.read_text(encoding="utf-8") if path.exists() else ""
    original_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(original_content)

        set_key(
            str(temporary_path),
            LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
            str(settings.show_us_chicago_1_models).lower(),
            quote_mode="never",
        )
        set_key(
            str(temporary_path),
            LLM_SHOW_OPENAI_MODELS_ENV,
            str(settings.show_openai_models).lower(),
            quote_mode="never",
        )
        set_key(
            str(temporary_path),
            LLM_DEFAULT_MODEL_ENV,
            settings.explicit_default_model,
            quote_mode="never",
        )
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    os.environ[LLM_SHOW_US_CHICAGO_1_MODELS_ENV] = str(
        settings.show_us_chicago_1_models
    ).lower()
    os.environ[LLM_SHOW_OPENAI_MODELS_ENV] = str(
        settings.show_openai_models
    ).lower()
    os.environ[LLM_DEFAULT_MODEL_ENV] = settings.explicit_default_model


def load_llm_model_settings(request: gr.Request):
    """Load persisted values when the settings tab is selected."""
    require_admin(request)
    settings = load_persisted_llm_model_settings()
    return (
        settings.show_us_chicago_1_models,
        settings.show_openai_models,
        settings.explicit_default_model,
    )


def build_llm_model_settings_tab(tab):
    """Build the accessible LLM model settings form."""
    settings = get_llm_model_settings()
    with gr.Accordion(
        label=(
            "ℹ️ モデル一覧に表示するモデルグループと"
            "デフォルトモデルを設定します。"
        ),
        open=True,
    ):
        show_chicago_checkbox = gr.Checkbox(
            value=settings.show_us_chicago_1_models,
            label="シカゴリージョン提供モデルを表示",
            info=(
                "シカゴリージョン（us-chicago-1）にて利用できる "
                "xai.grok-4.3 と meta.llama-4-scout-17b-16e-instruct "
                "がモデル一覧に追加されます。"
            ),
            interactive=True,
        )
        show_openai_checkbox = gr.Checkbox(
            value=settings.show_openai_models,
            label="OpenAI APIモデルを表示",
            info=(
                "gpt-4o、gpt-5.1がモデル一覧に追加されます。"
                "別途 OpenAI の APIキーが必要です"
                "（「OpenAI設定」タブで設定）。"
            ),
            interactive=True,
        )
        default_model_input = gr.Textbox(
            value=settings.explicit_default_model,
            label="デフォルトモデル",
            info=(
                "未入力の場合は自動で選択されます"
                "（シカゴリージョン提供モデルがオン: xai.grok-4.3 / "
                "オフ: openai.gpt-oss-120b）"
            ),
            placeholder="モデル名を入力（空欄の場合は自動選択）",
            lines=1,
            interactive=True,
        )
        save_button = gr.Button(value="保存", variant="primary")
        status_markdown = gr.Markdown(visible=False)

    return LlmModelSettingsControls(
        tab=tab,
        show_chicago_checkbox=show_chicago_checkbox,
        show_openai_checkbox=show_openai_checkbox,
        default_model_input=default_model_input,
        save_button=save_button,
        status_markdown=status_markdown,
    )


def bind_llm_model_settings_events(app, controls):
    """Bind load/save events after every model dropdown has been registered."""
    dropdowns = get_registered_model_dropdowns()
    profile_region_dropdowns = get_registered_profile_region_dropdowns()

    def refresh_from_persisted_settings():
        return create_persisted_llm_ui_updates(
            len(dropdowns), profile_region_count=len(profile_region_dropdowns)
        )

    app.load(
        refresh_from_persisted_settings,
        outputs=[
            controls.show_chicago_checkbox,
            controls.show_openai_checkbox,
            controls.default_model_input,
            *dropdowns,
            *profile_region_dropdowns,
        ],
        queue=False,
        show_progress="hidden",
    )

    controls.tab.select(
        load_llm_model_settings,
        outputs=[
            controls.show_chicago_checkbox,
            controls.show_openai_checkbox,
            controls.default_model_input,
        ],
    )

    def save_and_refresh(
        show_chicago, show_openai, explicit_default_model, request: gr.Request
    ):
        require_admin(request)
        skipped_updates = [
            gr.skip() for _ in (*dropdowns, *profile_region_dropdowns)
        ]
        yield [
            gr.Markdown(visible=True, value="⏳ LLMモデル設定を保存しています..."),
            *skipped_updates,
        ]

        settings = LlmModelSettings(
            show_us_chicago_1_models=bool(show_chicago),
            show_openai_models=bool(show_openai),
            explicit_default_model=str(explicit_default_model or "").strip(),
        )
        if not validate_explicit_default_model(settings):
            yield [
                gr.Markdown(
                    visible=True,
                    value=(
                        "❌ デフォルトモデルは現在の表示対象に含まれていません。"
                        "対応するモデルグループを有効にするか、表示対象のモデル名を"
                        "入力してください。"
                    ),
                ),
                *skipped_updates,
            ]
            return

        try:
            persist_llm_model_settings(settings)
        except Exception as exc:
            logger.exception("Failed to save LLM model settings")
            yield [
                gr.Markdown(
                    visible=True,
                    value=f"❌ LLMモデル設定の保存に失敗しました: {exc}",
                ),
                *skipped_updates,
            ]
            return

        default_model = get_effective_default_model(settings)
        yield [
            gr.Markdown(
                visible=True,
                value=(
                    "✅ LLMモデル設定を保存しました。"
                    f"デフォルトモデル: `{default_model}`"
                ),
            ),
            *create_model_dropdown_updates(settings, len(dropdowns)),
            *create_profile_region_dropdown_updates(
                settings, len(profile_region_dropdowns)
            ),
        ]

    controls.save_button.click(
        save_and_refresh,
        inputs=[
            controls.show_chicago_checkbox,
            controls.show_openai_checkbox,
            controls.default_model_input,
        ],
        outputs=[controls.status_markdown, *dropdowns, *profile_region_dropdowns],
    )
