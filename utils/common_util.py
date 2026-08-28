"""共通ユーティリティモジュール.

プロジェクト全体で使用される共通の関数を提供します。
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LLM_SHOW_US_CHICAGO_1_MODELS_ENV = "LLM_SHOW_US_CHICAGO_1_MODELS"
LLM_SHOW_OPENAI_MODELS_ENV = "LLM_SHOW_OPENAI_MODELS"
LLM_DEFAULT_MODEL_ENV = "LLM_DEFAULT_MODEL"

CHICAGO_DEFAULT_MODEL = "xai.grok-4.3"
BASE_DEFAULT_MODEL = "openai.gpt-oss-120b"

MODEL_GROUP_BASE = "base"
MODEL_GROUP_CHICAGO = "us-chicago-1"
MODEL_GROUP_OPENAI = "openai"

# Keep this order stable so enabling every group reproduces the existing UI.
CHAT_MODEL_CATALOG = (
    ("xai.grok-4.3", MODEL_GROUP_CHICAGO),
    ("cohere.command-a-03-2025", MODEL_GROUP_BASE),
    ("google.gemini-2.5-flash", MODEL_GROUP_BASE),
    ("google.gemini-2.5-pro", MODEL_GROUP_BASE),
    ("meta.llama-4-scout-17b-16e-instruct", MODEL_GROUP_CHICAGO),
    (BASE_DEFAULT_MODEL, MODEL_GROUP_BASE),
    ("gpt-4o", MODEL_GROUP_OPENAI),
    ("gpt-5.1", MODEL_GROUP_OPENAI),
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class LlmModelSettings:
    """LLM model visibility and default selection settings."""

    show_us_chicago_1_models: bool = True
    show_openai_models: bool = False
    explicit_default_model: str = ""


def parse_env_bool(value, default):
    """Parse a dotenv-style boolean, falling back for missing/invalid values."""
    if value is None or not str(value).strip():
        return default

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    logger.warning("Invalid boolean setting %r; using default %s", value, default)
    return default


def get_llm_model_settings(environ=None):
    """Load LLM model settings from an environment-like mapping."""
    values = os.environ if environ is None else environ
    return LlmModelSettings(
        show_us_chicago_1_models=parse_env_bool(
            values.get(LLM_SHOW_US_CHICAGO_1_MODELS_ENV), True
        ),
        show_openai_models=parse_env_bool(
            values.get(LLM_SHOW_OPENAI_MODELS_ENV), False
        ),
        explicit_default_model=str(
            values.get(LLM_DEFAULT_MODEL_ENV) or ""
        ).strip(),
    )


def get_chat_model_choices(settings=None):
    """Return the visible model choices for the supplied settings."""
    current = settings or get_llm_model_settings()
    enabled_groups = {MODEL_GROUP_BASE}
    if current.show_us_chicago_1_models:
        enabled_groups.add(MODEL_GROUP_CHICAGO)
    if current.show_openai_models:
        enabled_groups.add(MODEL_GROUP_OPENAI)

    return [
        model_name
        for model_name, group in CHAT_MODEL_CATALOG
        if group in enabled_groups
    ]


def get_automatic_default_model(settings=None):
    """Resolve the automatic default from the Chicago visibility switch."""
    current = settings or get_llm_model_settings()
    if current.show_us_chicago_1_models:
        return CHICAGO_DEFAULT_MODEL
    return BASE_DEFAULT_MODEL


def validate_explicit_default_model(settings):
    """Return whether an explicit default is empty or currently visible."""
    return (
        not settings.explicit_default_model
        or settings.explicit_default_model in get_chat_model_choices(settings)
    )


def get_effective_default_model(settings=None):
    """Return a safe default, falling back when external config is invalid."""
    current = settings or get_llm_model_settings()
    if validate_explicit_default_model(current) and current.explicit_default_model:
        return current.explicit_default_model

    if current.explicit_default_model:
        logger.warning(
            "Configured default model %r is hidden or unknown; using automatic default",
            current.explicit_default_model,
        )
    return get_automatic_default_model(current)


def get_dict_value(dictionary, key, default_value=None):
    """辞書から値を安全に取得する.

    Args:
        dictionary (dict): 値を取得する辞書
        key: 辞書で検索するキー
        default_value: キーが見つからない場合に返す値（デフォルト: None）

    Returns:
        キーが存在する場合はその値、存在しない場合はdefault_value
    """
    try:
        return dictionary[key]
    except KeyError:
        return default_value


def remove_comments(sql_str: str) -> str:
    """SQLからコメントを除去する.

    Args:
        sql_str (str): コメントを除去するSQL

    Returns:
        str: コメントが除去されたSQL
    """
    if not sql_str:
        return ""

    # 行単位で処理
    lines = sql_str.split("\n")
    result_lines = []

    for line in lines:
        # '--'が含まれるか確認
        if "--" in line:
            # '--'の位置を探す
            # ただし、文字列リテラル内の'--'は無視する必要がある
            # 簡易的な実装として、シングルクォートの外にある'--'以降を削除する

            in_quote = False
            comment_start = -1

            for i, char in enumerate(line):
                if char == "'":
                    in_quote = not in_quote
                elif (
                    char == "-"
                    and i + 1 < len(line)
                    and line[i + 1] == "-"
                    and not in_quote
                ):
                    comment_start = i
                    break

            if comment_start != -1:
                line = line[:comment_start]

        result_lines.append(line)

    return "\n".join(result_lines)
