"""Tests for configurable LLM model visibility and defaults."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from utils.common_util import (
    BASE_DEFAULT_MODEL,
    BASE_PROFILE_DEFAULT_REGION,
    CHAT_MODEL_CATALOG,
    CHICAGO_DEFAULT_MODEL,
    CHICAGO_PROFILE_DEFAULT_REGION,
    LLM_DEFAULT_MODEL_ENV,
    LLM_SHOW_OPENAI_MODELS_ENV,
    LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
    LlmModelSettings,
    get_chat_model_choices,
    get_effective_default_model,
    get_llm_model_settings,
    get_profile_default_region,
    validate_explicit_default_model,
)
from utils.llm_model_util import (
    bind_llm_model_settings_events,
    build_llm_model_settings_tab,
    create_chat_model_dropdown,
    create_model_dropdown_updates,
    create_persisted_llm_ui_updates,
    create_profile_region_dropdown,
    create_profile_region_dropdown_updates,
    load_persisted_llm_model_settings,
    persist_llm_model_settings,
    reset_model_dropdown_registry,
)


class LlmModelSettingsTest(unittest.TestCase):
    def test_llm_settings_ui_uses_requested_copy(self):
        with gr.Blocks() as demo:
            with gr.TabItem(label="LLM設定") as settings_tab:
                controls = build_llm_model_settings_tab(settings_tab)

        accordion = next(
            component
            for component in demo.config["components"]
            if component["type"] == "accordion"
        )
        self.assertEqual(
            accordion["props"]["label"],
            "ℹ️ モデル一覧に表示するモデルグループと"
            "デフォルトモデルを設定します。",
        )
        self.assertEqual(controls.tab.label, "LLM設定")
        self.assertEqual(
            controls.show_chicago_checkbox.label,
            "シカゴリージョン提供モデルを表示",
        )
        self.assertEqual(
            controls.show_chicago_checkbox.info,
            "シカゴリージョン（us-chicago-1）にて利用できる "
            "xai.grok-4.3 と meta.llama-4-scout-17b-16e-instruct "
            "がモデル一覧に追加されます。",
        )
        self.assertEqual(
            controls.show_openai_checkbox.label,
            "OpenAI APIモデルを表示",
        )
        self.assertEqual(
            controls.show_openai_checkbox.info,
            "gpt-4o、gpt-5.1がモデル一覧に追加されます。"
            "別途 OpenAI の APIキーが必要です"
            "（「OpenAI設定」タブで設定）。",
        )
        self.assertEqual(
            controls.default_model_input.info,
            "未入力の場合は自動で選択されます"
            "（シカゴリージョン提供モデルがオン: xai.grok-4.3 / "
            "オフ: openai.gpt-oss-120b）",
        )

    def test_missing_settings_use_requested_defaults(self):
        settings = get_llm_model_settings({})

        self.assertTrue(settings.show_us_chicago_1_models)
        self.assertFalse(settings.show_openai_models)
        self.assertEqual(settings.explicit_default_model, "")
        self.assertEqual(get_effective_default_model(settings), CHICAGO_DEFAULT_MODEL)
        self.assertIn(CHICAGO_DEFAULT_MODEL, get_chat_model_choices(settings))
        self.assertNotIn("gpt-4o", get_chat_model_choices(settings))

    def test_blank_default_uses_gpt_oss_when_chicago_is_hidden(self):
        settings = LlmModelSettings(show_us_chicago_1_models=False)

        self.assertEqual(get_effective_default_model(settings), BASE_DEFAULT_MODEL)
        self.assertIn(BASE_DEFAULT_MODEL, get_chat_model_choices(settings))

    def test_profile_default_region_follows_chicago_visibility(self):
        hidden = LlmModelSettings(show_us_chicago_1_models=False)
        visible = LlmModelSettings(show_us_chicago_1_models=True)

        self.assertEqual(
            get_profile_default_region(hidden), BASE_PROFILE_DEFAULT_REGION
        )
        self.assertEqual(
            get_profile_default_region(visible),
            CHICAGO_PROFILE_DEFAULT_REGION,
        )

    def test_model_groups_cover_all_switch_combinations(self):
        expected_catalog = [model for model, _group in CHAT_MODEL_CATALOG]
        cases = (
            (False, False, False, False),
            (False, True, False, True),
            (True, False, True, False),
            (True, True, True, True),
        )

        for show_chicago, show_openai, has_chicago, has_openai in cases:
            with self.subTest(
                show_chicago=show_chicago, show_openai=show_openai
            ):
                settings = LlmModelSettings(
                    show_us_chicago_1_models=show_chicago,
                    show_openai_models=show_openai,
                )
                choices = get_chat_model_choices(settings)
                self.assertEqual(
                    CHICAGO_DEFAULT_MODEL in choices, has_chicago
                )
                self.assertEqual(
                    "meta.llama-4-scout-17b-16e-instruct" in choices,
                    has_chicago,
                )
                self.assertEqual("gpt-4o" in choices, has_openai)
                self.assertEqual("gpt-5.1" in choices, has_openai)
                self.assertIn(BASE_DEFAULT_MODEL, choices)
                self.assertEqual(
                    choices,
                    [model for model in expected_catalog if model in choices],
                )

    def test_explicit_default_must_be_visible(self):
        override = LlmModelSettings(
            show_us_chicago_1_models=True,
            explicit_default_model="cohere.command-a-03-2025",
        )
        valid = LlmModelSettings(
            show_us_chicago_1_models=False,
            explicit_default_model=BASE_DEFAULT_MODEL,
        )
        hidden = LlmModelSettings(
            show_us_chicago_1_models=False,
            explicit_default_model=CHICAGO_DEFAULT_MODEL,
        )
        unknown = LlmModelSettings(explicit_default_model="unknown-model")

        self.assertTrue(validate_explicit_default_model(override))
        self.assertEqual(
            get_effective_default_model(override), "cohere.command-a-03-2025"
        )
        self.assertTrue(validate_explicit_default_model(valid))
        self.assertEqual(get_effective_default_model(valid), BASE_DEFAULT_MODEL)
        self.assertFalse(validate_explicit_default_model(hidden))
        self.assertEqual(get_effective_default_model(hidden), BASE_DEFAULT_MODEL)
        self.assertFalse(validate_explicit_default_model(unknown))
        self.assertEqual(
            get_effective_default_model(unknown), CHICAGO_DEFAULT_MODEL
        )

    def test_persist_round_trip_preserves_other_dotenv_values(self):
        settings = LlmModelSettings(
            show_us_chicago_1_models=False,
            show_openai_models=True,
            explicit_default_model="gpt-4o",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text("EXISTING_SETTING=keep\n", encoding="utf-8")
            old_environment = {
                key: os.environ.get(key)
                for key in (
                    LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
                    LLM_SHOW_OPENAI_MODELS_ENV,
                    LLM_DEFAULT_MODEL_ENV,
                )
            }
            try:
                persist_llm_model_settings(settings, env_path)
                loaded = load_persisted_llm_model_settings(env_path)
            finally:
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertEqual(loaded, settings)
            persisted_text = env_path.read_text(encoding="utf-8")
            self.assertIn("EXISTING_SETTING=keep", persisted_text)
            self.assertIn("LLM_SHOW_US_CHICAGO_1_MODELS=false", persisted_text)
            self.assertIn("LLM_SHOW_OPENAI_MODELS=true", persisted_text)
            self.assertIn("LLM_DEFAULT_MODEL=gpt-4o", persisted_text)

    def test_failed_atomic_save_does_not_modify_existing_dotenv(self):
        settings = LlmModelSettings()
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            original = "EXISTING_SETTING=keep\n"
            env_path.write_text(original, encoding="utf-8")

            with patch(
                "utils.llm_model_util.set_key",
                side_effect=RuntimeError("write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    persist_llm_model_settings(settings, env_path)

            self.assertEqual(env_path.read_text(encoding="utf-8"), original)

    def test_dropdown_updates_share_choices_and_effective_default(self):
        settings = LlmModelSettings(
            show_us_chicago_1_models=False,
            show_openai_models=True,
        )

        updates = create_model_dropdown_updates(settings, 3)

        self.assertEqual(len(updates), 3)
        self.assertTrue(all(update.choices == updates[0].choices for update in updates))
        self.assertTrue(
            all(update.value == BASE_DEFAULT_MODEL for update in updates)
        )

    def test_profile_region_updates_share_effective_default(self):
        settings = LlmModelSettings(show_us_chicago_1_models=False)

        updates = create_profile_region_dropdown_updates(settings, 2)

        self.assertEqual(len(updates), 2)
        for update in updates:
            choices = [value for _label, value in update.choices]
            self.assertEqual(
                choices,
                [BASE_PROFILE_DEFAULT_REGION, CHICAGO_PROFILE_DEFAULT_REGION],
            )
            self.assertEqual(update.value, BASE_PROFILE_DEFAULT_REGION)

    def test_page_load_updates_use_latest_persisted_settings(self):
        cases = (
            (False, False, "", BASE_DEFAULT_MODEL),
            (False, True, "gpt-4o", "gpt-4o"),
            (True, False, CHICAGO_DEFAULT_MODEL, CHICAGO_DEFAULT_MODEL),
            (True, True, "gpt-5.1", "gpt-5.1"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            for show_chicago, show_openai, explicit_default, expected in cases:
                with self.subTest(
                    show_chicago=show_chicago,
                    show_openai=show_openai,
                    explicit_default=explicit_default,
                ):
                    env_path.write_text(
                        "\n".join(
                            (
                                "LLM_SHOW_US_CHICAGO_1_MODELS="
                                f"{str(show_chicago).lower()}",
                                "LLM_SHOW_OPENAI_MODELS="
                                f"{str(show_openai).lower()}",
                                f"LLM_DEFAULT_MODEL={explicit_default}",
                                "",
                            )
                        ),
                        encoding="utf-8",
                    )
                    stale_process_values = {
                        LLM_SHOW_US_CHICAGO_1_MODELS_ENV: str(
                            not show_chicago
                        ).lower(),
                        LLM_SHOW_OPENAI_MODELS_ENV: str(
                            not show_openai
                        ).lower(),
                        LLM_DEFAULT_MODEL_ENV: "stale-model",
                    }
                    with patch.dict(
                        os.environ, stale_process_values, clear=False
                    ):
                        updates = create_persisted_llm_ui_updates(
                            2, env_path, profile_region_count=1
                        )

                    self.assertEqual(
                        updates[:3],
                        [show_chicago, show_openai, explicit_default],
                    )
                    dropdown_updates = updates[3:5]
                    self.assertEqual(len(dropdown_updates), 2)
                    for dropdown_update in dropdown_updates:
                        choices = [
                            value
                            for _label, value in dropdown_update.choices
                        ]
                        self.assertEqual(
                            CHICAGO_DEFAULT_MODEL in choices, show_chicago
                        )
                        self.assertEqual(
                            "meta.llama-4-scout-17b-16e-instruct" in choices,
                            show_chicago,
                        )
                        self.assertEqual("gpt-4o" in choices, show_openai)
                        self.assertEqual("gpt-5.1" in choices, show_openai)
                        self.assertIn(BASE_DEFAULT_MODEL, choices)
                        self.assertEqual(dropdown_update.value, expected)
                    region_update = updates[5]
                    self.assertEqual(
                        region_update.value,
                        CHICAGO_PROFILE_DEFAULT_REGION
                        if show_chicago
                        else BASE_PROFILE_DEFAULT_REGION,
                    )

    def test_page_load_event_refreshes_controls_and_registered_dropdowns(self):
        reset_model_dropdown_registry()
        self.addCleanup(reset_model_dropdown_registry)
        with gr.Blocks() as demo:
            with gr.TabItem(label="LLM設定") as settings_tab:
                controls = build_llm_model_settings_tab(settings_tab)
            first_dropdown = create_chat_model_dropdown(label="Model 1")
            second_dropdown = create_chat_model_dropdown(label="Model 2")
            profile_region = create_profile_region_dropdown(label="Region")
            bind_llm_model_settings_events(demo, controls)

        load_dependencies = [
            dependency
            for dependency in demo.config["dependencies"]
            if any(target[1] == "load" for target in dependency["targets"])
        ]
        self.assertEqual(len(load_dependencies), 1)
        load_dependency = load_dependencies[0]
        self.assertEqual(
            load_dependency["outputs"],
            [
                controls.show_chicago_checkbox._id,
                controls.show_openai_checkbox._id,
                controls.default_model_input._id,
                first_dropdown._id,
                second_dropdown._id,
                profile_region._id,
            ],
        )
        self.assertFalse(load_dependency["queue"])
        self.assertEqual(load_dependency["show_progress"], "hidden")

        click_dependencies = [
            dependency
            for dependency in demo.config["dependencies"]
            if any(target[1] == "click" for target in dependency["targets"])
        ]
        save_dependency = next(
            dependency
            for dependency in click_dependencies
            if dependency["outputs"]
            and dependency["outputs"][0] == controls.status_markdown._id
        )
        self.assertEqual(
            save_dependency["outputs"],
            [
                controls.status_markdown._id,
                first_dropdown._id,
                second_dropdown._id,
                profile_region._id,
            ],
        )


if __name__ == "__main__":
    unittest.main()
