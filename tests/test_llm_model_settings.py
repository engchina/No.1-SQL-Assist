"""Tests for configurable LLM model visibility and defaults."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.common_util import (
    BASE_DEFAULT_MODEL,
    CHAT_MODEL_CATALOG,
    CHICAGO_DEFAULT_MODEL,
    LLM_DEFAULT_MODEL_ENV,
    LLM_SHOW_OPENAI_MODELS_ENV,
    LLM_SHOW_US_CHICAGO_1_MODELS_ENV,
    LlmModelSettings,
    get_chat_model_choices,
    get_effective_default_model,
    get_llm_model_settings,
    validate_explicit_default_model,
)
from utils.llm_model_util import (
    create_model_dropdown_updates,
    load_persisted_llm_model_settings,
    persist_llm_model_settings,
)


class LlmModelSettingsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
