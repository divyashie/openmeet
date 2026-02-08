"""
Tests for utils/config.py
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestConfigPaths:
    """Test path constants"""

    def test_resources_dir_is_valid_directory(self):
        from utils.config import RESOURCES_DIR
        assert RESOURCES_DIR.exists()
        assert RESOURCES_DIR.is_dir()

    def test_transcripts_dir_exists(self):
        from utils.config import TRANSCRIPTS_DIR
        assert TRANSCRIPTS_DIR.exists()

    def test_logs_dir_exists(self):
        from utils.config import LOGS_DIR
        assert LOGS_DIR.exists()

    def test_whisper_model_path_has_bin_extension(self):
        from utils.config import WHISPER_MODEL_PATH
        assert WHISPER_MODEL_PATH.suffix == ".bin"

    def test_whisper_model_path_contains_model_name(self):
        from utils.config import WHISPER_MODEL_PATH, settings
        model_name = settings.get("whisper_model")
        assert model_name in WHISPER_MODEL_PATH.name


class TestAudioSettings:
    """Test audio configuration constants"""

    def test_sample_rate(self):
        from utils.config import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_channels_mono(self):
        from utils.config import CHANNELS
        assert CHANNELS == 1

    def test_chunk_size(self):
        from utils.config import CHUNK_SIZE
        assert CHUNK_SIZE == 1024


class TestSettings:
    """Test Settings singleton"""

    def test_settings_has_defaults(self):
        from utils.config import settings
        assert settings.get("whisper_model") is not None
        assert settings.get("language") is not None
        assert settings.get("summary_format") is not None

    def test_settings_get_unknown_key_returns_none(self):
        from utils.config import settings
        assert settings.get("nonexistent_key_xyz") is None

    def test_settings_set_and_get(self):
        from utils.config import settings
        original = settings.get("language")
        settings.set("language", "fr")
        assert settings.get("language") == "fr"
        # Restore
        settings.set("language", original)

    def test_settings_all_returns_dict(self):
        from utils.config import settings
        all_settings = settings.all()
        assert isinstance(all_settings, dict)
        assert "whisper_model" in all_settings


class TestValidateSetup:
    """Test setup validation"""

    def test_validate_setup_returns_bool(self):
        from utils.config import validate_setup
        result = validate_setup()
        assert isinstance(result, bool)
