"""
Settings management for OpenMeet
3-layer precedence: defaults -> data/settings.json -> .env overrides
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
SETTINGS_FILE = PROJECT_ROOT / "data" / "settings.json"

load_dotenv(ENV_PATH)


class Settings:
    """Runtime settings with persistence and .env overrides."""

    DEFAULTS = {
        "audio_device_index": None,
        "whisper_model": "tiny",
        "language": "en",
        "summary_format": "detailed",
        "diarization_enabled": False,
        "huggingface_token": "",
        "ollama_model": "llama3.2:latest",
        "log_level": "INFO",
    }

    def __init__(self):
        self._settings = dict(self.DEFAULTS)
        self._load_from_file()
        self._apply_env_overrides()

    def _load_from_file(self):
        """Load persisted settings from JSON."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                self._settings.update(saved)
            except (json.JSONDecodeError, IOError):
                pass

    def _apply_env_overrides(self):
        """.env variables override saved settings."""
        env_map = {
            "OPENMEET_WHISPER_MODEL": "whisper_model",
            "OPENMEET_LANGUAGE": "language",
            "OPENMEET_SUMMARY_FORMAT": "summary_format",
            "OPENMEET_OLLAMA_MODEL": "ollama_model",
            "OPENMEET_LOG_LEVEL": "log_level",
            "HUGGINGFACE_TOKEN": "huggingface_token",
        }
        for env_key, setting_key in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                self._settings[setting_key] = val

    def get(self, key):
        return self._settings.get(key, self.DEFAULTS.get(key))

    def set(self, key, value):
        self._settings[key] = value

    def save(self):
        """Persist current settings to JSON."""
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self._settings, f, indent=2)

    def all(self):
        return dict(self._settings)
