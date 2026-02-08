"""
Settings management for OpenMeet
3-layer precedence: defaults -> data/settings.json -> .env overrides
"""
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv


def _get_data_root():
    """Writable directory for user data (settings, transcripts)."""
    if getattr(sys, 'frozen', False):
        # Cross-platform data directory selection
        if sys.platform == 'darwin':
            # macOS
            d = Path.home() / "Library" / "Application Support" / "OpenMeet"
        elif sys.platform == 'win32':
            # Windows - use APPDATA if available, else LocalAppData
            appdata = os.environ.get('APPDATA')
            if appdata:
                d = Path(appdata) / "OpenMeet"
            else:
                d = Path.home() / "AppData" / "Local" / "OpenMeet"
        else:
            # Linux and other Unix-like systems
            xdg_data_home = os.environ.get('XDG_DATA_HOME')
            if xdg_data_home:
                d = Path(xdg_data_home) / "OpenMeet"
            else:
                d = Path.home() / ".local" / "share" / "OpenMeet"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).parent.parent.parent


def _get_resources_root():
    """Read-only directory for bundled resources."""
    if getattr(sys, 'frozen', False):
        # PyInstaller: sys._MEIPASS, py2app: RESOURCEPATH env var
        return Path(getattr(sys, '_MEIPASS', os.environ.get('RESOURCEPATH', '.')))
    return Path(__file__).parent.parent.parent


DATA_ROOT = _get_data_root()
RESOURCES_ROOT = _get_resources_root()

# .env: check user data dir first, fall back to bundled
ENV_PATH = DATA_ROOT / ".env" if (DATA_ROOT / ".env").exists() else RESOURCES_ROOT / ".env"
SETTINGS_FILE = DATA_ROOT / "data" / "settings.json"

load_dotenv(ENV_PATH)


class Settings:
    """Runtime settings with persistence and .env overrides."""

    DEFAULTS = {
        "audio_device_index": None,
        "whisper_model": "base",
        "language": "en",
        "summary_format": "detailed",
        "diarization_enabled": False,
        "huggingface_token": "",
        "llm_model": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
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
                # Migrate from legacy 'ollama_model' to 'llm_model' if needed
                if "ollama_model" in saved and ("llm_model" not in saved or not saved["llm_model"]):
                    saved["llm_model"] = saved["ollama_model"]
                    print(f"Migrating legacy setting 'ollama_model' to 'llm_model': {saved['llm_model']}")
                self._settings.update(saved)
            except (json.JSONDecodeError, IOError):
                pass

    def _apply_env_overrides(self):
        """.env variables override saved settings."""
        env_map = {
            "OPENMEET_WHISPER_MODEL": "whisper_model",
            "OPENMEET_LANGUAGE": "language",
            "OPENMEET_SUMMARY_FORMAT": "summary_format",
            "OPENMEET_LLM_MODEL": "llm_model",
            "OPENMEET_LOG_LEVEL": "log_level",
            "HUGGINGFACE_TOKEN": "huggingface_token",
            "OPENMEET_DIARIZATION_ENABLED": "diarization_enabled",
        }
        for env_key, setting_key in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                # Handle boolean environment variables
                if setting_key == "diarization_enabled":
                    self._settings[setting_key] = val.lower() in ('true', '1', 'yes', 'on')
                else:
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
