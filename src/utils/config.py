"""
Configuration management for OpenMeet
"""
import sys
import os
from pathlib import Path
from utils.settings import Settings
from utils.logger import setup_logger, get_logger


def is_frozen():
    """Check if running inside a bundled .app (PyInstaller or py2app)."""
    return getattr(sys, 'frozen', False)


def get_resources_dir():
    """Return the directory containing read-only bundled resources."""
    if is_frozen():
        # PyInstaller stores resources in sys._MEIPASS
        # py2app uses RESOURCEPATH env var
        return Path(getattr(sys, '_MEIPASS', os.environ.get(
            'RESOURCEPATH',
            str(Path(sys.executable).parent.parent / 'Resources')
        )))
    return Path(__file__).parent.parent.parent


def get_app_data_dir():
    """Return a writable directory for user data."""
    if is_frozen():
        d = Path.home() / "Library" / "Application Support" / "OpenMeet"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).parent.parent.parent


# Directories — read-only resources vs writable user data
RESOURCES_DIR = get_resources_dir()
APP_DATA_DIR = get_app_data_dir()

SRC_DIR = RESOURCES_DIR / "src" if not is_frozen() else RESOURCES_DIR
DATA_DIR = APP_DATA_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LOGS_DIR = APP_DATA_DIR / "logs"
WHISPER_DIR = RESOURCES_DIR / "whisper.cpp"

# Create directories if they don't exist
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Settings singleton
settings = Settings()

# Initialize root logger
LOG_FILE = LOGS_DIR / "openmeet.log"
setup_logger("root", log_file=LOG_FILE, level=settings.get("log_level"))
logger = get_logger(__name__)

# Whisper settings (dynamic based on settings)
WHISPER_MODEL_PATH = WHISPER_DIR / "models" / f"ggml-{settings.get('whisper_model')}.bin"
WHISPER_EXECUTABLE = WHISPER_DIR / "build" / "bin" / "whisper-cli"

# Audio settings
SAMPLE_RATE = 16000  # Whisper prefers 16kHz
CHANNELS = 1  # Mono
CHUNK_SIZE = 1024
CHUNK_DURATION = 10  # seconds

# LLM settings (bundled model) - only construct path if model name is configured
llm_model_name = settings.get("llm_model")
LLM_MODEL_PATH = RESOURCES_DIR / "models" / llm_model_name if llm_model_name else None

# UI settings
TRANSCRIPT_WINDOW_WIDTH = 500
TRANSCRIPT_WINDOW_HEIGHT = 600
SUMMARY_WINDOW_WIDTH = 700
SUMMARY_WINDOW_HEIGHT = 800


def validate_setup():
    """Check if all required components are installed"""
    errors = []

    if not WHISPER_MODEL_PATH.exists():
        model_name = settings.get("whisper_model")
        errors.append(f"Whisper model not found at {WHISPER_MODEL_PATH}")
        errors.append(f"  Run: cd whisper.cpp && bash ./models/download-ggml-model.sh {model_name}")

    if not WHISPER_EXECUTABLE.exists():
        errors.append(f"Whisper executable not found at {WHISPER_EXECUTABLE}")
        errors.append("  Run: cd whisper.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release")

    if LLM_MODEL_PATH is None:
        errors.append("LLM model not configured (llm_model is missing from settings)")
        errors.append("  Configure llm_model in settings or set OPENMEET_LLM_MODEL environment variable")
    elif not LLM_MODEL_PATH.exists():
        errors.append(f"LLM model not found at {LLM_MODEL_PATH}")
        errors.append(f"  Download a GGUF model into the models/ directory")

    if errors:
        logger.error("Setup validation failed:")
        for error in errors:
            logger.error(f"  {error}")
        return False

    logger.info("All components validated successfully!")
    return True


if __name__ == "__main__":
    validate_setup()
