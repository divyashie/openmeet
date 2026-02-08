"""
Configuration management for OpenMeet
"""
import requests
from pathlib import Path
from utils.settings import Settings
from utils.logger import setup_logger, get_logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LOGS_DIR = PROJECT_ROOT / "logs"
WHISPER_DIR = PROJECT_ROOT / "whisper.cpp"

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

# Ollama settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = settings.get("ollama_model")

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

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            errors.append("Ollama is not responding")
    except Exception:
        errors.append("Ollama is not running. Start it with: ollama serve")

    if errors:
        logger.error("Setup validation failed:")
        for error in errors:
            logger.error(f"  {error}")
        return False

    logger.info("All components validated successfully!")
    return True


if __name__ == "__main__":
    validate_setup()
