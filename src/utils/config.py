# Create the config file for the project
"""
Configuration management for OpenMeet
"""
import os
import requests
from pathlib import Path

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

# Whisper settings
WHISPER_MODEL_PATH = WHISPER_DIR / "models" / "ggml-tiny.bin"
WHISPER_EXECUTABLE = WHISPER_DIR / "build" / "bin" / "whisper-cli"

# Audio settings
SAMPLE_RATE = 16000  # Whisper prefers 16kHz
CHANNELS = 1  # Mono
CHUNK_SIZE = 1024
CHUNK_DURATION = 10  # seconds

# Ollama settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# UI settings
TRANSCRIPT_WINDOW_WIDTH = 500
TRANSCRIPT_WINDOW_HEIGHT = 600
SUMMARY_WINDOW_WIDTH = 700
SUMMARY_WINDOW_HEIGHT = 800

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "openmeet.log"

def validate_setup():
    """Check if all required components are installed"""
    errors = []
    
    # Check Whisper model exists
    if not WHISPER_MODEL_PATH.exists():
        errors.append(f"Whisper model not found at {WHISPER_MODEL_PATH}")
        errors.append("  Run: cd whisper.cpp && bash ./models/download-ggml-model.sh tiny")
    
    # Check Whisper executable exists
    if not WHISPER_EXECUTABLE.exists():
        errors.append(f"Whisper executable not found at {WHISPER_EXECUTABLE}")
        errors.append("  Run: cd whisper.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release")
    
    # Check Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            errors.append("Ollama is not responding")
    except:
        errors.append("Ollama is not running. Start it with: ollama serve")
        errors.append("  (Or it will start automatically when you run: ollama pull llama3.2:3b)")
    
    if errors:
        print("❌ Setup validation failed:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print("✅ All components validated successfully!")
    return True

if __name__ == "__main__":
    validate_setup()
