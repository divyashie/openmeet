"""
Speech-to-text using Whisper.cpp
"""
import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
import time
import logging

from utils.config import (
    WHISPER_MODEL_PATH,
    WHISPER_EXECUTABLE,
    SAMPLE_RATE,
    CHANNELS
)

logger = logging.getLogger(__name__)


class Transcriber:
    """Real-time transcription using Whisper.cpp"""

    def __init__(self):
        self.model_path = WHISPER_MODEL_PATH
        self.executable = WHISPER_EXECUTABLE

        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")
        if not self.executable.exists():
            raise FileNotFoundError(f"Whisper executable not found: {self.executable}")

        logger.info("Transcriber initialized (model: %s)", self.model_path.name)

    def transcribe_file(self, wav_path, language="en"):
        """
        Transcribe a complete WAV file.

        Args:
            wav_path: Path to WAV file
            language: Language code (en, fr, es, etc.)

        Returns:
            Transcription text
        """
        try:
            cmd = [
                str(self.executable),
                "-m", str(self.model_path),
                "-f", str(wav_path),
                "-l", language,
                "--output-txt",
                "-t", "4"
            ]

            logger.info("Transcribing: %s", wav_path.name if hasattr(wav_path, 'name') else wav_path)
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error("Whisper error: %s", result.stderr)
                return ""

            output_lines = result.stdout.strip().split('\n')

            transcript_lines = []
            for line in output_lines:
                if not line.strip():
                    continue

                # Extract text from timestamp lines
                if line.startswith('[') and '-->' in line:
                    bracket_end = line.index(']')
                    text = line[bracket_end + 1:].strip()
                    if text:
                        transcript_lines.append(text)
                    continue

                # Skip system messages
                if any(skip in line.lower() for skip in [
                    'whisper_init', 'processing', 'system info',
                    'load time', 'sample time', 'encode time'
                ]):
                    continue

                clean_line = line.strip()
                if clean_line and len(clean_line) > 1:
                    transcript_lines.append(clean_line)

            transcript = ' '.join(transcript_lines)
            transcript = ' '.join(transcript.split())

            logger.info("Transcribed in %.1fs (%d characters)", elapsed, len(transcript))

            return transcript

        except subprocess.TimeoutExpired:
            logger.error("Transcription timed out for %s", wav_path)
            return ""
        except Exception as e:
            logger.error("Transcription error: %s", e, exc_info=True)
            return ""

    def transcribe_chunk(self, audio_data):
        """
        Transcribe a small audio chunk (for real-time).

        Args:
            audio_data: Numpy array of audio samples

        Returns:
            Transcription text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = Path(temp_wav.name)

            try:
                with wave.open(str(temp_path), 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data.tobytes())

                return self.transcribe_file(temp_path)

            finally:
                if temp_path.exists():
                    temp_path.unlink()

    def transcribe_stream(self, audio_capture, callback):
        """
        Transcribe audio stream in real-time.

        Args:
            audio_capture: AudioCapture instance
            callback: Function to call with each transcript chunk
        """
        logger.info("Starting real-time transcription (10-second chunks)")

        chunk_count = 0

        while audio_capture.is_recording:
            audio_chunk = audio_capture.get_audio_chunk(duration_seconds=10)

            if audio_chunk is not None and len(audio_chunk) > 0:
                chunk_count += 1
                logger.debug("Processing chunk #%d...", chunk_count)

                transcript = self.transcribe_chunk(audio_chunk)

                if transcript:
                    callback(transcript)
                else:
                    logger.debug("No speech detected in chunk #%d", chunk_count)

            time.sleep(1)

        logger.info("Real-time transcription stopped (%d chunks processed)", chunk_count)


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger("openmeet", level="DEBUG")

    print("Testing Transcriber\n")

    try:
        transcriber = Transcriber()
    except FileNotFoundError as e:
        print(f"Setup error: {e}")
        print("\nPlease complete setup first:")
        print("  cd whisper.cpp")
        print("  mkdir build && cd build")
        print("  cmake .. && cmake --build . --config Release")
        print("  cd .. && bash ./models/download-ggml-model.sh tiny")
        exit(1)

    sample_file = Path("whisper.cpp/samples/jfk.wav")

    if sample_file.exists():
        print(f"Testing with sample file: {sample_file.name}\n")
        transcript = transcriber.transcribe_file(sample_file)

        print(f"\n{'='*60}")
        print("TRANSCRIPT:")
        print('='*60)
        print(transcript)
        print('='*60)

    else:
        print(f"Sample file not found: {sample_file}")
        print("\nLooking for your recorded audio...\n")

        from utils.config import TRANSCRIPTS_DIR
        recordings = list(TRANSCRIPTS_DIR.glob("meeting_*.wav"))

        if recordings:
            latest = max(recordings, key=lambda p: p.stat().st_mtime)
            print(f"Found recording: {latest.name}\n")

            transcript = transcriber.transcribe_file(latest)

            print(f"\n{'='*60}")
            print("TRANSCRIPT:")
            print('='*60)
            print(transcript)
            print('='*60)

        else:
            print("No recordings found!")
            print("\nPlease record some audio first:")
            print("  python src/audio_capture.py")
