"""
Audio capture from system audio or microphone
"""
import pyaudio
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import queue
import time
import logging

from utils.config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    TRANSCRIPTS_DIR
)

logger = logging.getLogger(__name__)

MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECONDS = 1.0


class AudioCapture:
    """Captures audio and saves to WAV files"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.current_session_id = None
        self.audio_queue = queue.Queue()
        self._device_index = None

        logger.info("AudioCapture initialized")

    def list_audio_devices(self):
        """List all available audio devices"""
        logger.debug("Listing audio devices")
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': info['defaultSampleRate']
            })
            logger.debug(
                "  [%d] %s (Channels: %d, Rate: %s)",
                i, info['name'], info['maxInputChannels'], info['defaultSampleRate']
            )
        return devices

    def find_input_device(self):
        """Find the best input device. Prefers: microphone input."""
        try:
            default_input = self.audio.get_default_input_device_info()
            device_index = default_input['index']
            device_name = default_input['name']

            logger.info("Using device: [%d] %s", device_index, device_name)
            return device_index

        except Exception as e:
            logger.error("No default input device found: %s", e)
            return None

    def start_recording(self):
        """Start capturing audio"""
        if self.is_recording:
            logger.warning("Already recording!")
            return False

        try:
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            device_index = self.find_input_device()
            if device_index is None:
                logger.error("No audio input device available")
                return False

            self._device_index = device_index
            logger.info("Starting recording session: %s", self.current_session_id)

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )

            self.is_recording = True
            self.frames = []

            self.stream.start_stream()

            logger.info("Recording started! Speak into your microphone...")
            return True

        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            logger.info("Tip: Check System Preferences > Security & Privacy > Microphone")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream (called by PyAudio)"""
        if self.is_recording:
            self.frames.append(in_data)
            self.audio_queue.put(in_data)

        return (in_data, pyaudio.paContinue)

    def _reconnect_stream(self):
        """Attempt to reconnect the audio stream after a failure."""
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            logger.warning(
                "Attempting stream reconnection (%d/%d)",
                attempt + 1, MAX_RECONNECT_ATTEMPTS
            )
            try:
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception:
                        pass

                device_index = self._device_index or self.find_input_device()
                if device_index is None:
                    continue

                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=self._audio_callback
                )
                self.stream.start_stream()
                logger.info("Stream reconnected successfully")
                return True
            except Exception as e:
                logger.error("Reconnection attempt %d failed: %s", attempt + 1, e)
                time.sleep(RECONNECT_DELAY_SECONDS)

        logger.error("All reconnection attempts failed")
        return False

    def stop_recording(self):
        """Stop capturing audio and save to file"""
        if not self.is_recording:
            logger.warning("Not recording!")
            return None

        try:
            logger.info("Stopping recording...")
            self.is_recording = False

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            wav_path = self._save_audio()

            logger.info("Recording saved: %s", wav_path.name)
            logger.info("File size: %.1f KB", wav_path.stat().st_size / 1024)

            return wav_path

        except Exception as e:
            logger.error("Failed to stop recording: %s", e)
            return None

    def _save_audio(self):
        """Save recorded audio to WAV file"""
        filename = f"meeting_{self.current_session_id}.wav"
        wav_path = TRANSCRIPTS_DIR / filename

        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        return wav_path

    def get_audio_chunk(self, duration_seconds=5):
        """
        Get accumulated audio chunk for transcription.
        Returns numpy array of audio data.
        """
        chunks_needed = int(SAMPLE_RATE * duration_seconds / CHUNK_SIZE)
        audio_data = []
        start_time = time.time()

        while len(audio_data) < chunks_needed:
            if time.time() - start_time > duration_seconds * 2:
                break
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                audio_data.append(np.frombuffer(chunk, dtype=np.int16))
            except queue.Empty:
                if not self.is_recording:
                    break

        if audio_data:
            return np.concatenate(audio_data)
        return None

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()
        logger.info("AudioCapture cleaned up")


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger("openmeet", level="DEBUG")

    print("Testing Audio Capture\n")

    capture = AudioCapture()
    capture.list_audio_devices()

    print("\nStarting 5-second test recording...")
    print("Say something into your microphone!")

    if capture.start_recording():
        time.sleep(5)
        wav_path = capture.stop_recording()

        if wav_path:
            print(f"\nTest successful! Saved to: {wav_path}")
            print(f"Play it back with: afplay {wav_path}")
        else:
            print("\nTest failed - could not save recording")
    else:
        print("\nTest failed - could not start recording")

    capture.cleanup()
