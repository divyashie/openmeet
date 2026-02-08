"""
Integration tests - test full pipeline with mocked external dependencies
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestTranscriptionPipeline:
    """Test audio -> transcription flow"""

    def test_audio_chunk_to_transcript(self, mock_pyaudio, mock_whisper_paths):
        model, exe = mock_whisper_paths

        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from audio_capture import AudioCapture
            from transcriber import Transcriber

            # Set up audio capture with fake data in queue
            ac = AudioCapture()
            ac.is_recording = True

            chunk = np.zeros(1024, dtype=np.int16).tobytes()
            for _ in range(20):
                ac.audio_queue.put(chunk)

            # Get audio chunk
            audio_data = ac.get_audio_chunk(duration_seconds=0.5)
            assert audio_data is not None
            assert len(audio_data) > 0

            # Transcribe chunk
            t = Transcriber()
            whisper_output = "[00:00:00.000 --> 00:00:05.000]   Test transcription output\n"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=whisper_output, stderr=""
                )
                transcript = t.transcribe_chunk(audio_data)

            assert "Test transcription output" in transcript


class TestSummarizationPipeline:
    """Test transcript -> summary flow"""

    def test_transcript_to_summary(self, mock_ollama, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()

        summary = s.generate_summary(sample_transcript)
        assert summary is not None
        assert len(summary) > 0
        assert "Meeting Summary" in summary

    def test_all_summary_formats_work(self, mock_ollama, sample_transcript):
        from summarizer import Summarizer, SUMMARY_FORMATS
        s = Summarizer()

        for fmt in SUMMARY_FORMATS:
            summary = s.generate_summary(sample_transcript, fmt=fmt)
            assert summary is not None
            assert len(summary) > 0


class TestDiarizationPipeline:
    """Test transcript -> diarization alignment flow (no ML models)"""

    def test_parse_and_align(self):
        from diarizer import Diarizer

        transcript = (
            "[00:00:00.000 --> 00:00:05.000]   Hello everyone\n"
            "[00:00:05.000 --> 00:00:10.000]   Let's get started\n"
            "[00:00:10.000 --> 00:00:15.000]   Sounds good\n"
        )

        # Simulated speaker segments
        speaker_segments = [
            (0.0, 6.0, "Speaker 1"),
            (6.0, 11.0, "Speaker 2"),
            (11.0, 15.0, "Speaker 1"),
        ]

        # Parse transcript
        transcript_segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(transcript_segments) == 3

        # Align
        labeled = Diarizer.align_speakers_with_transcript(
            speaker_segments, transcript_segments
        )
        assert labeled[0] == ("Speaker 1", "Hello everyone")
        assert labeled[1] == ("Speaker 2", "Let's get started")
        assert labeled[2] == ("Speaker 1", "Sounds good")


class TestFullPipeline:
    """Test complete audio -> transcript -> summary flow"""

    def test_end_to_end_with_mocks(
        self, mock_pyaudio, mock_whisper_paths, mock_ollama, sample_transcript
    ):
        model, exe = mock_whisper_paths

        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from audio_capture import AudioCapture
            from transcriber import Transcriber
            from summarizer import Summarizer

            # 1. Audio capture produces data
            ac = AudioCapture()
            ac.is_recording = True
            chunk = np.zeros(1024, dtype=np.int16).tobytes()
            for _ in range(20):
                ac.audio_queue.put(chunk)
            audio_data = ac.get_audio_chunk(duration_seconds=0.5)
            assert audio_data is not None

            # 2. Transcription produces text
            t = Transcriber()
            whisper_output = (
                "[00:00:00.000 --> 00:00:05.000]   Hello everyone\n"
                "[00:00:05.000 --> 00:00:10.000]   Meeting discussion\n"
            )
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=whisper_output, stderr=""
                )
                transcript = t.transcribe_chunk(audio_data)
            assert len(transcript) > 0

            # 3. Summary generation
            s = Summarizer()
            summary = s.generate_summary(transcript)
            assert "Meeting Summary" in summary

    def test_empty_recording_produces_placeholder(
        self, mock_pyaudio, mock_whisper_paths, mock_ollama
    ):
        model, exe = mock_whisper_paths

        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            from summarizer import Summarizer

            # Empty transcription
            t = Transcriber()
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="", stderr=""
                )
                transcript = t.transcribe_file(Path("empty.wav"))
            assert transcript == ""

            # Summary of empty transcript
            s = Summarizer()
            summary = s.generate_summary(transcript)
            assert "No transcript available" in summary
