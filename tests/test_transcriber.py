"""
Tests for transcriber.py
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestTranscriberInit:
    """Test Transcriber initialization"""

    def test_init_with_valid_paths(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()
            assert t.model_path == model
            assert t.executable == exe

    def test_init_raises_on_missing_model(self, tmp_path):
        missing = tmp_path / "nonexistent.bin"
        exe = tmp_path / "whisper-cli"
        exe.write_text("fake")
        with patch("transcriber.WHISPER_MODEL_PATH", missing), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            with pytest.raises(FileNotFoundError, match="model"):
                Transcriber()

    def test_init_raises_on_missing_executable(self, tmp_path):
        model = tmp_path / "ggml-tiny.bin"
        model.write_text("fake")
        missing = tmp_path / "nonexistent"
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", missing):
            from transcriber import Transcriber
            with pytest.raises(FileNotFoundError, match="executable"):
                Transcriber()


class TestTranscriberParsing:
    """Test the output parsing logic in transcribe_file"""

    def test_parses_timestamp_lines(self, mock_whisper_paths, sample_whisper_output):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=sample_whisper_output, stderr=""
                )
                result = t.transcribe_file(Path("test.wav"))

            assert "Hello everyone" in result
            assert "project status" in result
            assert "meeting adjourned" in result

    def test_filters_system_messages(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        whisper_output = (
            "whisper_init_from_file: loading model\n"
            "system info: test\n"
            "[00:00:00.000 --> 00:00:03.000]   Actual text\n"
            "processing: done\n"
            "load time = 100ms\n"
        )
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=whisper_output, stderr=""
                )
                result = t.transcribe_file(Path("test.wav"))

            assert result == "Actual text"
            assert "whisper_init" not in result
            assert "system info" not in result
            assert "processing" not in result

    def test_handles_empty_output(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="", stderr=""
                )
                result = t.transcribe_file(Path("test.wav"))

            assert result == ""

    def test_handles_whisper_error(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stdout="", stderr="error occurred"
                )
                result = t.transcribe_file(Path("test.wav"))

            assert result == ""

    def test_handles_subprocess_timeout(self, mock_whisper_paths):
        import subprocess
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="whisper", timeout=300)
                result = t.transcribe_file(Path("test.wav"))

            assert result == ""


class TestTranscribeChunk:
    """Test transcribe_chunk method"""

    def test_returns_empty_for_none_input(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()
            assert t.transcribe_chunk(None) == ""

    def test_returns_empty_for_empty_array(self, mock_whisper_paths):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()
            assert t.transcribe_chunk(np.array([], dtype=np.int16)) == ""

    def test_creates_temp_wav_and_transcribes(self, mock_whisper_paths, audio_chunk):
        model, exe = mock_whisper_paths
        with patch("transcriber.WHISPER_MODEL_PATH", model), \
             patch("transcriber.WHISPER_EXECUTABLE", exe):
            from transcriber import Transcriber
            t = Transcriber()

            with patch.object(t, "transcribe_file", return_value="hello world") as mock_tf:
                result = t.transcribe_chunk(audio_chunk)
                assert result == "hello world"
                # Verify transcribe_file was called with a temp path
                call_path = mock_tf.call_args[0][0]
                assert str(call_path).endswith(".wav")
