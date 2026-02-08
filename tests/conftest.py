"""
Shared pytest fixtures for OpenMeet tests
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for audio_capture tests"""
    with patch("pyaudio.PyAudio") as mock:
        instance = MagicMock()
        instance.get_device_count.return_value = 2
        instance.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'MacBook Pro Microphone',
            'maxInputChannels': 1,
            'defaultSampleRate': 16000.0
        }
        instance.get_device_info_by_index.side_effect = lambda i: {
            0: {
                'index': 0, 'name': 'MacBook Pro Microphone',
                'maxInputChannels': 1, 'defaultSampleRate': 16000.0
            },
            1: {
                'index': 1, 'name': 'External Mic',
                'maxInputChannels': 2, 'defaultSampleRate': 44100.0
            },
        }[i]
        instance.get_sample_size.return_value = 2
        instance.open.return_value = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_whisper_paths(tmp_path):
    """Create fake whisper model and executable paths"""
    model = tmp_path / "ggml-tiny.bin"
    model.write_text("fake model")
    exe = tmp_path / "whisper-cli"
    exe.write_text("fake exe")
    exe.chmod(0o755)
    return model, exe


@pytest.fixture
def mock_ollama():
    """Mock Ollama API responses"""
    with patch("requests.get") as mock_get, \
         patch("requests.post") as mock_post:
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "# Meeting Summary\n\n## Overview\nTest summary content."}
        )
        yield mock_get, mock_post


@pytest.fixture
def sample_transcript():
    """Sample transcript text for testing"""
    return (
        "[10:00:00] Hello everyone, welcome to the meeting.\n"
        "[10:01:00] Let's discuss the project status.\n"
        "[10:02:00] Great, meeting adjourned."
    )


@pytest.fixture
def sample_whisper_output():
    """Sample whisper-cli stdout output"""
    return (
        "whisper_init_from_file: loading model\n"
        "[00:00:00.000 --> 00:00:03.000]   Hello everyone, welcome to the meeting.\n"
        "[00:00:03.000 --> 00:00:06.000]   Let's discuss the project status.\n"
        "[00:00:06.000 --> 00:00:09.000]   Great, meeting adjourned.\n"
        "whisper_print_timings: load time = 100ms\n"
    )


@pytest.fixture
def audio_chunk():
    """Generate a fake audio chunk (1 second of silence at 16kHz)"""
    return np.zeros(16000, dtype=np.int16)
