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
def mock_llm(tmp_path):
    """Mock llama-cpp-python for summarizer tests"""
    # Create a fake model file so path.exists() passes
    fake_model = tmp_path / "fake_model.gguf"
    fake_model.write_text("fake")

    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = {
        'choices': [{'text': '# Meeting Summary\n\n## Overview\nTest summary content.'}]
    }

    mock_llama_class = MagicMock(return_value=mock_llama_instance)

    mock_module = MagicMock()
    mock_module.Llama = mock_llama_class

    with patch.dict('sys.modules', {'llama_cpp': mock_module}), \
         patch('summarizer.LLM_MODEL_PATH', fake_model):
        yield mock_llama_instance


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
