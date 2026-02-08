"""
Tests for audio_capture.py
"""
import pytest
import numpy as np
import queue
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestAudioCaptureInit:
    """Test AudioCapture initialization"""

    def test_initializes_pyaudio(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        assert ac.audio is not None

    def test_initial_state(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        assert ac.is_recording is False
        assert ac.frames == []
        assert ac.current_session_id is None
        assert ac.stream is None


class TestListDevices:
    """Test device listing"""

    def test_returns_device_list(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        devices = ac.list_audio_devices()
        assert len(devices) == 2
        assert devices[0]['name'] == 'MacBook Pro Microphone'
        assert devices[1]['name'] == 'External Mic'

    def test_device_has_required_keys(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        devices = ac.list_audio_devices()
        for d in devices:
            assert 'index' in d
            assert 'name' in d
            assert 'channels' in d
            assert 'sample_rate' in d


class TestFindInputDevice:
    """Test input device detection"""

    def test_returns_default_device_index(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        idx = ac.find_input_device()
        assert idx == 0

    def test_returns_none_on_error(self, mock_pyaudio):
        mock_pyaudio.get_default_input_device_info.side_effect = Exception("No device")
        from audio_capture import AudioCapture
        ac = AudioCapture()
        idx = ac.find_input_device()
        assert idx is None


class TestGetAudioChunk:
    """Test audio chunk retrieval from queue"""

    def test_returns_numpy_array_when_data_available(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        ac.is_recording = True

        # Put some fake audio data in the queue
        chunk = np.zeros(1024, dtype=np.int16).tobytes()
        for _ in range(10):
            ac.audio_queue.put(chunk)

        result = ac.get_audio_chunk(duration_seconds=0.5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16

    def test_returns_none_when_no_data_and_not_recording(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        ac.is_recording = False
        result = ac.get_audio_chunk(duration_seconds=0.1)
        assert result is None


class TestSessionId:
    """Test session ID generation"""

    def test_session_id_format(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()

        # Mock start_recording enough to set session_id
        with patch.object(ac, 'find_input_device', return_value=0):
            ac.start_recording()

        assert ac.current_session_id is not None
        # Format: YYYYMMDD_HHMMSS
        assert len(ac.current_session_id) == 15
        assert ac.current_session_id[8] == '_'

        # Cleanup
        ac.is_recording = False

    def test_new_session_id_per_recording(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()

        with patch.object(ac, 'find_input_device', return_value=0):
            ac.start_recording()
            id1 = ac.current_session_id
            ac.is_recording = False

            import time
            time.sleep(1.1)

            ac.start_recording()
            id2 = ac.current_session_id
            ac.is_recording = False

        assert id1 != id2


class TestCleanup:
    """Test resource cleanup"""

    def test_cleanup_terminates_pyaudio(self, mock_pyaudio):
        from audio_capture import AudioCapture
        ac = AudioCapture()
        ac.cleanup()
        mock_pyaudio.terminate.assert_called_once()
