"""
Tests for diarizer.py - static methods only (no ML models needed)
"""
import pytest
from diarizer import Diarizer


class TestParseWhisperTimestamps:
    """Test timestamp parsing from different formats"""

    def test_parses_whisper_cpp_format(self):
        transcript = (
            "[00:00:00.000 --> 00:00:03.000]   Hello everyone\n"
            "[00:00:03.000 --> 00:00:06.000]   Welcome to the meeting\n"
        )
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 2
        assert segments[0] == (0.0, 3.0, "Hello everyone")
        assert segments[1] == (3.0, 6.0, "Welcome to the meeting")

    def test_parses_runtime_format(self):
        transcript = (
            "[10:00:00] Hello everyone\n"
            "[10:01:00] Let's discuss\n"
        )
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 2
        # 10:00:00 = 36000 seconds
        assert segments[0][0] == 36000.0
        assert segments[0][2] == "Hello everyone"
        # Runtime format estimates 10s duration
        assert segments[0][1] == 36010.0

    def test_handles_empty_input(self):
        segments = Diarizer.parse_whisper_timestamps("")
        assert segments == []

    def test_skips_empty_lines(self):
        transcript = "\n\n[00:00:00.000 --> 00:00:03.000]   Hello\n\n"
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 1

    def test_skips_lines_without_timestamps(self):
        transcript = (
            "Some random text\n"
            "[00:00:00.000 --> 00:00:03.000]   Hello\n"
            "More random text\n"
        )
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 1
        assert segments[0][2] == "Hello"

    def test_skips_timestamp_lines_with_no_text(self):
        transcript = "[00:00:00.000 --> 00:00:03.000]   \n"
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 0

    def test_handles_mixed_formats(self):
        transcript = (
            "[00:00:00.000 --> 00:00:03.000]   Whisper format\n"
            "[10:05:30] Runtime format\n"
        )
        segments = Diarizer.parse_whisper_timestamps(transcript)
        assert len(segments) == 2
        assert segments[0][2] == "Whisper format"
        assert segments[1][2] == "Runtime format"


class TestAlignSpeakers:
    """Test temporal overlap alignment"""

    def test_perfect_overlap(self):
        speakers = [(0.0, 5.0, "Speaker 1"), (5.0, 10.0, "Speaker 2")]
        transcript = [(0.0, 5.0, "Hello"), (5.0, 10.0, "World")]

        result = Diarizer.align_speakers_with_transcript(speakers, transcript)
        assert result == [("Speaker 1", "Hello"), ("Speaker 2", "World")]

    def test_partial_overlap_picks_best(self):
        speakers = [(0.0, 4.0, "Speaker 1"), (4.0, 10.0, "Speaker 2")]
        # Transcript segment 3-7 overlaps Speaker 1 (3-4=1s) and Speaker 2 (4-7=3s)
        transcript = [(3.0, 7.0, "Hello")]

        result = Diarizer.align_speakers_with_transcript(speakers, transcript)
        assert result[0][0] == "Speaker 2"  # More overlap

    def test_no_overlap_returns_unknown(self):
        speakers = [(0.0, 5.0, "Speaker 1")]
        transcript = [(10.0, 15.0, "Hello")]

        result = Diarizer.align_speakers_with_transcript(speakers, transcript)
        assert result[0][0] == "Unknown"

    def test_empty_speakers(self):
        transcript = [(0.0, 5.0, "Hello")]
        result = Diarizer.align_speakers_with_transcript([], transcript)
        assert result[0][0] == "Unknown"

    def test_empty_transcript(self):
        speakers = [(0.0, 5.0, "Speaker 1")]
        result = Diarizer.align_speakers_with_transcript(speakers, [])
        assert result == []

    def test_multiple_speakers_multiple_segments(self):
        speakers = [
            (0.0, 3.0, "Speaker 1"),
            (3.0, 6.0, "Speaker 2"),
            (6.0, 9.0, "Speaker 1"),
        ]
        transcript = [
            (0.0, 3.0, "First part"),
            (3.0, 6.0, "Second part"),
            (6.0, 9.0, "Third part"),
        ]

        result = Diarizer.align_speakers_with_transcript(speakers, transcript)
        assert result[0] == ("Speaker 1", "First part")
        assert result[1] == ("Speaker 2", "Second part")
        assert result[2] == ("Speaker 1", "Third part")


class TestDiarizerInit:
    """Test Diarizer initialization (without loading models)"""

    def test_init_without_token(self):
        d = Diarizer()
        assert d.hf_token is None
        assert d.pipeline is None
        assert d._initialized is False

    def test_init_with_token(self):
        d = Diarizer(hf_token="test_token")
        assert d.hf_token == "test_token"
        assert d._initialized is False
