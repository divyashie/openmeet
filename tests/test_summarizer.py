"""
Tests for summarizer.py
"""
import pytest
from unittest.mock import patch, MagicMock
import requests


class TestSummarizerInit:
    """Test Summarizer initialization"""

    def test_init_success(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        assert s.model is not None
        assert s.api_url is not None

    def test_init_raises_when_ollama_down(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            from summarizer import Summarizer
            with pytest.raises(ConnectionError):
                Summarizer()


class TestBuildSummaryPrompt:
    """Test prompt building"""

    def test_detailed_prompt_contains_transcript(self, mock_ollama, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt(sample_transcript, fmt="detailed")
        assert "Hello everyone" in prompt
        assert "project status" in prompt

    def test_detailed_prompt_contains_date(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test transcript", fmt="detailed")
        assert "202" in prompt  # Year

    def test_prompt_includes_duration(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", duration=30, fmt="detailed")
        assert "30 minutes" in prompt

    def test_prompt_excludes_duration_when_none(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", duration=None, fmt="detailed")
        assert "minutes" not in prompt

    def test_bullets_format(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="bullets")
        assert "bullet-point" in prompt.lower() or "Key Points" in prompt

    def test_executive_format(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="executive")
        assert "executive" in prompt.lower() or "Executive Brief" in prompt

    def test_email_format(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="email")
        assert "email" in prompt.lower() or "recap" in prompt.lower()

    def test_unknown_format_falls_back_to_detailed(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="nonexistent")
        # Should use detailed format (contains structured sections)
        assert "Overview" in prompt


class TestCallOllama:
    """Test Ollama API calling"""

    def test_successful_api_call(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_ollama("test prompt")
        assert result is not None
        assert "Meeting Summary" in result

    def test_retries_on_failure(self, mock_ollama):
        mock_get, mock_post = mock_ollama
        mock_post.side_effect = [
            MagicMock(status_code=500),
            MagicMock(status_code=500),
            MagicMock(status_code=200, json=lambda: {"response": "success"})
        ]
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_ollama("test")
        assert result == "success"

    def test_returns_none_after_max_retries(self, mock_ollama):
        mock_get, mock_post = mock_ollama
        mock_post.return_value = MagicMock(status_code=500)
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_ollama("test")
        assert result is None

    def test_handles_timeout(self, mock_ollama):
        mock_get, mock_post = mock_ollama
        mock_post.side_effect = requests.exceptions.Timeout()
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_ollama("test")
        assert result is None


class TestGenerateSummary:
    """Test full summary generation"""

    def test_short_transcript_returns_placeholder(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary("hi")
        assert "No transcript available" in result

    def test_empty_transcript_returns_placeholder(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary("")
        assert "No transcript available" in result

    def test_none_transcript_returns_placeholder(self, mock_ollama):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary(None)
        assert "No transcript available" in result

    def test_valid_transcript_calls_ollama(self, mock_ollama, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary(sample_transcript)
        assert "Meeting Summary" in result

    def test_format_parameter_passed_through(self, mock_ollama, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        # Should not raise
        result = s.generate_summary(sample_transcript, fmt="bullets")
        assert result is not None
