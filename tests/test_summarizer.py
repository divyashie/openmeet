"""
Tests for summarizer.py
"""
import pytest
from unittest.mock import patch, MagicMock


class TestSummarizerInit:
    """Test Summarizer initialization"""

    def test_init_success(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        assert s.llm is not None
        assert s.model_path is not None

    def test_init_raises_when_model_missing(self, tmp_path):
        missing_model = tmp_path / "nonexistent.gguf"
        with patch("summarizer.LLM_MODEL_PATH", missing_model):
            from summarizer import Summarizer
            with pytest.raises(RuntimeError):
                Summarizer()


class TestBuildSummaryPrompt:
    """Test prompt building"""

    def test_detailed_prompt_contains_transcript(self, mock_llm, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt(sample_transcript, fmt="detailed")
        assert "Hello everyone" in prompt
        assert "project status" in prompt

    def test_detailed_prompt_contains_date(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test transcript", fmt="detailed")
        assert "202" in prompt  # Year

    def test_prompt_includes_duration(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", duration=30, fmt="detailed")
        assert "30 minutes" in prompt

    def test_prompt_excludes_duration_when_none(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", duration=None, fmt="detailed")
        assert "minutes" not in prompt

    def test_bullets_format(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="bullets")
        assert "bullet-point" in prompt.lower() or "Key Points" in prompt

    def test_executive_format(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="executive")
        assert "executive" in prompt.lower() or "Executive Brief" in prompt

    def test_email_format(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="email")
        assert "email" in prompt.lower() or "recap" in prompt.lower()

    def test_unknown_format_falls_back_to_detailed(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        prompt = s._build_summary_prompt("test", fmt="nonexistent")
        # Should use detailed format (contains structured sections)
        assert "Overview" in prompt


class TestCallLLM:
    """Test local LLM calling"""

    def test_successful_llm_call(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_llm("test prompt")
        assert result is not None
        assert "Meeting Summary" in result

    def test_returns_none_on_empty_response(self, mock_llm):
        mock_llm.return_value = {'choices': [{'text': '   '}]}
        from summarizer import Summarizer
        s = Summarizer()
        result = s._call_llm("test")
        assert result is None

    def test_handles_llm_exception(self, mock_llm):
        mock_llm.side_effect = Exception("Out of memory")
        from summarizer import Summarizer
        s = Summarizer()
        # _call_llm should propagate the exception (caught by generate_summary)
        with pytest.raises(Exception):
            s._call_llm("test")


class TestGenerateSummary:
    """Test full summary generation"""

    def test_short_transcript_returns_placeholder(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary("hi")
        assert "No transcript available" in result

    def test_empty_transcript_returns_placeholder(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary("")
        assert "No transcript available" in result

    def test_none_transcript_returns_placeholder(self, mock_llm):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary(None)
        assert "No transcript available" in result

    def test_valid_transcript_calls_llm(self, mock_llm, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        result = s.generate_summary(sample_transcript)
        assert "Meeting Summary" in result

    def test_format_parameter_passed_through(self, mock_llm, sample_transcript):
        from summarizer import Summarizer
        s = Summarizer()
        # Should not raise
        result = s.generate_summary(sample_transcript, fmt="bullets")
        assert result is not None
