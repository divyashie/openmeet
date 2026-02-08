"""
AI-powered meeting summarization using Ollama
"""
import requests
from datetime import datetime
import logging

from utils.config import OLLAMA_API_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SUMMARY_FORMATS = ("detailed", "bullets", "executive", "email")


class Summarizer:
    """Generate meeting summaries using local LLM (Ollama)"""

    def __init__(self, model=OLLAMA_MODEL, summary_format="detailed"):
        self.model = model
        self.api_url = OLLAMA_API_URL
        self.summary_format = summary_format

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info("Summarizer initialized (model: %s)", self.model)
            else:
                logger.warning("Ollama responded with status %d", response.status_code)
        except Exception:
            logger.error("Ollama is not running! Start it with: ollama serve")
            raise ConnectionError("Ollama is not running")

    def generate_summary(self, transcript, meeting_duration_minutes=None, fmt=None):
        """
        Generate a meeting summary from transcript.

        Args:
            transcript: Full meeting transcript text
            meeting_duration_minutes: Optional meeting duration
            fmt: Summary format (detailed/bullets/executive/email). Falls back to self.summary_format.

        Returns:
            Formatted summary as markdown string
        """
        if not transcript or len(transcript.strip()) < 10:
            return "# Meeting Summary\n\nNo transcript available to summarize."

        fmt = fmt or self.summary_format
        prompt = self._build_summary_prompt(transcript, meeting_duration_minutes, fmt)

        logger.info("Generating %s summary (%d chars, model: %s)", fmt, len(transcript), self.model)

        try:
            response = self._call_ollama(prompt)

            if response:
                logger.info("Summary generated successfully")
                return response
            else:
                return "# Meeting Summary\n\nFailed to generate summary."

        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            return f"# Meeting Summary\n\nError: {e}"

    def _build_summary_prompt(self, transcript, duration=None, fmt="detailed"):
        """Build format-specific prompt for the LLM."""
        builder = {
            "detailed": self._prompt_detailed,
            "bullets": self._prompt_bullets,
            "executive": self._prompt_executive,
            "email": self._prompt_email,
        }

        build_fn = builder.get(fmt, self._prompt_detailed)
        return build_fn(transcript, duration)

    def _prompt_detailed(self, transcript, duration=None):
        """Full structured markdown format."""
        duration_text = f"\nMeeting Duration: ~{duration} minutes" if duration else ""

        return f"""You are an expert meeting assistant. Analyze this meeting transcript and provide a clear, structured summary.

{duration_text}

TRANSCRIPT:
{transcript}

Please provide a professional meeting summary with these sections:

# Meeting Summary
Date: {datetime.now().strftime("%B %d, %Y")}

## Overview
[2-3 sentence high-level summary of what was discussed]

## Key Discussion Points
- [Main topic 1]
- [Main topic 2]
- [Main topic 3]
[etc.]

## Decisions Made
- [Decision 1]
- [Decision 2]
[If no decisions, write "No formal decisions recorded"]

## Action Items
- [ ] @Person: [Specific task] (Due: [date if mentioned])
- [ ] @Person: [Specific task] (Due: [date if mentioned])
[If no action items, write "No action items identified"]

## Next Steps
- [What happens next]
- [Follow-up items]

## Open Questions
- [Unresolved question 1]?
- [Unresolved question 2]?
[If none, write "No open questions"]

IMPORTANT RULES:
1. Be concise but complete
2. Extract action items with assignees if names are mentioned
3. Use bullet points for clarity
4. ONLY use information that is explicitly stated in the transcript. NEVER invent names, topics, decisions, or details that are not in the transcript. If a section has no relevant info, write "Not mentioned in transcript"
5. Use markdown formatting
6. Keep the tone professional
7. If the transcript is unclear or garbled, summarize only what you can confidently understand

Generate the summary now:"""

    def _prompt_bullets(self, transcript, duration=None):
        """Concise bullet points + action items."""
        duration_text = f"\nMeeting Duration: ~{duration} minutes" if duration else ""

        return f"""You are an expert meeting assistant. Summarize this meeting as a concise bullet-point list.
{duration_text}

TRANSCRIPT:
{transcript}

Format your response EXACTLY like this:

# Meeting Notes - {datetime.now().strftime("%B %d, %Y")}

## Key Points
- [Point 1]
- [Point 2]
- [Point 3]

## Action Items
- [ ] [Person]: [Task] [Due date if mentioned]

## Takeaways
- [Key takeaway 1]
- [Key takeaway 2]

RULES: Be concise. Max 10 bullet points for Key Points. Only include action items explicitly mentioned. Use markdown."""

    def _prompt_executive(self, transcript, duration=None):
        """Short 2-3 paragraph executive summary."""
        duration_text = f"\nMeeting Duration: ~{duration} minutes" if duration else ""

        return f"""You are an executive assistant. Write a brief executive summary of this meeting in 2-3 paragraphs.
{duration_text}

TRANSCRIPT:
{transcript}

Format:

# Executive Brief - {datetime.now().strftime("%B %d, %Y")}

[2-3 paragraphs summarizing: what was discussed, what was decided, what happens next]

**Key Decision:** [Most important decision, or "None"]
**Critical Action:** [Most urgent action item, or "None"]

RULES: Keep it under 200 words. No bullet points. Professional tone. Focus on outcomes not process."""

    def _prompt_email(self, transcript, duration=None):
        """Email-ready follow-up recap."""
        duration_text = f" ({duration} min)" if duration else ""

        return f"""You are a professional meeting coordinator. Write a follow-up email summarizing this meeting.

TRANSCRIPT:
{transcript}

Format the email EXACTLY like this:

Subject: Meeting Recap - {datetime.now().strftime("%B %d, %Y")}{duration_text}

Hi team,

Thank you for joining today's meeting. Here's a quick recap:

**What we discussed:**
- [Topic 1]
- [Topic 2]

**What we decided:**
- [Decision 1]

**Action items:**
- [Person]: [Task] (by [date])

**Next meeting:** [Date/time if mentioned, otherwise "TBD"]

Let me know if I missed anything or if you have questions.

Best regards,
[Meeting Organizer]

RULES: Keep it professional and concise. Only include information from the transcript. Use a warm but professional tone."""

    def _call_ollama(self, prompt, max_retries=3):
        """Call Ollama API with retry logic."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.85,
                "num_predict": 1000
            }
        }

        for attempt in range(max_retries):
            try:
                logger.info("Calling Ollama (attempt %d/%d)...", attempt + 1, max_retries)

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    logger.warning("Ollama returned status %d", response.status_code)

            except requests.exceptions.Timeout:
                logger.warning("Request timed out")

            except Exception as e:
                logger.error("Ollama error: %s", e)

            if attempt < max_retries - 1:
                logger.info("Retrying...")

        return None


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger("openmeet", level="DEBUG")

    print("Testing Summarizer\n")

    try:
        summarizer = Summarizer()
    except ConnectionError as e:
        print(f"\n{e}")
        print("\nPlease start Ollama:")
        print("  ollama serve")
        exit(1)

    sample_transcript = """
    [10:00:23] Thanks everyone for joining today's standup. Let's go around and share updates.
    [10:00:45] Sarah here. I finished the authentication module yesterday. It's ready for code review.
    [10:01:23] I've been working on the database optimization. Should be done by end of day.
    [10:02:05] Let's deploy to staging on Friday afternoon.
    [10:03:00] Yes I'll take care of it. I'll send a deploy checklist to the team by Wednesday.
    [10:03:25] Great. Thanks everyone. Let's sync again tomorrow same time.
    """

    print("Sample Transcript:")
    print("="*60)
    print(sample_transcript[:200] + "...")
    print("="*60 + "\n")

    for fmt in SUMMARY_FORMATS:
        print(f"\n--- {fmt.upper()} FORMAT ---")
        summary = summarizer.generate_summary(sample_transcript, meeting_duration_minutes=3, fmt=fmt)
        print(summary)
        print("="*60)
