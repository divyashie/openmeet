"""
AI-powered meeting summarization using local LLM (llama-cpp-python)
"""
from datetime import datetime
import logging

from utils.config import LLM_MODEL_PATH

logger = logging.getLogger(__name__)

SUMMARY_FORMATS = ("detailed", "bullets", "executive", "email")


class Summarizer:
    """Generate meeting summaries using a bundled local LLM"""

    def __init__(self, summary_format="detailed"):
        self.summary_format = summary_format
        self.model_path = LLM_MODEL_PATH

        if not self.model_path.exists():
            logger.error("LLM model not found at %s", self.model_path)
            raise RuntimeError(f"LLM model not found: {self.model_path}")

        from llama_cpp import Llama
        logger.info("Loading LLM model: %s", self.model_path.name)
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        logger.info("Summarizer initialized")

    def generate_summary(self, transcript, meeting_duration_minutes=None, fmt=None):
        """
        Generate a meeting summary from transcript.

        Args:
            transcript: Full meeting transcript text
            meeting_duration_minutes: Optional meeting duration
            fmt: Summary format (detailed/bullets/executive/email)

        Returns:
            Formatted summary as markdown string
        """
        if not transcript or len(transcript.strip()) < 10:
            return "# Meeting Summary\n\nNo transcript available to summarize."

        fmt = fmt or self.summary_format
        prompt = self._build_summary_prompt(transcript, meeting_duration_minutes, fmt)

        logger.info("Generating %s summary (%d chars)", fmt, len(transcript))

        try:
            response = self._call_llm(prompt)

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

    def _call_llm(self, prompt):
        """Call the local LLM model."""
        logger.info("Calling local LLM...")

        response = self.llm(
            prompt,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.85,
            echo=False,
        )

        # Defensively validate response structure
        if not isinstance(response, dict):
            logger.error("LLM response is not a dict: %s", type(response))
            logger.error("Raw response: %s", response)
            return None

        if 'choices' not in response or not isinstance(response['choices'], list) or len(response['choices']) == 0:
            logger.error("LLM response missing or invalid 'choices' key: %s", response)
            return None

        choice = response['choices'][0]
        if not isinstance(choice, dict) or 'text' not in choice:
            logger.error("LLM choice[0] missing or invalid 'text' key: %s", choice)
            return None

        text = choice['text'].strip()
        if not isinstance(text, str) or not text:
            logger.error("LLM text is not a non-empty string: %s", text)
            return None

        return text
