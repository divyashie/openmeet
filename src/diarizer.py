"""
Speaker diarization using pyannote.audio
Identifies who spoke when, aligned with whisper transcription timestamps.
"""
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Diarizer:
    """
    Speaker diarization using pyannote.audio pipeline.

    Workflow:
        1. Run pyannote pipeline on WAV file -> speaker segments
        2. Parse whisper timestamps from transcript
        3. Align whisper segments with speaker segments via temporal overlap
        4. Output speaker-labeled transcript lines
    """

    def __init__(self, hf_token=None):
        """
        Initialize diarizer. Downloads model on first run.

        Args:
            hf_token: Hugging Face token for model access.
        """
        self.hf_token = hf_token
        self.pipeline = None
        self._initialized = False

    def _ensure_pipeline(self):
        """Lazy-load the pyannote pipeline."""
        if self._initialized:
            return

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote speaker diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            self._initialized = True
            logger.info("Pyannote pipeline loaded successfully")

        except ImportError:
            logger.error(
                "pyannote.audio not installed. "
                "Install with: pip install pyannote.audio"
            )
            raise
        except Exception as e:
            logger.error("Failed to load pyannote pipeline: %s", e)
            raise

    def diarize(self, wav_path):
        """
        Run speaker diarization on a WAV file.

        Args:
            wav_path: Path to the WAV audio file

        Returns:
            List of (start_seconds, end_seconds, speaker_label) tuples
        """
        self._ensure_pipeline()

        logger.info("Running diarization on %s...", wav_path.name)
        diarization = self.pipeline(str(wav_path))

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        # Normalize speaker labels to "Speaker 1", "Speaker 2", etc.
        speaker_map = {}
        counter = 1
        normalized = []
        for start, end, speaker in segments:
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {counter}"
                counter += 1
            normalized.append((start, end, speaker_map[speaker]))

        logger.info(
            "Diarization complete: %d segments, %d speakers",
            len(normalized), len(speaker_map)
        )
        return normalized

    @staticmethod
    def parse_whisper_timestamps(raw_transcript):
        """
        Parse whisper output that contains timestamp markers.

        Handles:
            [HH:MM:SS.mmm --> HH:MM:SS.mmm] text  (whisper-cpp output)
            [HH:MM:SS] text                         (app runtime format)

        Returns:
            List of (start_sec, end_sec, text) tuples
        """
        segments = []

        # whisper-cpp format
        whisper_pattern = re.compile(
            r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*'
            r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s*(.*)'
        )

        # app runtime format
        runtime_pattern = re.compile(
            r'\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)'
        )

        for line in raw_transcript.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            m = whisper_pattern.match(line)
            if m:
                start = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                         + int(m.group(3)) + int(m.group(4)) / 1000)
                end = (int(m.group(5)) * 3600 + int(m.group(6)) * 60
                       + int(m.group(7)) + int(m.group(8)) / 1000)
                text = m.group(9).strip()
                if text:
                    segments.append((start, end, text))
                continue

            m = runtime_pattern.match(line)
            if m:
                start = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                         + int(m.group(3)))
                text = m.group(4).strip()
                if text:
                    segments.append((start, start + 10.0, text))

        return segments

    @staticmethod
    def align_speakers_with_transcript(speaker_segments, transcript_segments):
        """
        Align speaker diarization segments with transcript segments
        using maximum temporal overlap.

        Args:
            speaker_segments: List of (start, end, speaker_label)
            transcript_segments: List of (start, end, text)

        Returns:
            List of (speaker_label, text) tuples
        """
        labeled = []

        for t_start, t_end, text in transcript_segments:
            best_speaker = "Unknown"
            best_overlap = 0.0

            for s_start, s_end, speaker in speaker_segments:
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            labeled.append((best_speaker, text))

        return labeled

    def process(self, wav_path, raw_transcript):
        """
        Full diarization pipeline: diarize audio, parse transcript, align, format.

        Args:
            wav_path: Path to WAV file
            raw_transcript: Raw transcript text (with timestamps)

        Returns:
            Speaker-labeled transcript as formatted string
        """
        speaker_segments = self.diarize(wav_path)

        transcript_segments = self.parse_whisper_timestamps(raw_transcript)

        if not transcript_segments:
            logger.warning("No timestamped segments found; returning unlabeled transcript")
            return raw_transcript

        labeled = self.align_speakers_with_transcript(speaker_segments, transcript_segments)

        lines = []
        for speaker, text in labeled:
            lines.append(f"{speaker}: {text}")

        return "\n".join(lines)
