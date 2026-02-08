"""
OpenMeet - Main Application
Real-time meeting transcription and AI summaries
"""
import rumps
import sys
from pathlib import Path
import threading
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import validate_setup, settings
from audio_capture import AudioCapture
from transcriber import Transcriber
from summarizer import Summarizer

logger = logging.getLogger(__name__)


class OpenMeetApp(rumps.App):
    """Main menu bar application"""

    def __init__(self):
        logger.info("Validating setup...")
        if not validate_setup():
            rumps.alert(
                title="Setup Incomplete",
                message="Please complete setup before running OpenMeet. Check terminal for details."
            )
            rumps.quit_application()
            return

        super().__init__(
            name="OpenMeet",
            icon=None,
            quit_button=None
        )

        # State
        self.is_recording = False
        self.transcript_window = None
        self.full_transcript = []

        # Audio capture
        self.audio_capture = AudioCapture()
        self.recording_thread = None

        # Transcriber
        self.transcriber = Transcriber()
        self.transcription_thread = None

        # Summarizer
        try:
            self.summarizer = Summarizer()
        except ConnectionError:
            logger.warning("Summarizer not available (Ollama not running)")
            self.summarizer = None

        # Diarizer (lazy, only if enabled)
        self.diarizer = None
        if settings.get("diarization_enabled"):
            try:
                from diarizer import Diarizer
                self.diarizer = Diarizer(hf_token=settings.get("huggingface_token"))
                logger.info("Diarizer enabled")
            except Exception as e:
                logger.warning("Diarizer not available: %s", e)

        # Set up menu
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.start_recording),
            rumps.MenuItem("Stop Recording", callback=self.stop_recording),
            rumps.separator,
            rumps.MenuItem("Show Transcript", callback=self.show_transcript),
            rumps.MenuItem("View Past Meetings", callback=self.view_past_meetings),
            rumps.MenuItem("Open Latest Summary", callback=self.open_latest_summary),
            rumps.separator,
            rumps.MenuItem("Settings", callback=self.open_settings),
            rumps.MenuItem("About", callback=self.show_about),
            rumps.separator,
            rumps.MenuItem("Quit OpenMeet", callback=self.quit_app)
        ]

        # Initial state - disable until recording
        self.menu["Stop Recording"].set_callback(None)
        self.menu["Show Transcript"].set_callback(None)

        self.title = "🎙️"

        logger.info("OpenMeet initialized!")

    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        """Start recording meeting audio"""
        if self.is_recording:
            return

        logger.info("=" * 50)
        logger.info("STARTING NEW RECORDING SESSION")
        logger.info("=" * 50)

        success = self.audio_capture.start_recording()

        if not success:
            rumps.alert(
                title="Recording Failed",
                message="Could not start audio capture.\n\nPlease check:\n"
                        "- Microphone permission in System Preferences\n"
                        "- Microphone is connected"
            )
            return

        self.is_recording = True
        self.full_transcript = []

        # Update UI
        self.title = "🔴"
        self.menu["Start Recording"].set_callback(None)
        self.menu["Stop Recording"].set_callback(self.stop_recording)
        self.menu["Show Transcript"].set_callback(self.show_transcript)

        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Started",
            message="Meeting audio is being captured and transcribed"
        )

        # Start real-time transcription in background
        def transcription_worker():
            self.transcriber.transcribe_stream(
                self.audio_capture,
                self._on_transcript_chunk
            )

        self.transcription_thread = threading.Thread(
            target=transcription_worker,
            daemon=True
        )
        self.transcription_thread.start()
        logger.info("Real-time transcription started")

    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        """Stop recording and generate summary"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Ask user to choose summary format
        format_window = rumps.Window(
            title="Summary Format",
            message=(
                "Choose a summary format:\n\n"
                "  1 - Detailed (full structured summary)\n"
                "  2 - Bullet Points (key points + action items)\n"
                "  3 - Executive Brief (2-3 paragraph overview)\n"
                "  4 - Email Recap (ready to send to attendees)\n"
            ),
            default_text="1",
            ok="Generate",
            cancel="Skip Summary"
        )
        format_response = format_window.run()

        format_map = {"1": "detailed", "2": "bullets", "3": "executive", "4": "email"}
        if format_response.clicked:
            chosen_format = format_map.get(format_response.text.strip(), "detailed")
        else:
            chosen_format = None  # Skip summary

        # Update UI
        self.title = "⏳"
        self.menu["Start Recording"].set_callback(self.start_recording)
        self.menu["Stop Recording"].set_callback(None)

        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Stopped",
            message="Saving audio and transcript..."
        )

        def process_recording():
            logger.info("=" * 50)
            logger.info("STOPPING RECORDING SESSION")
            logger.info("=" * 50)

            wav_path = self.audio_capture.stop_recording()

            if wav_path:
                logger.info("Audio saved: %s", wav_path.name)

                # Transcribe the full recording for final accuracy
                logger.info("Generating final transcript...")
                full_transcript_text = self.transcriber.transcribe_file(wav_path)

                # Combine with real-time transcript if available
                if self.full_transcript:
                    combined_transcript = "\n".join(self.full_transcript)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    combined_transcript = f"[{timestamp}] {full_transcript_text}"

                # Speaker diarization (if enabled)
                if self.diarizer and combined_transcript:
                    try:
                        logger.info("Running speaker diarization...")
                        combined_transcript = self.diarizer.process(
                            wav_path, combined_transcript
                        )
                    except Exception as e:
                        logger.error("Diarization failed, using unlabeled transcript: %s", e)

                # Save transcript to text file
                txt_path = wav_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write("Meeting Transcript\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(combined_transcript)
                    f.write("\n")

                logger.info("Transcript saved: %s", txt_path.name)

                # Generate AI summary
                if chosen_format and self.summarizer and combined_transcript:
                    logger.info("Generating %s summary...", chosen_format)

                    summary = self.summarizer.generate_summary(
                        combined_transcript,
                        meeting_duration_minutes=None,
                        fmt=chosen_format
                    )

                    # Save with format indicator in filename
                    format_suffix = "" if chosen_format == "detailed" else f"_{chosen_format}"
                    summary_path = wav_path.with_name(
                        wav_path.stem + format_suffix + ".md"
                    )
                    with open(summary_path, 'w') as f:
                        f.write(summary)

                    logger.info("Summary saved: %s", summary_path.name)

                    rumps.notification(
                        title="OpenMeet",
                        subtitle="Summary Ready!",
                        message=f"{chosen_format.title()} summary generated successfully"
                    )

                    logger.info("=" * 60)
                    logger.info("MEETING SUMMARY:")
                    logger.info("=" * 60)
                    logger.info("\n%s", summary)
                else:
                    if not chosen_format:
                        logger.info("Summary skipped by user")
                    else:
                        logger.warning("Summary not generated (Ollama not available)")

                    rumps.notification(
                        title="OpenMeet",
                        subtitle="Session Complete",
                        message="Audio and transcript saved!"
                    )
            else:
                logger.error("Failed to save audio")

            self.title = "🎙️"
            logger.info("=" * 50)

        threading.Thread(target=process_recording, daemon=True).start()

    def _on_transcript_chunk(self, transcript):
        """Callback for each transcribed chunk"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {transcript}"

        self.full_transcript.append(entry)

        logger.info("=" * 60)
        logger.info("TRANSCRIPT: %s", entry)
        logger.info("=" * 60)

    @rumps.clicked("Open Latest Summary")
    def open_latest_summary(self, _):
        """Open the most recent summary"""
        from utils.config import TRANSCRIPTS_DIR
        import subprocess

        summaries = list(TRANSCRIPTS_DIR.glob("meeting_*.md"))

        if not summaries:
            rumps.alert(
                title="No Summaries",
                message="No meeting summaries found yet.\n\nRecord a meeting to generate a summary!"
            )
            return

        latest = max(summaries, key=lambda p: p.stat().st_mtime)
        subprocess.run(["open", "-t", str(latest)])

    @rumps.clicked("Show Transcript")
    def show_transcript(self, _):
        """Show current transcript"""
        if not self.full_transcript:
            rumps.alert(
                title="Transcript",
                message="No transcript yet.\n\nTranscript will appear as you speak."
            )
            return

        transcript_text = "\n\n".join(self.full_transcript)

        if len(transcript_text) > 500:
            transcript_text = transcript_text[-500:] + "\n\n(showing last 500 characters)"

        rumps.alert(
            title="Current Transcript",
            message=transcript_text
        )

    @rumps.clicked("View Past Meetings")
    def view_past_meetings(self, _):
        """Open folder with past transcripts"""
        from utils.config import TRANSCRIPTS_DIR
        import subprocess

        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(TRANSCRIPTS_DIR)])

    @rumps.clicked("Settings")
    def open_settings(self, _):
        """Open settings dialogs"""
        current = settings.all()

        window = rumps.Window(
            title="OpenMeet Settings",
            message=(
                f"Current Settings:\n\n"
                f"  Whisper Model: {current['whisper_model']}\n"
                f"  Language: {current['language']}\n"
                f"  Summary Format: {current['summary_format']}\n"
                f"  Diarization: {'On' if current['diarization_enabled'] else 'Off'}\n"
                f"\nEnter setting to change:\n"
                f"  model / language / format / diarization"
            ),
            default_text="",
            ok="Next",
            cancel="Close"
        )
        response = window.run()

        if response.clicked:
            self._handle_setting_change(response.text.strip().lower())

    def _handle_setting_change(self, setting_name):
        """Handle individual setting change via dialog"""
        if setting_name == "model":
            window = rumps.Window(
                title="Whisper Model",
                message="Choose model size:\n  tiny / base / small / medium",
                default_text=settings.get("whisper_model")
            )
            resp = window.run()
            if resp.clicked and resp.text.strip() in ("tiny", "base", "small", "medium"):
                settings.set("whisper_model", resp.text.strip())
                settings.save()
                rumps.alert("Settings", f"Whisper model set to: {resp.text.strip()}\n\nRestart app to apply.")

        elif setting_name == "language":
            window = rumps.Window(
                title="Language",
                message="Enter language code:\n  en / fr / es / de / ja / zh / etc.",
                default_text=settings.get("language")
            )
            resp = window.run()
            if resp.clicked and resp.text.strip():
                settings.set("language", resp.text.strip())
                settings.save()
                rumps.alert("Settings", f"Language set to: {resp.text.strip()}")

        elif setting_name == "format":
            window = rumps.Window(
                title="Summary Format",
                message="Choose format:\n  detailed / bullets / executive / email",
                default_text=settings.get("summary_format")
            )
            resp = window.run()
            if resp.clicked and resp.text.strip() in ("detailed", "bullets", "executive", "email"):
                settings.set("summary_format", resp.text.strip())
                settings.save()
                rumps.alert("Settings", f"Summary format set to: {resp.text.strip()}")

        elif setting_name == "diarization":
            current = settings.get("diarization_enabled")
            new_val = not current
            settings.set("diarization_enabled", new_val)
            settings.save()
            state = "enabled" if new_val else "disabled"
            rumps.alert("Settings", f"Speaker diarization {state}.\n\nRestart app to apply.")

        else:
            rumps.alert("Settings", f"Unknown setting: '{setting_name}'\n\nTry: model / language / format / diarization")

    @rumps.clicked("About")
    def show_about(self, _):
        """Show about dialog"""
        rumps.alert(
            title="About OpenMeet",
            message="OpenMeet v0.2.0\n\nOpen-source meeting transcription\n100% local, privacy-first\n\nBuilt with ❤️ by Divyashie"
        )

    @rumps.clicked("Quit OpenMeet")
    def quit_app(self, _):
        """Quit application"""
        if self.is_recording:
            response = rumps.alert(
                title="Recording in Progress",
                message="You are currently recording. Do you want to stop and quit?",
                ok="Stop & Quit",
                cancel="Cancel"
            )
            if response == 0:
                return

        logger.info("Quitting OpenMeet...")
        self.audio_capture.cleanup()
        rumps.quit_application()


def main():
    """Entry point"""
    logger.info("Starting OpenMeet...")
    app = OpenMeetApp()
    app.run()


if __name__ == "__main__":
    main()
