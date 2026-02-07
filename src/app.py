"""
OpenMeet - Main Application
Real-time meeting transcription and AI summaries
"""
import rumps
import sys
from pathlib import Path
from audio_capture import AudioCapture
import threading

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import validate_setup

class OpenMeetApp(rumps.App):
    """Main menu bar application"""
    
    def __init__(self):
        # Validate setup before starting
        print("🔍 Validating setup...")
        if not validate_setup():
            rumps.alert(
                title="Setup Incomplete",
                message="Please complete setup before running OpenMeet. Check terminal for details."
            )
            rumps.quit_application()
            return
        
        super().__init__(
            name="OpenMeet",
            icon=None,  # We'll add icon later
            quit_button=None  # Custom quit button
        )
        
        # State
        self.is_recording = False
        self.transcript_window = None
        self.full_transcript = []
        
        # Audio capture
        self.audio_capture = AudioCapture()
        self.recording_thread = None
        
        # Set up menu
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.start_recording),
            rumps.MenuItem("Stop Recording", callback=self.stop_recording),
            rumps.separator,
            rumps.MenuItem("Show Transcript", callback=self.show_transcript),
            rumps.MenuItem("View Past Meetings", callback=self.view_past_meetings),
            rumps.separator,
            rumps.MenuItem("Settings", callback=self.open_settings),
            rumps.MenuItem("About", callback=self.show_about),
            rumps.separator,
            rumps.MenuItem("Quit OpenMeet", callback=self.quit_app)
        ]
        
        # Initial state - disable until recording
        self.menu["Stop Recording"].set_callback(None)
        self.menu["Show Transcript"].set_callback(None)
        
        # Set initial title
        self.title = "🎙️"
        
        print("✅ OpenMeet initialized!")
    
    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        """Start recording meeting audio"""
        if self.is_recording:
            return
        
        # Start audio capture
        print("\n" + "="*50)
        print("🎬 STARTING NEW RECORDING SESSION")
        print("="*50)
        
        success = self.audio_capture.start_recording()
        
        if not success:
            rumps.alert(
                title="Recording Failed",
                message="Could not start audio capture.\n\nPlease check:\n• Microphone permission in System Preferences\n• Microphone is connected"
            )
            return
        
        self.is_recording = True
        self.full_transcript = []
        
        # Update UI
        self.title = "🔴"
        self.menu["Start Recording"].set_callback(None)
        self.menu["Stop Recording"].set_callback(self.stop_recording)
        self.menu["Show Transcript"].set_callback(self.show_transcript)
        
        # Show notification
        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Started",
            message="Meeting audio is being captured"
        )
        
        # Update UI
        self.title = "🔴"  # Red dot when recording
        self.menu["Start Recording"].set_callback(None)  # Disable
        self.menu["Stop Recording"].set_callback(self.stop_recording)  # Enable
        self.menu["Show Transcript"].set_callback(self.show_transcript)  # Enable
        
        # Show notification
        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Started",
            message="Meeting audio is being captured and transcribed"
        )
        
        print("🔴 Recording started...")
    
    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        """Stop recording and generate summary"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Update UI
        self.title = "⏳"
        self.menu["Start Recording"].set_callback(self.start_recording)
        self.menu["Stop Recording"].set_callback(None)
        
        # Show notification
        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Stopped",
            message="Saving audio file..."
        )
        
        # Stop audio capture in background thread
        def process_recording():
            print("\n" + "="*50)
            print("🛑 STOPPING RECORDING SESSION")
            print("="*50)
            
            wav_path = self.audio_capture.stop_recording()
            
            if wav_path:
                print(f"\n💾 Audio saved: {wav_path.name}")
                
                # TODO: Phase 3 will add transcription here
                
                # Show completion notification
                rumps.notification(
                    title="OpenMeet",
                    subtitle="Recording Saved",
                    message=f"Saved as {wav_path.name}"
                )
            else:
                print(f"\n❌ Failed to save audio")
            
            # Restore icon
            self.title = "🎙️"
            
            print("\n" + "="*50 + "\n")
        
        threading.Thread(target=process_recording, daemon=True).start()
        
        # Update UI
        self.title = "⏳"  # Hourglass while processing
        self.menu["Start Recording"].set_callback(self.start_recording)  # Enable
        self.menu["Stop Recording"].set_callback(None)  # Disable
        
        # Show notification
        rumps.notification(
            title="OpenMeet",
            subtitle="Recording Stopped",
            message="Processing complete!"
        )
        
        print("⏹️ Recording stopped...")
        
        # Restore icon
        self.title = "🎙️"
    
    @rumps.clicked("Show Transcript")
    def show_transcript(self, _):
        """Show/hide transcript window"""
        rumps.alert(
            title="Transcript",
            message="Transcript window will appear here.\n\nComing in Phase 5!"
        )
    
    @rumps.clicked("View Past Meetings")
    def view_past_meetings(self, _):
        """Open folder with past transcripts"""
        from utils.config import TRANSCRIPTS_DIR
        import subprocess
        
        # Create directory if it doesn't exist
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Open in Finder
        subprocess.run(["open", str(TRANSCRIPTS_DIR)])
    
    @rumps.clicked("Settings")
    def open_settings(self, _):
        """Open settings window"""
        rumps.alert(
            title="Settings",
            message="Settings coming soon!\n\n- Choose audio device\n- Select Whisper model\n- Configure Ollama model\n- Set save location"
        )
    
    @rumps.clicked("About")
    def show_about(self, _):
        """Show about dialog"""
        rumps.alert(
            title="About OpenMeet",
            message="OpenMeet v0.1.0\n\nOpen-source meeting transcription\n100% local, privacy-first\n\nBuilt with ❤️ by Divyashie"
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
            if response == 0:  # User clicked Cancel
                return
        
        print("👋 Quitting OpenMeet...")
        # Cleanup audio resources
        self.audio_capture.cleanup()
        rumps.quit_application()

def main():
    """Entry point"""
    print("🎙️ Starting OpenMeet...\n")
    app = OpenMeetApp()
    app.run()

if __name__ == "__main__":
    main()