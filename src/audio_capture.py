"""
Audio capture from system audio or microphone
"""
import pyaudio
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import queue
import time

from utils.config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    TRANSCRIPTS_DIR
)

class AudioCapture:
    """Captures audio and saves to WAV files"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.current_session_id = None
        self.audio_queue = queue.Queue()
        
        print("🎤 AudioCapture initialized")
    
    def list_audio_devices(self):
        """List all available audio devices"""
        print("\n🔊 Available audio devices:")
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': info['defaultSampleRate']
            })
            print(f"  [{i}] {info['name']} "
                  f"(Channels: {info['maxInputChannels']}, "
                  f"Rate: {info['defaultSampleRate']})")
        return devices
    
    def find_input_device(self):
        """
        Find the best input device
        Prefers: microphone input
        """
        try:
            # Get default input device
            default_input = self.audio.get_default_input_device_info()
            device_index = default_input['index']
            device_name = default_input['name']
            
            print(f"🎤 Using device: [{device_index}] {device_name}")
            return device_index
            
        except Exception as e:
            print(f"❌ No default input device found: {e}")
            return None
    
    def start_recording(self):
        """Start capturing audio"""
        if self.is_recording:
            print("⚠️ Already recording!")
            return False
        
        try:
            # Create session ID
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Find audio device
            device_index = self.find_input_device()
            if device_index is None:
                print("❌ No audio input device available")
                return False
            
            print(f"\n📼 Starting recording session: {self.current_session_id}")
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.frames = []
            
            # Start stream
            self.stream.start_stream()
            
            print(f"✅ Recording started!")
            print(f"💬 Speak into your microphone...")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            print("\n💡 Tip: Check System Preferences → Security & Privacy → Microphone")
            print("   Make sure Terminal (or your IDE) has microphone permission")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream (called by PyAudio)"""
        if self.is_recording:
            # Store raw audio
            self.frames.append(in_data)
            
            # Also put in queue for real-time transcription
            self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop capturing audio and save to file"""
        if not self.is_recording:
            print("⚠️ Not recording!")
            return None
        
        try:
            print("\n⏹️ Stopping recording...")
            self.is_recording = False
            
            # Stop and close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # Save to WAV file
            wav_path = self._save_audio()
            
            print(f"✅ Recording saved: {wav_path.name}")
            print(f"📊 File size: {wav_path.stat().st_size / 1024:.1f} KB")
            
            return wav_path
            
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            return None
    
    def _save_audio(self):
        """Save recorded audio to WAV file"""
        # Create filename
        filename = f"meeting_{self.current_session_id}.wav"
        wav_path = TRANSCRIPTS_DIR / filename
        
        # Save as WAV
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))
        
        return wav_path
    
    def get_audio_chunk(self, duration_seconds=5):
        """
        Get accumulated audio chunk for transcription
        Returns numpy array of audio data
        """
        chunks_needed = int(SAMPLE_RATE * duration_seconds / CHUNK_SIZE)
        audio_data = []
        
        try:
            for _ in range(chunks_needed):
                chunk = self.audio_queue.get(timeout=0.1)
                audio_data.append(np.frombuffer(chunk, dtype=np.int16))
        except queue.Empty:
            pass
        
        if audio_data:
            return np.concatenate(audio_data)
        return None
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()
        print("🧹 AudioCapture cleaned up")

# Test the module
if __name__ == "__main__":
    print("🎤 Testing Audio Capture\n")
    
    capture = AudioCapture()
    
    # List devices
    capture.list_audio_devices()
    
    print("\n📼 Starting 5-second test recording...")
    print("💬 Say something into your microphone!")
    
    if capture.start_recording():
        # Record for 5 seconds
        time.sleep(5)
        
        # Stop and save
        wav_path = capture.stop_recording()
        
        if wav_path:
            print(f"\n✅ Test successful!")
            print(f"📁 Saved to: {wav_path}")
            print(f"\n🎧 Play it back with: afplay {wav_path}")
        else:
            print("\n❌ Test failed - could not save recording")
    else:
        print("\n❌ Test failed - could not start recording")
    
    capture.cleanup()
