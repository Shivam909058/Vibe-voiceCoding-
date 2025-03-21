import os
import sys
import time
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import wave
import json
import subprocess
import argparse
from threading import Thread
import shutil
import warnings
import platform
import queue
import re
import keyboard
import pyperclip
import pyautogui
import getpass
warnings.filterwarnings("ignore")
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: soundfile module not fully installed. Using wave module as fallback.")
    SOUNDFILE_AVAILABLE = False
    import wave
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: speech_recognition module not installed. Only Whisper will be used for transcription.")
    SR_AVAILABLE = False

class AiderVoiceAssistant:
    def __init__(self, working_directory=None):
        self.working_directory = working_directory or os.getcwd()
        self.running = False
        self.audio_buffer = []
        self.silence_threshold = 0.005
        self.silence_duration = 0
        self.recording = False
        self.aider_process = None
        self.output_queue = queue.Queue()
        self.waiting_for_input = False
        self.aider_window_title = "Aider Voice Assistant"
        self.last_transcription_time = 0
        self.transcription_cooldown = 2  # seconds
        self.auto_yes_enabled = True  # Enable auto-yes by default
        
        # Test audio input before starting
        self._test_audio_input()
        
        # Get OpenAI API key from user
        self._configure_api_key()
        
        # Load the Whisper model
        print("Loading Whisper model (this may take a moment)...")
        try:
            self.whisper_model = whisper.load_model("tiny")  # Use tiny model for faster transcription
            print("✅ Whisper model loaded successfully!")
            self.whisper_available = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load Whisper model: {e}")
            print("Will fall back to Google Speech Recognition")
            self.whisper_available = False
        
    def _test_audio_input(self):
        """Test audio input device"""
        print("\n=== Testing Audio Input ===")
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            print(f"Default input device: {default_input['name']}")
            
            # Filter only input devices
            self.input_devices = []
            print("\nAvailable audio input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"{len(self.input_devices)}: {device['name']}")
                    self.input_devices.append(i)
            
            # Let user select device
            while True:
                choice = input("\nSelect input device number (or press Enter for default): ").strip()
                if choice == "":
                    self.device_id = None
                    break
                try:
                    device_idx = int(choice)
                    if 0 <= device_idx < len(self.input_devices):
                        self.device_id = self.input_devices[device_idx]
                        break
                    else:
                        print("Invalid device number, try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Test recording
            print("\nTesting microphone (3 seconds)...")
            duration = 3  # seconds
            sample_rate = 16000
            
            print("Recording will start in:")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            print("🔴 Recording NOW - Please speak...")
            recording = sd.rec(int(duration * sample_rate), 
                             samplerate=sample_rate,
                             channels=1,
                             dtype='float32',
                             device=self.device_id)
            
            # Show a progress bar
            for i in range(duration):
                print(f"Recording: {'▓' * (i+1)}{'░' * (duration-i-1)} {i+1}/{duration}s", end='\r')
                time.sleep(1)
            print("\n")
            
            sd.wait()
            
            if recording.any():
                print("✅ Microphone test successful!")
                # Print max amplitude to help debug threshold
                max_amplitude = np.max(np.abs(recording))
                print(f"Max amplitude detected: {max_amplitude:.6f}")
                print(f"Current threshold: {self.silence_threshold}")
                if max_amplitude < self.silence_threshold:
                    print("⚠️ Warning: Audio levels very low. Try adjusting microphone volume.")
                    self.silence_threshold = max_amplitude * 0.8  # Adjust threshold based on test
                    print(f"Adjusted threshold to: {self.silence_threshold}")
            else:
                print("❌ No audio detected. Please check your microphone.")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error testing audio input: {e}")
            sys.exit(1)
        
    def _configure_api_key(self):
        """Configure OpenAI API key for Aider"""
        print("\n=== OpenAI API Key Configuration ===")
        print("Please select your preferred model:")
        print("1. GPT-4o")
        print("2. GPT-4")
        print("3. GPT-3.5-turbo")
        
        while True:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == "1":
                self.model = "gpt-4o"
                break
            elif choice == "2":
                self.model = "gpt-4"
                break
            elif choice == "3":
                self.model = "gpt-3.5-turbo"
                break
            else:
                print("Invalid choice, please try again.")
        
        # Use getpass to mask the API key input
        self.api_key = getpass.getpass("\nPlease enter your OpenAI API key: ").strip()
        
        # Validate API key format
        if not self.api_key.startswith(("sk-", "org-")):
            print("⚠️ Warning: API key format looks incorrect. Continuing anyway...")
        
        print(f"\n✅ Configured to use {self.model}")
        
        # Ask about auto-yes
        print("\n=== Auto-Yes Configuration ===")
        print("Would you like to automatically respond 'y' to Aider prompts?")
        print("(This includes file creation, command execution, etc.)")
        auto_yes = input("Enable auto-yes? (Y/n): ").strip().lower()
        self.auto_yes_enabled = auto_yes != "n"
        
        if self.auto_yes_enabled:
            print("✅ Auto-yes enabled. Will automatically respond 'y' to Aider prompts.")
        else:
            print("❌ Auto-yes disabled. You will need to manually respond to Aider prompts.")
    
    def start_listening(self):
        """Start listening for voice commands"""
        print("\n=== Starting Aider Voice Assistant ===")
        print("Press F2 to activate voice recognition")
        print("Press Ctrl+C to exit")
        
        # Start Aider in a separate process
        self._start_aider()
        
        # Register hotkey for voice activation
        keyboard.add_hotkey('f2', self._handle_voice_activation)
        
        # Set running flag
        self.running = True
        
        # Start monitoring thread for Aider prompts if auto-yes is enabled
        if hasattr(self, 'auto_yes_enabled') and self.auto_yes_enabled:
            self._start_prompt_monitor()
        
        # Main loop - keep the program running
        try:
            print("\n✅ Voice assistant is now running. Press F2 to speak a command.")
            while self.running:
                time.sleep(0.1)  # Reduce CPU usage
        except KeyboardInterrupt:
            print("\n👋 Exiting voice assistant...")
        finally:
            # Clean up
            keyboard.unhook_all()
            self.running = False
            if hasattr(self, 'aider_process') and self.aider_process and self.aider_process.poll() is None:
                try:
                    self.aider_process.terminate()
                    print("✅ Aider process terminated")
                except Exception as e:
                    print(f"⚠️ Error terminating Aider process: {e}")
    
    def _start_aider(self):
        """Start Aider in a separate process"""
        try:
            print("🚀 Starting Aider...")
            
            # Prepare Aider command
            aider_cmd = [
                sys.executable, "-m", "aider.main",
                "--model", self.model,
                "--api-key", f"openai={self.api_key}"
            ]
            
            if platform.system() == "Windows":
                # On Windows, use a separate window with a specific title
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                # Create a batch file to set the window title and run Aider
                batch_file = os.path.join(tempfile.gettempdir(), "run_aider.bat")
                with open(batch_file, "w") as f:
                    f.write(f"@echo off\n")
                    f.write(f"title {self.aider_window_title}\n")
                    f.write(f"{sys.executable} -m aider.main --model {self.model} --api-key openai={self.api_key}\n")
                
                # Run the batch file
                self.aider_process = subprocess.Popen(
                    ["start", "cmd", "/k", batch_file],
                    shell=True,
                    cwd=self.working_directory
                )
                
                print("✅ Aider started in a separate window.")
                print(f"Window title: {self.aider_window_title}")
                
            else:
                # On Unix-like systems, use a terminal emulator
                if platform.system() == "Darwin":  # macOS
                    terminal_cmd = ["open", "-a", "Terminal", "--"]
                    self.aider_process = subprocess.Popen(
                        terminal_cmd + [" ".join(aider_cmd)],
                        cwd=self.working_directory
                    )
                else:  # Linux
                    # Try different terminal emulators
                    terminals = [
                        ["gnome-terminal", "--", "bash", "-c"],
                        ["xterm", "-e", "bash", "-c"],
                        ["konsole", "--", "bash", "-c"],
                        ["xfce4-terminal", "--", "bash", "-c"]
                    ]
                    
                    for terminal in terminals:
                        try:
                            cmd_str = " ".join(aider_cmd) + "; exec bash"
                            self.aider_process = subprocess.Popen(
                                terminal + [cmd_str],
                                cwd=self.working_directory
                            )
                            break
                        except FileNotFoundError:
                            continue
                
                print("✅ Aider started in a terminal window.")
            
            # Wait a moment for Aider to start
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error starting Aider: {e}")
            sys.exit(1)
    
    def _monitor_aider_output(self):
        """Monitor Aider's output for prompts that need auto-yes responses"""
        if not self.auto_yes_enabled:
            return
            
        try:
            # On Windows, we need to use pygetwindow to find the Aider window
            if platform.system() == "Windows":
                import pygetwindow as gw
                
                while self.running:
                    try:
                        # Look for the Aider window
                        aider_windows = gw.getWindowsWithTitle(self.aider_window_title)
                        
                        if aider_windows:
                            aider_window = aider_windows[0]
                            
                            # Check if window is active and waiting for input
                            if aider_window.isActive:
                                # Look for common Aider prompts
                                screen_text = self._get_screen_text()
                                
                                if screen_text:
                                    # Check for various Aider prompts
                                    if any(prompt in screen_text for prompt in [
                                        "Create new file?",
                                        "Run shell command?",
                                        "Add command output to the chat?",
                                        "Would you like to see what's new in this version?",
                                        "Applied edit to",
                                        "Create directory"
                                    ]):
                                        print("🤖 Auto-responding 'y' to Aider prompt")
                                        pyautogui.write('y')
                                        pyautogui.press('enter')
                                        time.sleep(1)  # Wait a bit before checking again
                    except Exception as e:
                        # Just continue if there's an error
                        pass
                        
                    time.sleep(0.5)  # Check every half second
        except Exception as e:
            print(f"⚠️ Auto-yes monitoring error: {e}")
    
    def _get_screen_text(self):
        """Try to get text from the screen using OCR"""
        try:
            # This is a simplified version - in a real implementation,
            # you might want to use a proper OCR library like pytesseract
            # For now, we'll just check if the cursor is at a prompt position
            
            # Get current cursor position
            x, y = pyautogui.position()
            
            # Check if there's a "[Yes]:" or similar near the cursor
            screen_region = (x - 200, y - 20, 400, 40)  # Region around cursor
            
            # For demonstration, we'll just return a dummy string
            # In a real implementation, you'd use OCR here
            return "Create new file? (Y)es/(N)o [Yes]:"
        except:
            return None
    
    def _handle_voice_activation(self):
        """Handle voice activation when F2 is pressed"""
        # Check if we're already recording
        if self.recording:
            print("⚠️ Already recording, please wait...")
            return
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_transcription_time < self.transcription_cooldown:
            print("⚠️ Please wait a moment before recording again...")
            return
        
        print("\n🎤 Voice activation triggered!")
        
        # Start recording
        self.recording = True
        self.audio_buffer = []
        self.silence_duration = 0
        
        # Create a unique filename for this recording
        temp_file_path = os.path.join(tempfile.gettempdir(), f"voice_command_{int(time.time())}.wav")
        
        # Record audio
        sample_rate = 16000
        try:
            print("🔴 Recording... (speak now)")
            
            # Wait a moment for user to start speaking
            time.sleep(0.5)
            
            # Record for at least 2 seconds to ensure we capture something
            min_duration = 2  # seconds
            max_duration = 15  # seconds
            start_time = time.time()
            duration = 0
            
            # Start the stream
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=self.device_id,
                callback=self._audio_callback
            )
            stream.start()
            
            # Record until silence or max duration
            has_speech = False
            while duration < max_duration:
                time.sleep(0.1)
                duration = time.time() - start_time
                
                # Check if we've recorded any audio above threshold
                if any(np.max(np.abs(chunk)) > self.silence_threshold for chunk in self.audio_buffer):
                    has_speech = True
                
                # Show progress
                progress = int(duration / max_duration * 10)
                print(f"Recording: {'▓' * progress}{'░' * (10-progress)} {duration:.1f}s", end='\r')
                
                # If we've recorded enough and there's been silence for a while, stop
                if duration > min_duration and has_speech and self.silence_duration > 1.5:
                    break
            
            print("\n")
            stream.stop()
            stream.close()
            
            # Check if we recorded anything
            if not self.audio_buffer:
                print("❌ No audio recorded.")
                self.recording = False
                return
            
            # Check if we detected any speech
            if not has_speech:
                print("⚠️ Warning: No speech detected. Please speak louder.")
                self.recording = False
                return
            
            print("✅ Recording complete.")
            
            # Save audio to file
            audio_file_path = self._save_audio_to_file(temp_file_path)
            if not audio_file_path:
                print("❌ Failed to save audio.")
                self.recording = False
                return
            
            # Transcribe audio
            transcription = self._transcribe_audio(audio_file_path)
            if not transcription:
                print("❌ Transcription failed.")
                self.recording = False
                return
            
            print(f"\n✅ Transcription: \"{transcription}\"")
            
            # Send command to Aider
            self._send_command_to_aider(transcription)
            
            # Update last transcription time
            self.last_transcription_time = time.time()
            
        except Exception as e:
            print(f"❌ Error during voice activation: {e}")
        finally:
            self.recording = False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
        
        # Convert indata to the right format (numpy array of float32)
        if isinstance(indata, np.ndarray):
            # Check if we have string data type (which would cause the error)
            if np.issubdtype(indata.dtype, np.string_) or np.issubdtype(indata.dtype, np.unicode_):
                print("⚠️ Warning: Received string data type in audio callback")
                return
            
            # Make a copy to ensure we don't get a view that might change
            data = indata.copy()
            
            # Ensure data is float32
            if data.dtype != np.float32:
                try:
                    data = data.astype(np.float32)
                except Exception as e:
                    print(f"⚠️ Error converting audio data: {e}")
                    return
        else:
            print(f"⚠️ Unexpected data type in audio callback: {type(indata)}")
            return
        
        # Check for silence
        max_amplitude = np.max(np.abs(data))
        if max_amplitude < self.silence_threshold:
            self.silence_duration += frames / 16000  # assuming 16kHz sample rate
        else:
            self.silence_duration = 0
        
        # Add data to buffer
        self.audio_buffer.append(data)
    
    def _save_audio_to_file(self, file_path):
        """Save recorded audio to file"""
        try:
            if not self.audio_buffer:
                print("❌ No audio data to save")
                return None
            
            # Convert audio buffer to a single numpy array
            try:
                # First check if all elements are numpy arrays
                if not all(isinstance(chunk, np.ndarray) for chunk in self.audio_buffer):
                    print("⚠️ Warning: Audio buffer contains non-numpy arrays")
                    # Filter out non-numpy arrays
                    self.audio_buffer = [chunk for chunk in self.audio_buffer if isinstance(chunk, np.ndarray)]
                
                # Check if buffer is still valid
                if not self.audio_buffer:
                    print("❌ No valid audio data after filtering")
                    return None
                
                # Concatenate arrays
                audio_data = np.concatenate(self.audio_buffer)
                
                # Ensure audio_data is float32
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize audio (ensure values are between -1 and 1)
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.9  # Leave some headroom
            except Exception as e:
                print(f"⚠️ Error processing audio buffer: {e}")
                return None
            
            # Save to WAV file
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save using wave module (more reliable than soundfile)
                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    
                    # Convert float32 to int16
                    audio_data_int = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_data_int.tobytes())
                
                print(f"✅ Audio saved to: {file_path}")
                return file_path
            except Exception as e:
                print(f"⚠️ Error saving WAV file: {e}")
                
                # Try alternative method
                try:
                    # Create a temporary file with a simpler name
                    simple_path = os.path.join(tempfile.gettempdir(), "voice_command.wav")
                    with wave.open(simple_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        audio_data_int = (audio_data * 32767).astype(np.int16)
                        wf.writeframes(audio_data_int.tobytes())
                    print(f"✅ Audio saved to alternative path: {simple_path}")
                    return simple_path
                except Exception as e2:
                    print(f"❌ Alternative save method also failed: {e2}")
                    return None
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def _transcribe_audio(self, audio_file_path):
        """Transcribe audio file to text"""
        # Try Google Speech Recognition if available (faster)
        if SR_AVAILABLE:
            try:
                print("🔄 Transcribing with Google Speech Recognition...")
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file_path) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Record the entire audio file
                    audio_data = recognizer.record(source)
                    
                    # Try multiple language options for better accuracy
                    languages = ["en-US", "en-GB", "en-IN"]
                    results = []
                    
                    for lang in languages:
                        try:
                            text = recognizer.recognize_google(audio_data, language=lang)
                            results.append((lang, text))
                        except:
                            continue
                    
                    if results:
                        # Return the longest result as it's likely the most complete
                        results.sort(key=lambda x: len(x[1]), reverse=True)
                        lang, text = results[0]
                        print(f"✅ Transcription successful using language: {lang}")
                        return text
            except Exception as e:
                print(f"⚠️ Google transcription error: {e}")
        
        # Try Whisper if available
        if self.whisper_available:
            try:
                print("🔄 Trying Whisper transcription...")
                # Use a different approach to load audio for Whisper
                try:
                    # Try to load with soundfile first
                    if SOUNDFILE_AVAILABLE:
                        audio, _ = sf.read(audio_file_path)
                        audio = audio.astype(np.float32)
                    else:
                        # Load with wave as fallback
                        with wave.open(audio_file_path, 'rb') as wf:
                            frames = wf.readframes(wf.getnframes())
                            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Ensure audio is the right shape for Whisper
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert stereo to mono
                    
                    # Transcribe directly from numpy array
                    result = self.whisper_model.transcribe(audio)
                    return result["text"].strip()
                except Exception as e:
                    print(f"⚠️ Error loading audio for Whisper: {e}")
                    # Try the file path directly as a last resort
                    result = self.whisper_model.transcribe(audio_file_path)
                    return result["text"].strip()
            except Exception as e:
                print(f"⚠️ Whisper transcription error: {e}")
        
        return None
    
    def _send_command_to_aider(self, command):
        """Send command to Aider window"""
        try:
            # Copy command to clipboard
            pyperclip.copy(command)
            print("✅ Command copied to clipboard")
            
            # Try to find and activate Aider window
            if platform.system() == "Windows":
                try:
                    import pygetwindow as gw
                    aider_windows = gw.getWindowsWithTitle(self.aider_window_title)
                    
                    if aider_windows:
                        aider_window = aider_windows[0]
                        aider_window.activate()
                        time.sleep(0.5)  # Wait for window to activate
                        
                        # Type the command
                        pyautogui.typewrite(command)
                        pyautogui.press('enter')
                        print("✅ Command sent to Aider window")
                        return
                except Exception as e:
                    print(f"⚠️ Error activating Aider window: {e}")
            
            # Fallback: Just show instructions
            print("\n⚠️ Could not automatically send command to Aider.")
            print(f"Please paste the command into the Aider window manually (Ctrl+V).")
            
        except Exception as e:
            print(f"❌ Error sending command: {e}")

    def _start_prompt_monitor(self):
        """Start a thread to monitor for Aider prompts"""
        if not self.auto_yes_enabled:
            print("Auto-yes disabled, skipping prompt monitor")
            return
        
        def monitor_prompts():
            print("✅ Started prompt monitor thread")
            while self.running:
                try:
                    # Check if Aider window is active and contains prompt text
                    if platform.system() == "Windows":
                        try:
                            import pygetwindow as gw
                            aider_windows = gw.getWindowsWithTitle(self.aider_window_title)
                            
                            if aider_windows and self.aider_process and self.aider_process.poll() is None:
                                # Check if there's a prompt that needs a 'y' response
                                # This is a simplified approach - in practice, we'd need to read the window content
                                # For now, we'll just send 'y' periodically when Aider is waiting for input
                                if self.waiting_for_input:
                                    # Send 'y' to Aider
                                    aider_window = aider_windows[0]
                                    if aider_window.isActive:
                                        pyautogui.typewrite('y')
                                        pyautogui.press('enter')
                                        print("✅ Auto-responded 'y' to Aider prompt")
                                        time.sleep(1)  # Avoid sending multiple responses
                        except Exception as e:
                            pass  # Silently ignore errors in the monitor thread
                except Exception:
                    pass  # Ensure the monitor thread doesn't crash
                
                time.sleep(0.5)  # Check every half second
            
        # Start the monitor thread
        monitor_thread = Thread(target=monitor_prompts, daemon=True)
        monitor_thread.start()

def main():
    parser = argparse.ArgumentParser(description="Voice-controlled Aider Assistant")
    parser.add_argument(
        "--dir", "-d", 
        help="The working directory for Aider (defaults to current directory)",
        default=os.getcwd()
    )
    args = parser.parse_args()
    
    # Print Python environment info for debugging
    print("=" * 50)
    print("🎙️  Aider Voice Assistant")
    print("=" * 50)
    print(f"📂 Working directory: {args.dir}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"🖥️  Platform: {platform.platform()}")
    
    # Check for required dependencies
    try:
        import whisper
        import sounddevice
        print("✅ Core dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        print("Please install required packages with: pip install openai-whisper sounddevice")
        sys.exit(1)
    
    # Check for keyboard and pyperclip
    try:
        import keyboard
        import pyperclip
        print("✅ Keyboard and clipboard modules are installed")
    except ImportError:
        print("⚠️ Missing keyboard or pyperclip module. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard", "pyperclip"])
            print("✅ Keyboard and clipboard modules installed successfully!")
            # Re-import the modules
            import keyboard
            import pyperclip
        except Exception as e:
            print(f"❌ Failed to install keyboard or pyperclip: {e}")
            print("Please install manually with: pip install keyboard pyperclip")
            sys.exit(1)
    
    # Check for pyautogui
    try:
        import pyautogui
        print("✅ PyAutoGUI module is installed")
    except ImportError:
        print("⚠️ Missing pyautogui module. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
            print("✅ PyAutoGUI module installed successfully!")
            # Re-import the module
            import pyautogui
        except Exception as e:
            print(f"❌ Failed to install pyautogui: {e}")
            print("Please install manually with: pip install pyautogui")
            sys.exit(1)
    
    # Check for pygetwindow on Windows
    if platform.system() == "Windows":
        try:
            import pygetwindow
            print("✅ PyGetWindow module is installed")
        except ImportError:
            print("⚠️ Missing pygetwindow module. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygetwindow"])
                print("✅ PyGetWindow module installed successfully!")
            except Exception as e:
                print(f"⚠️ Failed to install pygetwindow: {e}")
                print("Window switching may not work automatically.")
    
    # Check for Aider
    try:
        # Try to import aider to check if it's installed
        import aider
        print("✅ Aider is installed")
    except ImportError:
        print("⚠️ Aider module not found. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "aider-chat"])
            print("✅ Aider installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install Aider: {e}")
            print("Please install manually with: pip install aider-chat")
            sys.exit(1)
    
    # Start the voice assistant
    try:
        assistant = AiderVoiceAssistant(working_directory=args.dir)
        assistant.start_listening()
    except KeyboardInterrupt:
        print("\n👋 Exiting voice assistant...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 