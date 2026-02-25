<div align="center">

# 🎙️ OpenMeet

**Privacy-first meeting transcription and AI summaries — entirely on your machine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS-lightgrey?logo=apple)](https://www.apple.com/macos/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/divyashie/openmeet?style=social)](https://github.com/divyashie/openmeet)

[Features](#features) · [Quick Start](#quick-start) · [Installation](#installation) · [Roadmap](#roadmap) · [Contributing](#contributing)

---

> OpenMeet is a **100% local**, open-source macOS menu bar app that transcribes your meetings in real time and generates AI-powered summaries — no cloud, no subscriptions, no data leaving your device.

</div>

---

## Why OpenMeet?

Most meeting transcription tools send your audio to the cloud. OpenMeet is different:

| | OpenMeet | Cloud Tools |
|---|---|---|
| **Privacy** | Everything runs locally | Audio sent to remote servers |
| **Cost** | Free forever | Subscription required |
| **Internet** | Works offline | Requires connection |
| **Auditability** | Open source | Closed source |
| **Customizable** | Yes — fork & modify | No |

---

## Features

- **Real-time transcription** — live captions powered by [Whisper.cpp](https://github.com/ggerganov/whisper.cpp), running fully on-device
- **AI meeting summaries** — automatic summaries, action items, and key decisions using a local LLM
- **Multiple summary formats** — detailed, bullet points, executive brief, or email-ready
- **Menu bar native** — lives quietly in your macOS menu bar; one click to start/stop
- **Works with any app** — Zoom, Google Meet, Microsoft Teams, Slack, and any audio source
- **Speaker diarization** *(in progress)* — identify who said what via [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **Zero telemetry** — no analytics, no tracking, no external API calls

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Speech-to-Text | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) (C++, local) |
| AI Summarization | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (GGUF models) |
| Menu Bar UI | [Rumps](https://github.com/jaredks/rumps) |
| Windows & Dialogs | Tkinter |
| Audio Capture | PyAudio + SoundDevice |
| Speaker ID | [Pyannote.audio](https://github.com/pyannote/pyannote-audio) *(optional)* |

---

## Quick Start

### Prerequisites

- macOS 12.0+
- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for larger models)
- Xcode Command Line Tools (for building Whisper.cpp)

```bash
xcode-select --install
```

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/divyashie/openmeet.git
cd openmeet

# 2. Run the one-line setup script
./setup.sh

# 3. Launch
python src/app.py
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies
- Build Whisper.cpp
- Download the default Whisper model (`base`)

---

## Installation

### Manual Setup

<details>
<summary><strong>Step 1 — Clone & create environment</strong></summary>

```bash
git clone https://github.com/divyashie/openmeet.git
cd openmeet
python3 -m venv venv
source venv/bin/activate
```
</details>

<details>
<summary><strong>Step 2 — Install Python dependencies</strong></summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Step 3 — Build Whisper.cpp</strong></summary>

```bash
cd whisper.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
cd ..

# Download a model (options: tiny, base, small, medium, large)
bash ./models/download-ggml-model.sh base
cd ..
```
</details>

<details>
<summary><strong>Step 4 — Download an LLM (for summaries)</strong></summary>

Place a GGUF model in the `models/` directory. The default expected model is:

```
models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

You can override this with an environment variable:

```bash
export OPENMEET_LLM_MODEL=your-model.gguf
```

Recommended models from [HuggingFace](https://huggingface.co/models?library=gguf):
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` — fast, low memory
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` — higher quality summaries
</details>

### Build a Standalone App (macOS)

```bash
./build.sh
open dist/OpenMeet.app
```

---

## Usage

1. **Start OpenMeet** — the microphone icon appears in your menu bar
2. **Join a meeting** in Zoom, Meet, Teams, or Slack
3. **Click "Start Recording"** in the menu bar
4. **Talk** — the transcript updates in real time in the transcript window
5. **Click "Stop & Summarize"** when your meeting ends
6. **Copy or export** your transcript and summary

### Keyboard Shortcuts

| Action | Shortcut |
|---|---|
| Start / Stop | Coming soon |
| Open transcript | Coming soon |

---

## Roadmap

### Done ✅
- [x] macOS menu bar app
- [x] Audio capture with device selection
- [x] Real-time transcription (Whisper.cpp)
- [x] Local LLM summarization (llama-cpp-python)
- [x] Multiple summary formats (detailed, bullets, executive, email)
- [x] Transcript window UI
- [x] Local file storage for transcripts and settings
- [x] PyInstaller bundling for `.app` distribution

### In Progress 🚧
- [ ] Speaker diarization (who said what)
- [ ] Settings panel UI
- [ ] Transcript search and filtering

### Planned 📋
- [ ] Custom summary prompts
- [ ] Export to PDF, Markdown, DOCX
- [ ] Keyboard shortcuts
- [ ] Calendar app integration
- [ ] Batch processing of audio files
- [ ] Optional encrypted cloud sync

---

## Privacy & Security

OpenMeet is designed from the ground up to keep your data private:

- **No cloud uploads** — audio never leaves your machine
- **No external API calls** — transcription and summarization are fully local
- **No telemetry** — we collect nothing
- **Works offline** — no internet connection required
- **Open source** — every line of code is auditable

The only optional exception is downloading speaker diarization models from HuggingFace, which requires a free HuggingFace token.

---

## Troubleshooting

<details>
<summary><strong>"LLM model not found" error</strong></summary>

Download a GGUF model to the `models/` directory. The default path is:
```
models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```
Or set the environment variable:
```bash
export OPENMEET_LLM_MODEL=your-model.gguf
```
</details>

<details>
<summary><strong>"Whisper executable not found"</strong></summary>

Rebuild Whisper.cpp:
```bash
cd whisper.cpp && mkdir -p build && cd build && cmake .. && cmake --build . --config Release
```
</details>

<details>
<summary><strong>Audio capture issues</strong></summary>

List available audio devices:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```
Set the device manually:
```bash
export OPENMEET_AUDIO_DEVICE_INDEX=<number>
```
</details>

<details>
<summary><strong>Settings not persisting</strong></summary>

Settings are stored at:
- **macOS:** `~/Library/Application Support/OpenMeet/data/settings.json`
- **Linux:** `~/.local/share/OpenMeet/data/settings.json`
- **Windows:** `%APPDATA%\OpenMeet\data\settings.json`
</details>

---

## Contributing

Contributions are very welcome! OpenMeet is a community project and benefits from every improvement, no matter how small.

### How to Contribute

1. **Fork** this repository
2. **Create a branch** for your change: `git checkout -b feat/my-feature`
3. **Make your changes** and add tests where appropriate
4. **Run the tests:** `pytest`
5. **Submit a pull request** — describe what you changed and why

### Good First Issues

Look for issues tagged [`good first issue`](https://github.com/divyashie/openmeet/issues?q=label%3A%22good+first+issue%22) to get started.

### Development Setup

```bash
git clone https://github.com/divyashie/openmeet.git
cd openmeet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

---

## Acknowledgments

OpenMeet stands on the shoulders of great open source projects:

- [**Whisper.cpp**](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov — blazing-fast local speech-to-text
- [**llama-cpp-python**](https://github.com/abetlen/llama-cpp-python) — local LLM inference with GGUF models
- [**Pyannote.audio**](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [**Rumps**](https://github.com/jaredks/rumps) — macOS menu bar apps in Python
- [**PyInstaller**](https://pyinstaller.org) — packaging Python apps

---

## License

[MIT](LICENSE) © 2025 Bhoj Rani Soopal

---

<div align="center">

Built with ❤️ for privacy-conscious professionals.

**[Report a Bug](https://github.com/divyashie/openmeet/issues) · [Request a Feature](https://github.com/divyashie/openmeet/issues) · [Join the Discussion](https://github.com/divyashie/openmeet/discussions)**

</div>
