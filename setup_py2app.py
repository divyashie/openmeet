"""
py2app setup script for OpenMeet
Build with: python setup_py2app.py py2app
"""
import sys
sys.setrecursionlimit(10000)  # torch's deep AST tree needs this

from setuptools import setup

APP = ['src/app.py']
APP_NAME = 'OpenMeet'

DATA_FILES = [
    # Icon assets -> Contents/Resources/assets/
    ('assets', [
        'src/assets/openmeet.png',
        'src/assets/openmeet_menu.png',
        'src/assets/openmeet_meeting.png',
        'src/assets/openmeet_speechbubble.png',
    ]),
    # Whisper binary -> Contents/Resources/whisper.cpp/build/bin/
    ('whisper.cpp/build/bin', [
        'whisper.cpp/build/bin/whisper-cli',
    ]),
    # Whisper model -> Contents/Resources/whisper.cpp/models/
    ('whisper.cpp/models', [
        'whisper.cpp/models/ggml-base.bin',
    ]),
    # LLM model -> Contents/Resources/models/
    ('models', [
        'models/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    ]),
]

OPTIONS = {
    'argv_emulation': False,  # Must be False for rumps menu bar apps
    'iconfile': 'src/assets/openmeet.icns',
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleIdentifier': 'com.divyashie.openmeet',
        'CFBundleVersion': '0.2.0',
        'CFBundleShortVersionString': '0.2.0',
        'LSUIElement': True,  # Menu bar only, no Dock icon
        'NSMicrophoneUsageDescription': (
            'OpenMeet needs microphone access to record '
            'and transcribe meeting audio.'
        ),
        'NSHumanReadableCopyright': 'Copyright 2026 Divyashie.',
    },
    'includes': [
        'rumps',
        'pyaudio',
        'sounddevice',
        'numpy',
        'llama_cpp',
        'dotenv',
        'wave',
        'json',
        'queue',
        'threading',
    ],
    'packages': [
        'torch',
        'torchaudio',
        'numpy',
        'scipy',
        'sklearn',
        'sounddevice',
        'certifi',
        'urllib3',
        'charset_normalizer',
        'pytorch_lightning',
        'pytorch_metric_learning',
        'torchmetrics',
    ],
    # pyannote is a namespace package — exclude from scan, copied in build.sh
    'excludes': [
        'pyannote',
        'pyannote.audio',
        'pyannote.core',
        'pyannote.pipeline',
        'pyannote.database',
        'pyannote.metrics',
    ],
    'semi_standalone': False,
    'site_packages': True,
    'strip': True,
    'optimize': 1,
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
