# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OpenMeet
Build with: pyinstaller OpenMeet.spec
"""

a = Analysis(
    ['src/app.py'],
    pathex=['src'],
    binaries=[
        ('whisper.cpp/build/bin/whisper-cli', 'whisper.cpp/build/bin'),
    ],
    datas=[
        ('src/assets/*.png', 'assets'),
        ('whisper.cpp/models/ggml-base.bin', 'whisper.cpp/models'),
        ('models/Llama-3.2-3B-Instruct-Q4_K_M.gguf', 'models'),
    ],
    hiddenimports=[
        'rumps',
        'pyaudio',
        'sounddevice',
        'numpy',
        'llama_cpp',
        'dotenv',
        'pyannote.audio',
        'pyannote.core',
        'pyannote.pipeline',
        'pyannote.database',
        'pyannote.metrics',
        'pytorch_lightning',
        'pytorch_metric_learning',
        'torchmetrics',
        'sklearn',
        'sklearn.cluster',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter'],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenMeet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    name='OpenMeet',
)

app = BUNDLE(
    coll,
    name='OpenMeet.app',
    icon='src/assets/openmeet.icns',
    bundle_identifier='com.divyashie.openmeet',
    info_plist={
        'CFBundleName': 'OpenMeet',
        'CFBundleDisplayName': 'OpenMeet',
        'CFBundleVersion': '0.2.0',
        'CFBundleShortVersionString': '0.2.0',
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': (
            'OpenMeet needs microphone access to record '
            'and transcribe meeting audio.'
        ),
        'NSHumanReadableCopyright': 'Copyright 2026 Divyashie.',
    },
)
