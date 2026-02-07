#!/bin/bash
set -e

echo "🚀 Setting up OpenMeet..."

# Check macOS version
if [[ $(sw_vers -productVersion | cut -d. -f1) -lt 12 ]]; then
    echo "❌ macOS 12.0+ required"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.10+ required"
    exit 1
fi

echo "✅ System requirements met"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clone and build Whisper.cpp
if [ ! -d "whisper.cpp" ]; then
    echo "📦 Installing Whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
    mkdir -p build && cd build
    cmake ..
    cmake --build . --config Release
    cd ..
    echo "📥 Downloading Whisper model..."
    bash ./models/download-ggml-model.sh tiny
    cd ..
else
    echo "✅ Whisper.cpp already installed"
fi

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Pull Ollama model
echo "📥 Downloading Ollama model..."
ollama pull llama3.2:3b

# Validate setup
echo "🔍 Validating setup..."
python src/utils/config.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run OpenMeet:"
echo "  source venv/bin/activate"
echo "  python src/app.py"
echo ""
EOF

# Make it executable
chmod +x setup.sh