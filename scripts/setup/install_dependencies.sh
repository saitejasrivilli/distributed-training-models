#!/bin/bash

echo "🚀 Installing dependencies for distributed LLM training..."

# Update pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo "📦 Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "📦 Installing package..."
pip install -e .

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

echo ""
echo "✅ Installation complete!"
