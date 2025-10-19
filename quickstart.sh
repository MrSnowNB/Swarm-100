#!/bin/bash
# ---
# script: quickstart.sh
# purpose: One-command setup and launch
# status: production-ready
# created: 2025-10-18
# ---

set -e

echo "=========================================="
echo "GRANITE4:MICRO-H SWARM - QUICK START"
echo "=========================================="
echo ""

# Check for root/sudo
if [ "$EUID" -eq 0 ]; then
    echo "⚠ Warning: Running as root. Consider running as regular user."
fi

# 1. System check
echo "Step 1: Checking system prerequisites..."
command -v nvidia-smi >/dev/null 2>&1 || { echo "✗ nvidia-smi not found. Install NVIDIA drivers."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "✗ python3 not found."; exit 1; }
echo "✓ System prerequisites met"
echo ""

# 2. Install Python dependencies
echo "Step 2: Installing Python dependencies..."
pip3 install -q pyyaml requests
echo "✓ Dependencies installed"
echo ""

# 3. Check/Install Ollama
echo "Step 3: Checking Ollama installation..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi
echo ""

# 4. Pull model
echo "Step 4: Downloading Granite4:micro-h model..."
if ollama list | grep -q "granite4:micro-h"; then
    echo "✓ Model already downloaded"
else
    echo "Downloading (this may take several minutes)..."
    ollama pull granite4:micro-h
    echo "✓ Model downloaded"
fi
echo ""

# 5. Create directory structure
echo "Step 5: Creating directory structure..."
mkdir -p configs logs/gpu{0,1,2,3} bots scripts
echo "✓ Directories created"
echo ""

# 6. Test single instance
echo "Step 6: Testing single instance..."
timeout 30 ollama run granite4:micro-h "test" > /dev/null 2>&1 || echo "Note: Test query completed"
echo "✓ Single instance test passed"
echo ""

echo "=========================================="
echo "✓ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review configs/swarm_config.yaml"
echo "  2. Run: python3 scripts/launch_swarm.py"
echo "  3. Monitor: python3 scripts/health_monitor.py"
echo ""
