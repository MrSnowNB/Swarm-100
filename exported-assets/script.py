
# Create comprehensive setup documentation and all necessary files for 100-bot swarm
# on Quad Ada6000 Z8 Ubuntu system using Granite4:micro-h

import json

# Document structure with AI-first YAML frontmatter approach
setup_guide = """---
project: granite4-microh-swarm
purpose: Deploy 100-bot distributed memory swarm on Quad Ada6000 Z8
status: implementation-ready
architecture: 4x Ada6000 GPUs, 25 bots per GPU, Ollama + Granite4:micro-h Q4
validation_gates:
  - system_dependencies_installed
  - ollama_installed_verified
  - gpu_detection_confirmed
  - model_downloaded
  - single_bot_test_passed
  - multi_bot_stress_test_passed
  - full_swarm_operational
created: 2025-10-18
updated: 2025-10-18
---

# Granite4:Micro-H 100-Bot Swarm Setup Guide
## Quad Ada6000 Z8 Ubuntu System

This guide provides complete step-by-step instructions and all necessary files to deploy a 100-bot distributed memory swarm using IBM Granite4:micro-h on a Quad Ada6000 workstation.

---

## System Specifications

**Hardware:**
- 4x NVIDIA RTX 6000 Ada Generation GPUs
- 48GB GDDR6 ECC per GPU (192GB total VRAM)
- Ubuntu 22.04+ LTS

**Software Stack:**
- Ollama (latest)
- Granite4:micro-h Q4 quantization
- Python 3.10+
- NVIDIA CUDA drivers 535+
- Docker (optional, for containerized deployment)

---

## Prerequisites Validation

Run these commands to verify your system:

```bash
# Check GPU detection
nvidia-smi

# Expected output: 4x RTX 6000 Ada, 48GB each

# Check CUDA version
nvcc --version

# Check Ubuntu version
lsb_release -a

# Check available disk space (need ~50GB for models + overhead)
df -h
```

---

## STEP 1: System Dependencies Installation

### 1.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install NVIDIA Drivers (if not present)
```bash
# Check current driver
nvidia-smi

# If needed, install latest
sudo apt install nvidia-driver-535 nvidia-utils-535 -y
sudo reboot
```

### 1.3 Install Essential Tools
```bash
sudo apt install -y \\
  curl \\
  wget \\
  git \\
  build-essential \\
  python3-pip \\
  python3-venv \\
  htop \\
  tmux \\
  jq
```

### 1.4 Install Docker (Optional - for containerized bots)
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

**Validation Gate 1:** ✓ System dependencies installed

---

## STEP 2: Install Ollama

### 2.1 Download and Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2.2 Verify Installation
```bash
ollama --version
systemctl status ollama
```

### 2.3 Configure Ollama for Multi-GPU
```bash
# Check Ollama service file
sudo systemctl cat ollama

# If needed, edit to ensure no GPU restrictions
sudo systemctl edit ollama
```

Add environment overrides if needed:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
```

Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Validation Gate 2:** ✓ Ollama installed and verified

---

## STEP 3: Verify GPU Detection

### 3.1 Test Ollama GPU Recognition
```bash
# Check Ollama sees all GPUs
ollama ps

# You should see GPU resources listed
```

### 3.2 Manual CUDA Test
```bash
# Test each GPU individually
CUDA_VISIBLE_DEVICES=0 nvidia-smi
CUDA_VISIBLE_DEVICES=1 nvidia-smi
CUDA_VISIBLE_DEVICES=2 nvidia-smi
CUDA_VISIBLE_DEVICES=3 nvidia-smi
```

**Validation Gate 3:** ✓ GPU detection confirmed (4 GPUs visible)

---

## STEP 4: Download Granite4:Micro-H Model

### 4.1 Pull Model with Q4 Quantization
```bash
# Pull the instruction-tuned micro-h variant with Q4 quant
ollama pull granite4:micro-h

# Verify download
ollama list | grep granite
```

Expected output:
```
granite4:micro-h    latest    [size]    [date]
```

### 4.2 Check Model Size
```bash
ls -lh ~/.ollama/models/
```

**Validation Gate 4:** ✓ Model downloaded (~2-3GB for Q4)

---

## STEP 5: Single Bot Test

### 5.1 Test Single Instance
```bash
# Run single bot test
ollama run granite4:micro-h "What is 2+2? Answer concisely."
```

### 5.2 Test with CUDA Device Assignment
```bash
# Test on GPU 0
CUDA_VISIBLE_DEVICES=0 ollama run granite4:micro-h "Explain quantum computing in one sentence."

# Test on GPU 3
CUDA_VISIBLE_DEVICES=3 ollama run granite4:micro-h "What is machine learning?"
```

### 5.3 Monitor VRAM Usage
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
```

Expected VRAM per instance: ~2-4GB

**Validation Gate 5:** ✓ Single bot test passed on all GPUs

---

## STEP 6: Multi-Bot Architecture Setup

### 6.1 Create Project Directory Structure
```bash
mkdir -p ~/granite-swarm/{scripts,logs,configs,bots}
cd ~/granite-swarm
```

### 6.2 Architecture Overview
```
├── configs/
│   ├── swarm_config.yaml
│   └── bot_template.yaml
├── scripts/
│   ├── launch_swarm.py
│   ├── bot_worker.py
│   ├── health_monitor.py
│   └── stop_swarm.sh
├── logs/
│   ├── gpu0/
│   ├── gpu1/
│   ├── gpu2/
│   └── gpu3/
└── bots/
    └── [runtime state files]
```

---

## STEP 7: Configuration Files

All configuration files are provided in the next sections.

**Validation Gate 6:** Setup complete, ready for swarm launch

---

## STEP 8: Launch 100-Bot Swarm

### 8.1 Start Swarm Manager
```bash
cd ~/granite-swarm
python3 scripts/launch_swarm.py
```

### 8.2 Monitor Startup
```bash
# Watch logs
tail -f logs/swarm_manager.log

# Watch GPU usage
watch -n 2 nvidia-smi
```

### 8.3 Verify All Bots Running
```bash
python3 scripts/health_monitor.py --check-all
```

**Validation Gate 7:** ✓ Full swarm operational (100 bots)

---

## STEP 9: Test Swarm Functionality

### 9.1 Simple Distributed Query Test
```bash
# Send test query to swarm
python3 scripts/test_swarm.py --query "What is AI?" --bots 10
```

### 9.2 Stress Test
```bash
# Concurrent stress test
python3 scripts/stress_test.py --duration 300 --concurrent 50
```

### 9.3 Monitor Consensus
```bash
# Watch swarm consensus in real-time
python3 scripts/monitor_consensus.py
```

---

## STEP 10: Shutdown and Cleanup

### 10.1 Graceful Shutdown
```bash
bash scripts/stop_swarm.sh
```

### 10.2 Force Kill (if needed)
```bash
pkill -f "ollama.*granite4"
```

### 10.3 Clean Logs
```bash
rm -rf logs/*/*.log
```

---

## Troubleshooting

### Issue: VRAM Overflow
**Solution:** Reduce bots per GPU or use smaller context windows

```bash
# Check current VRAM
nvidia-smi

# Reduce to 20 bots per GPU
# Edit configs/swarm_config.yaml: bots_per_gpu: 20
```

### Issue: Bot Won't Start
**Solution:** Check logs and port conflicts

```bash
# Check specific bot log
tail -f logs/gpu0/bot_00.log

# Check port availability
netstat -tulpn | grep 11
```

### Issue: Slow Inference
**Solution:** Verify Q4 quantization and reduce context

```bash
# Confirm quant level
ollama show granite4:micro-h | grep quant

# Reduce context in bot_template.yaml
```

---

## Performance Tuning

### Optimize for Maximum Concurrency
```yaml
# In swarm_config.yaml
performance:
  max_context_length: 2048  # Reduce from default
  batch_size: 1
  num_threads: 4
  gpu_memory_fraction: 0.95
```

### Monitor Performance Metrics
```bash
python3 scripts/performance_stats.py --interval 60
```

---

## Next Steps

1. **Implement Gossip Protocol:** Add bot-to-bot communication layer
2. **Add Vector Storage:** Integrate per-bot vector memory (512-float embeddings)
3. **Consensus Logic:** Implement 4-hop flooding with confidence scoring
4. **Gatekeeper Service:** Deploy threshold-based consensus aggregator
5. **API Layer:** Expose swarm via REST API for external queries

---

## File Manifest

All required files are generated in subsequent sections:
- swarm_config.yaml
- bot_template.yaml
- launch_swarm.py
- bot_worker.py
- health_monitor.py
- stop_swarm.sh
- test_swarm.py
- stress_test.py
- monitor_consensus.py
- performance_stats.py

"""

print("Setup guide created successfully")
print("\n" + "="*60)
print("SETUP GUIDE PREVIEW")
print("="*60)
print(setup_guide[:2000] + "\n...[truncated]...")
