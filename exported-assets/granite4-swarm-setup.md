# Granite4:Micro-H 100-Bot Swarm Setup Guide

---
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

## Quick Start (5 Minutes)

```bash
# 1. Clone/create project directory
mkdir ~/granite-swarm && cd ~/granite-swarm

# 2. Run automated setup
curl -O https://raw.githubusercontent.com/your-repo/granite-swarm/main/quickstart.sh
chmod +x quickstart.sh
./quickstart.sh

# 3. Launch swarm
python3 scripts/launch_swarm.py

# 4. Monitor
python3 scripts/health_monitor.py
```

---

## System Requirements

**Hardware:**
- 4x NVIDIA RTX 6000 Ada Generation GPUs (48GB VRAM each)
- 64GB+ System RAM
- 100GB+ free disk space
- Ubuntu 22.04+ LTS

**Software:**
- NVIDIA drivers 535+
- CUDA 12.0+
- Python 3.10+
- Ollama latest

---

## Detailed Setup Instructions

### STEP 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y curl wget git build-essential python3-pip python3-venv htop tmux jq

# Verify GPUs
nvidia-smi
# Expected: 4x RTX 6000 Ada, 48GB each

# Install Python packages
pip3 install pyyaml requests
```

**Validation Gate 1:** ✓ System dependencies installed

### STEP 2: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
systemctl status ollama

# Pull Granite4:micro-h model
ollama pull granite4:micro-h

# Verify model
ollama list | grep granite4
```

**Validation Gate 2:** ✓ Ollama installed and model downloaded

### STEP 3: Project Setup

```bash
# Create directory structure
mkdir -p ~/granite-swarm/{scripts,logs/{gpu0,gpu1,gpu2,gpu3},configs,bots}
cd ~/granite-swarm

# Create all configuration files (see File Manifest below)
```

**Validation Gate 3:** ✓ Project structure created

### STEP 4: Configuration Files

Create the following files in your project directory:

#### configs/swarm_config.yaml
```yaml
---
project: granite4-microh-swarm
version: 1.0.0
status: active
created: 2025-10-18

# Hardware Configuration
hardware:
  gpus:
    - id: 0
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 1
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 2
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 3
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
  total_bots: 100

# Model Configuration
model:
  name: "granite4:micro-h"
  quantization: "Q4"
  context_length: 4096
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  
# Swarm Architecture
swarm:
  gossip_hops: 4
  fanout: 5
  confidence_threshold: 0.5
  ttl: 4
  
# Bot Configuration
bot:
  base_port: 11400
  api_timeout: 30
  max_retries: 3
  health_check_interval: 60
  
# Performance Tuning
performance:
  max_concurrent_requests: 10
  batch_size: 1
  num_threads: 4
  gpu_memory_fraction: 0.95
  enable_kv_cache: true
  
# Logging
logging:
  level: "INFO"
  dir: "logs"
  rotation: "daily"
  retention_days: 7
```

#### scripts/launch_swarm.py
```python
#!/usr/bin/env python3
"""
Swarm launcher - deploys 100 bots across 4 GPUs
"""
import subprocess
import yaml
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/swarm_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SwarmManager')

class SwarmManager:
    def __init__(self, config_path='configs/swarm_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.bots = []
        
    def check_prerequisites(self):
        """Verify system is ready"""
        # Check Ollama
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, check=True)
            if 'granite4' not in result.stdout:
                logger.error("granite4:micro-h not found. Run: ollama pull granite4:micro-h")
                return False
        except Exception:
            logger.error("Ollama not responding")
            return False
            
        # Check GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count'], 
                                  capture_output=True, text=True, check=True)
            if len(result.stdout.strip().split('\n')) < 4:
                logger.error("Need 4 GPUs, found fewer")
                return False
        except Exception:
            logger.error("nvidia-smi failed")
            return False
            
        return True
        
    def launch_bot(self, bot_id, gpu_id, port):
        """Launch single bot"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        log_file = open(f'logs/gpu{gpu_id}/bot_{bot_id:02d}.log', 'w')
        
        cmd = ['python3', 'scripts/bot_worker.py', 
               '--bot-id', f"{gpu_id}_{bot_id}",
               '--gpu', str(gpu_id),
               '--port', str(port)]
        
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        
        return {
            'bot_id': f"bot_{gpu_id:02d}_{bot_id:02d}",
            'gpu_id': gpu_id,
            'port': port,
            'pid': proc.pid,
            'process': proc,
            'log_file': log_file
        }
        
    def launch_swarm(self):
        """Launch all 100 bots"""
        logger.info("Starting swarm launch...")
        
        base_port = self.config['bot']['base_port']
        
        for gpu in self.config['hardware']['gpus']:
            gpu_id = gpu['id']
            num_bots = gpu['bots']
            
            logger.info(f"Launching {num_bots} bots on GPU {gpu_id}...")
            
            for bot_idx in range(num_bots):
                port = base_port + (gpu_id * 100) + bot_idx
                bot_info = self.launch_bot(bot_idx, gpu_id, port)
                self.bots.append(bot_info)
                
                time.sleep(0.5)  # Staggered launch
                
                if (bot_idx + 1) % 5 == 0:
                    logger.info(f"  {bot_idx + 1}/{num_bots} bots launched on GPU {gpu_id}")
        
        logger.info(f"✓ Swarm complete: {len(self.bots)} bots deployed")
        
    def save_state(self):
        """Save swarm state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'total_bots': len(self.bots),
            'bots': [
                {
                    'bot_id': b['bot_id'],
                    'gpu_id': b['gpu_id'],
                    'port': b['port'],
                    'pid': b['pid']
                }
                for b in self.bots
            ]
        }
        
        with open('bots/swarm_state.yaml', 'w') as f:
            yaml.dump(state, f)
            
    def monitor(self):
        """Monitor swarm health"""
        try:
            while True:
                time.sleep(60)
                alive = sum(1 for b in self.bots if b['process'].poll() is None)
                logger.info(f"Health: {alive}/{len(self.bots)} bots alive")
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            
    def run(self):
        """Main execution"""
        logger.info("="*60)
        logger.info("GRANITE4:MICRO-H SWARM MANAGER")
        logger.info("="*60)
        
        if not self.check_prerequisites():
            sys.exit(1)
            
        Path('logs/gpu0').mkdir(parents=True, exist_ok=True)
        Path('logs/gpu1').mkdir(parents=True, exist_ok=True)
        Path('logs/gpu2').mkdir(parents=True, exist_ok=True)
        Path('logs/gpu3').mkdir(parents=True, exist_ok=True)
        Path('bots').mkdir(exist_ok=True)
        
        self.launch_swarm()
        self.save_state()
        
        time.sleep(10)
        logger.info("Swarm operational. Monitoring...")
        self.monitor()

if __name__ == '__main__':
    manager = SwarmManager()
    manager.run()
```

### STEP 5: Launch Swarm

```bash
# Start the swarm
python3 scripts/launch_swarm.py

# Monitor in another terminal
python3 scripts/health_monitor.py

# Watch GPU usage
watch -n 2 nvidia-smi
```

**Validation Gate 4:** ✓ Full swarm operational (100 bots)

### STEP 6: Test Functionality

```bash
# Simple test
python3 scripts/test_swarm.py --query "What is 2+2?"

# Concurrent test
python3 scripts/test_swarm.py --bots 20

# Stress test
python3 scripts/stress_test.py --duration 300
```

**Validation Gate 5:** ✓ Swarm functionality verified

### STEP 7: Shutdown

```bash
# Graceful shutdown
bash scripts/stop_swarm.sh

# Force kill if needed
pkill -f "granite4"
```

---

## Expected Performance

- **Concurrent bots:** 100 (25 per GPU)
- **Memory per bot:** ~2-4GB (Q4 quantization)
- **Total VRAM usage:** ~150-180GB (of 192GB available)
- **Response latency:** 1-5 seconds per query
- **Throughput:** 20-50 concurrent queries/second

---

## Troubleshooting

### Issue: VRAM Overflow
**Symptoms:** Bots fail to start, CUDA out of memory
**Solution:** 
```bash
# Reduce bots per GPU in configs/swarm_config.yaml
# Change from 25 to 20 bots per GPU
vim configs/swarm_config.yaml
```

### Issue: Model Loading Slow
**Symptoms:** Long startup times
**Solution:**
```bash
# Pre-warm model
ollama run granite4:micro-h "test" 

# Check disk I/O
iostat -x 1
```

### Issue: Port Conflicts
**Symptoms:** Bot startup failures
**Solution:**
```bash
# Check port usage
netstat -tulpn | grep 114

# Change base port in config if needed
```

---

## Performance Optimization

### Memory Optimization
```yaml
# In swarm_config.yaml
model:
  context_length: 2048  # Reduce from 4096
performance:
  gpu_memory_fraction: 0.90  # Leave 10% headroom
```

### Throughput Optimization
```yaml
performance:
  max_concurrent_requests: 20  # Increase if stable
  batch_size: 2  # Test higher batch sizes
```

---

## Next Steps: Distributed Memory Implementation

Once the basic swarm is operational, implement distributed memory features:

1. **Gossip Protocol**: Add bot-to-bot communication
2. **Vector Storage**: Per-bot 512-float embeddings
3. **Consensus Logic**: 4-hop flooding with confidence scores
4. **Gatekeeper**: Threshold-based response aggregation

---

## File Manifest

**Required Files:**
- `configs/swarm_config.yaml` - Main configuration
- `configs/bot_template.yaml` - Individual bot template
- `scripts/launch_swarm.py` - Swarm launcher
- `scripts/bot_worker.py` - Individual bot worker
- `scripts/health_monitor.py` - Health monitoring
- `scripts/stop_swarm.sh` - Shutdown script
- `scripts/test_swarm.py` - Functionality tests
- `scripts/quickstart.sh` - Automated setup

**Generated Files:**
- `bots/swarm_state.yaml` - Runtime state
- `logs/gpu*/bot_*.log` - Individual bot logs
- `logs/swarm_manager.log` - Manager logs

---

## Support

For issues:
1. Check `logs/swarm_manager.log`
2. Monitor GPU usage with `nvidia-smi`
3. Verify model availability: `ollama list`
4. Test single instance: `ollama run granite4:micro-h "test"`

The swarm is now ready for distributed memory and gossip protocol implementation!