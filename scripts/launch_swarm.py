#!/usr/bin/env python3
"""
---
script: launch_swarm.py
purpose: Deploy 100-bot Granite4:micro-h swarm across 4 Ada6000 GPUs
status: production-ready
created: 2025-10-18
---
"""

import subprocess
import yaml
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

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
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'swarm_config.yaml'
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"Failed to load config from {config_path}")
        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a dictionary")
        self.config: Dict[str, Any] = config_data

        self.bots = []
        self.processes = []

    def create_directory_structure(self):
        """Create necessary directories"""
        dirs = ['logs/gpu0', 'logs/gpu1', 'logs/gpu2', 'logs/gpu3', 'bots']
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure created")

    def check_ollama(self):
        """Verify Ollama is running"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True,
                                  text=True,
                                  check=True)
            if 'granite4' in result.stdout:
                logger.info("✓ Ollama running, granite4:micro-h available")
                return True
            else:
                logger.error("✗ granite4:micro-h model not found")
                return False
        except Exception as e:
            logger.error(f"✗ Ollama check failed: {e}")
            return False

    def check_gpus(self):
        """Verify GPU availability"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                   '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True)
            gpus = result.stdout.strip().split('\n')
            if len(gpus) >= 4:
                logger.info(f"✓ Detected {len(gpus)} GPUs")
                for gpu in gpus[:4]:
                    logger.info(f"  {gpu}")
                return True
            else:
                logger.error(f"✗ Only {len(gpus)} GPUs detected, need 4")
                return False
        except Exception as e:
            logger.error(f"✗ GPU check failed: {e}")
            return False

    def launch_bot(self, bot_id, gpu_id, port):
        """Launch a single bot instance"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['BOT_ID'] = f"bot_{gpu_id:02d}_{bot_id:02d}"
        env['GPU_ID'] = str(gpu_id)
        env['BOT_PORT'] = str(port)

        log_file = open(f'logs/gpu{gpu_id}/bot_{bot_id:02d}.log', 'w')

        # Launch bot worker
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
        """Launch all 100 bots across 4 GPUs"""
        logger.info("Starting swarm launch sequence...")

        base_port = self.config['bot']['base_port']
        stagger_seconds = self.config['bot']['launch_stagger_seconds']

        for gpu in self.config['hardware']['gpus']:
            gpu_id = gpu['id']
            num_bots = gpu['bots']

            logger.info(f"\nLaunching {num_bots} bots on GPU {gpu_id}...")

            for bot_idx in range(num_bots):
                port = base_port + (gpu_id * 100) + bot_idx
                bot_info = self.launch_bot(bot_idx, gpu_id, port)
                self.bots.append(bot_info)

                # Staggered launch to avoid overwhelming the system and model loading spikes
                time.sleep(stagger_seconds)

                if (bot_idx + 1) % 5 == 0:
                    logger.info(f"  Launched {bot_idx + 1}/{num_bots} bots on GPU {gpu_id}")

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Swarm launch complete: {len(self.bots)} bots deployed")
        logger.info(f"{'='*60}")

    def monitor_health(self):
        """Check if all bots are still running"""
        alive = 0
        dead = []

        for bot in self.bots:
            if bot['process'].poll() is None:
                alive += 1
            else:
                dead.append(bot['bot_id'])

        logger.info(f"Health check: {alive}/{len(self.bots)} bots alive")

        if dead:
            logger.warning(f"Dead bots: {dead}")

        return alive, dead

    def save_swarm_state(self):
        """Save current swarm state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'startup_grace_period_seconds': self.config['bot']['startup_grace_period_seconds'],
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

        logger.info("Swarm state saved to bots/swarm_state.yaml")

    def run(self):
        """Main execution flow"""
        logger.info("="*60)
        logger.info("GRANITE4:MICRO-H SWARM MANAGER")
        logger.info("="*60)

        # Pre-flight checks
        self.create_directory_structure()

        if not self.check_ollama():
            logger.error("Ollama not ready. Run: ollama pull granite4:micro-h")
            sys.exit(1)

        if not self.check_gpus():
            logger.error("GPU configuration invalid")
            sys.exit(1)

        # Launch swarm
        self.launch_swarm()
        self.save_swarm_state()

        # Initial health check
        time.sleep(10)
        self.monitor_health()

        logger.info("\nSwarm is operational. Press Ctrl+C to monitor or stop.")
        logger.info("Run 'python3 scripts/health_monitor.py' for detailed monitoring")

        try:
            while True:
                time.sleep(60)
                self.monitor_health()
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received")

if __name__ == '__main__':
    manager = SwarmManager()
    manager.run()
