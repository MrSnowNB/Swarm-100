#!/usr/bin/env python3
"""
---
script: launch_swarm.py
purpose: Deploy swarm (Granite4:micro-h or Zombie CA bots) across 4 GPUs
status: production-ready
created: 2025-10-18
---
"""

import subprocess
import yaml
import time
import os
import sys
import socket
import argparse
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
    def __init__(self, config_path=None, zombie_active=False):
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'swarm_config.yaml'
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"Failed to load config from {config_path}")
        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a dictionary")
        self.config: Dict[str, Any] = config_data

        self.zombie_active = zombie_active
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
            models_available = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    models_available.append(line.split()[0])

            if self.zombie_active:
                required = ['gemma3:270m']
            else:
                required = ['granite4:micro-h']

            for model in required:
                if model not in models_available:
                    logger.error(f"✗ Required model {model} not found")
                    return False

            logger.info(f"✓ Ollama running, models available: {', '.join(required)}")
            return True

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

    def check_port_free(self, port):
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return True
        except OSError:
            return False

    def launch_bot(self, bot_id, gpu_id, port, grid_x, grid_y):
        """Launch a single bot instance"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['BOT_ID'] = f"bot_{gpu_id:02d}_{bot_id:02d}"
        env['GPU_ID'] = str(gpu_id)
        env['BOT_PORT'] = str(port)

        log_file = open(f'logs/gpu{gpu_id}/bot_{bot_id:02d}.log', 'w')

        # Choose bot script based on zombie_active flag
        if self.zombie_active:
            script_path = 'scripts/bot_worker_zombie.py'
            bot_type = "ZombieBot"
        else:
            script_path = 'scripts/bot_worker.py'
            bot_type = "Granite4"

        # Launch bot worker
        cmd = ['python3', script_path,
               '--bot-id', f"{gpu_id}_{bot_id}",
               '--gpu', str(gpu_id),
               '--port', str(port)]

        logger.debug(f"Launching {bot_type} bot: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)

        return {
            'bot_id': f"bot_{gpu_id:02d}_{bot_id:02d}",
            'gpu_id': gpu_id,
            'port': port,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'pid': proc.pid,
            'process': proc,
            'log_file': log_file
        }

    def launch_swarm(self):
        """Launch all bots across GPUs with CA grid mapping"""
        logger.info("Starting swarm launch sequence...")

        base_port = self.config['bot']['base_port']
        stagger_seconds = self.config['bot']['launch_stagger_seconds']
        grid_width = self.config['swarm']['grid_width']
        grid_height = self.config['swarm']['grid_height']

        global_bot_idx = 0

        for gpu in self.config['hardware']['gpus']:
            gpu_id = gpu['id']
            num_bots = gpu['bots']

            logger.info(f"\nLaunching {num_bots} bots on GPU {gpu_id}...")

            for bot_idx in range(num_bots):
                port = base_port + (gpu_id * 100) + bot_idx
                if not self.check_port_free(port):
                    logger.warning(f"Port {port} in use for bot_{gpu_id:02d}_{bot_idx:02d}, skipping deployment")
                    continue
                grid_x = global_bot_idx % grid_width
                grid_y = global_bot_idx // grid_width
                bot_info = self.launch_bot(bot_idx, gpu_id, port, grid_x, grid_y)
                self.bots.append(bot_info)

                global_bot_idx += 1

                # Staggered launch to avoid overwhelming the system and model loading spikes
                time.sleep(stagger_seconds)

                if (bot_idx + 1) % 5 == 0:
                    logger.info(f"  Launched {bot_idx + 1}/{num_bots} bots on GPU {gpu_id}")

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Swarm launch complete: {len(self.bots)} bots deployed")
        logger.info(f"✓ CA grid configured: {grid_width}x{grid_height}")
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
            'grid_width': self.config['swarm']['grid_width'],
            'grid_height': self.config['swarm']['grid_height'],
            'total_bots': len(self.bots),
            'bots': [
                {
                    'bot_id': b['bot_id'],
                    'gpu_id': b['gpu_id'],
                    'port': b['port'],
                    'grid_x': b['grid_x'],
                    'grid_y': b['grid_y'],
                    'pid': b['pid']
                }
                for b in self.bots
            ]
        }

        with open('bots/swarm_state.yaml', 'w') as f:
            yaml.dump(state, f)

        logger.info("Swarm state saved to bots/swarm_state.yaml with CA grid mapping")

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
    parser = argparse.ArgumentParser(description='Swarm Deployment Manager')
    parser.add_argument('--ca-active', action='store_true',
                       help='Launch CA-enabled ZombieBots with Gemma3:270M')
    parser.add_argument('--zombie-active', action='store_true',
                       help='Launch zombie recovery enabled bots (implies --ca-active)')

    args = parser.parse_args()

    # Zombie-active implies CA-active for backward compatibility
    zombie_active = args.zombie_active or args.ca_active

    manager = SwarmManager(zombie_active=zombie_active)
    manager.run()
