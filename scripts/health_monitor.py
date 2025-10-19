#!/usr/bin/env python3
"""
---
script: health_monitor.py
purpose: Monitor swarm health and performance
status: production-ready
created: 2025-10-18
---
"""

import yaml
import subprocess
import sys
import argparse
from datetime import datetime

class HealthMonitor:
    def __init__(self):
        self.load_swarm_state()

    def load_swarm_state(self):
        """Load current swarm state"""
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                self.state = yaml.safe_load(f)
        except FileNotFoundError:
            print("✗ Swarm state file not found. Is the swarm running?")
            sys.exit(1)

    def check_processes(self):
        """Check which bot processes are alive"""
        alive = []
        dead = []

        for bot in self.state['bots']:
            try:
                # Check if process exists
                subprocess.run(['ps', '-p', str(bot['pid'])],
                             check=True,
                             capture_output=True)
                alive.append(bot)
            except subprocess.CalledProcessError:
                dead.append(bot)

        return alive, dead

    def check_gpu_usage(self):
        """Get GPU memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )

            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                gpus.append({
                    'id': int(parts[0].strip()),
                    'mem_used': int(parts[1].strip()),
                    'mem_total': int(parts[2].strip()),
                    'utilization': int(parts[3].strip())
                })
            return gpus

        except Exception as e:
            print(f"✗ GPU check failed: {e}")
            return []

    def print_status(self):
        """Print comprehensive status"""
        print("="*70)
        print("GRANITE4:MICRO-H SWARM HEALTH MONITOR")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Swarm started: {self.state['timestamp']}")
        print()

        # Process status
        alive, dead = self.check_processes()
        print(f"Bot Processes: {len(alive)}/{self.state['total_bots']} alive")

        if dead:
            print(f"\n⚠ Dead bots ({len(dead)}):")
            for bot in dead[:10]:  # Show first 10
                print(f"  - {bot['bot_id']} (PID {bot['pid']}, GPU {bot['gpu_id']})")
        else:
            print("✓ All bots operational")

        # GPU status
        print("\nGPU Status:")
        print("-"*70)
        print(f"{'GPU':<5} {'Memory Used':<15} {'Memory Total':<15} {'Utilization':<12}")
        print("-"*70)

        gpus = self.check_gpu_usage()
        for gpu in gpus[:4]:
            mem_pct = (gpu['mem_used'] / gpu['mem_total']) * 100
            print(f"{gpu['id']:<5} {gpu['mem_used']:>5} MB ({mem_pct:>5.1f}%)  "
                  f"{gpu['mem_total']:>5} MB       {gpu['utilization']:>3}%")

        print("="*70)

    def run(self, check_all=False):
        """Execute monitoring"""
        self.print_status()

        if check_all:
            alive, dead = self.check_processes()
            sys.exit(0 if len(dead) == 0 else 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-all', action='store_true',
                       help='Check all bots and exit with status code')
    args = parser.parse_args()

    monitor = HealthMonitor()
    monitor.run(check_all=args.check_all)
