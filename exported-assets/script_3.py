
# Generate bot worker script
bot_worker = """#!/usr/bin/env python3
\"\"\"
---
script: bot_worker.py
purpose: Individual bot worker for Granite4:micro-h swarm
status: production-ready
created: 2025-10-18
---
\"\"\"

import argparse
import requests
import time
import json
import os
import logging
from datetime import datetime

class BotWorker:
    def __init__(self, bot_id, gpu_id, port):
        self.bot_id = bot_id
        self.gpu_id = gpu_id
        self.port = port
        self.ollama_url = "http://localhost:11434"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Bot-{bot_id}] %(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(f'Bot-{bot_id}')
        
        self.memory = []
        self.stats = {
            'requests': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
    def query_ollama(self, prompt):
        \"\"\"Query Ollama with prompt\"\"\"
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "granite4:micro-h",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.stats['requests'] += 1
                return response.json()['response']
            else:
                self.stats['errors'] += 1
                self.logger.error(f"Ollama error: {response.status_code}")
                return None
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Query failed: {e}")
            return None
            
    def health_check(self):
        \"\"\"Perform health check\"\"\"
        result = self.query_ollama("ping")
        if result:
            self.logger.info(f"✓ Health check passed - Stats: {self.stats}")
            return True
        else:
            self.logger.error("✗ Health check failed")
            return False
            
    def run(self):
        \"\"\"Main bot loop\"\"\"
        self.logger.info(f"Bot starting on GPU {self.gpu_id}, Port {self.port}")
        self.logger.info(f"Connecting to Ollama at {self.ollama_url}")
        
        # Initial health check
        time.sleep(2)
        if not self.health_check():
            self.logger.error("Initial health check failed, exiting")
            return
            
        self.logger.info("Bot operational, entering main loop")
        
        # Main event loop
        try:
            while True:
                time.sleep(60)  # Periodic health check
                self.health_check()
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    
    args = parser.parse_args()
    
    bot = BotWorker(args.bot_id, args.gpu, args.port)
    bot.run()
"""

# Generate health monitor script
health_monitor = """#!/usr/bin/env python3
\"\"\"
---
script: health_monitor.py
purpose: Monitor swarm health and performance
status: production-ready
created: 2025-10-18
---
\"\"\"

import yaml
import subprocess
import sys
import argparse
from datetime import datetime

class HealthMonitor:
    def __init__(self):
        self.load_swarm_state()
        
    def load_swarm_state(self):
        \"\"\"Load current swarm state\"\"\"
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                self.state = yaml.safe_load(f)
        except FileNotFoundError:
            print("✗ Swarm state file not found. Is the swarm running?")
            sys.exit(1)
            
    def check_processes(self):
        \"\"\"Check which bot processes are alive\"\"\"
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
        \"\"\"Get GPU memory usage\"\"\"
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\\n'):
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
        \"\"\"Print comprehensive status\"\"\"
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
            print(f"\\n⚠ Dead bots ({len(dead)}):")
            for bot in dead[:10]:  # Show first 10
                print(f"  - {bot['bot_id']} (PID {bot['pid']}, GPU {bot['gpu_id']})")
        else:
            print("✓ All bots operational")
            
        # GPU status
        print("\\nGPU Status:")
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
        \"\"\"Execute monitoring\"\"\"
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
"""

print("bot_worker.py generated")
print("="*60)
print(bot_worker[:1500] + "\\n...[script continues]...\\n")
print("\\nhealth_monitor.py generated")
print("="*60)
print(health_monitor[:1500] + "\\n...[script continues]...")
