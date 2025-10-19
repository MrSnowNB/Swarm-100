#!/usr/bin/env python3
"""
---
script: zombie_supervisor.py
purpose: Self-healing Zombie Protocol supervisor for Gemma3 swarm recovery
status: development
created: 2025-10-19
---
"""

import yaml
import subprocess
import time
import requests
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import Dict, List, Any, Optional, cast

class ZombieSupervisor:
    def __init__(self, config_path: str = "configs/gemma3-zombie-swarm.yaml"):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError("Config file is empty")
        if not isinstance(config_data, dict):
            raise ValueError("Config must be a dict")
        self.config: Dict[str, Any] = config_data

        self.swarm_state: Optional[Dict[str, Any]] = None
        self.setup_logging()
        self.load_swarm_state()
        self.dead_bots = {}
        self.metrics = {
            'zombies_reborn': 0,
            'zombies_failed': 0,
            'recovery_times': []
        }

    def setup_logging(self):
        """Configure zombie supervisor logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='[ZombieSupervisor] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['modules'][1]['logging']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ZombieSupervisor')

    def load_swarm_state(self):
        """Load current swarm state"""
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                state_data = yaml.safe_load(f)
            if state_data is not None and isinstance(state_data, dict):
                self.swarm_state = cast(Dict[str, Any], state_data)
            else:
                self.swarm_state = None
        except FileNotFoundError:
            self.logger.error("No swarm state found - is swarm running?")
            self.swarm_state = None

    def get_neighbors(self, dead_bot_id: str) -> List[str]:
        """Find K nearest neighbors for a dead bot using spatial proximity"""
        if not self.swarm_state:
            return []

        # Parse dead bot coordinates
        gpu_id = int(dead_bot_id.split('_')[1])
        bot_idx = int(dead_bot_id.split('_')[2])

        k = self.config['modules'][1]['reconstruction']['parameters']['k']
        neighbors = []

        # Simple spatial neighbor finding (can be made more sophisticated)
        # Current: left/right on same GPU, then same index on adjacent GPUs
        candidates = []

        # Same GPU neighbors
        for idx in range(max(0, bot_idx-1), min(25, bot_idx+2)):
            if idx != bot_idx:
                candidates.append(f"bot_{gpu_id:02d}_{idx:02d}")

        # Adjacent GPU neighbors
        for g in [gpu_id-1, gpu_id+1]:
            if 0 <= g <= 3:
                candidates.append(f"bot_{g:02d}_{bot_idx:02d}")

        # Filter to existing running bots
        alive_candidates = []
        for candidate in candidates[:k*2]:  # Get more than needed
            if candidate in [b['bot_id'] for b in self.swarm_state.get('bots', [])]:
                alive_candidates.append(candidate)

        return alive_candidates[:k]

    def query_neighbor_state(self, neighbor_bot_id: str) -> Optional[np.ndarray]:
        """Query a neighbor bot for its current state vectors"""
        if self.swarm_state is None:
            return None
        # Find neighbor's port
        for bot in self.swarm_state.get('bots', []):
            if bot['bot_id'] == neighbor_bot_id:
                port = bot['port']
                break
        else:
            self.logger.warning(f"Port not found for {neighbor_bot_id}")
            return None

        try:
            response = requests.get(f"http://localhost:{port}/state",
                                  timeout=5)
            if response.status_code == 200:
                state_data = response.json()
                # Convert to numpy array (assuming vector list)
                return np.array(state_data['vectors'])
            else:
                self.logger.warning(f"Bad response from {neighbor_bot_id}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to query {neighbor_bot_id}: {e}")
            return None

    def calculate_averaged_state(self, neighbor_states: List[np.ndarray]) -> np.ndarray:
        """Calculate time-decayed average of neighbor states"""
        if not neighbor_states:
            return np.zeros(self.config['modules'][0]['memory']['embeddings_vector_dim'])

        # Simple mean for now (can add time decay later)
        return np.mean(neighbor_states, axis=0)

    def trigger_ca_update_on_rebirth(self):
        """Trigger a CA rule update after zombie rebirth for grid stabilization"""
        try:
            result = subprocess.run([
                sys.executable, 'scripts/rule_engine.py'
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info("✅ CA rules updated after zombie rebirth")
            else:
                self.logger.warning(f"CA update failed: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Error triggering CA update: {e}")

    def reconstruct_bot(self, bot_id: str, averaged_state: np.ndarray) -> bool:
        """Reconstruct a dead bot with averaged neighbor state"""
        if self.swarm_state is None:
            self.logger.error(f"Cannot reconstruct {bot_id}: swarm state not available")
            return False

        # Find bot details from swarm state
        for bot in self.swarm_state.get('bots', []):
            if bot['bot_id'] == bot_id:
                gpu_id = bot['gpu_id']
                port = bot['port']
                break
        else:
            self.logger.error(f"No details found for {bot_id}")
            return False

        # Convert averaged state to string for env var
        state_str = ','.join(map(str, averaged_state.tolist()))

        # Launch bot with reconstructed state
        env = {
            'CUDA_VISIBLE_DEVICES': str(gpu_id),
            'BOT_ID': bot_id,
            'GPU_ID': str(gpu_id),
            'BOT_PORT': str(port),
            'RECONSTRUCTED_STATE': state_str,
            'REBIRTH_TIME': datetime.now().isoformat()
        }

        try:
            subprocess.Popen(
                ['python3', 'scripts/bot_worker_zombie.py'],
                env={**os.environ.copy(), **env}
            )

            self.logger.info(f"🧟 Reconstructed zombie {bot_id} with averaged state")

            # Trigger CA rule update after resurrection for grid stabilization
            self.trigger_ca_update_on_rebirth()

            return True

        except Exception as e:
            self.logger.error(f"Failed to reconstruct {bot_id}: {e}")
            return False

    def detect_dead_bots(self) -> List[Dict[str, Any]]:
        """Check ps for dead bots with grace period"""
        if not self.swarm_state:
            return []

        grace_period = self.config['modules'][1]['detection']['grace_period_s']
        now = datetime.now()
        dead = []

        for bot in self.swarm_state.get('bots', []):
            try:
                # Check if PID exists
                subprocess.run(['ps', '-p', str(bot['pid'])],
                             check=True, capture_output=True)

                # Bot is alive, remove from dead list if present
                if bot['bot_id'] in self.dead_bots:
                    del self.dead_bots[bot['bot_id']]

            except subprocess.CalledProcessError:
                # Bot is dead or not found
                if bot['bot_id'] not in self.dead_bots:
                    # First detection
                    self.dead_bots[bot['bot_id']] = now
                    self.logger.warning(f"🚨 Detected dead bot: {bot['bot_id']}")
                else:
                    # Check if grace period expired
                    death_time = self.dead_bots[bot['bot_id']]
                    if (now - death_time).total_seconds() > grace_period:
                        dead.append({
                            'bot_id': bot['bot_id'],
                            'death_time': death_time,
                            'gpu_id': bot['gpu_id'],
                            'port': bot['port']
                        })

        return dead

    def run_recovery_cycle(self):
        """One cycle of dead bot detection and recovery"""
        start_time = time.time()

        # Find bots eligible for resurrection
        dead_bots = self.detect_dead_bots()

        recoveries_initiated = 0

        for dead_bot in dead_bots:
            bot_id = dead_bot['bot_id']
            death_duration = (datetime.now() - dead_bot['death_time']).total_seconds()

            if death_duration < 300:  # Hot recovery
                self.logger.info(f"🌡️ Hot recovery for {bot_id} (dead {death_duration:.1f}s)")

                # Get neighbors and their states
                neighbors = self.get_neighbors(bot_id)
                if len(neighbors) < 1:
                    self.logger.warning(f"Not enough neighbors for {bot_id}")
                    continue

                # Query neighbor states
                neighbor_states = []
                for neighbor in neighbors:
                    state = self.query_neighbor_state(neighbor)
                    if state is not None:
                        neighbor_states.append(state)

                if len(neighbor_states) < 1:
                    self.logger.warning(f"Failed to get any neighbor states for {bot_id}")
                    continue

                # Calculate averaged state
                averaged_state = self.calculate_averaged_state(neighbor_states)

                # Reconstruct bot
                if self.reconstruct_bot(bot_id, averaged_state):
                    recoveries_initiated += 1
                    self.metrics['zombies_reborn'] += 1
                    recovery_time = time.time() - start_time
                    self.metrics['recovery_times'].append(recovery_time)
                else:
                    self.metrics['zombies_failed'] += 1

            else:  # Cold recovery for long-dead bots
                self.logger.info(f"❄️ Cold restart for {bot_id} (dead {death_duration:.1f}s)")
                # TODO: Implement cold restart without state

        return recoveries_initiated

    def log_metrics(self):
        """Log current zombie metrics"""
        self.logger.info("📊 Zombie Metrics:")
        self.logger.info(f"  Zombies reborn: {self.metrics['zombies_reborn']}")
        self.logger.info(f"  Recovery failures: {self.metrics['zombies_failed']}")
        if self.metrics['recovery_times']:
            avg_recovery = np.mean(self.metrics['recovery_times'])
            self.logger.info(f"  Mean recovery time: {avg_recovery:.2f}s")

    def run(self):
        """Main supervisor loop"""
        check_interval = self.config['process_supervision']['check_interval_s']
        max_concurrent = self.config['process_supervision']['recovery_threshold']['max_concurrent_restarts']

        self.logger.info("🧟 Zombie Supervisor started - monitoring for dead bots")

        while True:
            try:
                recoveries = self.run_recovery_cycle()
                if recoveries > 0:
                    self.log_metrics()

                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.logger.info("Shutdown signal received")
                break
            except Exception as e:
                self.logger.error(f"Supervisor error: {e}")
                time.sleep(10)  # Back off on errors

if __name__ == '__main__':
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/gemma3-zombie-swarm.yaml',
                       help='Path to zombie config YAML')
    args = parser.parse_args()

    supervisor = ZombieSupervisor(args.config)
    supervisor.run()
