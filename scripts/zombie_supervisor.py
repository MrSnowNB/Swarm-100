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
import socketio

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

        # Socket.IO client for dashboard communication
        self.sio = socketio.Client()
        self.setup_socketio_handlers()
        self.connect_to_dashboard()

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
        """Find K nearest neighbors for a dead bot using CA grid adjacency"""
        if not self.swarm_state:
            return []

        k = self.config['modules'][1]['reconstruction']['parameters']['k']
        grid_width = self.swarm_state.get('grid_width', 10)
        neighbors = []

        # Find dead bot's grid coordinates
        dead_x, dead_y = None, None
        for bot in self.swarm_state.get('bots', []):
            if bot['bot_id'] == dead_bot_id:
                dead_x = bot.get('grid_x')
                dead_y = bot.get('grid_y')
                break

        if dead_x is None or dead_y is None:
            self.logger.warning(f"Grid coordinates not found for {dead_bot_id}")
            return []

        # CA grid adjacency: Von Neumann neighborhood (4 cardinal directions)
        directions = [
            (0, -1),   # North
            (0, 1),    # South
            (-1, 0),   # West
            (1, 0)     # East
        ]

        for dx, dy in directions:
            nx = (dead_x + dx) % grid_width  # Toroidal wrap around
            ny = dead_y + dy
            # For height, prevent wrap-around (assume flat grid top/bottom)
            if 0 <= ny < grid_width:  # Assuming square grid for simplicity
                # Find bot at this position
                for bot in self.swarm_state['bots']:
                    if bot.get('grid_x') == nx and bot.get('grid_y') == ny:
                        neighbors.append(bot['bot_id'])
                        break

        # If we don't have enough grid neighbors, supplement with nearby bots
        if len(neighbors) < k:
            self.logger.info(f"Found {len(neighbors)} grid neighbors, supplementing for {dead_bot_id}")
            # Supplement with closest running bots by Euclidean distance
            distances = []
            for bot in self.swarm_state.get('bots', []):
                if bot['bot_id'] == dead_bot_id:
                    continue
                bx = bot.get('grid_x', 0)
                by = bot.get('grid_y', 0)
                dist = ((bx - dead_x)**2 + (by - dead_y)**2)**0.5
                distances.append((dist, bot['bot_id']))

            distances.sort()
            for _, bot_id in distances[:k - len(neighbors)]:
                if bot_id not in neighbors:
                    neighbors.append(bot_id)

        self.logger.debug(f"Found {len(neighbors)} neighbors for {dead_bot_id}: {neighbors[:k]}")
        return neighbors[:k]

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

            # Emit event to dashboard
            self.emit_zombie_event('reborn', f'Bot {bot_id} resurrected with averaged neighbor state')

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

    def setup_socketio_handlers(self):
        """Configure Socket.IO event handlers"""
        @self.sio.event
        def connect():
            self.logger.info("Connected to dashboard server")

        @self.sio.event
        def disconnect():
            self.logger.info("Disconnected from dashboard server")

    def connect_to_dashboard(self):
        """Establish connection to dashboard Socket.IO server"""
        try:
            self.sio.connect('http://localhost:5000')
            self.logger.info("Connected to dashboard for event broadcasting")
        except Exception as e:
            self.logger.warning(f"Could not connect to dashboard: {e}")

    def emit_zombie_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Emit zombie event to dashboard"""
        if self.sio.connected:
            self.sio.emit('zombie_event', {
                'type': event_type,
                'message': message,
                'details': details,
                'timestamp': time.time()
            })
        else:
            self.logger.debug(f"Dashboard not connected - zombie event: {event_type}: {message}")

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
