#!/usr/bin/env python3
"""
---
script: collect_ca_metrics.py
purpose: Collect cellular automata metrics during experimentation
status: development
created: 2025-10-19
---
"""

import numpy as np
import yaml
import csv
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import threading

logger = logging.getLogger('CAMetricsCollector')

class CAMetricsCollector:
    """
    Collects CA evolution metrics during stability testing (G5) and emergent behavior analysis (G6).

    Metrics collected:
    - State entropy (distribution diversity)
    - Neighbor similarity index
    - Zombie recovery events and wave propagation
    - State vector variance
    - Bot survival statistics
    - GPU utilization
    - Tick timing latency
    """

    def __init__(self, output_csv: str = "logs/experimentation/ca_metrics.csv"):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Zombie metrics tracking
        self.zombie_events = []
        self._load_existing_zombie_events()

        # CSV header
        self.csv_headers = [
            'tick_id', 'timestamp', 'mean_state_entropy', 'neighbor_similarity_index',
            'zombie_recovery_rate', 'gpu_utilization_percent', 'mean_state_magnitude',
            'state_variance', 'tick_latency_ms', 'active_bot_count'
        ]

        # Initialize CSV if not exists
        if not self.output_csv.exists():
            with open(self.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def _load_existing_zombie_events(self):
        """Load any existing zombie recovery stats"""
        zombie_yaml = Path('logs/experimentation/zombie_recovery_stats.yaml')
        if zombie_yaml.exists():
            try:
                with open(zombie_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and data and 'events' in data:
                        self.zombie_events = data['events']
            except Exception as e:
                logger.warning(f"Could not load zombie events: {e}")

    def load_swarm_state(self) -> Dict[str, Any] | None:
        """Load current swarm state"""
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None

    def calculate_state_entropy(self, swarm_state: Dict) -> float | None:
        """Calculate entropy of all bot state distributions"""
        if not swarm_state or 'bots' not in swarm_state:
            return None

        all_states = []
        for bot in swarm_state['bots']:
            if 'state_vector' in bot:
                all_states.extend(bot['state_vector'])

        if not all_states:
            return 0.0

        all_states = np.array(all_states)

        # Calculate histogram entropy
        hist, _ = np.histogram(all_states, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist))

        return float(entropy)

    def calculate_neighbor_similarity(self, swarm_state: Dict) -> float | None:
        """Calculate average similarity between neighbors"""
        if not swarm_state or 'bots' not in swarm_state:
            return None

        similarities = []

        # Group bots by position for easier neighbor lookup
        bots_by_pos = {}
        for bot in swarm_state['bots']:
            if 'grid_x' in bot and 'grid_y' in bot:
                pos = (bot['grid_x'], bot['grid_y'])
                bots_by_pos[pos] = bot

        for bot in swarm_state['bots']:
            if 'state_vector' not in bot or 'grid_x' not in bot or 'grid_y' not in bot:
                continue

            bot_state = np.array(bot['state_vector'])
            neighbors = self._get_neighbors(bot['grid_x'], bot['grid_y'], swarm_state)

            for neighbor_pos in neighbors:
                if neighbor_pos in bots_by_pos:
                    neighbor_bot = bots_by_pos[neighbor_pos]
                    if 'state_vector' in neighbor_bot:
                        neighbor_state = np.array(neighbor_bot['state_vector'])
                        # Cosine similarity for high-dimensional vectors
                        similarity = np.dot(bot_state, neighbor_state) / (
                            np.linalg.norm(bot_state) * np.linalg.norm(neighbor_state)
                        )
                        similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else None

    def _get_neighbors(self, grid_x: int, grid_y: int, swarm_state: Dict) -> List[Tuple[int, int]]:
        """Get neighbor positions (4-way)"""
        neighbors = []
        grid_width = swarm_state.get('grid_width', 10)
        grid_height = swarm_state.get('grid_height', 4)

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E

        for dx, dy in directions:
            nx = (grid_x + dx) % grid_width
            ny = (grid_y + dy) % grid_height
            neighbors.append((nx, ny))

        return neighbors

    def calculate_state_variance(self, swarm_state: Dict) -> float | None:
        """Calculate variance of state vector components"""
        if not swarm_state or 'bots' not in swarm_state:
            return None

        all_states = []
        for bot in swarm_state['bots']:
            if 'state_vector' in bot:
                all_states.append(bot['state_vector'])

        if not all_states:
            return 0.0

        all_states = np.array(all_states)
        return float(np.var(all_states))

    def calculate_mean_state_magnitude(self, swarm_state: Dict) -> float | None:
        """Calculate average magnitude of state vectors"""
        if not swarm_state or 'bots' not in swarm_state:
            return None

        magnitudes = []
        for bot in swarm_state['bots']:
            if 'state_vector' in bot:
                state = np.array(bot['state_vector'])
                magnitudes.append(np.linalg.norm(state))

        return float(np.mean(magnitudes)) if magnitudes else None

    def calculate_zombie_recovery_rate(self, tick: int) -> float:
        """Calculate recovery rate in recent window"""
        if not self.zombie_events:
            return 0.0

        # Count recoveries in last 50 ticks
        recent_recoveries = 0
        total_failures = 0

        for event in self.zombie_events:
            event_tick = event.get('tick', 0)
            if tick - 50 <= event_tick <= tick:
                if event.get('type') == 'reborn':
                    recent_recoveries += 1
                elif event.get('type') == 'failed':
                    total_failures += 1

        denominator = recent_recoveries + total_failures
        return recent_recoveries / max(denominator, 1)

    def get_gpu_utilization(self) -> float | None:
        """Get GPU utilization (if nvidia-smi available)"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Average across GPUs if multiple
                utilizations = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                return float(np.mean(utilizations)) if utilizations else 0.0
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None

    def collect_metrics(self, tick_id: int, tick_latency_ms: float = 0.0) -> Dict[str, Any]:
        """Collect all metrics for current tick"""
        start_time = time.time()
        swarm_state = self.load_swarm_state()

        if not swarm_state:
            logger.warning("Could not load swarm state")
            return {}

        metrics = {
            'tick_id': tick_id,
            'timestamp': time.time(),
            'mean_state_entropy': self.calculate_state_entropy(swarm_state),
            'neighbor_similarity_index': self.calculate_neighbor_similarity(swarm_state),
            'zombie_recovery_rate': self.calculate_zombie_recovery_rate(tick_id),
            'gpu_utilization_percent': self.get_gpu_utilization(),
            'mean_state_magnitude': self.calculate_mean_state_magnitude(swarm_state),
            'state_variance': self.calculate_state_variance(swarm_state),
            'tick_latency_ms': tick_latency_ms,
            'active_bot_count': len([b for b in swarm_state.get('bots', []) if b.get('alive', True)])
        }

        # Replace None values with 0 for CSV compatibility
        for key, value in metrics.items():
            if value is None:
                metrics[key] = 0.0

        collection_time = time.time() - start_time
        logger.debug(".2f")

        return metrics

    def save_metrics(self, metrics: Dict[str, Any]):
        """Append metrics to CSV file"""
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [metrics.get(header, 0.0) for header in self.csv_headers]
            writer.writerow(row)

    def update_zombie_events(self):
        """Check for new zombie events in logs"""
        zombie_yaml = Path('logs/experimentation/zombie_recovery_stats.yaml')
        if zombie_yaml.exists():
            try:
                with open(zombie_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and data and 'events' in data:
                        # Only add new events
                        new_events = [e for e in data['events'] if e not in self.zombie_events]
                        self.zombie_events.extend(new_events)
            except Exception as e:
                logger.debug(f"Could not update zombie events: {e}")

class BackgroundCollector:
    """Background metrics collector for long-running experiments"""

    def __init__(self, tick_interval: float = 1.0):
        self.collector = CAMetricsCollector()
        self.tick_interval = tick_interval
        self.running = False
        self.thread = None
        self.last_tick = 0
        self.start_time = time.time()

    def start_collecting(self):
        """Start background collection"""
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info(".1f")

    def stop_collecting(self):
        """Stop background collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Background metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Get current tick from swarm state
                swarm_state = self.collector.load_swarm_state()
                current_tick = swarm_state.get('tick', 0) if swarm_state else 0

                if current_tick > self.last_tick:
                    # Calculate latency (how long since last collection)
                    latency = 0.0  # Default
                    if self.last_tick > 0:
                        latency = (self.tick_interval - (time.time() - self.start_time) % self.tick_interval) * 1000

                    # Collect metrics
                    metrics = self.collector.collect_metrics(current_tick, latency)
                    if metrics:
                        self.collector.save_metrics(metrics)
                        self.collector.update_zombie_events()

                    self.last_tick = current_tick

                time.sleep(self.tick_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(1)

def main():
    """Standalone metrics collection for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='CA Metrics Collector')
    parser.add_argument('--output', default='logs/experimentation/ca_metrics.csv',
                       help='Output CSV file')
    parser.add_argument('--background', action='store_true',
                       help='Run in background during experiment')
    parser.add_argument('--single', action='store_true',
                       help='Collect single metrics snapshot')
    parser.add_argument('--tick-id', type=int, default=0,
                       help='Tick ID for single collection')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                       format='%(asctime)s - CAMetrics - %(levelname)s - %(message)s')

    collector = CAMetricsCollector(args.output)

    if args.background:
        bg_collector = BackgroundCollector()
        try:
            bg_collector.start_collecting()
            print("Background metrics collection started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bg_collector.stop_collecting()

    elif args.single:
        metrics = collector.collect_metrics(args.tick_id)
        if metrics:
            collector.save_metrics(metrics)
            print(f"Metrics collected for tick {args.tick_id}:")
            for key, value in metrics.items():
                print(".4f")
        else:
            print("Could not collect metrics")
    else:
        # Interactive mode
        collector = CAMetricsCollector(args.output)
        metrics = collector.collect_metrics(0)
        if metrics:
            print("Current metrics:")
            for key, value in metrics.items():
                print(".4f")
        else:
            print("No swarm state available")

if __name__ == '__main__':
    main()
