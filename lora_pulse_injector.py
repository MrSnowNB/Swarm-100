#!/usr/bin/env python3
"""
---
script: lora_pulse_injector.py
purpose: Inject LoRA pulse into running zombie swarm and record propagation
status: experimental
created: 2025-10-21
---
"""

import requests
import time
import json
import logging
import yaml
import numpy as np
from datetime import datetime
import concurrent.futures
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - LoRA-Pulse - %(levelname)s - %(message)s')
logger = logging.getLogger('LoRAPulseInjector')

class LoRAPulseInjector:
    """Injects and monitors LoRA pulse propagation through zombie swarm"""

    def __init__(self, swarm_state_file='bots/swarm_state.yaml'):
        self.load_swarm_state(swarm_state_file)
        self.results = {
            'pulse_start_time': None,
            'pulse_end_time': None,
            'initial_state': {},
            'propagation_data': [],
            'bot_responses': {},
            'metrics': {
                'total_bots': len(self.bots),
                'successful_injections': 0,
                'response_time_avg': 0,
                'pulse_propagation_speed': 0,
                'energy_attenuation': 0
            }
        }

    def load_swarm_state(self, filename):
        """Load current swarm configuration"""
        try:
            with open(filename, 'r') as f:
                state = yaml.safe_load(f)
                if state and isinstance(state, dict):
                    self.bots = state['bots']
                    self.grid_size = state.get('grid_width', 10)
                    logger.info(f"Loaded {len(self.bots)} bots from swarm state")
                else:
                    logger.error("Invalid swarm state format")
                    self.bots = []
        except Exception as e:
            logger.error(f"Failed to load swarm state: {e}")
            self.bots = []

    def get_bot_state(self, bot):
        """Get current state from a bot"""
        try:
            url = f"http://localhost:{bot['port']}/state"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get state from bot {bot['bot_id']}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Error getting state from bot {bot['bot_id']}: {e}")
            return None

    def inject_pulse(self, target_bot, pulse_energy=1.0, pulse_radius=3):
        """Inject high-energy LoRA pulse into target bot and neighbors"""
        logger.info(f"🌀 Injecting LoRA pulse (E={pulse_energy}) at bot {target_bot['bot_id']} (grid: {target_bot['grid_x']},{target_bot['grid_y']})")

        # Find target bot and nearby bots for pulse injection
        target_bots = self.find_bots_in_radius(target_bot['grid_x'], target_bot['grid_y'], pulse_radius)

        pulse_data = {
            'pulse_energy': pulse_energy,
            'pulse_radius': pulse_radius,
            'target_coordinates': (target_bot['grid_x'], target_bot['grid_y']),
            'affected_bots': len(target_bots),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"💥 Pulse will affect {len(target_bots)} bots in radius {pulse_radius}")

        # Record pre-pulse states
        logger.info("📊 Recording pre-pulse swarm state...")
        self.results['initial_state'] = self.record_swarm_snapshot()

        # Start pulse injection
        self.results['pulse_start_time'] = datetime.now().isoformat()
        start_time = time.time()

        successful_injections = 0
        response_times = []

        # Inject pulse into all affected bots concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = []
            for bot in target_bots:
                # Calculate pulse energy based on distance from center
                distance = self.calculate_distance(bot, target_bot)
                adjusted_energy = pulse_energy * max(0.1, 1.0 - distance/pulse_radius)

                futures.append(executor.submit(self.inject_energy_into_bot, bot, adjusted_energy))

            # Wait for all injections to complete
            for future in concurrent.futures.as_completed(futures):
                success, response_time, bot_id = future.result()
                if success:
                    successful_injections += 1
                    response_times.append(response_time)

        injection_duration = time.time() - start_time

        # Record results
        self.results['metrics']['successful_injections'] = successful_injections
        self.results['metrics']['response_time_avg'] = np.mean(response_times) if response_times else 0

        logger.info("✅ Pulse injection complete!")
        logger.info(".2f")
        logger.info(".2%")
        logger.info(".3f")

        return pulse_data

    def inject_energy_into_bot(self, bot, energy_level):
        """Inject energy parameter into individual bot"""
        try:
            url = f"http://localhost:{bot['port']}/parameters/update"
            payload = {
                'updates': {
                    'diffusion_rate': min(1.0, energy_level),  # Use diffusion_rate as energy proxy
                    'noise_level': max(0.0, energy_level - 0.5),  # Additional energy effect
                    'entropy_threshold': max(0.5, 2.0 - energy_level)  # Energy affects entropy threshold
                },
                'pulse_timestamp': datetime.now().isoformat()
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                logger.debug(f"✅ Bot {bot['bot_id']}: Energy injection successful ({response_time:.3f}s)")
                return True, response_time, bot['bot_id']
            else:
                logger.warning(f"❌ Bot {bot['bot_id']}: Inject failed ({response.status_code})")
                return False, response_time, bot['bot_id']

        except Exception as e:
            logger.warning(f"❌ Bot {bot['bot_id']}: Inject error ({e})")
            return False, 0, bot['bot_id']

    def find_bots_in_radius(self, center_x, center_y, radius):
        """Find all bots within radius of center coordinates"""
        target_bots = []
        for bot in self.bots:
            distance = np.sqrt((bot['grid_x'] - center_x)**2 + (bot['grid_y'] - center_y)**2)
            if distance <= radius:
                target_bots.append(bot)
        return target_bots

    def calculate_distance(self, bot1, bot2):
        """Calculate grid distance between two bots"""
        return np.sqrt((bot1['grid_x'] - bot2['grid_x'])**2 + (bot1['grid_y'] - bot2['grid_y'])**2)

    def record_swarm_snapshot(self):
        """Take snapshot of entire swarm state"""
        snapshot = {}
        for bot in self.bots[:20]:  # Sample first 20 bots to avoid overwhelming
            state = self.get_bot_state(bot)
            if state:
                snapshot[bot['bot_id']] = {
                    'vectors_sample': state.get('vectors', [])[:10],  # First 10 dims
                    'memory_entries': state.get('memory_entries', 0),
                    'timestamp': state.get('timestamp')
                }
        return snapshot

    def monitor_pulse_propagation(self, duration_seconds=60):
        """Monitor how pulse propagates through swarm over time"""
        logger.info(f"🔍 Monitoring pulse propagation for {duration_seconds}s...")

        snapshots = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'elapsed': time.time() - start_time,
                'bot_states': self.record_swarm_snapshot()
            }
            snapshots.append(snapshot)
            time.sleep(5)  # Monitor every 5 seconds

            progress = (time.time() - start_time) / duration_seconds * 100
            logger.info(".1f")

        self.results['propagation_data'] = snapshots
        self.results['pulse_end_time'] = datetime.now().isoformat()

        # Analyze propagation patterns
        self.analyze_propagation_results()

        return snapshots

    def analyze_propagation_results(self):
        """Analyze pulse propagation patterns and record metrics"""
        if not self.results['propagation_data']:
            return

        logger.info("🧮 Analyzing pulse propagation results...")

        # Calculate propagation speed (how quickly energy spreads)
        initial_states = self.results['initial_state']
        final_states = self.results['propagation_data'][-1]['bot_states']

        energy_changes = []
        for bot_id in initial_states:
            if bot_id in final_states:
                initial_energy = np.mean(initial_states[bot_id].get('vectors_sample', [0]))
                final_energy = np.mean(final_states[bot_id].get('vectors_sample', [0]))
                energy_changes.append(abs(final_energy - initial_energy))

        if energy_changes:
            avg_energy_change = np.mean(energy_changes)
            attenuation_rate = 1.0 - (avg_energy_change / 1.0)  # Assuming max energy of 1.0

            self.results['metrics']['energy_attenuation'] = attenuation_rate
            self.results['metrics']['pulse_propagation_speed'] = 1.0 - attenuation_rate  # Simpler metric

            logger.info(".3f")
            logger.info(".3f")

            if attenuation_rate < 0.3:
                logger.info("🎯 HIGH ENERGY RETENTION - Pulse propagated effectively!")
            elif attenuation_rate < 0.7:
                logger.info("⚖️ MEDIUM ATTENUATION - Balanced energy diffusion")
            else:
                logger.info("🌊 HIGH ATTENUATION - Energy dissipated quickly (normal for CA systems)")

    def save_results(self, filename=None):
        """Save pulse injection results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/pulse_experiment_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"💾 Results saved to {filename}")
        return filename

    def run_pulse_experiment(self, target_coords=(5, 5), energy=1.0, radius=3, monitor_seconds=30):
        """Run complete pulse injection experiment"""
        logger.info("🧪 STARTING LORA PULSE INJECTION EXPERIMENT")
        logger.info("="*60)

        # Find target bot
        target_bot = None
        for bot in self.bots:
            if bot['grid_x'] == target_coords[0] and bot['grid_y'] == target_coords[1]:
                target_bot = bot
                break

        if not target_bot:
            logger.error(f"Target bot at coordinates {target_coords} not found!")
            return None

        # Inject pulse
        pulse_data = self.inject_pulse(target_bot, energy, radius)

        # Monitor propagation
        self.monitor_pulse_propagation(monitor_seconds)

        # Save results
        result_file = self.save_results()

        logger.info("🧪 PULSE EXPERIMENT COMPLETE")
        logger.info(f"Results: {result_file}")
        logger.info("="*60)

        return result_file

def main():
    """CLI interface for pulse injection"""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Pulse Injector for Zombie Swarm")
    parser.add_argument('--target-x', type=int, default=5, help='Target bot X coordinate')
    parser.add_argument('--target-y', type=int, default=5, help='Target bot Y coordinate')
    parser.add_argument('--energy', type=float, default=1.0, help='Pulse energy level (0-1)')
    parser.add_argument('--radius', type=int, default=3, help='Pulse radius in grid cells')
    parser.add_argument('--monitor', type=int, default=30, help='Monitoring duration (seconds)')

    args = parser.parse_args()

    injector = LoRAPulseInjector()
    result_file = injector.run_pulse_experiment(
        target_coords=(args.target_x, args.target_y),
        energy=args.energy,
        radius=args.radius,
        monitor_seconds=args.monitor
    )

    if result_file:
        print(f"✅ Pulse experiment completed successfully!")
        print(f"📊 Results saved to: {result_file}")
    else:
        print("❌ Pulse experiment failed!")
        exit(1)

if __name__ == '__main__':
    main()
