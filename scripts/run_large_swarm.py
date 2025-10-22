#!/usr/bin/env python3
"""
---
script: run_large_swarm.py
purpose: Scale swarm CA experiments from 10x10 to 20x20 to test computational limits
status: experimental
created: 2025-10-19
---
"""

import numpy as np
import time
import yaml
import json
import logging
import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from rule_engine import CellularRuleEngine

logger = logging.getLogger('LargeSwarmScaling')

class LargeSwarmExperiment:
    """
    Scale CA experiments from 10x10 to 20x20 and beyond.
    Tests computational limits of distributed swarm CA system.
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.total_bots = grid_size * grid_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"large_swarm_{grid_size}x{grid_size}_{self.timestamp}"

        self.rule_engine = CellularRuleEngine()
        self.metrics_history = []
        self.performance_metrics = {}

    def _generate_initial_swarm_state(self) -> dict:
        """Generate initial N×N grid of bots with realistic parameters"""
        logger.info(f"Generating initial {self.grid_size}x{self.grid_size} swarm state ({self.total_bots} bots)")

        bots = []

        for bot_id in range(1, self.total_bots + 1):
            grid_x = (bot_id - 1) % self.grid_size
            grid_y = (bot_id - 1) // self.grid_size

            # Create more realistic state vectors (simulating LLM embeddings)
            # Gaussian initialization as proxy for emergent behavior
            state_vector = np.random.normal(0, 0.5, 512).tolist()

            # Add some spatial correlation for more interesting CA dynamics
            # Adjacent bots have related states
            if grid_x > 0:
                # Correlate with left neighbor (very weak for complexity)
                state_vector[0] += np.random.normal(0, 0.01)
            if grid_y > 0:
                # Correlate with bottom neighbor
                state_vector[1] += np.random.normal(0, 0.01)

            # Initialize LoRA pulse energy (simulating recent LLM activity)
            initial_energy = np.random.normal(0.5, 0.2)
            initial_energy = np.clip(initial_energy, 0, 1)  # Bind to [0,1]

            bots.append({
                'bot_id': bot_id,
                'grid_x': grid_x,
                'grid_y': grid_y,
                'port': 9000 + bot_id,
                'alive': True,
                'state_vector': state_vector,
                'state_magnitude': float(np.linalg.norm(state_vector)),
                'lora_energy': float(initial_energy),
                'last_zombie_state': False
            })

        return {
            'total_bots': self.total_bots,
            'grid_width': self.grid_size,
            'grid_height': self.grid_size,
            'bots': bots,
            'tick': 0,
            'latest_update': 'initialization',
            'experiment_start': datetime.now().isoformat(),
            'experiment_id': self.experiment_id
        }

    def _calculate_advanced_metrics(self, swarm_state: dict) -> dict:
        """Calculate comprehensive CA scaling metrics"""
        tick = swarm_state.get('tick', 0)
        bots = swarm_state.get('bots', [])
        alive_bots = [bot for bot in bots if bot.get('alive', True)]

        start_time = time.time()

        # Basic state metrics
        magnitudes = [bot['state_magnitude'] for bot in alive_bots]
        energies = [bot.get('lora_energy', 0) for bot in alive_bots]

        # Entropy and convergence metrics
        vector_entropy = float(np.std([np.array(bot['state_vector']) for bot in alive_bots])) / np.sqrt(512)

        # Spatial correlation analysis
        grid_correlation = self._calculate_grid_correlation(swarm_state)
        neighborhood_similarity = self._calculate_neighborhood_similarity(swarm_state)

        # LoRA pulse propagation metrics
        energy_flux = self._calculate_energy_flux(swarm_state)
        pulse_attenuation = self._calculate_pulse_attenuation(swarm_state)

        # Zombie dynamics (if applicable)
        zombie_rate = sum(1 for bot in bots if not bot.get('alive', True)) / len(bots)

        # Computational complexity metrics
        state_complexity = self._estimate_state_complexity(swarm_state)

        calc_time = time.time() - start_time

        return {
            'tick': tick,
            'timestamp': datetime.now().isoformat(),

            'basic_metrics': {
                'alive_bots': len(alive_bots),
                'zombie_rate': float(zombie_rate),
                'mean_magnitude': float(np.mean(magnitudes)),
                'std_magnitude': float(np.std(magnitudes)),
                'mean_energy': float(np.mean(energies)),
                'vector_entropy': vector_entropy
            },

            'correlational_metrics': {
                'grid_correlation': grid_correlation,
                'neighborhood_similarity': neighborhood_similarity,
                'spatial_dependence': float(grid_correlation * neighborhood_similarity)
            },

            'lorapulse_metrics': {
                'energy_flux': energy_flux,
                'pulse_attenuation': pulse_attenuation,
                'energy_gradient': self._calculate_energy_gradient(swarm_state)
            },

            'complexity_metrics': {
                'state_complexity': state_complexity,
                'computational_load': self._estimate_computational_load(swarm_state),
                'bifurcation_potential': self._estimate_bifurcation_potential(swarm_state)
            },

            'performance_metrics': {
                'metrics_calculation_time': float(calc_time),
                'memory_estimation': self._estimate_memory_usage(swarm_state),
                'scalability_index': self._calculate_scalability_index(swarm_state)
            }
        }

    def _calculate_grid_correlation(self, swarm_state: dict) -> float:
        """Calculate global grid correlation coefficient"""
        try:
            bots = swarm_state.get('bots', [])
            vectors = [bot['state_vector'][:50] for bot in bots if bot.get('alive', True)]  # First 50 dims

            if len(vectors) < 2:
                return 0.0

            # Compute pairwise correlations and take mean
            corr_matrix = np.corrcoef(np.array(vectors))
            mean_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])

            return float(np.abs(mean_corr)) if not np.isnan(mean_corr) else 0.0
        except:
            return 0.0

    def _calculate_neighborhood_similarity(self, swarm_state: dict) -> float:
        """Calculate average similarity with direct neighbors"""
        similarities = []
        grid = {(bot['grid_x'], bot['grid_y']): bot for bot in swarm_state['bots'] if bot.get('alive', True)}

        for pos, bot in grid.items():
            x, y = pos
            self_vec = np.array(bot['state_vector'])

            # Check orthogonal neighbors
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            neighbor_sims = []

            for nx, ny in neighbors:
                if (nx, ny) in grid:
                    neighbor_vec = np.array(grid[(nx, ny)]['state_vector'])
                    similarity = np.dot(self_vec, neighbor_vec) / (
                        np.linalg.norm(self_vec) * np.linalg.norm(neighbor_vec) + 1e-8)
                    neighbor_sims.append(similarity)

            if neighbor_sims:
                similarities.append(np.mean(neighbor_sims))

        return float(np.mean(similarities)) if similarities else 0.0

    def _calculate_energy_flux(self, swarm_state: dict) -> dict:
        """Calculate LoRA energy flux patterns"""
        bots = swarm_state.get('bots', [])
        energies = {}

        for bot in bots:
            if bot.get('alive', True):
                energies[(bot['grid_x'], bot['grid_y'])] = bot.get('lora_energy', 0)

        if not energies:
            return {'mean_flux': 0.0, 'flux_variance': 0.0}

        # Calculate local energy gradients
        fluxes = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in energies:
                    center_e = energies[(x, y)]

                    # Sum gradients to neighbors
                    neighbor_sum = 0
                    neighbor_count = 0

                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in energies:
                            neighbor_sum += energies[(nx, ny)] - center_e
                            neighbor_count += 1

                    if neighbor_count > 0:
                        fluxes.append(neighbor_sum / neighbor_count)

        flux_array = np.array(fluxes)
        return {
            'mean_flux': float(np.mean(flux_array)) if len(flux_array) > 0 else 0.0,
            'flux_variance': float(np.var(flux_array)) if len(flux_array) > 0 else 0.0
        }

    def _calculate_pulse_attenuation(self, swarm_state: dict) -> dict:
        """Measure how LoRA pulses attenuate with distance"""
        # Find local energy maxima (pulse centers)
        bots = swarm_state.get('bots', [])
        energies = {(bot['grid_x'], bot['grid_y']): bot.get('lora_energy', 0)
                   for bot in bots if bot.get('alive', True)}

        attenuation_coeffs = []

        # For each potential pulse center
        for center_x in range(1, self.grid_size - 1):
            for center_y in range(1, self.grid_size - 1):
                if (center_x, center_y) in energies:
                    center_e = energies[(center_x, center_y)]

                    # Check if it's a local maximum
                    is_max = True
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = center_x + dx, center_y + dy
                            if (nx, ny) in energies and energies[(nx, ny)] > center_e:
                                is_max = False
                                break
                        if not is_max:
                            break

                    if is_max and center_e > 0.5:  # Significant pulse
                        # Measure attenuation along axes
                        x_attenuation = self._measure_attenuation(center_x, center_y, energies, 'x')
                        y_attenuation = self._measure_attenuation(center_x, center_y, energies, 'y')

                        if x_attenuation and y_attenuation:
                            attenuation_coeffs.append((x_attenuation + y_attenuation) / 2)

        mean_attenuation = float(np.mean(attenuation_coeffs)) if attenuation_coeffs else 0.0
        return {
            'mean_attenuation': mean_attenuation,
            'pulse_count': len(attenuation_coeffs),
            'attenuation_variance': float(np.var(attenuation_coeffs)) if attenuation_coeffs else 0.0
        }

    def _measure_attenuation(self, cx: int, cy: int, energies: dict, axis: str) -> float:
        """Measure attenuation coefficient along one axis"""
        center_e = energies.get((cx, cy), 0)
        if center_e <= 0:
            return 0.0

        # Sample points at increasing distances
        samples = []
        for dist in range(1, min(cx, cy, self.grid_size - 1 - cx, self.grid_size - 1 - cy) + 1):
            if axis == 'x':
                pos1 = (cx + dist, cy)
                pos2 = (cx - dist, cy)
            else:  # y
                pos1 = (cx, cy + dist)
                pos2 = (cx, cy - dist)

            e1 = energies.get(pos1, 0)
            e2 = energies.get(pos2, 0)

            if e1 > 0.01:  # Still detectable
                attenuation = np.log(center_e / (e1 + 0.001)) / dist
                samples.append(max(0, attenuation))  # Only positive attenuation

        return float(np.mean(samples)) if samples else 0.0

    def _calculate_energy_gradient(self, swarm_state: dict) -> float:
        """Calculate mean energy gradient across grid"""
        bots = swarm_state.get('bots', [])
        energies = [bot.get('lora_energy', 0) for bot in bots if bot.get('alive', True)]

        if len(energies) < 4:
            return 0.0

        return float(np.std(energies) / (np.mean(energies) + 0.001))

    def _estimate_state_complexity(self, swarm_state: dict) -> float:
        """Estimate Kolmogorov complexity of swarm state"""
        bots = swarm_state.get('bots', [])
        vectors = [bot['state_vector'][:100] for bot in bots if bot.get('alive', True)]  # Sample dims

        if len(vectors) < 2:
            return 0.0

        # Rough complexity estimate: distance to nearest patterns
        vectors_array = np.array(vectors)
        distances = []

        for i, v1 in enumerate(vectors_array):
            min_dist = float('inf')
            for j, v2 in enumerate(vectors_array):
                if i != j:
                    dist = np.linalg.norm(v1 - v2)
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Higher mean distance = higher complexity
        return float(np.mean(distances) / np.sqrt(100))

    def _estimate_computational_load(self, swarm_state: dict) -> dict:
        """Estimate computational complexity of current state"""
        num_bots = len([bot for bot in swarm_state.get('bots', []) if bot.get('alive', True)])

        # Rough estimates based on N×N scaling
        neighbor_operations = 4 * num_bots  # 4 neighbors per bot
        state_updates = num_bots * 512  # Vector operations
        communication_overhead = num_bots * num_bots / self.grid_size  # Approximate

        return {
            'neighbor_operations': int(neighbor_operations),
            'state_vector_operations': int(state_updates),
            'communication_complexity': int(communication_overhead),
            'total_complexity': int(neighbor_operations + state_updates + communication_overhead)
        }

    def _estimate_bifurcation_potential(self, swarm_state: dict) -> float:
        """Estimate how close system is to bifurcation points"""
        # Simplified measure: variance in local correlations
        correlations = []

        bots_by_pos = {(bot['grid_x'], bot['grid_y']): bot
                      for bot in swarm_state.get('bots', []) if bot.get('alive', True)}

        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if (x, y) in bots_by_pos:
                    bot = bots_by_pos[(x, y)]
                    neighbors = []

                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        npos = (x + dx, y + dy)
                        if npos in bots_by_pos:
                            neighbors.append(bots_by_pos[npos])

                    if len(neighbors) >= 2:
                        center_vec = np.array(bot['state_vector'])[:50]
                        neighbor_vecs = [np.array(n['state_vector'])[:50] for n in neighbors]

                        # Measure correlation variance between neighbors
                        local_corr = np.corrcoef([center_vec] + neighbor_vecs)[0, 1:]

                        if len(local_corr) > 1:
                            correlations.append(np.var(local_corr))

        return float(np.mean(correlations)) if correlations else 0.0

    def _estimate_memory_usage(self, swarm_state: dict) -> dict:
        """Estimate memory usage of swarm state"""
        bots = swarm_state.get('bots', [])

        # Rough memory calculations
        bot_size = 4 * 512 * 8  # state vector as float64
        bot_metadata = 4 * 100  # other fields
        per_bot_memory = bot_size + bot_metadata

        total_bot_memory = len(bots) * per_bot_memory

        # Grid structures
        grid_memory = self.grid_size * self.grid_size * 8  # Pointers or indices

        return {
            'per_bot_kb': per_bot_memory / 1024,
            'total_bots_kb': total_bot_memory / 1024,
            'grid_overhead_kb': grid_memory / 1024,
            'total_estimated_kb': (total_bot_memory + grid_memory) / 1024
        }

    def _calculate_scalability_index(self, swarm_state: dict) -> float:
        """Calculate index of how well system scales with size"""
        num_bots = len([bot for bot in swarm_state.get('bots', []) if bot.get('alive', True)])

        # Scaling metrics: efficiency = useful work per computational step
        # For CA systems, efficiency approximately N/log(N) for optimal scaling
        theoretical_optimal = num_bots / np.log(num_bots + 1)

        # Measured efficiency: alive bots / total capacity
        measured_efficiency = num_bots / (self.grid_size * self.grid_size)

        # Scalability = measured / theoretical
        return float(min(1.0, measured_efficiency * np.log(num_bots + 1) / num_bots))

    def run_scaling_experiment(self, max_ticks: int = 200) -> dict:
        """Execute large swarm scaling experiment"""
        logger.info(f"🚀 Starting {self.grid_size}x{self.grid_size} Large Swarm Scaling Experiment")
        logger.info(f"Bots: {self.total_bots}")
        logger.info(f"Ticks: {max_ticks}")

        # Initialize
        swarm_state = self._generate_initial_swarm_state()
        metrics_history = []

        # Performance tracking
        experiment_start = time.time()
        tick_times = []
        memory_usage = []

        # Main experiment loop
        for tick in range(max_ticks):
            tick_start = time.time()

            # Apply CA rules
            swarm_state = self.rule_engine.update_swarm_state(swarm_state)
            swarm_state['tick'] = tick + 1

            # Calculate metrics
            metrics = self._calculate_advanced_metrics(swarm_state)
            metrics_history.append(metrics)

            tick_time = time.time() - tick_start
            tick_times.append(tick_time)

            # Progress and monitoring
            if tick % 50 == 0:
                alive_count = metrics['basic_metrics']['alive_bots']
                entropy = metrics['basic_metrics']['vector_entropy']
                energy = metrics['basic_metrics']['mean_energy']

                logger.info(f"Tick {tick+1}: Alive={alive_count}, Entropy={entropy:.4f}, Energy={energy:.3f}")

        total_experiment_time = time.time() - experiment_start
        avg_tick_time = np.mean(tick_times)
        avg_memory_mb = np.mean(memory_usage) / (1024 * 1024) if memory_usage else 0

        # Final analysis
        self._analyze_scaling_results(metrics_history)

        # Save comprehensive results
        results = {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'grid_size': self.grid_size,
                'total_bots': self.total_bots,
                'max_ticks': max_ticks,
                'timestamp': datetime.now().isoformat()
            },
            'performance_summary': {
                'total_experiment_time': total_experiment_time,
                'avg_tick_time': avg_tick_time,
                'ticks_per_second': 1.0 / avg_tick_time if avg_tick_time > 0 else 0,
                'scalability_assessment': self._assess_scalability(metrics_history, max_ticks)
            },
            'final_metrics': metrics_history[-1] if metrics_history else {},
            'convergence_analysis': self._analyze_convergence(metrics_history),
            'time_series': metrics_history[::10]  # Every 10th tick for space efficiency
        }

        # Save to file
        output_path = Path(f'logs/scaling/experiment_{self.experiment_id}.yaml')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)

        logger.info(f"📊 Scaling results saved to {output_path}")

        return results

    def _assess_scalability(self, metrics_history, max_ticks: int = 200) -> dict:
        """Assess how well the system scales with grid size"""
        if not metrics_history:
            return {'scalability_score': 0.0}

        final_metrics = metrics_history[-1]

        # Scalability factors
        computation_time = final_metrics.get('performance_metrics', {}).get('metrics_calculation_time', 0)
        scalability_idx = final_metrics.get('performance_metrics', {}).get('scalability_index', 0)

        # Time complexity assessment
        time_scaling = max_ticks / computation_time if computation_time > 0 else 0

        # Memory scaling
        memory_scaling = final_metrics.get('performance_metrics', {}).get('total_estimated_kb', 0)

        # Communication scaling (approximate as O(N²/grid_size))
        comm_scaling = self.total_bots ** 2 / self.grid_size

        return {
            'scalability_score': float(scalability_idx),
            'time_complexity': float(time_scaling),
            'memory_scalability': float(memory_scaling / self.total_bots),  # Per bot
            'communication_overhead': int(comm_scaling),
            'grid_size_feasibility': 'good' if scalability_idx > 0.7 else 'challenging' if scalability_idx > 0.4 else 'problematic'
        }

    def _analyze_convergence(self, metrics_history) -> dict:
        """Analyze convergence patterns in scaling experiment"""
        if len(metrics_history) < 10:
            return {'analysis': 'insufficient_data'}

        # Extract key convergence indicators
        entropies = [m['basic_metrics']['vector_entropy'] for m in metrics_history]
        correlations = [m['correlational_metrics']['grid_correlation'] for m in metrics_history]
        energies = [m['basic_metrics']['mean_energy'] for m in metrics_history]

        # Convergence metrics
        entropy_change = abs(entropies[-1] - entropies[0]) / max(abs(entropies[0]), 0.001)
        correlation_growth = correlations[-1] - correlations[0]
        energy_stability = np.std(energies[-50:]) / np.mean(energies[-50:]) if len(energies) >= 50 else 1.0

        # Classification
        if entropy_change < 0.1 and correlation_growth > 0.5:
            convergence_type = 'strong_convergence'
        elif entropy_change < 0.3 and energy_stability < 0.2:
            convergence_type = 'moderate_convergence'
        elif correlation_growth < 0.1:
            convergence_type = 'no_convergence'
        else:
            convergence_type = 'partial_convergence'

        return {
            'convergence_type': convergence_type,
            'entropy_reduction': float(entropy_change),
            'correlation_increase': float(correlation_growth),
            'energy_stability': float(energy_stability),
            'convergence_score': min(1.0, (correlation_growth + (1 - entropy_change) + (1 - energy_stability)) / 3)
        }

    def _analyze_scaling_results(self, metrics_history):
        """Print detailed analysis of scaling experiment results"""
        if not metrics_history:
            logger.warning("No metrics history available")
            return

        final_metrics = metrics_history[-1]
        convergence_analysis = self._analyze_convergence(metrics_history)
        scalability_assessment = self._assess_scalability(metrics_history, 200)

        logger.info("\n🔬 LARGE SWARM SCALING ANALYSIS")
        logger.info(f"{'='*50}")
        logger.info(f"Grid Size: {self.grid_size}x{self.grid_size} ({self.total_bots} bots)")
        logger.info(f"Convergence: {convergence_analysis['convergence_type']}")
        logger.info(f"Convergence Score: {convergence_analysis['convergence_score']:.3f}")
        logger.info(f"Scalability: {scalability_assessment['grid_size_feasibility']}")
        logger.info(f"Time Complexity: {scalability_assessment['time_complexity']:.1f} ops/s")
        logger.info(f"Memory per Bot: {scalability_assessment['memory_scalability']:.2f} KB")

        # Detailed metrics
        logger.info("\nFinal State Metrics:")
        basic = final_metrics['basic_metrics']
        logger.info(f"  Alive Bots: {basic['alive_bots']}/{self.total_bots} ({basic['zombie_rate']:.1%} zombie rate)")
        logger.info(f"  Vector Entropy: {basic['vector_entropy']:.4f}")
        logger.info(f"  Mean LoRA Energy: {basic['mean_energy']:.3f}")

        corr = final_metrics['correlational_metrics']
        logger.info(f"  Grid Correlation: {corr['grid_correlation']:.4f}")
        logger.info(f"  Neighborhood Similarity: {corr['neighborhood_similarity']:.4f}")

        perf = final_metrics['performance_metrics']
        logger.info(f"  Metrics Calc Time: {perf['metrics_calculation_time']:.3f}s")
        logger.info(f"  Scalability Index: {perf['scalability_index']:.3f}")

def main():
    """Entry point for large swarm scaling experiments"""
    parser = argparse.ArgumentParser(description='Large Swarm Scaling Experiment')
    parser.add_argument('--size', '-s', type=int, default=20, help='Grid size (NxN)')
    parser.add_argument('--ticks', '-t', type=int, default=200, help='Max ticks to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - LARGE_SWARM - %(levelname)s - %(message)s'
    )

    experiment = LargeSwarmExperiment(grid_size=args.size)
    results = experiment.run_scaling_experiment(max_ticks=args.ticks)

    print(f"\n{'='*60}")
    print(f"LARGE SWARM SCALING EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Grid Size: {args.size}x{args.size}")
    print(f"Bots Simulated: {args.size**2}")
    print(f"Ticks: {args.ticks}")
    print(f"Results: logs/scaling/experiment_{experiment.experiment_id}.yaml")
    print(f"\nDetermine if {args.size}x{args.size} grid is computationally feasible!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
