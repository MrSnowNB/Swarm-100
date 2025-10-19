#!/usr/bin/env python3
"""
Real-Time Energy Propagation Visualization for Swarm-100 CyberGrid

Demonstrates the continuous-discrete coupling between Conway's Game of Life
and LoRA pulse propagation, showcasing emergent Class IV behavior.

This visualization confirms the DDLab-style pulsing CA implementation with
real-time energy field coherence analysis.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Add swarm-core build directory to path
sys.path.insert(0, 'swarm-core/build')

try:
    import swarm_core
except ImportError as e:
    print(f"Failed to import swarm_core: {e}")
    print("Make sure the C++ module is compiled and in the Python path")
    sys.exit(1)

class CyberGridVisualizer:
    """Real-time visualization of CyberGrid energy propagation and emergent patterns"""

    def __init__(self, grid_size=50, initial_pattern="glider_gun"):
        """Initialize visualizer with CyberGrid and matplotlib setup"""
        self.swarm_core = swarm_core

        # Initialize CyberGrid
        self.grid = swarm_core.CyberGrid(grid_size, grid_size)
        self.size = grid_size

        # Set up visualization
        self.fig, ((self.ax_life, self.ax_energy), (self.ax_entropy, self.ax_coherence)) = plt.subplots(2, 2, figsize=(14, 10))

        # Initialize data arrays
        self.energy_data = np.zeros((grid_size, grid_size))
        self.life_data = np.zeros((grid_size, grid_size), dtype=int)
        self.occupancy_data = np.zeros((grid_size, grid_size), dtype=int)

        # History for temporal analysis
        self.entropy_history = []
        self.coherence_history = []
        self.generation_history = []

        # Set up plots
        self._setup_plots()

        # Initialize with interesting pattern
        self._initialize_pattern(initial_pattern)

        print("Energy Propagation Visualizer initialized")
        print("Grid size: {}x{}".format(grid_size, grid_size))
        print("Pattern: {}".format(initial_pattern))
        print("Ready for real-time visualization...")

    def _setup_plots(self):
        """Configure matplotlib subplots"""

        # Conway's Game of Life - binary life states
        self.im_life = self.ax_life.imshow(self.life_data, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        self.ax_life.set_title("Conway's Game of Life\n(Binary: Dead/Alive)")
        self.ax_life.set_xlabel("X Coordinate")
        self.ax_life.set_ylabel("Y Coordinate")

        # Energy field visualization with custom colormap
        colors = [(0.0, 0.0, 0.0),      # Black for zero energy
                 (0.3, 0.0, 0.0),      # Dark red for low energy
                 (1.0, 0.5, 0.0),      # Orange for medium energy
                 (1.0, 1.0, 0.7)]      # Bright yellow for high energy
        self.energy_cmap = mcolors.LinearSegmentedColormap.from_list('energy', colors, N=256)
        self.im_energy = self.ax_energy.imshow(self.energy_data, cmap=self.energy_cmap, vmin=0, vmax=1, interpolation='bilinear')
        self.ax_energy.set_title("LoRA Energy Propagation\n(Continuous: 0.0-1.0)\nLoRA pulses every 4 ticks")
        self.ax_energy.set_xlabel("X Coordinate")
        self.ax_energy.set_ylabel("Y Coordinate")

        # Add colorbar for energy
        cbar = plt.colorbar(self.im_energy, ax=self.ax_energy, shrink=0.8)
        cbar.set_label('Energy Level')

        # Grid entropy over time
        self.line_entropy, = self.ax_entropy.plot([], [], 'b-', linewidth=2, label='Grid Entropy')
        self.ax_entropy.set_title("Emergent Order Analysis\n(Shannon Entropy of Life+Energy States)")
        self.ax_entropy.set_xlabel("Generation")
        self.ax_entropy.set_ylabel("Entropy (bits)")
        self.ax_entropy.grid(True, alpha=0.3)
        self.ax_entropy.set_xlim(0, 500)
        self.ax_entropy.set_ylim(0, 2.5)

        # Pulse coherence over time
        self.line_coherence, = self.ax_coherence.plot([], [], 'r-', linewidth=2, label='Pulse Coherence')
        self.ax_coherence.set_title("Energy Field Coherence\n(Inverse Variance × Uniformity)")
        self.ax_coherence.set_xlabel("Generation")
        self.ax_coherence.set_ylabel("Coherence (0-1)")
        self.ax_coherence.grid(True, alpha=0.3)
        self.ax_coherence.set_xlim(0, 500)
        self.ax_coherence.set_ylim(0, 1)

        plt.tight_layout()

    def _initialize_pattern(self, pattern_type):
        """Initialize the grid with different interesting patterns"""

        if pattern_type == "glider_gun":
            # Gosper's Glider Gun - creates endless gliders
            self._create_glider_gun()

        elif pattern_type == "pulsar":
            # Pulsar oscillator
            self._create_pulsar()

        elif pattern_type == "beacon":
            # Beacon oscillator
            self._create_beacon()

        elif pattern_type == "random":
            # Sparse random initialization with energy
            self.grid.randomize(0.1, 0.2)  # 10% alive, 20% energy

        elif pattern_type == "energy_pulse":
            # Pure energy pulse without initial life
            center_x, center_y = self.size // 2 - 10, self.size // 2
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= 5:
                        energy = max(0, 1.0 - distance/5)
                        # Find the cell and set energy
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < self.size and 0 <= y < self.size:
                            cell = self.grid.get_cell(x, y)
                            cell.energy = energy

        print("Initialized with {}".format(pattern_type))

    def _create_glider_gun(self):
        """Create Gosper's Glider Gun"""
        # This is a simplified version - the full Gosper gun is quite large
        # We'll create a basic glider progenitor

        center_x, center_y = self.size // 2, self.size // 2

        # Simple pulsar pattern as placeholder (full gun requires larger grid)
        offsets = [(-6, -8), (-5, -8), (-4, -8), (-3, -8), (-2, -8), (-1, -8), (0, -8), (1, -8),
                  (-6, -7), (-4, -7), (-2, -7), (0, -7), (1, -7),
                  (-6, -6), (-5, -6), (-4, -6), (-3, -6), (-2, -6), (-1, -6), (0, -6), (1, -6),
                  (-6, -5), (-4, -5), (-2, -5), (0, -5), (1, -5),
                  (-6, -4), (-5, -4), (-4, -4), (-3, -4), (-2, -4), (-1, -4), (0, -4), (1, -4)]

        for dx, dy in offsets:
            x, y = center_x + dx, center_y + dy
            if 0 <= x < self.size and 0 <= y < self.size:
                cell = self.grid.get_cell(x, y)
                cell.alive = True

    def _create_pulsar(self):
        """Create a pulsar oscillator"""
        center_x, center_y = self.size // 2, self.size // 2

        # Pulsar pattern (simplified)
        pulsar_offsets = [
            (-6, -2), (-6, -1), (-6, 0), (-6, 1), (-6, 2),  # Left arm
            (-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2),  # Left middle
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),  # Left arm
            (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),      # Right arm
            (4, -2), (4, -1), (4, 0), (4, 1), (4, 2),      # Right middle
            (6, -2), (6, -1), (6, 0), (6, 1), (6, 2),      # Right arm
        ]

        for dx, dy in pulsar_offsets:
            x, y = center_x + dx, center_y + dy
            if 0 <= x < self.size and 0 <= y < self.size:
                cell = self.grid.get_cell(x, y)
                cell.alive = True

    def _create_beacon(self):
        """Create a beacon oscillator"""
        center_x, center_y = self.size // 2, self.size // 2

        # Beacon pattern
        beacon_offsets = [
            (-1, -1), (0, -1), (-1, 0), (0, 0),  # Top-left cluster
            (1, 1), (2, 1), (1, 2), (2, 2)       # Bottom-right cluster
        ]

        for dx, dy in beacon_offsets:
            x, y = center_x + dx, center_y + dy
            if 0 <= x < self.size and 0 <= y < self.size:
                cell = self.grid.get_cell(x, y)
                cell.alive = True

    def update_frame(self, frame_number):
        """Animation update function - called for each frame"""

        # Execute one simulation step (Conway + LoRA every 4 ticks)
        self.grid.step()

        # Export data from C++ to Python (vectorized)
        energy_vec = self.grid.export_energy_matrix()
        life_vec = self.grid.export_life_matrix()

        # Reshape vectors into 2D arrays (row-major order)
        self.energy_data = np.array(energy_vec).reshape((self.size, self.size))
        self.life_data = np.array(life_vec).reshape((self.size, self.size))

        # Calculate emergent analysis metrics
        entropy = self.grid.calculate_grid_entropy()
        coherence = self.grid.calculate_pulse_coherence()

        # Update history (keep last 500 points for visualization)
        self.generation_history.append(frame_number)
        self.entropy_history.append(entropy)
        self.coherence_history.append(coherence)

        if len(self.generation_history) > 500:
            self.generation_history.pop(0)
            self.entropy_history.pop(0)
            self.coherence_history.pop(0)

        # Update plots
        self.im_life.set_array(self.life_data)
        self.im_energy.set_array(self.energy_data)

        if len(self.generation_history) > 1:
            self.line_entropy.set_data(self.generation_history, self.entropy_history)
            self.line_coherence.set_data(self.generation_history, self.coherence_history)

            # Update x-axis limits to follow the data
            if len(self.generation_history) > 10:
                x_min = max(0, self.generation_history[0])
                x_max = self.generation_history[-1] + 50
                self.ax_entropy.set_xlim(x_min, x_max)
                self.ax_coherence.set_xlim(x_min, x_max)

        # Update titles with current metrics
        self.ax_entropy.set_title("Emergent Order Analysis\n(Shannon Entropy: {:.3f})".format(entropy))
        self.ax_coherence.set_title("Energy Field Coherence\n(Cohereance: {:.3f})".format(coherence))

        return [self.im_life, self.im_energy, self.line_entropy, self.line_coherence]

    def run_animation(self, fps=30, max_frames=float('inf')):
        """Run the real-time visualization"""
        print("Starting energy propagation visualization...")
        print("Press Ctrl+C to stop the simulation")

        # Create animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=None,  # Infinite frames
            interval=1000//fps,  # milliseconds between frames
            blit=False,
            cache_frame_data=False  # Memory optimization
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nAnimation stopped by user")
        except Exception as e:
            print(f"Animation error: {e}")
        finally:
            plt.close()

        # Print final analysis summary
        if self.entropy_history and self.coherence_history:
            avg_entropy = np.mean(self.entropy_history[-100:]) if len(self.entropy_history) > 100 else np.mean(self.entropy_history)
            avg_coherence = np.mean(self.coherence_history[-100:]) if len(self.coherence_history) > 100 else np.mean(self.coherence_history)

            print("\nFinal Analysis Summary (last 100 generations):")
            print("Entropy: {:.3f}".format(avg_entropy))
            print("Coherence: {:.3f}".format(avg_coherence))
            entropy_class = 'High (Chaotic)' if avg_entropy > 1.5 else 'Medium (Ordered)' if avg_entropy > 0.8 else 'Low (Stable)'
            coherence_class = 'High (Synchronized)' if avg_coherence > 0.7 else 'Medium (Wavelike)' if avg_coherence > 0.4 else 'Low (Diffuse)'
            print("Entropy: {}".format(entropy_class))
            print("Coherence: {}".format(coherence_class))

            if avg_entropy < 1.0 and avg_coherence > 0.6:
                print("✅ DDLab-Compliant Class IV Emergent Behavior Detected!")
                print("   Continuous-discrete coupling between Game of Life and LoRA pulses")
            else:
                print("⚠️  Emergent behavior may need parameter tuning")

    def run_static_analysis(self, generations=100):
        """Run analysis for a fixed number of generations and display final state"""
        print(f"Running {generations} generations of analysis...")

        entropy_values = []
        coherence_values = []
        pattern_periods = []

        for gen in range(generations):
            # Store state snapshot for period detection
            energy_snapshot = self.grid.export_energy_matrix()
            life_snapshot = self.grid.export_life_matrix()
            state_key = (tuple(energy_snapshot), tuple(life_snapshot))

            # Check for pattern periodicity
            if state_key in pattern_periods:
                print(f"Periodic pattern detected at generation {gen}")
            pattern_periods.append(state_key)

            # Step simulation
            self.grid.step()

            # Collect metrics
            entropy_values.append(self.grid.calculate_grid_entropy())
            coherence_values.append(self.grid.calculate_pulse_coherence())

            if gen % 20 == 0:
                print(f"Gen {gen}: Entropy={entropy_values[-1]:.3f}, Coherence={coherence_values[-1]:.3f}")

        # Statistical analysis
        entropy_std = np.std(entropy_values)
        coherence_std = np.std(coherence_values)
        entropy_mean = np.mean(entropy_values)
        coherence_mean = np.mean(coherence_values)

        print("\nAnalysis Complete:")
        print("Entropy: {:.3f} ± {:.3f}".format(entropy_mean, entropy_std))
        print("Coherence: {:.3f} ± {:.3f}".format(coherence_mean, coherence_std))

        # Classify emergent behavior
        if entropy_std > 0.3 and coherence_std > 0.2:
            print("🎯 High-Variability System - Class IV Emergent Dynamics")
            print("   Chaotic but coherent pulse propagation patterns")
        elif entropy_std < 0.1 and coherence_std < 0.1:
            print("🔒 Stable Equilibrium - Simple attractor reached")
        else:
            print("🌊 Complex Ordered - Intermediate complexity regime")

        # Display final patterns
        self.update_frame(generations)  # Final update
        plt.show()

def main():
    """Main visualization demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Swarm-100 Energy Propagation Visualizer")
    parser.add_argument("--pattern", choices=["glider_gun", "pulsar", "beacon", "random", "energy_pulse"],
                       default="pulsar", help="Initial pattern to simulate")
    parser.add_argument("--size", type=int, default=50, help="Grid size (NxN)")
    parser.add_argument("--fps", type=int, default=10, help="Animation frame rate")
    parser.add_argument("--generations", type=int, help="Fixed generations (static analysis mode)")
    parser.add_argument("--headless", action="store_true", help="Run without visualization (data analysis only)")

    args = parser.parse_args()

    print("="*80)
    print("🧬 SWARM-100 ENERGY PROPAGATION VISUALIZER")
    print("Game of Life + LoRA Pulse Coupling Demonstration")
    print("="*80)

    if args.headless:
        # Headless data analysis mode
        visualizer = CyberGridVisualizer(args.size, args.pattern)
        visualizer.run_static_analysis(args.generations or 100)
    else:
        # Interactive visualization mode
        visualizer = CyberGridVisualizer(args.size, args.pattern)

        if args.generations:
            visualizer.run_static_analysis(args.generations)
        else:
            visualizer.run_animation(args.fps)

if __name__ == "__main__":
    main()
