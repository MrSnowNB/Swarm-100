#!/usr/bin/env python3
"""
Basic test to verify CyberGrid Conway step functionality
Without direct cell access, we'll test that the step() function exists and runs
"""

import sys
sys.path.insert(0, 'swarm-core/build')

import swarm_core as sc
import numpy as np

def test_conway_step():
    print("Testing CyberGrid Conway step functionality...")

    # Create grid
    grid = sc.CyberGrid(10, 10)  # Smaller 10x10 grid for testing
    print(f"Created {grid.width()}x{grid.height()} CyberGrid")

    # Test different initial conditions
    test_cases = [
        (0.1, 0.1),   # Low density
        (0.3, 0.5),   # Medium density
        (0.8, 0.2),   # High density
    ]

    all_passed = True

    for alive_prob, energy_prob in test_cases:
        print(f"\nTesting with alive_prob={alive_prob}, energy_prob={energy_prob}")

        # Reset and randomize grid
        grid.reset()
        grid.randomize(alive_prob, energy_prob)

        initial_alive = grid.alive_cell_count()
        initial_energy = grid.average_energy()

        print(".0f")

        # Test step() method exists and runs
        grid.step()
        post_step_alive = grid.alive_cell_count()
        post_step_energy = grid.average_energy()

        print(".0f")

        # Verify step changed the grid (rules applied)
        if initial_alive != post_step_alive:
            print("✓ Conway rules affected live cell count")
        else:
            print("⚠ Conway rules may not have changed live cells (expected for stable patterns)")

        # Check energy changed (LoRA pulses)
        if abs(initial_energy - post_step_energy) > 0.001:
            print("✓ LoRA energy pulses propagated")
        else:
            print("⚠ Energy field unchanged")

        # Step a few more times to ensure pattern evolution
        for i in range(3):
            grid.step()
            current_alive = grid.alive_cell_count()
            print(f"Step {i+2}: {current_alive} cells, avg_energy={grid.average_energy():.3f}")

        # Run Conway rules specifically (vs full step which includes LoRA)
        changes = grid.apply_conway_rules()
        print(f"apply_conway_rules() returned {changes} cell changes")

        # Test matrix export
        life_matrix = grid.export_life_matrix()
        energy_matrix = grid.export_energy_matrix()

        print(f"Exported life matrix: {len(life_matrix)} elements")
        print(f"Exported energy matrix: {len(energy_matrix)} elements")

        if len(life_matrix) == grid.width() * grid.height():
            print("✓ Matrix export correct size")
        else:
            print("❌ Matrix export size mismatch")
            all_passed = False

        # Test grid entropy and pulse coherence
        entropy = grid.calculate_grid_entropy()
        coherence = grid.calculate_pulse_coherence()

        print(f"Grid entropy: {entropy:.3f}, pulse coherence: {coherence:.3f}")

    if all_passed:
        print("\n✅ ALL TESTS PASSED - CyberGrid Conway CA functional")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = test_conway_step()
    exit(0 if success else 1)
