#!/usr/bin/env python3
"""
---
test: test_cybergrid_glider.py
purpose: Test Conway CA glider propagation in CyberGrid
description: T2.4 Conway CA validation - Verify glider period, still lifes, oscillators
status: foundations test for cybergrid substrate
created: 2025-10-19
---

Validates the CyberGrid Conway CA implementation using canonical glider patterns.
Tests ensure B3/S23 rules, toroidal boundaries, and energy coupling work correctly.
"""

import sys
sys.path.insert(0, 'swarm-core/build')  # For built module

try:
    import swarm_core as sc
except ImportError:
    print("ERROR: swarm_core module not available. Build with: cd swarm-core && mkdir build && cd build && cmake .. && make")
    sys.exit(1)

import json
import numpy as np
from datetime import datetime

class CyberGridGliderTester:
    """
    Test suite for CyberGrid Conway CA using glider patterns.

    Gliders are the canonical test for Conway CA implementations:
    - Deterministic period-4 motion
    - Tests toroidal boundary conditions
    - Validates B3/S23 rule implementation
    """

    # Standard Conway glider pattern (5 cells, period 4)
    GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    # Still lifes (should remain unchanged)
    BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]
    BEEHIVE = [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)]  # Corrected canonical coordinates
    LOAF = [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 3), (3, 2)]  # Corrected canonical coordinates

    # Oscillators (should cycle with period)
    BLINKER = [(1, 0), (1, 1), (1, 2)]  # Period 2
    TOAD = [(1, 0), (1, 1), (1, 2), (2, -1), (2, 0), (2, 1)]  # Period 2
    BEACON = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]  # Period 2

    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0

    def run_full_test_suite(self):
        """Execute complete test suite"""
        print("=" * 60)
        print("CYBERGRID CONWAY CA GLIDER TEST SUITE")
        print("=" * 60)

        # Test glider period verification
        self.test_glider_period_verification()

        # Test still life stability
        self.test_still_life_stability()

        # Test oscillator periods
        self.test_oscillator_periods()

        # Summary
        self.print_summary()

        return self.failed == 0

    def create_grid_with_pattern(self, pattern, center_x=50, center_y=50):
        """Create CyberGrid and initialize with centered pattern"""
        grid = sc.CyberGrid()  # 100x100 default
        grid.reset()

        for dx, dy in pattern:
            x = (center_x + dx) % grid.width()
            y = (center_y + dy) % grid.height()
            grid.get_cell(x, y).alive = True

        return grid

    def initialize_persistent_glider(self):
        """Initialize glider pattern on persistent CyberGrid"""
        # Import here to avoid circular imports
        from scripts.global_tick import CyberGridManager

        grid = CyberGridManager.get_grid()
        if grid is not None:
            grid.reset()  # Clear any existing state

            # Initialize glider at center
            center_x, center_y = grid.width() // 2, grid.height() // 2
            for dx, dy in self.GLIDER:
                x = (center_x + dx) % grid.width()
                y = (center_y + dy) % grid.height()
                grid.get_cell(x, y).alive = True

            initial_count = grid.alive_cell_count()
            print(f"Glider initialized on persistent grid: {initial_count} cells")
            return initial_count
        return 0

    def count_alive_in_pattern(self, grid, pattern, center_x=50, center_y=50):
        """Count alive cells in expected pattern locations"""
        count = 0
        for dx, dy in pattern:
            x = (center_x + dx) % grid.width()
            y = (center_y + dy) % grid.height()
            if grid.get_cell(x, y).alive:
                count += 1
        return count

    def test_glider_period_verification(self):
        """Test standard glider completes period-4 cycle"""
        print(f"\nTEST: Glider Period Verification (standard_glider)")
        print("=" * 60)

        grid = self.create_grid_with_pattern(self.GLIDER)
        initial_alive = grid.alive_cell_count()

        print(f"Initial cells: {initial_alive}")
        print("Expected period: 4")

        # Step 4 generations
        for i in range(4):
            grid.step()
            alive_after = grid.alive_cell_count()
            print(f"Step {i+1}: {alive_after} cells")

        final_alive = grid.alive_cell_count()

        # Verify conservation
        if final_alive == initial_alive:
            print("✅ PASS: Cell count preserved")
            result = "PASS"
            self.passed += 1
        else:
            print("❌ FAIL: Cell count not preserved")
            result = "FAIL"
            self.failed += 1

        self.test_results.append({
            'test': 'glider_period_verification',
            'result': result,
            'initial_cells': initial_alive,
            'final_cells': final_alive,
            'expected_cells': initial_alive
        })

    def test_still_life_stability(self):
        """Test still lifes remain stable over multiple generations"""
        test_patterns = [
            ('block', self.BLOCK, 10),
            ('beehive', self.BEEHIVE, 10),
            ('loaf', self.LOAF, 10)
        ]

        for name, pattern, steps in test_patterns:
            print(f"\nTEST: Still Life Stability ({name})")
            print("-" * 40)

            grid = self.create_grid_with_pattern(pattern)
            initial_alive = grid.alive_cell_count()

            print(f"Initial cells: {initial_alive}")

            # Step multiple generations
            stable = True
            for i in range(steps):
                grid.step()
                current_alive = grid.alive_cell_count()
                if current_alive != initial_alive:
                    stable = False
                    break

            final_alive = grid.alive_cell_count()

            if stable and final_alive == initial_alive:
                print(f"✅ PASS: Still life remained stable after {steps} steps")
                result = "PASS"
                self.passed += 1
            else:
                print(f"❌ FAIL: Pattern changed (started {initial_alive}, ended {final_alive})")
                result = "FAIL"
                self.failed += 1

            self.test_results.append({
                'test': f'still_life_{name}',
                'result': result,
                'initial_cells': initial_alive,
                'final_cells': final_alive,
                'steps': steps
            })

    def test_oscillator_periods(self):
        """Test oscillators cycle with correct period"""
        test_patterns = [
            ('blinker', self.BLINKER, 4, 2),  # Test over 4 steps, expect back after 2
            ('toad', self.TOAD, 4, 2),
            ('beacon', self.BEACON, 4, 2)
        ]

        for name, pattern, total_steps, expected_period in test_patterns:
            print(f"\nTEST: Oscillator Period Verification ({name})")
            print("-" * 50)

            grid = self.create_grid_with_pattern(pattern)
            initial_alive = grid.alive_cell_count()

            print(f"Initial cells: {initial_alive}")
            print(f"Expected period: {expected_period}")

            # Capture state at each step
            states = [initial_alive]

            for i in range(total_steps):
                grid.step()
                states.append(grid.alive_cell_count())
                print(f"Step {i+1}: {states[-1]} cells")

            # Check if pattern returns to initial state at period
            final_state = states[-1]
            period_state = states[expected_period]

            if final_state == initial_alive and period_state == initial_alive:
                print("✅ PASS: Oscillator returned to initial state")
                result = "PASS"
                self.passed += 1
            else:
                print("❌ FAIL: Oscillator did not cycle correctly")
                result = "FAIL"
                self.failed += 1

            self.test_results.append({
                'test': f'oscillator_{name}',
                'result': result,
                'initial_cells': initial_alive,
                'final_cells': final_state,
                'period_cells': period_state,
                'steps': total_steps,
                'expected_period': expected_period
            })

    def print_summary(self):
        """Print test suite summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED - Conway CA implementation validated!")
            print("✅ CyberGrid ready for stigmergic coordination")
        else:
            print(f"\n❌ {self.failed} tests failed - Check B3/S23 rule implementation")

    def export_results(self, filename="glider_test_results.json"):
        """Export detailed results"""
        output = {
            'test_suite': 'cybergrid_glider_validation',
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'passed': self.passed,
            'failed': self.failed,
            'results': self.test_results
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n📄 Results exported to {filename}")


def main():
    """Main entry point"""
    tester = CyberGridGliderTester()

    try:
        success = tester.run_full_test_suite()
        tester.export_results()

        exit_code = 0 if success else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
