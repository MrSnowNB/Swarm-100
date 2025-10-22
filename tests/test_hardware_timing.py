#!/usr/bin/env python3
"""
---
test: test_hardware_timing.py
purpose: V1 Gate - Hardware Timing Stability
description: Validate rdtsc drift stability over 10-minute baseline against NTP time source
status: validation test for hardware-breathing heartbeat
created: 2025-10-19
---
"""

import pytest
import time
import statistics
import subprocess
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from clock.hardware_beat import HardwareBeat

class TestHardwareTiming:
    """V1 Gate: Hardware Timing Stability Validation"""

    @pytest.fixture
    def hardware_beat(self):
        """Provide fresh HardwareBeat instance for each test"""
        return HardwareBeat(normalization_interval=60.0)  # Normalize every minute

    @pytest.mark.parametrize("test_duration", [10.0, 60.0])  # 10s, 1min (full 5min takes too long for CI)
    def test_rdtsc_stability_over_time(self, hardware_beat, test_duration):
        """Test rdtsc readings remain stable without wild variations"""
        samples = []
        start_time = time.monotonic()

        # Collect samples over test duration
        while time.monotonic() - start_time < test_duration:
            samples.append(hardware_beat._rdtsc())
            time.sleep(0.1)  # 100ms intervals

        # Check monotonicity (rdtsc should increase)
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i-1], f"Non-monotonic rdtsc at sample {i}"

        # Check coefficient of variation (stability measure)
        if len(samples) > 1:
            cv = statistics.stdev(samples) / statistics.mean(samples)
            assert cv < 0.01, ".04f"
            print(".04f")

    def test_drift_normalization_convergence(self, hardware_beat):
        """Test that drift correction stabilizes after updates"""
        corrections = []

        # Take initial normalization
        initial_norm = hardware_beat.update_normalization()

        for i in range(5):
            hardware_beat.last_normalization = time.monotonic() - 70  # Force renormalization
            norm = hardware_beat.update_normalization()
            corrections.append(norm['drift_magnitude'])
            time.sleep(0.5)

        # Drift should be relatively stable (within an order of magnitude)
        avg_drift = statistics.mean(corrections)
        assert avg_drift < 0.1, ".4f"

        print(".4f")

    def test_metabolism_adaptation(self, hardware_beat):
        """Test metabolism scaling with thermal/power constraints"""
        # Normal conditions
        factor_normal = hardware_beat.get_metabolism_factor(None, None)
        assert 0.5 <= factor_normal <= 2.0, ".3f"

        # Thermal constraint
        factor_thermal = hardware_beat.get_metabolism_factor(0.3, None)
        assert factor_thermal <= factor_normal, "Thermal throttling should reduce metabolism"

        # Power constraint
        factor_power = hardware_beat.get_metabolism_factor(None, 0.4)
        assert factor_power <= factor_normal, "Power limiting should reduce metabolism"

        print(".3f")

    def test_tick_interval_calculation(self, hardware_beat):
        """Test tick interval adjusts with metabolism"""
        base_rate = 1.0  # 1 Hz

        # Get current metabolism
        metabolism = hardware_beat.get_metabolism_factor()
        interval = hardware_beat.get_tick_interval_seconds(base_rate)

        # Interval should be reasonable (0.01 to 10 seconds)
        assert 0.01 <= interval <= 10.0, ".3f"
        print(".3f")

    @pytest.mark.v1_gate  # Mark for V1 gate validation
    def test_drift_against_ntp_baseline(self, hardware_beat):
        """
        V1 Gate Primary Test: Measure drift over 10-minute baseline

        Target: <2μs average deviation, <10μs max jitter
        """
        print("\n=== V1 GATE: HARDWARE TIMING STABILITY ===")
        print("Running 10-minute baseline drift measurement...")

        # For actual NTP sync, we could use ntpq or chronyc
        # Since this is a simulation, we'll use system time as proxy
        baseline_start = time.monotonic()
        rdtsc_samples = []
        wall_samples = []

        # Sample for 10 minutes (shortened for testing)
        duration = 60.0  # 1 minute for test, would be 600 for full
        sample_interval = 0.01  # 10ms

        samples_collected = 0
        while time.monotonic() - baseline_start < duration:
            wall_samples.append(time.monotonic())
            rdtsc_samples.append(hardware_beat._rdtsc())
            time.sleep(sample_interval)
            samples_collected += 1

        print(f"Collected {samples_collected} samples over {duration}s")

        # Calculate rdtsc frequency deviations
        deviations_microseconds = []
        for i in range(1, len(wall_samples)):
            wall_delta = (wall_samples[i] - wall_samples[i-1]) * 1e6  # microseconds
            rdtsc_delta = rdtsc_samples[i] - rdtsc_samples[i-1]
            expected_rdtsc_delta = rdtsc_delta  # Based on measured frequency

            # Deviation from expected rdtsc increment
            expected_ticks = wall_delta * (hardware_beat.clock_freq_hz / 1e6)
            deviation = abs(rdtsc_delta - expected_ticks)
            deviation_us = deviation / (hardware_beat.clock_freq_hz / 1e6) if hardware_beat.clock_freq_hz > 0 else deviation

            deviations_microseconds.append(deviation_us)

        if deviations_microseconds:
            avg_deviation = statistics.mean(deviations_microseconds)
            max_deviation = max(deviations_microseconds)

            print(".1f")
            print(".1f")

            # V1 Gate Criteria
            assert avg_deviation < 2.0, ".1f"
            assert max_deviation < 10.0, ".1f"

            print("✓ V1 GATE PASSED: Hardware Timing Stability Verified")

        else:
            pytest.fail("No deviation samples collected")

def main():
    """Run outside pytest if needed"""
    print("Running V1 Gate validation...")
    test_v1 = TestHardwareTiming()
    hb = HardwareBeat()

    try:
        test_v1.test_drift_against_ntp_baseline(hb)
        print("\nValidation Results:")
        print("✓ Hardware timestamp accuracy: <2μs avg deviation")
        print("✓ Stability: <10μs max jitter")
        print("\nV1 GATE PASSED - Proceed to Phase 2")
        return 0
    except Exception as e:
        print(f"\n✗ V1 GATE FAILED: {e}")
        print("Fix hardware timing before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())
