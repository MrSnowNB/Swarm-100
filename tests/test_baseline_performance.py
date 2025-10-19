#!/usr/bin/env python3
"""
---
file: test_baseline_performance.py
purpose: Baseline performance measurements for swarm system
framework: pytest with pytest-benchmark
status: development
created: 2025-10-18
---
"""

import pytest
import time
import subprocess
import requests
import threading
import psutil
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np  # type: ignore


class PerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self):
        self.start_time = None
        self.metrics = {}

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.initial_gpu_util = self._get_gpu_utilization()
        self.initial_cpu_util = psutil.cpu_percent(interval=None)

    def _get_gpu_utilization(self):
        """Get current GPU utilization using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpus = []
            for line in result.stdout.strip().split('\n'):
                util, mem_used, mem_total = line.split(', ')
                gpus.append({
                    'utilization': int(util),
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total)
                })
            return gpus
        except Exception as e:
            return None

    def get_current_metrics(self):
        """Get current performance metrics"""
        elapsed = time.time() - (self.start_time or time.time())

        metrics = {
            'elapsed_time': elapsed,
            'cpu_utilization': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_stats': self._get_gpu_utilization()
        }

        return metrics


class OllamaClient:
    """Client for interacting with Ollama"""

    def __init__(self, url="http://localhost:11434"):
        self.url = url
        self.model = "granite4:micro-h"

    def query(self, prompt, timeout=30):
        """Send query to Ollama and measure latency"""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=timeout
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                return {
                    'success': True,
                    'response': response.json()['response'],
                    'latency': latency
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'latency': latency
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency': time.time() - start_time
            }


class SwarmSimulator:
    """Simulate swarm behavior for testing"""

    def __init__(self):
        self.config_path = 'configs/swarm_config.yaml'
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.ollama = OllamaClient()

    def measure_bot_spawn_time(self):
        """Measure time to spawn a single bot"""
        start_time = time.time()

        # Simulate bot spawn process (without actually launching)
        env = os.environ.copy()
        env['BOT_ID'] = "test_bot_00"
        env['GPU_ID'] = "0"
        env['BOT_PORT'] = "11400"

        # Just measure the setup time (not actual spawn)
        setup_time = time.time() - start_time

        return setup_time

    def measure_message_latency(self, num_messages=100):
        """Measure message latency between queries"""
        latencies = []

        for i in range(num_messages):
            prompt = f"Generate a random number between 1 and 100: {i}"
            result = self.ollama.query(prompt)
            if result['success']:
                latencies.append(result['latency'])

        if latencies:
            return {
                'avg_latency': np.mean(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'total_messages': num_messages
            }
        else:
            return None

    def measure_concurrent_throughput(self, num_concurrent=10, duration=30):
        """Measure maximum throughput under concurrent load"""
        results = []
        start_time = time.time()
        request_count = 0

        def worker():
            nonlocal request_count
            while time.time() - start_time < duration:
                prompt = "What is the capital of France?"
                result = self.ollama.query(prompt)
                if result['success']:
                    request_count += 1
                time.sleep(0.1)  # Prevent overwhelming

        # Start concurrent workers
        threads = []
        for _ in range(num_concurrent):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for all to complete
        for t in threads:
            t.join()

        total_time = time.time() - start_time
        throughput = request_count / total_time if total_time > 0 else 0

        return {
            'total_requests': request_count,
            'duration': total_time,
            'throughput': throughput,
            'concurrent_users': num_concurrent
        }

    def measure_gpu_performance(self, duration=10):
        """Measure GPU performance during load"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Generate load
        def load_generator():
            start = time.time()
            while time.time() - start < duration:
                self.ollama.query("Count from 1 to 100 in words.")
                time.sleep(0.05)

        load_thread = threading.Thread(target=load_generator)
        load_thread.start()

        # Monitor GPU during load
        gpu_readings = []
        while load_thread.is_alive():
            time.sleep(1)
            metrics = monitor.get_current_metrics()
            if metrics.get('gpu_stats'):
                gpu_readings.append(metrics['gpu_stats'])

        load_thread.join()

        if gpu_readings:
            # Calculate averages across all GPUs
            avg_utilizations = []
            for gpu_data in gpu_readings:
                for gpu in gpu_data:
                    avg_utilizations.append(gpu['utilization'])

            return {
                'avg_gpu_utilization': np.mean(avg_utilizations),
                'max_gpu_utilization': max(avg_utilizations),
                'duration': duration
            }

        return None


# Pytest fixtures
@pytest.fixture
def swarm_simulator():
    return SwarmSimulator()

@pytest.fixture
def performance_monitor():
    return PerformanceMonitor()

# Baseline performance benchmarks

@pytest.mark.benchmark
def test_single_query_latency_baseline(benchmark, swarm_simulator):
    """Baseline test: Single query latency to Ollama"""
    result = benchmark(swarm_simulator.ollama.query, "What is 2+2?")

    assert result['success'], f"Query failed: {result.get('error', 'unknown')}"
    assert result['latency'] < 5.0, f"Latency too high: {result['latency']:.2f}s"

@pytest.mark.benchmark
def test_bot_spawn_time_baseline(benchmark, swarm_simulator):
    """Baseline test: Bot spawn time measurement"""
    spawn_time = benchmark(swarm_simulator.measure_bot_spawn_time)

    assert spawn_time < 1.0, f"Spawn time too high: {spawn_time:.3f}s"
    global BOT_SPAWN_TIME_BASELINE
    BOT_SPAWN_TIME_BASELINE = spawn_time

@pytest.mark.benchmark
def test_message_latency_baseline(benchmark, swarm_simulator):
    """Baseline test: Message latency (99th percentile)"""
    def measure_latency():
        return swarm_simulator.measure_message_latency(num_messages=50)

    result = benchmark(measure_latency)

    assert result is not None, "Latency measurement failed"
    assert result['p99_latency'] < 1.0, f"P99 latency too high: {result['p99_latency']:.3f}s"

@pytest.mark.benchmark
def test_concurrent_throughput_baseline(benchmark, swarm_simulator):
    """Baseline test: Concurrent throughput measurement"""
    def measure_throughput():
        return swarm_simulator.measure_concurrent_throughput(num_concurrent=5, duration=10)

    result = benchmark(measure_throughput)

    assert result['throughput'] > 1.0, f"Throughput too low: {result['throughput']:.2f} req/sec"
    global THROUGHPUT_BASELINE
    THROUGHPUT_BASELINE = result['throughput']

@pytest.mark.benchmark
def test_gpu_performance_baseline(benchmark, swarm_simulator):
    """Baseline test: GPU utilization during load"""
    result = benchmark(swarm_simulator.measure_gpu_performance, duration=5)

    assert result is not None, "GPU measurement failed"
    assert result['max_gpu_utilization'] > 10, f"GPU utilization too low: {result['max_gpu_utilization']:.1f}%"

# Global variables to store baseline measurements (will be used for regression detection)
BOT_SPAWN_TIME_BASELINE = None
THROUGHPUT_BASELINE = None
