#!/usr/bin/env python3
"""
---
script: test_swarm.py
purpose: Test swarm functionality
status: production-ready
created: 2025-10-18
---
"""

import requests
import time
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

class SwarmTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"

    def query_bot(self, prompt):
        """Send query to Ollama"""
        try:
            start = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "granite4:micro-h",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                return {
                    'success': True,
                    'response': response.json()['response'],
                    'latency': elapsed
                }
            else:
                return {'success': False, 'error': response.status_code}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_single_query(self):
        """Test single query"""
        print("Testing single query...")
        result = self.query_bot("What is 2+2? Answer in one word.")

        if result['success']:
            print(f"✓ Query successful")
            print(f"  Response: {result['response'][:100]}")
            print(f"  Latency: {result['latency']:.2f}s")
        else:
            print(f"✗ Query failed: {result['error']}")

    def test_concurrent(self, num_queries=10):
        """Test concurrent queries"""
        print(f"\nTesting {num_queries} concurrent queries...")

        prompts = [f"Count to {i}" for i in range(1, num_queries+1)]

        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=num_queries) as executor:
            futures = [executor.submit(self.query_bot, p) for p in prompts]

            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start_time

        successes = sum(1 for r in results if r.get('success'))
        avg_latency = sum(r.get('latency', 0) for r in results if r.get('success')) / max(successes, 1)

        print(f"✓ Concurrent test complete")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Success rate: {successes}/{num_queries}")
        print(f"  Avg latency: {avg_latency:.2f}s")
        print(f"  Throughput: {num_queries/elapsed:.2f} queries/sec")

    def run(self, num_concurrent=10):
        """Run all tests"""
        print("="*60)
        print("SWARM FUNCTIONALITY TEST")
        print("="*60)

        self.test_single_query()
        self.test_concurrent(num_concurrent)

        print("\n" + "="*60)
        print("✓ Testing complete")
        print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='Single query to test')
    parser.add_argument('--bots', type=int, default=10,
                       help='Number of concurrent queries')

    args = parser.parse_args()

    tester = SwarmTester()

    if args.query:
        result = tester.query_bot(args.query)
        print(result)
    else:
        tester.run(args.bots)
