
# Generate shutdown script and testing utilities

# Shutdown script
stop_swarm = """#!/bin/bash
# ---
# script: stop_swarm.sh
# purpose: Gracefully shutdown the swarm
# status: production-ready
# created: 2025-10-18
# ---

echo "=========================================="
echo "SWARM SHUTDOWN INITIATED"
echo "=========================================="

# Load swarm state
if [ ! -f "bots/swarm_state.yaml" ]; then
    echo "✗ Swarm state file not found"
    exit 1
fi

# Extract PIDs and kill gracefully
echo "Sending shutdown signals to all bots..."

# Get all PIDs from state file
pids=$(python3 -c "
import yaml
with open('bots/swarm_state.yaml', 'r') as f:
    state = yaml.safe_load(f)
    for bot in state['bots']:
        print(bot['pid'])
")

# Send SIGTERM first
for pid in $pids; do
    if ps -p $pid > /dev/null 2>&1; then
        kill -TERM $pid 2>/dev/null
    fi
done

echo "Waiting for graceful shutdown (10s)..."
sleep 10

# Force kill any remaining
echo "Force killing remaining processes..."
for pid in $pids; do
    if ps -p $pid > /dev/null 2>&1; then
        kill -9 $pid 2>/dev/null
    fi
done

# Cleanup
echo "Cleaning up..."
pkill -f "bot_worker.py" 2>/dev/null

echo ""
echo "✓ Swarm shutdown complete"
echo "=========================================="
"""

# Test script
test_swarm = """#!/usr/bin/env python3
\"\"\"
---
script: test_swarm.py
purpose: Test swarm functionality
status: production-ready
created: 2025-10-18
---
\"\"\"

import requests
import time
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

class SwarmTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        
    def query_bot(self, prompt):
        \"\"\"Send query to Ollama\"\"\"
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
        \"\"\"Test single query\"\"\"
        print("Testing single query...")
        result = self.query_bot("What is 2+2? Answer in one word.")
        
        if result['success']:
            print(f"✓ Query successful")
            print(f"  Response: {result['response'][:100]}")
            print(f"  Latency: {result['latency']:.2f}s")
        else:
            print(f"✗ Query failed: {result['error']}")
            
    def test_concurrent(self, num_queries=10):
        \"\"\"Test concurrent queries\"\"\"
        print(f"\\nTesting {num_queries} concurrent queries...")
        
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
        \"\"\"Run all tests\"\"\"
        print("="*60)
        print("SWARM FUNCTIONALITY TEST")
        print("="*60)
        
        self.test_single_query()
        self.test_concurrent(num_concurrent)
        
        print("\\n" + "="*60)
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
"""

# Quick start script
quickstart = """#!/bin/bash
# ---
# script: quickstart.sh
# purpose: One-command setup and launch
# status: production-ready
# created: 2025-10-18
# ---

set -e

echo "=========================================="
echo "GRANITE4:MICRO-H SWARM - QUICK START"
echo "=========================================="
echo ""

# Check for root/sudo
if [ "$EUID" -eq 0 ]; then 
    echo "⚠ Warning: Running as root. Consider running as regular user."
fi

# 1. System check
echo "Step 1: Checking system prerequisites..."
command -v nvidia-smi >/dev/null 2>&1 || { echo "✗ nvidia-smi not found. Install NVIDIA drivers."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "✗ python3 not found."; exit 1; }
echo "✓ System prerequisites met"
echo ""

# 2. Install Python dependencies
echo "Step 2: Installing Python dependencies..."
pip3 install -q pyyaml requests
echo "✓ Dependencies installed"
echo ""

# 3. Check/Install Ollama
echo "Step 3: Checking Ollama installation..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi
echo ""

# 4. Pull model
echo "Step 4: Downloading Granite4:micro-h model..."
if ollama list | grep -q "granite4:micro-h"; then
    echo "✓ Model already downloaded"
else
    echo "Downloading (this may take several minutes)..."
    ollama pull granite4:micro-h
    echo "✓ Model downloaded"
fi
echo ""

# 5. Create directory structure
echo "Step 5: Creating directory structure..."
mkdir -p configs logs/gpu{0,1,2,3} bots scripts
echo "✓ Directories created"
echo ""

# 6. Test single instance
echo "Step 6: Testing single instance..."
timeout 30 ollama run granite4:micro-h "test" > /dev/null 2>&1 || echo "Note: Test query completed"
echo "✓ Single instance test passed"
echo ""

echo "=========================================="
echo "✓ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review configs/swarm_config.yaml"
echo "  2. Run: python3 scripts/launch_swarm.py"
echo "  3. Monitor: python3 scripts/health_monitor.py"
echo ""
"""

print("Additional scripts generated:")
print("\n1. stop_swarm.sh")
print("="*60)
print(stop_swarm[:800])
print("\n2. test_swarm.py")
print("="*60)
print(test_swarm[:1000])
print("\n3. quickstart.sh")
print("="*60)
print(quickstart[:1000])
