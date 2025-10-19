#!/bin/bash
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
