#!/usr/bin/env python3
"""
---
script: swarm_monitor.py
purpose: Web dashboard for real-time swarm monitoring
status: development
created: 2025-10-19
---
"""

import os
import time
import yaml
import psutil
from flask import Flask, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit
import subprocess

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def get_swarm_state():
    """Read current swarm state from YAML"""
    try:
        with open('bots/swarm_state.yaml', 'r') as f:
            state = yaml.safe_load(f)
        if not isinstance(state, dict):
            return {"error": "invalid yaml format: expected dict", "total_bots": 0, "bots": []}
        return state
    except Exception as e:
        return {"error": str(e), "total_bots": 0, "bots": []}

def get_gpu_usage():
    """Get GPU memory and utilization using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu = {
                    'id': int(parts[0].strip()) if len(parts) > 0 else 0,
                    'mem_used_mb': int(parts[1].strip()) if len(parts) > 1 else 0,
                    'mem_total_mb': int(parts[2].strip()) if len(parts) > 2 else 0,
                    'utilization_percent': int(parts[3].strip()) if len(parts) > 3 else 0,
                    'mem_used_gb': 0.0,
                    'mem_total_gb': 0.0
                }
                gpu['mem_used_gb'] = round(gpu['mem_used_mb'] / 1024, 1)
                gpu['mem_total_gb'] = round(gpu['mem_total_mb'] / 1024, 1)
                gpus.append(gpu)
        return gpus
    except Exception as e:
        return [{"error": str(e)}]

def count_processes():
    """Count swarm-related processes"""
    try:
        # Count bot workers
        bots = subprocess.run(
            ['pgrep', '-f', 'bot_worker'],
            capture_output=True, text=True
        )
        bot_count = len(bots.stdout.strip().split('\n')) if bots.stdout.strip() else 0

        # Count zombie supervisor
        supervisor = subprocess.run(
            ['pgrep', '-f', 'zombie_supervisor'],
            capture_output=True, text=True
        )
        supervisor_count = len(supervisor.stdout.strip().split('\n')) if supervisor.stdout.strip() else 0

        # Count launch processes
        launch = subprocess.run(
            ['pgrep', '-f', 'launch_swarm'],
            capture_output=True, text=True
        )
        launch_count = len(launch.stdout.strip().split('\n')) if launch.stdout.strip() else 0

        return {
            'bot_workers': bot_count,
            'zombie_supervisor': supervisor_count,
            'launcher': launch_count,
            'total_swarm_processes': bot_count + supervisor_count + launch_count
        }
    except Exception as e:
        return {"error": str(e)}

# HTML Template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemma3-Zombie-Swarm Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .status-section {
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .status-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .status-item {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .metric {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        .metric.label {
            color: #6c757d;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .danger { color: #dc3545; }
        .info { color: #17a2b8; }
        .last-update {
            text-align: center;
            color: #6c757d;
            font-size: 12px;
            margin-top: 20px;
        }
        #ca-grid div {
            min-height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧟 Gemma3-Zombie-Swarm Live Dashboard</h1>

        <div class="status-section">
            <div class="status-header">📊 Process Monitor</div>
            <div class="status-grid">
                <div class="status-item">
                    <div class="metric" id="bot-count">0</div>
                    <div class="metric label">Active Bot Workers</div>
                </div>
                <div class="status-item">
                    <div class="metric" id="supervisor-count">0</div>
                    <div class="metric label">Zombie Supervisors</div>
                </div>
                <div class="status-item">
                    <div class="metric" id="total-processes">0</div>
                    <div class="metric label">Total Swarm Processes</div>
                </div>
            </div>
        </div>

        <div class="status-section">
            <div class="status-header">💾 GPU Memory Usage</div>
            <table>
                <thead>
                    <tr>
                        <th>GPU</th>
                        <th>Memory Used</th>
                        <th>Memory Total</th>
                        <th>Utilization</th>
                    </tr>
                </thead>
                <tbody id="gpu-table">
                    <tr><td colspan="4">Loading GPU data...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="status-section">
            <div class="status-header">🤖 Bot Health Status</div>
            <div class="status-grid">
                <div class="status-item">
                    <div class="metric success" id="alive-bots">0</div>
                    <div class="metric label">Bots Alive</div>
                </div>
                <div class="status-item">
                    <div class="metric danger" id="dead-bots">0</div>
                    <div class="metric label">Bots Dead</div>
                </div>
                <div class="status-item">
                    <div class="metric warning" id="zombie-recoveries">0</div>
                    <div class="metric label">Zombie Recoveries</div>
                </div>
            </div>
        </div>

        <div class="status-section">
            <div class="status-header">🔳 Cellular Automata Grid (Tick: <span id="ca-tick">0</span>)</div>
            <div id="ca-grid" style="display: grid; grid-template-columns: repeat(var(--grid-width, 10), 1fr); gap: 1px; max-width: 800px; margin: 0 auto;">
                <!-- CA grid cells will be populated here -->
            </div>
        </div>

        <div class="status-section">
            <div class="status-header">📋 Recent Events</div>
            <div id="events-log" style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; height: 200px; overflow-y: auto;">
                Connecting to live event stream...<br>
            </div>
        </div>

        <div class="last-update">
            Last updated: <span id="timestamp">-</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();

        function updateDashboard(data) {
            // Process counts
            if (data.processes) {
                document.getElementById('bot-count').textContent = data.processes.bot_workers || 0;
                document.getElementById('supervisor-count').textContent = data.processes.zombie_supervisor || 0;
                document.getElementById('total-processes').textContent = data.processes.total_swarm_processes || 0;
            }

            // GPU usage
            if (data.gpus && data.gpus.length > 0) {
                const gpuRows = data.gpus.map(gpu => {
                    if (gpu.error) {
                        return `<tr><td colspan="4" class="danger">GPU Error: ${gpu.error}</td></tr>`;
                    }
                    const usedGB = gpu.mem_used_gb.toFixed(1);
                    const totalGB = gpu.mem_total_gb.toFixed(1);
                    const utilization = gpu.utilization_percent + '%';
                    return `<tr>
                        <td>GPU ${gpu.id}</td>
                        <td>${usedGB} GB</td>
                        <td>${totalGB} GB</td>
                        <td>${utilization}</td>
                    </tr>`;
                }).join('');
                document.getElementById('gpu-table').innerHTML = gpuRows;
            }

            // Bot status
            if (data.swarm) {
                document.getElementById('alive-bots').textContent = data.swarm.total_bots || 0;
                document.getElementById('dead-bots').textContent = 0;  // Will be updated when zombie events come
                document.getElementById('zombie-recoveries').textContent = 0;  // Will be updated when recovery events come
            }

            // CA Grid visualization
            if (data.ca_grid) {
                document.documentElement.style.setProperty('--grid-width', data.ca_grid.width);
                document.getElementById('ca-tick').textContent = data.ca_grid.tick;

                const gridElement = document.getElementById('ca-grid');
                const cellHTML = data.ca_grid.cells.map(cell => {
                    const hue = cell.alive ? 220 : 0;  // Blue for alive, red for dead
                    const lightness = 50 + (cell.intensity * 40);  // Adjust intensity
                    const style = `background-color: hsl(${hue}, 100%, ${lightness}%); font-size: 10px;`;
                    return `<div style="${style}" title="${cell.bot_id}">${cell.x},${cell.y}</div>`;
                }).join('');
                gridElement.innerHTML = cellHTML;
            }

            // Timestamp
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
        }

        function addEvent(event) {
            const eventsDiv = document.getElementById('events-log');
            const timestamp = new Date().toLocaleTimeString();
            eventsDiv.innerHTML += `[${timestamp}] ${event}<br>`;
            eventsDiv.scrollTop = eventsDiv.scrollHeight;
        }

        // Listen for dashboard updates
        socket.on('update', function(data) {
            updateDashboard(data);
        });

        // Listen for zombie events
        socket.on('zombie_event', function(data) {
            addEvent(`🧟 ${data.type}: ${data.message}`);
            // Update metrics if it's a recovery
            if (data.type === 'reborn') {
                const current = parseInt(document.getElementById('zombie-recoveries').textContent) || 0;
                document.getElementById('zombie-recoveries').textContent = current + 1;
            }
        });

        // Initial load
        socket.on('connect', function() {
            addEvent('Connected to swarm event stream ✅');
        });

        // Fetch initial data every 5 seconds
        function fetchInitialData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => {
                    console.error('Failed to fetch dashboard data:', error);
                    addEvent(`❌ Data fetch failed: ${error.message}`);
                });
        }

        // Start periodic updates
        setInterval(fetchInitialData, 2000);
        setTimeout(fetchInitialData, 1000); // Initial fetch
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Serve the main dashboard HTML"""
    return render_template_string(HTML_TEMPLATE)

def get_ca_grid():
    """Build CA grid data for visualization"""
    swarm_state = get_swarm_state()
    if not swarm_state or 'error' in swarm_state:
        return {'width': 10, 'height': 4, 'cells': [], 'tick': 0}

    width = swarm_state.get('grid_width', 10)
    height = swarm_state.get('grid_height', 4)
    bots = swarm_state.get('bots', [])
    tick = swarm_state.get('tick', 0)

    # Initialize grid with bots by position
    grid_bots = {}
    for bot in bots:
        x, y = bot.get('grid_x', 0), bot.get('grid_y', 0)
        grid_bots[(x, y)] = bot

    # Build cell data
    cells = []
    for y in range(height):
        for x in range(width):
            bot = grid_bots.get((x, y))
            if bot and 'state_magnitude' in bot:
                alive = True
                intensity = min(float(bot['state_magnitude']) / 10, 1.0)  # Normalize
                bot_id = bot['bot_id']
            else:
                alive = x*height + y < len(bots)  # Assume alive if within count
                intensity = 0.0
                bot_id = f"empt_{x}_{y}"

            cells.append({
                'x': x,
                'y': y,
                'alive': alive,
                'intensity': intensity,
                'bot_id': bot_id
            })

    return {
        'width': width,
        'height': height,
        'cells': cells,
        'tick': tick
    }

@app.route('/data')
def get_data():
    """Return JSON data for dashboard"""
    return jsonify({
        'swarm': get_swarm_state(),
        'gpus': get_gpu_usage(),
        'processes': count_processes(),
        'ca_grid': get_ca_grid(),
        'timestamp': time.time()
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected to dashboard')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from dashboard')

@socketio.on('global_tick')
def handle_global_tick(tick_data):
    """Handle CA tick updates from global coordinator"""
    print(f"Received global tick: {tick_data}")
    # Trigger dashboard update
    socketio.emit('update', {
        'swarm': get_swarm_state(),
        'gpus': get_gpu_usage(),
        'processes': count_processes(),
        'ca_grid': get_ca_grid(),
        'tick': tick_data
    })

# Function to trigger updates (called by external services)
@socketio.on('trigger_update')
def trigger_update():
    """Endpoint that supervisors can call to push updates"""
    socketio.emit('update', {
        'swarm': get_swarm_state(),
        'gpus': get_gpu_usage(),
        'processes': count_processes(),
        'triggered': True
    })

# Function to broadcast zombie events
def emit_zombie_event(event_type, message, details=None):
    """Function that supervisors can call to announce zombie events"""
    socketio.emit('zombie_event', {
        'type': event_type,
        'message': message,
        'details': details,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("🧟 Starting Gemma3-Zombie-Swarm Web Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("💡 Open multiple browser tabs to test real-time updates")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
