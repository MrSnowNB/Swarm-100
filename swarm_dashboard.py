#!/usr/bin/env python3
"""
---
script: swarm_dashboard.py
purpose: Conway LoRA Visualization Dashboard - Zombie Swarm Web Portal
status: development
created: 2025-10-21
---
"""

import os
import time
import json
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
import requests
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for web
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

class SwarmDashboard:
    """Swarm-100 CyberGrid Conway Visualization Dashboard"""

    def __init__(self):
        self.grid_size = 10
        self.grid_data = np.zeros((10, 10), dtype=int)  # Conway binary states
        self.energy_matrix = np.zeros((10, 10))         # LoRA continuous energy
        self.zombie_states = np.zeros((10, 10))         # Bot alive/zombie status
        self.last_update = datetime.now()

        # Mock data for initial testing (replace with real bot polling)
        self.initialize_mock_data()

    def initialize_mock_data(self):
        """Initialize with some mock Conway patterns for demonstration"""
        # Add a glider in top-left
        self.grid_data[1:4, 1:4] = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]

        # Add some energy pulses
        self.energy_matrix[2, 2] = 1.0
        self.energy_matrix[5, 5] = 0.7
        self.energy_matrix[8, 3] = 0.5

        # Mark some bots as alive
        self.zombie_states.fill(1)  # All alive initially

    def poll_swarm_bots(self):
        """Poll all 100 zombie bots for current state - REAL DATA FOR DIAGNOSTICS"""
        # Load swarm configuration
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                state = yaml.safe_load(f)
                bots = state['bots']  # Poll ALL 100 bots for full diagnostics
        except Exception as e:
            print(f"Failed to load swarm state: {e}")
            return

        # Reset zombie states (assume alive by default)
        self.zombie_states.fill(1)

        # Poll each bot's /state endpoint concurrently
        updated_bots = 0
        total_bots = len(bots)

        for bot in bots:
            try:
                url = f"http://localhost:{bot['port']}/state"
                response = requests.get(url, timeout=2)

                if response.status_code == 200:
                    state_data = response.json()

                    # Extract position and REAL bot data for 10x10 grid
                    x, y = bot['grid_x'], bot['grid_y']
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        # Extract real memory vectors and compute grid state
                        memory_entries = state_data.get('memory_entries', 0)
                        vectors = state_data.get('vectors', [0] * 512)

                        # Conway state: based on memory activity (alive if recent activity)
                        memory_activity = sum(abs(v) for v in vectors[:10]) / 10
                        self.grid_data[y, x] = 1 if memory_activity > 0.01 else 0

                        # Energy level: based on vector magnitude and memory entries
                        vector_magnitude = np.linalg.norm(vectors[:50]) / 50
                        energy_level = min(1.0, (vector_magnitude * memory_entries) / 1000)
                        self.energy_matrix[y, x] = energy_level

                        updated_bots += 1

                else:
                    # Bot is dead/zombie if endpoint not responding
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.zombie_states[y, x] = 0
                        self.grid_data[y, x] = 0
                        self.energy_matrix[y, x] = 0.0

            except Exception as e:
                # Mark as zombie if unreachable
                if 0 <= bot['grid_x'] < self.grid_size and 0 <= bot['grid_y'] < self.grid_size:
                    self.zombie_states[bot['grid_y'], bot['grid_x']] = 0
                    self.grid_data[bot['grid_y'], bot['grid_x']] = 0
                    self.energy_matrix[bot['grid_y'], bot['grid_x']] = 0.0
                continue

        self.last_update = datetime.now()
        alive_bots = int(self.zombie_states.sum())
        active_cells = int(self.grid_data.sum())
        avg_energy = float(self.energy_matrix.mean())

        print(f"🔍 DIAGNOSTIC POLL: {updated_bots}/{total_bots} bots responding, {alive_bots}/100 alive, {active_cells}/100 Conway cells, {avg_energy:.3f} avg energy")

    def apply_conway_rules(self):
        """Apply Conway's Game of Life rules with LoRA energy influence"""
        new_grid = self.grid_data.copy()

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Count living neighbors
                neighbors = [
                    ((x+dx) % self.grid_size, (y+dy) % self.grid_size)
                    for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                    if not (dx == 0 and dy == 0)
                ]

                alive_neighbors = sum(self.grid_data[ny, nx] for nx, ny in neighbors)
                current_energy = self.energy_matrix[y, x]

                # Conway rules with LoRA energy boost
                if self.grid_data[y, x] == 1:
                    # Survival: needs 2-3 neighbors, boosted by energy
                    energy_boost = int(current_energy > 0.5)
                    if alive_neighbors < 2 + energy_boost or alive_neighbors > 3:
                        new_grid[y, x] = 0
                else:
                    # Birth: exactly 3 neighbors, or 2 with high energy
                    if alive_neighbors == 3 or (alive_neighbors == 2 and current_energy > 0.7):
                        new_grid[y, x] = 1

        self.grid_data = new_grid

    def generate_conway_plot(self):
        """Generate matplotlib plot of Conway grid with LoRA overlays"""
        fig, (ax_conway, ax_energy) = plt.subplots(1, 2, figsize=(12, 6))

        # Conway binary grid
        im_conway = ax_conway.imshow(self.grid_data, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_conway.set_title('Conway\'s Game of Life\nBinary States')
        ax_conway.set_xlabel('X Grid Position')
        ax_conway.set_ylabel('Y Grid Position')
        ax_conway.grid(True, alpha=0.3, color='gray')

        # Add grid lines and labels
        ax_conway.set_xticks(range(10))
        ax_conway.set_yticks(range(10))
        ax_conway.set_xticklabels(range(10))
        ax_conway.set_yticklabels(range(10))

        # Mark zombie dead bots
        for y in range(10):
            for x in range(10):
                if self.zombie_states[y, x] == 0:
                    ax_conway.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                                    fill=False, edgecolor='red', linewidth=2))

        # LoRA energy field
        colors = [(0.0, 0.0, 1.0, 0.3),   # Blue (low energy)
                 (0.5, 0.5, 1.0, 0.5),   # Light blue
                 (1.0, 0.5, 0.0, 0.7),   # Orange (medium)
                 (1.0, 0.0, 0.0, 1.0)]   # Red (high energy)
        import matplotlib.colors as mcolors
        energy_cmap = mcolors.LinearSegmentedColormap.from_list('energy', colors, N=256)
        im_energy = ax_energy.imshow(self.energy_matrix, cmap=energy_cmap, vmin=0, vmax=1, interpolation='bilinear')
        ax_energy.set_title('LoRA Energy Propagation\nContinuous Field')
        ax_energy.set_xlabel('X Grid Position')
        ax_energy.set_ylabel('Y Grid Position')

        # Add colorbar for energy
        cbar = plt.colorbar(im_energy, ax=ax_energy, shrink=0.8)
        cbar.set_label('Energy Level (0-1)')

        plt.tight_layout()

        # Convert to base64 for web display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f"data:image/png;base64,{image_base64}"

# Global dashboard instance
dashboard = SwarmDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/grid/state')
def get_grid_state():
    """API endpoint returning current grid state"""
    return jsonify({
        'grid_size': 10,
        'conway_states': dashboard.grid_data.tolist(),
        'energy_matrix': dashboard.energy_matrix.tolist(),
        'zombie_states': dashboard.zombie_states.tolist(),
        'last_update': dashboard.last_update.isoformat(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/grid/update')
def update_grid():
    """Force grid update and return plots"""
    dashboard.poll_swarm_bots()
    dashboard.apply_conway_rules()
    plot_url = dashboard.generate_conway_plot()

    return jsonify({
        'success': True,
        'plot_url': plot_url,
        'metrics': {
            'active_bots': int(dashboard.zombie_states.sum()),
            'avg_energy': float(dashboard.energy_matrix.mean()),
            'alive_cells': int(dashboard.grid_data.sum())
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/pulse/inject', methods=['POST'])
def inject_pulse():
    """Inject LoRA pulse at specified coordinates"""
    data = request.json
    x, y, energy = data.get('x', 5), data.get('y', 5), data.get('energy', 1.0)

    if 0 <= x < 10 and 0 <= y < 10:
        dashboard.energy_matrix[y, x] = min(1.0, energy)
        dashboard.grid_data[y, x] = 1  # Activate cell for pulse

        # Propagate pulse to neighbors (simple diffusion)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 10 and 0 <= ny < 10:
                    distance = abs(dx) + abs(dy)
                    attenuation = energy * max(0, 1.0 - distance * 0.3)
                    dashboard.energy_matrix[ny, nx] = min(1.0,
                        dashboard.energy_matrix[ny, nx] + attenuation * 0.5)

        dashboard.last_update = datetime.now()

        plot_url = dashboard.generate_conway_plot()

        return jsonify({
            'success': True,
            'message': f'Pulse injected at ({x},{y}) with E={energy}',
            'plot_url': plot_url,
            'affected_cells': 9  # 3x3 area
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid coordinates'})

@app.route('/api/grid/reset')
def reset_grid():
    """Reset grid to initial state"""
    dashboard.initialize_mock_data()
    plot_url = dashboard.generate_conway_plot()

    return jsonify({
        'success': True,
        'message': 'Grid reset to initial patterns',
        'plot_url': plot_url
    })

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>S.W.A.R.M. - Conway LoRA Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin: 0; }
        .header p { color: #7f8c8d; margin: 5px 0; }
        .grid-section { margin: 20px 0; padding: 20px; border: 1px solid #ecf0f1; border-radius: 5px; }
        .controls { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .control-group { background: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef; }
        .button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px; }
        .button:hover { background: #2980b9; }
        .button.danger { background: #e74c3c; }
        .button.danger:hover { background: #c0392b; }
        .status { margin: 10px 0; padding: 10px; background: #e8f5e8; border: 1px solid #d4edda; border-radius: 5px; color: #155724; }
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .footer { margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧟 S.W.A.R.M. - Conway LoRA Dashboard</h1>
            <p>Zombie Swarm Intelligence with Cellular Automaton Visualization</p>
            <div id="status" class="status">Loading dashboard...</div>
        </div>

        <div class="controls">
            <div class="control-group">
                <h3>🔄 Grid Controls</h3>
                <button class="button" onclick="updateGrid()">Refresh Data</button>
                <button class="button" onclick="resetGrid()">Reset Grid</button>
            </div>

            <div class="control-group">
                <h3>⚡ LoRA Pulse Injection</h3>
                <label>X: <input type="number" id="pulseX" value="5" min="0" max="9"></label>
                <label>Y: <input type="number" id="pulseY" value="5" min="0" max="9"></label>
                <label>Energy: <input type="range" id="pulseEnergy" min="0.1" max="1.0" step="0.1" value="1.0">
                    <span id="energyValue">1.0</span></label>
                </br>
                <button class="button" onclick="injectPulse()">Inject Pulse</button>
            </div>
        </div>

        <div class="grid-section">
            <h2>🧬 CyberGrid Conway State</h2>
            <div id="plot-container" class="plot-container">
                <p>Click "Refresh Data" to load current swarm state...</p>
            </div>
        </div>

        <div class="footer">
            <p>Swarm-100 • Conway LoRA CyberGrid Visualization • Real-time Bot Monitoring • AI-First Architecture</p>
        </div>
    </div>

    <script>
        let currentPlotUrl = null;

        function updateStatus(message, isError = false) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.style.background = isError ? '#f8d7da' : '#e8f5e8';
            statusDiv.style.color = isError ? '#721c24' : '#155724';
            statusDiv.style.borderColor = isError ? '#f5c6cb' : '#d4edda';
        }

        function updateEnergyValue() {
            document.getElementById('energyValue').textContent = document.getElementById('pulseEnergy').value;
        }

        // Initialize energy slider
        document.getElementById('pulseEnergy').addEventListener('input', updateEnergyValue);
        updateEnergyValue();

        async function updateGrid() {
            updateStatus('🔄 Updating grid data from swarm bots...');

            try {
                const response = await fetch('/api/grid/update');
                const data = await response.json();

                if (data.success) {
                    currentPlotUrl = data.plot_url;
                    document.getElementById('plot-container').innerHTML =
                        `<img src="${currentPlotUrl}" alt="Conway Grid Visualization" />`;
                    updateStatus(`✅ Updated: ${data.metrics.active_bots}/100 bots active, ${data.metrics.alive_cells} Conway cells alive, ${data.metrics.avg_energy.toFixed(2)} avg energy`);
                } else {
                    throw new Error('Update failed');
                }
            } catch (error) {
                updateStatus('❌ Failed to update grid data', true);
                console.error('Update error:', error);
            }
        }

        async function resetGrid() {
            updateStatus('🔄 Resetting grid...');

            try {
                const response = await fetch('/api/grid/reset');
                const data = await response.json();

                if (data.success) {
                    document.getElementById('plot-container').innerHTML =
                        `<img src="${data.plot_url}" alt="Conway Grid Visualization" />`;
                    updateStatus('✅ Grid reset to initial patterns');
                } else {
                    throw new Error('Reset failed');
                }
            } catch (error) {
                updateStatus('❌ Failed to reset grid', true);
                console.error('Reset error:', error);
            }
        }

        async function injectPulse() {
            const x = parseInt(document.getElementById('pulseX').value);
            const y = parseInt(document.getElementById('pulseY').value);
            const energy = parseFloat(document.getElementById('pulseEnergy').value);

            updateStatus(`⚡ Injecting pulse at (${x},${y}) with E=${energy}...`);

            try {
                const response = await fetch('/api/pulse/inject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ x: x, y: y, energy: energy })
                });

                const data = await response.json();

                if (data.success) {
                    document.getElementById('plot-container').innerHTML =
                        `<img src="${data.plot_url}" alt="Conway Grid Visualization" />`;
                    updateStatus(data.message);
                } else {
                    throw new Error(data.error || 'Injection failed');
                }
            } catch (error) {
                updateStatus('❌ Failed to inject pulse', true);
                console.error('Injection error:', error);
            }
        }

        // Auto-update every 60 seconds
        setInterval(() => {
            if (!document.querySelector('.button:hover')) { // Don't auto-update while hovering buttons
                updateGrid();
            }
        }, 60000);

        // Load initial data
        updateStatus('🎯 Ready for zombie swarm visualization');
    </script>
</body>
</html>
"""

def background_grid_updater():
    """Background thread to update grid state periodically"""
    while True:
        try:
            time.sleep(30)  # Update every 30 seconds
            dashboard.poll_swarm_bots()
            dashboard.apply_conway_rules()
        except Exception as e:
            print(f"Background update error: {e}")

if __name__ == '__main__':
    print("🚀 Starting S.W.A.R.M. Conway LoRA Dashboard...")
    print("Navigate to http://localhost:5000 for the web interface")
    print("Features:")
    print("- Real-time Conway grid visualization")
    print("- LoRA energy propagation display")
    print("- Interactive pulse injection controls")
    print("- Zombie bot health monitoring")

    # Start background updater
    background_thread = threading.Thread(target=background_grid_updater, daemon=True)
    background_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
