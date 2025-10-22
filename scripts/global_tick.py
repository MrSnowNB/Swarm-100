#!/usr/bin/env python3
"""
---
script: global_tick.py
purpose: Global clock synchronization for CA-based swarm updates
status: development
created: 2025-10-19
---
"""

import time
import logging
import socketio
import threading
import signal
import sys
from typing import Dict, Any
import requests
import yaml

logger = logging.getLogger('GlobalTick')

class CyberGridManager:
    """Persistent manager for CyberGrid state across ticks"""
    _instance = None

    @classmethod
    def get_grid(cls):
        """Get or create persistent CyberGrid instance"""
        if cls._instance is None:
            try:
                import swarm_core as sc
                cls._instance = sc.CyberGrid()
                logger.info("CyberGrid instance created and ready for CA operations")
            except ImportError:
                logger.error("swarm_core not available - cannot create CyberGrid")
                return None
        return cls._instance

    @classmethod
    def reset_grid(cls):
        """Reset grid state for new experiments"""
        cls._instance = None

class GlobalTickCoordinator:
    """
    Coordinator for global tick synchronization across all swarm bots.

    Emits periodic tick events via WebSocket/Socket.IO to trigger CA transitions.
    Runs as a separate process to maintain timing independence.
    """

    def __init__(self, tick_interval_ms: int = 1000, port: int = 5001):
        """Initialize global tick coordinator"""
        self.tick_interval_ms = tick_interval_ms
        self.port = port
        self.tick_count = 0
        self.running = False

        self.sio = socketio.Client()

        # State for metrics
        self.last_ca_update = 0
        self.bots_connected = 0
        self.last_heartbeat = time.time()

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Handle graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_socketio_handlers(self):
        """Configure Socket.IO event handlers"""
        @self.sio.event
        def connect():
            logger.info("Connected to dashboard server")
            self.sio.emit('register_service', {'service': 'global_tick'})

        @self.sio.event
        def disconnect():
            logger.info("Disconnected from dashboard server")

        @self.sio.event
        def heartbeat(data):
            """Receive heartbeat from dashboard"""
            self.last_heartbeat = time.time()
            logger.debug(f"Heartbeat: {data}")

        @self.sio.event
        def status_request():
            """Respond to status requests from dashboard"""
            status = {
                'service': 'global_tick',
                'tick_count': self.tick_count,
                'tick_interval_ms': self.tick_interval_ms,
                'last_ca_update': self.last_ca_update,
                'bots_connected': self.bots_connected,
                'status': 'active' if self.running else 'stopped'
            }
            self.sio.emit('status_response', status)

    def _tick_loop(self):
        """Main tick generation loop"""
        logger.info(f"Starting global tick loop with {self.tick_interval_ms}ms interval")

        while self.running:
            start_time = time.time()

            # Emit global tick event
            self.emit_tick()

            # Apply rules locally (call rule_engine)
            self.apply_ca_update()

            # Wait for next tick, accounting for processing time
            processing_time = time.time() - start_time
            wait_time = max(0.001, (self.tick_interval_ms / 1000.0) - processing_time)

            time.sleep(wait_time)

    def emit_tick(self):
        """Emit global tick event to all connected clients"""
        self.tick_count += 1

        tick_data = {
            'tick': self.tick_count,
            'timestamp': time.time(),
            'tick_interval_ms': self.tick_interval_ms
        }

        # Emit via Socket.IO if connected
        if self.sio.connected:
            self.sio.emit('global_tick', tick_data)

        # Also broadcast tick to all bot ports directly (fallback)
        self._notify_bots_directly(tick_data)

        logger.debug(f"Global tick {self.tick_count} emitted")

    def _notify_bots_directly(self, tick_data: Dict[str, Any]):
        """Directly notify running bots via REST (fallback to Socket.IO primary)"""
        # Load current swarm state to get bot ports
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                swarm_state = yaml.safe_load(f)

            if swarm_state and isinstance(swarm_state, dict) and 'bots' in swarm_state:
                for bot in swarm_state['bots']:
                    bot_port = bot.get('port')
                    if bot_port is None:
                        continue
                    try:
                        # Send tick notification via HTTP POST
                        requests.post(f"http://localhost:{bot_port}/tick",
                                    json=tick_data,
                                    timeout=0.1)  # Short timeout
                    except Exception as e:
                        logger.debug(f"Failed to notify bot on port {bot_port}: {e}")

        except FileNotFoundError:
            logger.warning("Swarm state not found - no direct bot notifications")

    def apply_ca_update(self):
        """Apply CA rules update via rule engine and CyberGrid"""
        try:
            # First apply Rule Engine (bot state vectors)
            import subprocess
            result = subprocess.run([
                sys.executable, 'scripts/rule_engine.py'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                logger.debug("Bot state vector CA rules applied successfully")
            else:
                logger.warning(f"Bot state vector CA update failed: {result.stderr}")

            # Then apply CyberGrid Conway CA step
            try:
                grid = CyberGridManager.get_grid()
                if grid is not None:
                    grid.step()  # Advance Conway CA one generation
                    self.last_ca_update = time.time()
                    logger.debug("CyberGrid Conway CA step completed")
                else:
                    logger.warning("CyberGrid not initialized - cannot step")

            except Exception as e:
                logger.error(f"Error stepping CyberGrid: {e}")

        except Exception as e:
            logger.error(f"Error applying CA rules: {e}")

    def connect_to_dashboard(self):
        """Establish connection to dashboard Socket.IO server"""
        try:
            # Assume dashboard runs on same machine, port 5000
            self.sio.connect('http://localhost:5000')
            logger.info("Connected to dashboard for broadcasting")
        except Exception as e:
            logger.warning(f"Could not connect to dashboard: {e}")

    def start(self):
        """Start the global tick coordinator"""
        logger.info("="*50)
        logger.info("GLOBAL TICK COORDINATOR STARTING")
        logger.info("="*50)
        logger.info(f"Tick interval: {self.tick_interval_ms}ms")
        logger.info(f"Dashboard port: {self.port}")

        self.running = True
        self._setup_socketio_handlers()

        # Connect to dashboard (optional)
        self.connect_to_dashboard()

        # Start tick loop in separate thread
        tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        tick_thread.start()

        logger.info("Global tick coordinator active. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)  # Keep main thread alive

                # Periodic heartbeat check
                if time.time() - self.last_heartbeat > 30:
                    logger.warning("Lost dashboard heartbeat - attempting reconnect")
                    self.connect_to_dashboard()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the coordinator gracefully"""
        logger.info("Stopping global tick coordinator...")
        self.running = False

        if self.sio.connected:
            self.sio.disconnect()

        logger.info("Global tick coordinator stopped.")

def main():
    """Entry point for global tick coordinator"""
    import argparse

    parser = argparse.ArgumentParser(description='Global Tick Coordinator for Swarm CA')
    parser.add_argument('--interval-ms', type=int, default=1000,
                       help='Tick interval in milliseconds (default: 1000)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Status port for coordinator (default: 5001)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                       format='%(asctime)s - GlobalTick - %(levelname)s - %(message)s')

    coordinator = GlobalTickCoordinator(args.interval_ms, args.port)
    coordinator.start()

if __name__ == '__main__':
    main()
