#!/usr/bin/env python3
"""
---
script: swarm_supervisor.py
purpose: Layer 1 Supervisor - Orchestrates Gemma3 swarm with Granite4
gate: Phase 2 implementation
status: development
created: 2025-10-19
---

Hierarchical Swarm Supervisor (Granite4 orchestrator)

Monitors Layer 0 (Gemma3 swarm) and provides strategic coordination:
- Performance monitoring (tick rate, bot survival, entropy dynamics)
- Anomaly detection (zombie waves, synchronization failures, parameter drift)
- Autonomous parameter adjustment (alpha, noise, diffusion rates)
- Human-readable status reports for Layer 2 interface

Research foundation: ArXiv 2508.12683, NVIDIA 2025, ArXiv 2509.05355
"""

import time
import json
import logging
import subprocess
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yaml

import requests  # For Ollama API calls
import pandas as pd

# Supervisor configuration
SUPERVISOR_CONFIG = {
    "ollama_endpoint": "http://localhost:11434",
    "model": "granite4:micro-h",
    "monitoring_interval": 5.0,  # seconds
    "anomaly_threshold": {
        "entropy_spike": 2.0,      # standard deviations
        "bot_death_rate": 0.1,     # per tick
        "tick_timeout": 2.0,       # seconds
        "param_drift": 0.2         # relative change
    },
    "adjustment_delay": 30.0,     # seconds after detection
    "human_approval_required": True  # for critical changes
}

class SwarmSupervisor:
    """
    Granite4-powered orchestrator for hierarchical swarm control.

    Phase 2: Layer 1 Supervisor implementation
    """

    def __init__(self):
        self.logger = logging.getLogger('SwarmSupervisor')
        self.config = SUPERVISOR_CONFIG

        # Monitoring state
        self.baseline_metrics = {}
        self.anomalies = []
        self.adjustments = []
        self.last_tick = 0
        self.running = False

        # Ollama process
        self.ollama_process = None
        self.ollama_ready = False

        # Signal handling
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.logger.info("Swarm Supervisor initialized")

    def _shutdown_handler(self, signal, frame):
        """Graceful shutdown"""
        self.logger.info("Shutdown signal received")
        self.stop_supervision()

    def start_ollama(self):
        """Ensure Ollama service is running with Granite4 model"""
        self.logger.info("Checking Ollama service...")

        # Check if Ollama is responding
        try:
            response = requests.get(f"{self.config['ollama_endpoint']}/api/tags",
                                  timeout=5, verify=False)
            if response.status_code == 200:
                self.ollama_ready = True
                self.logger.info("Ollama service ready")

                # Pull model if needed
                if not self._model_available():
                    self.logger.info("Pulling Granite4:micro-h model...")
                    result = requests.post(f"{self.config['ollama_endpoint']}/api/pull",
                                         json={"name": "granite4:micro-h"}, timeout=300)
                    if hasattr(result, 'status_code') and result.status_code == 200:
                        raise RuntimeError("Failed to pull Granite4 model")
                self.logger.info("Granite4 model ready")
                return
        except requests.RequestException:
            pass

        # Start Ollama if not running
        self.logger.info("Starting Ollama service...")
        try:
            self.ollama_process = subprocess.Popen(['ollama', 'serve'],
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL)
            # Wait for service to be ready
            for _ in range(60):  # 60 attempts, 5 seconds each
                try:
                    response = requests.get(f"{self.config['ollama_endpoint']}/api/tags",
                                          timeout=5, verify=False)
                    if response.status_code == 200:
                        self.ollama_ready = True
                        self.logger.info("Ollama started successfully")
                        # Pull model
                        self.logger.info("Pulling Granite4:micro-h model...")
                        result = requests.post(f"{self.config['ollama_endpoint']}/api/pull",
                                             json={"name": "granite4:micro-h"}, timeout=180)
                        if result.status_code == 200:
                            self.logger.info("Model pulled successfully")
                            return
                except requests.RequestException:
                    pass
                time.sleep(5)

            raise RuntimeError("Failed to start Ollama service")

        except FileNotFoundError:
            raise RuntimeError("Ollama not installed. Install from https://ollama.ai/")

    def _model_available(self) -> bool:
        """Check if Granite4 model is available"""
        try:
            response = requests.get(f"{self.config['ollama_endpoint']}/api/tags", verify=False)
            data: Dict[str, Any] = response.json()
            if isinstance(data, dict):
                models = data.get('models', [])
                if isinstance(models, list):
                    for m in models:
                        if isinstance(m, dict):
                            name = m.get('name', '')
                            if name.startswith('granite4:micro-h'):
                                return True
            return False
        except:
            return False

    def start_supervision(self):
        """Start the supervision loop"""
        self.logger.info("Starting swarm supervision...")

        # Ensure Ollama is ready
        self.start_ollama()

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info("Swarm supervisor active")

        # Keep running
        while self.running:
            time.sleep(1)

    def stop_supervision(self):
        """Stop supervision and cleanup"""
        self.logger.info("Stopping supervision...")
        self.running = False

        if self.ollama_process:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=10)
                self.logger.info("Ollama stopped")
            except:
                self.ollama_process.kill()
                self.ollama_process.wait()

        self.logger.info("Supervision stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Monitoring loop started")

        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_metrics()

                # Analyze anomalies
                anomalies = self._detect_anomalies(metrics)

                if anomalies:
                    self.logger.info(f"Detected {len(anomalies)} anomalies")
                    # Analyze with Granite4
                    analysis = self._analyze_with_llm(metrics, anomalies)
                    # Plan adjustments
                    adjustments = self._plan_adjustments(analysis)
                    # Apply adjustments
                    if adjustments:
                        self._apply_adjustments(adjustments)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

            time.sleep(self.config['monitoring_interval'])

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics from Layer 0"""
        metrics = {
            'timestamp': datetime.now(),
            'tick': None,
            'bot_count': 0,
            'active_bots': 0,
            'dead_bots': 0,
            'entropy': 0.0,
            'tick_rate': 0.0,
            'tick_time_delta': 0.0
        }

        # Read swarm state
        state_file = Path('bots/swarm_state.yaml')
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = yaml.safe_load(f)

                if isinstance(state, dict):
                    current_tick = state.get('tick', 0)
                    metrics['tick'] = current_tick

                    # Count active/dead bots
                    bots = state.get('bots', [])
                    metrics['bot_count'] = len(bots)
                    metrics['active_bots'] = sum(1 for b in bots if b.get('alive', True))
                    metrics['dead_bots'] = metrics['bot_count'] - metrics['active_bots']

                    # Calculate tick rate
                    if self.last_tick > 0:
                        tick_delta = current_tick - self.last_tick
                        time_delta = time.time() - getattr(self, 'last_tick_time', time.time())
                        metrics['tick_rate'] = tick_delta / time_delta if time_delta > 0 else 0
                        metrics['tick_time_delta'] = time_delta

                    self.last_tick = current_tick
                    setattr(self, 'last_tick_time', time.time())
                else:
                    self.logger.warning(f"Invalid state file format: expected dict, got {type(state)}")

            except Exception as e:
                self.logger.warning(f"Error reading swarm state: {e}")

        # Read entropy from metrics CSV
        metrics_file = Path('logs/experimentation/ca_metrics.csv')
        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                if len(df) > 10:
                    # Get recent entropy average - handle different column names
                    entropy_col = 'state_entropy' if 'state_entropy' in df.columns else 'mean_state_entropy'
                    recent_entropy = df[entropy_col].tail(10).mean()
                    metrics['entropy'] = recent_entropy
            except Exception as e:
                self.logger.warning(f"Error reading metrics: {e}")

        return metrics

    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[str]:
        """Detect anomalies in metrics"""
        anomalies = []

        # Establish baselines if not set
        for key in ['entropy', 'tick_rate']:
            if key not in self.baseline_metrics and metrics[key] is not None:
                self.baseline_metrics[key] = metrics[key]

        # Entropy spike detection
        if 'entropy' in self.baseline_metrics and metrics['entropy']:
            baseline = self.baseline_metrics['entropy']
            current = metrics['entropy']
            threshold = baseline * self.config['anomaly_threshold']['entropy_spike']

            if current > baseline + threshold:
                anomalies.append(f"Entropy spike: {current:.3f} > {baseline:.3f} + {threshold:.3f}")

        # High death rate
        if metrics['bot_count'] > 0:
            death_rate = metrics['dead_bots'] / metrics['bot_count']
            if death_rate > self.config['anomaly_threshold']['bot_death_rate']:
                anomalies.append(f"High death rate: {death_rate:.2f} > {self.config['anomaly_threshold']['bot_death_rate']}")

        # Tick timeout
        if metrics['tick_time_delta'] > self.config['anomaly_threshold']['tick_timeout']:
            anomalies.append(f"Tick timeout: {metrics['tick_time_delta']:.1f}s > {self.config['anomaly_threshold']['tick_timeout']}s")

        # Log anomalies
        if anomalies:
            self.anomalies.extend(anomalies)

        return anomalies

    def _analyze_with_llm(self, metrics: Dict[str, Any], anomalies: List[str]) -> str:
        """Use Granite4 to analyze system state and anomalies"""
        if not self.ollama_ready:
            return "Ollama not ready - skipping LLM analysis"

        try:
            analysis_prompt = f"""
You are Granite4, the Layer 1 Supervisor for a hierarchical multi-agent swarm system.

SYSTEM STATE:
- Tick: {metrics['tick']}
- Bots: {metrics['active_bots']}/{metrics['bot_count']} alive
- Entropy: {metrics['entropy']:.3f}
- Tick Rate: {metrics['tick_rate']:.2f}
- Tick Time Delta: {metrics['tick_time_delta']:.1f}s

DETECTED ANOMALIES:
{chr(10).join(f"- {a}" for a in anomalies)}

Based on research-validated swarm intelligence patterns:
1. Analyze what these anomalies indicate about the CA system health
2. Recommend parameter adjustments to restore stability
3. Suggest monitoring priorities for next 5 minutes

Be specific about which parameters to adjust and why.
Be conservative - suggest incremental changes.
"""

            response = requests.post(
                f"{self.config['ollama_endpoint']}/api/generate",
                json={"model": self.config['model'], "prompt": analysis_prompt, "stream": False},
                timeout=30, verify=False
            )

            if response.status_code == 200:
                result = response.json()['response']
                self.logger.info(f"LLM Analysis: {result[:200]}...")
                return result
            else:
                self.logger.error(f"LLM analysis failed: {response.status_code}")
                return "LLM analysis failed"

        except Exception as e:
            self.logger.error(f"LLM analysis error: {e}")
            return "Error analyzing with LLM"

    def _plan_adjustments(self, analysis: str) -> List[Dict[str, Any]]:
        """Plan parameter adjustments based on LLM analysis"""
        adjustments = []

        # Simple rule-based parsing for now (could be made more sophisticated)
        if "entropy" in analysis.lower() and ("spike" in analysis.lower() or "increase" in analysis.lower()):
            # Entropy spike - reduce diffusion
            adjustments.append({
                'type': 'parameter_adjustment',
                'parameter': 'diffusion_rate',
                'new_value': 0.8,  # Scale down
                'reason': 'Entropy spike detected - reducing diffusion to stabilize',
                'confidence': 0.85
            })

        if "death rate" in analysis.lower():
            adjustments.append({
                'type': 'parameter_adjustment',
                'parameter': 'noise_level',
                'new_value': 0.1,  # Reduce noise
                'reason': 'High bot death rate - reducing noise to increase stability',
                'confidence': 0.9
            })

        if "tick timeout" in analysis.lower():
            adjustments.append({
                'type': 'scaling_adjustment',
                'action': 'increase_workers',
                'parameter': 'bot_instances',
                'delta': 0.2,  # +20%
                'reason': 'Tick timeout - increasing worker capacity',
                'confidence': 0.75
            })

        self.adjustments.extend(adjustments)
        return adjustments

    def _apply_adjustments(self, adjustments: List[Dict[str, Any]]):
        """Apply approved adjustments to Layer 0"""
        for adj in adjustments:
            if self.config['human_approval_required']:
                self.logger.info(f"AUTOMATIC ADJUSTMENT PENDING APPROVAL: {adj}")
                # For now, just log - would need approval mechanism
            else:
                self.logger.info(f"Applying adjustment: {adj}")
                success = self._send_parameter_update(adj)
                if success:
                    self.logger.info("✅ Parameter update applied to Layer 0")
                else:
                    self.logger.error("❌ Failed to apply parameter update")

    def _send_parameter_update(self, adjustment: Dict[str, Any]) -> bool:
        """Send parameter update to all active Layer 0 bots"""
        # Read swarm state to get bot endpoints
        state_file = Path('bots/swarm_state.yaml')
        if not state_file.exists():
            self.logger.warning("Swarm state not available, cannot send updates")
            return False

        try:
            with open(state_file, 'r') as f:
                swarm_state = yaml.safe_load(f)

            if not isinstance(swarm_state, dict):
                self.logger.warning("Invalid swarm state format")
                return False

            active_bots = [b for b in swarm_state.get('bots', []) if self._check_bot_alive('localhost', b['port'])]

            if not active_bots:
                self.logger.warning("No active bots found")
                return False

            # Prepare update payload
            updates = {}
            if adjustment['type'] == 'parameter_adjustment':
                param_name = adjustment.get('parameter')
                new_value = adjustment.get('new_value')
                if param_name and new_value is not None:
                    updates[param_name] = new_value

            if not updates:
                self.logger.warning(f"No valid updates in adjustment: {adjustment}")
                return False

            payload = {'updates': updates}
            self.logger.info(f"Sending update to {len(active_bots)} bots: {payload}")

            # Send to all active bots
            success_count = 0
            for bot in active_bots:
                try:
                    response = requests.post(
                        f"http://localhost:{bot['port']}/parameters/update",
                        json=payload,
                        timeout=3, verify=False
                    )
                    if response.status_code == 200:
                        success_count += 1
                        self.logger.debug(f"Updated bot {bot['bot_id']}")
                    else:
                        self.logger.warning(f"Failed to update bot {bot['bot_id']}: {response.status_code}")

                except Exception as e:
                    self.logger.warning(f"Error updating bot {bot['bot_id']}: {e}")

            self.logger.info(f"Successfully updated {success_count}/{len(active_bots)} bots")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error sending parameter update: {e}")
            return False

    def _check_bot_alive(self, host: str, port: int) -> bool:
        """Quick health check for bot"""
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2, verify=False)
            return response.status_code == 200
        except:
            return False

def main():
    """Entry point for swarm supervisor"""
    import argparse

    parser = argparse.ArgumentParser(description='Swarm Supervisor - Layer 1 Orchestrator')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-human-approval', action='store_true',
                       help='Disable human approval for adjustments (dangerous)')

    args = parser.parse_args()

    if args.no_human_approval:
        SUPERVISOR_CONFIG['human_approval_required'] = False

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - SUPERVISOR - %(levelname)s - %(message)s'
    )

    supervisor = SwarmSupervisor()

    try:
        supervisor.start_supervision()
    except KeyboardInterrupt:
        supervisor.stop_supervision()
    except Exception as e:
        supervisor.logger.error(f"Supervisor failed: {e}")
        supervisor.stop_supervision()
        raise

if __name__ == '__main__':
    main()
