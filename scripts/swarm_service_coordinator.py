#!/usr/bin/env python3
"""
---
script: swarm_service_coordinator.py
purpose: Microservice for coordinating swarm manager ownership and preventing conflicts
status: development
created: 2025-10-19
---
"""

from flask import Flask, jsonify, request
import psutil
import os
import time
import threading
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Coordinator - %(levelname)s - %(message)s')
logger = logging.getLogger('SwarmCoordinator')

app = Flask(__name__)

# Configuration constants
LEASE_SECONDS = int(os.getenv('SWARM_COORDINATOR_LEASE_SECONDS', '120'))
API_TOKEN = os.getenv('SWARM_COORDINATOR_API_TOKEN')  # Optional security token
COORDINATOR_HOST = os.getenv('SWARM_COORDINATOR_HOST', 'localhost')
COORDINATOR_PORT = int(os.getenv('SWARM_COORDINATOR_PORT', '5050'))

STATE: Dict[str, Any] = {
    'owner': None,
    'last_checkin': 0,
    'start_time': time.time(),
    'process_id': os.getpid()
}

# Thread synchronization lock
state_lock = threading.Lock()

# Background cleanup monitor
def cleanup_monitor():
    """Monitor for dead/stale owners and clean up"""
    while True:
        try:
            time.sleep(30)  # Check every 30 seconds
            
            with state_lock:
                if STATE['owner'] and (time.time() - STATE['last_checkin'] > LEASE_SECONDS):
                    logger.warning(f"Owner {STATE['owner']} appears stale (>{LEASE_SECONDS}s) - releasing ownership")
                    STATE['owner'] = None
                    STATE['last_checkin'] = 0
                    
        except Exception as e:
            logger.error(f"Cleanup monitor error: {e}")

@app.route("/status", methods=["GET"])
def get_status():
    """Get current coordinator status"""
    with state_lock:
        return jsonify({
            "current_owner": STATE['owner'],
            "last_checkin": STATE['last_checkin'],
            "timestamp": time.time(),
            "age_seconds": time.time() - STATE['start_time'],
            "lease_seconds": LEASE_SECONDS,
            "healthy": STATE['owner'] is None or (time.time() - STATE['last_checkin'] < LEASE_SECONDS)
        })

@app.route("/claim", methods=["POST"])
def claim_ownership():
    """Attempt to claim ownership of swarm management"""
    data = request.json or {}
    claimant = data.get('name', 'unknown')
    claimant_pid = data.get('pid')
    
    # Verify claimant is actually running
    if claimant_pid:
        try:
            psutil.Process(claimant_pid).name()  # Check if process exists and accessible
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning(f"Claimant process {claimant_pid} not found or inaccessible")
            return jsonify({
                "status": "denied",
                "reason": "claimant_process_not_found",
                "claimant_pid": claimant_pid
            })
    
    with state_lock:
        current_time = time.time()
        
        # If no current owner, grant ownership
        if not STATE['owner']:
            STATE['owner'] = claimant
            STATE['last_checkin'] = current_time
            logger.info(f"Granted ownership to {claimant} (pid: {claimant_pid})")
            return jsonify({
                "status": "granted",
                "owner_since": current_time,
                "lease_seconds": LEASE_SECONDS
            })
        
        # If current owner is stale (>LEASE_SECONDS), reassign ownership
        elif current_time - STATE['last_checkin'] > LEASE_SECONDS:
            old_owner = STATE['owner']
            STATE['owner'] = claimant
            STATE['last_checkin'] = current_time
            logger.warning(f"Reassigned ownership from stale owner {old_owner} to {claimant}")
            return jsonify({
                "status": "granted",
                "owner_since": current_time,
                "lease_seconds": LEASE_SECONDS,
                "reassigned_from": old_owner
            })
        
        # Otherwise deny claim
        else:
            lock_duration = current_time - STATE['last_checkin']
            return jsonify({
                "status": "denied",
                "reason": "ownership_locked",
                "current_owner": STATE['owner'],
                "lock_duration_seconds": lock_duration,
                "lease_remaining_seconds": LEASE_SECONDS - lock_duration
            })

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    """Send heartbeat to maintain ownership"""
    data = request.json or {}
    claimant = data.get('name')
    
    with state_lock:
        if claimant == STATE['owner']:
            STATE['last_checkin'] = time.time()
            return jsonify({"status": "heartbeat_acknowledged"})
        else:
            return jsonify({
                "status": "not_owner",
                "current_owner": STATE['owner']
            })

@app.route("/release", methods=["POST"])
def release_ownership():
    """Release ownership of swarm management"""
    data = request.json or {}
    claimant = data.get('name')
    
    with state_lock:
        if claimant == STATE['owner']:
            logger.info(f"Ownership released by {claimant}")
            STATE['owner'] = None
            STATE['last_checkin'] = 0
            return jsonify({"status": "released"})
        else:
            return jsonify({"status": "not_owner", "current_owner": STATE['owner']})

@app.route("/force_release", methods=["POST"])
def force_release():
    """Force release ownership (admin use - optionally secured)"""
    # Check API token if configured
    if API_TOKEN:
        data = request.json or {}
        provided_token = request.headers.get('X-API-Token') or data.get('api_token')
        if provided_token != API_TOKEN:
            return jsonify({"status": "unauthorized"}), 401
    
    with state_lock:
        old_owner = STATE['owner']
        STATE['owner'] = None
        STATE['last_checkin'] = 0
        logger.warning(f"Force released ownership from {old_owner}")
        return jsonify({
            "status": "force_released",
            "previous_owner": old_owner
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    with state_lock:
        return jsonify({
            "status": "healthy",
            "uptime_seconds": time.time() - STATE['start_time'],
            "has_owner": STATE['owner'] is not None,
            "timestamp": time.time(),
            "version": "1.0"
        })

@app.route("/config", methods=["GET"])
def get_config():
    """Get coordinator configuration"""
    config = {
        "lease_seconds": LEASE_SECONDS,
        "api_token_configured": API_TOKEN is not None,
        "host": COORDINATOR_HOST,
        "port": COORDINATOR_PORT,
        "pid": STATE['process_id']
    }
    return jsonify(config)

def main():
    """Start the swarm coordinator service"""
    logger.info("Starting Swarm Service Coordinator...")
    logger.info(f"Configuration:")
    logger.info(f"  Host: {COORDINATOR_HOST}:{COORDINATOR_PORT}")
    logger.info(f"  Lease: {LEASE_SECONDS}s")
    logger.info(f"  Security: {'enabled' if API_TOKEN else 'disabled'}")
    logger.info("Endpoints:")
    logger.info("  GET  /status        - Get current status")
    logger.info("  GET  /config        - Get configuration")
    logger.info("  GET  /health        - Health check")
    logger.info("  POST /claim         - Attempt to claim ownership")
    logger.info("  POST /heartbeat     - Renew ownership lease")
    logger.info("  POST /release       - Release ownership")
    logger.info("  POST /force_release - Force release (admin)")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_monitor, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup monitor started")
    
    try:
        app.run(host=COORDINATOR_HOST, port=COORDINATOR_PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("Coordinator shutdown requested")
    except Exception as e:
        logger.error(f"Coordinator failed: {e}")
        raise

if __name__ == '__main__':
    main()
