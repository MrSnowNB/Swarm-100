#!/usr/bin/env python3
"""
---
script: swarm_coordinator_client.py
purpose: Client utilities for interacting with the Swarm Service Coordinator
status: development
created: 2025-10-19
---
"""

import requests
import threading
import time
import socket
import os
import atexit
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - SwarmClient - %(levelname)s - %(message)s')
logger = logging.getLogger('SwarmCoordinatorClient')

# Configuration - match coordinator defaults
COORDINATOR_HOST = os.getenv('SWARM_COORDINATOR_HOST', 'localhost')
COORDINATOR_PORT = int(os.getenv('SWARM_COORDINATOR_PORT', '5050'))
COORDINATOR_URL = f"http://{COORDINATOR_HOST}:{COORDINATOR_PORT}"

class SwarmCoordinatorClient:
    """Client for interacting with the Swarm Service Coordinator"""

    def __init__(self, claimant_name: Optional[str] = None):
        """
        Initialize coordinator client

        Args:
            claimant_name: Unique identifier for this claimant process
        """
        if claimant_name is None:
            claimant_name = f"{socket.gethostname()}-{os.getpid()}"
        self.claimant_name = claimant_name
        self.owned = False
        self.heartbeat_thread: Optional[threading.Thread] = None

        # Disable SSL warnings for localhost requests
        try:
            requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)  # type: ignore
        except AttributeError:
            pass

    def is_coordinator_available(self) -> bool:
        """Check if coordinator is running"""
        try:
            resp = requests.get(f"{COORDINATOR_URL}/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def acquire_ownership(self) -> bool:
        """
        Acquire exclusive ownership of swarm management

        Returns:
            True if ownership granted, False otherwise
        """
        if not self.is_coordinator_available():
            logger.warning("Swarm coordinator not available - proceeding without coordination")
            return True

        try:
            resp = requests.post(
                f"{COORDINATOR_URL}/claim",
                json={"name": self.claimant_name, "pid": os.getpid()},
                timeout=10
            )
            resp.raise_for_status()

            data = resp.json()
            if data["status"] == "granted":
                self.owned = True

                # Start heartbeat thread
                self.heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True,
                    name=f"heartbeat-{self.claimant_name}"
                )
                self.heartbeat_thread.start()

                # Release ownership on exit
                atexit.register(self.release_ownership)

                logger.info(f"Swarm management ownership granted: {self.claimant_name}")
                return True
            else:
                logger.warning(f"Ownership denied: {data}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to acquire ownership: {e}")
            return False

    def release_ownership(self) -> bool:
        """Release ownership of swarm management"""
        if not self.owned:
            return True

        if not self.is_coordinator_available():
            logger.warning("Coordinator not available for release")
            self.owned = False
            return True

        try:
            resp = requests.post(
                f"{COORDINATOR_URL}/release",
                json={"name": self.claimant_name},
                timeout=5
            )
            resp.raise_for_status()

            data = resp.json()
            if data["status"] == "released":
                logger.info(f"Ownership released: {self.claimant_name}")
                self.owned = False
                return True
            else:
                logger.warning(f"Release failed: {data}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to release ownership: {e}")
            return False

    def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain ownership"""
        while self.owned:
            try:
                time.sleep(30)  # Heartbeat every 30s

                if not self.is_coordinator_available():
                    logger.error("Coordinator unavailable during heartbeat")
                    self.owned = False
                    break

                resp = requests.post(
                    f"{COORDINATOR_URL}/heartbeat",
                    json={"name": self.claimant_name},
                    timeout=5
                )

                if resp.status_code != 200:
                    data = resp.json()
                    if data.get("status") != "heartbeat_acknowledged":
                        logger.warning(f"Heartbeat failed: {data}")
                        self.owned = False
                        break

            except requests.exceptions.RequestException as e:
                logger.error(f"Heartbeat failed: {e}")
                self.owned = False
                break

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current coordinator status"""
        if not self.is_coordinator_available():
            return None

        try:
            resp = requests.get(f"{COORDINATOR_URL}/status", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            return None

    def has_ownership(self) -> bool:
        """Check if this client currently has ownership"""
        return self.owned

# Global client instance for convenience
_coordinator_client: Optional[SwarmCoordinatorClient] = None

def get_coordinator_client(claimant_name: Optional[str] = None) -> SwarmCoordinatorClient:
    """Get singleton coordinator client instance"""
    global _coordinator_client
    if _coordinator_client is None:
        _coordinator_client = SwarmCoordinatorClient(claimant_name)
    return _coordinator_client

def acquire_swarm_ownership(claimant_name: Optional[str] = None) -> bool:
    """Convenience function to acquire swarm ownership"""
    client = get_coordinator_client(claimant_name)
    return client.acquire_ownership()

def release_swarm_ownership() -> bool:
    """Convenience function to release swarm ownership"""
    client = get_coordinator_client()
    return client.release_ownership()

def has_swarm_ownership() -> bool:
    """Convenience function to check ownership status"""
    client = get_coordinator_client()
    return client.has_ownership()

if __name__ == '__main__':
    # Test the client
    client = SwarmCoordinatorClient("test-client")

    if not client.is_coordinator_available():
        print("Coordinator not running. Start with: python3 scripts/swarm_service_coordinator.py")
        exit(1)

    print("Attempting to acquire ownership...")
    if client.acquire_ownership():
        print("Ownership acquired!")

        # Hold ownership for 10 seconds
        print("Holding ownership for 10 seconds...")
        time.sleep(10)

        # Release ownership
        if client.release_ownership():
            print("Ownership released successfully!")
        else:
            print("Failed to release ownership!")
    else:
        print("Failed to acquire ownership!")
        status = client.get_status()
        if status:
            print(f"Current owner: {status.get('current_owner')}")
