#!/usr/bin/env python3
"""
---
script: bot_worker.py
purpose: Individual bot worker for Granite4:micro-h swarm
status: production-ready
created: 2025-10-18
---
"""

import argparse
import requests
import time
import json
import os
import logging
from datetime import datetime

class BotWorker:
    def __init__(self, bot_id, gpu_id, port):
        self.bot_id = bot_id
        self.gpu_id = gpu_id
        self.port = port
        self.ollama_url = "http://localhost:11434"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Bot-{bot_id}] %(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(f'Bot-{bot_id}')

        self.memory = []
        self.stats = {
            'requests': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }

    def query_ollama(self, prompt):
        """Query Ollama with prompt"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "granite4:micro-h",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                self.stats['requests'] += 1
                return response.json()['response']
            else:
                self.stats['errors'] += 1
                self.logger.error(f"Ollama error: {response.status_code}")
                return None

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Query failed: {e}")
            return None

    def health_check(self):
        """Perform health check"""
        result = self.query_ollama("ping")
        if result:
            self.logger.info(f"✓ Health check passed - Stats: {self.stats}")
            return True
        else:
            self.logger.error("✗ Health check failed")
            return False

    def run(self):
        """Main bot loop"""
        self.logger.info(f"Bot starting on GPU {self.gpu_id}, Port {self.port}")
        self.logger.info(f"Connecting to Ollama at {self.ollama_url}")

        # Initial health check
        time.sleep(2)
        if not self.health_check():
            self.logger.error("Initial health check failed, exiting")
            return

        self.logger.info("Bot operational, entering main loop")

        # Main event loop
        try:
            while True:
                time.sleep(60)  # Periodic health check
                self.health_check()

        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)

    args = parser.parse_args()

    bot = BotWorker(args.bot_id, args.gpu, args.port)
    bot.run()
