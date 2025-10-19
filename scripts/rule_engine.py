#!/usr/bin/env python3
"""
---
script: rule_engine.py
purpose: Cellular automata rule engine for swarm bots
status: development
created: 2025-10-19
---
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import yaml
import logging

logger = logging.getLogger('RuleEngine')

class CellularRuleEngine:
    """
    Rule engine implementing cellular automata transitions for swarm bots.

    Each bot has an internal state vector (512-dimensional embedding).
    Rules are applied based on local neighbor interactions within the CA grid.
    """

    def __init__(self, config_path: str = None):
        """Initialize rule engine with configuration"""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str | None) -> Dict[str, Any]:
        """Load CA configuration"""
        if config_path is None:
            from pathlib import Path
            config_path = str(Path(__file__).parent.parent / 'configs' / 'swarm_config.yaml')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError("Failed to load config file or config is empty")

        # Extract CA-specific settings
        ca_config = {
            'grid_width': config['swarm']['grid_width'],
            'grid_height': config['swarm']['grid_height'],
            'state_vector_dim': 512,  # Temporary embedding size
            'decay_factor': 0.3,      # Alpha in averaging rule
            'rule_type': 'diffusion_damping'  # Type of CA rule
        }

        return ca_config

    def load_swarm_state(self) -> Dict[str, Any]:
        """Load current swarm state with grid mapping"""
        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                state = yaml.safe_load(f)
            return state
        except FileNotFoundError:
            logger.error("Swarm state file not found. Run launch_swarm.py first.")
            return None

    def get_neighbors(self, grid_x: int, grid_y: int, swarm_state: Dict) -> List[Dict[str, Any]]:
        """
        Get 4-way Von Neumann neighbors for a bot at (grid_x, grid_y).

        Returns list of neighbor bot dictionaries, empty if no neighbor exists.
        """
        neighbors = []
        grid_width = swarm_state.get('grid_width', 10)
        grid_height = swarm_state.get('grid_height', 4)

        # Cardinal directions
        directions = [
            (0, -1),   # North
            (0, 1),    # South
            (-1, 0),   # West
            (1, 0)     # East
        ]

        for dx, dy in directions:
            nx = grid_x + dx
            ny = grid_y + dy

            # Wrap around toroidal topology
            nx = nx % grid_width
            ny = ny % grid_height

            # Find bot at this position
            for bot in swarm_state['bots']:
                if bot['grid_x'] == nx and bot['grid_y'] == ny:
                    neighbors.append(bot)
                    break

        logger.debug(f"Bot at ({grid_x},{grid_y}) has {len(neighbors)} neighbors")
        return neighbors

    def diffusion_damping_rule(self, self_state: np.ndarray, neighbor_states: List[np.ndarray]) -> np.ndarray:
        """
        Simple diffusion-damping cellular automata rule.

        new_state = alpha * mean(neighbor_states) + (1-alpha) * self_state + noise
        """
        alpha = self.config['decay_factor']

        if not neighbor_states:
            # No neighbors: purge state toward zero with decay
            new_state = (1 - alpha) * self_state
        else:
            neighbor_mean = np.mean(neighbor_states, axis=0)
            new_state = alpha * neighbor_mean + (1 - alpha) * self_state

        # Add small noise to prevent stagnation
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, size=new_state.shape)
        new_state += noise

        return new_state

    def apply_rule_to_bot(self, bot: Dict[str, Any], swarm_state: Dict) -> np.ndarray:
        """
        Apply CA rule to a single bot based on its current state and neighbors.

        For now, state is randomly initialized. In production, this would load
        from bot's persistent memory or Gemma model internal state.
        """
        # Get current state (placeholder - initialize if not exists)
        state_dim = self.config['state_vector_dim']
        if 'state_vector' not in bot:
            # Initialize with random values
            bot['state_vector'] = np.random.randn(state_dim).tolist()

        self_state = np.array(bot['state_vector'])

        # Get neighbors and their states
        neighbors = self.get_neighbors(bot['grid_x'], bot['grid_y'], swarm_state)
        neighbor_states = []

        for neighbor in neighbors:
            if 'state_vector' not in neighbor:
                # Initialize neighbor state if not present
                neighbor['state_vector'] = np.random.randn(state_dim).tolist()
            neighbor_states.append(np.array(neighbor['state_vector']))

        # Apply CA rule
        new_state = self.diffusion_damping_rule(self_state, neighbor_states)

        # Clip to reasonable bounds to prevent divergence
        new_state = np.clip(new_state, -3.0, 3.0)

        logger.debug(f"Bot {bot['bot_id']} at ({bot['grid_x']},{bot['grid_y']}) rule applied")

        return new_state

    def update_swarm_state(self, swarm_state: Dict) -> Dict:
        """
        Apply CA rules to the entire swarm for one tick.

        Returns updated swarm_state with new state vectors.
        """
        for bot in swarm_state['bots']:
            new_state = self.apply_rule_to_bot(bot, swarm_state)
            bot['state_vector'] = new_state.tolist()

            # Log state magnitude as simple health indicator
            state_magnitude = np.linalg.norm(new_state)
            bot['state_magnitude'] = float(state_magnitude)

        swarm_state['tick'] = swarm_state.get('tick', 0) + 1
        swarm_state['latest_update'] = 'rule_engine'

        logger.info(f"CA rule update complete for tick {swarm_state['tick']}")

        return swarm_state

def main():
    """Demonstration of rule engine on current swarm"""
    logging.basicConfig(level=logging.INFO)

    engine = CellularRuleEngine()

    # Load swarm state
    swarm_state = engine.load_swarm_state()
    if not swarm_state:
        return

    logger.info(f"Applying CA rules to {swarm_state['total_bots']} bots")
    logger.info(f"Grid size: {swarm_state['grid_width']}x{swarm_state['grid_height']}")

    # Apply one update tick
    updated_state = engine.update_swarm_state(swarm_state)

    # Save back to file
    with open('bots/swarm_state.yaml', 'w') as f:
        yaml.dump(updated_state, f)

    logger.info("Swarm state updated with CA rule results")

if __name__ == '__main__':
    main()
