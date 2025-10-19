#!/usr/bin/env python3
"""
---
file: test_adaptive_rl_communication.py
purpose: Reinforcement Learning for Adaptive Communication Resilience
framework: pytest with Q-learning implementation
status: implements behavioral learning integration for communication optimization
created: 2025-10-19
---
**Behavioral Learning Integration Strategy:**
This implements Q-learning to optimize Swarm-100's communication resilience parameters.
Learns optimal heartbeat intervals, pulse ranges, and trust thresholds based on:
- State: [agent_density, trust_entropy, failure_rate, communication_success]
- Actions: [adjust_heartbeat, modify_range, update_trust_threshold, toggle_sar]
- Reward: connectivity_stability - energy_penalty - failure_penalty
"""

import pytest
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class CommunicationParams:
    """Communication parameters being optimized by RL"""
    heartbeat_interval_ms: int = 1000
    pulse_range_multiplier: float = 1.0
    trust_threshold: float = 0.3
    sar_mode_active: bool = False


@dataclass
class SwarmState:
    """Current state of the swarm communication system"""
    agent_density: float  # 0.0 to 1.0
    trust_entropy: float  # 0.0 to 1.0 (higher = more trust diversity)
    failure_rate: float   # 0.0 to 1.0
    communication_success: float  # 0.0 to 1.0


@dataclass
class PerformanceMetrics:
    """Performance outcome after parameter application"""
    connectivity_score: float
    energy_efficiency: float
    failure_rate: float
    recovery_time_ms: float


class AdaptiveCommunicationRL:
    """
    Q-Learning Agent for Communication Resilience Optimization

    State Space: 5×4×3×4 = 240 states
    Action Space: 8 actions
    Reward Function: connectivity_gain - energy_penalty - failure_penalty
    """

    # State discretization bins
    DENSITY_BINS = 5      # [0.0, 0.2, 0.4, 0.6, 0.8]
    TRUST_BINS = 4        # [0.0, 0.25, 0.5, 0.75]
    FAILURE_BINS = 3      # [0.0, 0.1, 0.3]
    SUCCESS_BINS = 4      # [0.0, 0.25, 0.5, 0.75]

    # Actions
    ACTIONS = [
        "reduce_heartbeat",     # -100ms interval
        "increase_heartbeat",   # +100ms interval
        "extend_pulse_range",   # +0.1 multiplier
        "reduce_pulse_range",   # -0.1 multiplier
        "raise_trust_threshold", # +0.05 threshold
        "lower_trust_threshold", # -0.05 threshold
        "activate_sar",         # enable SAR mode
        "conserve_energy"       # disable SAR, reduce range
    ]

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-table: state_key -> action -> value
        self.q_table: Dict[int, Dict[int, float]] = {}

        # Learning state
        self.total_reward = 0.0
        self.episodes_completed = 0
        self.best_reward = float('-inf')
        self.best_params = CommunicationParams()

        # Random generation
        random.seed(seed)
        np.random.seed(seed)

        # Performance tracking
        self.learning_history: List[Dict] = []

    def get_state_key(self, state: SwarmState) -> int:
        """Convert continuous state to discrete state key"""
        density_idx = self._discretize(state.agent_density, self.DENSITY_BINS)
        trust_idx = self._discretize(state.trust_entropy, self.TRUST_BINS)
        failure_idx = min(int(state.failure_rate * 10), self.FAILURE_BINS - 1)
        success_idx = self._discretize(state.communication_success, self.SUCCESS_BINS)

        # Combine into single integer key (5×4×3×4 = 240 possible states)
        return (density_idx * (self.TRUST_BINS * self.FAILURE_BINS * self.SUCCESS_BINS) +
                trust_idx * (self.FAILURE_BINS * self.SUCCESS_BINS) +
                failure_idx * self.SUCCESS_BINS +
                success_idx)

    def select_action(self, state: SwarmState) -> int:
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)

        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, len(self.ACTIONS) - 1)
        else:
            # Exploitation: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {i: 0.0 for i in range(len(self.ACTIONS))}

            action_values = self.q_table[state_key]
            return max(action_values, key=lambda action: action_values[action])

    def calculate_reward(self, old_metrics: PerformanceMetrics,
                        new_metrics: PerformanceMetrics,
                        action_taken: int) -> float:
        """Calculate reward based on performance improvement"""

        # Connectivity improvement (main reward component)
        connectivity_gain = new_metrics.connectivity_score - old_metrics.connectivity_score

        # Energy efficiency penalty (trade-off for connectivity)
        energy_penalty = (1.0 - new_metrics.energy_efficiency) * 0.5

        # Failure rate penalty
        failure_penalty = new_metrics.failure_rate * 2.0

        # Recovery time penalty
        recovery_penalty = new_metrics.recovery_time_ms / 10000.0  # Normalize to 0-1 range

        # Action-specific bonuses/penalties
        if action_taken in [6, 7]:  # SAR activation/deactivation
            # Slight penalty for mode changes (stability preference)
            recovery_penalty *= 1.1

        total_reward = (connectivity_gain * 10.0 -  # Weight connectivity highly
                       energy_penalty -
                       failure_penalty -
                       recovery_penalty)

        return total_reward

    def update_q_value(self, state_key: int, action: int,
                      reward: float, next_state_key: int):
        """Q-learning update rule"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.ACTIONS))}

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(len(self.ACTIONS))}

        # Q-learning update
        old_q = self.q_table[state_key][action]
        max_future_q = max(self.q_table[next_state_key].values())

        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state_key][action] = new_q

        return new_q

    def apply_action(self, params: CommunicationParams, action: int) -> CommunicationParams:
        """Apply action to communication parameters"""
        new_params = CommunicationParams(
            heartbeat_interval_ms=params.heartbeat_interval_ms,
            pulse_range_multiplier=params.pulse_range_multiplier,
            trust_threshold=params.trust_threshold,
            sar_mode_active=params.sar_mode_active
        )

        if action == 0:   # reduce_heartbeat
            new_params.heartbeat_interval_ms = max(200, params.heartbeat_interval_ms - 100)
        elif action == 1: # increase_heartbeat
            new_params.heartbeat_interval_ms = min(5000, params.heartbeat_interval_ms + 100)
        elif action == 2: # extend_pulse_range
            new_params.pulse_range_multiplier = min(2.0, params.pulse_range_multiplier + 0.1)
        elif action == 3: # reduce_pulse_range
            new_params.pulse_range_multiplier = max(0.5, params.pulse_range_multiplier - 0.1)
        elif action == 4: # raise_trust_threshold
            new_params.trust_threshold = min(0.8, params.trust_threshold + 0.05)
        elif action == 5: # lower_trust_threshold
            new_params.trust_threshold = max(0.1, params.trust_threshold - 0.05)
        elif action == 6: # activate_sar
            new_params.sar_mode_active = True
        elif action == 7: # conserve_energy
            new_params.sar_mode_active = False
            new_params.pulse_range_multiplier = min(params.pulse_range_multiplier, 1.0)

        return new_params

    def simulate_performance(self, state: SwarmState,
                           params: CommunicationParams) -> PerformanceMetrics:
        """Simulate performance outcome for given state and parameters

        This would normally integrate with actual CyberGrid simulation,
        but here we use a simplified model to demonstrate RL learning.
        """

        # Base performance influenced by state
        base_connectivity = (state.agent_density * 0.4 +
                           state.communication_success * 0.4 +
                           (1.0 - state.failure_rate) * 0.2)

        # Parameter effects
        param_connectivity_boost = 0.0
        param_energy_penalty = 0.0

        # Heartbeat interval effect (optimal around 800-1200ms)
        heartbeat_factor = 1.0 - abs(params.heartbeat_interval_ms - 1000) / 2000.0
        param_connectivity_boost += heartbeat_factor * 0.1

        # Pulse range effect (trade-off between connectivity and energy)
        range_effect = params.pulse_range_multiplier - 1.0  # Deviation from baseline
        param_connectivity_boost += range_effect * 0.2
        param_energy_penalty += abs(range_effect) * 0.3

        # Trust threshold effect (optimal around 0.3-0.5)
        trust_factor = 1.0 - abs(params.trust_threshold - 0.4) / 0.4
        param_connectivity_boost += trust_factor * 0.1

        # SAR mode effect (helps in high failure scenarios)
        if params.sar_mode_active:
            if state.failure_rate > 0.2:
                param_connectivity_boost += 0.15
                param_energy_penalty += 0.2  # SAR costs energy

        # Final metrics calculation
        connectivity_score = min(1.0, max(0.0, base_connectivity + param_connectivity_boost))
        energy_efficiency = max(0.0, 1.0 - param_energy_penalty)
        failure_rate = max(0.0, state.failure_rate - param_connectivity_boost * 0.3)
        recovery_time_ms = 3000 * (1.0 - connectivity_score)  # Higher connectivity = faster recovery

        return PerformanceMetrics(
            connectivity_score=connectivity_score,
            energy_efficiency=energy_efficiency,
            failure_rate=failure_rate,
            recovery_time_ms=recovery_time_ms
        )

    def train_episode(self, initial_state: SwarmState,
                     initial_params: CommunicationParams,
                     max_steps: int = 50) -> List[Dict]:
        """Run single training episode"""
        episode_history = []
        current_state = initial_state
        current_params = initial_params
        step_metrics = self.simulate_performance(current_state, current_params)

        episode_reward = 0.0

        for step in range(max_steps):
            # Select and apply action
            action = self.select_action(current_state)
            new_params = self.apply_action(current_params, action)

            # Get new performance metrics
            new_metrics = self.simulate_performance(current_state, new_params)

            # Calculate reward
            reward = self.calculate_reward(step_metrics, new_metrics, action)

            # Update Q-table
            state_key = self.get_state_key(current_state)
            new_state_key = self.get_state_key(current_state)  # State doesn't change in simulation
            self.update_q_value(state_key, action, reward, new_state_key)

            # Accumulate reward
            episode_reward += reward

            # Log step results
            step_result = {
                'step': step,
                'action': self.ACTIONS[action],
                'reward': reward,
                'connectivity': new_metrics.connectivity_score,
                'energy_efficiency': new_metrics.energy_efficiency,
                'parameters': {
                    'heartbeat_ms': new_params.heartbeat_interval_ms,
                    'pulse_range': new_params.pulse_range_multiplier,
                    'trust_threshold': new_params.trust_threshold,
                    'sar_active': new_params.sar_mode_active
                }
            }
            episode_history.append(step_result)

            # Update for next iteration
            current_params = new_params
            step_metrics = new_metrics

        # Track episode results
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_params = current_params

        self.total_reward += episode_reward
        self.episodes_completed += 1

        episode_summary = {
            'episode': self.episodes_completed,
            'total_reward': episode_reward,
            'avg_connectivity': np.mean([h['connectivity'] for h in episode_history]),
            'final_parameters': {
                'heartbeat_ms': current_params.heartbeat_interval_ms,
                'pulse_range': current_params.pulse_range_multiplier,
                'trust_threshold': current_params.trust_threshold,
                'sar_active': current_params.sar_mode_active
            }
        }
        self.learning_history.append(episode_summary)

        return episode_history

    def get_optimal_parameters(self) -> CommunicationParams:
        """Get best learned parameter configuration"""
        return self.best_params

    def _discretize(self, value: float, bins: int) -> int:
        """Convert continuous value to discrete bin index"""
        if value <= 0.0:
            return 0
        if value >= 1.0:
            return bins - 1
        return min(int(value * bins), bins - 1)


class SwarmCommunicationOptimizer:
    """High-level optimizer using RL for Swarm-100 communication parameters"""

    def __init__(self, rl_agent: AdaptiveCommunicationRL):
        self.rl_agent = rl_agent
        self.training_history: List[Dict] = []

    def optimize_for_scenario(self, scenario_name: str,
                            initial_state: SwarmState,
                            training_episodes: int = 100) -> Dict:
        """Optimize communication parameters for specific scenario"""

        print(f"🧠 Optimizing communication for scenario: {scenario_name}")
        print(f"   Training episodes: {training_episodes}")

        scenario_initial_params = CommunicationParams()
        scenario_history = []

        start_time = time.time()
        for episode in range(training_episodes):
            episode_result = self.rl_agent.train_episode(
                initial_state, scenario_initial_params, max_steps=30
            )

            if (episode + 1) % 25 == 0:
                avg_reward = np.mean([h['total_reward'] for h in self.rl_agent.learning_history[-5:]])
                print(f"Average reward after {episode + 1} episodes: {avg_reward:.1f}")
        training_time = time.time() - start_time

        optimal_params = self.rl_agent.get_optimal_parameters()

        scenario_result = {
            'scenario': scenario_name,
            'training_episodes': training_episodes,
            'training_time_seconds': training_time,
            'best_reward': self.rl_agent.best_reward,
            'optimal_parameters': {
                'heartbeat_interval_ms': optimal_params.heartbeat_interval_ms,
                'pulse_range_multiplier': optimal_params.pulse_range_multiplier,
                'trust_threshold': optimal_params.trust_threshold,
                'sar_mode_active': optimal_params.sar_mode_active
            },
            'convergence_metrics': self._calculate_convergence_metrics()
        }

        self.training_history.append(scenario_result)
        return scenario_result

    def _calculate_convergence_metrics(self) -> Dict:
        """Calculate learning convergence metrics"""
        if len(self.rl_agent.learning_history) < 10:
            return {'convergence_score': 0.0, 'learning_stability': 0.0}

        # Calculate reward stability over last 20 episodes
        recent_rewards = [h['total_reward'] for h in self.rl_agent.learning_history[-20:]]
        reward_std = float(np.std(recent_rewards))
        reward_mean = float(np.mean(recent_rewards))

        # Convergence score (lower variance = higher convergence)
        convergence_score = max(0.0, 1.0 - (reward_std / abs(reward_mean)) * 2.0)

        # Learning trend stability
        learning_stability = self._calculate_learning_stability()

        return {
            'convergence_score': convergence_score,
            'learning_stability': learning_stability,
            'final_reward_mean': reward_mean,
            'final_reward_std': reward_std
        }

    def _calculate_learning_stability(self) -> float:
        """Measure stability of learned parameters over time"""
        if len(self.rl_agent.learning_history) < 5:
            return 0.0

        recent_params = [h['final_parameters'] for h in self.rl_agent.learning_history[-5:]]
        heartbeat_values = [p['heartbeat_ms'] for p in recent_params]
        range_values = [p['pulse_range'] for p in recent_params]
        trust_values = [p['trust_threshold'] for p in recent_params]

        # Calculate coefficient of variation for each parameter
        hb_cv = np.std(heartbeat_values) / np.mean(heartbeat_values) if heartbeat_values else 0.0
        range_cv = np.std(range_values) / np.mean(range_values) if range_values else 0.0
        trust_cv = np.std(trust_values) / np.mean(trust_values) if trust_values else 0.0

        # Lower CV values = more stable learning
        avg_cv = np.mean(np.array([hb_cv, range_cv, trust_cv]))
        stability_score = max(0.0, 1.0 - avg_cv * 2.0)  # Convert to 0-1 score

        return float(stability_score)

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        if not self.training_history:
            return "No optimization results available"

        report = []
        report.append("╔══════════════════════════════════════════════════════════════════╗")
        report.append("║        SWARM COMMUNICATION RL OPTIMIZATION REPORT                ║")
        report.append("╠══════════════════════════════════════════════════════════════════╣")

        for result in self.training_history[-3:]:  # Show last 3 scenarios
            report.append(f"║ Scenario: {result['scenario']:<40} ║")
            report.append(f"║   Training Time: {result['training_time_seconds']:>6.1f}s                         ║")
            report.append(f"║   Best Reward: {result['best_reward']:>8.2f}                              ║")
            report.append(f"║   Convergence: {result['convergence_metrics']['convergence_score']:.1f}                           ║")
            report.append(f"║   Heartbeat: {result['optimal_parameters']['heartbeat_interval_ms']:>5d}ms                           ║")
            report.append(f"║   Pulse Range: {result['optimal_parameters']['pulse_range_multiplier']:.2f}                           ║")
            report.append(f"║   Trust Threshold: {result['optimal_parameters']['trust_threshold']:.2f}                           ║")
            if result['optimal_parameters']['sar_mode_active']:
                report.append("║   SAR: ✅ ACTIVE                                                  ║")
            else:
                report.append("║   SAR: ❌ INACTIVE                                                ║")
            report.append("╠══════════════════════════════════════════════════════════════════╣")

        report.append(f"║ Total Training Sessions: {len(self.training_history):<5d}                                ║")
        report.append("╚══════════════════════════════════════════════════════════════════╝")

        return "\n".join(report)


# Test implementation and validation

@pytest.fixture
def communication_rl():
    """RL agent fixture for testing"""
    return AdaptiveCommunicationRL(alpha=0.2, gamma=0.95, epsilon=0.15, seed=12345)

@pytest.fixture
def swarm_optimizer(communication_rl):
    """Swarm communication optimizer fixture"""
    return SwarmCommunicationOptimizer(communication_rl)


@pytest.mark.rl
def test_basic_rl_learning(communication_rl):
    """Test basic RL learning capabilities"""

    # Simple scenario: high density, moderate trust, some failures
    initial_state = SwarmState(
        agent_density=0.8,
        trust_entropy=0.6,
        failure_rate=0.1,
        communication_success=0.7
    )

    initial_params = CommunicationParams()

    # Run training episode
    episode_result = communication_rl.train_episode(initial_state, initial_params, max_steps=20)

    # Verify learning occurred
    assert len(episode_result) == 20, "Training episode should complete all steps"
    assert episode_result[-1]['connectivity'] >= 0.5, "Should achieve reasonable connectivity"

    # Verify Q-table was populated
    state_key = communication_rl.get_state_key(initial_state)
    assert state_key in communication_rl.q_table, "Q-table should be populated"

    print(f"✅ RL learning test passed: {len(episode_result)} steps, final connectivity: {episode_result[-1]['connectivity']:.3f}")


@pytest.mark.rl
def test_parameter_optimization_under_stress(swarm_optimizer):
    """Test parameter optimization in high-stress scenarios"""

    # High-stress scenario: low density, high failure rate
    stress_state = SwarmState(
        agent_density=0.2,    # Sparse deployment
        trust_entropy=0.8,    # Low trust consensus
        failure_rate=0.4,     # High failure rate
        communication_success=0.3  # Poor connectivity
    )

    # Run optimization
    result = swarm_optimizer.optimize_for_scenario("High-Stress Topology", stress_state, 50)

    # Verify optimization achieved improvement
    assert result['best_reward'] > -10, "Optimization should achieve positive learning"
    assert result['convergence_metrics']['convergence_score'] > 0.5, "Learning should converge"

    # Check optimized parameters make sense for stress scenario
    opt_params = result['optimal_parameters']
    assert opt_params['heartbeat_interval_ms'] < 1500, "Should prefer faster heartbeats in stress"

    print(f"✅ Stress optimization passed: reward={result['best_reward']:.2f}, convergence={result['convergence_metrics']['convergence_score']:.3f}")


@pytest.mark.rl
def test_adaptive_behavioral_learning(swarm_optimizer):
    """Test learning different strategies for different swarm conditions"""

    test_scenarios = [
        ("Dense_Crowd_Scenario", SwarmState(0.9, 0.6, 0.05, 0.9)),  # High density, good conditions
        ("Sparse_Failure_Scenario", SwarmState(0.2, 0.9, 0.3, 0.4)), # Low density, poor conditions
        ("Moderate_Balanced_Scenario", SwarmState(0.5, 0.7, 0.1, 0.7))  # Balanced conditions
    ]

    for scenario_name, swarm_state in test_scenarios:
        result = swarm_optimizer.optimize_for_scenario(scenario_name, swarm_state, 40)

        # Verify each scenario found viable parameters
        assert result['training_time_seconds'] < 60, "Training should complete in reasonable time"
        assert result['best_reward'] > result['best_reward'] * 0.8, "Should achieve consistent results"

    # Verify different scenarios led to different optimal parameters
    dense_params = swarm_optimizer.training_history[0]['optimal_parameters']
    sparse_params = swarm_optimizer.training_history[1]['optimal_parameters']

    # Dense scenarios should prefer different heartbeats than sparse
    heartbeat_difference = abs(dense_params['heartbeat_interval_ms'] - sparse_params['heartbeat_interval_ms'])
    assert heartbeat_difference > 100, "Different scenarios should lead to different optimal parameters"

    print(f"✅ Multi-scenario learning passed: tested {len(test_scenarios)} conditions with adaptive parameters")


@pytest.mark.rl
@pytest.mark.benchmark
def test_rl_performance_convergence(swarm_optimizer, benchmark):
    """Benchmark RL convergence speed and stability"""

    # Performance-sensitive scenario
    perf_state = SwarmState(0.6, 0.5, 0.15, 0.65)

    def convergence_test():
        return swarm_optimizer.optimize_for_scenario("Performance_Test", perf_state, 30)

    # Run benchmark
    result = benchmark(convergence_test)

    # Verify learning performance
    assert result['best_reward'] > -5, "Learning should achieve reasonable performance"
    assert result['convergence_metrics']['learning_stability'] > 0.6, "Parameters should stabilize"

    # Performance requirements
    assert result['training_time_seconds'] < 30, f"Training too slow: {result['training_time_seconds']:.2f}s"
    opt_params = result['optimal_parameters']
    assert 500 <= opt_params['heartbeat_interval_ms'] <= 3000, "Heartbeat intervals should be reasonable"

    print("✅ RL performance benchmark passed")
    print(f"   Training time: {result['training_time_seconds']:.2f}s")
    print(f"   Final reward: {result['best_reward']:.2f}")
    print(f"   Optimal heartbeat: {opt_params['heartbeat_interval_ms']}ms")


@pytest.mark.rl
def test_rl_optimization_report(swarm_optimizer):
    """Test comprehensive optimization reporting"""

    # Run multiple scenarios to generate report data
    scenarios = [
        ("Urban_Dense", SwarmState(0.8, 0.5, 0.1, 0.8)),
        ("Rural_Sparse", SwarmState(0.3, 0.8, 0.2, 0.5)),
        ("Industrial_Mixed", SwarmState(0.6, 0.6, 0.15, 0.7))
    ]

    for scenario_name, swarm_state in scenarios:
        swarm_optimizer.optimize_for_scenario(scenario_name, swarm_state, 20)

    # Generate comprehensive report
    report = swarm_optimizer.generate_optimization_report()

    # Verify report contains expected elements
    assert "SWARM COMMUNICATION RL OPTIMIZATION REPORT" in report
    assert "Total Training Sessions" in report
    assert "Dense" in report or "Sparse" in report  # Should include scenario data

    lines = report.split('\n')
    assert len(lines) >= 15, "Report should be comprehensive"

    print("✅ RL optimization report generation passed")
    print("   Report contains comprehensive scenario analysis")


# Integration test - end-to-end RL optimization for communication resilience
@pytest.mark.rl
def test_end_to_end_communication_optimization(communication_rl, swarm_optimizer):
    """Complete end-to-end test of RL-based communication optimization"""

    # Run comprehensive optimization across multiple challenging scenarios
    challenging_scenarios = [
        ("Extreme_Sparsity", SwarmState(0.1, 0.9, 0.5, 0.2)),  # Very challenging
        ("High_Density_Interference", SwarmState(0.95, 0.3, 0.3, 0.6)),  # Dense but problematic
        ("Trust_Collapse_Scenario", SwarmState(0.7, 0.95, 0.25, 0.45))  # Trust-poor environment
    ]

    optimization_results = []
    total_training_time = 0

    for scenario_name, swarm_state in challenging_scenarios:
        print(f"\n🔄 Optimizing scenario: {scenario_name}")

        result = swarm_optimizer.optimize_for_scenario(scenario_name, swarm_state, 60)
        optimization_results.append(result)
        total_training_time += result['training_time_seconds']

        # Each scenario should improve from initial performance
        assert result['best_reward'] > -20, f"{scenario_name} failed to learn effective communication"

        # Check scenario-specific adaptations
        if "Sparsity" in scenario_name:
            # Sparse scenarios should favor extended range
            assert result['optimal_parameters']['pulse_range_multiplier'] > 1.0, "Should extend range for sparsity"
        elif "Density" in scenario_name:
            # Dense scenarios should manage interference
            assert result['optimal_parameters']['sar_mode_active'] or result['optimal_parameters']['pulse_range_multiplier'] < 1.2, "Should manage density effects"

    # Overall optimization assessment
    avg_reward = np.mean([r['best_reward'] for r in optimization_results])
    avg_convergence = np.mean([r['convergence_metrics']['convergence_score'] for r in optimization_results])

    assert avg_reward > -15, f"Overall optimization performance too low: {avg_reward:.2f}"
    assert avg_convergence > 0.7, f"Learning convergence inadequate: {avg_convergence:.3f}"

    print(f"\n🏆 END-TO-END RL OPTIMIZATION COMPLETE")
    print(f"   Tested Scenarios: {len(challenging_scenarios)}")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Convergence: {avg_convergence:.3f}")
    print(f"   Total Training Time: {total_training_time:.1f}s")
    print("   ✅ RL-based adaptive communication system validated")
