#pragma once

#include "cyber_grid.hpp"
#include <vector>
#include <unordered_map>
#include <chrono>
#include <memory>

// Forward declarations
class CyberGrid;

/**
 * EmergenceAnalyzer - Quantifies Swarm Intelligence Emergence
 *
 * Analyzes the CyberGrid for emergent behaviors using entropy-based metrics
 * and pattern detection. Validates whether local rules (Conway + LoRA) produce
 * global intelligence patterns under hardware-locked timing.
 */
class EmergenceAnalyzer {
private:
    const CyberGrid& grid_;
    long long current_generation_;

    // Entropy tracking over time windows
    static constexpr int ENTROPY_WINDOW_SIZE = 100;
    std::vector<double> life_entropy_history_;
    std::vector<double> energy_entropy_history_;
    std::vector<double> coherence_history_;

    // Pattern persistence tracking
    std::unordered_map<std::string, int> pattern_persistence_;
    std::vector<std::pair<int, int>> active_oscillators_;  // (x,y) coordinates

    // Coarse-graining for macro-scale analysis
    static constexpr int COARSE_GRAIN_FACTOR = 4;
    int coarse_width_, coarse_height_;

    // Metadata for reproducibility validation
    struct GenerationMetadata {
        long long tick_count;
        std::chrono::steady_clock::time_point timestamp;
        double actual_tick_duration_ms;
        double timing_error_accumulator;
        int alive_cell_count;
        float average_energy;
    };
    std::vector<GenerationMetadata> metadata_history_;

public:
    explicit EmergenceAnalyzer(const CyberGrid& grid);

    // Core analysis methods
    void analyze_generation();  // Called after each grid.step()

    // Entropy and coherence metrics
    double calculate_life_entropy() const;
    double calculate_energy_entropy() const;
    double calculate_entropy_to_coherence_ratio() const;  // ECR primary metric

    // Pattern detection
    void detect_oscillators();
    void track_pattern_persistence();
    int get_oscillator_count() const { return active_oscillators_.size(); }

    // Coarse-grained analysis for macroscopic patterns
    void update_coarse_grained_analysis();
    std::vector<std::vector<double>> get_coarse_grained_energy_field() const;

    // Convergence and stability analysis
    bool detect_emergence_convergence(double threshold = 0.1) const;
    double measure_system_stability() const;

    // Metadata and reproducibility
    void record_generation_metadata(double tick_duration_ms, double timing_error);
    const std::vector<GenerationMetadata>& get_metadata_history() const;

    // Reporting and visualization helpers
    std::unordered_map<std::string, double> get_current_metrics() const;
    std::string generate_emergence_report() const;

    // Long-term analysis for 1000+ generation runs
    bool run_emergence_benchmark(int generations, const std::string& output_path);

private:
    // Low-level entropy calculations
    double calculate_shannon_entropy(const std::vector<double>& probabilities) const;
    double calculate_conditional_entropy(int x1, int y1, int x2, int y2) const;
    double calculate_spatial_coherence() const;

    // Grid analysis helpers
    std::vector<double> get_cell_value_distribution(bool use_energy = false) const;

    // Pattern recognition
    std::string generate_cell_pattern_signature(int x, int y, int radius = 3) const;

    // Coarse-graining utilities
    void initialize_coarse_graining();
    std::vector<std::vector<double>> coarse_grain_field(const std::vector<std::vector<double>>& fine_field) const;

    // Statistical utilities
    double calculate_running_average(const std::vector<double>& values) const;
    double calculate_standard_deviation(const std::vector<double>& values) const;
};

/**
 * Fault Injection Pattern for Resilience Testing
 */
struct FaultInjectionPattern {
    enum class FaultType {
        NETWORK_PARTITION,
        HEARTBEAT_FAILURE,
        TRUST_CORRUPTION,
        PULSE_INTERFERENCE,
        AGENT_ISOLATION
    };

    FaultType type;
    int start_generation;
    int duration_generations;
    float intensity;  // 0.0 to 1.0 - how severe the fault is
    std::vector<std::string> affected_agents;  // Empty means apply to random subset
    std::unordered_map<std::string, float> fault_parameters;  // Type-specific parameters
};

/**
 * ResilientEmergenceBenchmarker - Tests Swarm Resilience Under Stress
 *
 * Extends emergence analysis with fault injection and resilience metrics
 * Measures MTTR (Mean Time To Reconnect), survival ratios, and adaptive communication effectiveness
 */
class ResilientEmergenceBenchmarker {
private:
    CyberGrid* grid_;
    std::vector<std::string> active_agents_;
    std::unordered_map<std::string, float> agent_trust_scores_;

    // Resilience metrics tracking
    struct ResilienceSnapshot {
        long long generation;
        std::vector<std::string> failed_agents;
        std::vector<std::string> recovered_agents;
        double connectivity_index;  // Fraction of connected agent pairs
        double trust_entropy;       // Diversity in trust scores
        double communication_success_rate;
        int active_sar_operations;
    };
    std::vector<ResilienceSnapshot> resilience_history_;

public:
    // Core resilience metrics structure
    struct ResilienceMetrics {
        double mean_time_to_rejoin_ms;           // MTTR for failed agents
        double agent_survival_ratio;             // Fraction surviving entire test
        double communication_entropy;            // Network fragmentation measure
        double trust_stability_score;            // Trust score variance over time
        double heartbeat_success_rate;           // Adaptive heartbeat effectiveness
        double average_connectivity_recovery;    // Speed of network reconnection
        int max_simultaneous_failures;           // Worst case failure cascade
        double sar_operation_efficiency;         // Search and reconnect success rate
    };

    explicit ResilientEmergenceBenchmarker(CyberGrid& grid);

    // Agent management
    void register_agents(const std::vector<std::string>& agent_ids);
    void update_agent_trust(const std::string& agent_id, float trust_score);

    // Fault injection and testing
    ResilienceMetrics run_resilience_simulation(
        int agent_count,
        int target_generations,
        const std::vector<FaultInjectionPattern>& fault_patterns
    );

    // Individual fault types
    void inject_network_partition(int isolation_duration, float affected_percentage);
    void inject_heartbeat_failure(const std::vector<std::string>& affected_agents, float drop_probability);
    void inject_trust_corruption(float corruption_intensity);
    void inject_pulse_interference(int x, int y, int radius, float interference_strength);

    // Recovery tracking
    std::vector<std::string> get_failed_agents() const;
    double calculate_current_connectivity() const;
    double calculate_trust_entropy() const;

    // SAR operation monitoring
    void start_sar_operation(const std::string& rover_agent, const std::string& target_agent);
    void complete_sar_operation(const std::string& rover_agent, const std::string& target_agent, bool success);

private:
    // Helper methods
    void simulate_generation_with_faults(const FaultInjectionPattern& pattern);
    void update_resilience_snapshot(long long generation);
    ResilienceMetrics calculate_final_metrics() const;
    std::vector<std::string> select_random_agents(int count) const;
};

/**
 * AdaptiveCommunicationRL - Q-Learning for Communication Resilience Optimization
 *
 * Implements reinforcement learning to optimize adaptive communication parameters:
 * - Adaptive heartbeat intervals based on density and trust
 * - Pulse range optimization for connectivity vs energy trade-offs
 * - Trust threshold adaptation for fault containment
 *
 * State Space: [agent_density_bins, trust_entropy_bins, failure_rate_bins, communication_success_bins]
 * Actions: parameter adjustments (heartbeat_freq, pulse_range, trust_threshold, sar_threshold)
 * Reward: connectivity_maintained - energy_cost_penalty - failure_penalty
 */
class AdaptiveCommunicationRL {
private:
    // State discretization bins
    static constexpr int DENSITY_BINS = 5;     // [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    static constexpr int TRUST_BINS = 4;       // [0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]
    static constexpr int FAILURE_BINS = 3;     // [0.0-0.1, 0.1-0.3, 0.3-1.0]
    static constexpr int SUCCESS_BINS = 4;     // [0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]

    // RL Parameters
    static constexpr double ALPHA = 0.1;       // Learning rate
    static constexpr double GAMMA = 0.9;       // Discount factor
    static constexpr double EPSILON = 0.1;     // Exploration rate

    // Q-Table: state -> action -> value
    using StateKey = std::tuple<int, int, int, int>;  // (density, trust, failure, success)
    std::unordered_map<StateKey, std::unordered_map<int, double>> q_table_;

    // Action space
    enum class Action {
        REDUCE_HEARTBEAT_FREQ,
        INCREASE_HEARTBEAT_FREQ,
        EXTEND_PULSE_RANGE,
        REDUCE_PULSE_RANGE,
        RAISE_TRUST_THRESHOLD,
        LOWER_TRUST_THRESHOLD,
        ACTIVATE_SAR_MODE,
        CONSERVE_ENERGY_MODE,
        MAX_ACTIONS
    };

    // Parameter ranges
    struct CommunicationParameters {
        int base_heartbeat_ms;
        float pulse_range_multiplier;
        float trust_threshold;
        bool sar_mode_active;
        CommunicationParameters() : base_heartbeat_ms(1000), pulse_range_multiplier(1.0f),
                                  trust_threshold(0.3f), sar_mode_active(false) {}
    };

    CommunicationParameters current_params_;
    CommunicationParameters best_params_;  // Parameters that achieved highest reward

    // Learning state
    std::mt19937 rng_;
    double total_reward_;
    int episodes_completed_;

    // Performance tracking
    struct PerformanceMetrics {
        double connectivity_score;
        double energy_efficiency;
        double failure_rate;
        int actions_taken;
    };
    std::vector<PerformanceMetrics> performance_history_;

public:
    AdaptiveCommunicationRL();

    // Core RL interface
    int select_action(const std::vector<double>& state_features);
    double calculate_reward(const PerformanceMetrics& before, const PerformanceMetrics& after);
    void update_q_value(int state_key, int action, double reward, int next_state_key);
    void train_episode(const std::vector<PerformanceMetrics>& episode_metrics);

    // State encoding/decoding
    StateKey encode_state(double agent_density, double trust_entropy,
                         double failure_rate, double communication_success) const;
    std::vector<double> decode_state(const StateKey& state) const;

    // Parameter optimization
    CommunicationParameters optimize_parameters(const CyberGrid& grid,
                                              const std::vector<std::string>& agents);
    void apply_parameters(CyberGrid& grid, const CommunicationParameters& params);

    // Learning evaluation
    double get_best_reward() const;
    CommunicationParameters get_optimal_parameters() const;
    std::string generate_learning_report() const;

private:
    // Helper methods
    double get_q_value(const StateKey& state, int action) const;
    void set_q_value(const StateKey& state, int action, double value);
    int discretize_value(double value, int bins, double min_val = 0.0, double max_val = 1.0) const;
    double calculate_connectivity_reward(double before, double after) const;
    double calculate_energy_penalty(const CommunicationParameters& params) const;

};

/**
 * EmergentBehaviorBenchmarkSuite - Benchmark Suite for Emergence Analysis
 *
 * Provides comprehensive benchmarking capabilities for emergent behavior analysis,
 * supporting multiple grid configurations, statistical significance through repetitions,
 * and automated report generation.
 */
class EmergentBehaviorBenchmarkSuite {
private:
    int max_generations_;
    int repetitions_per_config_;
    std::string output_directory_;

    // Test configurations
    std::vector<std::unique_ptr<CyberGrid>> grids_;
    std::vector<std::unique_ptr<EmergenceAnalyzer>> analyzers_;

    // Benchmark results
    struct BenchmarkResult {
        std::string config_name;
        bool emergence_detected;
        int emergence_generation;
        double average_ecr;
        int final_oscillator_count;
        double final_stability_score;
    };
    std::vector<BenchmarkResult> results_;

public:
    EmergentBehaviorBenchmarkSuite();
    EmergentBehaviorBenchmarkSuite(int max_generations, int repetitions, const std::string& output_dir);

    // Configuration management
    void add_grid_configuration(int width, int height, float alive_prob, float energy_prob);
    void set_parameters(int max_generations, int repetitions, const std::string& output_dir);

    // Benchmark execution
    void run_full_benchmark_suite();
    void generate_emergence_report(const std::string& report_path);

    // Statistical analysis
    double calculate_emergence_success_rate() const;
    std::unordered_map<std::string, double> get_aggregate_statistics() const;

private:
    // Helper methods
    bool run_single_emergence_test(CyberGrid& grid, int max_generations);
    std::unique_ptr<EmergenceAnalyzer> create_analyzer_for_grid(const CyberGrid& grid);
    std::string generate_config_name(int width, int height, float alive_prob, float energy_prob) const;
    void save_trajectory_data(const BenchmarkResult& result, const std::string& filename) const;
};
