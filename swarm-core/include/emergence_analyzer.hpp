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
 * EmergentBehaviorBenchmarkSuite - Automated Validation Framework
 *
 * Runs comprehensive emergence testing across multiple initial conditions,
 * grid sizes, and parameter settings. Produces statistical analysis of
 * emergence success rates and characteristic signatures.
 */
class EmergentBehaviorBenchmarkSuite {
private:
    std::vector<std::unique_ptr<EmergenceAnalyzer>> analyzers_;
    std::vector<std::unique_ptr<CyberGrid>> grids_;

    // Benchmark configuration
    int max_generations_;
    int repetitions_per_config_;
    std::string output_directory_;

    // Results storage
    struct BenchmarkResult {
        std::string config_name;
        bool emergence_detected;
        int emergence_generation;
        double average_ecr;
        int final_oscillator_count;
        double final_stability_score;
        std::vector<double> ecr_trajectory;
    };
    std::vector<BenchmarkResult> results_;

public:
    EmergentBehaviorBenchmarkSuite();

    // Benchmark setup
    void add_grid_configuration(int width, int height, float alive_prob, float energy_prob);
    void set_parameters(int max_generations, int repetitions, const std::string& output_dir);

    // Execution
    void run_full_benchmark_suite();

    // Analysis and reporting
    void generate_emergence_report(const std::string& report_path);
    double calculate_emergence_success_rate() const;
    std::unordered_map<std::string, double> get_aggregate_statistics() const;

    // Individual test utilities
    bool run_single_emergence_test(CyberGrid& grid, int max_generations);
    std::unique_ptr<EmergenceAnalyzer> create_analyzer_for_grid(const CyberGrid& grid);

private:
    std::string generate_config_name(int width, int height, float alive_prob, float energy_prob) const;
    void save_trajectory_data(const BenchmarkResult& result, const std::string& filename) const;
};
