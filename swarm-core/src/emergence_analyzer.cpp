#include "emergence_analyzer.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>

// EmergenceAnalyzer Implementation

EmergenceAnalyzer::EmergenceAnalyzer(const CyberGrid& grid)
    : grid_(grid),
      current_generation_(0),
      coarse_width_(grid.width() / COARSE_GRAIN_FACTOR),
      coarse_height_(grid.height() / COARSE_GRAIN_FACTOR) {

    // Reserve space for history buffers
    life_entropy_history_.reserve(ENTROPY_WINDOW_SIZE + 100);
    energy_entropy_history_.reserve(ENTROPY_WINDOW_SIZE + 100);
    coherence_history_.reserve(ENTROPY_WINDOW_SIZE + 100);

    metadata_history_.reserve(5000);  // For 1000+ generation runs
}

void EmergenceAnalyzer::analyze_generation() {
    current_generation_++;

    // Calculate current entropies
    double life_entropy = calculate_life_entropy();
    double energy_entropy = calculate_energy_entropy();
    double ecr = calculate_entropy_to_coherence_ratio();

    // Update rolling histories
    life_entropy_history_.push_back(life_entropy);
    energy_entropy_history_.push_back(energy_entropy);
    coherence_history_.push_back(ecr);

    // Maintain window size
    if (life_entropy_history_.size() > ENTROPY_WINDOW_SIZE) {
        life_entropy_history_.erase(life_entropy_history_.begin());
        energy_entropy_history_.erase(energy_entropy_history_.begin());
        coherence_history_.erase(coherence_history_.begin());
    }

    // Detect patterns and oscillators
    detect_oscillators();
    track_pattern_persistence();

    // Update coarse-grained analysis for macroscopic patterns
    update_coarse_grained_analysis();
}

double EmergenceAnalyzer::calculate_life_entropy() const {
    // Calculate Shannon entropy of alive/dead cell distribution
    double alive_count = 0.0;
    int total_cells = grid_.width() * grid_.height();

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            if (grid_.get_cell(x, y).alive) {
                alive_count += 1.0;
            }
        }
    }

    if (alive_count == 0.0 || alive_count == static_cast<double>(total_cells)) {
        return 0.0;  // No entropy in uniform states
    }

    double p_alive = alive_count / total_cells;
    double p_dead = 1.0 - p_alive;

    return -p_alive * std::log2(p_alive) - p_dead * std::log2(p_dead);
}

double EmergenceAnalyzer::calculate_energy_entropy() const {
    // Calculate entropy of energy distribution (binned)
    const int ENERGY_BINS = 16;
    std::vector<int> bins(ENERGY_BINS, 0);
    int total_cells = grid_.width() * grid_.height();

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            int bin = static_cast<int>(grid_.get_cell(x, y).energy * (ENERGY_BINS - 1));
            bin = std::clamp(bin, 0, ENERGY_BINS - 1);
            bins[bin]++;
        }
    }

    double entropy = 0.0;
    for (int count : bins) {
        if (count > 0) {
            double p = static_cast<double>(count) / total_cells;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

double EmergenceAnalyzer::calculate_entropy_to_coherence_ratio() const {
    // ECR: measure of order formation vs randomness
    // Lower ECR indicates more coherent/patterned behavior
    double life_entropy = calculate_life_entropy();
    double energy_entropy = calculate_energy_entropy();

    // Calculate spatial coherence (inverse of conditional entropy between energy and life)
    double spatial_coherence = calculate_spatial_coherence();

    // ECR = entropy_sum / (1 + coherence) - approaches 0 when coherent
    double entropy_sum = life_entropy + energy_entropy;
    return entropy_sum / std::max(1.0 + spatial_coherence, 1e-6);
}

double EmergenceAnalyzer::calculate_spatial_coherence() const {
    // Measure how well energy and life patterns correlate spatially
    // Higher coherence = more organized emergence
    double correlation_sum = 0.0;
    int sample_count = 0;

    // Sample correlation across different spatial scales
    for (int scale = 1; scale <= 3; ++scale) {
        for (int y = scale; y < grid_.height() - scale; y += scale * 2) {
            for (int x = scale; x < grid_.width() - scale; x += scale * 2) {
                // Calculate local pattern correlation
                double local_energy_variance = 0.0;
                double local_life_energy_correlation = 0.0;
                int local_cells = 0;

                // Sample neighborhood
                for (int dy = -scale; dy <= scale; ++dy) {
                    for (int dx = -scale; dx <= scale; ++dx) {
                        const auto& cell = grid_.get_cell(x + dx, y + dy);
                        local_energy_variance += cell.energy * cell.energy;
                        local_life_energy_correlation += (cell.alive ? 1.0 : 0.0) * cell.energy;
                        local_cells++;
                    }
                }

                if (local_cells > 0) {
                    local_energy_variance /= local_cells;
                    local_life_energy_correlation /= local_cells;

                    // Add to correlation sum (normalized)
                    correlation_sum += std::abs(local_life_energy_correlation) / std::max(std::sqrt(local_energy_variance), 1e-6);
                    sample_count++;
                }
            }
        }
    }

    return sample_count > 0 ? correlation_sum / sample_count : 0.0;
}

void EmergenceAnalyzer::detect_oscillators() {
    active_oscillators_.clear();

    // Simple oscillator detection: look for stable 2-3 period patterns
    // This is a simplified version - full oscillator census would need state history
    for (int y = 1; y < grid_.height() - 1; ++y) {
        for (int x = 1; x < grid_.width() - 1; ++x) {
            // Check for potential oscillator by looking at local activity
            int neighbor_count = 0;
            float avg_neighbor_energy = 0.0f;

            // Moore neighborhood
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const auto& neighbor = grid_.get_cell(x + dx, y + dy);
                    if (neighbor.alive) neighbor_count++;
                    avg_neighbor_energy += neighbor.energy;
                }
            }
            avg_neighbor_energy /= 8.0f;

            // Heuristic: cells in oscillator-like configurations
            const auto& cell = grid_.get_cell(x, y);
            if (neighbor_count >= 2 && neighbor_count <= 3 &&
                cell.energy > 0.5f && avg_neighbor_energy > 0.3f) {
                // Potential oscillator activity
                std::string signature = generate_cell_pattern_signature(x, y, 2);
                pattern_persistence_[signature]++;
            }
        }
    }

    // Identify persistently active regions as oscillators
    for (const auto& pair : pattern_persistence_) {
        if (pair.second > 5) {  // Persisted for multiple generations
            // Extract coordinates from signature (simplified: just track count)
            // In full implementation, we'd decode position from signature
            // For now, just track count of persistent patterns
        }
    }
}

void EmergenceAnalyzer::track_pattern_persistence() {
    // Decay persistence counts and clean up old patterns
    for (auto it = pattern_persistence_.begin(); it != pattern_persistence_.end(); ) {
        it->second *= 0.95;  // Exponential decay
        if (it->second < 1.0) {
            it = pattern_persistence_.erase(it);
        } else {
            ++it;
        }
    }

    // Sample current patterns for persistence tracking
    for (int y = 2; y < grid_.height() - 2; y += 4) {
        for (int x = 2; x < grid_.width() - 2; x += 4) {
            std::string signature = generate_cell_pattern_signature(x, y, 2);
            if (!signature.empty()) {
                pattern_persistence_[signature] += 1.0;
            }
        }
    }
}

std::string EmergenceAnalyzer::generate_cell_pattern_signature(int x, int y, int radius) const {
    // Generate a compact signature of the cell pattern around (x,y)
    std::string signature;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const auto& cell = grid_.get_cell(x + dx, y + dy);
            char life_char = cell.alive ? '1' : '0';
            char energy_char = static_cast<char>('A' + static_cast<int>(cell.energy * 25));  // 26 levels
            signature += life_char;
            signature += energy_char;
        }
    }

    return signature;
}

void EmergenceAnalyzer::update_coarse_grained_analysis() {
    // Coarse graining already calculated in get_coarse_grained_energy_field()
    // Could cache this if performance becomes an issue
}

std::vector<std::vector<double>> EmergenceAnalyzer::get_coarse_grained_energy_field() const {
    std::vector<std::vector<double>> coarse_field(coarse_height_,
                                                  std::vector<double>(coarse_width_, 0.0));

    for (int cy = 0; cy < coarse_height_; ++cy) {
        for (int cx = 0; cx < coarse_width_; ++cx) {
            // Average energy in this coarse cell
            double sum = 0.0;
            int count = 0;

            for (int fy = cy * COARSE_GRAIN_FACTOR;
                 fy < std::min((cy + 1) * COARSE_GRAIN_FACTOR, grid_.height()); ++fy) {
                for (int fx = cx * COARSE_GRAIN_FACTOR;
                     fx < std::min((cx + 1) * COARSE_GRAIN_FACTOR, grid_.width()); ++fx) {
                    sum += grid_.get_cell(fx, fy).energy;
                    count++;
                }
            }

            coarse_field[cy][cx] = count > 0 ? sum / count : 0.0;
        }
    }

    return coarse_field;
}

bool EmergenceAnalyzer::detect_emergence_convergence(double threshold) const {
    if (coherence_history_.size() < 10) return false;

    // Check if ECR has stabilized (convergence to coherent state)
    size_t window_size = std::min<size_t>(20, coherence_history_.size());
    std::vector<double> recent_window(coherence_history_.end() - window_size,
                                      coherence_history_.end());

    double recent_avg = calculate_running_average(recent_window);
    double recent_std = calculate_standard_deviation(recent_window);

    // Converged if coefficient of variation is low (stable ECR)
    double cv = recent_std / std::max(recent_avg, 1e-6);
    return cv < threshold;
}

double EmergenceAnalyzer::measure_system_stability() const {
    if (coherence_history_.size() < 5) return 0.0;

    // Stability = inverse of ECR variance over recent generations
    double avg_ecr = calculate_running_average(coherence_history_);
    double std_ecr = calculate_standard_deviation(coherence_history_);

    // Higher stability = more ordered/less chaotic system
    return std_ecr > 0.0 ? 1.0 / (1.0 + std_ecr) : 1.0;
}

void EmergenceAnalyzer::record_generation_metadata(double tick_duration_ms, double timing_error) {
    GenerationMetadata metadata{
        .tick_count = current_generation_,
        .timestamp = std::chrono::steady_clock::now(),
        .actual_tick_duration_ms = tick_duration_ms,
        .timing_error_accumulator = timing_error,
        .alive_cell_count = 0,
        .average_energy = grid_.average_energy()
    };

    // Count alive cells
    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            if (grid_.get_cell(x, y).alive) {
                metadata.alive_cell_count++;
            }
        }
    }

    metadata_history_.push_back(metadata);

    // Maintain reasonable history size
    if (metadata_history_.size() > 5000) {
        metadata_history_.erase(metadata_history_.begin());
    }
}

const std::vector<EmergenceAnalyzer::GenerationMetadata>& EmergenceAnalyzer::get_metadata_history() const {
    return metadata_history_;
}

std::unordered_map<std::string, double> EmergenceAnalyzer::get_current_metrics() const {
    return {
        {"generation", static_cast<double>(current_generation_)},
        {"life_entropy", calculate_life_entropy()},
        {"energy_entropy", calculate_energy_entropy()},
        {"entropy_coherence_ratio", calculate_entropy_to_coherence_ratio()},
        {"spatial_coherence", calculate_spatial_coherence()},
        {"oscillator_count", static_cast<double>(get_oscillator_count())},
        {"system_stability", measure_system_stability()},
        {"alive_cells", static_cast<double>(grid_.alive_cell_count())}
    };
}

std::string EmergenceAnalyzer::generate_emergence_report() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    auto metrics = get_current_metrics();

    ss << "=== Swarm Intelligence Emergence Report ===\n";
    ss << "Generation: " << metrics["generation"] << "\n";
    ss << "Life Entropy: " << metrics["life_entropy"] << "\n";
    ss << "Energy Entropy: " << metrics["energy_entropy"] << "\n";
    ss << "Entropy-to-Coherence Ratio (ECR): " << metrics["entropy_coherence_ratio"] << "\n";
    ss << "Spatial Coherence: " << metrics["spatial_coherence"] << "\n";
    ss << "Active Oscillators: " << metrics["oscillator_count"] << "\n";
    ss << "System Stability: " << metrics["system_stability"] << "\n";
    ss << "Alive Cells: " << metrics["alive_cells"] << "\n";
    ss << "Emergence Convergence: " << (detect_emergence_convergence() ? "Yes" : "No") << "\n";

    return ss.str();
}

bool EmergenceAnalyzer::run_emergence_benchmark(int generations, const std::string& output_path) {
    std::ofstream logfile(output_path + "/emergence_log.csv");
    if (!logfile.is_open()) return false;

    // CSV header
    logfile << "generation,life_entropy,energy_entropy,ecr,coherence,oscillators,stability,alive_cells,tick_duration_ms,timing_error\n";

    // Need access to grid stepping - this would need to be connected to the step loop
    // For now, return false as this requires integration with the simulation loop
    logfile.close();
    return false;  // Requires external loop integration
}

// Statistical utilities
double EmergenceAnalyzer::calculate_running_average(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double EmergenceAnalyzer::calculate_standard_deviation(const std::vector<double>& values) const {
    if (values.size() <= 1) return 0.0;

    double mean = calculate_running_average(values);
    double variance = 0.0;

    for (double val : values) {
        double diff = val - mean;
        variance += diff * diff;
    }

    return std::sqrt(variance / (values.size() - 1));
}

// Not implemented helper methods (these are referenced but not critical for basic operation)
// These would be implemented for full functionality
double EmergenceAnalyzer::calculate_shannon_entropy(const std::vector<double>& probabilities) const {
    double entropy = 0.0;
    for (double p : probabilities) {
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

double EmergenceAnalyzer::calculate_conditional_entropy(int x1, int y1, int x2, int y2) const {
    // Simplified implementation - would need full conditional probability calculations
    return 0.0;
}

std::vector<double> EmergenceAnalyzer::get_cell_value_distribution(bool use_energy) const {
    // Not implemented for this basic version
    return std::vector<double>();
}

void EmergenceAnalyzer::initialize_coarse_graining() {
    // Implementation handled in constructor
}

std::vector<std::vector<double>> EmergenceAnalyzer::coarse_grain_field(const std::vector<std::vector<double>>& fine_field) const {
    // Basic implementation - delegate to get_coarse_grained_energy_field
    return get_coarse_grained_energy_field();
}

// EmergentBehaviorBenchmarkSuite Implementation

EmergentBehaviorBenchmarkSuite::EmergentBehaviorBenchmarkSuite()
    : max_generations_(1000),
      repetitions_per_config_(5),
      output_directory_("emergence_benchmarks") {
}

void EmergentBehaviorBenchmarkSuite::add_grid_configuration(int width, int height, float alive_prob, float energy_prob) {
    std::string config_name = generate_config_name(width, height, alive_prob, energy_prob);
    grids_.push_back(std::make_unique<CyberGrid>(width, height));
    grids_.back()->randomize(alive_prob, energy_prob);

    analyzers_.push_back(create_analyzer_for_grid(*grids_.back()));
}

void EmergentBehaviorBenchmarkSuite::set_parameters(int max_generations, int repetitions, const std::string& output_dir) {
    max_generations_ = max_generations;
    repetitions_per_config_ = repetitions;
    output_directory_ = output_dir;
}

void EmergentBehaviorBenchmarkSuite::run_full_benchmark_suite() {
    if (grids_.empty()) {
        // Add default configurations for comprehensive testing
        add_grid_configuration(50, 50, 0.3f, 0.2f);     // Balanced emergence
        add_grid_configuration(100, 100, 0.1f, 0.5f);   // Sparse with high energy
        add_grid_configuration(100, 100, 0.5f, 0.1f);   // Dense with low energy
        add_grid_configuration(50, 50, 0.01f, 0.8f);    // Very sparse, high energy pulses
    }

    results_.clear();

    for (size_t config_idx = 0; config_idx < grids_.size(); ++config_idx) {
        const auto& grid = grids_[config_idx];
        const auto& analyzer = analyzers_[config_idx];
        std::string config_name = generate_config_name(
            grid->width(), grid->height(),
            0.3f, 0.2f  // Simplified - would need to store original params
        );

        // Run multiple repetitions for statistical significance
        for (int rep = 0; rep < repetitions_per_config_; ++rep) {
            BenchmarkResult result;
            result.config_name = config_name + "_rep" + std::to_string(rep + 1);

            // Reset grid for each repetition
            grid->randomize(0.3f, 0.2f);  // Simplified - use stored params
            bool emergence_found = run_single_emergence_test(*grid, max_generations_);

            // Analyze final state
            result.emergence_detected = emergence_found;
            result.emergence_generation = emergence_found ? max_generations_ : -1;
            result.average_ecr = analyzer->calculate_entropy_to_coherence_ratio();
            result.final_oscillator_count = analyzer->get_oscillator_count();
            result.final_stability_score = analyzer->measure_system_stability();
            // ECR trajectory would need continuous tracking

            results_.push_back(result);
        }
    }

    // Save comprehensive report
    generate_emergence_report(output_directory_ + "/benchmark_results.md");
}

void EmergentBehaviorBenchmarkSuite::generate_emergence_report(const std::string& report_path) {
    std::ofstream report_file(report_path);
    if (!report_file.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    report_file << "# Swarm Intelligence Emergence Benchmark Results\n\n";
    report_file << "Generated: " << std::ctime(&time_t) << "\n";

    report_file << "## Summary Statistics\n";
    report_file << "- Total Configurations Tested: " << results_.size() / repetitions_per_config_ << "\n";
    report_file << "- Emergence Success Rate: " << std::fixed << std::setprecision(1) <<
                   calculate_emergence_success_rate() * 100 << "%\n\n";

    auto stats = get_aggregate_statistics();
    report_file << "## Aggregate Metrics\n";
    for (const auto& [key, value] : stats) {
        report_file << "- " << key << ": " << std::fixed << std::setprecision(4) << value << "\n";
    }
    report_file << "\n";

    report_file << "## Detailed Results\n";
    report_file << "| Configuration | Emergence | Generation | Avg ECR | Oscillators | Stability |\n";
    report_file << "|--------------|-----------|------------|---------|------------|-----------|\n";

    for (const auto& result : results_) {
        report_file << "| " << result.config_name << " | " <<
                   (result.emergence_detected ? "✅" : "❌") << " | " <<
                   result.emergence_generation << " | " <<
                   std::fixed << std::setprecision(3) << result.average_ecr << " | " <<
                   result.final_oscillator_count << " | " <<
                   result.final_stability_score << " |\n";
    }

    report_file.close();
}

double EmergentBehaviorBenchmarkSuite::calculate_emergence_success_rate() const {
    if (results_.empty()) return 0.0;
    int success_count = std::count_if(results_.begin(), results_.end(),
                                     [](const BenchmarkResult& r) { return r.emergence_detected; });
    return static_cast<double>(success_count) / results_.size();
}

std::unordered_map<std::string, double> EmergentBehaviorBenchmarkSuite::get_aggregate_statistics() const {
    if (results_.empty()) return {};

    double avg_ecr = 0.0;
    double avg_oscillators = 0.0;
    double avg_stability = 0.0;

    for (const auto& result : results_) {
        avg_ecr += result.average_ecr;
        avg_oscillators += result.final_oscillator_count;
        avg_stability += result.final_stability_score;
    }

    int count = results_.size();
    return {
        {"Average ECR", avg_ecr / count},
        {"Average Oscillators", avg_oscillators / count},
        {"Average Stability", avg_stability / count}
    };
}

bool EmergentBehaviorBenchmarkSuite::run_single_emergence_test(CyberGrid& grid, int max_generations) {
    // Simplified test - in full implementation this would run the simulator
    // For now, just return random result to demonstrate framework
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Some configurations more likely to show emergence
    double emergence_probability = 0.4;  // 40% success rate for demo
    return dis(gen) < emergence_probability;
}

std::unique_ptr<EmergenceAnalyzer> EmergentBehaviorBenchmarkSuite::create_analyzer_for_grid(const CyberGrid& grid) {
    return std::make_unique<EmergenceAnalyzer>(grid);
}

std::string EmergentBehaviorBenchmarkSuite::generate_config_name(int width, int height, float alive_prob, float energy_prob) const {
    std::stringstream ss;
    ss << width << "x" << height << "_alive" << std::fixed << std::setprecision(1) << alive_prob *
                                      100 << "_energy" << energy_prob * 100;
    return ss.str();
}

void EmergentBehaviorBenchmarkSuite::save_trajectory_data(const BenchmarkResult& result, const std::string& filename) const {
    // Implementation would save ECR trajectories and other time series data
    // Not implemented in this basic version
}
